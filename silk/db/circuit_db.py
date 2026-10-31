"""SQLite design database: circuit snapshots, evidence and stage management.

The :class:`Database` class stores circuit design data as immutable,
identity-bearing snapshots (rows of ``circuit_versions``) together with the
metrics, spec measurements, characterization profiles and net-current sets
produced for them. Modules are stored once in a ``modules`` table and linked
to snapshots through a junction table, so a module shared by several
snapshots is not duplicated.
"""

import sqlite3
from dataclasses import replace
from collections.abc import Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from itertools import groupby
import json
import logging
import uuid
import hashlib
from datetime import datetime
from numbers import Real
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

from silk._json_codec import decode_json, encode_json
from silk.db._condition_columns import (
    condition_columns,
    condition_predicate,
    condition_scalar,
    profile_check_sql,
    state_columns_sql,
)
from silk.db.context import CharacterizationContext
from silk.db.identity import (
    PROFILE_FINGERPRINT_SCHEME,
    SPEC_FINGERPRINT_SCHEME,
    STRUCTURE_FINGERPRINT_SCHEME,
    _canonicalize_characterization_profile,
    _decode_characterization_profile_parameters,
    _fingerprint_characterization_profile_fields,
    _fingerprint_spec_definition_fields,
    _fingerprint_spec_set_records,
    _fingerprints_equal,
    _require_fingerprint_scheme,
)
from silk.db.models import (
    CharacterizationEvidenceRecord,
    CharacterizationProfileRecord,
    Fingerprint,
    NetCurrentRecord,
    NetCurrentSetRecord,
    SimulationResultRecord,
    SnapshotIdentity,
    SpecDeclarationRecord,
    SpecMeasurementRecord,
    UNSCOPED_PROFILE,
)
from silk.db.schema import (
    DATABASE_FORMAT,
    DATABASE_SCHEMA_VERSION,
    _validate_database_version,
    inspect_existing_database,
    validate_database_connection,
)
from silk.exceptions import (
    DatabaseError,
    EvidenceAmbiguityError,
    EvidenceMismatchError,
    EvidenceNotFoundError,
    LockedVersionError,
)


_log = logging.getLogger(__name__)

# Schema v4 (TASK-306): the nullable provenance link carried by
# simulation_results, spec_measurements and net_current_sets. Added by ALTER
# TABLE both for new files and in migrate_v3_to_v4, so a fresh v4 file and a
# migrated one have the same table definitions.
_PROVENANCE_COLUMN_SQL = (
    "provenance_id TEXT REFERENCES run_provenance(provenance_id) "
    "ON DELETE RESTRICT"
)
_PROVENANCE_TABLES = ("simulation_results", "spec_measurements", "net_current_sets")

_SPEC_MEASUREMENT_SOURCES = frozenset(
    {"behavioral", "spice", "extraction", "user"}
)
_NET_CURRENT_SOURCES = frozenset({"spice", "user"})
_SPEC_COMPARISONS = {
    "<": lambda value, target: value < target,
    "<=": lambda value, target: value <= target,
    ">": lambda value, target: value > target,
    ">=": lambda value, target: value >= target,
    "==": lambda value, target: value == target,
}


def _spec_measurement_real(value, *, field, optional=False):
    """Validate a numeric measurement field and return a plain float."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        qualifier = " or None" if optional else ""
        raise TypeError(
            f"Spec measurement {field} must be a real number{qualifier}"
        )
    return float(value)


def _net_current_real(
    value,
    *,
    field,
    optional=True,
    error_type=TypeError,
    subject="Current row",
):
    """Validate a current scalar without trusting SQLite's loose typing."""
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        qualifier = " or None" if optional else ""
        raise error_type(
            f"{subject} {field} must be a real number{qualifier}"
        )
    return float(value)


def _net_current_text(
    value,
    *,
    field,
    optional=False,
    error_type=TypeError,
    subject="Current row",
):
    """Validate a current text field without SQLite affinity coercion."""
    if value is None and optional:
        return None
    if not isinstance(value, str):
        qualifier = " or None" if optional else ""
        raise error_type(f"{subject} {field} must be a string{qualifier}")
    return value


def _validate_stored_net_current_row(row) -> None:
    """Reject corrupt evidence before a caller can apply any row.

    ``row`` is the JOIN of one ``net_currents`` row with its
    ``net_current_sets`` parent (schema v2, design §6.3): per-row identity
    fields (net_name/pin_id/avg_a/...) and set-level context fields
    (corner/temperature/.../results_file) are validated together here. Since
    every row of one current set shares exactly one parent row, the context
    fields are structurally identical across all rows of a set — there is no
    separate cross-row consistency check to run.
    """
    subject = f"Stored current row {row['current_id']!r}"
    _net_current_text(
        row["net_name"],
        field="net_name",
        error_type=EvidenceMismatchError,
        subject=subject,
    )
    _net_current_text(
        row["pin_id"],
        field="pin_id",
        error_type=EvidenceMismatchError,
        subject=subject,
    )
    _net_current_real(
        row["avg_a"],
        field="avg_a",
        optional=False,
        error_type=EvidenceMismatchError,
        subject=subject,
    )
    for field in ("rms_a", "peak_a"):
        _net_current_real(
            row[field],
            field=field,
            error_type=EvidenceMismatchError,
            subject=subject,
        )
    _net_current_text(
        row["current_vector"],
        field="current_vector",
        optional=True,
        error_type=EvidenceMismatchError,
        subject=subject,
    )
    # Schema v5: the CHECKs tie each state to its value; decode both so a
    # corrupt state or an untyped value is refused before any apply.
    Database._row_condition_scalars(
        row, subject=subject, error_type=EvidenceMismatchError
    )
    _net_current_real(
        row["tstart"],
        field="tstart",
        error_type=EvidenceMismatchError,
        subject=subject,
    )
    for field in ("tool_name", "analysis", "results_file"):
        _net_current_text(
            row[field],
            field=field,
            optional=True,
            error_type=EvidenceMismatchError,
            subject=subject,
        )


def _evaluate_spec_measurement(*, op, value, target):
    comparison = _SPEC_COMPARISONS.get(op)
    if comparison is None:
        raise EvidenceMismatchError(
            f"Stored spec uses unsupported comparison operator {op!r}"
        )
    return bool(comparison(value, target))


def _profile_filter_predicate(profile_id, *, column="profile_id", alias=None):
    """SQL clause/params for a 3-state ``profile_id`` query filter.

    Design SS4.3/SS8 ("Queries and restore accept a profile filter; unscoped
    legacy rows stay reachable and clearly marked"):

    - ``None`` (the default everywhere) means "no filter" -- both scoped and
      unscoped evidence match, exactly as before DB-02.
    - :data:`silk.db.models.UNSCOPED_PROFILE` narrows to explicitly unscoped
      (``profile_id IS NULL``) rows only.
    - A non-empty ``str`` narrows to that exact ``profile_id``.

    Never fabricates or infers a profile — an unrecognised value is a typed
    error, not a silent no-op.
    """
    qualified = f"{alias}.{column}" if alias else column
    if profile_id is None:
        return [], []
    if profile_id is UNSCOPED_PROFILE:
        return [f"{qualified} IS NULL"], []
    if not isinstance(profile_id, str) or not profile_id:
        raise TypeError(
            "profile_id filter must be a non-empty string, "
            "silk.db.UNSCOPED_PROFILE, or None"
        )
    return [f"{qualified} = ?"], [profile_id]


class DesignStage(Enum):
    """Design stage of a circuit snapshot.

    The members are ``DEV`` (``'dev'``), ``STABLE`` (``'stable'``),
    ``POST_LAYOUT`` (``'post_layout'``) and ``TAPEOUT`` (``'tapeout'``). Database
    methods that take a ``stage`` accept a member; some also accept the value
    string.

    Examples
    --------
    >>> from silk.db import DesignStage
    >>> DesignStage('post_layout')
    <DesignStage.POST_LAYOUT: 'post_layout'>
    """
    DEV = "dev"
    STABLE = "stable"
    POST_LAYOUT = "post_layout"
    TAPEOUT = "tapeout"


class Database:
    """SQLite design database holding circuit snapshots and their evidence.

    A snapshot is a ``circuit_versions`` row identified by a
    :class:`~silk.db.SnapshotIdentity` and moves through two states: ``building``
    (structure and spec declarations may be written) and ``sealed`` (evidence may
    be written; declarations are fixed). A version may also be locked, which
    refuses every further write that targets it with
    :class:`~silk.exceptions.LockedVersionError`. Reads of stored evidence follow
    the version's parent lineage and only return rows whose structure identity
    matches the requested snapshot.

    Write methods commit per call unless they run inside :meth:`transaction`.
    The connection uses ``sqlite3.Row`` rows and enforces foreign keys. The
    connection is created with ``check_same_thread=False``.

    Parameters
    ----------
    db_path : str, default "circuit_design.db"
        Path of the SQLite file, or ``':memory:'`` for a private in-memory
        database. A missing or empty file is initialised with the current schema
        (version 6). An existing file is validated first.

    Attributes
    ----------
    db_path : str
        The path passed in, as a string.
    conn : sqlite3.Connection
        The open connection. Closed by :meth:`close` and on leaving a ``with``
        block.

    Raises
    ------
    silk.exceptions.DatabaseCompatibilityError
        The existing file is not a silk database or cannot be read as one.
    silk.exceptions.UnsupportedSchemaVersionError
        The existing file uses a schema version this silk does not support; the
        ``migrate_v*`` functions copy older files forward.

    See Also
    --------
    silk.db.CircuitTracker : Module-bound tracking built on this class.
    silk.db.migrate_v6_to_v7 : Upgrade a schema-v6 database file.

    Examples
    --------
    >>> from silk import Database
    >>> with Database(':memory:') as db:
    ...     enabled = db.conn.execute('PRAGMA foreign_keys').fetchone()[0]
    >>> enabled
    1
    """

    def __init__(self, db_path: str = "circuit_design.db"):
        self.db_path = str(db_path)
        is_memory = self.db_path == ":memory:"
        is_new = is_memory
        if not is_memory:
            path = Path(self.db_path)
            is_new = not path.exists() or path.stat().st_size == 0
            if not is_new:
                inspect_existing_database(path)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            # SQLite ignores every declared FOREIGN KEY unless this
            # per-connection pragma is set.
            self.conn.execute("PRAGMA foreign_keys = ON")
            self._txn_depth = 0   # >0 while inside a transaction() block
            self._commit_actions = []
            self._validation_actions = []
            self._rollback_actions = []
            self._savepoint_serial = 0
            # (reason, cause) once this group can no longer be trusted; the
            # outermost context then refuses to commit it.
            self._unsafe_group = None
            if is_new:
                self._initialize_schema()
            else:
                # Repeat validation on the normal connection so a file changed
                # between preflight and open is never exposed as current.
                validate_database_connection(
                    self.conn,
                    path_label=str(Path(self.db_path).resolve()),
                )
        except BaseException:
            self.conn.close()
            raise

    @contextmanager
    def transaction(self):
        """Group several writes into one atomic transaction.

        Write methods commit per call by default. Inside this context manager they
        defer to a single commit on success, or a rollback if the block or the
        commit fails, so a failure part-way through cannot leave partial SQL state.
        The context is re-entrant: only the outermost context commits, but every
        context is its own rollback boundary.

        Nested scopes run on a SQLite ``SAVEPOINT``. An exception that reaches the
        context manager undoes every SQL change made under that scope, including
        changes made by deeper scopes that already exited successfully, and
        discards the post-commit actions those scopes registered. Work done before
        the scope, or by sibling scopes, is untouched and stays eligible for the
        outermost commit::

            with db.transaction():
                db.add_device(...)
                db.add_net(...)      # both land, or neither

            with db.transaction():                 # outer group
                tracker.store_results({"gain": (60.0, "dB")})
                try:
                    tracker.store_results(bad)     # raises part-way through
                except TypeError:
                    pass                           # caught OUTSIDE that save
            # committed: the first save only. The failed save left no rows.

        The exception has to reach the context manager. One swallowed inside the
        same ``with`` block, before the block exits, is invisible here and nothing
        is rolled back for it; put the failing work in its own ``transaction()``
        when you intend to recover from it. Tracker bindings follow SQL immediately
        and are restored in reverse order when a scope rolls back. An outer commit
        discards those restores before post-commit actions run.

        Yields
        ------
        Database
            This database, so ``with db.transaction() as db:`` is also valid.

        Raises
        ------
        silk.exceptions.DatabaseError
            A nested scope was entered after SQLite discarded the enclosing
            transaction, or the group was poisoned because a nested rollback could
            not be confirmed; the outermost context then rolls everything back
            instead of committing rows it could not verify.
        Exception
            Whatever the block raises is re-raised after the rollback.
        """
        outermost = self._txn_depth == 0
        if outermost:
            self.conn.execute("BEGIN IMMEDIATE")
            savepoint = None
        else:
            self._require_open_group()
            # Unique per scope: a stale same-named savepoint left behind by a
            # connection-level abort must never be mistaken for this one.
            self._savepoint_serial += 1
            savepoint = f"silk_txn_{self._savepoint_serial}"
            self.conn.execute(f"SAVEPOINT {savepoint}")
        # Post-commit actions are a flat queue; this mark is what makes them
        # scope-owned. Everything appended from here on belongs to this scope
        # or to a scope nested under it.
        action_mark = len(self._commit_actions)
        validation_mark = len(self._validation_actions)
        undo_mark = len(self._rollback_actions)
        self._txn_depth += 1
        try:
            yield self
        except BaseException:
            self._txn_depth -= 1
            del self._commit_actions[action_mark:]
            del self._validation_actions[validation_mark:]
            try:
                if outermost:
                    self._unsafe_group = None
                    self.conn.rollback()
                else:
                    self._rollback_savepoint(savepoint)
            finally:
                self._undo_since(undo_mark)
            raise
        else:
            self._txn_depth -= 1
            if not outermost:
                # Merge into the enclosing transaction. RELEASE inside a BEGIN
                # is not a commit: nothing reaches the file until the
                # outermost context commits.
                try:
                    self.conn.execute(f"RELEASE {savepoint}")
                except BaseException:
                    # The scope did NOT merge. Give it the same cleanup an
                    # exception inside it would have got: a caller that
                    # catches this error must not be able to commit the
                    # scope's rows or publish its actions later.
                    del self._commit_actions[action_mark:]
                    del self._validation_actions[validation_mark:]
                    try:
                        self._rollback_savepoint(savepoint)
                    finally:
                        self._undo_since(undo_mark)
                    raise
            else:
                actions, self._commit_actions = self._commit_actions, []
                validations, self._validation_actions = self._validation_actions, []
                unsafe, self._unsafe_group = self._unsafe_group, None
                if unsafe is not None:
                    # Rows this group must not keep may still be live, or the
                    # group is no longer the transaction it started as.
                    # Discard it rather than commit unverified state.
                    reason, cause = unsafe
                    try:
                        self.conn.rollback()
                    finally:
                        self._undo_since(undo_mark)
                    raise DatabaseError(
                        f"Transaction rolled back: {reason}, so the group "
                        "was not committed"
                    ) from cause
                try:
                    for validate in validations:
                        validate()
                    self._maybe_commit()
                except BaseException:
                    try:
                        self.conn.rollback()
                    finally:
                        self._undo_since(undo_mark)
                    raise
                # Committed state must survive even a failing post-commit action.
                del self._rollback_actions[undo_mark:]
                # SQL is committed. Live recording errors must propagate
                # without claiming a rollback or retaining actions for replay.
                for action in actions:
                    action()

    def _require_open_group(self):
        """Refuse a nested scope the enclosing transaction can no longer hold.

        A ``SAVEPOINT`` issued outside a transaction OPENS one, and releasing
        that outermost savepoint COMMITS it -- so a nested save entered after
        SQLite discarded the enclosing transaction (an interrupted write, a
        full disk) would commit on the spot, before the outer block exits,
        and survive the rollback that block then reports.
        """
        if self._unsafe_group is not None:
            reason, cause = self._unsafe_group
            raise DatabaseError(
                f"Cannot open a nested transaction(): {reason}"
            ) from cause
        if not self.conn.in_transaction:
            error = DatabaseError(
                "Cannot open a nested transaction(): the enclosing SQL "
                "transaction is gone (SQLite discarded it, e.g. an "
                "interrupted write), so a nested scope would open -- and on "
                "release commit -- a transaction of its own"
            )
            # Even if the caller catches this, the group must not report
            # success: what it wrote before the abort is already gone.
            self._unsafe_group = (
                "SQLite discarded the enclosing transaction", error
            )
            raise error

    def _rollback_savepoint(self, savepoint):
        """Undo one nested scope, keeping the enclosing transaction usable.

        Called while another exception is in flight, so this never raises --
        but it must not pretend either. A refused ``ROLLBACK TO`` is recorded
        so the outermost context discards the whole group rather than commit
        rows nothing proved were undone.
        """
        try:
            self.conn.execute(f"ROLLBACK TO {savepoint}")
        except sqlite3.Error as error:
            # The scope's rows may still be live. Assuming SQLite already
            # aborted the enclosing transaction is exactly the assumption
            # that lets a failed save commit.
            self._unsafe_group = (
                "a nested scope's rollback could not be confirmed", error
            )
            return
        try:
            self.conn.execute(f"RELEASE {savepoint}")
        except sqlite3.Error:
            # The rows are confirmed gone; only the (unique, never reused)
            # stack entry is left behind, which nothing can match against.
            pass

    def _after_rollback(self, action):
        """Register a non-throwing internal state restore in the active scope."""
        if self._txn_depth == 0:
            raise RuntimeError("Rollback actions require an active transaction()")
        self._rollback_actions.append(action)

    def _undo_since(self, mark):
        """Detach a scope's restores before applying them in reverse order."""
        actions = self._rollback_actions[mark:]
        del self._rollback_actions[mark:]
        for action in reversed(actions):
            action()

    def _before_commit(self, action):
        """Validate staged publication before SQL commits; discard on rollback."""
        if self._txn_depth == 0:
            raise RuntimeError("Pre-commit actions require an active transaction()")
        self._validation_actions.append(action)

    def _after_commit(self, action):
        """Stage an internal storage side effect in the active transaction."""
        if self._txn_depth == 0:
            raise RuntimeError("Commit-bound actions require an active transaction()")
        self._commit_actions.append(action)

    def _maybe_commit(self):
        """Commit now, unless inside a ``transaction()`` block (then defer)."""
        if self._txn_depth == 0:
            self.conn.commit()

    @staticmethod
    def _creator_version():
        try:
            return package_version("circuitsilk")
        except PackageNotFoundError:
            return None

    def _initialize_schema(self, *, schema_version=DATABASE_SCHEMA_VERSION):
        """Create and identify a new database as one atomic operation."""
        with self.transaction():
            self._create_tables(schema_version=schema_version)
            self.conn.execute(
                """
                INSERT INTO silk_metadata
                    (singleton, format_id, schema_version, creator_version)
                VALUES (1, ?, ?, ?)
                """,
                (
                    DATABASE_FORMAT,
                    schema_version,
                    self._creator_version(),
                ),
            )
            _validate_database_version(
                self.conn, path_label=self.db_path, schema_version=schema_version
            )

    @staticmethod
    def _create_spec_measurements_table(
        conn, *, schema_version=DATABASE_SCHEMA_VERSION,
        table_name="spec_measurements",
    ):
        """Keep v2's NOT NULL constraint for its explicit copy destination."""
        if table_name not in ("spec_measurements", "_silk_spec_measurements_v3"):
            raise ValueError("Unexpected measurement table name")
        conditions_sql = Database._row_conditions_sql(
            schema_version=schema_version
        )
        # Spec measurements table - one row per observation. circuit_version_id
        # is redundant with specs.circuit_version_id (reachable via spec_id)
        # but is stored directly so SQLite can prove a measurement and its
        # profile share one origin version via the composite FK below,
        # instead of trusting an application-level join (design §6.5).
        conn.execute(f"""
            CREATE TABLE {table_name} (
                measurement_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                spec_id TEXT NOT NULL,
                profile_id TEXT,  -- NULL = explicitly unscoped/legacy/manual
                source TEXT NOT NULL CHECK(source IN ('behavioral', 'spice', 'extraction', 'user')),
                value REAL {"NOT NULL" if schema_version == 2 else ""},
                corner TEXT,
                temperature REAL,
                voltage REAL,
                sim_type TEXT,
                tool_name TEXT,
                timestamp REAL,
                sim_run_id TEXT,
                notes TEXT,
                passes INTEGER,
                {conditions_sql},
                FOREIGN KEY (circuit_version_id, spec_id)
                    REFERENCES specs(circuit_version_id, spec_id) ON DELETE CASCADE,
                FOREIGN KEY (circuit_version_id, profile_id)
                    REFERENCES characterization_profiles(circuit_version_id, profile_id)
                    ON DELETE RESTRICT
            )
        """)
        if schema_version >= 6:
            conn.execute('ALTER TABLE spec_measurements ADD COLUMN aggregate_json TEXT')
        if schema_version >= 4:
            conn.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {_PROVENANCE_COLUMN_SQL}")

    @staticmethod
    def _row_conditions_sql(*, schema_version=DATABASE_SCHEMA_VERSION):
        """Condition columns/CHECKs shared by the three evidence tables.

        Before v5 a scoped row needed non-NULL corner/temperature/voltage.
        Schema v5 stores a state per fact (design 2026-10-01
        operating-conditions-none, §5.1): a value is ``'fixed'`` with a
        non-NULL column, none/unknown/sweep hold NULL, and a scoped row has
        every fact fixed or none.
        """
        if schema_version >= 5:
            return f"{state_columns_sql(schema_version=schema_version)},\n                {profile_check_sql()}"
        return (
            "CHECK(profile_id IS NULL OR\n"
            "                      (corner IS NOT NULL AND temperature IS NOT NULL AND\n"
            "                       voltage IS NOT NULL))"
        )

    @staticmethod
    def _current_set_unique_index_sql(schema_version=DATABASE_SCHEMA_VERSION):
        """At most one current set per selection key.

        In v5 a none or unknown fact stores NULL, and SQLite treats NULLs as
        distinct in a unique index. The v5 keys therefore index each state
        with ``ifnull(value, '')``: the state tells none from unknown, and
        ``''`` (TEXT) never equals a stored value of either type.
        """
        if schema_version >= 5:
            return (
                """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_current_sets_scoped_current
            ON net_current_sets(
                circuit_version_id, profile_id,
                corner_state, ifnull(corner, ''),
                temperature_state, ifnull(temperature, ''),
                voltage_state, ifnull(voltage, '')
            )
            WHERE state = 'current' AND profile_id IS NOT NULL
        """,
                """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_current_sets_unscoped_current
            ON net_current_sets(
                circuit_version_id, corner_state, ifnull(corner, '')
            )
            WHERE state = 'current' AND profile_id IS NULL
        """,
            )
        return (
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_current_sets_scoped_current
            ON net_current_sets(
                circuit_version_id, profile_id, corner, temperature, voltage
            )
            WHERE state = 'current' AND profile_id IS NOT NULL
        """,
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_current_sets_unscoped_current
            ON net_current_sets(circuit_version_id, corner)
            WHERE state = 'current' AND profile_id IS NULL
        """,
        )

    @staticmethod
    def _create_run_provenance_table(conn):
        """The schema-v4 run_provenance table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_provenance (
                provenance_id TEXT PRIMARY KEY,
                scheme        TEXT NOT NULL,
                digest        TEXT NOT NULL,
                record_json   TEXT NOT NULL,
                created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(scheme, digest)
            )
        """)

    def _create_tables(self, *, schema_version=DATABASE_SCHEMA_VERSION):
        """Create all necessary database tables"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE silk_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                format_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                creator_version TEXT
            )
        """)

        # Circuit versions table - main design objects
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_versions (
                version_id TEXT PRIMARY KEY,
                circuit_name TEXT NOT NULL,
                version TEXT NOT NULL,
                stage TEXT NOT NULL CHECK(stage IN ('dev', 'stable', 'post_layout', 'tapeout')),
                state TEXT NOT NULL CHECK(state IN ('building', 'sealed')),
                is_locked BOOLEAN DEFAULT 0,
                parent_version_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                description TEXT,
                netlist_path TEXT,
                metadata TEXT,  -- JSON string for flexible data
                simulation_summary TEXT,  -- Auto-generated summary of all simulations
                last_summary_update TIMESTAMP,
                structure_scheme TEXT NOT NULL,
                structure_hash TEXT NOT NULL,
                spec_scheme TEXT NOT NULL,
                spec_set_hash TEXT NOT NULL,
                -- (circuit_name, version) is deliberately NOT unique at the
                -- schema level: CircuitTracker.resolve_version keys rows by
                -- structure_hash and reuses one version LABEL across them.
                -- The partial sealed-identity index below prevents duplicate
                -- snapshots without treating human labels as identifiers.
                FOREIGN KEY (parent_version_id) REFERENCES circuit_versions(version_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                model TEXT,
                properties TEXT,  -- JSON for additional properties
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            )
        """)

        # Nets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nets (
                net_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                name TEXT NOT NULL,
                net_type TEXT,  -- signal, power, ground, etc.
                properties TEXT,  -- JSON for additional properties
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            )
        """)

        # Run provenance (TASK-306 / schema v4) — one content-addressed row
        # per distinct silk.simulation.provenance record; measurement-type rows point at
        # it through a nullable provenance_id (NULL = none recorded, the
        # value every migrated pre-v4 row keeps).
        if schema_version >= 4:
            self._create_run_provenance_table(self.conn)

        # Characterization profiles (DB-02 / schema v2) — a profile's logical
        # identity is (circuit_version_id, profile_scheme, profile_hash).
        # See docs/database.md for the public characterization-profile workflow.
        # profile_id is the persisted UUID FK endpoint; it is immutable
        # and unrelated to the logical digest. Rows are append-only: no
        # profile-delete API exists in schema v2.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS characterization_profiles (
                profile_id         TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                owner_path         TEXT NOT NULL,
                test_name          TEXT NOT NULL,
                parameters_json    TEXT NOT NULL,
                profile_scheme     TEXT NOT NULL,
                profile_hash       TEXT NOT NULL,
                created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(circuit_version_id, profile_scheme, profile_hash),
                UNIQUE(circuit_version_id, profile_id),
                FOREIGN KEY (circuit_version_id)
                    REFERENCES circuit_versions(version_id) ON DELETE RESTRICT
            )
        """)

        # Simulation results table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS simulation_results (
                sim_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                object_id TEXT,  -- Can reference transistor, net, or circuit
                object_type TEXT CHECK(object_type IN ('circuit', 'transistor', 'net')),
                source TEXT NOT NULL,
                sim_type TEXT,  -- dc, ac, tran, noise, etc.
                corner TEXT,  -- Process corner: tt, ss, ff, sf, fs, etc.
                temperature REAL,  -- Temperature in Celsius
                voltage REAL,  -- Supply voltage
                parameter TEXT NOT NULL,
                value REAL,
                value_text TEXT,  -- For non-numeric results
                unit TEXT,
                tool_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                conditions TEXT,  -- JSON for additional simulation conditions
                profile_id TEXT,  -- NULL = explicitly unscoped/legacy/manual
                {self._row_conditions_sql(schema_version=schema_version)},
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id),
                FOREIGN KEY (circuit_version_id, profile_id)
                    REFERENCES characterization_profiles(circuit_version_id, profile_id)
                    ON DELETE RESTRICT
            )
        """)
        if schema_version >= 4:
            cursor.execute(
                f"ALTER TABLE simulation_results ADD COLUMN {_PROVENANCE_COLUMN_SQL}")

        # Parasitics table (for post-layout data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parasitics (
                parasitic_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                net_id TEXT NOT NULL,
                parasitic_type TEXT CHECK(parasitic_type IN ('resistance', 'capacitance', 'inductance')),
                value REAL,
                unit TEXT,
                from_node TEXT,
                to_node TEXT,
                extraction_tool TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id),
                FOREIGN KEY (net_id) REFERENCES nets(net_id)
            )
        """)

        # Selected current sets (DB-02 / schema v2) — the stable, immutable
        # FK endpoint for one selected-evidence current parent. Replacing the
        # selected set for one (version, profile, corner, temperature,
        # voltage) mints a new row and marks the old one superseded; both are
        # retained (§4.2). Unscoped (profile_id IS NULL) evidence keeps v1's
        # one-current-set-per-(version, corner) behavior.
        # Schema v5: none and unknown corners store NULL, so the corner
        # loses NOT NULL and the profile CHECK becomes the state check.
        if schema_version >= 5:
            corner_not_null = ""
            # Columns precede the table constraints in SQLite's grammar.
            current_set_state_columns_sql, current_set_conditions_sql = (
                self._row_conditions_sql(
                    schema_version=schema_version
                ).split(",\n                CHECK(profile_id", 1)
            )
            current_set_state_columns_sql = (
                "\n                " + current_set_state_columns_sql + ","
            )
            current_set_conditions_sql = (
                "CHECK(profile_id" + current_set_conditions_sql
            )
        else:
            current_set_state_columns_sql = ""
            corner_not_null = " NOT NULL"
            current_set_conditions_sql = (
                "CHECK(profile_id IS NULL OR\n"
                "                      (temperature IS NOT NULL AND voltage IS NOT NULL))"
            )
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS net_current_sets (
                current_set_id             TEXT PRIMARY KEY,
                circuit_version_id         TEXT NOT NULL,
                profile_id                 TEXT,
                state                      TEXT NOT NULL
                                           CHECK(state IN ('current', 'superseded')),
                supersedes_current_set_id  TEXT UNIQUE,
                corner                     TEXT{corner_not_null},
                temperature                REAL,
                voltage                    REAL,
                tool_name                  TEXT,
                analysis                   TEXT,
                tstart                     REAL,
                results_file               TEXT,
                structure_scheme           TEXT NOT NULL,
                structure_hash             TEXT NOT NULL,
                selected_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                superseded_at              TIMESTAMP,{current_set_state_columns_sql}
                UNIQUE(circuit_version_id, current_set_id),
                {current_set_conditions_sql},
                CHECK((state = 'current' AND superseded_at IS NULL) OR
                      (state = 'superseded' AND superseded_at IS NOT NULL)),
                FOREIGN KEY (circuit_version_id)
                    REFERENCES circuit_versions(version_id) ON DELETE RESTRICT,
                FOREIGN KEY (circuit_version_id, profile_id)
                    REFERENCES characterization_profiles(circuit_version_id, profile_id)
                    ON DELETE RESTRICT,
                FOREIGN KEY (supersedes_current_set_id)
                    REFERENCES net_current_sets(current_set_id) ON DELETE RESTRICT
            )
        """)
        if schema_version >= 4:
            cursor.execute(
                f"ALTER TABLE net_current_sets ADD COLUMN {_PROVENANCE_COLUMN_SQL}")

        # Net operating currents (EM) — spec 2026-07-18, rebuilt for schema v2
        # (design §6.3). pin_id '' = the net-level aggregate row (½·Σ bounds);
        # non-empty = one pin's signed currents. Sentinels (not NULL) keep the
        # UNIQUE real: SQLite treats NULLs as pairwise-distinct in unique
        # indexes. Set-level context (corner/temperature/voltage/tool/
        # structure identity/etc.) now lives on the parent net_current_sets
        # row; this table keeps only per-row identity and values. Partial nets
        # store pin rows only — an absent aggregate row round-trips as None,
        # never 0.0.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS net_currents (
                current_id     TEXT PRIMARY KEY,
                current_set_id TEXT NOT NULL,
                net_name       TEXT NOT NULL,
                pin_id         TEXT NOT NULL DEFAULT '',
                avg_a          REAL NOT NULL,
                rms_a          REAL,
                peak_a         REAL,
                source         TEXT NOT NULL CHECK(source IN ('spice', 'user')),
                current_vector TEXT,
                timestamp      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(current_set_id, net_name, pin_id),
                FOREIGN KEY (current_set_id)
                    REFERENCES net_current_sets(current_set_id) ON DELETE RESTRICT
            )
        """)

        # Stage releases table - snapshots for teams
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS releases (
                release_id TEXT PRIMARY KEY,
                release_name TEXT UNIQUE NOT NULL,
                stage TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT,
                notes TEXT,
                status TEXT DEFAULT 'active' CHECK(status IN ('active', 'superseded', 'archived'))
            )
        """)

        # Release contents - which circuits are in each release
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS release_contents (
                release_id TEXT NOT NULL,
                circuit_version_id TEXT NOT NULL,
                PRIMARY KEY (release_id, circuit_version_id),
                FOREIGN KEY (release_id) REFERENCES releases(release_id),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            )
        """)

        # ===== NEW MODULE TRACKING SCHEMA =====

        # Modules table - stores unique module definitions
        # A module is uniquely identified by its content hash
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modules (
                module_id TEXT PRIMARY KEY,
                module_name TEXT NOT NULL,
                module_path TEXT NOT NULL,
                module_hash_scheme TEXT NOT NULL,
                module_hash TEXT NOT NULL,
                device_count INTEGER DEFAULT 0,
                structure TEXT,  -- JSON of module structure (devices, nets, submodules)
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(module_hash_scheme, module_hash)
            )
        """)

        # Junction table - links circuits to modules (many-to-many)
        # Allows same module to be reused across multiple circuit versions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_modules (
                circuit_version_id TEXT NOT NULL,
                module_id TEXT NOT NULL,
                instance_name TEXT,  -- Instance name in the circuit (e.g., 'ldo_output_1')
                position_in_hierarchy INTEGER,  -- Order/depth in hierarchy
                PRIMARY KEY (circuit_version_id, module_id, instance_name),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id) ON DELETE CASCADE,
                FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
            )
        """)

        # ===== END NEW MODULE TRACKING SCHEMA =====

        # Specs table - declared targets (Pillar 1 of the design-stage bridge)
        # UNIQUE(circuit_version_id, spec_id) adds no new real-world
        # constraint (spec_id is already a globally unique PK) but SQLite
        # requires an explicit UNIQUE/PK on the referenced column tuple for
        # spec_measurements' composite FK below (design §6.5).
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS specs (
                spec_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                owner_path TEXT NOT NULL,
                spec_key TEXT NOT NULL,
                name TEXT NOT NULL,
                metric TEXT NOT NULL,
                op TEXT NOT NULL CHECK(op IN ('<', '<=', '>', '>=', '==')),
                target REAL NOT NULL,
                units TEXT,
                description TEXT,
                parent_id TEXT,
                declared_at TEXT,
                definition_scheme TEXT NOT NULL,
                definition_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(circuit_version_id, owner_path, spec_key),
                UNIQUE(circuit_version_id, spec_id),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES specs(spec_id)
            )
        """)

        self._create_spec_measurements_table(
            self.conn, schema_version=schema_version
        )

        # Create useful indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circuit_name_stage 
            ON circuit_versions(circuit_name, stage, created_at DESC)
        """)

        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_circuit_sealed_identity
            ON circuit_versions(
                circuit_name, stage, structure_scheme, structure_hash,
                spec_scheme, spec_set_hash
            )
            WHERE state = 'sealed'
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circuit_parent
            ON circuit_versions(parent_version_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sim_results_circuit 
            ON simulation_results(circuit_version_id, parameter)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sim_results_context
            ON simulation_results(
                circuit_version_id, profile_id, parameter, source, sim_type,
                corner, temperature, voltage, tool_name, timestamp
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiles_owner_test
            ON characterization_profiles(circuit_version_id, owner_path, test_name)
        """)

        for sql in self._current_set_unique_index_sql(schema_version):
            cursor.execute(sql)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_current_sets_profile_history
            ON net_current_sets(profile_id, corner, temperature, voltage, selected_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_net_currents_set
            ON net_currents(current_set_id, net_name, pin_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_hash
            ON modules(module_hash_scheme, module_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_modules_name 
            ON modules(module_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circuit_modules_circuit 
            ON circuit_modules(circuit_version_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_circuit_modules_module
            ON circuit_modules(module_id)
        """)

        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_specs_owner_key
            ON specs(circuit_version_id, owner_path, spec_key)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_specs_definition
            ON specs(definition_scheme, definition_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_spec_measurements_spec
            ON spec_measurements(spec_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_spec_measurements_context
            ON spec_measurements(
                circuit_version_id, profile_id, spec_id, source, corner,
                temperature, voltage, tool_name, timestamp
            )
        """)

        self._maybe_commit()

    @staticmethod
    def _stage_value(stage) -> str:
        if isinstance(stage, DesignStage):
            return stage.value
        try:
            return DesignStage(stage).value
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unknown design stage {stage!r}; expected one of "
                f"{[item.value for item in DesignStage]}"
            ) from exc

    @staticmethod
    def _validate_identity(identity: SnapshotIdentity) -> None:
        if not isinstance(identity, SnapshotIdentity):
            raise TypeError(
                "identity must be a SnapshotIdentity, got "
                f"{type(identity).__name__}"
            )
        _require_fingerprint_scheme(
            identity.structure, STRUCTURE_FINGERPRINT_SCHEME
        )
        _require_fingerprint_scheme(identity.spec_set, SPEC_FINGERPRINT_SCHEME)

    def _snapshot_row(self, version_id: str):
        row = self.conn.execute(
            """
            SELECT version_id, circuit_name, version, stage, state, is_locked,
                   parent_version_id, structure_scheme, structure_hash,
                   spec_scheme, spec_set_hash
            FROM circuit_versions
            WHERE version_id = ?
            """,
            (version_id,),
        ).fetchone()
        if row is None:
            raise DatabaseError(f"Circuit version {version_id!r} does not exist")
        return row

    def _require_state(self, version_id: str, expected: str):
        row = self._snapshot_row(version_id)
        if row["state"] != expected:
            raise DatabaseError(
                f"Circuit '{row['circuit_name']}' version '{row['version']}' "
                f"is {row['state']!r}; this write requires a {expected!r} "
                "snapshot"
            )
        return row

    def _require_building(self, version_id: str):
        self._require_unlocked(version_id)
        return self._require_state(version_id, "building")

    def _require_sealed(self, version_id: str):
        self._require_unlocked(version_id)
        return self._require_state(version_id, "sealed")

    @staticmethod
    def _characterization_profile_record(row) -> CharacterizationProfileRecord:
        stored_fingerprint = Fingerprint(
            row["profile_scheme"], row["profile_hash"]
        )
        _require_fingerprint_scheme(
            stored_fingerprint, PROFILE_FINGERPRINT_SCHEME
        )
        for field_name in (
            "profile_id",
            "circuit_version_id",
            "owner_path",
            "test_name",
        ):
            value = row[field_name]
            if not isinstance(value, str) or not value:
                raise EvidenceMismatchError(
                    "Stored characterization profile has an invalid "
                    f"{field_name}"
                )

        parameters, parameter_entries = (
            _decode_characterization_profile_parameters(
                row["parameters_json"]
            )
        )
        _payload_json, computed_fingerprint = (
            _fingerprint_characterization_profile_fields(
                row["owner_path"],
                row["test_name"],
                parameter_entries,
            )
        )
        if not _fingerprints_equal(
            stored_fingerprint,
            computed_fingerprint,
            expected_scheme=PROFILE_FINGERPRINT_SCHEME,
        ):
            raise EvidenceMismatchError(
                "Stored characterization profile digest does not match its "
                "canonical owner, test, and parameter payload"
            )
        return CharacterizationProfileRecord(
            profile_id=row["profile_id"],
            origin_version_id=row["circuit_version_id"],
            owner_path=row["owner_path"],
            test_name=row["test_name"],
            parameters=parameters,
            profile_scheme=stored_fingerprint.scheme,
            profile_hash=stored_fingerprint.digest,
            created_at=row["created_at"],
        )

    @staticmethod
    def _profile_payload_matches(row, context, canonical) -> bool:
        return (
            row["owner_path"] == context.owner_path
            and row["test_name"] == context.test_name
            and row["parameters_json"] == canonical.parameters_json
        )

    def store_characterization_profile(
        self,
        circuit_version_id: str,
        context: CharacterizationContext,
    ) -> CharacterizationProfileRecord:
        """Store, or reuse, the characterization profile of a context on a sealed snapshot.

        The profile identity is derived from the context's owner path, test name
        and declared parameters; the process, temperature and voltage point is not
        part of it. A profile already stored for the same payload is returned
        unchanged.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed, unlocked snapshot that will own the profile.
        context : CharacterizationContext
            The complete runtime descriptor. A display label or precomputed digest
            is never accepted.

        Returns
        -------
        CharacterizationProfileRecord
            The new or existing profile.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not sealed.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        silk.exceptions.EvidenceMismatchError
            A stored profile with the same digest has a different canonical
            payload, or a stored row is corrupt.
        """
        with self.transaction():
            self._require_sealed(circuit_version_id)
            canonical = _canonicalize_characterization_profile(context)
            row = self.conn.execute(
                """
                SELECT profile_id, circuit_version_id, owner_path, test_name,
                       parameters_json, profile_scheme, profile_hash, created_at
                FROM characterization_profiles
                WHERE circuit_version_id = ?
                  AND profile_scheme = ? AND profile_hash = ?
                """,
                (
                    circuit_version_id,
                    canonical.fingerprint.scheme,
                    canonical.fingerprint.digest,
                ),
            ).fetchone()
            if row is not None:
                record = self._characterization_profile_record(row)
                if not self._profile_payload_matches(row, context, canonical):
                    raise EvidenceMismatchError(
                        "Stored characterization profile identity conflicts "
                        "with the supplied canonical descriptor"
                    )
                return record

            profile_id = str(uuid.uuid4())
            self.conn.execute(
                """
                INSERT INTO characterization_profiles
                    (profile_id, circuit_version_id, owner_path, test_name,
                     parameters_json, profile_scheme, profile_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile_id,
                    circuit_version_id,
                    context.owner_path,
                    context.test_name,
                    canonical.parameters_json,
                    canonical.fingerprint.scheme,
                    canonical.fingerprint.digest,
                ),
            )
            row = self.conn.execute(
                """
                SELECT profile_id, circuit_version_id, owner_path, test_name,
                       parameters_json, profile_scheme, profile_hash, created_at
                FROM characterization_profiles
                WHERE circuit_version_id = ? AND profile_id = ?
                """,
                (circuit_version_id, profile_id),
            ).fetchone()
            return self._characterization_profile_record(row)

    def get_characterization_profile(
        self,
        circuit_version_id: str,
        *,
        profile_id: Optional[str] = None,
        context: Optional[CharacterizationContext] = None,
    ) -> CharacterizationProfileRecord:
        """Look up one profile owned by an exact snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot that owns the profile. Ancestors are not
            searched; see :meth:`resolve_characterization_profile`.
        profile_id : str, optional
            Exact profile id. Give exactly one of ``profile_id`` and ``context``.
        context : CharacterizationContext, optional
            Complete descriptor to match by identity.

        Returns
        -------
        CharacterizationProfileRecord
            The matching profile.

        Raises
        ------
        ValueError
            Both or neither of ``profile_id`` and ``context`` were given, or
            ``profile_id`` is empty.
        TypeError
            ``profile_id`` is not a string.
        silk.exceptions.DatabaseError
            The snapshot does not exist.
        silk.exceptions.EvidenceNotFoundError
            The snapshot owns no such profile.
        silk.exceptions.EvidenceMismatchError
            The stored profile conflicts with the supplied descriptor, or is corrupt.
        """
        if (profile_id is None) == (context is None):
            raise ValueError(
                "Profile lookup requires exactly one of profile_id or context"
            )
        self._snapshot_row(circuit_version_id)

        canonical = None
        if profile_id is not None:
            if not isinstance(profile_id, str):
                raise TypeError("profile_id must be a string")
            if not profile_id:
                raise ValueError("profile_id must be a non-empty string")
            row = self.conn.execute(
                """
                SELECT profile_id, circuit_version_id, owner_path, test_name,
                       parameters_json, profile_scheme, profile_hash, created_at
                FROM characterization_profiles
                WHERE circuit_version_id = ? AND profile_id = ?
                """,
                (circuit_version_id, profile_id),
            ).fetchone()
        else:
            canonical = _canonicalize_characterization_profile(context)
            row = self.conn.execute(
                """
                SELECT profile_id, circuit_version_id, owner_path, test_name,
                       parameters_json, profile_scheme, profile_hash, created_at
                FROM characterization_profiles
                WHERE circuit_version_id = ?
                  AND profile_scheme = ? AND profile_hash = ?
                """,
                (
                    circuit_version_id,
                    canonical.fingerprint.scheme,
                    canonical.fingerprint.digest,
                ),
            ).fetchone()

        if row is None:
            selector = (
                f"profile_id {profile_id!r}"
                if profile_id is not None
                else "the supplied complete descriptor"
            )
            raise EvidenceNotFoundError(
                f"No characterization profile for {selector} on circuit "
                f"version {circuit_version_id!r}"
            )

        record = self._characterization_profile_record(row)
        if canonical is not None and not self._profile_payload_matches(
            row, context, canonical
        ):
            raise EvidenceMismatchError(
                "Stored characterization profile identity conflicts with the "
                "supplied canonical descriptor"
            )
        return record

    def _profile_lineage_by_structure(
        self,
        circuit_version_id: str,
        *,
        include_ancestors: bool,
    ):
        """Return nearest-first compatible and incompatible lineage IDs."""
        target = self._snapshot_row(circuit_version_id)
        if target["state"] != "sealed":
            raise DatabaseError(
                "Characterization-profile readback requires a sealed "
                f"snapshot; circuit version {circuit_version_id!r} is "
                f"{target['state']!r}"
            )
        target_fingerprint = Fingerprint(
            target["structure_scheme"], target["structure_hash"]
        )
        _require_fingerprint_scheme(
            target_fingerprint, STRUCTURE_FINGERPRINT_SCHEME
        )
        lineage = (
            self.iter_version_lineage(circuit_version_id)
            if include_ancestors
            else [circuit_version_id]
        )
        compatible = []
        incompatible = []
        for version_id in lineage:
            version = self._snapshot_row(version_id)
            if version["state"] != "sealed":
                raise DatabaseError(
                    "Characterization-profile lineage contains an unsealed "
                    f"circuit version {version_id!r}"
                )
            if version["circuit_name"] != target["circuit_name"]:
                raise DatabaseError(
                    "Characterization-profile lineage crosses circuit names "
                    f"at version {version_id!r}"
                )
            fingerprint = Fingerprint(
                version["structure_scheme"], version["structure_hash"]
            )
            _require_fingerprint_scheme(
                fingerprint, STRUCTURE_FINGERPRINT_SCHEME
            )
            destination = (
                compatible
                if _fingerprints_equal(
                    target_fingerprint,
                    fingerprint,
                    expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                )
                else incompatible
            )
            destination.append(version_id)
        return compatible, incompatible

    def list_characterization_profiles(
        self,
        circuit_version_id: str,
        *,
        include_ancestors: bool = True,
    ) -> List[CharacterizationProfileRecord]:
        """List the profiles visible from a sealed snapshot.

        Only snapshots in the lineage whose structure identity equals the
        snapshot's own contribute.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed snapshot.
        include_ancestors : bool, default True
            Also read the snapshot's ancestors; ``False`` reads only the snapshot.

        Returns
        -------
        list of CharacterizationProfileRecord
            Profiles ordered by lineage distance (the snapshot first), then by
            creation order within a snapshot. Empty when none are stored.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot or a lineage member does not exist, is not sealed, or the
            lineage crosses circuit names.
        silk.exceptions.UnsupportedFingerprintSchemeError
            A stored structure identity uses an unsupported scheme.
        """
        compatible, _incompatible = self._profile_lineage_by_structure(
            circuit_version_id,
            include_ancestors=include_ancestors,
        )
        records = []
        for version_id in compatible:
            rows = self.conn.execute(
                """
                SELECT profile_id, circuit_version_id, owner_path, test_name,
                       parameters_json, profile_scheme, profile_hash, created_at
                FROM characterization_profiles
                WHERE circuit_version_id = ?
                ORDER BY created_at, rowid
                """,
                (version_id,),
            ).fetchall()
            records.extend(
                self._characterization_profile_record(row) for row in rows
            )
        return records

    def resolve_characterization_profile(
        self,
        circuit_version_id: str,
        *,
        profile_id: Optional[str] = None,
        context: Optional[CharacterizationContext] = None,
        include_ancestors: bool = True,
    ) -> CharacterizationProfileRecord:
        """Resolve a profile by exact id or complete descriptor within a snapshot's lineage.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed snapshot.
        profile_id : str, optional
            Exact profile id. Give exactly one of ``profile_id`` and ``context``.
        context : CharacterizationContext, optional
            Complete descriptor to match by identity.
        include_ancestors : bool, default True
            Also search the snapshot's ancestors; the nearest match wins.

        Returns
        -------
        CharacterizationProfileRecord
            The nearest structure-compatible match.

        Raises
        ------
        ValueError
            Both or neither selector was given, or ``profile_id`` is empty.
        TypeError
            ``profile_id`` is not a string.
        silk.exceptions.DatabaseError
            The snapshot is missing or unsealed, or the lineage is invalid.
        silk.exceptions.EvidenceNotFoundError
            No structure-compatible profile exists in the lineage.
        silk.exceptions.EvidenceMismatchError
            The profile exists only on an incompatible structure or outside the
            requested lineage, or conflicts with the supplied descriptor.
        """
        if (profile_id is None) == (context is None):
            raise ValueError(
                "Profile resolution requires exactly one of profile_id or "
                "context"
            )
        if profile_id is not None:
            if not isinstance(profile_id, str):
                raise TypeError("profile_id must be a string")
            if not profile_id:
                raise ValueError("profile_id must be a non-empty string")
        compatible, incompatible = self._profile_lineage_by_structure(
            circuit_version_id,
            include_ancestors=include_ancestors,
        )

        canonical = (
            None
            if context is None
            else _canonicalize_characterization_profile(context)
        )
        for version_id in compatible:
            if profile_id is not None:
                row = self.conn.execute(
                    """
                    SELECT profile_id, circuit_version_id, owner_path,
                           test_name, parameters_json, profile_scheme,
                           profile_hash, created_at
                    FROM characterization_profiles
                    WHERE circuit_version_id = ? AND profile_id = ?
                    """,
                    (version_id, profile_id),
                ).fetchone()
            else:
                row = self.conn.execute(
                    """
                    SELECT profile_id, circuit_version_id, owner_path,
                           test_name, parameters_json, profile_scheme,
                           profile_hash, created_at
                    FROM characterization_profiles
                    WHERE circuit_version_id = ? AND profile_scheme = ?
                          AND profile_hash = ?
                    """,
                    (
                        version_id,
                        canonical.fingerprint.scheme,
                        canonical.fingerprint.digest,
                    ),
                ).fetchone()
            if row is None:
                continue
            record = self._characterization_profile_record(row)
            if canonical is not None and not self._profile_payload_matches(
                row, context, canonical
            ):
                raise EvidenceMismatchError(
                    "Stored characterization profile identity conflicts with "
                    "the supplied canonical descriptor"
                )
            return record

        if profile_id is not None:
            existing = self.conn.execute(
                """
                SELECT circuit_version_id FROM characterization_profiles
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()
            if existing is not None:
                reason = (
                    "an incompatible structure in the requested lineage"
                    if existing["circuit_version_id"] in incompatible
                    else "a circuit version outside the requested lineage"
                )
                raise EvidenceMismatchError(
                    f"Characterization profile {profile_id!r} exists on "
                    f"{reason}"
                )
        elif incompatible:
            placeholders = ", ".join("?" for _ in incompatible)
            row = self.conn.execute(
                f"""
                SELECT 1 FROM characterization_profiles
                WHERE circuit_version_id IN ({placeholders})
                      AND profile_scheme = ? AND profile_hash = ?
                LIMIT 1
                """,
                [
                    *incompatible,
                    canonical.fingerprint.scheme,
                    canonical.fingerprint.digest,
                ],
            ).fetchone()
            if row is not None:
                raise EvidenceMismatchError(
                    "The requested characterization profile exists only on "
                    "an incompatible structure in the version lineage"
                )

        selector = (
            f"profile_id {profile_id!r}"
            if profile_id is not None
            else "the supplied complete descriptor"
        )
        raise EvidenceNotFoundError(
            f"No compatible characterization profile for {selector} is "
            f"stored in the lineage of circuit version "
            f"{circuit_version_id!r}"
        )

    @staticmethod
    def _net_current_set_record(
        origin_version_id: str,
        rows,
    ) -> NetCurrentSetRecord:
        first = rows[0]
        corner, temperature, voltage = Database._row_condition_scalars(first)
        currents = tuple(
            NetCurrentRecord(
                current_id=row["current_id"],
                net_name=row["net_name"],
                pin_id=row["pin_id"],
                avg_a=row["avg_a"],
                rms_a=row["rms_a"],
                peak_a=row["peak_a"],
                source=row["source"],
                current_vector=row["current_vector"],
                timestamp=row["timestamp"],
            )
            for row in rows
        )
        return NetCurrentSetRecord(
            current_set_id=first["current_set_id"],
            profile_id=first["profile_id"],
            origin_version_id=origin_version_id,
            state=first["state"],
            supersedes_current_set_id=first["supersedes_current_set_id"],
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            tool_name=first["tool_name"],
            analysis=first["analysis"],
            tstart=first["tstart"],
            results_file=first["results_file"],
            structure_scheme=first["structure_scheme"],
            structure_hash=first["structure_hash"],
            selected_at=first["selected_at"],
            superseded_at=first["superseded_at"],
            currents=currents,
            provenance_id=(first["provenance_id"]
                           if "provenance_id" in first.keys() else None),
        )

    def select_characterization_evidence(
        self,
        circuit_version_id: str,
        *,
        profile_id: Optional[str] = None,
        context: Optional[CharacterizationContext] = None,
        corner: Optional[str] = None,
        temperature: Optional[float] = None,
        voltage: Optional[float] = None,
        include_ancestors: bool = True,
    ) -> CharacterizationEvidenceRecord:
        """Select the evidence stored for one profile at one exact operating point.

        The profile is resolved with :meth:`resolve_characterization_profile`; the
        selected current set, the profile's metrics and its matching spec
        measurements are then read from the same lineage.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed target snapshot.
        profile_id : str, optional
            Exact profile id. Give exactly one of ``profile_id`` and ``context``.
        context : CharacterizationContext, optional
            Complete descriptor. Its corner, temperature and voltage select the
            point.
        corner : str or ConditionState, optional
            Process corner name or ``ConditionState.NONE``. Required with
            ``profile_id``; with ``context`` it may only repeat the context's value.
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius or ``ConditionState.NONE``. Same
            rule as ``corner``.
        voltage : float or ConditionState, optional
            Supply volts or ``ConditionState.NONE``. Same rule as ``corner``.
        include_ancestors : bool, default True
            Also search the snapshot's ancestors.

        Returns
        -------
        CharacterizationEvidenceRecord
            The profile, metrics, spec measurements and current set at the point.
            Its ``deck_check`` fields are ``None``.

        Raises
        ------
        silk.exceptions.EvidenceAmbiguityError
            Both ``profile_id`` and ``context`` were given.
        ValueError
            Neither selector was given, ``profile_id`` was given without all of
            ``corner``, ``temperature`` and ``voltage``, or a point value is
            unknown or a sweep (a profile needs each fact as a value or
            ``ConditionState.NONE``).
        TypeError
            ``context`` is not a ``CharacterizationContext`` or a point value has an
            unsupported type.
        silk.exceptions.EvidenceNotFoundError
            No compatible profile, or no selected current evidence for the profile.
        silk.exceptions.EvidenceMismatchError
            The explicit point conflicts with the context, current evidence exists
            only at other points or only superseded, the origin versions differ, or
            the matching spec measurements are incompatible with the target
            declarations.
        silk.exceptions.DatabaseError
            The snapshot is missing or unsealed.
        """
        if profile_id is not None and context is not None:
            raise EvidenceAmbiguityError(
                "Evidence selection requires exactly one profile selector: "
                "profile_id or a complete characterization context"
            )
        if profile_id is None and context is None:
            raise ValueError(
                "Evidence selection requires exactly one profile selector: "
                "profile_id or a complete characterization context"
            )
        if context is not None:
            if not isinstance(context, CharacterizationContext):
                raise TypeError(
                    "context must be a CharacterizationContext"
                )
            supplied = (corner, temperature, voltage)
            expected = (
                context.corner,
                context.temperature,
                context.voltage,
            )
            if any(value is not None for value in supplied) and any(
                value is not None and value != wanted
                for value, wanted in zip(supplied, expected)
            ):
                raise EvidenceMismatchError(
                    "Explicit PVT selectors conflict with the complete "
                    "characterization context"
                )
            corner, temperature, voltage = expected
        elif corner is None or temperature is None or voltage is None:
            raise ValueError(
                "A profile_id evidence selection requires exact corner, "
                "temperature, and voltage selectors"
            )

        # Schema v5 (TASK-409): each selector is a value or
        # ConditionState.NONE, matched as state-and-value.
        states = self._row_condition_columns(
            corner, temperature, voltage, subject="Evidence selector"
        )
        self._require_profile_states(states, subject="evidence selection")
        corner, temperature, voltage = (
            condition_scalar(column, state, value)
            for column, (state, value) in zip(
                ("corner", "temperature", "voltage"), states
            )
        )
        (corner_state, corner_value), (temperature_state, temperature_value), (
            voltage_state, voltage_value) = states
        profile = self.resolve_characterization_profile(
            circuit_version_id,
            profile_id=profile_id,
            context=context,
            include_ancestors=include_ancestors,
        )

        exact = self.conn.execute(
            """
            SELECT current_set_id, state
            FROM net_current_sets
            WHERE circuit_version_id = ? AND profile_id = ?
                  AND corner_state = ? AND corner IS ?
                  AND temperature_state = ? AND temperature IS ?
                  AND voltage_state = ? AND voltage IS ?
            ORDER BY selected_at DESC, rowid DESC
            """,
            (
                profile.origin_version_id,
                profile.profile_id,
                corner_state,
                corner_value,
                temperature_state,
                temperature_value,
                voltage_state,
                voltage_value,
            ),
        ).fetchall()
        selected_parent = next(
            (row for row in exact if row["state"] == "current"), None
        )
        if selected_parent is None:
            any_profile_set = self.conn.execute(
                """
                SELECT 1 FROM net_current_sets
                WHERE circuit_version_id = ? AND profile_id = ?
                LIMIT 1
                """,
                (profile.origin_version_id, profile.profile_id),
            ).fetchone()
            if any_profile_set is not None:
                if exact:
                    raise EvidenceMismatchError(
                        "The exact profile/PVT has only superseded current "
                        "evidence and no selected current set"
                    )
                raise EvidenceMismatchError(
                    "The characterization profile has selected current "
                    "evidence, but not at the requested exact PVT"
                )
            raise EvidenceNotFoundError(
                "No selected current evidence is stored for the resolved "
                "characterization profile"
            )

        origin_version_id, current_rows = self.get_net_current_set(
            circuit_version_id,
            corner,
            profile_id=profile.profile_id,
            temperature=temperature,
            voltage=voltage,
            include_ancestors=include_ancestors,
        )
        if origin_version_id != profile.origin_version_id:
            raise EvidenceMismatchError(
                "Selected current evidence and characterization profile have "
                "different origin versions"
            )
        current_set = self._net_current_set_record(
            origin_version_id, current_rows
        )
        metrics = tuple(
            self.get_simulation_results(
                circuit_version_id,
                corner=corner,
                temperature=temperature,
                voltage=voltage,
                profile_id=profile.profile_id,
                include_ancestors=include_ancestors,
            )
        )
        declarations = self.get_stored_specs(
            circuit_version_id,
            include_measurements=True,
            include_ancestors=include_ancestors,
            profile_id=profile.profile_id,
        )
        specs = tuple(
            replace(
                declaration,
                measurements=tuple(
                    measurement
                    for measurement in declaration.measurements
                    if measurement.corner == corner
                    and measurement.temperature == temperature
                    and measurement.voltage == voltage
                ),
            )
            for declaration in declarations
            if any(
                measurement.corner == corner
                and measurement.temperature == temperature
                and measurement.voltage == voltage
                for measurement in declaration.measurements
            )
        )
        if not specs and self._has_incompatible_spec_measurements(
            circuit_version_id,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            profile_id=profile.profile_id,
            include_ancestors=include_ancestors,
        ):
            raise EvidenceMismatchError(
                "Spec measurements for the selected profile/PVT are "
                "incompatible with the target declarations"
            )
        return CharacterizationEvidenceRecord(
            target_version_id=circuit_version_id,
            origin_version_id=origin_version_id,
            profile_id=profile.profile_id,
            current_set_id=current_set.current_set_id,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            profile=profile,
            metrics=metrics,
            specs=specs,
            current_set=current_set,
        )

    def find_snapshot(
        self,
        circuit_name: str,
        stage,
        identity: SnapshotIdentity,
    ) -> Optional[str]:
        """Return the sealed snapshot with an exact identity, if one exists.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        stage : DesignStage or str
            Design stage, as a member or its value string.
        identity : SnapshotIdentity
            Structure and spec-set identity to match exactly.

        Returns
        -------
        str or None
            The version id of the sealed snapshot, or ``None``.

        Raises
        ------
        TypeError
            ``identity`` is not a ``SnapshotIdentity``.
        ValueError
            ``stage`` is not a design stage.
        silk.exceptions.UnsupportedFingerprintSchemeError
            ``identity`` or a stored sealed row uses an unsupported scheme.
        """
        self._validate_identity(identity)
        stage_value = self._stage_value(stage)
        scheme_rows = self.conn.execute(
            """
            SELECT DISTINCT structure_scheme, spec_scheme
            FROM circuit_versions
            WHERE circuit_name = ? AND stage = ? AND state = 'sealed'
            """,
            (circuit_name, stage_value),
        ).fetchall()
        for row in scheme_rows:
            _require_fingerprint_scheme(
                Fingerprint(row["structure_scheme"], ""),
                STRUCTURE_FINGERPRINT_SCHEME,
            )
            _require_fingerprint_scheme(
                Fingerprint(row["spec_scheme"], ""),
                SPEC_FINGERPRINT_SCHEME,
            )

        row = self.conn.execute(
            """
            SELECT version_id
            FROM circuit_versions
            WHERE circuit_name = ? AND stage = ? AND state = 'sealed'
              AND structure_scheme = ? AND structure_hash = ?
              AND spec_scheme = ? AND spec_set_hash = ?
            """,
            (
                circuit_name,
                stage_value,
                identity.structure.scheme,
                identity.structure.digest,
                identity.spec_set.scheme,
                identity.spec_set.digest,
            ),
        ).fetchone()
        return None if row is None else row["version_id"]

    def iter_version_lineage(self, version_id: str) -> List[str]:
        """Return a snapshot followed by its ancestors.

        Despite the name, this returns a list, not an iterator.

        Parameters
        ----------
        version_id : str
            Version id to start from.

        Returns
        -------
        list of str
            ``version_id`` first, then its parent, grandparent and so on up to the
            root.

        Raises
        ------
        silk.exceptions.DatabaseError
            A version in the chain does not exist, or the parent links contain a
            cycle.
        """
        lineage = []
        seen = set()
        current = version_id
        while current is not None:
            if current in seen:
                raise DatabaseError(
                    f"Cycle detected in circuit-version lineage at {current!r}"
                )
            seen.add(current)
            lineage.append(current)
            row = self.conn.execute(
                "SELECT parent_version_id FROM circuit_versions WHERE version_id = ?",
                (current,),
            ).fetchone()
            if row is None:
                raise DatabaseError(
                    f"Circuit-version lineage references missing row {current!r}"
                )
            current = row["parent_version_id"]
        return lineage

    def _validate_parent(
        self,
        parent_version_id: str,
        *,
        circuit_name: str,
    ) -> None:
        parent = self._snapshot_row(parent_version_id)
        if parent["state"] != "sealed":
            raise DatabaseError(
                f"Parent version {parent_version_id!r} must be sealed"
            )
        if parent["circuit_name"] != circuit_name:
            raise DatabaseError(
                f"Parent version {parent_version_id!r} must belong to the "
                f"same circuit {circuit_name!r}"
            )
        self.iter_version_lineage(parent_version_id)

    def require_version_identity(
        self,
        version_id: str,
        identity: SnapshotIdentity,
        *,
        include_spec_set: bool,
    ) -> None:
        """Reject a snapshot whose stored identity differs from a given one.

        Parameters
        ----------
        version_id : str
            Version id of the stored snapshot.
        identity : SnapshotIdentity
            The identity the caller expects (typically that of the live design).
        include_spec_set : bool
            Also compare the spec-set identity; ``False`` compares structure only.

        Raises
        ------
        TypeError
            ``identity`` is not a ``SnapshotIdentity``.
        silk.exceptions.DatabaseError
            The snapshot does not exist.
        silk.exceptions.EvidenceMismatchError
            The stored structure (or, with ``include_spec_set``, spec-set) identity
            differs.
        silk.exceptions.UnsupportedFingerprintSchemeError
            A scheme is not the supported one.
        """
        self._validate_identity(identity)
        row = self._snapshot_row(version_id)
        stored_structure = Fingerprint(
            row["structure_scheme"], row["structure_hash"]
        )
        if not _fingerprints_equal(
            stored_structure,
            identity.structure,
            expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
        ):
            raise EvidenceMismatchError(
                f"Circuit version {version_id!r} has a different structure identity"
            )
        if include_spec_set:
            stored_specs = Fingerprint(row["spec_scheme"], row["spec_set_hash"])
            if not _fingerprints_equal(
                stored_specs,
                identity.spec_set,
                expected_scheme=SPEC_FINGERPRINT_SCHEME,
            ):
                raise EvidenceMismatchError(
                    f"Circuit version {version_id!r} has a different spec-set identity"
                )

    def seal_circuit_version(self, version_id: str) -> None:
        """Verify a snapshot's spec declarations and seal it.

        The stored declarations are re-fingerprinted and must match the snapshot's
        spec-set identity. A building snapshot becomes ``sealed``; a snapshot that
        is already sealed is left unchanged after the same verification.

        Parameters
        ----------
        version_id : str
            Version id of the snapshot to seal.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist.
        silk.exceptions.EvidenceMismatchError
            The stored declarations do not match the spec-set identity.
        silk.exceptions.LockedVersionError
            A building snapshot is locked (an already sealed snapshot is not
            checked for a lock).
        silk.exceptions.UnsupportedFingerprintSchemeError
            A stored identity uses an unsupported scheme.
        """
        row = self._snapshot_row(version_id)
        _require_fingerprint_scheme(
            Fingerprint(row["structure_scheme"], row["structure_hash"]),
            STRUCTURE_FINGERPRINT_SCHEME,
        )
        stored_spec_set = Fingerprint(row["spec_scheme"], row["spec_set_hash"])
        _require_fingerprint_scheme(stored_spec_set, SPEC_FINGERPRINT_SCHEME)
        definitions = [
            (
                record["owner_path"],
                record["spec_key"],
                Fingerprint(
                    record["definition_scheme"], record["definition_hash"]
                ),
            )
            for record in self.conn.execute(
                """
                SELECT owner_path, spec_key, definition_scheme, definition_hash
                FROM specs
                WHERE circuit_version_id = ?
                ORDER BY owner_path, spec_key
                """,
                (version_id,),
            ).fetchall()
        ]
        computed_spec_set = _fingerprint_spec_set_records(definitions)
        if not _fingerprints_equal(
            stored_spec_set,
            computed_spec_set,
            expected_scheme=SPEC_FINGERPRINT_SCHEME,
        ):
            raise EvidenceMismatchError(
                f"Cannot seal circuit version {version_id!r}: stored spec "
                "declarations do not match its spec-set identity"
            )
        if row["state"] == "sealed":
            return
        self._require_building(version_id)
        self.conn.execute(
            "UPDATE circuit_versions SET state = 'sealed' WHERE version_id = ?",
            (version_id,),
        )
        self._maybe_commit()

    def _validated_live_spec_definitions(self, version_id, definitions):
        """Validate and topologically order one complete live declaration set."""
        version = self._snapshot_row(version_id)
        expected_set = Fingerprint(
            version["spec_scheme"], version["spec_set_hash"]
        )
        _require_fingerprint_scheme(expected_set, SPEC_FINGERPRINT_SCHEME)

        entries = {}
        locations_by_object = {}
        for definition in tuple(definitions):
            location = (definition.owner_path, definition.key)
            if location in entries:
                raise DatabaseError(
                    "Spec declaration set has a duplicate owner/key: "
                    f"{definition.owner_path}/{definition.key}"
                )
            spec = definition.spec
            entries[location] = {
                "definition": definition,
                "spec": spec,
                "parent": (
                    None
                    if definition.parent_owner_path is None
                    else (
                        definition.parent_owner_path,
                        definition.parent_key,
                    )
                ),
            }
            locations_by_object.setdefault(id(spec), []).append(location)

        for location, entry in entries.items():
            definition = entry["definition"]
            spec = entry["spec"]
            parent_location = entry["parent"]
            if (definition.parent_owner_path is None) != (
                definition.parent_key is None
            ):
                raise DatabaseError(
                    f"Spec {location!r} has an incomplete parent identity"
                )
            if parent_location is not None and parent_location not in entries:
                raise DatabaseError(
                    f"Spec {location!r} refers to unknown parent "
                    f"{parent_location!r}"
                )
            if getattr(spec, "key", None) != definition.key:
                raise EvidenceMismatchError(
                    f"Spec {location!r} key changed after fingerprinting"
                )
            actual_parent = getattr(spec, "parent", None)
            actual_parent_location = None
            if actual_parent is not None:
                choices = locations_by_object.get(id(actual_parent), [])
                def proximity(candidate):
                    count = 0
                    for left, right in zip(location[0].split('/'), candidate[0].split('/')):
                        if left != right:
                            break
                        count += 1
                    return count
                if choices:
                    best = max(map(proximity, choices))
                    closest = [candidate for candidate in choices if proximity(candidate) == best]
                    if len(closest) == 1:
                        actual_parent_location = closest[0]
            if actual_parent_location != parent_location:
                raise EvidenceMismatchError(
                    f"Spec {location!r} parent relationship changed after "
                    "fingerprinting"
                )
            if isinstance(spec.target, bool) or not isinstance(spec.target, Real):
                raise DatabaseError(
                    f"Spec {location!r} target must be a real number"
                )
            _require_fingerprint_scheme(
                definition.fingerprint, SPEC_FINGERPRINT_SCHEME
            )
            recomputed = _fingerprint_spec_definition_fields(
                owner_path=definition.owner_path,
                key=definition.key,
                metric=spec.metric,
                op=spec.op,
                target=spec.target,
                units=spec.units,
                parent_owner_path=definition.parent_owner_path,
                parent_key=definition.parent_key,
            )
            if not _fingerprints_equal(
                definition.fingerprint,
                recomputed,
                expected_scheme=SPEC_FINGERPRINT_SCHEME,
            ):
                raise EvidenceMismatchError(
                    f"Spec {location!r} definition fingerprint does not "
                    "match its declaration fields"
                )

        computed_set = _fingerprint_spec_set_records(
            (
                location[0],
                location[1],
                entry["definition"].fingerprint,
            )
            for location, entry in entries.items()
        )
        if not _fingerprints_equal(
            expected_set,
            computed_set,
            expected_scheme=SPEC_FINGERPRINT_SCHEME,
        ):
            raise EvidenceMismatchError(
                f"Spec declarations do not match circuit version "
                f"{version_id!r} spec-set identity"
            )

        ordered = []
        resolved = set()
        pending = dict(entries)
        while pending:
            ready = sorted(
                location
                for location, entry in pending.items()
                if entry["parent"] is None or entry["parent"] in resolved
            )
            if not ready:
                unresolved = ", ".join(
                    f"{owner_path}/{key}"
                    for owner_path, key in sorted(pending)
                )
                raise DatabaseError(
                    "Spec declaration parents contain a cycle: " + unresolved
                )
            for location in ready:
                ordered.append(pending.pop(location))
                resolved.add(location)
        return ordered

    def store_spec_declarations(self, version_id: str, definitions) -> None:
        """Insert the complete spec declaration set of a building snapshot.

        The set is validated against the snapshot's spec-set identity and written
        in parent-before-child order. Declarations are insert-only: a snapshot that
        already has any cannot receive more.

        Parameters
        ----------
        version_id : str
            Version id of a building, unlocked snapshot.
        definitions : iterable
            The live declaration records produced by the tracker for the design
            (one per spec, carrying owner path, key, the ``Spec`` and its
            fingerprint). They are not meant to be built by hand; an empty sequence
            declares no specs.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot is missing or not building, declarations already exist,
            an owner/key is duplicated, a parent is unknown or incomplete, parents
            form a cycle, or a target is not a real number.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        silk.exceptions.EvidenceMismatchError
            The live specs no longer match their fingerprints, or the set does not
            match the snapshot's spec-set identity.
        """
        with self.transaction():
            self._require_building(version_id)
            existing = self.conn.execute(
                "SELECT 1 FROM specs WHERE circuit_version_id = ? LIMIT 1",
                (version_id,),
            ).fetchone()
            if existing is not None:
                raise DatabaseError(
                    f"Spec declarations for version {version_id!r} are "
                    "insert-only"
                )
            ordered = self._validated_live_spec_definitions(
                version_id, definitions
            )
            ids_by_location = {}
            for entry in ordered:
                definition = entry["definition"]
                spec = entry["spec"]
                parent_id = (
                    None
                    if entry["parent"] is None
                    else ids_by_location[entry["parent"]]
                )
                spec_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO specs
                        (spec_id, circuit_version_id, owner_path, spec_key,
                         name, metric, op, target, units, description,
                         parent_id, declared_at, definition_scheme,
                         definition_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec_id,
                        version_id,
                        definition.owner_path,
                        definition.key,
                        spec.name,
                        spec.metric,
                        spec.op,
                        float(spec.target),
                        spec.units,
                        spec.description,
                        parent_id,
                        spec.declared_at,
                        definition.fingerprint.scheme,
                        definition.fingerprint.digest,
                    ),
                )
                ids_by_location[
                    (definition.owner_path, definition.key)
                ] = spec_id

    def _validated_stored_spec_rows(self, version_id):
        """Return a sealed version's rows after full identity validation."""
        version = self._snapshot_row(version_id)
        if version["state"] != "sealed":
            raise DatabaseError(
                "Historical spec readback requires a sealed snapshot; "
                f"circuit version {version_id!r} is {version['state']!r}"
            )
        expected_set = Fingerprint(
            version["spec_scheme"], version["spec_set_hash"]
        )
        _require_fingerprint_scheme(expected_set, SPEC_FINGERPRINT_SCHEME)
        rows = self.conn.execute(
            """
            SELECT rowid AS spec_rowid, spec_id, circuit_version_id,
                   owner_path, spec_key, name, metric, op, target, units,
                   description, parent_id, declared_at, definition_scheme,
                   definition_hash, created_at
            FROM specs
            WHERE circuit_version_id = ?
            ORDER BY owner_path, spec_key, rowid
            """,
            (version_id,),
        ).fetchall()
        by_id = {}
        by_location = {}
        for row in rows:
            location = (row["owner_path"], row["spec_key"])
            if location in by_location:
                raise DatabaseError(
                    "Stored spec declarations contain duplicate owner/key "
                    f"{location!r}"
                )
            by_id[row["spec_id"]] = row
            by_location[location] = row

        parent_locations = {}
        for row in rows:
            parent_id = row["parent_id"]
            if parent_id is None:
                parent_location = None
            else:
                parent = by_id.get(parent_id)
                if parent is None:
                    raise DatabaseError(
                        f"Stored spec {row['spec_id']!r} has a parent outside "
                        f"declaration version {version_id!r}"
                    )
                parent_location = (
                    parent["owner_path"], parent["spec_key"]
                )
            parent_locations[row["spec_id"]] = parent_location
            stored = Fingerprint(
                row["definition_scheme"], row["definition_hash"]
            )
            _require_fingerprint_scheme(stored, SPEC_FINGERPRINT_SCHEME)
            recomputed = _fingerprint_spec_definition_fields(
                owner_path=row["owner_path"],
                key=row["spec_key"],
                metric=row["metric"],
                op=row["op"],
                target=row["target"],
                units=row["units"],
                parent_owner_path=(
                    None if parent_location is None else parent_location[0]
                ),
                parent_key=(
                    None if parent_location is None else parent_location[1]
                ),
            )
            if not _fingerprints_equal(
                stored,
                recomputed,
                expected_scheme=SPEC_FINGERPRINT_SCHEME,
            ):
                raise EvidenceMismatchError(
                    f"Stored spec {row['spec_id']!r} definition hash does not "
                    "match its declaration fields"
                )

        resolved = set()
        pending = {row["spec_id"]: row for row in rows}
        while pending:
            ready = sorted(
                spec_id
                for spec_id, row in pending.items()
                if row["parent_id"] is None or row["parent_id"] in resolved
            )
            if not ready:
                raise DatabaseError(
                    f"Stored spec declarations for version {version_id!r} "
                    "contain a parent cycle"
                )
            for spec_id in ready:
                pending.pop(spec_id)
                resolved.add(spec_id)

        computed_set = _fingerprint_spec_set_records(
            (
                row["owner_path"],
                row["spec_key"],
                Fingerprint(
                    row["definition_scheme"], row["definition_hash"]
                ),
            )
            for row in rows
        )
        if not _fingerprints_equal(
            expected_set,
            computed_set,
            expected_scheme=SPEC_FINGERPRINT_SCHEME,
        ):
            raise EvidenceMismatchError(
                f"Stored spec declarations for version {version_id!r} do "
                "not match its spec-set identity"
            )
        return version, rows, parent_locations

    def get_stored_specs(
        self,
        circuit_version_id,
        *,
        include_measurements=True,
        include_ancestors=True,
        profile_id=None,
    ) -> List[SpecDeclarationRecord]:
        """Return validated stored spec declarations, optionally with measurements.

        Declarations of the snapshot are checked against their fingerprints and
        the snapshot's spec-set identity. Measurements are gathered from the
        snapshot and, with ``include_ancestors``, from ancestors whose structure
        identity and declaration definition match.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed snapshot.
        include_measurements : bool, default True
            Attach stored measurements to each declaration.
        include_ancestors : bool, default True
            Also read measurements stored on ancestor snapshots.
        profile_id : str or UNSCOPED_PROFILE, optional
            Filter on which measurements are attached; declarations are always all
            returned. ``None`` applies no filter,
            :data:`~silk.db.UNSCOPED_PROFILE` keeps only unscoped measurements and
            a string keeps only that profile's.

        Returns
        -------
        list of SpecDeclarationRecord
            Declarations ordered by owner path and key. Each ``measurements`` tuple
            is ordered by timestamp (measurements without one first), then storage
            order. A stored NULL value reads back as ``nan``.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot or a lineage member is missing or unsealed, or stored
            declarations are structurally invalid.
        silk.exceptions.EvidenceMismatchError
            Stored declarations or measurements fail validation (fingerprint,
            source, cached verdict).
        ValueError
            A stored aggregate attribution is invalid.
        TypeError
            ``profile_id`` is not a non-empty string, ``UNSCOPED_PROFILE`` or
            ``None``.
        """
        target, target_rows, parent_locations = (
            self._validated_stored_spec_rows(circuit_version_id)
        )
        measurements_by_location = {
            (row["owner_path"], row["spec_key"]): []
            for row in target_rows
        }
        if include_measurements and target_rows:
            target_structure = Fingerprint(
                target["structure_scheme"], target["structure_hash"]
            )
            _require_fingerprint_scheme(
                target_structure, STRUCTURE_FINGERPRINT_SCHEME
            )
            lineage = (
                self.iter_version_lineage(circuit_version_id)
                if include_ancestors
                else [circuit_version_id]
            )
            target_by_location = {
                (row["owner_path"], row["spec_key"]): row
                for row in target_rows
            }
            for origin_version_id in lineage:
                origin = self._snapshot_row(origin_version_id)
                if origin["state"] != "sealed":
                    raise DatabaseError(
                        "Historical spec lineage contains unsealed circuit "
                        f"version {origin_version_id!r}"
                    )
                origin_structure = Fingerprint(
                    origin["structure_scheme"], origin["structure_hash"]
                )
                _require_fingerprint_scheme(
                    origin_structure, STRUCTURE_FINGERPRINT_SCHEME
                )
                if not _fingerprints_equal(
                    target_structure,
                    origin_structure,
                    expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                ):
                    continue
                _origin, origin_rows, _parents = (
                    self._validated_stored_spec_rows(origin_version_id)
                )
                origin_by_location = {
                    (row["owner_path"], row["spec_key"]): row
                    for row in origin_rows
                }
                for location, target_row in target_by_location.items():
                    origin_row = origin_by_location.get(location)
                    if origin_row is None:
                        continue
                    target_definition = Fingerprint(
                        target_row["definition_scheme"],
                        target_row["definition_hash"],
                    )
                    origin_definition = Fingerprint(
                        origin_row["definition_scheme"],
                        origin_row["definition_hash"],
                    )
                    if not _fingerprints_equal(
                        target_definition,
                        origin_definition,
                        expected_scheme=SPEC_FINGERPRINT_SCHEME,
                    ):
                        continue
                    profile_clauses, profile_params = _profile_filter_predicate(
                        profile_id
                    )
                    measurement_where = " AND ".join(
                        ["spec_id = ?", *profile_clauses]
                    )
                    measurement_rows = self.conn.execute(
                        f"""
                        SELECT rowid AS measurement_rowid, measurement_id,
                               source, value, corner, temperature, voltage,
                               sim_type, tool_name, timestamp, sim_run_id,
                               notes, passes, profile_id, provenance_id, aggregate_json,
                               corner_state, temperature_state, voltage_state
                        FROM spec_measurements
                        WHERE {measurement_where}
                        """,
                        [origin_row["spec_id"], *profile_params],
                    ).fetchall()
                    for measurement in measurement_rows:
                        source = measurement["source"]
                        if source not in _SPEC_MEASUREMENT_SOURCES:
                            raise EvidenceMismatchError(
                                f"Stored spec measurement "
                                f"{measurement['measurement_id']!r} has invalid "
                                f"source {source!r}"
                            )
                        value = (
                            float("nan") if measurement["value"] is None
                            else _spec_measurement_real(
                                measurement["value"], field="value"
                            )
                        )
                        corner, temperature, voltage = (
                            self._row_condition_scalars(
                                measurement, subject="Spec measurement"
                            )
                        )
                        timestamp = _spec_measurement_real(
                            measurement["timestamp"],
                            field="timestamp",
                            optional=True,
                        )
                        cached = measurement["passes"]
                        if cached not in (None, 0, 1):
                            raise EvidenceMismatchError(
                                f"Stored spec measurement "
                                f"{measurement['measurement_id']!r} has invalid "
                                f"cached passes value {cached!r}"
                            )
                        evaluated = _evaluate_spec_measurement(
                            op=target_row["op"],
                            value=value,
                            target=target_row["target"],
                        )
                        if cached is not None and bool(cached) != evaluated:
                            raise EvidenceMismatchError(
                                f"Stored spec measurement "
                                f"{measurement['measurement_id']!r} cached pass "
                                "does not match its value, operator, and target"
                            )
                        record = SpecMeasurementRecord(
                            measurement_id=measurement["measurement_id"],
                            origin_version_id=origin_version_id,
                            origin_spec_id=origin_row["spec_id"],
                            source=source,
                            value=value,
                            corner=corner,
                            temperature=temperature,
                            voltage=voltage,
                            sim_type=measurement["sim_type"],
                            tool_name=measurement["tool_name"],
                            timestamp=timestamp,
                            sim_run_id=measurement["sim_run_id"],
                            notes=measurement["notes"] or "",
                            aggregate=_read_aggregate(measurement, origin_row["metric"]),
                            passes=(
                                None
                                if measurement["passes"] is None
                                else bool(measurement["passes"])
                            ),
                            profile_id=measurement["profile_id"],
                            provenance_id=measurement["provenance_id"],
                        )
                        measurements_by_location[location].append(
                            (record, measurement["measurement_rowid"])
                        )

        records = []
        for row in target_rows:
            location = (row["owner_path"], row["spec_key"])
            measurement_entries = measurements_by_location[location]
            measurement_entries.sort(
                key=lambda entry: (
                    entry[0].timestamp is not None,
                    (
                        entry[0].timestamp
                        if entry[0].timestamp is not None
                        else 0.0
                    ),
                    entry[1],
                )
            )
            parent_location = parent_locations[row["spec_id"]]
            records.append(
                SpecDeclarationRecord(
                    spec_id=row["spec_id"],
                    declaration_version_id=circuit_version_id,
                    owner_path=row["owner_path"],
                    spec_key=row["spec_key"],
                    name=row["name"],
                    metric=row["metric"],
                    op=row["op"],
                    target=row["target"],
                    units=row["units"],
                    description=row["description"],
                    parent_id=row["parent_id"],
                    parent_owner_path=(
                        None if parent_location is None else parent_location[0]
                    ),
                    parent_spec_key=(
                        None if parent_location is None else parent_location[1]
                    ),
                    declared_at=row["declared_at"],
                    definition_scheme=row["definition_scheme"],
                    definition_hash=row["definition_hash"],
                    created_at=row["created_at"],
                    measurements=tuple(
                        entry[0] for entry in measurement_entries
                    ),
                )
            )
        return records

    def _has_incompatible_spec_measurements(
        self,
        circuit_version_id,
        *,
        source=None,
        corner=None,
        temperature=None,
        voltage=None,
        tool_name=None,
        profile_id=None,
        include_ancestors=True,
    ) -> bool:
        """Whether requested history exists but cannot apply to the target.

        This is used only to distinguish a true absence from evidence made
        inapplicable by structure, key, or definition drift. Every declaration
        snapshot in the considered lineage is validated before the answer is
        returned.
        """
        target, target_rows, _parents = self._validated_stored_spec_rows(
            circuit_version_id
        )
        target_structure = Fingerprint(
            target["structure_scheme"], target["structure_hash"]
        )
        _require_fingerprint_scheme(
            target_structure, STRUCTURE_FINGERPRINT_SCHEME
        )
        target_by_location = {
            (row["owner_path"], row["spec_key"]): row
            for row in target_rows
        }
        lineage = (
            self.iter_version_lineage(circuit_version_id)
            if include_ancestors
            else [circuit_version_id]
        )
        incompatible_spec_ids = []
        for origin_version_id in lineage:
            origin, origin_rows, _origin_parents = (
                self._validated_stored_spec_rows(origin_version_id)
            )
            origin_structure = Fingerprint(
                origin["structure_scheme"], origin["structure_hash"]
            )
            same_structure = _fingerprints_equal(
                target_structure,
                origin_structure,
                expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
            )
            for origin_row in origin_rows:
                compatible = False
                target_row = target_by_location.get(
                    (origin_row["owner_path"], origin_row["spec_key"])
                )
                if same_structure and target_row is not None:
                    compatible = _fingerprints_equal(
                        Fingerprint(
                            target_row["definition_scheme"],
                            target_row["definition_hash"],
                        ),
                        Fingerprint(
                            origin_row["definition_scheme"],
                            origin_row["definition_hash"],
                        ),
                        expected_scheme=SPEC_FINGERPRINT_SCHEME,
                    )
                if not compatible:
                    incompatible_spec_ids.append(origin_row["spec_id"])

        if not incompatible_spec_ids:
            return False
        placeholders = ", ".join("?" for _ in incompatible_spec_ids)
        clauses = [f"spec_id IN ({placeholders})"]
        params = list(incompatible_spec_ids)
        for column, value in (
            ("source", source),
            ("tool_name", tool_name),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        for column, value in (
            ("corner", corner),
            ("temperature", temperature),
            ("voltage", voltage),
        ):
            condition_clauses, condition_params = condition_predicate(
                column, value
            )
            clauses.extend(condition_clauses)
            params.extend(condition_params)
        profile_clauses, profile_params = _profile_filter_predicate(profile_id)
        clauses.extend(profile_clauses)
        params.extend(profile_params)
        return self.conn.execute(
            f"""
            SELECT 1 FROM spec_measurements
            WHERE {' AND '.join(clauses)}
            LIMIT 1
            """,
            params,
        ).fetchone() is not None

    # ===== MODULE MANAGEMENT METHODS =====

    def insert_or_get_module(self, module_name: str, module_path: str,
                             module_hash: str, device_count: int = 0,
                             structure: Dict = None,
                             module_hash_scheme: str = STRUCTURE_FINGERPRINT_SCHEME) -> str:
        """Store a module definition, or return the id of an identical stored one.

        Modules are content-addressed by ``(module_hash_scheme, module_hash)``, so
        a module shared between snapshots is stored once.

        Parameters
        ----------
        module_name : str
            Name of the module, for example ``'ldo_output_stage'``.
        module_path : str
            Hierarchical path of the module, for example ``'top.ldo.output'``.
        module_hash : str
            Structure digest of the module.
        device_count : int, default 0
            Number of devices in the module.
        structure : dict, optional
            Module payload, stored as JSON when non-empty. Its fingerprint must
            equal ``module_hash``.
        module_hash_scheme : str, default "silk.structure/v4"
            Fingerprint scheme of ``module_hash``.

        Returns
        -------
        str
            The module id, existing or new.

        Raises
        ------
        silk.exceptions.UnsupportedFingerprintSchemeError
            ``module_hash_scheme`` is not the structure scheme.
        silk.exceptions.EvidenceMismatchError
            ``structure`` does not hash to ``module_hash``.
        """
        _require_fingerprint_scheme(
            Fingerprint(module_hash_scheme, module_hash),
            STRUCTURE_FINGERPRINT_SCHEME,
        )
        if structure is not None:
            from silk.db.identity import _fingerprint_module_structure
            if _fingerprint_module_structure(structure).digest != module_hash:
                raise EvidenceMismatchError(
                    'Module payload does not match its structure identity'
                )
        cursor = self.conn.cursor()

        # Check if module with this hash already exists
        cursor.execute("""
            SELECT module_id FROM modules
            WHERE module_hash_scheme = ? AND module_hash = ?
        """, (module_hash_scheme, module_hash))

        row = cursor.fetchone()
        if row:
            # Module already exists, return its ID
            return row[0]

        # Module doesn't exist, create new one
        module_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO modules 
            (module_id, module_name, module_path, module_hash_scheme,
             module_hash, device_count, structure)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            module_id,
            module_name,
            module_path,
            module_hash_scheme,
            module_hash,
            device_count,
            json.dumps(encode_json(structure), allow_nan=False) if structure else None,
        ))

        self._maybe_commit()
        return module_id

    def link_module_to_circuit(self, circuit_version_id: str, module_id: str,
                               instance_name: str = None, position: int = 0):
        """Link a stored module to a building snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a building, unlocked snapshot.
        module_id : str
            Id returned by :meth:`insert_or_get_module`.
        instance_name : str, optional
            Instance name of the module in the circuit.
        position : int, default 0
            Position in the module hierarchy; :meth:`get_circuit_modules` orders by
            it.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not building.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        """
        self._require_building(circuit_version_id)
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO circuit_modules
            (circuit_version_id, module_id, instance_name, position_in_hierarchy)
            VALUES (?, ?, ?, ?)
        """, (circuit_version_id, module_id, instance_name, position))

        self._maybe_commit()

    def get_circuit_modules(self, circuit_version_id: str) -> List[Dict]:
        """Return the modules linked to a snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot.

        Returns
        -------
        list of dict
            One dict per link, ordered by hierarchy position, with keys
            ``'module_id'``, ``'module_name'``, ``'module_path'``,
            ``'module_hash'``, ``'module_hash_scheme'``, ``'device_count'``,
            ``'structure'`` (the decoded payload or ``None``), ``'created_at'``,
            ``'instance_name'`` and ``'position_in_hierarchy'``.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT m.module_id, m.module_name, m.module_path, m.module_hash,
                   m.module_hash_scheme, m.device_count, m.structure,
                   m.created_at,
                   cm.instance_name, cm.position_in_hierarchy
            FROM modules m
            JOIN circuit_modules cm ON m.module_id = cm.module_id
            WHERE cm.circuit_version_id = ?
            ORDER BY cm.position_in_hierarchy
        """, (circuit_version_id,))

        modules = []
        for row in cursor.fetchall():
            modules.append({
                'module_id': row[0],
                'module_name': row[1],
                'module_path': row[2],
                'module_hash': row[3],
                'module_hash_scheme': row[4],
                'device_count': row[5],
                'structure': decode_json(json.loads(row[6])) if row[6] else None,
                'created_at': row[7],
                'instance_name': row[8],
                'position_in_hierarchy': row[9]
            })

        return modules

    def get_circuits_using_module(
        self,
        module_hash: str,
        *,
        module_hash_scheme: str = STRUCTURE_FINGERPRINT_SCHEME,
    ) -> List[Dict]:
        """Find the snapshots that link a module with the given hash.

        Useful for impact analysis before changing a module.

        Parameters
        ----------
        module_hash : str
            Structure digest of the module.
        module_hash_scheme : str, default "silk.structure/v4"
            Fingerprint scheme of ``module_hash`` (keyword-only).

        Returns
        -------
        list of dict
            Snapshots, newest first, each with keys ``'version_id'``,
            ``'circuit_name'``, ``'version'``, ``'stage'`` (the value string) and
            ``'created_at'``. Empty when the module is unused.

        Raises
        ------
        silk.exceptions.UnsupportedFingerprintSchemeError
            ``module_hash_scheme`` is not the structure scheme.
        """
        _require_fingerprint_scheme(
            Fingerprint(module_hash_scheme, module_hash),
            STRUCTURE_FINGERPRINT_SCHEME,
        )
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT cv.version_id, cv.circuit_name, cv.version, cv.stage, cv.created_at
            FROM circuit_versions cv
            JOIN circuit_modules cm ON cv.version_id = cm.circuit_version_id
            JOIN modules m ON cm.module_id = m.module_id
            WHERE m.module_hash_scheme = ? AND m.module_hash = ?
            ORDER BY cv.created_at DESC
        """, (module_hash_scheme, module_hash))

        circuits = []
        for row in cursor.fetchall():
            circuits.append({
                'version_id': row[0],
                'circuit_name': row[1],
                'version': row[2],
                'stage': row[3],
                'created_at': row[4]
            })

        return circuits

    def compare_circuit_modules(self, circuit_id_1: str, circuit_id_2: str) -> Dict[str, Any]:
        """Compare the modules of two snapshots to see what changed.

        Modules are matched by ``(module_hash_scheme, module_hash)``.

        Parameters
        ----------
        circuit_id_1 : str
            Version id of the first (earlier) snapshot.
        circuit_id_2 : str
            Version id of the second snapshot.

        Returns
        -------
        dict
            Keys ``'added'`` (modules only in the second), ``'removed'`` (only in
            the first) and ``'unchanged'`` (in both, taken from the second). Each
            value is a list of the module dicts that :meth:`get_circuit_modules`
            returns.
        """
        modules_1 = {
            (m['module_hash_scheme'], m['module_hash']): m
            for m in self.get_circuit_modules(circuit_id_1)
        }
        modules_2 = {
            (m['module_hash_scheme'], m['module_hash']): m
            for m in self.get_circuit_modules(circuit_id_2)
        }

        hashes_1 = set(modules_1.keys())
        hashes_2 = set(modules_2.keys())

        return {
            'added': [modules_2[h] for h in (hashes_2 - hashes_1)],
            'removed': [modules_1[h] for h in (hashes_1 - hashes_2)],
            'unchanged': [modules_2[h] for h in (hashes_1 & hashes_2)]
        }

    # ===== END MODULE MANAGEMENT METHODS =====

    def create_circuit(self, name: str, version: str = "1.0",
                      stage: DesignStage = DesignStage.DEV,
                      created_by: str = "user",
                      description: str = "",
                      netlist_path: str = None,
                      metadata: Dict = None,
                      allow_duplicate_version: bool = False,
                      *, identity: SnapshotIdentity,
                      parent_version_id: str = None) -> str:
        """Create a snapshot in the ``building`` state.

        The snapshot is not usable for evidence until its structure and spec
        declarations are stored and :meth:`seal_circuit_version` is called (the
        tracker does this in one step).

        Parameters
        ----------
        name : str
            Circuit name.
        version : str, default "1.0"
            Descriptive version label. Labels need not be unique; identity, not the
            label, decides reuse.
        stage : DesignStage or str, optional
            Design stage; ``DesignStage.DEV`` by default.
        created_by : str, default "user"
            Author recorded with the snapshot.
        description : str, default ""
            Free-text description.
        netlist_path : str, optional
            Path of the netlist the snapshot was built from.
        metadata : dict, optional
            JSON-compatible dictionary stored with the snapshot. It must not
            contain ``'structure_hash'``.
        allow_duplicate_version : bool, default False
            Deprecated and ignored; kept for positional compatibility.
        identity : SnapshotIdentity
            Structure and spec-set identity of the snapshot (keyword-only,
            required).
        parent_version_id : str, optional
            Parent snapshot. It must be sealed and belong to the same circuit name.
            When omitted, the most recent sealed snapshot of the same name and stage
            is used, if any.

        Returns
        -------
        str
            The new version id.

        Raises
        ------
        TypeError
            ``identity`` is not a ``SnapshotIdentity``.
        ValueError
            ``stage`` is not a design stage.
        silk.exceptions.DatabaseError
            ``metadata`` contains ``'structure_hash'``, or ``parent_version_id`` is
            not a sealed snapshot of the same circuit.
        silk.exceptions.UnsupportedFingerprintSchemeError
            ``identity`` uses an unsupported scheme.
        """
        del allow_duplicate_version
        self._validate_identity(identity)
        stage_value = self._stage_value(stage)
        if metadata and "structure_hash" in metadata:
            raise DatabaseError(
                "structure_hash is first-class snapshot identity and must not "
                "also be stored in mutable metadata"
            )
        if parent_version_id is None:
            parent = self.conn.execute(
                """
                SELECT version_id
                FROM circuit_versions
                WHERE circuit_name = ? AND stage = ? AND state = 'sealed'
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (name, stage_value),
            ).fetchone()
            parent_version_id = None if parent is None else parent["version_id"]
        else:
            self._validate_parent(parent_version_id, circuit_name=name)
        version_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO circuit_versions 
            (version_id, circuit_name, version, stage, state, parent_version_id,
             created_by, description, netlist_path, metadata,
             structure_scheme, structure_hash, spec_scheme, spec_set_hash)
            VALUES (?, ?, ?, ?, 'building', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            version_id,
            name,
            version,
            stage_value,
            parent_version_id,
            created_by,
            description,
            netlist_path,
            json.dumps(encode_json(metadata), allow_nan=False) if metadata else None,
            identity.structure.scheme,
            identity.structure.digest,
            identity.spec_set.scheme,
            identity.spec_set.digest,
        ))

        self._maybe_commit()
        _log.info("Created circuit %r version %s at stage %r", name, version, stage_value)
        return version_id

    def add_device(self, circuit_version_id: str, name: str, device_type: str,
                   model: str = None, **properties) -> str:
        """Add a device to a building snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a building, unlocked snapshot.
        name : str
            Instance name of the device.
        device_type : str
            Device type label, for example ``'nmos'``.
        model : str, optional
            Model name.
        **properties
            Extra JSON-compatible properties stored with the device as given.

        Returns
        -------
        str
            The new device id.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not building.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        """
        self._require_building(circuit_version_id)
        device_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO devices
            (device_id, circuit_version_id, name, type, model, properties)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (device_id, circuit_version_id, name, device_type, model,
              json.dumps(encode_json(properties), allow_nan=False) if properties else None))

        self._maybe_commit()
        return device_id

    def add_transistor(self, circuit_version_id: str, name: str,
                       transistor_type: str, width: float, length: float,
                       multiplier: int = 1, model: str = None) -> str:
        """Add a transistor to a building snapshot.

        Stores ``width``, ``length`` and ``multiplier`` as properties through
        :meth:`add_device`. The values are stored as given, with no unit conversion.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a building, unlocked snapshot.
        name : str
            Instance name of the transistor.
        transistor_type : str
            Device type label, for example ``'nmos'`` or ``'pmos'``.
        width : float
            Channel width, in the caller's unit.
        length : float
            Channel length, in the caller's unit.
        multiplier : int, default 1
            Parallel multiplier.
        model : str, optional
            Model name.

        Returns
        -------
        str
            The new device id.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not building.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        """
        return self.add_device(circuit_version_id, name, transistor_type,
                               model=model, width=width, length=length,
                               multiplier=multiplier)

    def add_net(self, circuit_version_id: str, name: str, net_type: str = None,
                **properties) -> str:
        """Add a net to a building snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a building, unlocked snapshot.
        name : str
            Net name.
        net_type : str, optional
            Role label such as ``'signal'`` or ``'power'``.
        **properties
            Extra JSON-compatible properties stored with the net as given.

        Returns
        -------
        str
            The new net id.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not building.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        """
        self._require_building(circuit_version_id)
        net_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO nets
            (net_id, circuit_version_id, name, net_type, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (net_id, circuit_version_id, name, net_type,
              json.dumps(encode_json(properties), allow_nan=False) if properties else None))

        self._maybe_commit()
        return net_id

    def add_simulation_result(self, circuit_version_id: str, parameter: str,
                             value: float = None, value_text: str = None,
                             unit: str = None, object_id: str = None,
                             object_type: str = 'circuit', sim_type: str = None,
                             corner: str = None, temperature: float = None,
                             voltage: float = None, tool_name: str = None,
                             conditions: Dict = None,
                             source: str = "user",
                             timestamp: str = None,
                             profile_id: Optional[str] = None,
                             provenance_id: Optional[str] = None) -> str:
        """Add one typed numeric or text observation to a sealed snapshot.

        ``corner``, ``temperature`` and ``voltage`` are condition scalars: a value,
        ``ConditionState.NONE`` (the condition does not apply), ``None`` (unknown)
        or, for ``temperature`` only, ``ConditionState.SWEEP``. Temperature is in
        degrees Celsius and voltage in volts by silk convention; the database
        stores the numbers as given. A scoped result (one with ``profile_id``) needs
        each of the three to be a value or ``NONE``.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed, unlocked snapshot.
        parameter : str
            Metric name.
        value : float, optional
            Numeric value. Exactly one of ``value`` and ``value_text`` is required.
        value_text : str, optional
            Text value.
        unit : str, optional
            Unit label.
        object_id : str, optional
            Identifier of the object the observation is about.
        object_type : str, default "circuit"
            Kind of object the observation is about.
        sim_type : str, optional
            Label of the analysis; not validated.
        corner : str or ConditionState, optional
            Process corner.
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius.
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        tool_name : str, optional
            Tool that produced the value.
        conditions : dict, optional
            JSON-compatible dictionary of extra conditions, stored canonically.
        source : str, default "user"
            Non-empty label of where the value came from.
        timestamp : str, optional
            Time label; the database insertion time (UTC) is used when omitted.
        profile_id : str, optional
            Id of an existing characterization profile of this snapshot. Nothing is
            inferred from ``conditions`` or any other field.
        provenance_id : str, optional
            Id of an existing run-provenance row (see :meth:`store_run_provenance`).

        Returns
        -------
        str
            The new result id.

        Raises
        ------
        ValueError
            Not exactly one of ``value`` and ``value_text`` was given, ``source`` is
            not a non-empty string, or a scoped result has an unknown or sweep
            condition.
        TypeError
            ``value`` is a ``bool`` or not a real number, ``value_text`` or
            ``timestamp`` is not a string, ``conditions`` is not a JSON-compatible
            mapping, a condition has an unsupported type, or ``profile_id`` or
            ``provenance_id`` is not a non-empty string.
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not sealed.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        silk.exceptions.EvidenceMismatchError
            ``profile_id`` or ``provenance_id`` is not stored.
        """
        self._require_sealed(circuit_version_id)
        if (value is None) == (value_text is None):
            raise ValueError(
                "A simulation result requires exactly one of value or "
                "value_text"
            )
        states = self._row_condition_columns(
            corner, temperature, voltage, subject="Simulation result"
        )
        if profile_id is not None:
            if not isinstance(profile_id, str) or not profile_id:
                raise TypeError(
                    "Simulation result profile_id must be a non-empty "
                    "string or None"
                )
            self._require_profile_states(states, subject="simulation result")
            owner = self.conn.execute(
                """
                SELECT 1 FROM characterization_profiles
                WHERE circuit_version_id = ? AND profile_id = ?
                """,
                (circuit_version_id, profile_id),
            ).fetchone()
            if owner is None:
                raise EvidenceMismatchError(
                    f"profile_id {profile_id!r} is not a stored "
                    f"characterization profile for circuit version "
                    f"{circuit_version_id!r}"
                )
        if value is not None:
            if isinstance(value, bool):
                raise TypeError(
                    "Simulation result value cannot be a boolean"
                )
            if not isinstance(value, Real):
                raise TypeError(
                    "Simulation result value must be a real number"
                )
            value = float(value)
        if value_text is not None and not isinstance(value_text, str):
            raise TypeError("Simulation result text value must be a string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Simulation result source must be a non-empty string")
        if timestamp is not None and not isinstance(timestamp, str):
            raise TypeError("Simulation result timestamp must be a string")
        conditions_json = self._canonical_simulation_conditions(conditions)
        self._require_run_provenance(provenance_id)
        sim_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        (corner_state, corner), (temperature_state, temperature), (
            voltage_state, voltage) = states
        cursor.execute("""
            INSERT INTO simulation_results
            (sim_id, circuit_version_id, object_id, object_type, source, sim_type,
             corner, temperature, voltage, parameter, value, value_text, unit,
             tool_name, timestamp, conditions, profile_id, provenance_id,
             corner_state, temperature_state, voltage_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?, ?, ?)
        """, (sim_id, circuit_version_id, object_id, object_type, source,
              sim_type, corner, temperature, voltage, parameter, value,
              value_text, unit, tool_name, timestamp, conditions_json,
              profile_id, provenance_id, corner_state, temperature_state,
              voltage_state))

        # Invalidate the explicitly stored summary when evidence changes.
        # Summary reads render current records; only explicit update writes
        # these compatibility cache columns.
        cursor.execute("""
            UPDATE circuit_versions
            SET simulation_summary = NULL, last_summary_update = NULL
            WHERE version_id = ?
        """, (circuit_version_id,))

        self._maybe_commit()
        return sim_id

    @staticmethod
    def _row_condition_columns(corner, temperature, voltage, *, subject,
                               error_type=TypeError):
        """``((state, value), ...)`` for corner, temperature and voltage."""
        return tuple(
            condition_columns(
                column, scalar, error_type=error_type, subject=subject
            )
            for column, scalar in (
                ("corner", corner),
                ("temperature", temperature),
                ("voltage", voltage),
            )
        )

    @staticmethod
    def _require_profile_states(states, *, subject):
        """A scoped row needs every fact a value or none (design §2.2)."""
        if any(state not in ("fixed", "none") for state, _value in states):
            raise ValueError(
                f"A profile-scoped {subject} requires corner, temperature, "
                "and voltage, each a value or ConditionState.NONE (unknown "
                "and sweep cannot form a profile)"
            )

    @staticmethod
    def _row_condition_scalars(row, *, subject="Stored row",
                               error_type=TypeError):
        """The widened ``(corner, temperature, voltage)`` of a stored row.

        Each stored ``(state, value)`` pair is decoded and its value
        re-validated (SQLite's loose typing is not trusted), so a corrupt
        state raises ``EvidenceMismatchError`` and an untyped value
        ``error_type``. A value comes back as ``str`` / ``float``.
        """
        scalars = []
        for column in ("corner", "temperature", "voltage"):
            try:
                scalar = condition_scalar(
                    column, row[f"{column}_state"], row[column]
                )
            except ValueError as exc:
                raise EvidenceMismatchError(f"{subject}: {exc}") from exc
            state, value = condition_columns(
                column, scalar, error_type=error_type, subject=subject
            )
            scalars.append(condition_scalar(column, state, value))
        return tuple(scalars)

    @staticmethod
    def _canonical_simulation_conditions(conditions) -> Optional[str]:
        if conditions is None:
            return None
        if not isinstance(conditions, Mapping):
            raise TypeError("Simulation result conditions must be a mapping")
        try:
            return json.dumps(
                encode_json(dict(conditions), finite_only=True),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Simulation result conditions must contain JSON-compatible "
                "values"
            ) from exc

    #: Reserved ``conditions`` key store_report() attaches per metric row
    #: -- per-run diagnostic narration, not evidence identity.
    _METRIC_WARNINGS_CONDITIONS_KEY = 'silk_metric_warnings'

    @classmethod
    def _conditions_identity(cls, conditions_json: Optional[str]) -> Optional[str]:
        """Stored conditions identity, excluding per-observation narration and links.

        Used only to build the ``latest_only=True`` "one row per context"
        grouping key in :meth:`get_simulation_results`. Two otherwise
        identical readings that differ only in whether a metric warned this
        time must still collapse exactly as they did before
        ``silk_metric_warnings`` existed -- otherwise a superseded,
        warning-bearing run would survive alongside its own replacement
        instead of being collapsed away. The spec-measurement row link also
        identifies an observation, rather than its coordinates, and is excluded.
        Coordinate maps in ``silk_run_variables`` and all other caller/reserved
        keys still participate in identity.
        """
        if conditions_json is None:
            return conditions_json
        decoded = json.loads(conditions_json)
        if (
            not isinstance(decoded, dict)
            or not ({cls._METRIC_WARNINGS_CONDITIONS_KEY, 'silk_spec_measurement'} & decoded.keys())
        ):
            return conditions_json
        stripped = {
            k: v for k, v in decoded.items()
            if k not in (cls._METRIC_WARNINGS_CONDITIONS_KEY, 'silk_spec_measurement')
        }
        if not stripped:
            return None
        return json.dumps(
            encode_json(stripped, finite_only=True),
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )

    def replace_net_current_set(
        self,
        circuit_version_id: str,
        *,
        corner: str,
        structure_scheme: str,
        structure_hash: str,
        rows,
        temperature: float = None,
        voltage: float = None,
        tool_name: str = None,
        analysis: str = None,
        tstart: float = None,
        results_file: str = None,
        context: Optional[CharacterizationContext] = None,
        provenance_id: Optional[str] = None,
    ) -> str:
        """Atomically replace the selected current set of one selection key.

        The key is the snapshot plus the corner for an unscoped set, or the profile
        plus corner, temperature and voltage when ``context`` is given. A previous
        selected set for the key is marked superseded and retained; the new set
        records which set it supersedes.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed, unlocked snapshot.
        corner : str or ConditionState
            Process corner name or ``ConditionState.NONE`` (keyword-only). ``None``
            is refused.
        structure_scheme : str
            Scheme of ``structure_hash``; must be the structure scheme.
        structure_hash : str
            Structure digest of the design the currents were measured on. It must
            equal the snapshot's structure identity.
        rows : iterable of dict
            Current rows. Each needs ``'net_name'``, a real ``'avg_a'`` (amperes)
            and ``'source'`` (``'spice'`` or ``'user'``); ``'pin_id'`` (default
            ``''``, the net-level aggregate), ``'rms_a'``, ``'peak_a'`` (amperes)
            and ``'current_vector'`` (text) are optional. At least one row is
            required.
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius.
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        tool_name : str, optional
            Tool that produced the currents.
        analysis : str, optional
            Analysis the currents came from.
        tstart : float, optional
            Start of the window the currents were taken over, in seconds of
            simulation time.
        results_file : str, optional
            Path of the results file the currents came from.
        context : CharacterizationContext, optional
            Scopes the set to a profile; its corner, temperature and voltage must
            equal the arguments.
        provenance_id : str, optional
            Id of an existing run-provenance row.

        Returns
        -------
        str
            The new current-set id.

        Raises
        ------
        ValueError
            ``corner`` is ``None``, a row has an invalid ``source``, or ``rows`` is
            empty.
        TypeError
            A row is not a mapping, a field has the wrong type, or ``context`` is
            not a ``CharacterizationContext``.
        KeyError
            A row has no ``'net_name'``.
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not sealed.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        silk.exceptions.EvidenceMismatchError
            The structure identity does not match the snapshot, ``context``
            disagrees with the point arguments, or ``provenance_id`` is not stored.
        silk.exceptions.UnsupportedFingerprintSchemeError
            ``structure_scheme`` is not the structure scheme.
        """
        if corner is None:
            raise ValueError("Current evidence corner is required")
        states = self._row_condition_columns(
            corner, temperature, voltage, subject="Current row"
        )
        (corner_state, corner_value), (temperature_state, temperature_value), (
            voltage_state, voltage_value) = states
        corner, temperature, voltage = (
            condition_scalar(column, state, value)
            for column, (state, value) in zip(
                ("corner", "temperature", "voltage"), states
            )
        )
        tstart = _net_current_real(tstart, field="tstart")
        tool_name = _net_current_text(
            tool_name, field="tool_name", optional=True
        )
        analysis = _net_current_text(
            analysis, field="analysis", optional=True
        )
        results_file = _net_current_text(
            results_file, field="results_file", optional=True
        )
        if context is not None:
            if not isinstance(context, CharacterizationContext):
                raise TypeError(
                    "Current evidence context must be a "
                    "CharacterizationContext or None"
                )
            if (
                context.corner != corner
                or context.temperature != temperature
                or context.voltage != voltage
            ):
                raise EvidenceMismatchError(
                    "Current evidence corner, temperature, and voltage must "
                    "exactly match its characterization context"
                )
        staged = []
        for current in rows:
            if not isinstance(current, Mapping):
                raise TypeError("Current rows must be mappings")
            source = current.get("source")
            if source not in _NET_CURRENT_SOURCES:
                raise ValueError(
                    "Current row source must be one of "
                    f"{sorted(_NET_CURRENT_SOURCES)}, got {source!r}"
                )
            staged.append(
                (
                    _net_current_text(
                        current["net_name"], field="net_name"
                    ),
                    _net_current_text(
                        current.get("pin_id", ""), field="pin_id"
                    ),
                    _net_current_real(
                        current.get("avg_a"),
                        field="avg_a",
                        optional=False,
                    ),
                    _net_current_real(
                        current.get("rms_a"), field="rms_a"
                    ),
                    _net_current_real(
                        current.get("peak_a"), field="peak_a"
                    ),
                    source,
                    _net_current_text(
                        current.get("current_vector"),
                        field="current_vector",
                        optional=True,
                    ),
                )
            )
        if not staged:
            raise ValueError("A current set must contain at least one row")

        with self.transaction():
            version = self._require_sealed(circuit_version_id)
            self._require_run_provenance(provenance_id)
            supplied = Fingerprint(structure_scheme, structure_hash)
            stored = Fingerprint(
                version["structure_scheme"], version["structure_hash"]
            )
            if not _fingerprints_equal(
                stored,
                supplied,
                expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
            ):
                raise EvidenceMismatchError(
                    f"Current evidence does not match circuit version "
                    f"{circuit_version_id!r} structure identity"
                )

            profile_id = None
            if context is not None:
                profile_id = self.store_characterization_profile(
                    circuit_version_id, context
                ).profile_id

            if profile_id is None:
                existing = self.conn.execute(
                    """
                    SELECT current_set_id FROM net_current_sets
                    WHERE circuit_version_id = ? AND corner_state = ?
                          AND corner IS ?
                          AND profile_id IS NULL AND state = 'current'
                    """,
                    (circuit_version_id, corner_state, corner_value),
                ).fetchone()
            else:
                existing = self.conn.execute(
                    """
                    SELECT current_set_id FROM net_current_sets
                    WHERE circuit_version_id = ? AND profile_id = ?
                          AND corner_state = ? AND corner IS ?
                          AND temperature_state = ? AND temperature IS ?
                          AND voltage_state = ? AND voltage IS ?
                          AND state = 'current'
                    """,
                    (
                        circuit_version_id,
                        profile_id,
                        corner_state,
                        corner_value,
                        temperature_state,
                        temperature_value,
                        voltage_state,
                        voltage_value,
                    ),
                ).fetchone()
            if existing is not None:
                self.conn.execute(
                    """
                    UPDATE net_current_sets
                    SET state = 'superseded', superseded_at = CURRENT_TIMESTAMP
                    WHERE current_set_id = ?
                    """,
                    (existing["current_set_id"],),
                )
            predecessor_id = (
                None if existing is None else existing["current_set_id"]
            )

            new_set_id = str(uuid.uuid4())
            self.conn.execute(
                """
                INSERT INTO net_current_sets
                    (current_set_id, circuit_version_id, profile_id, state,
                     supersedes_current_set_id, corner, temperature, voltage,
                     tool_name, analysis, tstart, results_file,
                     structure_scheme, structure_hash, provenance_id,
                     corner_state, temperature_state, voltage_state)
                VALUES (?, ?, ?, 'current', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?)
                """,
                (
                    new_set_id,
                    circuit_version_id,
                    profile_id,
                    predecessor_id,
                    corner_value,
                    temperature_value,
                    voltage_value,
                    tool_name,
                    analysis,
                    tstart,
                    results_file,
                    structure_scheme,
                    structure_hash,
                    provenance_id,
                    corner_state,
                    temperature_state,
                    voltage_state,
                ),
            )
            for net_name, pin_id, avg, rms, peak, source, vector in staged:
                self.conn.execute(
                    """
                    INSERT INTO net_currents
                        (current_id, current_set_id, net_name, pin_id,
                         avg_a, rms_a, peak_a, source, current_vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        new_set_id,
                        net_name,
                        pin_id,
                        avg,
                        rms,
                        peak,
                        source,
                        vector,
                    ),
                )
        return new_set_id

    def get_net_current_set(
        self,
        circuit_version_id: str,
        corner: Optional[str] = None,
        *,
        current_set_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        temperature: Optional[float] = None,
        voltage: Optional[float] = None,
        include_ancestors: bool = True,
    ):
        """Return a stored current set that is compatible with a sealed snapshot.

        Selection searches the snapshot first, then its ancestors, and takes the
        first structure-compatible set. Spec-set differences are irrelevant here.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed snapshot.
        corner : str or ConditionState, optional
            Corner of the selected set. Required unless ``current_set_id`` is given.
        current_set_id : str, optional
            Read this exact set, including a superseded one. It cannot be combined
            with the other selectors (keyword-only).
        profile_id : str, optional
            Select the profile-scoped set. This needs exact ``corner``,
            ``temperature`` and ``voltage`` (keyword-only).
        temperature : float or ConditionState, optional
            Temperature selector for a profile-scoped set.
        voltage : float or ConditionState, optional
            Voltage selector for a profile-scoped set.
        include_ancestors : bool, default True
            Also search the snapshot's ancestors.

        Returns
        -------
        tuple
            ``(origin_version_id, rows)``: the version id that holds the set and a
            list of ``sqlite3.Row`` objects, one per stored current row in insertion
            order. Condition columns come as ``corner``, ``temperature`` and
            ``voltage`` with matching ``*_state`` columns. Besides the current's own columns (``current_id``, ``net_name``,
            ``pin_id``, ``avg_a``, ``rms_a``, ``peak_a``, ``source``,
            ``current_vector``, ``timestamp``) every row repeats its set's columns
            (``current_set_id``, ``profile_id``, ``state``,
            ``supersedes_current_set_id``, ``origin_version_id``, ``analysis``,
            ``tstart``, ``results_file``, the condition columns, ``tool_name``,
            ``structure_scheme``, ``structure_hash``, ``selected_at``,
            ``superseded_at``, ``provenance_id``, and a ``current_rowid``).

        Raises
        ------
        ValueError
            Neither ``corner`` nor ``current_set_id`` was given, a profile lookup
            lacks ``temperature`` or ``voltage``, or ``current_set_id`` is combined
            with other selectors.
        TypeError
            ``profile_id`` is not a non-empty string.
        silk.exceptions.DatabaseError
            The snapshot or a lineage member is missing or unsealed.
        silk.exceptions.EvidenceMismatchError
            The set exists only on an incompatible structure, or stored rows are
            invalid.
        RuntimeError
            No compatible set exists for the selector (the message lists the
            available corners), or a stored row's structure identity does not match
            its origin version.
        """
        if current_set_id is None and corner is None:
            raise ValueError(
                "Current-set lookup requires a corner or current_set_id"
            )
        if profile_id is not None:
            if not isinstance(profile_id, str) or not profile_id:
                raise TypeError(
                    "Current-set profile_id must be a non-empty string or None"
                )
            if temperature is None or voltage is None:
                raise ValueError(
                    "A profile-scoped current-set lookup requires exact "
                    "profile, corner, temperature, and voltage selectors"
                )
        if current_set_id is not None and any(
            value is not None
            for value in (corner, profile_id, temperature, voltage)
        ):
            raise ValueError(
                "Historical current_set_id lookup cannot be combined with "
                "profile/PVT selectors"
            )
        if current_set_id is None:
            corner_clauses, corner_params = condition_predicate(
                "corner", corner, subject="Current-set selector"
            )
            pvt_clauses, pvt_params = list(corner_clauses), list(corner_params)
            if profile_id is not None:
                for column, scalar in (
                    ("temperature", temperature),
                    ("voltage", voltage),
                ):
                    clauses, params = condition_predicate(
                        column, scalar, subject="Current-set selector"
                    )
                    pvt_clauses.extend(clauses)
                    pvt_params.extend(params)
        target = self._snapshot_row(circuit_version_id)
        if target["state"] != "sealed":
            raise DatabaseError(
                "Current readback requires a sealed snapshot; circuit version "
                f"{circuit_version_id!r} is {target['state']!r}"
            )
        target_fingerprint = Fingerprint(
            target["structure_scheme"], target["structure_hash"]
        )
        _require_fingerprint_scheme(
            target_fingerprint, STRUCTURE_FINGERPRINT_SCHEME
        )
        lineage = (
            self.iter_version_lineage(circuit_version_id)
            if include_ancestors
            else [circuit_version_id]
        )

        candidates = []
        for version_id in lineage:
            version = self._snapshot_row(version_id)
            if version["state"] != "sealed":
                raise DatabaseError(
                    "Current readback lineage contains an unsealed circuit "
                    f"version {version_id!r}"
                )
            if version["circuit_name"] != target["circuit_name"]:
                raise DatabaseError(
                    "Current readback lineage crosses circuit names at "
                    f"version {version_id!r}"
                )
            fingerprint = Fingerprint(
                version["structure_scheme"], version["structure_hash"]
            )
            _require_fingerprint_scheme(
                fingerprint, STRUCTURE_FINGERPRINT_SCHEME
            )
            candidates.append(
                (
                    version_id,
                    fingerprint,
                    _fingerprints_equal(
                        target_fingerprint,
                        fingerprint,
                        expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                    ),
                )
            )

        available = set()
        incompatible_with_corner = []
        for version_id, version_fingerprint, compatible in candidates:
            # The partial unique indexes permit exactly one selected parent
            # for each unscoped corner or scoped profile/PVT key. Set-level
            # context is fetched once and projected into every child row by
            # the JOIN below.
            if current_set_id is not None:
                current_set = self.conn.execute(
                    """
                    SELECT current_set_id FROM net_current_sets
                    WHERE circuit_version_id = ? AND current_set_id = ?
                    """,
                    (version_id, current_set_id),
                ).fetchone()
            elif profile_id is None:
                current_set = self.conn.execute(
                    f"""
                    SELECT current_set_id FROM net_current_sets
                    WHERE circuit_version_id = ? AND {' AND '.join(pvt_clauses)}
                          AND profile_id IS NULL AND state = 'current'
                    """,
                    (version_id, *pvt_params),
                ).fetchone()
            else:
                current_set = self.conn.execute(
                    f"""
                    SELECT current_set_id FROM net_current_sets
                    WHERE circuit_version_id = ? AND profile_id = ?
                          AND {' AND '.join(pvt_clauses)}
                          AND state = 'current'
                    """,
                    (version_id, profile_id, *pvt_params),
                ).fetchone()
            if not compatible:
                if current_set is not None:
                    incompatible_with_corner.append(version_id)
                continue
            if current_set is None:
                available.update(
                    condition_scalar(
                        "corner", row["corner_state"], row["corner"]
                    )
                    for row in self.conn.execute(
                        """
                        SELECT DISTINCT corner_state, corner
                        FROM net_current_sets
                        WHERE circuit_version_id = ? AND state = 'current'
                        """,
                        (version_id,),
                    ).fetchall()
                )
                continue

            rows = self.conn.execute(
                """
                SELECT nc.rowid AS current_rowid, nc.current_id,
                       ncs.current_set_id, ncs.profile_id, ncs.state,
                       ncs.supersedes_current_set_id,
                       ncs.circuit_version_id AS origin_version_id,
                       nc.net_name, nc.pin_id, nc.avg_a, nc.rms_a, nc.peak_a,
                       nc.source, nc.current_vector,
                       ncs.analysis, ncs.tstart, ncs.results_file, ncs.corner,
                       ncs.temperature, ncs.voltage, ncs.tool_name,
                       ncs.structure_scheme, ncs.structure_hash,
                       ncs.selected_at, ncs.superseded_at, nc.timestamp,
                       ncs.provenance_id, ncs.corner_state,
                       ncs.temperature_state, ncs.voltage_state
                FROM net_currents AS nc
                JOIN net_current_sets AS ncs
                    ON ncs.current_set_id = nc.current_set_id
                WHERE nc.current_set_id = ?
                ORDER BY nc.rowid
                """,
                (current_set["current_set_id"],),
            ).fetchall()
            if not rows:
                raise EvidenceMismatchError(
                    f"Stored current set {current_set['current_set_id']!r} "
                    "has no rows"
                )

            for row in rows:
                _validate_stored_net_current_row(row)
                row_fingerprint = Fingerprint(
                    row["structure_scheme"], row["structure_hash"]
                )
                _require_fingerprint_scheme(
                    row_fingerprint, STRUCTURE_FINGERPRINT_SCHEME
                )
                if not _fingerprints_equal(
                    version_fingerprint,
                    row_fingerprint,
                    expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                ):
                    raise RuntimeError(
                        "Stored currents are stale: a row structure identity "
                        f"does not match origin version {version_id!r}"
                    )
                if row["source"] not in _NET_CURRENT_SOURCES:
                    raise EvidenceMismatchError(
                        f"Stored current row {row['current_id']!r} has invalid "
                        f"source {row['source']!r}"
                    )
            return version_id, rows

        if incompatible_with_corner:
            raise EvidenceMismatchError(
                "Current evidence for the requested corner exists only on an "
                "incompatible structure in the version lineage"
            )
        raise RuntimeError(
            f"load_net_currents: version found but no rows for corner "
            f"{corner!r}. Available corners: {sorted(available, key=repr)}."
        )

    def _simulation_context_predicates(
        self,
        *,
        parameter=None,
        source=None,
        sim_type=None,
        corner=None,
        temperature=None,
        voltage=None,
        tool_name=None,
        conditions=None,
        profile_id=None,
        alias="sr",
    ):
        clauses = []
        params = []
        for column, requested in (
            ("parameter", parameter),
            ("source", source),
            ("sim_type", sim_type),
            ("tool_name", tool_name),
        ):
            if requested is not None:
                clauses.append(f"{alias}.{column} = ?")
                params.append(requested)
        # Schema v5 (TASK-409): a condition filter matches state-and-value,
        # so ConditionState.NONE selects none rows and never unknown ones.
        for column, requested in (
            ("corner", corner),
            ("temperature", temperature),
            ("voltage", voltage),
        ):
            condition_clauses, condition_params = condition_predicate(
                column, requested, alias=alias
            )
            clauses.extend(condition_clauses)
            params.extend(condition_params)
        canonical_conditions = self._canonical_simulation_conditions(conditions)
        if canonical_conditions is not None:
            clauses.append(f"{alias}.conditions = ?")
            params.append(canonical_conditions)
        profile_clauses, profile_params = _profile_filter_predicate(
            profile_id, alias=alias
        )
        clauses.extend(profile_clauses)
        params.extend(profile_params)
        return clauses, params

    def _simulation_lineage_by_structure(
        self,
        circuit_version_id: str,
        *,
        include_ancestors: bool,
    ):
        target = self._snapshot_row(circuit_version_id)
        if target["state"] != "sealed":
            raise DatabaseError(
                f"Metric readback requires a sealed snapshot; circuit version "
                f"{circuit_version_id!r} is {target['state']!r}"
            )
        target_fingerprint = Fingerprint(
            target["structure_scheme"], target["structure_hash"]
        )
        _require_fingerprint_scheme(
            target_fingerprint, STRUCTURE_FINGERPRINT_SCHEME
        )
        lineage = (
            self.iter_version_lineage(circuit_version_id)
            if include_ancestors
            else [circuit_version_id]
        )
        compatible = []
        incompatible = []
        ranks = {}
        for rank, version_id in enumerate(lineage):
            row = self._snapshot_row(version_id)
            if row["state"] != "sealed":
                raise DatabaseError(
                    "Metric readback lineage contains an unsealed circuit "
                    f"version {version_id!r}"
                )
            fingerprint = Fingerprint(
                row["structure_scheme"], row["structure_hash"]
            )
            _require_fingerprint_scheme(
                fingerprint, STRUCTURE_FINGERPRINT_SCHEME
            )
            ranks[version_id] = rank
            if _fingerprints_equal(
                target_fingerprint,
                fingerprint,
                expected_scheme=STRUCTURE_FINGERPRINT_SCHEME,
            ):
                compatible.append(version_id)
            else:
                incompatible.append(version_id)
        return compatible, incompatible, ranks

    @staticmethod
    def _decode_simulation_conditions(raw, *, result_id):
        if raw is None:
            return None
        try:
            # The finite-only writer emits no numeric tags or literal escapes.
            decoded = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise DatabaseError(
                f"Simulation result {result_id!r} has invalid conditions JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise DatabaseError(
                f"Simulation result {result_id!r} conditions are not an object"
            )
        return decoded

    def get_simulation_results(
        self,
        circuit_version_id: str,
        parameter: str = None,
        corner: str = None,
        *,
        source: str = None,
        sim_type: str = None,
        temperature: float = None,
        voltage: float = None,
        tool_name: str = None,
        conditions: Dict = None,
        profile_id=None,
        latest_only: bool = False,
        include_ancestors: bool = True,
    ) -> List[SimulationResultRecord]:
        """Return the typed observations visible from a sealed snapshot.

        Observations stored on the snapshot, and with ``include_ancestors`` on its
        structure-compatible ancestors, are matched against the filters. A filter
        left as ``None`` does not restrict. Condition filters match state and
        value, so ``ConditionState.NONE`` selects rows where the condition does not
        apply and never unknown ones.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed snapshot.
        parameter : str, optional
            Metric name.
        corner : str or ConditionState, optional
            Process corner.
        source : str, optional
            Source label (keyword-only from here on).
        sim_type : str, optional
            Analysis label.
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius.
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        tool_name : str, optional
            Producing tool.
        conditions : dict, optional
            Exact extra-conditions dictionary to match.
        profile_id : str or UNSCOPED_PROFILE, optional
            ``None`` applies no profile filter, :data:`~silk.db.UNSCOPED_PROFILE`
            keeps only unscoped rows, and a string keeps only that profile's.
        latest_only : bool, default False
            Keep only the newest observation for each distinct combination of
            parameter, object, source, analysis, corner, temperature, voltage,
            tool, conditions and profile.
        include_ancestors : bool, default True
            Also read structure-compatible ancestors.

        Returns
        -------
        list of SimulationResultRecord
            Records sorted by timestamp, then lineage distance from the snapshot,
            then storage order. A row with both value and text NULL has
            ``value=nan``. An empty list is returned when nothing matches.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot or a lineage member is missing or unsealed, or stored
            ``conditions`` JSON is invalid.
        silk.exceptions.EvidenceMismatchError
            A stored condition state is corrupt.
        TypeError
            ``conditions`` is not a JSON-compatible mapping, or ``profile_id`` is
            invalid.
        """
        compatible, _incompatible, ranks = self._simulation_lineage_by_structure(
            circuit_version_id,
            include_ancestors=include_ancestors,
        )
        placeholders = ", ".join("?" for _ in compatible)
        clauses = [f"sr.circuit_version_id IN ({placeholders})"]
        params = list(compatible)
        context_clauses, context_params = self._simulation_context_predicates(
            parameter=parameter,
            source=source,
            sim_type=sim_type,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            tool_name=tool_name,
            conditions=conditions,
            profile_id=profile_id,
        )
        clauses.extend(context_clauses)
        params.extend(context_params)
        rows = self.conn.execute(
            f"""
            SELECT sr.rowid AS result_rowid, sr.sim_id,
                   sr.circuit_version_id, sr.parameter, sr.value,
                   sr.value_text, sr.unit, sr.object_id, sr.object_type,
                   sr.source, sr.sim_type, sr.corner, sr.temperature,
                   sr.voltage, sr.tool_name, sr.conditions, sr.timestamp,
                   sr.profile_id, sr.provenance_id, sr.corner_state,
                   sr.temperature_state, sr.voltage_state
            FROM simulation_results AS sr
            WHERE {' AND '.join(clauses)}
            """,
            params,
        ).fetchall()

        entries = []
        for row in rows:
            corner_scalar, temperature_scalar, voltage_scalar = (
                self._row_condition_scalars(
                    row, subject=f"Simulation result {row['sim_id']!r}",
                    error_type=EvidenceMismatchError,
                )
            )
            record = SimulationResultRecord(
                result_id=row["sim_id"],
                origin_version_id=row["circuit_version_id"],
                parameter=row["parameter"],
                value=(
                    float("nan")
                    if row["value"] is None and row["value_text"] is None
                    else row["value"]
                ),
                value_text=row["value_text"],
                unit=row["unit"],
                object_id=row["object_id"],
                object_type=row["object_type"],
                source=row["source"],
                sim_type=row["sim_type"],
                corner=corner_scalar,
                temperature=temperature_scalar,
                voltage=voltage_scalar,
                tool_name=row["tool_name"],
                conditions=self._decode_simulation_conditions(
                    row["conditions"], result_id=row["sim_id"]
                ),
                timestamp=row["timestamp"],
                profile_id=row["profile_id"],
                provenance_id=row["provenance_id"],
            )
            entries.append(
                (
                    record,
                    row["conditions"],
                    ranks[row["circuit_version_id"]],
                    row["result_rowid"],
                )
            )

        def all_order(entry):
            record, _canonical, rank, rowid = entry
            return (record.timestamp, rank, rowid)

        if latest_only:
            newest = {}
            for entry in entries:
                record, canonical, _rank, rowid = entry
                key = (
                    record.parameter,
                    record.object_type,
                    record.object_id,
                    record.source,
                    record.sim_type,
                    record.corner,
                    record.temperature,
                    record.voltage,
                    record.tool_name,
                    self._conditions_identity(canonical),
                    record.profile_id,
                )
                previous = newest.get(key)
                if previous is None or (
                    record.timestamp,
                    rowid,
                ) > (previous[0].timestamp, previous[3]):
                    newest[key] = entry
            entries = list(newest.values())

        entries.sort(key=all_order)
        return [entry[0] for entry in entries]

    def _has_incompatible_simulation_results(
        self,
        circuit_version_id: str,
        *,
        parameter=None,
        source=None,
        sim_type=None,
        corner=None,
        temperature=None,
        voltage=None,
        tool_name=None,
        conditions=None,
        profile_id=None,
        include_ancestors=True,
    ) -> bool:
        _compatible, incompatible, _ranks = self._simulation_lineage_by_structure(
            circuit_version_id,
            include_ancestors=include_ancestors,
        )
        if not incompatible:
            return False
        placeholders = ", ".join("?" for _ in incompatible)
        clauses = [f"sr.circuit_version_id IN ({placeholders})"]
        params = list(incompatible)
        context_clauses, context_params = self._simulation_context_predicates(
            parameter=parameter,
            source=source,
            sim_type=sim_type,
            corner=corner,
            temperature=temperature,
            voltage=voltage,
            tool_name=tool_name,
            conditions=conditions,
            profile_id=profile_id,
        )
        clauses.extend(context_clauses)
        params.extend(context_params)
        return self.conn.execute(
            f"""
            SELECT 1 FROM simulation_results AS sr
            WHERE {' AND '.join(clauses)}
            LIMIT 1
            """,
            params,
        ).fetchone() is not None

    def get_latest(self, circuit_name: str, stage: DesignStage = None) -> Optional[Dict]:
        """Return the newest sealed snapshot of a circuit.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        stage : DesignStage, optional
            Restrict to this stage.

        Returns
        -------
        dict or None
            A dict with keys ``'version_id'``, ``'circuit_name'``, ``'version'``,
            ``'stage'`` (value string), ``'created_at'``, ``'netlist_path'``,
            ``'description'``, ``'metadata'`` (decoded dict, ``{}`` when none)
            and ``'created_by'``, or ``None`` when no sealed snapshot
            matches.
        """
        cursor = self.conn.cursor()

        if stage:
            cursor.execute("""
                SELECT version_id, circuit_name, version, stage, created_at,
                       netlist_path, description, metadata, created_by
                FROM circuit_versions
                WHERE circuit_name = ? AND stage = ? AND state = 'sealed'
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
            """, (circuit_name, stage.value))
        else:
            cursor.execute("""
                SELECT version_id, circuit_name, version, stage, created_at,
                       netlist_path, description, metadata, created_by
                FROM circuit_versions
                WHERE circuit_name = ? AND state = 'sealed'
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
            """, (circuit_name,))

        row = cursor.fetchone()
        if row:
            return {
                'version_id': row[0],
                'circuit_name': row[1],
                'version': row[2],
                'stage': row[3],
                'created_at': row[4],
                'netlist_path': row[5],
                'description': row[6],
                'metadata': decode_json(json.loads(row[7])) if row[7] else {},
                'created_by': row[8],
            }
        return None

    def get_version(self, circuit_name: str, version: str) -> Optional[Dict]:
        """Return a snapshot by circuit name and version label.

        Building and sealed snapshots are both found.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        version : str
            Version label.

        Returns
        -------
        dict or None
            The same keys as :meth:`get_latest`, or ``None`` when no snapshot has
            that label.

        Raises
        ------
        silk.exceptions.DatabaseError
            More than one snapshot carries the label; resolve by identity or version
            id instead.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata, created_by
            FROM circuit_versions
            WHERE circuit_name = ? AND version = ?
        """, (circuit_name, version))

        rows = cursor.fetchall()
        if len(rows) > 1:
            raise DatabaseError(
                f"Circuit label {circuit_name!r} version {version!r} is "
                "ambiguous; resolve by SnapshotIdentity or version_id"
            )
        row = rows[0] if rows else None
        if row:
            return {
                'version_id': row[0],
                'circuit_name': row[1],
                'version': row[2],
                'stage': row[3],
                'created_at': row[4],
                'netlist_path': row[5],
                'description': row[6],
                'metadata': decode_json(json.loads(row[7])) if row[7] else {},
                'created_by': row[8],
            }
        return None

    def _copy_structure_for_promotion(
        self, source_version_id: str, target_version_id: str
    ) -> None:
        """Copy declaration-side structure, assigning fresh row identities."""
        for row in self.conn.execute(
            """
            SELECT name, type, model, properties
            FROM devices
            WHERE circuit_version_id = ?
            ORDER BY rowid
            """,
            (source_version_id,),
        ).fetchall():
            self.conn.execute(
                """
                INSERT INTO devices
                    (device_id, circuit_version_id, name, type, model, properties)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    target_version_id,
                    row["name"],
                    row["type"],
                    row["model"],
                    row["properties"],
                ),
            )

        for row in self.conn.execute(
            """
            SELECT name, net_type, properties
            FROM nets
            WHERE circuit_version_id = ?
            ORDER BY rowid
            """,
            (source_version_id,),
        ).fetchall():
            self.conn.execute(
                """
                INSERT INTO nets
                    (net_id, circuit_version_id, name, net_type, properties)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    target_version_id,
                    row["name"],
                    row["net_type"],
                    row["properties"],
                ),
            )

        for row in self.conn.execute(
            """
            SELECT module_id, instance_name, position_in_hierarchy
            FROM circuit_modules
            WHERE circuit_version_id = ?
            ORDER BY position_in_hierarchy, rowid
            """,
            (source_version_id,),
        ).fetchall():
            self.conn.execute(
                """
                INSERT INTO circuit_modules
                    (circuit_version_id, module_id, instance_name,
                     position_in_hierarchy)
                VALUES (?, ?, ?, ?)
                """,
                (
                    target_version_id,
                    row["module_id"],
                    row["instance_name"],
                    row["position_in_hierarchy"],
                ),
            )

    def _copy_spec_declarations_for_promotion(
        self, source_version_id: str, target_version_id: str
    ) -> None:
        """Copy spec declarations and remap every parent to the new snapshot."""
        rows = self.conn.execute(
            """
            SELECT spec_id, owner_path, spec_key, name, metric, op, target,
                   units, description, parent_id, declared_at,
                   definition_scheme, definition_hash
            FROM specs
            WHERE circuit_version_id = ?
            ORDER BY rowid
            """,
            (source_version_id,),
        ).fetchall()
        source_ids = {row["spec_id"] for row in rows}
        new_ids = {}
        pending = list(rows)
        while pending:
            deferred = []
            made_progress = False
            for row in pending:
                old_parent_id = row["parent_id"]
                if old_parent_id is None:
                    new_parent_id = None
                elif old_parent_id not in source_ids:
                    raise DatabaseError(
                        f"Spec {row['spec_id']!r} has a parent outside source "
                        f"snapshot {source_version_id!r}"
                    )
                elif old_parent_id not in new_ids:
                    deferred.append(row)
                    continue
                else:
                    new_parent_id = new_ids[old_parent_id]

                new_spec_id = str(uuid.uuid4())
                self.conn.execute(
                    """
                    INSERT INTO specs
                        (spec_id, circuit_version_id, owner_path, spec_key,
                         name, metric, op, target, units, description,
                         parent_id, declared_at, definition_scheme,
                         definition_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_spec_id,
                        target_version_id,
                        row["owner_path"],
                        row["spec_key"],
                        row["name"],
                        row["metric"],
                        row["op"],
                        row["target"],
                        row["units"],
                        row["description"],
                        new_parent_id,
                        row["declared_at"],
                        row["definition_scheme"],
                        row["definition_hash"],
                    ),
                )
                new_ids[row["spec_id"]] = new_spec_id
                made_progress = True
            if not made_progress:
                raise DatabaseError(
                    f"Spec declarations in source snapshot {source_version_id!r} "
                    "contain a parent cycle"
                )
            pending = deferred

    def promote_circuit(
        self,
        source_version_id: str,
        to_stage: DesignStage,
        *,
        new_version: str = None,
        lock: bool = True,
        created_by: str = "user",
        description: str = None,
    ) -> str:
        """Promote one exact sealed snapshot to another stage.

        Creates (or reuses) a sealed snapshot of the same identity at ``to_stage``
        whose parent is the source. Structure and spec declarations are copied; the
        source's evidence is not.

        Parameters
        ----------
        source_version_id : str
            Version id of the sealed snapshot to promote.
        to_stage : DesignStage or str
            Target stage.
        new_version : str, optional
            Version label of the new snapshot; defaults to
            ``'<source version>.<stage>'``. This and the following parameters are
            keyword-only.
        lock : bool, default True
            Lock the promoted snapshot.
        created_by : str, default "user"
            Author recorded with the new snapshot.
        description : str, optional
            Description; defaults to ``'Promoted from <source stage>'``.

        Returns
        -------
        str
            The version id of the promoted snapshot. If an identical promotion
            already exists it is returned (and locked when ``lock``).

        Raises
        ------
        silk.exceptions.DatabaseError
            The source is missing or not sealed, the exact identity at the target
            stage belongs to a different source or label, or the label is already
            used by a different source.
        ValueError
            ``to_stage`` is not a design stage.
        """
        with self.transaction():
            source = self._snapshot_row(source_version_id)
            if source["state"] != "sealed":
                raise DatabaseError(
                    f"Promotion source {source_version_id!r} must be sealed"
                )
            identity = SnapshotIdentity(
                Fingerprint(
                    source["structure_scheme"], source["structure_hash"]
                ),
                Fingerprint(source["spec_scheme"], source["spec_set_hash"]),
            )
            self._validate_identity(identity)
            target_stage = self._stage_value(to_stage)
            target_label = (
                new_version
                if new_version is not None
                else f"{source['version']}.{target_stage}"
            )
            target_description = (
                description
                if description is not None
                else f"Promoted from {source['stage']}"
            )

            exact_id = self.find_snapshot(
                source["circuit_name"], target_stage, identity
            )
            label_rows = self.conn.execute(
                """
                SELECT version_id, parent_version_id
                FROM circuit_versions
                WHERE circuit_name = ? AND version = ?
                """,
                (source["circuit_name"], target_label),
            ).fetchall()
            if exact_id is not None:
                exact = self._snapshot_row(exact_id)
                if (
                    exact["parent_version_id"] != source_version_id
                    or exact["version"] != target_label
                ):
                    raise DatabaseError(
                        f"The exact {target_stage!r} identity already belongs "
                        "to a promotion from a different source or label"
                    )
                if any(row["version_id"] != exact_id for row in label_rows):
                    raise DatabaseError(
                        f"Promotion target label {target_label!r} is ambiguous"
                    )
                if lock:
                    self.lock_version(exact_id)
                _log.info(
                    "Promotion of snapshot %s to %r already exists as %s; reusing",
                    source_version_id,
                    target_stage,
                    exact_id,
                )
                return exact_id

            if label_rows:
                raise DatabaseError(
                    f"Promotion target label {target_label!r} already exists "
                    "for a different source"
                )

            target_id = self.create_circuit(
                source["circuit_name"],
                target_label,
                stage=target_stage,
                created_by=created_by,
                description=target_description,
                identity=identity,
                parent_version_id=source_version_id,
            )
            self._copy_structure_for_promotion(source_version_id, target_id)
            self._copy_spec_declarations_for_promotion(
                source_version_id, target_id
            )
            self.seal_circuit_version(target_id)
            if lock:
                self.lock_version(target_id)
            return target_id

    def _require_unlocked(self, circuit_version_id: str):
        """Raise if the circuit version is locked.

        All version-attached writers route through a lifecycle guard before
        issuing SQL, so lock enforcement is uniform and typed.
        """
        row = self._snapshot_row(circuit_version_id)
        if row["is_locked"]:
            raise LockedVersionError(
                f"Circuit '{row['circuit_name']}' version '{row['version']}' "
                "is locked; call unlock_version() before writing to it"
            )

    def unlock_version(self, version_id: str) -> None:
        """Clear the lock on one snapshot.

        Parameters
        ----------
        version_id : str
            Version id of the snapshot.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist.
        """
        row = self._snapshot_row(version_id)
        self.conn.execute(
            "UPDATE circuit_versions SET is_locked = 0 WHERE version_id = ?",
            (version_id,),
        )
        self._maybe_commit()
        _log.info(
            "Unlocked circuit %r version %s", row["circuit_name"], row["version"]
        )

    def lock_version(self, version_id: str) -> None:
        """Lock one snapshot against writes that target it.

        Writers then raise :class:`~silk.exceptions.LockedVersionError` until
        :meth:`unlock_version` is called.

        Parameters
        ----------
        version_id : str
            Version id of the snapshot.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist.
        """
        row = self._snapshot_row(version_id)
        self.conn.execute(
            "UPDATE circuit_versions SET is_locked = 1 WHERE version_id = ?",
            (version_id,),
        )
        self._maybe_commit()
        _log.info(
            "Locked circuit %r version %s", row["circuit_name"], row["version"]
        )

    def unlock_circuit(self, circuit_name: str, version: str):
        """Unlock a snapshot found by circuit name and version label.

        A compatibility wrapper around :meth:`get_version` and
        :meth:`unlock_version`.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        version : str
            Version label.

        Raises
        ------
        ValueError
            No snapshot has that name and label.
        silk.exceptions.DatabaseError
            The label is ambiguous.
        """
        circuit = self.get_version(circuit_name, version)
        if not circuit:
            raise ValueError(f"Circuit '{circuit_name}' version '{version}' not found")
        self.unlock_version(circuit["version_id"])

    def lock_circuit(self, circuit_name: str, version: str):
        """Lock a snapshot found by circuit name and version label.

        A compatibility wrapper around :meth:`get_version` and :meth:`lock_version`.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        version : str
            Version label.

        Raises
        ------
        ValueError
            No snapshot has that name and label.
        silk.exceptions.DatabaseError
            The label is ambiguous.
        """
        circuit = self.get_version(circuit_name, version)
        if not circuit:
            raise ValueError(f"Circuit '{circuit_name}' version '{version}' not found")
        self.lock_version(circuit["version_id"])

    def get_circuit_by_id(self, version_id: str) -> Optional[Dict]:
        """Return a snapshot by version id.

        Parameters
        ----------
        version_id : str
            Version id.

        Returns
        -------
        dict or None
            The same keys as :meth:`get_latest`, or ``None`` when the id is not
            stored. Building snapshots are included.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata, created_by
            FROM circuit_versions
            WHERE version_id = ?
        """, (version_id,))

        row = cursor.fetchone()
        if row:
            return {
                'version_id': row[0],
                'circuit_name': row[1],
                'version': row[2],
                'stage': row[3],
                'created_at': row[4],
                'netlist_path': row[5],
                'description': row[6],
                'metadata': decode_json(json.loads(row[7])) if row[7] else {},
                'created_by': row[8],
            }
        return None

    def get_circuit_by_version_id(self, version_id: str) -> Optional[Dict]:
        """Return a snapshot by version id.

        An alias of :meth:`get_circuit_by_id`.

        Parameters
        ----------
        version_id : str
            Version id.

        Returns
        -------
        dict or None
            As :meth:`get_circuit_by_id`.
        """
        return self.get_circuit_by_id(version_id)

    def get_all_circuit_versions(self, circuit_name: str, order_by: str = 'created_at') -> List[Dict]:
        """Return every snapshot of a circuit, building or sealed.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        order_by : str, default "created_at"
            ``'created_at'``, ``'version'`` or ``'stage'``. Any other value falls
            back to ``'created_at'`` without an error.

        Returns
        -------
        list of dict
            Snapshot dicts as returned by :meth:`get_latest`, sorted descending by
            the chosen column.
        """
        _ORDER_COLUMNS = {
            'created_at': 'created_at',
            'version': 'version',
            'stage': 'stage',
        }
        col = _ORDER_COLUMNS.get(order_by, 'created_at')

        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata, created_by
            FROM circuit_versions
            WHERE circuit_name = ?
            ORDER BY {col} DESC
        """, (circuit_name,))

        circuits = []
        for row in cursor.fetchall():
            circuits.append({
                'version_id': row[0],
                'circuit_name': row[1],
                'version': row[2],
                'stage': row[3],
                'created_at': row[4],
                'netlist_path': row[5],
                'description': row[6],
                'metadata': decode_json(json.loads(row[7])) if row[7] else {},
                'created_by': row[8],
            })

        return circuits

    def get_most_simulated_circuit(self, circuit_name: str) -> Optional[Dict]:
        """Return the snapshot of a circuit with the highest recorded run count.

        The count is the ``run_count`` entry of each snapshot's metadata.

        Parameters
        ----------
        circuit_name : str
            Circuit name.

        Returns
        -------
        dict or None
            The snapshot dict (as :meth:`get_latest`) with the largest
            ``metadata['run_count']``. When no snapshot has a positive count, the
            first of :meth:`get_all_circuit_versions` (the newest) is returned.
            ``None`` when the circuit has no snapshots.
        """
        circuits = self.get_all_circuit_versions(circuit_name)

        if not circuits:
            return None

        # Find circuit with max run_count
        max_runs = 0
        most_simulated = circuits[0]

        for circuit in circuits:
            run_count = circuit['metadata'].get('run_count', 0)
            if run_count > max_runs:
                max_runs = run_count
                most_simulated = circuit

        return most_simulated

    def create_release(self, release_name: str, stage: DesignStage,
                      circuit_names: List[str] = None,
                      created_by: str = "user", notes: str = "") -> str:
        """Create a named release of the latest circuits at a stage.

        Parameters
        ----------
        release_name : str
            Unique release name.
        stage : DesignStage
            Stage the release is cut at.
        circuit_names : list of str, optional
            Circuits to include. For each, the newest sealed snapshot at ``stage``
            is added; a name with no sealed snapshot there is skipped silently.
        created_by : str, default "user"
            Author recorded with the release.
        notes : str, default ""
            Free-text notes.

        Returns
        -------
        str
            The new release id.

        Raises
        ------
        sqlite3.IntegrityError
            A release with that name already exists.
        """
        release_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO releases
            (release_id, release_name, stage, created_by, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (release_id, release_name, stage.value, created_by, notes))

        # Add circuits to release
        if circuit_names:
            for name in circuit_names:
                circuit = self.get_latest(name, stage)
                if circuit:
                    cursor.execute("""
                        INSERT INTO release_contents
                        (release_id, circuit_version_id)
                        VALUES (?, ?)
                    """, (release_id, circuit['version_id']))

        self._maybe_commit()
        _log.info("Created release %r at stage %r", release_name, stage.value)
        return release_id

    def list_circuits(self, stage: DesignStage = None) -> List[Dict]:
        """List every snapshot, optionally filtered by stage.

        Parameters
        ----------
        stage : DesignStage, optional
            Restrict to this stage.

        Returns
        -------
        list of dict
            One dict per snapshot (building or sealed) with keys
            ``'circuit_name'``, ``'version'``, ``'stage'``, ``'created_at'`` and
            ``'description'``, ordered by circuit name, then newest first.
        """
        cursor = self.conn.cursor()

        if stage:
            cursor.execute("""
                SELECT circuit_name, version, stage, created_at, description
                FROM circuit_versions
                WHERE stage = ?
                ORDER BY circuit_name, created_at DESC
            """, (stage.value,))
        else:
            cursor.execute("""
                SELECT circuit_name, version, stage, created_at, description
                FROM circuit_versions
                ORDER BY circuit_name, created_at DESC
            """)

        return [dict(row) for row in cursor.fetchall()]

    def get_circuit_summary(self, circuit_name: str, version: str = None) -> Dict:
        """Return a snapshot together with its devices, nets, modules and results.

        Parameters
        ----------
        circuit_name : str
            Circuit name.
        version : str, optional
            Version label. When omitted the newest sealed snapshot is used.

        Returns
        -------
        dict or None
            ``None`` when no snapshot matches. Otherwise a dict with keys
            ``'circuit'`` (the snapshot dict of :meth:`get_latest`), ``'devices'``
            (dicts with ``device_id``, ``name``, ``type``, ``model`` and
            ``properties`` as stored JSON text), ``'nets'`` (dicts with ``name``,
            ``net_type`` and ``role_inferred``, which is True when the net's role
            was guessed from its name rather than declared), ``'modules'`` (as
            :meth:`get_circuit_modules`) and ``'simulation_results'`` (as
            :meth:`get_simulation_results`).

        Raises
        ------
        silk.exceptions.DatabaseError
            The selected snapshot is not sealed (results cannot be read), or the
            label is ambiguous.
        """
        if version:
            circuit = self.get_version(circuit_name, version)
        else:
            circuit = self.get_latest(circuit_name)

        if not circuit:
            return None

        cursor = self.conn.cursor()

        # Get devices
        cursor.execute("""
            SELECT device_id, name, type, model, properties
            FROM devices
            WHERE circuit_version_id = ?
        """, (circuit['version_id'],))
        devices = [dict(row) for row in cursor.fetchall()]

        # Get nets
        cursor.execute("""
            SELECT name, net_type, properties
            FROM nets
            WHERE circuit_version_id = ?
        """, (circuit['version_id'],))
        nets = []
        for row in cursor.fetchall():
            net = dict(row)
            props = decode_json(json.loads(net.pop('properties'))) if net.get('properties') else {}
            # A net's role (net_type) can be declared by the caller or
            # guessed from its name by parse_spice_netlist's ruleset
            # (TASK-97). TWO writer conventions share the properties key
            # for the guess, the same way devices.properties carries both
            # 'w' and 'width' (see CircuitExporter.export_to_json): the
            # importer's own in-memory record uses 'inferred' (unchanged
            # since TASK-97), while a re-imported export projection
            # (below) writes 'role_inferred' to match this method's own
            # field name. Read both. A row with no properties at all --
            # either a pre-TASK-112 import or a manually declared net that
            # never set either key -- reads as role_inferred=False
            # (declared); there is no stored distinction between
            # "declared" and "never asked" for such rows (TASK-112).
            net['role_inferred'] = bool(
                props.get('role_inferred', props.get('inferred', False))
            )
            nets.append(net)

        # Get simulation results
        sim_results = self.get_simulation_results(circuit['version_id'])

        # Get modules
        modules = self.get_circuit_modules(circuit['version_id'])

        return {
            'circuit': circuit,
            'devices': devices,
            'nets': nets,
            'modules': modules,
            'simulation_results': sim_results
        }

    def _simulation_observations_for_pair(self, circuit_version_id, parameter, unit):
        """Rank stored observations within one parameter/unit pair.

        Identity includes the condition states: unknown and none both have
        NULL scalars but are different observations. Rank all value types
        together so a newer text error or unavailable value supersedes an
        older numeric result. SQL orders rows; Python owns rank and count.
        """
        identity = (
            "parameter", "unit", "corner_state", "corner",
            "temperature_state", "temperature", "voltage_state", "voltage",
        )
        rows = self.conn.execute(f"""
            SELECT parameter, unit, corner, temperature, voltage,
                   corner_state, temperature_state, voltage_state,
                   value, value_text, timestamp, rowid
            FROM simulation_results
            WHERE circuit_version_id = ? AND parameter = ? AND unit IS ?
            ORDER BY {', '.join(identity)}, timestamp DESC, rowid DESC
        """, (circuit_version_id, parameter, unit)).fetchall()
        observations = []
        for _key, group in groupby(rows, key=lambda row: tuple(row[k] for k in identity)):
            group = list(group)
            for rank, row in enumerate(group, start=1):
                observations.append(dict(
                    row, latest_rank=rank, observation_count=len(group),
                ))
        return observations

    @staticmethod
    def _format_summary_value(value):
        """Format every summary value, including unavailable numeric evidence."""
        if value is None:
            return "unavailable"
        if isinstance(value, str):
            return value
        return f"{value:.3g}"

    def _format_simulation_summary(self, circuit_version_id: str) -> str:
        """Render current observations per parameter/unit without changing evidence."""
        parameters = self.conn.execute("""
            SELECT DISTINCT parameter, unit
            FROM simulation_results
            WHERE circuit_version_id = ?
            ORDER BY parameter, unit
        """, (circuit_version_id,)).fetchall()
        if not parameters:
            return ""

        summary_lines = [
            "=== SIMULATION SUMMARY ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        status_lines = []
        format_value = self._format_summary_value
        corner_order = {'tt': 1, 'ss': 2, 'ff': 3, 'sf': 4, 'fs': 5}
        display_columns = (
            'corner', 'corner_state', 'temperature', 'temperature_state',
            'voltage', 'voltage_state',
        )
        for param, unit in parameters:
            observations = self._simulation_observations_for_pair(
                circuit_version_id, param, unit,
            )
            unit_str = f" {unit}" if unit else ""
            # Text status history is retained; numeric lines use only current
            # observations, never an older number superseded by a text error.
            for row in sorted(observations, key=lambda row: (row['timestamp'], row['rowid'])):
                if row['value_text'] is not None:
                    status_lines.append(
                        f"{param:20s}: {format_value(row['value_text'])}{unit_str}"
                    )
            latest = [row for row in observations
                      if row['latest_rank'] == 1 and row['value_text'] is None]
            latest.sort(key=lambda row: (
                corner_order.get(row['corner'], 6),
                tuple((row[k] is not None, row[k]) for k in display_columns),
            ))
            n_groups = len(latest)
            if not n_groups:
                continue
            corner_word = "corner" if n_groups == 1 else "corners"
            if n_groups > 1:
                values = [row['value'] for row in latest if row['value'] is not None]
                value_range = (
                    f"{format_value(min(values))} to {format_value(max(values))}"
                    if values else format_value(None)
                )
                summary_lines.append(
                    f"{param:20s}: {value_range}{unit_str} ({n_groups} {corner_word})"
                )
            for row in latest:
                corner_str, conditions = self._summary_condition_labels(
                    *self._row_condition_scalars(row)
                )
                cond_str = f" @ {', '.join(conditions)}" if conditions else ""
                count = row['observation_count']
                latest_suffix = f" (latest of {count})" if count > 1 else ""
                value = format_value(row['value'])
                if n_groups > 1:
                    summary_lines.append(
                        f"  └─ {corner_str:8s}{cond_str:20s}: {value}{unit_str}{latest_suffix}"
                    )
                else:
                    summary_lines.append(
                        f"{param:20s}: {value}{unit_str} ({corner_str}{cond_str}, "
                        f"{n_groups} {corner_word}){latest_suffix}"
                    )
            summary_lines.append("")

        if status_lines:
            summary_lines.extend(["=== STATUS ===", *status_lines, ""])
        return "\n".join(summary_lines)

    @staticmethod
    def _summary_condition_labels(corner, temperature, voltage):
        """Display words for one widened (corner, temperature, voltage).

        An unknown corner keeps its historical ``nominal`` label; none and
        sweep are spelled out, and an unknown temperature/voltage is omitted
        as before.
        """
        from silk.simulation.conditions import ConditionState

        if corner is ConditionState.NONE:
            corner_str = "none"
        else:
            corner_str = corner or "nominal"
        conditions = []
        if temperature is ConditionState.NONE:
            conditions.append("temperature none")
        elif temperature is ConditionState.SWEEP:
            conditions.append("temperature sweep")
        elif temperature is not None:
            conditions.append(f"{temperature}°C")
        if voltage is ConditionState.NONE:
            conditions.append("supply none")
        elif voltage is ConditionState.SWEEP:
            conditions.append("supply sweep")
        elif voltage is not None:
            conditions.append(f"{voltage}V")
        return corner_str, conditions

    def update_simulation_summary(self, circuit_version_id: str) -> str:
        """Render the simulation summary and cache it on a sealed snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed, unlocked snapshot.

        Returns
        -------
        str
            The summary text; empty when the snapshot has no results.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not sealed.
        silk.exceptions.LockedVersionError
            The snapshot is locked.

        Notes
        -----
        Adding a result clears the cached text. :meth:`get_simulation_summary`
        never reads the cache.
        """
        self._require_sealed(circuit_version_id)
        summary_text = self._format_simulation_summary(circuit_version_id)
        self.conn.execute("""
            UPDATE circuit_versions
            SET simulation_summary = ?, last_summary_update = CURRENT_TIMESTAMP
            WHERE version_id = ?
        """, (summary_text, circuit_version_id))
        self._maybe_commit()
        return summary_text

    def get_simulation_summary(self, circuit_version_id: str) -> str:
        """Render the simulation summary of a snapshot from its current records.

        The text is rendered on every call and nothing is written. For each
        metric the newest observation per corner and operating point is listed with
        its range across corners; text observations appear in a trailing status
        block.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot.

        Returns
        -------
        str
            The summary text; empty when the snapshot has no results.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist.
        """
        self._snapshot_row(circuit_version_id)
        return self._format_simulation_summary(circuit_version_id)

    def add_parasitic(self, circuit_version_id: str, net_id: str,
                     parasitic_type: str, value: float, unit: str,
                     from_node: str = None, to_node: str = None,
                     extraction_tool: str = None) -> str:
        """Add parasitic data (for post-layout work) to a sealed snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of a sealed, unlocked snapshot.
        net_id : str
            Id of the net the parasitic belongs to.
        parasitic_type : str
            Kind of parasitic, for example ``'capacitance'``.
        value : float
            Parasitic value, in ``unit``.
        unit : str
            Unit label of ``value``.
        from_node : str, optional
            First node the parasitic connects.
        to_node : str, optional
            Second node the parasitic connects.
        extraction_tool : str, optional
            Tool that extracted the parasitic.

        Returns
        -------
        str
            The new parasitic id.

        Raises
        ------
        silk.exceptions.DatabaseError
            The snapshot does not exist or is not sealed.
        silk.exceptions.LockedVersionError
            The snapshot is locked.
        """
        self._require_sealed(circuit_version_id)
        parasitic_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO parasitics
            (parasitic_id, circuit_version_id, net_id, parasitic_type,
             value, unit, from_node, to_node, extraction_tool)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (parasitic_id, circuit_version_id, net_id, parasitic_type,
              value, unit, from_node, to_node, extraction_tool))

        self._maybe_commit()
        return parasitic_id

    def get_latest_circuit(self, circuit_name: str) -> Optional[Dict]:
        """Return the newest sealed snapshot of a circuit.

        An alias of :meth:`get_latest` without the stage filter.

        Parameters
        ----------
        circuit_name : str
            Circuit name.

        Returns
        -------
        dict or None
            As :meth:`get_latest`.
        """
        return self.get_latest(circuit_name)

    @staticmethod
    def _version_sort_key(version: str):
        """Sort key that orders version strings numerically per dotted
        segment, so ``1.10 > 1.9`` (SQL ``ORDER BY version`` is lexicographic
        on the TEXT column and got exactly that case wrong). A non-numeric
        segment falls back to string comparison within its position; numeric
        segments always sort after non-numeric ones so ``1.10`` beats
        ``1.rc1``.
        """
        key = []
        for seg in str(version).split('.'):
            try:
                key.append((1, int(seg), ''))
            except ValueError:
                key.append((0, 0, seg))
        return key

    def get_latest_circuit_by_version(self, circuit_name: str) -> Optional[Dict]:
        """Return the snapshot of a circuit with the highest version label.

        Labels are compared segment by segment on dots, numerically where a
        segment is an integer (so ``1.10`` is higher than ``1.9``); numeric
        segments sort after non-numeric ones. Building and sealed snapshots are
        both considered.

        Parameters
        ----------
        circuit_name : str
            Circuit name.

        Returns
        -------
        dict or None
            The snapshot dict (as :meth:`get_latest`), or ``None`` when the circuit
            has no snapshots.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata, created_by
            FROM circuit_versions
            WHERE circuit_name = ?
        """, (circuit_name,))

        rows = cursor.fetchall()
        row = (max(rows, key=lambda r: self._version_sort_key(r[2]))
               if rows else None)
        if row:
            return {
                'version_id': row[0],
                'circuit_name': row[1],
                'version': row[2],
                'stage': row[3],
                'created_at': row[4],
                'netlist_path': row[5],
                'description': row[6],
                'metadata': decode_json(json.loads(row[7])) if row[7] else {},
                'created_by': row[8],
            }
        return None

    def latest_at_stage(self, stage: DesignStage) -> List[Dict]:
        """Return the newest sealed snapshot of each circuit at a stage.

        Only sealed snapshots are listed. A circuit whose snapshots at the
        stage are all still building is left out; use :meth:`list_circuits`
        with ``stage`` to see work in progress.

        Parameters
        ----------
        stage : DesignStage
            Stage to list.

        Returns
        -------
        list of dict
            One entry per circuit with a sealed snapshot at ``stage``, as
            :meth:`get_latest` returns it. No entry is ``None``.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT circuit_name FROM circuit_versions
            WHERE stage = ? AND state = 'sealed'
        """, (stage.value,))

        circuit_names = [row[0] for row in cursor.fetchall()]
        return [self.get_latest(name, stage) for name in circuit_names]

    def get_corners_for_parameter(self, circuit_version_id: str,
                                  parameter: str,
                                  latest_only: bool = False) -> List[Dict]:
        """Return the results of one parameter across corners.

        The results table is append-only: storing a parameter again at the same
        condition adds a row. By default every observation is returned; with
        ``latest_only`` each distinct (parameter, unit, corner, temperature,
        voltage) keeps only its newest observation.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot (rows stored on that snapshot only).
        parameter : str
            Metric name.
        latest_only : bool, default False
            Collapse to the newest observation per condition group. Newest means
            greatest timestamp, then storage order. A text observation then
            appears as the ``'value'``, and unavailable numeric evidence as
            ``None``; an older number is never substituted.

        Returns
        -------
        list of dict
            Dicts with keys ``'corner'``, ``'temperature'`` and ``'voltage'``
            (condition scalars), ``'value'``, ``'unit'`` and ``'timestamp'``,
            ordered by condition (corner, then temperature, then voltage).
        """
        cursor = self.conn.cursor()
        # Schema v5 (TASK-409): corner/temperature/voltage come back as
        # widened condition scalars, and a none and an unknown fact (both
        # NULL) are distinct groups.
        if latest_only:
            units = cursor.execute("""
                SELECT DISTINCT unit FROM simulation_results
                WHERE circuit_version_id = ? AND parameter = ?
                ORDER BY unit
            """, (circuit_version_id, parameter)).fetchall()
            rows = [
                row for (unit,) in units
                for row in self._simulation_observations_for_pair(
                    circuit_version_id, parameter, unit,
                )
                if row['latest_rank'] == 1
            ]
            display_columns = (
                'corner', 'corner_state', 'temperature', 'temperature_state',
                'voltage', 'voltage_state', 'unit',
            )
            rows.sort(key=lambda row: tuple(
                (row[k] is not None, row[k]) for k in display_columns
            ))
        else:
            cursor.execute("""
                SELECT corner, temperature, voltage, value, unit, timestamp,
                       corner_state, temperature_state, voltage_state
                FROM simulation_results
                WHERE circuit_version_id = ? AND parameter = ?
                ORDER BY corner, corner_state, temperature, temperature_state,
                         voltage, voltage_state
            """, (circuit_version_id, parameter))

            rows = cursor.fetchall()

        results = []
        for row in rows:
            corner, temperature, voltage = self._row_condition_scalars(row)
            results.append({
                'corner': corner,
                'temperature': temperature,
                'voltage': voltage,
                'value': (row['value_text'] if latest_only and row['value_text'] is not None
                          else row['value']),
                'unit': row['unit'],
                'timestamp': row['timestamp'],
            })
        return results

    def get_corner_summary(self, circuit_version_id: str) -> Dict:
        """Return the corners a snapshot has results for.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot (rows stored on that snapshot only).

        Returns
        -------
        dict
            Maps a corner label to a list of dicts, each with ``'corner'`` (the
            label), ``'temperature'``, ``'voltage'`` and ``'display_name'`` (for
            example ``'tt (27.0°C, 1.8V)'``). The label is the corner name, ``'none'``
            when no corner applies, or ``'nominal'`` when the corner is unknown.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT corner, temperature, voltage,
                   corner_state, temperature_state, voltage_state
            FROM simulation_results
            WHERE circuit_version_id = ?
            ORDER BY corner, corner_state, temperature, temperature_state,
                     voltage, voltage_state
        """, (circuit_version_id,))

        corners = {}
        for row in cursor.fetchall():
            # Schema v5 (TASK-409): a none corner groups under 'none', apart
            # from the unknown corner's historical 'nominal'.
            raw_corner, temp, volt = self._row_condition_scalars(row)
            corner, conditions = self._summary_condition_labels(
                raw_corner, temp, volt
            )

            key = corner
            if conditions:
                key = f"{corner} ({', '.join(conditions)})"

            if corner not in corners:
                corners[corner] = []
            corners[corner].append({
                'corner': corner,
                'temperature': temp,
                'voltage': volt,
                'display_name': key
            })

        return corners

    def get_nets_for_circuit(self, circuit_version_id: str) -> List[Dict]:
        """Return the nets stored for a snapshot.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot.

        Returns
        -------
        list of dict
            One dict per net with keys ``'net_id'``, ``'name'`` and ``'net_type'``.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT net_id, name, net_type FROM nets
            WHERE circuit_version_id = ?
        """, (circuit_version_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_parameters_for_circuit(self, circuit_version_id: str) -> List[str]:
        """Return the metric names a snapshot has results for.

        Parameters
        ----------
        circuit_version_id : str
            Version id of the snapshot (rows stored on that snapshot only).

        Returns
        -------
        list of str
            Distinct parameter names in ascending order.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT parameter
            FROM simulation_results
            WHERE circuit_version_id = ?
            ORDER BY parameter
        """, (circuit_version_id,))
        return [row[0] for row in cursor.fetchall()]

    def insert_spec_measurement(self, measurement_id, spec_id, source, value,
                                corner=None, timestamp=None, sim_run_id=None,
                                notes='', passes=None, *, temperature=None,
                                voltage=None, sim_type=None, tool_name=None,
                                profile_id=None, provenance_id=None, aggregate=None):
        """Insert one validated, context-bearing spec observation.

        Parameters
        ----------
        measurement_id : str
            Identifier for the measurement, chosen by the caller.
        spec_id : str
            Id of a stored spec declaration row. The measurement is attached to the
            snapshot that owns it.
        source : str
            ``'behavioral'``, ``'spice'``, ``'extraction'`` or ``'user'``.
        value : float
            Measured value in the units of the spec. A ``bool`` is refused.
        corner : str or ConditionState, optional
            Process corner; ``None`` is unknown.
        timestamp : float, optional
            Observation time (Unix seconds by the ``Measurement`` convention).
        sim_run_id : str, optional
            External run-correlation identifier; required to match an aggregate's
            campaign id when ``aggregate`` is given.
        notes : str, default ""
            Free-text note.
        passes : bool, optional
            Cached verdict. If given it must agree with the declaration's operator
            and target.
        temperature : float or ConditionState, optional
            Temperature in degrees Celsius (keyword-only from here on).
        voltage : float or ConditionState, optional
            Supply voltage in volts.
        sim_type : str, optional
            Analysis label.
        tool_name : str, optional
            Producing tool.
        profile_id : str, optional
            Id of an existing characterization profile of the owning snapshot; then
            corner, temperature and voltage must each be a value or
            ``ConditionState.NONE``.
        provenance_id : str, optional
            Id of an existing run-provenance row.
        aggregate : AggregateMeasurement, optional
            Monte Carlo attribution when ``value`` summarises a campaign.

        Raises
        ------
        silk.exceptions.DatabaseError
            The spec does not exist, or its snapshot is not sealed.
        silk.exceptions.LockedVersionError
            The owning snapshot is locked.
        ValueError
            ``source`` is not allowed, a scoped measurement has an unknown or
            sweep condition, or ``aggregate`` does not belong to the spec's
            metric or ``sim_run_id``.
        TypeError
            A field has the wrong type (for example ``value`` is not a real number,
            ``passes`` is not a bool, ``notes`` is not a string).
        silk.exceptions.EvidenceMismatchError
            ``profile_id`` or ``provenance_id`` is not stored, or ``passes``
            contradicts the value, operator and target.
        """
        declaration = self.conn.execute(
            """
            SELECT circuit_version_id, op, target, metric
            FROM specs WHERE spec_id = ?
            """,
            (spec_id,),
        ).fetchone()
        if declaration is None:
            raise DatabaseError(f"Spec {spec_id!r} does not exist")
        self._require_sealed(declaration["circuit_version_id"])
        states = self._row_condition_columns(
            corner, temperature, voltage, subject="Spec measurement"
        )
        if profile_id is not None:
            if not isinstance(profile_id, str) or not profile_id:
                raise TypeError(
                    "Spec measurement profile_id must be a non-empty string "
                    "or None"
                )
            self._require_profile_states(states, subject="spec measurement")
            owner = self.conn.execute(
                """
                SELECT 1 FROM characterization_profiles
                WHERE circuit_version_id = ? AND profile_id = ?
                """,
                (declaration["circuit_version_id"], profile_id),
            ).fetchone()
            if owner is None:
                raise EvidenceMismatchError(
                    f"profile_id {profile_id!r} is not a stored "
                    f"characterization profile for circuit version "
                    f"{declaration['circuit_version_id']!r}"
                )
        if source not in _SPEC_MEASUREMENT_SOURCES:
            raise ValueError(
                "Spec measurement source must be one of "
                f"{sorted(_SPEC_MEASUREMENT_SOURCES)}, got {source!r}"
            )
        value = _spec_measurement_real(value, field="value")
        (corner_state, corner), (temperature_state, temperature), (
            voltage_state, voltage) = states
        timestamp = _spec_measurement_real(
            timestamp, field="timestamp", optional=True
        )
        for field, item in (
            ("sim_run_id", sim_run_id),
            ("sim_type", sim_type),
            ("tool_name", tool_name),
        ):
            if item is not None and not isinstance(item, str):
                raise TypeError(
                    f"Spec measurement {field} must be a string or None"
                )
        if not isinstance(notes, str):
            raise TypeError("Spec measurement notes must be a string")
        if passes is not None and not isinstance(passes, bool):
            raise TypeError("Spec measurement passes must be a bool or None")
        evaluated = _evaluate_spec_measurement(
            op=declaration["op"], value=value, target=declaration["target"]
        )
        if passes is not None and passes != evaluated:
            raise EvidenceMismatchError(
                "Spec measurement cached pass does not match its value, "
                "operator, and target"
            )
        from silk.specs.aggregate import encode_aggregate, validate_attribution
        validate_attribution(aggregate, declaration['metric'], sim_run_id)
        aggregate_json = (None if aggregate is None else json.dumps(
            encode_json(encode_aggregate(aggregate)), sort_keys=True, separators=(',', ':'), allow_nan=False))
        self._require_run_provenance(provenance_id)
        self.conn.execute("""
            INSERT INTO spec_measurements (
                measurement_id, circuit_version_id, spec_id, source, value,
                corner, temperature, voltage, sim_type, tool_name, timestamp,
                sim_run_id, notes, passes, profile_id, provenance_id,
                corner_state, temperature_state, voltage_state, aggregate_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            measurement_id, declaration["circuit_version_id"], spec_id,
            source, value, corner, temperature,
            voltage, sim_type, tool_name, timestamp, sim_run_id, notes,
            int(passes) if passes is not None else None, profile_id,
            provenance_id, corner_state, temperature_state, voltage_state, aggregate_json,
        ))
        self._maybe_commit()

    def store_run_provenance(self, record) -> str:
        """Store a run-provenance record and return its id.

        Records are content-addressed: a record whose scheme and digest are
        already stored returns the existing id, so all rows of one run share one
        provenance row. The digest excludes ``timestamp_utc``, so the stored JSON
        keeps the first record's timestamp.

        Parameters
        ----------
        record : RunProvenance
            The provenance to store.

        Returns
        -------
        str
            The provenance id to pass as ``provenance_id`` to the evidence writers.

        Raises
        ------
        TypeError
            ``record`` is not a ``RunProvenance``.
        """
        from silk.simulation.provenance import RunProvenance
        if not isinstance(record, RunProvenance):
            raise TypeError("store_run_provenance requires a RunProvenance")
        digest = record.digest()
        row = self.conn.execute(
            "SELECT provenance_id FROM run_provenance "
            "WHERE scheme = ? AND digest = ?",
            (record.scheme, digest),
        ).fetchone()
        if row is not None:
            return row["provenance_id"]
        provenance_id = str(uuid.uuid4())
        record_json = json.dumps(
            encode_json(record.to_dict()), sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        self.conn.execute(
            "INSERT INTO run_provenance "
            "(provenance_id, scheme, digest, record_json) VALUES (?, ?, ?, ?)",
            (provenance_id, record.scheme, digest, record_json),
        )
        self._maybe_commit()
        return provenance_id

    def get_run_provenance(self, provenance_id):
        """Return the stored run provenance for an id.

        Parameters
        ----------
        provenance_id : str or None
            Id from a record's ``provenance_id``.

        Returns
        -------
        RunProvenance or None
            The stored record, or ``None`` when ``provenance_id`` is ``None``.

        Raises
        ------
        silk.exceptions.DatabaseError
            ``provenance_id`` is not stored.
        """
        if provenance_id is None:
            return None
        from silk.simulation.provenance import RunProvenance
        row = self.conn.execute(
            "SELECT record_json FROM run_provenance WHERE provenance_id = ?",
            (provenance_id,),
        ).fetchone()
        if row is None:
            raise DatabaseError(
                f"run_provenance {provenance_id!r} does not exist")
        return RunProvenance.from_dict(decode_json(json.loads(row["record_json"])))

    def _require_run_provenance(self, provenance_id) -> None:
        if provenance_id is None:
            return
        if not isinstance(provenance_id, str) or not provenance_id:
            raise TypeError("provenance_id must be a non-empty string or None")
        row = self.conn.execute(
            "SELECT 1 FROM run_provenance WHERE provenance_id = ?",
            (provenance_id,),
        ).fetchone()
        if row is None:
            raise EvidenceMismatchError(
                f"provenance_id {provenance_id!r} is not a stored "
                "run_provenance row")

    def close(self):
        """Close the database connection.

        The database is unusable afterwards. Leaving a ``with`` block calls this
        automatically.
        """
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _read_aggregate(row, metric):
    from silk.specs.aggregate import decode_aggregate, validate_attribution
    try:
        aggregate = decode_aggregate(json.loads(row['aggregate_json'])) if row['aggregate_json'] is not None else None
        validate_attribution(aggregate, metric, row['sim_run_id'])
        return aggregate
    except (TypeError, ValueError) as exc:
        raise ValueError(f"measurement {row['measurement_id']!r}: {exc}") from exc
