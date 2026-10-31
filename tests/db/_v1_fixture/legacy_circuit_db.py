"""
FROZEN schema-v1 fixture builder (TASK-4, DB-02-SCHEMA, 2026-08-31).

This is a verbatim snapshot of ``silk/db/circuit_db.py`` as it stood at
commit 04bebf7 (TASK-2's merge), immediately before the schema-v2 DDL
change. It exists ONLY so migration tests can build a genuine v1-shaped
database through the real v1 write path (``add_device``, ``add_net``,
``store_spec_declarations``, ``replace_net_current_set``, ...), instead of a
hand-authored SQLite file. Never import this from non-test code, and never
"fix" it to track the live (v2) ``circuit_db.py`` — its entire value is
being a stable snapshot of the v1 physical shape and behavior.

Circuit Design Database with Version Control and Stage Management
Supports tracking circuits, nets, transistors, and simulation results across design stages

UPDATED: Refactored module tracking to:
- Store unique modules in separate 'modules' table
- Use junction table 'circuit_modules' to link circuits to modules
- Avoid module duplication across circuit versions
"""

import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import logging
import uuid
import hashlib
from datetime import datetime
from numbers import Real
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

from silk.db.identity import (
    SPEC_FINGERPRINT_SCHEME,
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprint_spec_definition_fields,
    _fingerprint_spec_set_records,
    _fingerprints_equal,
    _require_fingerprint_scheme,
)
from silk.db.models import (
    Fingerprint,
    SimulationResultRecord,
    SnapshotIdentity,
    SpecDeclarationRecord,
    SpecMeasurementRecord,
)
from tests.db._v1_fixture.legacy_schema import (
    DATABASE_FORMAT,
    DATABASE_SCHEMA_VERSION,
    inspect_existing_database,
    validate_database_connection,
)
from silk.exceptions import (
    DatabaseError,
    EvidenceMismatchError,
    LockedVersionError,
)


_log = logging.getLogger(__name__)

_SPEC_MEASUREMENT_SOURCES = frozenset(
    {"behavioral", "spice", "extraction", "user"}
)
_NET_CURRENT_SOURCES = frozenset({"spice", "user"})
_NET_CURRENT_CONTEXT_FIELDS = (
    "corner",
    "temperature",
    "voltage",
    "tool_name",
    "analysis",
    "tstart",
    "results_file",
)
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
    """Reject corrupt evidence before a caller can apply any row."""
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
    _net_current_text(
        row["corner"],
        field="corner",
        error_type=EvidenceMismatchError,
        subject=subject,
    )
    for field in ("temperature", "voltage", "tstart"):
        _net_current_real(
            row[field],
            field=field,
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


class DesignStage(Enum):
    """Design stages in the flow"""
    DEV = "dev"
    STABLE = "stable"
    POST_LAYOUT = "post_layout"
    TAPEOUT = "tapeout"


class Database:
    """Main database interface for circuit design data"""

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
        """Group multiple writes into a single atomic transaction.

        Write methods auto-commit per call by default; inside this context they
        defer to one commit on success (or a rollback on any exception), so a
        mid-sequence failure or kill cannot leave partial state. Re-entrant.

        ::

            with db.transaction():
                db.add_device(...)
                db.add_net(...)      # both land, or neither
        """
        outermost = self._txn_depth == 0
        if outermost:
            self.conn.execute("BEGIN IMMEDIATE")
        self._txn_depth += 1
        try:
            yield self
        except BaseException:
            self._txn_depth -= 1
            if self._txn_depth == 0:
                self.conn.rollback()
            raise
        else:
            self._txn_depth -= 1
            if self._txn_depth == 0:
                self._maybe_commit()

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

    def _initialize_schema(self):
        """Create and identify a new database as one atomic operation."""
        with self.transaction():
            self._create_tables()
            self.conn.execute(
                """
                INSERT INTO silk_metadata
                    (singleton, format_id, schema_version, creator_version)
                VALUES (1, ?, ?, ?)
                """,
                (
                    DATABASE_FORMAT,
                    DATABASE_SCHEMA_VERSION,
                    self._creator_version(),
                ),
            )
            validate_database_connection(self.conn, path_label=self.db_path)

    def _create_tables(self):
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

        # Simulation results table
        cursor.execute("""
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
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
            )
        """)

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

        # Net operating currents (EM) — spec 2026-07-18. pin_id '' = the
        # net-level aggregate row (½·Σ bounds); non-empty = one pin's signed
        # currents. Sentinels (not NULL) keep the UNIQUE real: SQLite treats
        # NULLs as pairwise-distinct in unique indexes. structure_hash is
        # stored per-row so load can verify freshness independently of version
        # metadata. Shared operating context describes one complete unnamed
        # last-write-wins set per (version, corner). Partial nets store pin
        # rows only — an absent aggregate row round-trips as None, never 0.0.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS net_currents (
                current_id TEXT PRIMARY KEY,
                circuit_version_id TEXT NOT NULL,
                net_name TEXT NOT NULL,
                pin_id TEXT NOT NULL DEFAULT '',
                avg_a REAL,
                rms_a REAL,
                peak_a REAL,
                source TEXT NOT NULL CHECK(source IN ('spice', 'user')),
                current_vector TEXT,
                analysis TEXT,
                tstart REAL,
                results_file TEXT,
                corner TEXT NOT NULL,
                temperature REAL,
                voltage REAL,
                tool_name TEXT,
                structure_scheme TEXT NOT NULL,
                structure_hash TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(circuit_version_id, net_name, pin_id, corner),
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id)
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
                FOREIGN KEY (circuit_version_id) REFERENCES circuit_versions(version_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_id) REFERENCES specs(spec_id)
            )
        """)

        # Spec measurements table - one row per observation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spec_measurements (
                measurement_id TEXT PRIMARY KEY,
                spec_id TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('behavioral', 'spice', 'extraction', 'user')),
                value REAL NOT NULL,
                corner TEXT,
                temperature REAL,
                voltage REAL,
                sim_type TEXT,
                tool_name TEXT,
                timestamp REAL,
                sim_run_id TEXT,
                notes TEXT,
                passes INTEGER,
                FOREIGN KEY (spec_id) REFERENCES specs(spec_id) ON DELETE CASCADE
            )
        """)

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
                circuit_version_id, parameter, source, sim_type, corner,
                temperature, voltage, tool_name, timestamp
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_net_currents_context
            ON net_currents(circuit_version_id, corner, timestamp)
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
                spec_id, source, corner, temperature, voltage, tool_name,
                timestamp
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

    def find_snapshot(
        self,
        circuit_name: str,
        stage,
        identity: SnapshotIdentity,
    ) -> Optional[str]:
        """Return the exact sealed snapshot for an identity, if present."""
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
        """Return ``version_id`` followed by its parents, rejecting cycles."""
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
        """Reject a snapshot whose stored applicability differs."""
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
        """Verify the declaration set, then transition a building row only."""
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
            if id(spec) in locations_by_object:
                raise DatabaseError(
                    "One live Spec object appears more than once in the "
                    "declaration set"
                )
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
            locations_by_object[id(spec)] = location

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
            actual_parent_location = (
                None
                if actual_parent is None
                else locations_by_object.get(id(actual_parent))
            )
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
        """Atomically insert one complete validated declaration snapshot."""
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
    ) -> List[SpecDeclarationRecord]:
        """Return validated historical declarations without rebuilding Python."""
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
                    measurement_rows = self.conn.execute(
                        """
                        SELECT rowid AS measurement_rowid, measurement_id,
                               source, value, corner, temperature, voltage,
                               sim_type, tool_name, timestamp, sim_run_id,
                               notes, passes
                        FROM spec_measurements
                        WHERE spec_id = ?
                        """,
                        (origin_row["spec_id"],),
                    ).fetchall()
                    for measurement in measurement_rows:
                        source = measurement["source"]
                        if source not in _SPEC_MEASUREMENT_SOURCES:
                            raise EvidenceMismatchError(
                                f"Stored spec measurement "
                                f"{measurement['measurement_id']!r} has invalid "
                                f"source {source!r}"
                            )
                        value = _spec_measurement_real(
                            measurement["value"], field="value"
                        )
                        temperature = _spec_measurement_real(
                            measurement["temperature"],
                            field="temperature",
                            optional=True,
                        )
                        voltage = _spec_measurement_real(
                            measurement["voltage"],
                            field="voltage",
                            optional=True,
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
                            corner=measurement["corner"],
                            temperature=temperature,
                            voltage=voltage,
                            sim_type=measurement["sim_type"],
                            tool_name=measurement["tool_name"],
                            timestamp=timestamp,
                            sim_run_id=measurement["sim_run_id"],
                            notes=measurement["notes"] or "",
                            passes=(
                                None
                                if measurement["passes"] is None
                                else bool(measurement["passes"])
                            ),
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
            ("corner", corner),
            ("temperature", temperature),
            ("voltage", voltage),
            ("tool_name", tool_name),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
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
        """
        Insert a new module or return existing module_id if hash already exists.
        This prevents module duplication across circuit versions.

        Args:
            module_name: Name of the module (e.g., 'ldo_output_stage')
            module_path: Hierarchical path (e.g., 'top.ldo.output')
            module_hash: Content hash of the module structure
            device_count: Number of devices in the module
            structure: Dict containing module structure (devices, nets, submodules)

        Returns:
            str: module_id (existing or newly created)
        """
        _require_fingerprint_scheme(
            Fingerprint(module_hash_scheme, module_hash),
            STRUCTURE_FINGERPRINT_SCHEME,
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
            json.dumps(structure) if structure else None,
        ))

        self._maybe_commit()
        return module_id

    def link_module_to_circuit(self, circuit_version_id: str, module_id: str,
                               instance_name: str = None, position: int = 0):
        """
        Create a link between a circuit and a module in the junction table.

        Args:
            circuit_version_id: Circuit version ID
            module_id: Module ID to link
            instance_name: Instance name in the circuit (optional)
            position: Position in hierarchy (optional)
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
        """
        Get all modules associated with a circuit version.

        Args:
            circuit_version_id: Circuit version ID

        Returns:
            List of dicts containing module information
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
                'structure': json.loads(row[6]) if row[6] else None,
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
        """
        Find all circuits that use a module with the given hash.
        Useful for impact analysis when considering module changes.

        Args:
            module_hash: Hash of the module

        Returns:
            List of circuit version info dicts
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
        """
        Compare modules between two circuit versions to see what changed.

        Args:
            circuit_id_1: First circuit version ID
            circuit_id_2: Second circuit version ID

        Returns:
            Dict with 'added', 'removed', 'unchanged' module lists
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
        """Create one identity-bearing snapshot in ``building`` state.

        ``version`` is a descriptive label and may repeat. The deprecated
        ``allow_duplicate_version`` slot remains for positional compatibility;
        identity, rather than that flag, now controls reuse.
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
            json.dumps(metadata) if metadata else None,
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
        """Add a device (transistor, resistor, etc.) to a circuit version"""
        self._require_building(circuit_version_id)
        device_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO devices
            (device_id, circuit_version_id, name, type, model, properties)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (device_id, circuit_version_id, name, device_type, model,
              json.dumps(properties) if properties else None))

        self._maybe_commit()
        return device_id

    def add_transistor(self, circuit_version_id: str, name: str,
                       transistor_type: str, width: float, length: float,
                       multiplier: int = 1, model: str = None) -> str:
        """Add a transistor device to a circuit version"""
        return self.add_device(circuit_version_id, name, transistor_type,
                               model=model, width=width, length=length,
                               multiplier=multiplier)

    def add_net(self, circuit_version_id: str, name: str, net_type: str = None,
                **properties) -> str:
        """Add a net to a circuit version"""
        self._require_building(circuit_version_id)
        net_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO nets
            (net_id, circuit_version_id, name, net_type, properties)
            VALUES (?, ?, ?, ?, ?)
        """, (net_id, circuit_version_id, name, net_type,
              json.dumps(properties) if properties else None))

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
                             timestamp: str = None) -> str:
        """Add exactly one typed numeric or text observation."""
        self._require_sealed(circuit_version_id)
        if (value is None) == (value_text is None):
            raise ValueError(
                "A simulation result requires exactly one of value or "
                "value_text"
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
        sim_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO simulation_results
            (sim_id, circuit_version_id, object_id, object_type, source, sim_type,
             corner, temperature, voltage, parameter, value, value_text, unit,
             tool_name, timestamp, conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE(?, CURRENT_TIMESTAMP), ?)
        """, (sim_id, circuit_version_id, object_id, object_type, source,
              sim_type, corner, temperature, voltage, parameter, value,
              value_text, unit, tool_name, timestamp, conditions_json))

        # Invalidate the cached summary (get_simulation_summary regenerates
        # lazily on next read) — without this, a store after the first read
        # left view_summary() permanently stale.
        cursor.execute("""
            UPDATE circuit_versions
            SET simulation_summary = NULL, last_summary_update = NULL
            WHERE version_id = ?
        """, (circuit_version_id,))

        self._maybe_commit()
        return sim_id

    @staticmethod
    def _canonical_simulation_conditions(conditions) -> Optional[str]:
        if conditions is None:
            return None
        if not isinstance(conditions, Mapping):
            raise TypeError("Simulation result conditions must be a mapping")
        try:
            return json.dumps(
                dict(conditions),
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
    ) -> int:
        """Atomically replace one complete unnamed current set.

        The set identity is ``(circuit_version_id, corner)``. Operating
        context describes that last-write-wins set; named/multiple profiles
        remain deliberately out of scope for schema v1.
        """
        if corner is None:
            raise ValueError("Current evidence corner is required")
        corner = _net_current_text(corner, field="corner")
        temperature = _net_current_real(
            temperature, field="temperature"
        )
        voltage = _net_current_real(voltage, field="voltage")
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

            self.conn.execute(
                """
                DELETE FROM net_currents
                WHERE circuit_version_id = ? AND corner = ?
                """,
                (circuit_version_id, corner),
            )
            for net_name, pin_id, avg, rms, peak, source, vector in staged:
                self.conn.execute(
                    """
                    INSERT INTO net_currents
                        (current_id, circuit_version_id, net_name, pin_id,
                         avg_a, rms_a, peak_a, source, current_vector,
                         analysis, tstart, results_file, corner, temperature,
                         voltage, tool_name, structure_scheme, structure_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        circuit_version_id,
                        net_name,
                        pin_id,
                        avg,
                        rms,
                        peak,
                        source,
                        vector,
                        analysis,
                        tstart,
                        results_file,
                        corner,
                        temperature,
                        voltage,
                        tool_name,
                        structure_scheme,
                        structure_hash,
                    ),
                )
        return len(staged)

    def get_net_current_set(
        self,
        circuit_version_id: str,
        corner: str,
        *,
        include_ancestors: bool = True,
    ):
        """Return ``(origin_version_id, rows)`` for compatible current data.

        Selection is deterministic: the requested sealed version first, then
        its parent lineage. A candidate must have the same supported structure
        fingerprint; spec-set differences are intentionally irrelevant.
        """
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
            rows = self.conn.execute(
                """
                SELECT rowid AS current_rowid, current_id,
                       circuit_version_id AS origin_version_id,
                       net_name, pin_id, avg_a, rms_a, peak_a, source,
                       current_vector, analysis, tstart, results_file, corner,
                       temperature, voltage, tool_name, structure_scheme,
                       structure_hash, timestamp
                FROM net_currents
                WHERE circuit_version_id = ? AND corner = ?
                ORDER BY timestamp DESC, rowid DESC
                """,
                (version_id, corner),
            ).fetchall()
            if not compatible:
                if rows:
                    incompatible_with_corner.append(version_id)
                continue
            if not rows:
                available.update(
                    row["corner"]
                    for row in self.conn.execute(
                        """
                        SELECT DISTINCT corner FROM net_currents
                        WHERE circuit_version_id = ?
                        """,
                        (version_id,),
                    ).fetchall()
                )
                continue

            expected_context = tuple(
                rows[0][field] for field in _NET_CURRENT_CONTEXT_FIELDS
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
                context = tuple(
                    row[field] for field in _NET_CURRENT_CONTEXT_FIELDS
                )
                if context != expected_context:
                    raise EvidenceMismatchError(
                        "Stored current-set context is inconsistent across rows"
                    )
            return version_id, rows

        if incompatible_with_corner:
            raise EvidenceMismatchError(
                "Current evidence for the requested corner exists only on an "
                "incompatible structure in the version lineage"
            )
        raise RuntimeError(
            f"load_net_currents: version found but no rows for corner "
            f"{corner!r}. Available corners: {sorted(available)}."
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
        alias="sr",
    ):
        clauses = []
        params = []
        for column, requested in (
            ("parameter", parameter),
            ("source", source),
            ("sim_type", sim_type),
            ("corner", corner),
            ("temperature", temperature),
            ("voltage", voltage),
            ("tool_name", tool_name),
        ):
            if requested is not None:
                clauses.append(f"{alias}.{column} = ?")
                params.append(requested)
        canonical_conditions = self._canonical_simulation_conditions(conditions)
        if canonical_conditions is not None:
            clauses.append(f"{alias}.conditions = ?")
            params.append(canonical_conditions)
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
        latest_only: bool = False,
        include_ancestors: bool = True,
    ) -> List[SimulationResultRecord]:
        """Return typed compatible observations in deterministic order."""
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
        )
        clauses.extend(context_clauses)
        params.extend(context_params)
        rows = self.conn.execute(
            f"""
            SELECT sr.rowid AS result_rowid, sr.sim_id,
                   sr.circuit_version_id, sr.parameter, sr.value,
                   sr.value_text, sr.unit, sr.object_id, sr.object_type,
                   sr.source, sr.sim_type, sr.corner, sr.temperature,
                   sr.voltage, sr.tool_name, sr.conditions, sr.timestamp
            FROM simulation_results AS sr
            WHERE {' AND '.join(clauses)}
            """,
            params,
        ).fetchall()

        entries = []
        for row in rows:
            record = SimulationResultRecord(
                result_id=row["sim_id"],
                origin_version_id=row["circuit_version_id"],
                parameter=row["parameter"],
                value=row["value"],
                value_text=row["value_text"],
                unit=row["unit"],
                object_id=row["object_id"],
                object_type=row["object_type"],
                source=row["source"],
                sim_type=row["sim_type"],
                corner=row["corner"],
                temperature=row["temperature"],
                voltage=row["voltage"],
                tool_name=row["tool_name"],
                conditions=self._decode_simulation_conditions(
                    row["conditions"], result_id=row["sim_id"]
                ),
                timestamp=row["timestamp"],
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
                    canonical,
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
        """Get the latest version of a circuit, optionally filtered by stage"""
        cursor = self.conn.cursor()

        if stage:
            cursor.execute("""
                SELECT version_id, circuit_name, version, stage, created_at,
                       netlist_path, description, metadata
                FROM circuit_versions
                WHERE circuit_name = ? AND stage = ? AND state = 'sealed'
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
            """, (circuit_name, stage.value))
        else:
            cursor.execute("""
                SELECT version_id, circuit_name, version, stage, created_at,
                       netlist_path, description, metadata
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
                'metadata': json.loads(row[7]) if row[7] else {}
            }
        return None

    def get_version(self, circuit_name: str, version: str) -> Optional[Dict]:
        """Get a specific version of a circuit"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata
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
                'metadata': json.loads(row[7]) if row[7] else {}
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
        """Promote one exact sealed snapshot without copying its evidence."""
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
        """Clear the lock on one exact version UUID."""
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
        """Lock one exact version UUID against attached writes."""
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
        """Compatibility wrapper resolving an unambiguous human label."""
        circuit = self.get_version(circuit_name, version)
        if not circuit:
            raise ValueError(f"Circuit '{circuit_name}' version '{version}' not found")
        self.unlock_version(circuit["version_id"])

    def lock_circuit(self, circuit_name: str, version: str):
        """Compatibility wrapper resolving an unambiguous human label."""
        circuit = self.get_version(circuit_name, version)
        if not circuit:
            raise ValueError(f"Circuit '{circuit_name}' version '{version}' not found")
        self.lock_version(circuit["version_id"])

    def get_circuit_by_id(self, version_id: str) -> Optional[Dict]:
        """
        Get circuit information by version_id

        Args:
            version_id: Unique version ID

        Returns:
            dict: Circuit info or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata
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
                'metadata': json.loads(row[7]) if row[7] else {}
            }
        return None

    def get_circuit_by_version_id(self, version_id: str) -> Optional[Dict]:
        """
        Get circuit by its unique version_id (alias for get_circuit_by_id)

        Args:
            version_id: Unique version ID

        Returns:
            dict: Circuit info or None if not found
        """
        return self.get_circuit_by_id(version_id)

    def get_all_circuit_versions(self, circuit_name: str, order_by: str = 'created_at') -> List[Dict]:
        """
        Get all versions of a circuit

        Args:
            circuit_name: Name of the circuit
            order_by: Column to order by ('created_at', 'version', 'stage')

        Returns:
            list: List of circuit info dicts
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
                   netlist_path, description, metadata
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
                'metadata': json.loads(row[7]) if row[7] else {}
            })

        return circuits

    def get_most_simulated_circuit(self, circuit_name: str) -> Optional[Dict]:
        """
        Get the circuit version with the highest run count

        Args:
            circuit_name: Name of the circuit

        Returns:
            dict: Circuit info with highest run_count or None if not found
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
        """Create a release snapshot of circuits at a stage"""
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
        """List all circuits, optionally filtered by stage"""
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
        """Get a complete summary of a circuit including components and results"""
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
            SELECT name, net_type
            FROM nets
            WHERE circuit_version_id = ?
        """, (circuit['version_id'],))
        nets = [dict(row) for row in cursor.fetchall()]

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

    def _format_simulation_summary(self, circuit_version_id: str) -> str:
        """Render a summary without mutating the snapshot or its evidence."""
        cursor = self.conn.cursor()

        # Get all unique parameters
        cursor.execute("""
            SELECT DISTINCT parameter, unit
            FROM simulation_results
            WHERE circuit_version_id = ?
            ORDER BY parameter
        """, (circuit_version_id,))

        parameters = cursor.fetchall()

        if not parameters:
            return ""

        summary_lines = []
        summary_lines.append("=== SIMULATION SUMMARY ===")
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")

        for param, unit in parameters:
            # Get every observation for this parameter across corners. The
            # results table is append-only — re-running the same (corner,
            # temperature, voltage) leaves multiple rows — so group by that
            # triple and keep only the latest observation per group (by
            # timestamp, then rowid as the tiebreak for same-timestamp
            # writes) for both the printed line and the header's value range.
            cursor.execute("""
                SELECT corner, temperature, voltage, value, sim_type, timestamp, rowid
                FROM simulation_results
                WHERE circuit_version_id = ? AND parameter = ?
                ORDER BY
                    CASE corner
                        WHEN 'tt' THEN 1
                        WHEN 'ss' THEN 2
                        WHEN 'ff' THEN 3
                        WHEN 'sf' THEN 4
                        WHEN 'fs' THEN 5
                        ELSE 6
                    END,
                    temperature,
                    voltage,
                    timestamp,
                    rowid
            """, (circuit_version_id, param))

            rows = cursor.fetchall()

            if not rows:
                continue

            unit_str = f" {unit}" if unit else ""

            # Group consecutive rows sharing (corner, temperature, voltage)
            # — safe because the ORDER BY above sorts on that triple first,
            # so every group's rows are contiguous. Each group's LAST row is
            # its latest observation (timestamp, then rowid ascending).
            groups = []
            for row in rows:
                key = (row[0], row[1], row[2])
                if groups and groups[-1][0] == key:
                    groups[-1][1].append(row)
                else:
                    groups.append((key, [row]))

            n_groups = len(groups)
            corner_word = "corner" if n_groups == 1 else "corners"

            # If multiple distinct corners, show range
            if n_groups > 1:
                latest_values = [obs[-1][3] for _key, obs in groups if obs[-1][3] is not None]
                if latest_values:
                    min_val = min(latest_values)
                    max_val = max(latest_values)
                    summary_lines.append(f"{param:20s}: {min_val:.3g} to {max_val:.3g}{unit_str} ({n_groups} {corner_word})")

                    # Show each distinct corner's latest observation
                    for (corner, temp, volt), obs in groups:
                        _c, _t, _v, value, sim_type, ts, rid = obs[-1]
                        corner_str = corner or "nominal"
                        conditions = []
                        if temp is not None:
                            conditions.append(f"{temp}°C")
                        if volt is not None:
                            conditions.append(f"{volt}V")
                        cond_str = f" @ {', '.join(conditions)}" if conditions else ""
                        latest_suffix = f" (latest of {len(obs)})" if len(obs) > 1 else ""
                        summary_lines.append(f"  └─ {corner_str:8s}{cond_str:20s}: {value:.3g}{unit_str}{latest_suffix}")
            else:
                # Single distinct corner (possibly multiple superseded observations)
                (corner, temp, volt), obs = groups[0]
                _c, _t, _v, value, sim_type, ts, rid = obs[-1]
                if value is not None:
                    corner_str = corner or "nominal"
                    conditions = []
                    if temp is not None:
                        conditions.append(f"{temp}°C")
                    if volt is not None:
                        conditions.append(f"{volt}V")
                    cond_str = f" @ {', '.join(conditions)}" if conditions else ""
                    latest_suffix = f" (latest of {len(obs)})" if len(obs) > 1 else ""
                    summary_lines.append(f"{param:20s}: {value:.3g}{unit_str} ({corner_str}{cond_str}, {n_groups} {corner_word}){latest_suffix}")

            summary_lines.append("")

        # Get pass/fail status if available
        cursor.execute("""
            SELECT parameter, value_text
            FROM simulation_results
            WHERE circuit_version_id = ? AND value_text IS NOT NULL
        """, (circuit_version_id,))

        status_results = cursor.fetchall()
        if status_results:
            summary_lines.append("=== STATUS ===")
            for param, status in status_results:
                summary_lines.append(f"{param:20s}: {status}")
            summary_lines.append("")

        summary_text = "\n".join(summary_lines)

        return summary_text

    def update_simulation_summary(self, circuit_version_id: str) -> str:
        """Generate and cache a formatted summary on an unlocked snapshot."""
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
        """Get the cached simulation summary for a circuit"""
        snapshot = self._snapshot_row(circuit_version_id)
        row = self.conn.execute("""
            SELECT simulation_summary
            FROM circuit_versions
            WHERE version_id = ?
        """, (circuit_version_id,)).fetchone()
        if row and row[0]:
            return row[0]
        if snapshot["is_locked"]:
            return self._format_simulation_summary(circuit_version_id)
        return self.update_simulation_summary(circuit_version_id)

    def add_parasitic(self, circuit_version_id: str, net_id: str,
                     parasitic_type: str, value: float, unit: str,
                     from_node: str = None, to_node: str = None,
                     extraction_tool: str = None) -> str:
        """Add parasitic data (for post-layout)"""
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
        """
        Get the most recently created circuit version by circuit name
        (Alias for get_latest for backward compatibility)
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
        """
        Get the circuit with the highest version number

        Args:
            circuit_name: Name of the circuit

        Returns:
            dict: Circuit info or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT version_id, circuit_name, version, stage, created_at,
                   netlist_path, description, metadata
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
                'metadata': json.loads(row[7]) if row[7] else {}
            }
        return None

    def get_all_circuits_at_stage(self, stage: DesignStage) -> List[Dict]:
        """Get all circuits at a specific stage (latest version of each)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT circuit_name FROM circuit_versions
            WHERE stage = ?
        """, (stage.value,))

        circuit_names = [row[0] for row in cursor.fetchall()]
        return [self.get_latest(name, stage) for name in circuit_names]

    def get_corners_for_parameter(self, circuit_version_id: str,
                                  parameter: str,
                                  latest_only: bool = False) -> List[Dict]:
        """Get results for a specific parameter across all corners.

        The ``simulation_results`` table is append-only: re-running the same
        (corner, temperature, voltage) adds a new row rather than replacing
        the old one. This accessor has two modes:

        - ``latest_only=False`` (default): return every observation, in the
          order they were recorded — the full append-only truth, including
          any superseded re-runs.
        - ``latest_only=True``: collapse to one row per distinct (corner,
          temperature, voltage), keeping only the latest observation in
          each group (by ``timestamp``, then insertion order as the
          tiebreak for same-timestamp writes).
        """
        cursor = self.conn.cursor()
        if latest_only:
            cursor.execute("""
                SELECT corner, temperature, voltage, value, unit, timestamp
                FROM simulation_results
                WHERE circuit_version_id = ? AND parameter = ?
                AND rowid = (
                    SELECT inner_sr.rowid FROM simulation_results AS inner_sr
                    WHERE inner_sr.circuit_version_id = simulation_results.circuit_version_id
                      AND inner_sr.parameter = simulation_results.parameter
                      AND inner_sr.corner IS simulation_results.corner
                      AND inner_sr.temperature IS simulation_results.temperature
                      AND inner_sr.voltage IS simulation_results.voltage
                    ORDER BY inner_sr.timestamp DESC, inner_sr.rowid DESC
                    LIMIT 1
                )
                ORDER BY corner, temperature, voltage
            """, (circuit_version_id, parameter))
        else:
            cursor.execute("""
                SELECT corner, temperature, voltage, value, unit, timestamp
                FROM simulation_results
                WHERE circuit_version_id = ? AND parameter = ?
                ORDER BY corner, temperature, voltage
            """, (circuit_version_id, parameter))

        return [dict(row) for row in cursor.fetchall()]

    def get_corner_summary(self, circuit_version_id: str) -> Dict:
        """Get a summary of all corners simulated"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT DISTINCT corner, temperature, voltage
            FROM simulation_results
            WHERE circuit_version_id = ?
            ORDER BY corner, temperature, voltage
        """, (circuit_version_id,))

        corners = {}
        for row in cursor.fetchall():
            corner = row[0] or "nominal"
            temp = row[1]
            volt = row[2]

            key = corner
            if temp is not None or volt is not None:
                conditions = []
                if temp is not None:
                    conditions.append(f"{temp}°C")
                if volt is not None:
                    conditions.append(f"{volt}V")
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
        """Get all nets for a circuit version."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT net_id, name, net_type FROM nets
            WHERE circuit_version_id = ?
        """, (circuit_version_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_parameters_for_circuit(self, circuit_version_id: str) -> List[str]:
        """Get distinct parameter names for a circuit version."""
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
                                voltage=None, sim_type=None, tool_name=None):
        """Insert one validated, context-bearing Spec observation."""
        declaration = self.conn.execute(
            """
            SELECT circuit_version_id, op, target
            FROM specs WHERE spec_id = ?
            """,
            (spec_id,),
        ).fetchone()
        if declaration is None:
            raise DatabaseError(f"Spec {spec_id!r} does not exist")
        self._require_sealed(declaration["circuit_version_id"])
        if source not in _SPEC_MEASUREMENT_SOURCES:
            raise ValueError(
                "Spec measurement source must be one of "
                f"{sorted(_SPEC_MEASUREMENT_SOURCES)}, got {source!r}"
            )
        value = _spec_measurement_real(value, field="value")
        temperature = _spec_measurement_real(
            temperature, field="temperature", optional=True
        )
        voltage = _spec_measurement_real(
            voltage, field="voltage", optional=True
        )
        timestamp = _spec_measurement_real(
            timestamp, field="timestamp", optional=True
        )
        for field, item in (
            ("corner", corner),
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
        self.conn.execute("""
            INSERT INTO spec_measurements (
                measurement_id, spec_id, source, value, corner, temperature,
                voltage, sim_type, tool_name, timestamp, sim_run_id, notes,
                passes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            measurement_id, spec_id, source, value, corner, temperature,
            voltage, sim_type, tool_name, timestamp, sim_run_id, notes,
            int(passes) if passes is not None else None,
        ))
        self._maybe_commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Example usage
    db = Database("example_updated.db")

    # Create a circuit in dev stage
    amp_id = db.create_circuit(
        name="amplifier",
        version="1.0",
        stage=DesignStage.DEV,
        created_by="designer1",
        description="Initial differential amplifier design"
    )

    # Add some devices
    db.add_device(amp_id, "M1", "nmos", model="sky130_fd_pr__nfet_01v8",
                  width=10e-6, length=180e-9)
    db.add_device(amp_id, "M2", "nmos", model="sky130_fd_pr__nfet_01v8",
                  width=10e-6, length=180e-9)
    db.add_device(amp_id, "M3", "pmos", model="sky130_fd_pr__pfet_01v8",
                  width=20e-6, length=180e-9)

    # Create a module with a hash
    module_hash = hashlib.sha256("output_stage_v1".encode()).hexdigest()
    module_id = db.insert_or_get_module(
        module_name="output_stage",
        module_path="amplifier.output",
        module_hash=module_hash,
        device_count=2,
        structure={'devices': ['M4', 'M5'], 'nets': ['out', 'gnd']}
    )

    # Link module to circuit
    db.link_module_to_circuit(amp_id, module_id, instance_name="output_1", position=0)

    # Add simulation results
    db.add_simulation_result(amp_id, "gain", value=45.2, unit="dB", sim_type="ac")
    db.add_simulation_result(amp_id, "bandwidth", value=1e9, unit="Hz", sim_type="ac")

    # Create version 2 with same module
    amp_id_v2 = db.create_circuit(
        name="amplifier",
        version="1.1",
        stage=DesignStage.DEV,
        created_by="designer1",
        description="Updated amplifier - same output stage"
    )

    # Reuse the same module (no duplication!)
    db.link_module_to_circuit(amp_id_v2, module_id, instance_name="output_1", position=0)

    print("\n" + "="*60)
    print("MODULE REUSE DEMONSTRATION")
    print("="*60)

    # Show modules for both circuits
    print(f"\nModules in version 1.0:")
    for m in db.get_circuit_modules(amp_id):
        print(f"  - {m['module_name']} (hash: {m['module_hash'][:16]}...)")

    print(f"\nModules in version 1.1:")
    for m in db.get_circuit_modules(amp_id_v2):
        print(f"  - {m['module_name']} (hash: {m['module_hash'][:16]}...)")

    print(f"\nCircuits using output_stage module:")
    for c in db.get_circuits_using_module(module_hash):
        print(f"  - {c['circuit_name']} v{c['version']} [{c['stage']}]")

    # Compare modules between versions
    comparison = db.compare_circuit_modules(amp_id, amp_id_v2)
    print(f"\nModule comparison:")
    print(f"  Unchanged: {len(comparison['unchanged'])} modules")
    print(f"  Added: {len(comparison['added'])} modules")
    print(f"  Removed: {len(comparison['removed'])} modules")

    db.close()
# Backward-compat alias (pre-silk name). Internal silk/db callers now use
# `Database` directly; retained for external code + silk._compat. Delete once
# no external consumer imports it.
CircuitDatabase = Database
