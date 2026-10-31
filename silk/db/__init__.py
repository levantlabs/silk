"""Design database, version tracking and migration.

:class:`Database` stores circuit snapshots (identified by a
:class:`SnapshotIdentity`) with their metrics, spec measurements,
characterization profiles and net-current sets in SQLite.
:class:`CircuitTracker` binds a database to a live
:class:`~silk.circuit.Module`; :class:`CircuitImporter` and
:class:`CircuitExporter` move circuits in and out as JSON, CSV and SPICE
text. The ``migrate_v*`` functions copy an older database file to the next
schema version. Value objects returned by reads (:class:`SimulationResultRecord`
and its siblings) are immutable dataclasses, and failures raise the typed
errors re-exported here (rooted at :class:`DatabaseError`).
"""
from silk.db.circuit_db import Database, DesignStage
from silk.db.migration import (
    migrate_v1_to_v2,
    migrate_v2_to_v3,
    migrate_v3_to_v4,
    migrate_v4_to_v5,
    migrate_v5_to_v6,
    migrate_v6_to_v7,
)
from silk.db.tracking import CircuitTracker
from silk.db.circuit_import import CircuitImporter, CircuitExporter
from silk.db.models import (
    DeckCheck,
    CharacterizationEvidenceRecord,
    CharacterizationProfileRecord,
    CharacterizationRestoreResult,
    Fingerprint,
    NetCurrentRecord,
    NetCurrentSetRecord,
    SimulationResultRecord,
    SnapshotIdentity,
    SpecDeclarationRecord,
    SpecMeasurementRecord,
    UNSCOPED_PROFILE,
)
from silk.exceptions import (
    DatabaseCompatibilityError,
    DatabaseError,
    EvidenceError,
    EvidenceAmbiguityError,
    EvidenceMismatchError,
    EvidenceNotFoundError,
    LockedVersionError,
    ReportStorageError,
    UnsupportedFingerprintSchemeError,
    UnsupportedSchemaVersionError,
)

__all__ = [
    "DeckCheck",
    "Database",
    "migrate_v1_to_v2",
    "migrate_v2_to_v3",
    "migrate_v3_to_v4",
    "migrate_v4_to_v5",
    "migrate_v5_to_v6",
    "migrate_v6_to_v7",
    "CircuitTracker",
    "CircuitImporter",
    "CircuitExporter",
    "DesignStage",
    "CharacterizationProfileRecord",
    "CharacterizationEvidenceRecord",
    "CharacterizationRestoreResult",
    "Fingerprint",
    "NetCurrentRecord",
    "NetCurrentSetRecord",
    "SnapshotIdentity",
    "SimulationResultRecord",
    "SpecDeclarationRecord",
    "SpecMeasurementRecord",
    "UNSCOPED_PROFILE",
    "DatabaseError",
    "DatabaseCompatibilityError",
    "UnsupportedSchemaVersionError",
    "UnsupportedFingerprintSchemeError",
    "EvidenceError",
    "EvidenceAmbiguityError",
    "EvidenceNotFoundError",
    "EvidenceMismatchError",
    "LockedVersionError",
    "ReportStorageError",
]
