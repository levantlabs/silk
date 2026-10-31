"""Test-only builders for low-level database snapshot fixtures."""

from silk.db.circuit_db import DesignStage
from silk.db.identity import _fingerprint_spec_set_records, _fingerprint_structure
from silk.db.models import SnapshotIdentity


def empty_structure():
    return {
        "transistors": [],
        "nets": [],
        "hierarchy": [],
        "parameters": [],
        "passives": [],
        "modules": {},
    }


def structure_with_net(name):
    structure = empty_structure()
    structure["nets"] = [{"name": name, "type": "signal"}]
    return structure


def identity_for_structure(structure=None):
    structure_fingerprint, _module_fingerprints = _fingerprint_structure(
        empty_structure() if structure is None else structure
    )
    return SnapshotIdentity(
        structure=structure_fingerprint,
        spec_set=_fingerprint_spec_set_records(()),
    )


def create_building_snapshot(
    db,
    name="amp",
    version="1.0",
    stage=DesignStage.DEV,
    *,
    structure=None,
    **kwargs,
):
    return db.create_circuit(
        name=name,
        version=version,
        stage=stage,
        identity=identity_for_structure(structure),
        **kwargs,
    )


def create_sealed_snapshot(db, *args, **kwargs):
    structure = kwargs.get("structure")
    payload = empty_structure() if structure is None else structure
    version_id = create_building_snapshot(db, *args, **kwargs)
    from silk.db.structure_tracker import CircuitStructureTracker

    tracker = CircuitStructureTracker(db, version_id)
    structure_hash, module_hashes = tracker.compute_structure_hash(payload)
    tracker.store_structure(payload, structure_hash, module_hashes)
    db.store_spec_declarations(version_id, ())
    db.seal_circuit_version(version_id)
    return version_id
