"""Typed, validated historical Spec declaration and measurement readback."""

from dataclasses import FrozenInstanceError, replace

import pytest

import silk.db as silk_db
from silk.db import (
    Database,
    DatabaseError,
    DesignStage,
    LockedVersionError,
)
from silk.db.identity import (
    _fingerprint_spec_set,
    _fingerprint_spec_set_records,
)
from silk.db.models import SnapshotIdentity
from silk.specs import Spec
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
    identity_for_structure,
)


class _ElementManager:
    def __init__(self):
        self.elem_dict = {}


class _SpecOwner:
    def __init__(self):
        self._specs = []
        self._element_mgr = _ElementManager()

    def add_instance(self, name, child):
        self._element_mgr.elem_dict[name] = {"type": "m", "device": child}


def _add_spec(
    owner,
    *,
    name,
    key=None,
    metric="gain",
    op=">=",
    target=40.0,
    units="dB",
    description="",
    parent=None,
    declared_at="design.py:10",
):
    spec = Spec(
        name=name,
        module=owner,
        metric=metric,
        op=op,
        target=target,
        units=units,
        description=description,
        parent=parent,
        declared_at=declared_at,
        key=key,
    )
    owner._specs.append(spec)
    return spec


def _owner_with_hierarchy():
    root = _SpecOwner()
    child = _SpecOwner()
    root.add_instance("core", child)
    alpha = _add_spec(
        root,
        name="Alpha gain",
        key="alpha",
        description="root target",
    )
    _add_spec(root, name="Zeta gain", key="zeta", target=45.0)
    _add_spec(
        child,
        name="Child gain",
        key="child",
        target=50.0,
        parent=alpha,
    )
    return root


def _identity_for_definitions(definitions):
    spec_set = _fingerprint_spec_set_records(
        (
            definition.owner_path,
            definition.key,
            definition.fingerprint,
        )
        for definition in definitions
    )
    return SnapshotIdentity(
        structure=identity_for_structure().structure,
        spec_set=spec_set,
    )


def _create_building_for_definitions(
    db, definitions, *, name="spec_history", identity=None
):
    return db.create_circuit(
        name,
        identity=identity or _identity_for_definitions(definitions),
    )


def _store_owner(db, owner, *, name="spec_history"):
    _spec_set, definitions = _fingerprint_spec_set(owner)
    version_id = _create_building_for_definitions(
        db, definitions, name=name
    )
    db.store_spec_declarations(version_id, definitions)
    db.seal_circuit_version(version_id)
    return version_id


def _spec_ids(db, version_id):
    return {
        row["spec_key"]: row["spec_id"]
        for row in db.conn.execute(
            "SELECT spec_id, spec_key FROM specs WHERE circuit_version_id = ?",
            (version_id,),
        )
    }


def _insert_measurement(
    db,
    spec_id,
    measurement_id,
    *,
    value,
    timestamp,
    source="spice",
    corner="tt",
    passes=True,
):
    db.insert_spec_measurement(
        measurement_id,
        spec_id,
        source,
        value,
        corner=corner,
        timestamp=timestamp,
        sim_run_id=f"run-{measurement_id}",
        notes=f"note-{measurement_id}",
        passes=passes,
    )


def test_historical_query_survives_reopen_and_is_deterministic(tmp_path):
    db_path = tmp_path / "spec-history.db"
    with Database(db_path) as db:
        version_id = _store_owner(db, _owner_with_hierarchy())
        alpha_id = _spec_ids(db, version_id)["alpha"]
        _insert_measurement(
            db, alpha_id, "late", value=42.0, timestamp=20.0
        )
        _insert_measurement(
            db, alpha_id, "early-a", value=41.0, timestamp=10.0
        )
        _insert_measurement(
            db, alpha_id, "early-b", value=41.5, timestamp=10.0
        )

    with Database(db_path) as reopened:
        records = reopened.get_stored_specs(version_id)

    assert [(record.owner_path, record.spec_key) for record in records] == [
        ("$", "alpha"),
        ("$", "zeta"),
        ("$/core", "child"),
    ]
    alpha = records[0]
    assert isinstance(alpha, silk_db.SpecDeclarationRecord)
    assert [m.measurement_id for m in alpha.measurements] == [
        "early-a",
        "early-b",
        "late",
    ]
    assert all(
        isinstance(m, silk_db.SpecMeasurementRecord)
        for m in alpha.measurements
    )
    early = alpha.measurements[0]
    assert early.origin_version_id == version_id
    assert early.origin_spec_id == alpha.spec_id
    assert early.source == "spice"
    assert early.value == 41.0
    assert early.corner == "tt"
    assert early.timestamp == 10.0
    assert early.sim_run_id == "run-early-a"
    assert early.notes == "note-early-a"
    assert early.passes is True
    with pytest.raises(FrozenInstanceError):
        early.value = 0.0


def test_promotion_remaps_parent_ids_and_preserves_measurement_origins():
    with Database(":memory:") as db:
        source_id = _store_owner(db, _owner_with_hierarchy())
        source_ids = _spec_ids(db, source_id)
        _insert_measurement(
            db, source_ids["alpha"], "source", value=42.0, timestamp=1.0
        )
        target_id = db.promote_circuit(
            source_id, DesignStage.STABLE, lock=False
        )
        target_ids = _spec_ids(db, target_id)
        _insert_measurement(
            db, target_ids["alpha"], "target", value=43.0, timestamp=2.0
        )

        inherited = db.get_stored_specs(target_id)
        local = db.get_stored_specs(
            target_id, include_ancestors=False
        )
        declarations_only = db.get_stored_specs(
            target_id, include_measurements=False
        )

    by_key = {record.spec_key: record for record in inherited}
    assert all(
        record.declaration_version_id == target_id for record in inherited
    )
    assert by_key["child"].parent_id == target_ids["alpha"]
    assert by_key["child"].parent_id != source_ids["alpha"]
    assert [
        measurement.origin_version_id
        for measurement in by_key["alpha"].measurements
    ] == [source_id, target_id]
    local_alpha = next(record for record in local if record.spec_key == "alpha")
    assert [m.measurement_id for m in local_alpha.measurements] == ["target"]
    assert all(record.measurements == () for record in declarations_only)


def test_empty_historical_query_is_nonmutating_and_building_is_rejected():
    with Database(":memory:") as db:
        sealed_id = create_sealed_snapshot(db)
        building_id = create_building_snapshot(
            db,
            version="1.1",
            parent_version_id=sealed_id,
        )

        assert db.get_stored_specs(sealed_id) == []
        with pytest.raises(DatabaseError, match="sealed"):
            db.get_stored_specs(building_id)


def test_duplicate_owner_key_rolls_back_complete_declaration_set():
    owner = _SpecOwner()
    _add_spec(owner, name="First", key="first")
    _add_spec(owner, name="Second", key="second")
    _spec_set, definitions = _fingerprint_spec_set(owner)
    duplicate = (
        definitions[0],
        replace(
            definitions[1],
            owner_path=definitions[0].owner_path,
            key=definitions[0].key,
        ),
    )
    with Database(":memory:") as db:
        version_id = _create_building_for_definitions(
            db,
            duplicate,
            identity=SnapshotIdentity(
                identity_for_structure().structure,
                _fingerprint_spec_set_records(()),
            ),
        )

        with pytest.raises(DatabaseError, match="duplicate|owner|key"):
            db.store_spec_declarations(version_id, duplicate)

        assert db.conn.execute("SELECT COUNT(*) FROM specs").fetchone()[0] == 0


def test_unknown_parent_rolls_back_complete_declaration_set():
    owner = _SpecOwner()
    parent = _add_spec(owner, name="Parent", key="parent")
    _add_spec(owner, name="Child", key="child", parent=parent)
    _spec_set, definitions = _fingerprint_spec_set(owner)
    child_only = tuple(
        definition for definition in definitions if definition.key == "child"
    )
    with Database(":memory:") as db:
        version_id = _create_building_for_definitions(db, child_only)

        with pytest.raises(DatabaseError, match="parent|unresolved"):
            db.store_spec_declarations(version_id, child_only)

        assert db.conn.execute("SELECT COUNT(*) FROM specs").fetchone()[0] == 0


def test_parent_cycle_rolls_back_complete_declaration_set():
    owner = _SpecOwner()
    first = _add_spec(owner, name="First", key="first")
    second = _add_spec(owner, name="Second", key="second")
    first.parent = second
    second.parent = first
    _spec_set, definitions = _fingerprint_spec_set(owner)
    with Database(":memory:") as db:
        version_id = _create_building_for_definitions(db, definitions)

        with pytest.raises(DatabaseError, match="cycle"):
            db.store_spec_declarations(version_id, definitions)

        assert db.conn.execute("SELECT COUNT(*) FROM specs").fetchone()[0] == 0


def test_inconsistent_definition_hash_rolls_back_complete_set():
    owner = _SpecOwner()
    spec = _add_spec(owner, name="Gain", key="gain")
    _spec_set, definitions = _fingerprint_spec_set(owner)
    with Database(":memory:") as db:
        version_id = _create_building_for_definitions(db, definitions)
        # Bypass the public assignment guard to exercise the deeper hash check.
        object.__setattr__(spec, "target", 99.0)

        with pytest.raises(DatabaseError, match="definition|fingerprint|hash"):
            db.store_spec_declarations(version_id, definitions)

        assert db.conn.execute("SELECT COUNT(*) FROM specs").fetchone()[0] == 0


def test_sealed_and_locked_versions_reject_declaration_writes():
    owner = _SpecOwner()
    _add_spec(owner, name="Gain", key="gain")
    _spec_set, definitions = _fingerprint_spec_set(owner)
    with Database(":memory:") as db:
        sealed_id = _create_building_for_definitions(
            db, definitions, name="sealed"
        )
        db.store_spec_declarations(sealed_id, definitions)
        db.seal_circuit_version(sealed_id)
        with pytest.raises(DatabaseError, match="building"):
            db.store_spec_declarations(sealed_id, definitions)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM specs WHERE circuit_version_id = ?",
            (sealed_id,),
        ).fetchone()[0] == len(definitions)

        locked_id = _create_building_for_definitions(
            db, definitions, name="locked"
        )
        db.lock_version(locked_id)
        with pytest.raises(LockedVersionError):
            db.store_spec_declarations(locked_id, definitions)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM specs WHERE circuit_version_id = ?",
            (locked_id,),
        ).fetchone()[0] == 0
