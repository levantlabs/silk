"""Immutable snapshot lifecycle and deterministic identity resolution."""

import pytest

from silk.circuit.modules import Module
from silk.db import (
    Database,
    DatabaseError,
    DesignStage,
    UnsupportedFingerprintSchemeError,
)
from silk.db.identity import (
    SPEC_FINGERPRINT_SCHEME,
    STRUCTURE_FINGERPRINT_SCHEME,
)
from silk.db.models import Fingerprint, SnapshotIdentity
from silk.elements import elements


EMPTY_SPEC_SET_DIGEST = (
    "fa8829046c5e4c01929182f550e6445502dc868b21d0fe28076b8b9094fb61b3"
)


def _identity(structure_digest="a" * 64):
    return SnapshotIdentity(
        structure=Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, structure_digest),
        spec_set=Fingerprint(SPEC_FINGERPRINT_SCHEME, EMPTY_SPEC_SET_DIGEST),
    )


def _build(
    pdk,
    tmp_path,
    db_path,
    *,
    circuit_name="snapshot_amp",
    version="draft",
    resistance=1_000.0,
    spec_target=40.0,
):
    output = tmp_path / f"{circuit_name}-{resistance:g}-{spec_target:g}"
    module = Module(
        "PDK_skywater130",
        str(output),
        "snapshot.cir",
        pdk_details=pdk,
        module_name=circuit_name,
        db_path=str(db_path),
        version=version,
    )
    module.add(
        elements.resistor("r_load", p="out", n="0", value=resistance)
    )
    module.declare_spec(
        "DC gain",
        "dc_gain",
        ">=",
        spec_target,
        units="dB",
        key="dc_gain",
    )
    return module


def _row(db, version_id):
    return db.conn.execute(
        "SELECT * FROM circuit_versions WHERE version_id = ?",
        (version_id,),
    ).fetchone()


def test_snapshot_schema_v1_has_b2_identity_columns_and_indexes():
    with Database(":memory:") as db:
        expected_columns = {
            "circuit_versions": {
                "state",
                "structure_scheme",
                "structure_hash",
                "spec_scheme",
                "spec_set_hash",
            },
            "modules": {"module_hash_scheme"},
            "specs": {
                "circuit_version_id",
                "owner_path",
                "spec_key",
                "definition_scheme",
                "definition_hash",
            },
        }
        for table, expected in expected_columns.items():
            actual = {
                row[1]
                for row in db.conn.execute(f'PRAGMA table_info("{table}")')
            }
            assert expected <= actual

        indexes = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        assert {
            "idx_circuit_sealed_identity",
            "idx_circuit_parent",
            "idx_specs_owner_key",
            "idx_specs_definition",
        } <= indexes


def test_module_write_rejects_unsupported_fingerprint_scheme():
    with Database(":memory:") as db:
        with pytest.raises(UnsupportedFingerprintSchemeError):
            db.insert_or_get_module(
                "core",
                "$.core",
                "abc",
                module_hash_scheme="silk.structure/v999",
            )


def test_identical_rebuild_reuses_sealed_snapshot_after_reopen(
    pdk, tmp_path,
):
    db_path = tmp_path / "reuse.db"
    first = _build(pdk, tmp_path, db_path)
    first_identity = first._ensure_tracker().compute_snapshot_identity_for(first)
    first_id = first.create_circuit_version()
    first.close()

    rebuilt = _build(pdk, tmp_path, db_path)
    rebuilt_identity = rebuilt._ensure_tracker().compute_snapshot_identity_for(rebuilt)
    rebuilt_id = rebuilt.create_circuit_version()
    rebuilt.close()

    assert rebuilt_identity == first_identity
    assert rebuilt_id == first_id
    with Database(db_path) as db:
        assert _row(db, first_id)["state"] == "sealed"
        assert db.find_snapshot(
            "snapshot_amp", DesignStage.DEV, first_identity
        ) == first_id
        assert db.conn.execute(
            "SELECT COUNT(*) FROM circuit_versions"
        ).fetchone()[0] == 1


def test_device_edit_creates_child_with_new_structure_identity(pdk, tmp_path):
    db_path = tmp_path / "device-edit.db"
    first = _build(pdk, tmp_path, db_path, resistance=1_000.0)
    first_id = first.create_circuit_version()
    first.close()

    changed = _build(pdk, tmp_path, db_path, resistance=2_000.0)
    changed_id = changed.create_circuit_version()
    changed.close()

    with Database(db_path) as db:
        first_row = _row(db, first_id)
        changed_row = _row(db, changed_id)
        assert changed_id != first_id
        assert changed_row["parent_version_id"] == first_id
        assert changed_row["structure_hash"] != first_row["structure_hash"]
        assert changed_row["spec_set_hash"] == first_row["spec_set_hash"]


def test_spec_target_edit_keeps_structure_and_changes_spec_set(pdk, tmp_path):
    db_path = tmp_path / "spec-edit.db"
    first = _build(pdk, tmp_path, db_path, spec_target=40.0)
    first_id = first.create_circuit_version()
    first.close()

    changed = _build(pdk, tmp_path, db_path, spec_target=60.0)
    changed_id = changed.create_circuit_version()
    changed.close()

    with Database(db_path) as db:
        first_row = _row(db, first_id)
        changed_row = _row(db, changed_id)
        assert changed_row["structure_hash"] == first_row["structure_hash"]
        assert changed_row["spec_set_hash"] != first_row["spec_set_hash"]
        assert changed_row["parent_version_id"] == first_id


def test_building_version_is_not_resolvable_as_evidence():
    identity = _identity()
    with Database(":memory:") as db:
        version_id = db.create_circuit("amp", identity=identity)

        assert _row(db, version_id)["state"] == "building"
        assert db.find_snapshot("amp", DesignStage.DEV, identity) is None


def test_evidence_cannot_be_written_before_seal():
    with Database(":memory:") as db:
        version_id = db.create_circuit("amp", identity=_identity())

        with pytest.raises(DatabaseError, match="sealed"):
            db.add_simulation_result(version_id, "gain", value=40.0)

        db.seal_circuit_version(version_id)
        db.add_simulation_result(version_id, "gain", value=40.0)


def test_parasitic_evidence_cannot_be_written_before_seal():
    with Database(":memory:") as db:
        version_id = db.create_circuit("amp", identity=_identity())
        net_id = db.add_net(version_id, "out", "signal")

        with pytest.raises(DatabaseError, match="sealed"):
            db.add_parasitic(version_id, net_id, "capacitance", 1e-15, "F")

        db.seal_circuit_version(version_id)
        db.add_parasitic(version_id, net_id, "capacitance", 1e-15, "F")


def test_structure_cannot_be_changed_after_seal():
    with Database(":memory:") as db:
        version_id = db.create_circuit("amp", identity=_identity())
        db.add_net(version_id, "out", "signal")
        db.seal_circuit_version(version_id)

        with pytest.raises(DatabaseError, match="building"):
            db.add_net(version_id, "late", "signal")


def test_create_rolls_back_row_structure_and_specs_on_failure(
    pdk, tmp_path, monkeypatch,
):
    db_path = tmp_path / "rollback.db"
    module = _build(pdk, tmp_path, db_path)
    tracker = module._ensure_tracker()

    def fail_seal(_version_id):
        raise RuntimeError("injected seal failure")

    monkeypatch.setattr(tracker.db, "seal_circuit_version", fail_seal)
    with pytest.raises(RuntimeError, match="injected seal failure"):
        tracker.ensure_version(module)

    for table in ("circuit_versions", "devices", "nets", "circuit_modules", "specs"):
        assert tracker.db.conn.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0] == 0
    module.close()


def test_duplicate_human_label_does_not_affect_identity_resolution(pdk, tmp_path):
    db_path = tmp_path / "labels.db"
    first = _build(pdk, tmp_path, db_path, resistance=1_000.0, version="draft")
    first_tracker = first._ensure_tracker()
    first_identity = first_tracker.compute_snapshot_identity_for(first)
    first_id = first.create_circuit_version()
    first.close()

    second = _build(pdk, tmp_path, db_path, resistance=2_000.0, version="draft")
    second_tracker = second._ensure_tracker()
    second_identity = second_tracker.compute_snapshot_identity_for(second)
    second_id = second.create_circuit_version()
    second.close()

    with Database(db_path) as db:
        assert first_id != second_id
        assert db.find_snapshot(
            "snapshot_amp", DesignStage.DEV, first_identity
        ) == first_id
        assert db.find_snapshot(
            "snapshot_amp", DesignStage.DEV, second_identity
        ) == second_id


def test_ambiguous_label_lookup_fails_loudly(pdk, tmp_path):
    db_path = tmp_path / "ambiguous.db"
    first = _build(pdk, tmp_path, db_path, resistance=1_000.0, version="draft")
    first.create_circuit_version()
    first.close()
    second = _build(pdk, tmp_path, db_path, resistance=2_000.0, version="draft")
    second.create_circuit_version()
    second.close()

    with Database(db_path) as db:
        with pytest.raises(DatabaseError, match="ambiguous"):
            db.get_version("snapshot_amp", "draft")


def test_explicit_parent_is_validated(pdk, tmp_path):
    db_path = tmp_path / "parent.db"
    other = _build(
        pdk, tmp_path, db_path, circuit_name="other_amp", resistance=500.0
    )
    other_id = other.create_circuit_version()
    other.close()

    child = _build(pdk, tmp_path, db_path, circuit_name="snapshot_amp")
    with pytest.raises(DatabaseError, match="same circuit"):
        child.create_circuit_version(parent_version_id=other_id)
    assert child._tracker.db.conn.execute(
        "SELECT COUNT(*) FROM circuit_versions WHERE circuit_name = 'snapshot_amp'"
    ).fetchone()[0] == 0
    child.close()


def test_explicit_valid_parent_is_reported_by_lineage(pdk, tmp_path):
    db_path = tmp_path / "lineage.db"
    parent = _build(pdk, tmp_path, db_path, resistance=1_000.0)
    parent_id = parent.create_circuit_version()
    parent.close()

    child = _build(pdk, tmp_path, db_path, resistance=2_000.0)
    tracker = child._ensure_tracker()
    child_id = child.create_circuit_version(parent_version_id=parent_id)

    assert tracker.db.iter_version_lineage(child_id) == [child_id, parent_id]
    child.close()
