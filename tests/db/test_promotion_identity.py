"""Explicit promotion preserves snapshot identity and evidence provenance."""

import inspect
import uuid

import pytest

from silk.db import (
    Database,
    DatabaseError,
    DesignStage,
    LockedVersionError,
)
from silk.db.identity import (
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprint_module_structure,
    _fingerprint_spec_set,
    _fingerprint_structure,
)
from silk.db.models import SnapshotIdentity
from silk.db.structure_tracker import CircuitStructureTracker
from silk.specs import Spec


class _SpecOwner:
    def __init__(self):
        self._specs = []
        self._element_mgr = type("ElementManager", (), {"elem_dict": {}})()


def _spec_definitions(target):
    owner = _SpecOwner()
    parent = Spec(
        "system gain",
        owner,
        "gain",
        ">=",
        target,
        units="dB",
        key="system_gain",
    )
    child = Spec(
        "core gain",
        owner,
        "gain",
        ">=",
        target + 1.0,
        units="dB",
        parent=parent,
        key="core_gain",
    )
    owner._specs.extend((parent, child))
    return _fingerprint_spec_set(owner)


def _structure(tag):
    module_payload = {
        "devices": [
            {
                "name": f"inner_{tag}",
                "type": "nmos",
                "parameters": {"w": 1.0},
                "nets": {"d": f"out_{tag}"},
            }
        ],
        "nets": [f"out_{tag}"],
        "submodules": {},
        "passives": [],
    }
    module_hash = _fingerprint_module_structure(module_payload).digest
    return {
        "transistors": [
            {
                "name": f"m_{tag}",
                "type": "nmos",
                "model": "nfet",
                "parameters": {"w": float(len(tag) + 1)},
            }
        ],
        "nets": [{"name": f"out_{tag}", "type": "signal"}],
        "hierarchy": [],
        "parameters": [],
        "passives": [],
        "modules": {
            "core": {
                **module_payload,
                "hash": module_hash,
                "device_count": 1,
            }
        },
    }


def _insert_current_evidence(db, version_id, net_name, structure_hash):
    # Schema v2: set-level context (corner/structure identity) lives on the
    # net_current_sets parent; net_currents keeps only per-row identity/values.
    current_set_id = str(uuid.uuid4())
    db.conn.execute(
        """
        INSERT INTO net_current_sets
            (current_set_id, circuit_version_id, profile_id, state,
             corner, structure_scheme, structure_hash,
             corner_state, temperature_state, voltage_state)
        VALUES (?, ?, NULL, 'current', 'tt', ?, ?, 'fixed', 'unknown', 'unknown')
        """,
        (current_set_id, version_id, STRUCTURE_FINGERPRINT_SCHEME, structure_hash),
    )
    db.conn.execute(
        """
        INSERT INTO net_currents
            (current_id, current_set_id, net_name, pin_id, avg_a, rms_a,
             peak_a, source)
        VALUES (?, ?, ?, '', ?, ?, ?, 'spice')
        """,
        (str(uuid.uuid4()), current_set_id, net_name, 1e-3, 1e-3, 2e-3),
    )
    db.conn.commit()


def _create_source(
    db,
    *,
    tag="one",
    label="draft",
    target=40.0,
    evidence=False,
):
    structure = _structure(tag)
    structure_fingerprint, _module_fingerprints = _fingerprint_structure(
        structure
    )
    spec_set, definitions = _spec_definitions(target)
    identity = SnapshotIdentity(structure_fingerprint, spec_set)
    version_id = db.create_circuit(
        "amp",
        label,
        stage=DesignStage.DEV,
        identity=identity,
    )
    tracker = CircuitStructureTracker(db, version_id)
    structure_hash, module_hashes = tracker.compute_structure_hash(structure)
    tracker.store_structure(structure, structure_hash, module_hashes)
    db.store_spec_declarations(version_id, definitions)
    db.seal_circuit_version(version_id)

    net_id = db.conn.execute(
        "SELECT net_id FROM nets WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0]
    spec_id = db.conn.execute(
        """
        SELECT spec_id FROM specs
        WHERE circuit_version_id = ? AND spec_key = 'system_gain'
        """,
        (version_id,),
    ).fetchone()[0]
    if evidence:
        db.add_simulation_result(version_id, "gain", value=42.0, unit="dB")
        db.add_parasitic(
            version_id, net_id, "capacitance", 2e-15, "F"
        )
        db.insert_spec_measurement(
            str(uuid.uuid4()),
            spec_id,
            "spice",
            42.0,
            corner="tt",
            passes=True,
        )
        _insert_current_evidence(
            db, version_id, f"out_{tag}", identity.structure.digest
        )
    return {
        "version_id": version_id,
        "identity": identity,
        "definitions": definitions,
        "net_id": net_id,
        "spec_id": spec_id,
    }


def _count(db, table, version_id):
    if table == "net_currents":
        # Schema v2: circuit_version_id lives on the net_current_sets parent.
        return db.conn.execute(
            """
            SELECT COUNT(*) FROM net_currents AS nc
            JOIN net_current_sets AS ncs
                ON ncs.current_set_id = nc.current_set_id
            WHERE ncs.circuit_version_id = ?
            """,
            (version_id,),
        ).fetchone()[0]
    return db.conn.execute(
        f'SELECT COUNT(*) FROM "{table}" WHERE circuit_version_id = ?',
        (version_id,),
    ).fetchone()[0]


def test_promotion_api_requires_explicit_source_version_id():
    parameters = inspect.signature(Database.promote_circuit).parameters
    assert "source_version_id" in parameters
    assert "circuit_name" not in parameters
    assert "from_stage" not in parameters
    assert parameters["new_version"].kind is inspect.Parameter.KEYWORD_ONLY


def test_promotion_copies_identity_structure_and_specs_but_not_evidence():
    with Database(":memory:") as db:
        source = _create_source(db, evidence=True)
        target_id = db.promote_circuit(
            source["version_id"], DesignStage.STABLE, lock=False
        )

        source_row = db._snapshot_row(source["version_id"])
        target_row = db._snapshot_row(target_id)
        assert target_id != source["version_id"]
        assert target_row["parent_version_id"] == source["version_id"]
        assert target_row["state"] == "sealed"
        assert target_row["stage"] == "stable"
        assert (
            target_row["structure_scheme"],
            target_row["structure_hash"],
            target_row["spec_scheme"],
            target_row["spec_set_hash"],
        ) == (
            source_row["structure_scheme"],
            source_row["structure_hash"],
            source_row["spec_scheme"],
            source_row["spec_set_hash"],
        )

        for table in ("devices", "nets", "circuit_modules", "specs"):
            assert _count(db, table, target_id) == _count(
                db, table, source["version_id"]
            )
        child_parent = db.conn.execute(
            """
            SELECT parent.circuit_version_id
            FROM specs AS child
            JOIN specs AS parent ON parent.spec_id = child.parent_id
            WHERE child.circuit_version_id = ? AND child.spec_key = 'core_gain'
            """,
            (target_id,),
        ).fetchone()
        assert child_parent[0] == target_id

        for table in (
            "simulation_results",
            "net_currents",
            "parasitics",
        ):
            assert _count(db, table, target_id) == 0
        assert db.conn.execute(
            """
            SELECT COUNT(*)
            FROM spec_measurements AS sm
            JOIN specs AS s ON s.spec_id = sm.spec_id
            WHERE s.circuit_version_id = ?
            """,
            (target_id,),
        ).fetchone()[0] == 0


def test_promotion_lineage_finds_source_evidence_with_origin_uuid():
    with Database(":memory:") as db:
        source = _create_source(db, evidence=True)
        target_id = db.promote_circuit(
            source["version_id"], DesignStage.STABLE, lock=False
        )

        observations = db.get_simulation_results(
            target_id, parameter="gain"
        )
        assert len(observations) == 1
        assert observations[0].origin_version_id == source["version_id"]
        assert observations[0].value == 42.0


def test_exact_promotion_rerun_reuses_target():
    with Database(":memory:") as db:
        source = _create_source(db)
        first = db.promote_circuit(source["version_id"], DesignStage.STABLE)
        second = db.promote_circuit(source["version_id"], DesignStage.STABLE)

        assert second == first
        assert db.conn.execute(
            "SELECT COUNT(*) FROM circuit_versions WHERE stage = 'stable'"
        ).fetchone()[0] == 1


def test_duplicate_source_label_cannot_redirect_promotion():
    with Database(":memory:") as db:
        intended = _create_source(db, tag="one", label="draft")
        _create_source(db, tag="two", label="draft", target=41.0)

        target_id = db.promote_circuit(
            intended["version_id"], DesignStage.STABLE, lock=False
        )
        assert db._snapshot_row(target_id)["parent_version_id"] == intended[
            "version_id"
        ]


def test_target_label_collision_from_different_source_fails():
    with Database(":memory:") as db:
        first = _create_source(db, tag="one", label="draft")
        second = _create_source(db, tag="two", label="draft", target=41.0)
        db.promote_circuit(
            first["version_id"],
            DesignStage.STABLE,
            new_version="release",
            lock=False,
        )

        with pytest.raises(DatabaseError, match="label|source"):
            db.promote_circuit(
                second["version_id"],
                DesignStage.STABLE,
                new_version="release",
                lock=False,
            )


def test_promotion_rejects_unsealed_source():
    with Database(":memory:") as db:
        source = _create_source(db)
        identity = source["identity"]
        building_id = db.create_circuit(
            "amp", "building", identity=identity
        )
        with pytest.raises(DatabaseError, match="sealed"):
            db.promote_circuit(building_id, DesignStage.STABLE)


def test_failed_copy_rolls_back_target_and_partial_structure(monkeypatch):
    with Database(":memory:") as db:
        source = _create_source(db)
        source_device_count = db.conn.execute(
            "SELECT COUNT(*) FROM devices"
        ).fetchone()[0]

        def fail_specs(*_args, **_kwargs):
            raise RuntimeError("injected spec-copy failure")

        monkeypatch.setattr(
            db,
            "_copy_spec_declarations_for_promotion",
            fail_specs,
            raising=False,
        )
        with pytest.raises(RuntimeError, match="injected spec-copy failure"):
            db.promote_circuit(
                source["version_id"], DesignStage.STABLE, lock=False
            )

        assert db.conn.execute(
            "SELECT COUNT(*) FROM circuit_versions WHERE stage = 'stable'"
        ).fetchone()[0] == 0
        assert db.conn.execute(
            "SELECT COUNT(*) FROM devices"
        ).fetchone()[0] == source_device_count


@pytest.mark.parametrize(
    "writer_name",
    [
        "device",
        "net",
        "module_link",
        "result",
        "parasitic",
        "spec_declaration",
        "spec_measurement",
        "current",
        "run_metadata",
        "summary_cache",
    ],
)
def test_locked_version_rejects_every_attached_writer(writer_name):
    with Database(":memory:") as db:
        source = _create_source(db)
        target_id = db.promote_circuit(source["version_id"], DesignStage.STABLE)
        if writer_name == "summary_cache":
            db.unlock_version(target_id)
            db.add_simulation_result(target_id, "gain", value=43.0)
            db.lock_version(target_id)
        net_id = db.conn.execute(
            "SELECT net_id FROM nets WHERE circuit_version_id = ?",
            (target_id,),
        ).fetchone()[0]
        module_id = db.conn.execute(
            """
            SELECT module_id FROM circuit_modules
            WHERE circuit_version_id = ?
            """,
            (target_id,),
        ).fetchone()[0]
        spec_id = db.conn.execute(
            "SELECT spec_id FROM specs WHERE circuit_version_id = ? LIMIT 1",
            (target_id,),
        ).fetchone()[0]
        before = {
            table: db.conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in (
                "devices",
                "nets",
                "circuit_modules",
                "simulation_results",
                "parasitics",
                "specs",
                "spec_measurements",
                "net_currents",
            )
        }
        writers = {
            "device": lambda: db.add_device(target_id, "late_m", "nmos"),
            "net": lambda: db.add_net(target_id, "late_net"),
            "module_link": lambda: db.link_module_to_circuit(
                target_id, module_id, "late_core"
            ),
            "result": lambda: db.add_simulation_result(
                target_id, "gain", value=43.0
            ),
            "parasitic": lambda: db.add_parasitic(
                target_id, net_id, "capacitance", 1e-15, "F"
            ),
            "spec_declaration": lambda: db.store_spec_declarations(
                target_id, source["definitions"]
            ),
            "spec_measurement": lambda: db.insert_spec_measurement(
                str(uuid.uuid4()), spec_id, "spice", 43.0
            ),
            "current": lambda: db.replace_net_current_set(
                target_id,
                corner="tt",
                structure_scheme=STRUCTURE_FINGERPRINT_SCHEME,
                structure_hash=source["identity"].structure.digest,
                rows=[
                    {
                        "net_name": "out_one",
                        "pin_id": "",
                        "avg_a": 1e-3,
                        "rms_a": 1e-3,
                        "peak_a": 2e-3,
                        "source": "spice",
                        "current_vector": None,
                    }
                ],
            ),
            "run_metadata": lambda: CircuitStructureTracker(
                db, target_id
            ).increment_run_count(),
            "summary_cache": lambda: db.update_simulation_summary(target_id),
        }
        before_version = db.conn.execute(
            """
            SELECT metadata, simulation_summary, last_summary_update
            FROM circuit_versions WHERE version_id = ?
            """,
            (target_id,),
        ).fetchone()

        with pytest.raises(LockedVersionError):
            writers[writer_name]()
        after = {
            table: db.conn.execute(
                f'SELECT COUNT(*) FROM "{table}"'
            ).fetchone()[0]
            for table in before
        }
        assert after == before
        after_version = db.conn.execute(
            """
            SELECT metadata, simulation_summary, last_summary_update
            FROM circuit_versions WHERE version_id = ?
            """,
            (target_id,),
        ).fetchone()
        assert tuple(after_version) == tuple(before_version)


def test_id_locking_and_ambiguous_label_wrapper():
    with Database(":memory:") as db:
        first = _create_source(db, tag="one", label="draft")
        second = _create_source(db, tag="two", label="draft", target=41.0)

        db.lock_version(first["version_id"])
        assert db._snapshot_row(first["version_id"])["is_locked"] == 1
        db.unlock_version(first["version_id"])
        assert db._snapshot_row(first["version_id"])["is_locked"] == 0
        with pytest.raises(DatabaseError, match="ambiguous"):
            db.lock_circuit("amp", "draft")
        assert db._snapshot_row(second["version_id"])["is_locked"] == 0


def test_locked_summary_read_renders_without_mutating_cache():
    with Database(":memory:") as db:
        source = _create_source(db, evidence=True)
        version_id = source["version_id"]
        db.lock_version(version_id)
        before = db.conn.execute(
            """
            SELECT simulation_summary, last_summary_update
            FROM circuit_versions WHERE version_id = ?
            """,
            (version_id,),
        ).fetchone()

        summary = db.get_simulation_summary(version_id)

        assert "gain" in summary
        after = db.conn.execute(
            """
            SELECT simulation_summary, last_summary_update
            FROM circuit_versions WHERE version_id = ?
            """,
            (version_id,),
        ).fetchone()
        assert tuple(after) == tuple(before)
