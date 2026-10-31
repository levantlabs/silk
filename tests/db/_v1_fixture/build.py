"""Build a real, representative schema-v1 database through the genuine v1
write path (TASK-4, DB-02-SCHEMA migration tests).

Uses the FROZEN ``legacy_circuit_db.Database`` (a verbatim snapshot of v1
``circuit_db.py``) plus the live, schema-unaffected ``silk.db.identity`` /
``silk.db.models`` / ``silk.db.structure_tracker`` / ``silk.specs`` modules —
exactly the same machinery real v1 callers used — instead of hand-authoring
a SQLite file.

The resulting database exercises every table the v1-to-v2 migration must
handle: two circuit versions (parent/child lineage), devices, nets, a
parasitic, simulation_results at two corners, a parent+child spec pair with
two spec_measurements, net_currents at two corners (aggregate + pin rows,
mixed spice/user sources), one module + circuit_modules link, and one
release.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from silk.db.identity import (
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprint_module_structure,
    _fingerprint_spec_set,
    _fingerprint_structure,
)
from silk.db.models import SnapshotIdentity
from silk.db.structure_tracker import CircuitStructureTracker
from silk.specs import Spec

from tests.db._v1_fixture.legacy_circuit_db import Database, DesignStage


class _SpecOwner:
    def __init__(self):
        self._specs = []
        self._element_mgr = type("ElementManager", (), {"elem_dict": {}})()


def _structure(tag: str) -> Dict[str, Any]:
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
        "nets": [
            {"name": f"out_{tag}", "type": "signal"},
            {"name": f"mid_{tag}", "type": "signal"},
        ],
        "hierarchy": [],
        "parameters": [],
        "passives": [],
        "modules": {
            "core": {**module_payload, "hash": module_hash, "device_count": 1},
        },
    }


def _spec_definitions(target: float):
    owner = _SpecOwner()
    parent = Spec(
        "system gain", owner, "gain", ">=", target, units="dB", key="system_gain"
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


def build_v1_database(path: str) -> Dict[str, Any]:
    """Populate a fresh v1 database at ``path`` (must not already exist)."""

    db = Database(str(path))
    try:
        # --- Parent circuit version -------------------------------------
        structure_a = _structure("a")
        structure_fp_a, _ = _fingerprint_structure(structure_a)
        spec_set_a, definitions_a = _spec_definitions(40.0)
        identity_a = SnapshotIdentity(structure_fp_a, spec_set_a)

        version_a = db.create_circuit(
            "amp", "1.0", stage=DesignStage.DEV, identity=identity_a
        )
        tracker_a = CircuitStructureTracker(db, version_a)
        structure_hash_a, module_hashes_a = tracker_a.compute_structure_hash(
            structure_a
        )
        tracker_a.store_structure(structure_a, structure_hash_a, module_hashes_a)
        db.store_spec_declarations(version_a, definitions_a)

        # devices/nets already exist via store_structure; add one more
        # device directly to exercise add_device's write path too (both
        # require the still-'building' state).
        db.add_device(version_a, "m_extra", "nmos", model="nfet", width=2.0)
        db.add_net(version_a, "extra_net", net_type="signal")

        db.seal_circuit_version(version_a)

        net_id_a = db.conn.execute(
            "SELECT net_id FROM nets WHERE circuit_version_id = ? "
            "ORDER BY name LIMIT 1",
            (version_a,),
        ).fetchone()[0]
        spec_child_a = db.conn.execute(
            "SELECT spec_id FROM specs WHERE circuit_version_id = ? "
            "AND spec_key = 'core_gain'",
            (version_a,),
        ).fetchone()[0]

        db.add_simulation_result(
            version_a, "gain", value=41.5, unit="dB", corner="tt",
            temperature=27.0, voltage=1.8, tool_name="ngspice-44",
            source="spice",
        )
        db.add_simulation_result(
            version_a, "gain", value=39.0, unit="dB", corner="ss",
            temperature=125.0, voltage=1.62, tool_name="ngspice-44",
            source="spice",
        )
        db.add_parasitic(
            version_a, net_id_a, "capacitance", 2.5e-15, "F",
            extraction_tool="fixture",
        )

        db.insert_spec_measurement(
            str(uuid.uuid4()), spec_child_a, "spice", 41.0,
            corner="tt", temperature=27.0, voltage=1.8, sim_type="ac",
            tool_name="ngspice-44",
        )
        db.insert_spec_measurement(
            str(uuid.uuid4()), spec_child_a, "spice", 40.5,
            corner="ss", temperature=125.0, voltage=1.62, sim_type="ac",
            tool_name="ngspice-44",
        )

        db.replace_net_current_set(
            version_a,
            corner="tt",
            structure_scheme=STRUCTURE_FINGERPRINT_SCHEME,
            structure_hash=structure_fp_a.digest,
            rows=[
                {
                    "net_name": f"out_a", "pin_id": "", "avg_a": 1e-3,
                    "rms_a": 1e-3, "peak_a": 1.2e-3, "source": "spice",
                },
                {
                    "net_name": f"out_a", "pin_id": "m_a.d", "avg_a": 1e-3,
                    "rms_a": 1e-3, "peak_a": 1.2e-3, "source": "spice",
                },
            ],
            temperature=27.0, voltage=1.8, tool_name="ngspice-44",
            analysis="transient", tstart=1e-6, results_file="/tmp/a-tt.raw",
        )
        db.replace_net_current_set(
            version_a,
            corner="ss",
            structure_scheme=STRUCTURE_FINGERPRINT_SCHEME,
            structure_hash=structure_fp_a.digest,
            rows=[
                {
                    "net_name": f"out_a", "pin_id": "", "avg_a": 0.9e-3,
                    "rms_a": 0.9e-3, "peak_a": 1.0e-3, "source": "user",
                },
            ],
            temperature=125.0, voltage=1.62, tool_name="ngspice-44",
            analysis="transient", tstart=2e-6, results_file="/tmp/a-ss.raw",
        )

        release_id = db.create_release(
            "fixture-release", DesignStage.DEV, circuit_names=["amp"]
        )

        # --- Child circuit version (lineage) -----------------------------
        structure_b = _structure("b")
        structure_fp_b, _ = _fingerprint_structure(structure_b)
        spec_set_b, definitions_b = _spec_definitions(45.0)
        identity_b = SnapshotIdentity(structure_fp_b, spec_set_b)

        version_b = db.create_circuit(
            "amp", "2.0", stage=DesignStage.DEV, identity=identity_b,
            parent_version_id=version_a,
        )
        tracker_b = CircuitStructureTracker(db, version_b)
        structure_hash_b, module_hashes_b = tracker_b.compute_structure_hash(
            structure_b
        )
        tracker_b.store_structure(structure_b, structure_hash_b, module_hashes_b)
        db.store_spec_declarations(version_b, definitions_b)
        db.seal_circuit_version(version_b)

        db.add_simulation_result(
            version_b, "gain", value=46.0, unit="dB", corner="tt",
            temperature=27.0, voltage=1.8, tool_name="ngspice-44",
            source="spice",
        )
        db.replace_net_current_set(
            version_b,
            corner="tt",
            structure_scheme=STRUCTURE_FINGERPRINT_SCHEME,
            structure_hash=structure_fp_b.digest,
            rows=[
                {
                    "net_name": "out_b", "pin_id": "", "avg_a": 1.1e-3,
                    "rms_a": 1.1e-3, "peak_a": 1.3e-3, "source": "spice",
                },
            ],
            temperature=27.0, voltage=1.8, tool_name="ngspice-44",
            analysis="transient", tstart=1e-6, results_file="/tmp/b-tt.raw",
        )

        return {
            "version_a": version_a,
            "version_b": version_b,
            "net_id_a": net_id_a,
            "spec_child_a": spec_child_a,
            "release_id": release_id,
            "structure_fp_a": structure_fp_a,
            "structure_fp_b": structure_fp_b,
        }
    finally:
        db.close()
