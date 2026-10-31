"""Typed, deterministic simulation-metric persistence and live readback."""

from dataclasses import FrozenInstanceError
import inspect

import pytest

import silk.db as silk_db
from silk.circuit.modules import Module
from silk.db import (
    Database,
    DatabaseError,
    DesignStage,
    EvidenceMismatchError,
    EvidenceNotFoundError,
    UnsupportedFingerprintSchemeError,
)
from silk.db.schema import REQUIRED_COLUMNS
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
    structure_with_net,
)


@pytest.fixture
def tracked_module(pdk, tmp_dir):
    module = Module(
        "PDK_skywater130",
        tmp_dir,
        "metric_readback.cir",
        pdk_details=pdk,
        module_name="metric_readback",
        db_path=":memory:",
    )
    yield module
    module.close()


def _result_count(db, version_id):
    return db.conn.execute(
        """
        SELECT COUNT(*) FROM simulation_results
        WHERE circuit_version_id = ?
        """,
        (version_id,),
    ).fetchone()[0]


def test_schema_v1_requires_explicit_metric_source():
    with Database(":memory:") as db:
        source = next(
            row
            for row in db.conn.execute(
                "PRAGMA table_info(simulation_results)"
            ).fetchall()
            if row["name"] == "source"
        )

        assert source["notnull"] == 1
        assert "source" in REQUIRED_COLUMNS["simulation_results"]


def test_numeric_and_text_results_round_trip_with_complete_context(tmp_path):
    db_path = tmp_path / "metrics.db"
    with Database(db_path) as db:
        version_id = create_sealed_snapshot(
            db,
            name="amp",
            structure=structure_with_net("out"),
        )
        numeric_id = db.add_simulation_result(
            version_id,
            "gain",
            value=42.5,
            unit="dB",
            object_id="out",
            object_type="net",
            source="spice",
            sim_type="ac",
            corner="tt",
            temperature=27.0,
            voltage=1.8,
            tool_name="ngspice-44",
            conditions={"mode": "closed-loop", "load_a": 0.001},
            timestamp="2026-08-29 12:00:00.000001",
        )
        text_id = db.add_simulation_result(
            version_id,
            "convergence",
            value_text="settled",
            unit="",
            source="behavioral",
            sim_type="transient",
            corner="tt",
            temperature=27.0,
            voltage=1.8,
            tool_name="silk",
            conditions={"samples": 128},
            timestamp="2026-08-29 12:00:00.000002",
        )
        stored_conditions = db.conn.execute(
            "SELECT conditions FROM simulation_results WHERE sim_id = ?",
            (numeric_id,),
        ).fetchone()[0]
        assert stored_conditions == '{"load_a":0.001,"mode":"closed-loop"}'

    with Database(db_path) as reopened:
        records = reopened.get_simulation_results(
            version_id, include_ancestors=False
        )

    assert all(
        isinstance(record, silk_db.SimulationResultRecord)
        for record in records
    )
    assert [record.result_id for record in records] == [numeric_id, text_id]
    numeric, text = records
    assert numeric.origin_version_id == version_id
    assert numeric.parameter == "gain"
    assert numeric.value == 42.5
    assert numeric.value_text is None
    assert numeric.unit == "dB"
    assert numeric.object_id == "out"
    assert numeric.object_type == "net"
    assert numeric.source == "spice"
    assert numeric.sim_type == "ac"
    assert numeric.corner == "tt"
    assert numeric.temperature == 27.0
    assert numeric.voltage == 1.8
    assert numeric.tool_name == "ngspice-44"
    assert numeric.conditions == {"load_a": 0.001, "mode": "closed-loop"}
    assert numeric.timestamp == "2026-08-29 12:00:00.000001"
    assert text.value is None
    assert text.value_text == "settled"
    assert text.conditions == {"samples": 128}
    with pytest.raises(FrozenInstanceError):
        numeric.value = 0.0


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({}, ValueError, "exactly one"),
        ({"value": 1.0, "value_text": "one"}, ValueError, "exactly one"),
        ({"value": True}, TypeError, "boolean"),
        ({"value_text": 7}, TypeError, "text"),
        ({"value": 1.0, "conditions": ["not", "a", "mapping"]}, TypeError, "conditions"),
    ],
)
def test_invalid_result_payload_fails_before_insert(kwargs, error, message):
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db)

        with pytest.raises(error, match=message):
            db.add_simulation_result(version_id, "gain", **kwargs)

        assert _result_count(db, version_id) == 0


def test_store_results_rolls_back_all_rows_after_injected_failure(
    tracked_module, monkeypatch
):
    version_id = tracked_module.create_circuit_version()
    db = tracked_module._tracker.db
    real_add = db.add_simulation_result
    calls = 0

    def fail_after_first_real_insert(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected metric failure")
        return real_add(*args, **kwargs)

    monkeypatch.setattr(db, "add_simulation_result", fail_after_first_real_insert)
    with pytest.raises(RuntimeError, match="injected metric failure"):
        tracked_module.store_results(
            {"gain": (40.0, "dB"), "bandwidth": (1e6, "Hz")},
            source="spice",
        )

    assert _result_count(db, version_id) == 0


def test_store_results_rejects_a_live_design_changed_after_binding(
    tracked_module,
):
    version_id = tracked_module.create_circuit_version()
    tracked_module.add_net("late_edit")

    with pytest.raises(EvidenceMismatchError, match="structure identity"):
        tracked_module.store_results({"gain": (40.0, "dB")})

    assert _result_count(tracked_module._tracker.db, version_id) == 0


def test_all_results_order_by_timestamp_lineage_rank_then_rowid():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        first_id = db.add_simulation_result(
            source_id,
            "sequence",
            value=1.0,
            timestamp="2026-08-29 12:00:00.000001",
        )
        source_tie_id = db.add_simulation_result(
            source_id,
            "sequence",
            value=3.0,
            timestamp="2026-08-29 12:00:00.000002",
        )
        source_tie_2_id = db.add_simulation_result(
            source_id,
            "sequence",
            value=4.0,
            timestamp="2026-08-29 12:00:00.000002",
        )
        target_id = db.promote_circuit(
            source_id, DesignStage.STABLE, lock=False
        )
        target_tie_id = db.add_simulation_result(
            target_id,
            "sequence",
            value=2.0,
            timestamp="2026-08-29 12:00:00.000002",
        )

        records = db.get_simulation_results(
            target_id,
            parameter="sequence",
            include_ancestors=True,
        )

    assert [record.result_id for record in records] == [
        first_id,
        target_tie_id,
        source_tie_id,
        source_tie_2_id,
    ]


def test_latest_groups_by_the_complete_logical_context():
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db)
        base = {
            "parameter": "gain",
            "object_type": "circuit",
            "object_id": None,
            "source": "spice",
            "sim_type": "ac",
            "corner": "tt",
            "temperature": 27.0,
            "voltage": 1.8,
            "tool_name": "ngspice",
            "conditions": {"load": "nominal", "sweep": 1},
        }

        def store(value, timestamp, **changes):
            context = {**base, **changes}
            parameter = context.pop("parameter")
            return db.add_simulation_result(
                version_id,
                parameter,
                value=value,
                timestamp=timestamp,
                **context,
            )

        store(1.0, "2026-08-29 12:00:00.000001")
        newest_base = store(
            2.0,
            "2026-08-29 12:00:00.000002",
            conditions={"sweep": 1, "load": "nominal"},
        )
        variations = [
            {"parameter": "bandwidth"},
            {"object_type": "net"},
            {"object_id": "out"},
            {"source": "behavioral"},
            {"sim_type": "transient"},
            {"corner": "ss"},
            {"temperature": 125.0},
            {"voltage": 1.62},
            {"tool_name": "xyce"},
            {"conditions": {"load": "heavy", "sweep": 1}},
        ]
        for offset, variation in enumerate(variations, start=3):
            store(
                float(offset),
                f"2026-08-29 12:00:00.{offset:06d}",
                **variation,
            )

        latest = db.get_simulation_results(
            version_id, latest_only=True, include_ancestors=False
        )

    assert len(latest) == 11
    base_records = [
        record
        for record in latest
        if (
            record.parameter,
            record.object_type,
            record.object_id,
            record.source,
            record.sim_type,
            record.corner,
            record.temperature,
            record.voltage,
            record.tool_name,
            record.conditions,
        )
        == (
            "gain",
            "circuit",
            None,
            "spice",
            "ac",
            "tt",
            27.0,
            1.8,
            "ngspice",
            {"load": "nominal", "sweep": 1},
        )
    ]
    assert [record.result_id for record in base_records] == [newest_base]


def test_database_filters_are_exact_and_missing_history_is_an_empty_list():
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db)
        db.add_simulation_result(
            version_id,
            "gain",
            value=42.0,
            source="spice",
            sim_type="ac",
            corner="tt",
            temperature=27.0,
            voltage=1.8,
            tool_name="ngspice",
            conditions={"b": 2, "a": 1},
        )

        matches = db.get_simulation_results(
            version_id,
            parameter="gain",
            source="spice",
            sim_type="ac",
            corner="tt",
            temperature=27.0,
            voltage=1.8,
            tool_name="ngspice",
            conditions={"a": 1, "b": 2},
            include_ancestors=False,
        )
        missing = db.get_simulation_results(
            version_id,
            parameter="gain",
            corner="ss",
            include_ancestors=False,
        )

    assert len(matches) == 1
    assert missing == []


def test_inherited_results_keep_their_origin_and_are_not_copied():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        result_id = db.add_simulation_result(
            source_id, "gain", value=42.0, source="spice", corner="tt"
        )
        target_id = db.promote_circuit(
            source_id, DesignStage.STABLE, lock=False
        )

        inherited = db.get_simulation_results(
            target_id, parameter="gain", include_ancestors=True
        )
        local = db.get_simulation_results(
            target_id, parameter="gain", include_ancestors=False
        )

        assert _result_count(db, target_id) == 0
    assert [record.result_id for record in inherited] == [result_id]
    assert inherited[0].origin_version_id == source_id
    assert local == []


def test_building_snapshot_cannot_expose_inherited_results():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        db.add_simulation_result(source_id, "gain", value=42.0)
        building_id = create_building_snapshot(
            db,
            name="amp",
            version="1.1",
            parent_version_id=source_id,
        )

        with pytest.raises(DatabaseError, match="sealed"):
            db.get_simulation_results(building_id, parameter="gain")


def test_live_query_distinguishes_missing_context_from_incompatible_structure(
    tracked_module,
):
    source_id = tracked_module.create_circuit_version()
    tracked_module.store_result(
        "gain", 42.0, "dB", corner="tt", source="spice"
    )

    with pytest.raises(EvidenceNotFoundError, match="corner|context|stored"):
        tracked_module.query_results(parameter="gain", corner="ss")

    tracked_module.add_net("structural_edit")
    child_id = tracked_module.create_circuit_version(
        parent_version_id=source_id
    )
    assert child_id != source_id
    with pytest.raises(EvidenceMismatchError, match="structure"):
        tracked_module.query_results(parameter="gain", corner="tt")


def test_live_query_returns_typed_inherited_results(tracked_module):
    source_id = tracked_module.create_circuit_version()
    tracked_module.store_result(
        "gain",
        42.0,
        "dB",
        source="spice",
        sim_type="ac",
        corner="tt",
        conditions={"load": "nominal"},
    )
    target_id = tracked_module.promote("stable", lock=False)

    records = tracked_module.query_results(parameter="gain")

    assert len(records) == 1
    assert records[0].origin_version_id == source_id
    assert records[0].origin_version_id != target_id


def test_unsupported_structure_scheme_fails_before_result_lookup(
    tracked_module,
):
    version_id = tracked_module.create_circuit_version()
    tracked_module._tracker.db.conn.execute(
        """
        UPDATE circuit_versions SET structure_scheme = 'silk.structure/v999'
        WHERE version_id = ?
        """,
        (version_id,),
    )
    tracked_module._tracker.db.conn.commit()

    with pytest.raises(UnsupportedFingerprintSchemeError):
        tracked_module.query_results(parameter="never-stored")


def test_public_query_signature_is_keyword_only():
    signature = inspect.signature(Module.query_results)
    assert str(signature) == (
        "(self, *, parameter=None, source=None, sim_type=None, corner=None, "
        "temperature=None, voltage=None, tool_name=None, conditions=None, "
        "profile_id=None, latest_only=False, include_ancestors=True)"
    )


def test_store_pass_fail_keeps_its_existing_public_signature():
    assert str(inspect.signature(Module.store_pass_fail)) == (
        "(self, spec_name, status, details='')"
    )
