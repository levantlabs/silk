"""Complete Spec declarations are part of every sealed circuit snapshot."""

from dataclasses import FrozenInstanceError
import inspect

import pytest

import silk.db as silk_db
from silk.circuit.modules import Module
from silk.db import Database
from silk.specs import Measurement


def _module(pdk, directory, name, *, db_path=None):
    return Module(
        "PDK_skywater130",
        str(directory / name),
        f"{name}.cir",
        pdk_details=pdk,
        module_name=name,
        db_path=str(db_path) if db_path is not None else None,
    )


def _build_spec_tree(pdk, tmp_path, db_path):
    top = _module(pdk, tmp_path, "spec_history", db_path=db_path)
    child_template = _module(pdk, tmp_path, "core")
    top.add_module(child_template, "core1", nets={})
    child = top._element_mgr.elem_dict["core1"]["device"]

    root = top.declare_spec(
        "Readable DC gain",
        "dc_gain",
        ">=",
        55.0,
        units="dB",
        description="System gain target",
        key="dc_gain",
    )
    root.record_measurement(
        Measurement(source="spice", value=61.0, timestamp=10.0)
    )
    child.declare_spec(
        "Settling time",
        "settling_time",
        "<=",
        2e-6,
        units="s",
        description="Core settling target",
        parent=root,
    )
    return top


def test_snapshot_creation_persists_measured_and_unmeasured_declarations(
    pdk, tmp_path,
):
    db_path = tmp_path / "spec-history.db"
    module = _build_spec_tree(pdk, tmp_path, db_path)
    version_id = module.create_circuit_version()
    module.close()

    with Database(db_path) as db:
        records = db.get_stored_specs(version_id)

    assert all(
        isinstance(record, silk_db.SpecDeclarationRecord)
        for record in records
    )
    assert [(record.owner_path, record.spec_key) for record in records] == [
        ("$", "dc_gain"),
        ("$/core1", "Settling time"),
    ]
    root, child = records
    assert root.declaration_version_id == version_id
    assert root.name == "Readable DC gain"
    assert root.metric == "dc_gain"
    assert root.op == ">="
    assert root.target == 55.0
    assert root.units == "dB"
    assert root.description == "System gain target"
    assert "test_db_persistence.py" in root.declared_at
    assert root.parent_id is None
    assert root.measurements == ()

    assert child.name == "Settling time"
    assert child.metric == "settling_time"
    assert child.op == "<="
    assert child.target == 2e-6
    assert child.units == "s"
    assert child.description == "Core settling target"
    assert child.parent_id == root.spec_id
    assert child.parent_owner_path == "$"
    assert child.parent_spec_key == "dc_gain"
    assert child.measurements == ()
    with pytest.raises(FrozenInstanceError):
        root.target = 0.0


def test_bound_module_exposes_exact_historical_query_surface(
    pdk, tmp_path,
):
    module = _build_spec_tree(pdk, tmp_path, tmp_path / "bound.db")
    version_id = module.create_circuit_version()

    records = module.get_stored_specs(include_measurements=False)

    assert len(records) == 2
    assert {record.declaration_version_id for record in records} == {version_id}
    assert all(record.measurements == () for record in records)
    assert str(inspect.signature(Module.get_stored_specs)) == (
        "(self, *, include_measurements=True, include_ancestors=True, "
        "profile_id=None)"
    )
    module.close()
