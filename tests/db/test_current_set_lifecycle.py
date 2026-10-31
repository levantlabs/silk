"""DB-02 selected-current-set identity and lifecycle proofs."""

import sqlite3

import pytest

from silk.db.circuit_db import Database
from silk.db.context import CharacterizationContext
from silk.exceptions import EvidenceMismatchError
from tests.db_snapshot_helpers import create_sealed_snapshot


@pytest.fixture
def db():
    with Database(":memory:") as database:
        yield database


def _context(
    *,
    test_name="load_step",
    parameters=None,
    corner="tt",
    temperature=27.0,
    voltage=1.8,
):
    return CharacterizationContext(
        owner_path="$/amp",
        test_name=test_name,
        parameters={} if parameters is None else parameters,
        corner=corner,
        temperature=temperature,
        voltage=voltage,
    )


def _row(avg_a):
    return {
        "net_name": "out",
        "pin_id": "",
        "avg_a": avg_a,
        "rms_a": None,
        "peak_a": None,
        "source": "spice",
        "current_vector": None,
    }


def _store(db, version_id, context, avg_a):
    version = db._snapshot_row(version_id)
    return db.replace_net_current_set(
        version_id,
        corner=context.corner,
        temperature=context.temperature,
        voltage=context.voltage,
        structure_scheme=version["structure_scheme"],
        structure_hash=version["structure_hash"],
        context=context,
        rows=[_row(avg_a)],
    )


def test_scoped_sets_at_one_corner_coexist_and_are_exactly_retrievable(db):
    version_id = create_sealed_snapshot(db, name="amp")
    nominal = _context(parameters={"cload_f": 1e-12})
    hot = _context(
        test_name="startup",
        parameters={"cload_f": 2e-12},
        temperature=85.0,
        voltage=1.62,
    )

    nominal_set_id = _store(db, version_id, nominal, 1e-3)
    hot_set_id = _store(db, version_id, hot, 2e-3)
    nominal_profile = db.get_characterization_profile(
        version_id, context=nominal
    )
    hot_profile = db.get_characterization_profile(version_id, context=hot)

    nominal_origin, nominal_rows = db.get_net_current_set(
        version_id,
        "tt",
        profile_id=nominal_profile.profile_id,
        temperature=27.0,
        voltage=1.8,
    )
    hot_origin, hot_rows = db.get_net_current_set(
        version_id,
        "tt",
        profile_id=hot_profile.profile_id,
        temperature=85.0,
        voltage=1.62,
    )

    assert nominal_set_id != hot_set_id
    assert nominal_profile.profile_id != hot_profile.profile_id
    assert nominal_origin == hot_origin == version_id
    assert {row["current_set_id"] for row in nominal_rows} == {
        nominal_set_id
    }
    assert len(nominal_rows) == 1
    assert nominal_rows[0]["avg_a"] == pytest.approx(1e-3)
    assert {row["current_set_id"] for row in hot_rows} == {hot_set_id}
    assert len(hot_rows) == 1
    assert hot_rows[0]["avg_a"] == pytest.approx(2e-3)


@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
def test_replacement_retains_immutable_history_and_selects_only_successor(
    db, scoped,
):
    version_id = create_sealed_snapshot(db, name="amp")
    context = _context(parameters={"mode": "max"}) if scoped else None
    version = db._snapshot_row(version_id)
    common = {
        "corner": "tt",
        "temperature": 27.0,
        "voltage": 1.8,
        "structure_scheme": version["structure_scheme"],
        "structure_hash": version["structure_hash"],
        "context": context,
    }

    old_set_id = db.replace_net_current_set(
        version_id, rows=[_row(1e-3)], **common
    )
    new_set_id = db.replace_net_current_set(
        version_id, rows=[_row(2e-3)], **common
    )

    parents = db.conn.execute(
        """
        SELECT current_set_id, state, supersedes_current_set_id, superseded_at
        FROM net_current_sets
        WHERE circuit_version_id = ?
        ORDER BY selected_at, rowid
        """,
        (version_id,),
    ).fetchall()
    assert new_set_id != old_set_id
    assert [tuple(row) for row in parents] == [
        (old_set_id, "superseded", None, parents[0]["superseded_at"]),
        (new_set_id, "current", old_set_id, None),
    ]
    assert parents[0]["superseded_at"] is not None

    old_child = db.conn.execute(
        """
        SELECT current_id, avg_a FROM net_currents
        WHERE current_set_id = ?
        """,
        (old_set_id,),
    ).fetchone()
    new_child = db.conn.execute(
        """
        SELECT current_id, avg_a FROM net_currents
        WHERE current_set_id = ?
        """,
        (new_set_id,),
    ).fetchone()
    assert old_child["avg_a"] == pytest.approx(1e-3)
    assert new_child["avg_a"] == pytest.approx(2e-3)
    assert old_child["current_id"] != new_child["current_id"]

    selector = {}
    if scoped:
        profile = db.get_characterization_profile(version_id, context=context)
        selector = {
            "profile_id": profile.profile_id,
            "temperature": 27.0,
            "voltage": 1.8,
        }
    _origin, selected = db.get_net_current_set(
        version_id, "tt", **selector
    )
    _historical_origin, historical = db.get_net_current_set(
        version_id, current_set_id=old_set_id
    )

    assert {row["current_set_id"] for row in selected} == {new_set_id}
    assert selected[0]["avg_a"] == pytest.approx(2e-3)
    assert {row["current_set_id"] for row in historical} == {old_set_id}
    assert historical[0]["state"] == "superseded"
    assert historical[0]["avg_a"] == pytest.approx(1e-3)


def test_profile_selection_requires_complete_exact_pvt(db):
    version_id = create_sealed_snapshot(db, name="amp")
    context = _context(parameters={"mode": "max"})
    _store(db, version_id, context, 1e-3)
    profile = db.get_characterization_profile(version_id, context=context)

    with pytest.raises(ValueError, match="profile.*temperature.*voltage"):
        db.get_net_current_set(
            version_id,
            "tt",
            profile_id=profile.profile_id,
        )

    with pytest.raises(ValueError, match="profile.*temperature.*voltage"):
        db.get_net_current_set(
            version_id,
            "tt",
            profile_id=profile.profile_id,
            temperature=27.0,
        )

    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_current_sets WHERE state = 'current'"
    ).fetchone()[0] == 1


def test_same_profile_at_two_pvt_points_has_two_current_selections(db):
    version_id = create_sealed_snapshot(db, name="amp")
    nominal = _context(parameters={"cload_f": 2e-12})
    hot = _context(
        parameters={"cload_f": 2e-12},
        temperature=85.0,
        voltage=1.62,
    )

    nominal_set_id = _store(db, version_id, nominal, 1e-3)
    hot_set_id = _store(db, version_id, hot, 2e-3)
    nominal_profile = db.get_characterization_profile(
        version_id, context=nominal
    )
    hot_profile = db.get_characterization_profile(version_id, context=hot)
    assert nominal_profile.profile_id == hot_profile.profile_id

    _origin, nominal_rows = db.get_net_current_set(
        version_id,
        "tt",
        profile_id=nominal_profile.profile_id,
        temperature=27.0,
        voltage=1.8,
    )
    _origin, hot_rows = db.get_net_current_set(
        version_id,
        "tt",
        profile_id=hot_profile.profile_id,
        temperature=85.0,
        voltage=1.62,
    )

    assert nominal_set_id != hot_set_id
    assert nominal_rows[0]["current_set_id"] == nominal_set_id
    assert nominal_rows[0]["avg_a"] == pytest.approx(1e-3)
    assert hot_rows[0]["current_set_id"] == hot_set_id
    assert hot_rows[0]["avg_a"] == pytest.approx(2e-3)


def test_context_pvt_mismatch_fails_before_profile_or_current_mutation(db):
    version_id = create_sealed_snapshot(db, name="amp")
    context = _context(temperature=85.0, voltage=1.62)
    version = db._snapshot_row(version_id)

    with pytest.raises(
        EvidenceMismatchError,
        match="exactly match.*characterization context",
    ):
        db.replace_net_current_set(
            version_id,
            corner="tt",
            temperature=27.0,
            voltage=1.8,
            structure_scheme=version["structure_scheme"],
            structure_hash=version["structure_hash"],
            context=context,
            rows=[_row(1e-3)],
        )

    assert db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_current_sets"
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_currents"
    ).fetchone()[0] == 0


@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
def test_failed_child_insert_rolls_back_supersession(db, scoped):
    version_id = create_sealed_snapshot(db, name="amp")
    context = _context(parameters={"mode": "max"}) if scoped else None
    version = db._snapshot_row(version_id)
    common = {
        "corner": "tt",
        "temperature": 27.0,
        "voltage": 1.8,
        "structure_scheme": version["structure_scheme"],
        "structure_hash": version["structure_hash"],
        "context": context,
    }
    old_set_id = db.replace_net_current_set(
        version_id, rows=[_row(1e-3)], **common
    )

    with pytest.raises(sqlite3.IntegrityError):
        db.replace_net_current_set(
            version_id,
            rows=[_row(2e-3), _row(3e-3)],
            **common,
        )

    parents = db.conn.execute(
        """
        SELECT current_set_id, state, supersedes_current_set_id, superseded_at
        FROM net_current_sets
        """
    ).fetchall()
    children = db.conn.execute(
        "SELECT current_set_id, avg_a FROM net_currents"
    ).fetchall()
    assert [tuple(row) for row in parents] == [
        (old_set_id, "current", None, None)
    ]
    assert len(children) == 1
    assert children[0]["current_set_id"] == old_set_id
    assert children[0]["avg_a"] == pytest.approx(1e-3)
