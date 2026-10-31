"""Tests for the constraint-checker dispatch mechanism (stubs only)."""
import pytest


@pytest.fixture(autouse=True)
def _patch():
    import silk.constraints  # noqa: F401


@pytest.mark.parametrize("ctor", [
    lambda: __import__("silk.constraints", fromlist=["SymmetricBlocks"]).SymmetricBlocks(
        pairs=[["A", "B"]], direction="V"),
    lambda: __import__("silk.constraints", fromlist=["GroupBlocks"]).GroupBlocks(
        name="g", instances=["A"]),
    lambda: __import__("silk.constraints", fromlist=["SameTemplate"]).SameTemplate(
        instances=["A", "B"]),
    lambda: __import__("silk.constraints", fromlist=["ProximityConstraint"]).ProximityConstraint(
        instance_a="A", instance_b="B", max_distance_um=2),
])
def test_dispatch_finds_checker_for_every_type(ctor):
    """Every shipped constraint type has a registered (stub) checker."""
    from silk.constraints import checkers
    constraint = ctor()
    with pytest.raises(NotImplementedError):
        checkers.check(constraint, layout=None)


def test_unknown_type_raises_lookup_error():
    """A constraint whose type has no checker registered must surface
    a clear error — not silently no-op."""
    from silk.constraints import checkers
    from silk.constraints.envelope import ConstraintEnvelope
    from typing import Literal

    class MysteryConstraint(ConstraintEnvelope):
        constraint: Literal["MysteryConstraint"] = "MysteryConstraint"

    with pytest.raises(LookupError):
        checkers.check(MysteryConstraint(), layout=None)


def test_register_lets_callers_add_their_own_checkers():
    """Open dispatch — layout consumers register concrete
    checkers later without modifying silk.constraints."""
    from silk.constraints import checkers
    from silk.constraints.types import SymmetricBlocks

    sentinel = object()
    calls = []

    def my_checker(c, layout):
        calls.append((c, layout))
        return sentinel

    token = checkers.register(SymmetricBlocks, my_checker)
    try:
        result = checkers.check(
            SymmetricBlocks(pairs=[["A", "B"]], direction="V"),
            layout="LAYOUT",
        )
        assert result is sentinel
        assert len(calls) == 1
        assert calls[0][1] == "LAYOUT"
    finally:
        checkers.unregister(SymmetricBlocks, token)


def test_unregister_restores_original_stub():
    from silk.constraints import checkers
    from silk.constraints.types import SymmetricBlocks

    token = checkers.register(SymmetricBlocks, lambda c, layout: "shadow")
    checkers.unregister(SymmetricBlocks, token)
    with pytest.raises(NotImplementedError):
        checkers.check(SymmetricBlocks(pairs=[["A"]], direction="V"), layout=None)


def test_check_all_runs_each_constraint_against_layout(empty_netlist):
    from silk.constraints import checkers
    from silk.constraints.types import SymmetricBlocks, GroupBlocks

    received = []

    def my_checker(c, layout):
        received.append(c.constraint)
        return "ok"

    t1 = checkers.register(SymmetricBlocks, my_checker)
    t2 = checkers.register(GroupBlocks, my_checker)
    try:
        empty_netlist.add_constraint(SymmetricBlocks(pairs=[["A"]], direction="V"))
        empty_netlist.add_constraint(GroupBlocks(name="g", instances=["A"]))
        results = checkers.check_all(empty_netlist, layout="L")
        assert len(results) == 2
        assert received == ["SymmetricBlocks", "GroupBlocks"]
    finally:
        checkers.unregister(SymmetricBlocks, t1)
        checkers.unregister(GroupBlocks, t2)
