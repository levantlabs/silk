"""Tests for the Module.add_constraint / list_constraints monkey-patch."""
import pytest


@pytest.fixture(autouse=True)
def _ensure_patch_installed():
    import silk.constraints  # noqa: F401


def test_module_gets_add_constraint_on_import():
    from silk.circuit.modules import Module
    assert hasattr(Module, "add_constraint")
    assert hasattr(Module, "list_constraints")


def test_add_constraint_appends_and_returns(empty_netlist):
    from silk.constraints import SymmetricBlocks
    c = SymmetricBlocks(pairs=[["MN0", "MN1"]], direction="V")
    returned = empty_netlist.add_constraint(c)
    assert returned is c
    assert empty_netlist.list_constraints() == [c]


def test_add_constraint_rejects_non_envelope(empty_netlist):
    with pytest.raises(TypeError):
        empty_netlist.add_constraint({"not": "a constraint"})


def test_add_multiple_constraints_preserves_order(empty_netlist):
    from silk.constraints import SymmetricBlocks, GroupBlocks, SameTemplate
    a = SymmetricBlocks(pairs=[["A", "B"]], direction="V")
    b = GroupBlocks(name="g", instances=["A", "B"])
    c = SameTemplate(instances=["A", "B"])
    empty_netlist.add_constraint(a)
    empty_netlist.add_constraint(b)
    empty_netlist.add_constraint(c)
    assert empty_netlist.list_constraints() == [a, b, c]


def test_list_constraints_returns_copy(empty_netlist):
    from silk.constraints import SymmetricBlocks
    empty_netlist.add_constraint(SymmetricBlocks(pairs=[["A"]], direction="V"))
    snapshot = empty_netlist.list_constraints()
    snapshot.clear()
    assert len(empty_netlist.list_constraints()) == 1


def test_install_is_idempotent():
    from silk.constraints._module_patch import install
    from silk.circuit.modules import Module
    install()
    install()
    assert callable(Module.add_constraint)


def test_validate_against_real_module(basic_netlist):
    """End-to-end: validators see a real silk Module's instances."""
    from silk.constraints import SymmetricBlocks, GroupBlocks
    sym = SymmetricBlocks(pairs=[["mn1", "mp1"]], direction="V")
    assert sym.validate_against(basic_netlist) == []

    bad = SymmetricBlocks(pairs=[["mn1", "ghost"]], direction="V")
    assert any("ghost" in e for e in bad.validate_against(basic_netlist))

    gb = GroupBlocks(name="diff", instances=["mn1", "mp1"])
    assert gb.validate_against(basic_netlist) == []
