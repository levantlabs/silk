"""Tests for the four device-relational constraint types.

Port semantics moved to PortRole (see test_module_api.py for the
PortRole serialization bridge tests). Net budgets moved to Net fields (see
tests/test_net_layout_budgets.py).
"""
import pytest
from pydantic import ValidationError


class StubModule:
    def __init__(self, instances=(), nets=()):
        self.elem_dict = {n: object() for n in instances}
        self.net_dict = {n: object() for n in nets}


@pytest.fixture
def small_netlist():
    return StubModule(
        instances=("MN0", "MN1", "MN2", "MN3", "MN4"),
        nets=("vdd", "gnd", "vbias"),
    )


# ---------------------------------------------------------------------------
# Discriminator + envelope inheritance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name", [
    "SymmetricBlocks", "GroupBlocks", "SameTemplate", "ProximityConstraint",
])
def test_each_type_carries_envelope_fields(cls_name):
    import silk.constraints.types as types_mod
    from silk.constraints.envelope import ConstraintEnvelope
    cls = getattr(types_mod, cls_name)
    assert issubclass(cls, ConstraintEnvelope)


def test_constraint_discriminator_is_literal_per_type():
    from silk.constraints.types import (
        SymmetricBlocks, GroupBlocks, SameTemplate, ProximityConstraint,
    )
    assert SymmetricBlocks(pairs=[["A", "B"]], direction="V").constraint == "SymmetricBlocks"
    assert GroupBlocks(name="g", instances=["A"]).constraint == "GroupBlocks"
    assert SameTemplate(instances=["A", "B"]).constraint == "SameTemplate"
    assert ProximityConstraint(
        instance_a="A", instance_b="B", max_distance_um=1
    ).constraint == "ProximityConstraint"


def test_discriminator_cannot_be_overridden():
    from silk.constraints.types import SymmetricBlocks
    with pytest.raises(ValidationError):
        SymmetricBlocks(constraint="SomethingElse",
                        pairs=[["A", "B"]], direction="V")


# ---------------------------------------------------------------------------
# SymmetricBlocks
# ---------------------------------------------------------------------------

def test_symmetric_blocks_accepts_mirror_and_self_pairs(small_netlist):
    from silk.constraints.types import SymmetricBlocks
    c = SymmetricBlocks(
        pairs=[["MN0", "MN1"], ["MN2", "MN3"], ["MN4"]],
        direction="V",
    )
    assert c.validate_against(small_netlist) == []


def test_symmetric_blocks_rejects_unknown_instance(small_netlist):
    from silk.constraints.types import SymmetricBlocks
    c = SymmetricBlocks(pairs=[["MN0", "MGHOST"]], direction="V")
    errs = c.validate_against(small_netlist)
    assert len(errs) == 1
    assert "MGHOST" in errs[0]


def test_symmetric_blocks_rejects_pair_size_other_than_one_or_two():
    from silk.constraints.types import SymmetricBlocks
    with pytest.raises(ValidationError):
        SymmetricBlocks(pairs=[["A", "B", "C"]], direction="V")
    with pytest.raises(ValidationError):
        SymmetricBlocks(pairs=[[]], direction="V")


def test_symmetric_blocks_direction_restricted():
    from silk.constraints.types import SymmetricBlocks
    SymmetricBlocks(pairs=[["A"]], direction="H")
    SymmetricBlocks(pairs=[["A"]], direction="V")
    with pytest.raises(ValidationError):
        SymmetricBlocks(pairs=[["A"]], direction="diagonal")


def test_symmetric_blocks_requires_at_least_one_pair():
    from silk.constraints.types import SymmetricBlocks
    with pytest.raises(ValidationError):
        SymmetricBlocks(pairs=[], direction="V")


# ---------------------------------------------------------------------------
# GroupBlocks
# ---------------------------------------------------------------------------

def test_group_blocks_with_generator(small_netlist):
    from silk.constraints.types import GroupBlocks
    c = GroupBlocks(
        name="input_pair",
        instances=["MN0", "MN1"],
        generator={"name": "MOS", "parameters": {"pattern": "cc"}},
    )
    assert c.validate_against(small_netlist) == []
    assert c.generator == {"name": "MOS", "parameters": {"pattern": "cc"}}


def test_group_blocks_generator_optional():
    from silk.constraints.types import GroupBlocks
    c = GroupBlocks(name="foo", instances=["MN0"])
    assert c.generator is None


def test_group_blocks_requires_at_least_one_instance():
    from silk.constraints.types import GroupBlocks
    with pytest.raises(ValidationError):
        GroupBlocks(name="g", instances=[])


def test_group_blocks_rejects_unknown_instance(small_netlist):
    from silk.constraints.types import GroupBlocks
    c = GroupBlocks(name="g", instances=["MN0", "MGHOST"])
    errs = c.validate_against(small_netlist)
    assert any("MGHOST" in e for e in errs)


# ---------------------------------------------------------------------------
# SameTemplate
# ---------------------------------------------------------------------------

def test_same_template_validates(small_netlist):
    from silk.constraints.types import SameTemplate
    c = SameTemplate(instances=["MN0", "MN1", "MN2"])
    assert c.validate_against(small_netlist) == []


def test_same_template_needs_at_least_two():
    from silk.constraints.types import SameTemplate
    with pytest.raises(ValidationError):
        SameTemplate(instances=["MN0"])


def test_same_template_unknown_instance(small_netlist):
    from silk.constraints.types import SameTemplate
    c = SameTemplate(instances=["MN0", "MGHOST"])
    assert any("MGHOST" in e for e in c.validate_against(small_netlist))


# ---------------------------------------------------------------------------
# ProximityConstraint
# ---------------------------------------------------------------------------

def test_proximity_constraint_ok(small_netlist):
    from silk.constraints.types import ProximityConstraint
    c = ProximityConstraint(
        instance_a="MN0", instance_b="MN1", max_distance_um=5.0,
    )
    assert c.validate_against(small_netlist) == []


def test_proximity_unknown_instance(small_netlist):
    from silk.constraints.types import ProximityConstraint
    errs = ProximityConstraint(
        instance_a="MNX", instance_b="MGHOST", max_distance_um=1.0,
    ).validate_against(small_netlist)
    assert any("MNX" in e for e in errs)
    assert any("MGHOST" in e for e in errs)


def test_proximity_distance_must_be_positive():
    from silk.constraints.types import ProximityConstraint
    with pytest.raises(ValidationError):
        ProximityConstraint(instance_a="A", instance_b="B", max_distance_um=0)


def test_proximity_self_pairing_rejected():
    from silk.constraints.types import ProximityConstraint
    with pytest.raises(ValidationError):
        ProximityConstraint(instance_a="A", instance_b="A", max_distance_um=2)


# ---------------------------------------------------------------------------
# Severity defaults are hard/soft and can be overridden per instance
# ---------------------------------------------------------------------------

def test_hard_types_default_to_hard_severity():
    from silk.constraints.types import SymmetricBlocks, GroupBlocks
    assert SymmetricBlocks(pairs=[["A"]], direction="V").severity == "hard"
    assert GroupBlocks(name="g", instances=["A"]).severity == "hard"


def test_soft_types_default_to_soft_severity():
    from silk.constraints.types import SameTemplate, ProximityConstraint
    assert SameTemplate(instances=["A", "B"]).severity == "soft"
    assert ProximityConstraint(
        instance_a="A", instance_b="B", max_distance_um=1
    ).severity == "soft"


def test_severity_can_be_overridden_per_instance():
    from silk.constraints.types import SameTemplate
    c = SameTemplate(instances=["A", "B"], severity="hard")
    assert c.severity == "hard"
