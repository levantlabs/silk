import pytest
from silk.lint.exclusions import ScopeExclusion, ExclusionFile


def test_exclusion_requires_reason():
    with pytest.raises(ValueError, match="reason"):
        ScopeExclusion(target_type="instance", target_path="ldo.sram", reason="")


def test_exclusion_blank_reason_rejected():
    with pytest.raises(ValueError, match="reason"):
        ScopeExclusion(target_type="instance", target_path="ldo.sram", reason="   ")


def test_exclusion_matches_instance_by_path():
    e = ScopeExclusion(
        target_type="instance",
        target_path="ldo.sram_macro",
        reason="vendor IP",
    )
    assert e.matches_instance_path("ldo.sram_macro") is True
    assert e.matches_instance_path("ldo.opamp") is False


def test_exclusion_matches_instance_subtree():
    e = ScopeExclusion(
        target_type="instance",
        target_path="ldo.sram_macro",
        reason="vendor IP",
    )
    assert e.matches_instance_path("ldo.sram_macro.sense_amp") is True
    assert e.matches_instance_path("ldo.sram_macro.bank.cell") is True


def test_exclusion_module_target_does_not_match_instance():
    """target_type='module' should not match via matches_instance_path."""
    e = ScopeExclusion(
        target_type="module",
        target_path="some_module",
        reason="not used by instance check",
    )
    assert e.matches_instance_path("top.some_module") is False


def test_exclusion_file_loads(tmp_path):
    p = tmp_path / "exclusions.yaml"
    p.write_text(
        "exclusions:\n"
        "  - target:\n"
        "      type: instance\n"
        "      path: ldo.sram_macro\n"
        "    reason: Vendor SRAM, black box\n"
        "    owner: SRAM_vendor_v3.1\n"
    )
    ef = ExclusionFile.load(p)
    assert len(ef.exclusions) == 1
    assert ef.exclusions[0].owner == "SRAM_vendor_v3.1"
    assert ef.exclusions[0].target_path == "ldo.sram_macro"


def test_exclusion_file_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    ef = ExclusionFile.load(p)
    assert ef.exclusions == []
