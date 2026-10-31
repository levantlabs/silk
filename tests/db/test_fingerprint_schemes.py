"""Frozen contracts for versioned structure and specification identities."""

from dataclasses import FrozenInstanceError
import copy
import json
from pathlib import Path

import pytest

import silk.db.identity as identity
from silk.db.models import Fingerprint, SnapshotIdentity
from silk.exceptions import (
    CircuitError,
    DuplicateNameError,
    UnsupportedFingerprintSchemeError,
)
from silk.specs import Measurement, Spec


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "db_fingerprints_v1.json"


def _fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class _ElementManager:
    def __init__(self):
        self.elem_dict = {}


class _SpecOwner:
    """Small real-shape owner for exercising the identity walker."""

    def __init__(self):
        self._specs = []
        self._element_mgr = _ElementManager()
        self.module_dict = {}

    def add_instance(self, name, child):
        self._element_mgr.elem_dict[name] = {"type": "m", "device": child}


def _add_spec(
    owner,
    *,
    name="Gain",
    key="gain",
    metric="dc_gain",
    op=">=",
    target=60.0,
    units="dB",
    description="",
    parent=None,
    calibration=None,
    declared_at="",
):
    spec = Spec(
        name=name,
        module=owner,
        metric=metric,
        op=op,
        target=target,
        units=units,
        description=description,
        parent=parent,
        calibration=calibration,
        declared_at=declared_at,
        key=key,
    )
    owner._specs.append(spec)
    return spec


def _spec_fingerprints(root):
    return identity._fingerprint_spec_set(root)


def _definition_for(records, owner_path, key):
    return next(
        record for record in records
        if record.owner_path == owner_path and record.key == key
    )


def test_current_fingerprint_scheme_names_are_exact():
    assert identity.STRUCTURE_FINGERPRINT_SCHEME == "silk.structure/v4"
    assert identity.SPEC_FINGERPRINT_SCHEME == "silk.spec/v1"


def test_fingerprint_is_an_immutable_scheme_and_digest_pair():
    fingerprint = Fingerprint(scheme="silk.structure/v4", digest="abc123")

    assert fingerprint.scheme == "silk.structure/v4"
    assert fingerprint.digest == "abc123"
    with pytest.raises(FrozenInstanceError):
        fingerprint.digest = "changed"


def test_snapshot_identity_keeps_structure_and_spec_set_separate():
    structure = Fingerprint("silk.structure/v4", "structure-digest")
    spec_set = Fingerprint("silk.spec/v1", "spec-set-digest")
    snapshot = SnapshotIdentity(structure=structure, spec_set=spec_set)

    assert snapshot.structure is structure
    assert snapshot.spec_set is spec_set
    with pytest.raises(FrozenInstanceError):
        snapshot.structure = spec_set


def test_v4_flat_module_preserves_v1_digest_bytes_without_mutation():
    case = _fixture()["module"]
    module_structure = copy.deepcopy(case["input"])

    fingerprint = identity._fingerprint_module_structure(module_structure)

    assert fingerprint == identity.Fingerprint(
        scheme="silk.structure/v4",
        digest=case["digest"],
    )
    assert module_structure == case["input"]


def test_v4_flat_structure_preserves_v1_digest_bytes_without_mutation():
    case = _fixture()["structure"]
    structure = copy.deepcopy(case["input"])

    fingerprint, module_fingerprints = identity._fingerprint_structure(structure)

    assert fingerprint == identity.Fingerprint(
        scheme="silk.structure/v4",
        digest=case["digest"],
    )
    assert module_fingerprints == {
        path: identity.Fingerprint("silk.structure/v4", digest)
        for path, digest in case["module_digests"].items()
    }
    assert structure == case["input"]


def test_v1_empty_spec_set_matches_frozen_literal():
    fingerprint, definitions = _spec_fingerprints(_SpecOwner())

    assert fingerprint == identity.Fingerprint(
        "silk.spec/v1", _fixture()["spec"]["empty_set_digest"]
    )
    assert definitions == ()


def test_v1_spec_definition_and_set_match_frozen_canonical_fixture():
    fixture = _fixture()["spec"]
    root = _SpecOwner()
    child = _SpecOwner()
    root.add_instance("amp/top~rev", child)
    _add_spec(child, name="Display gain", key="gain/μ")

    set_fingerprint, definitions = _spec_fingerprints(root)

    assert len(definitions) == 1
    definition = definitions[0]
    assert definition.owner_path == fixture["definition"]["payload"]["owner_path"]
    assert definition.key == fixture["definition"]["payload"]["key"]
    assert definition.fingerprint == identity.Fingerprint(
        "silk.spec/v1", fixture["definition"]["digest"]
    )
    assert set_fingerprint == identity.Fingerprint(
        "silk.spec/v1", fixture["set"]["digest"]
    )


def test_spec_set_is_deterministic_across_rebuild_and_declaration_order():
    def build(order):
        root = _SpecOwner()
        for key in order:
            _add_spec(root, name=key.upper(), key=key, target=len(key))
        return root

    first_set, first_definitions = _spec_fingerprints(build(["zeta", "alpha"]))
    second_set, second_definitions = _spec_fingerprints(build(["alpha", "zeta"]))

    assert first_set == second_set
    assert [record.key for record in first_definitions] == ["alpha", "zeta"]
    assert first_definitions == second_definitions


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("target", 61.0),
        ("op", ">"),
        ("metric", "ac_gain"),
        ("units", "V/V"),
    ],
)
def test_definition_fields_change_spec_identity(field, changed):
    base = _SpecOwner()
    _add_spec(base)
    variant = _SpecOwner()
    _add_spec(variant, **{field: changed})

    base_set, base_definitions = _spec_fingerprints(base)
    variant_set, variant_definitions = _spec_fingerprints(variant)

    assert base_definitions[0].fingerprint != variant_definitions[0].fingerprint
    assert base_set != variant_set


def test_parent_reference_changes_child_definition_identity():
    def build(parent_key):
        root = _SpecOwner()
        child = _SpecOwner()
        root.add_instance("amp", child)
        parent_a = _add_spec(root, name="Budget A", key="budget_a")
        parent_b = _add_spec(root, name="Budget B", key="budget_b")
        selected = parent_a if parent_key == "budget_a" else parent_b
        _add_spec(child, key="child_gain", parent=selected)
        return root

    _first_set, first = _spec_fingerprints(build("budget_a"))
    _second_set, second = _spec_fingerprints(build("budget_b"))

    first_child = _definition_for(first, "$/amp", "child_gain")
    second_child = _definition_for(second, "$/amp", "child_gain")
    assert first_child.fingerprint != second_child.fingerprint


def test_display_and_provenance_fields_do_not_change_explicit_key_identity():
    base = _SpecOwner()
    _add_spec(base, name="Gain", key="stable_gain")
    variant = _SpecOwner()
    changed = _add_spec(
        variant,
        name="Readable DC gain target",
        key="stable_gain",
        description="clarified for release",
        declared_at="new_design.py:99",
        calibration=object(),
    )
    changed.record_measurement(Measurement(source="spice", value=62.0))

    base_set, base_definitions = _spec_fingerprints(base)
    variant_set, variant_definitions = _spec_fingerprints(variant)

    assert base_definitions[0].fingerprint == variant_definitions[0].fingerprint
    assert base_set == variant_set


def test_default_key_display_rename_changes_identity():
    first = _SpecOwner()
    _add_spec(first, name="Gain", key=None)
    second = _SpecOwner()
    _add_spec(second, name="DC Gain", key=None)

    first_set, first_definitions = _spec_fingerprints(first)
    second_set, second_definitions = _spec_fingerprints(second)

    assert first_definitions[0].fingerprint != second_definitions[0].fingerprint
    assert first_set != second_set


def test_hierarchy_instance_path_changes_identity():
    def build(instance_name):
        root = _SpecOwner()
        child = _SpecOwner()
        root.add_instance(instance_name, child)
        _add_spec(child, key="gain")
        return root

    first_set, first = _spec_fingerprints(build("amp1"))
    second_set, second = _spec_fingerprints(build("amp2"))

    assert first[0].owner_path == "$/amp1"
    assert second[0].owner_path == "$/amp2"
    assert first[0].fingerprint != second[0].fingerprint
    assert first_set != second_set


def test_walker_ignores_definition_only_module_dict_entries():
    root = _SpecOwner()
    live = _SpecOwner()
    template = _SpecOwner()
    root.add_instance("live", live)
    root.module_dict["template"] = {"device": template}
    _add_spec(live, key="live_spec")
    _add_spec(template, key="template_spec")

    _set_fingerprint, definitions = _spec_fingerprints(root)

    assert [(record.owner_path, record.key) for record in definitions] == [
        ("$/live", "live_spec")
    ]


def test_walker_rejects_duplicate_local_keys_even_if_objects_were_injected():
    root = _SpecOwner()
    _add_spec(root, name="First", key="same")
    _add_spec(root, name="Second", key="same")

    with pytest.raises(DuplicateNameError, match="same"):
        _spec_fingerprints(root)


def test_walker_rejects_one_spec_object_aliased_across_owners():
    root = _SpecOwner()
    child = _SpecOwner()
    root.add_instance("child", child)
    spec = _add_spec(root, key="aliased")
    child._specs.append(spec)

    with pytest.raises(CircuitError, match="same Spec object"):
        _spec_fingerprints(root)


def test_walker_rejects_parent_outside_the_live_instance_tree():
    root = _SpecOwner()
    external = _SpecOwner()
    parent = _add_spec(external, key="external")
    _add_spec(root, key="child", parent=parent)

    with pytest.raises(CircuitError, match="parent.*live instance tree"):
        _spec_fingerprints(root)


def test_unknown_fingerprint_scheme_raises_before_digest_comparison():
    unknown = identity.Fingerprint("silk.structure/v999", "same-digest")
    known = identity.Fingerprint("silk.structure/v4", "same-digest")

    with pytest.raises(UnsupportedFingerprintSchemeError, match="v999"):
        identity._fingerprints_equal(
            unknown,
            known,
            expected_scheme=identity.STRUCTURE_FINGERPRINT_SCHEME,
        )


def test_supported_fingerprint_comparison_uses_digest_after_scheme_validation():
    first = identity.Fingerprint("silk.structure/v4", "first")
    same = identity.Fingerprint("silk.structure/v4", "first")
    changed = identity.Fingerprint("silk.structure/v4", "changed")

    assert identity._fingerprints_equal(
        first, same, expected_scheme=identity.STRUCTURE_FINGERPRINT_SCHEME
    )
    assert not identity._fingerprints_equal(
        first, changed, expected_scheme=identity.STRUCTURE_FINGERPRINT_SCHEME
    )


@pytest.mark.parametrize('historical_scheme', ['silk.structure/v1', 'silk.structure/v2', 'silk.structure/v3'])
def test_historical_pair_cannot_compare_as_v4_even_when_digest_is_unchanged(historical_scheme):
    case = _fixture()["structure"]
    legacy = Fingerprint(historical_scheme, case["digest"])
    current, _ = identity._fingerprint_structure(case["input"])
    assert legacy.scheme == historical_scheme
    assert current.digest == legacy.digest
    with pytest.raises(UnsupportedFingerprintSchemeError, match="rebuild"):
        identity._fingerprints_equal(
            legacy, current, expected_scheme=identity.STRUCTURE_FINGERPRINT_SCHEME,
        )
