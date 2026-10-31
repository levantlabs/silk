"""Canonical characterization-profile identity and persistence (TASK-5)."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
import math
from uuid import UUID

import pytest

import silk.db as silk_db
from silk.db import identity
from silk.db import (
    Database,
    DatabaseError,
    EvidenceMismatchError,
    EvidenceNotFoundError,
    LockedVersionError,
    UnsupportedFingerprintSchemeError,
)
from silk.db.context import CharacterizationContext
from silk.exceptions import CharacterizationContextError
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
)


def _context(*, parameters=None, **overrides):
    kwargs = {
        "owner_path": "$/amp",
        "test_name": "load_step",
        "parameters": (
            {"cload_f": 2e-12, "enabled": True, "cycles": 4}
            if parameters is None
            else parameters
        ),
        "corner": "tt",
        "temperature": 27.0,
        "voltage": 1.8,
    }
    kwargs.update(overrides)
    return CharacterizationContext(**kwargs)


def _canonicalize(context):
    canonicalize = getattr(
        identity, "_canonicalize_characterization_profile", None
    )
    assert canonicalize is not None, (
        "canonical characterization-profile derivation is not implemented"
    )
    return canonicalize(context)


def _store_profile(db, version_id, context, **kwargs):
    store = getattr(db, "store_characterization_profile", None)
    assert store is not None, (
        "characterization-profile persistence is not implemented"
    )
    return store(version_id, context, **kwargs)


def _get_profile(db, version_id, **kwargs):
    get = getattr(db, "get_characterization_profile", None)
    assert get is not None, (
        "characterization-profile readback is not implemented"
    )
    return get(version_id, **kwargs)


class TestCanonicalProfileIdentity:
    def test_frozen_design_example_has_exact_json_and_digest(self):
        canonical = _canonicalize(_context())

        assert canonical.payload_json == (
            b'{"owner_path":"$/amp","parameters":['
            b'{"name":"cload_f","value":{"type":"float","value":'
            b'"0x1.19799812dea11p-39"}},'
            b'{"name":"cycles","value":{"type":"int","value":"4"}},'
            b'{"name":"enabled","value":{"type":"bool","value":true}}],'
            b'"test_name":"load_step"}'
        )
        assert canonical.parameters_json == (
            '[{"name":"cload_f","value":{"type":"float","value":'
            '"0x1.19799812dea11p-39"}},'
            '{"name":"cycles","value":{"type":"int","value":"4"}},'
            '{"name":"enabled","value":{"type":"bool","value":true}}]'
        )
        assert canonical.fingerprint.scheme == "silk.profile/v1"
        assert canonical.fingerprint.digest == (
            "fd6adae669e7684e04ab70830df22a438a58119978c6ffc3fd4f75674c74ba0a"
        )

    def test_input_mapping_order_does_not_change_identity(self):
        left = _canonicalize(
            _context(parameters={"cload_f": 2e-12, "enabled": True, "cycles": 4})
        )
        right = _canonicalize(
            _context(parameters={"cycles": 4, "enabled": True, "cload_f": 2e-12})
        )

        assert right.payload_json == left.payload_json
        assert right.fingerprint == left.fingerprint

    def test_pvt_is_excluded_from_profile_identity(self):
        nominal = _canonicalize(_context())
        hot_slow = _canonicalize(
            _context(corner="ss", temperature=125.0, voltage=1.62)
        )

        assert hot_slow.payload_json == nominal.payload_json
        assert hot_slow.fingerprint == nominal.fingerprint

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ({"value": 1}, {"value": 1.0}),
            ({"value": False}, {"value": 0}),
            ({"value": None}, {}),
            ({"value": [1, 2]}, {"value": [2, 1]}),
        ],
    )
    def test_declared_type_or_value_changes_identity(self, left, right):
        assert (
            _canonicalize(_context(parameters=left)).fingerprint
            != _canonicalize(_context(parameters=right)).fingerprint
        )

    def test_list_and_tuple_intentionally_share_identity(self):
        list_form = _canonicalize(_context(parameters={"value": [1, 2]}))
        tuple_form = _canonicalize(_context(parameters={"value": (1, 2)}))

        assert tuple_form.payload_json == list_form.payload_json
        assert tuple_form.fingerprint == list_form.fingerprint

    def test_recursive_builtin_value_set_is_canonicalized(self):
        canonical = _canonicalize(
            _context(
                parameters={
                    "none": None,
                    "bool": False,
                    "int": -(10**40),
                    "float": -0.0,
                    "str": "Ω",
                    "sequence": [1, (2,)],
                    "mapping": {"z": True, "a": "first"},
                }
            )
        )

        assert '"value":"-10000000000000000000000000000000000000000"' in (
            canonical.parameters_json
        )
        assert '"value":"-0x0.0p+0"' in canonical.parameters_json
        assert '"value":"Ω"' in canonical.parameters_json
        assert canonical.parameters_json.index('"name":"a"') < (
            canonical.parameters_json.index('"name":"z"')
        )

    def test_arbitrary_precision_int_exceeds_interpreter_decimal_limit(self):
        huge = 10**5000
        expected_digits = "1" + ("0" * 5000)

        try:
            canonical = _canonicalize(_context(parameters={"huge": huge}))
        except ValueError as exc:
            pytest.fail(f"arbitrary-precision integer was rejected: {exc}")

        assert (
            f'"type":"int","value":"{expected_digits}"'
            in canonical.parameters_json
        )

    @pytest.mark.parametrize(
        "bad_value",
        [
            math.nan,
            math.inf,
            -math.inf,
            Decimal("1.0"),
            b"bytes",
            {1, 2},
            object(),
        ],
        ids=["nan", "inf", "negative-inf", "decimal", "bytes", "set", "object"],
    )
    def test_unsupported_parameter_values_fail_loudly(self, bad_value):
        with pytest.raises(CharacterizationContextError, match="parameters.bad"):
            _canonicalize(_context(parameters={"bad": bad_value}))

    def test_nested_mapping_requires_exact_string_keys(self):
        with pytest.raises(CharacterizationContextError, match="parameters.mapping"):
            _canonicalize(_context(parameters={"mapping": {1: "bad"}}))

    @pytest.mark.parametrize("container_type", [list, dict])
    def test_recursive_parameter_cycles_fail_loudly(self, container_type):
        if container_type is list:
            value = []
            value.append(value)
        else:
            value = {}
            value["self"] = value

        with pytest.raises(CharacterizationContextError, match="cycle"):
            _canonicalize(_context(parameters={"cyclic": value}))

    @pytest.mark.parametrize(
        ("overrides", "parameters"),
        [
            ({"owner_path": "$/amp2"}, None),
            ({"test_name": "startup"}, None),
            ({}, {"cload_f": 3e-12, "enabled": True, "cycles": 4}),
        ],
    )
    def test_each_identity_fact_changes_the_digest(self, overrides, parameters):
        baseline = _canonicalize(_context()).fingerprint
        changed = _canonicalize(
            _context(parameters=parameters, **overrides)
        ).fingerprint

        assert changed != baseline

class TestProfilePersistence:
    def test_store_returns_immutable_typed_record_with_stable_fk_endpoint(self):
        context = _context(
            parameters={
                "flat": "value",
                "nested": {"sequence": [1, 2], "enabled": True},
            }
        )
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)

            record = _store_profile(db, version_id, context)
            sim_id = db.add_simulation_result(
                version_id,
                "gain",
                value=42.0,
                corner=context.corner,
                temperature=context.temperature,
                voltage=context.voltage,
            )
            db.conn.execute(
                "UPDATE simulation_results SET profile_id = ? WHERE sim_id = ?",
                (record.profile_id, sim_id),
            )
            db.conn.commit()
            linked_profile_id = db.conn.execute(
                "SELECT profile_id FROM simulation_results WHERE sim_id = ?",
                (sim_id,),
            ).fetchone()[0]

        record_type = getattr(silk_db, "CharacterizationProfileRecord", None)
        assert record_type is not None
        assert isinstance(record, record_type)
        assert UUID(record.profile_id).version == 4
        assert record.origin_version_id == version_id
        assert record.owner_path == "$/amp"
        assert record.test_name == "load_step"
        assert record.parameters == {
            "flat": "value",
            "nested": {"enabled": True, "sequence": (1, 2)},
        }
        assert record.profile_scheme == "silk.profile/v1"
        assert len(record.profile_hash) == 64
        assert linked_profile_id == record.profile_id
        with pytest.raises(FrozenInstanceError):
            record.test_name = "renamed"
        with pytest.raises(TypeError):
            record.parameters["flat"] = "changed"
        with pytest.raises(TypeError):
            record.parameters["nested"]["enabled"] = False

    def test_storing_same_derived_identity_is_idempotent_across_pvt(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            first = _store_profile(db, version_id, _context())
            second = _store_profile(db,
                version_id,
                _context(
                    parameters={
                        "cycles": 4,
                        "cload_f": 2e-12,
                        "enabled": True,
                    },
                    corner="ss",
                    temperature=125.0,
                    voltage=1.62,
                ),
            )
            count = db.conn.execute(
                "SELECT COUNT(*) FROM characterization_profiles"
            ).fetchone()[0]

        assert second == first
        assert second.profile_id == first.profile_id
        assert count == 1

    def test_arbitrary_precision_int_round_trips_profile_storage(self):
        huge = -(10**5000)
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)

            try:
                record = _store_profile(
                    db, version_id, _context(parameters={"huge": huge})
                )
            except ValueError as exc:
                pytest.fail(f"stored arbitrary-precision int was rejected: {exc}")

        assert record.parameters["huge"] == huge

    def test_changed_identity_facts_mint_distinct_profiles(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            original = _store_profile(db, version_id, _context())
            changed = _store_profile(db,
                version_id, _context(test_name="startup")
            )

            assert changed.profile_id != original.profile_id
            assert changed.profile_hash != original.profile_hash
            assert db.conn.execute(
                "SELECT COUNT(*) FROM characterization_profiles"
            ).fetchone()[0] == 2

    def test_lookup_accepts_exact_id_or_complete_descriptor(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            stored = _store_profile(db, version_id, _context())

            by_id = _get_profile(db,
                version_id, profile_id=stored.profile_id
            )
            by_descriptor = _get_profile(db,
                version_id,
                context=_context(corner="ff", temperature=-40.0, voltage=1.95),
            )

        assert by_id == stored
        assert by_descriptor == stored

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"profile_id": "id", "context": _context()},
        ],
    )
    def test_lookup_requires_exactly_one_complete_selector(self, kwargs):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            with pytest.raises(ValueError, match="exactly one"):
                _get_profile(db, version_id, **kwargs)

    def test_missing_profile_is_a_typed_not_found_error(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            with pytest.raises(EvidenceNotFoundError, match="profile"):
                _get_profile(db,
                    version_id,
                    profile_id="00000000-0000-4000-8000-000000000000",
                )

    def test_public_apis_reject_free_standing_label_and_raw_digest(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            with pytest.raises(TypeError, match="label"):
                _store_profile(db,
                    version_id, _context(), label="nominal"
                )
            with pytest.raises(TypeError, match="profile_hash"):
                _store_profile(db,
                    version_id, _context(), profile_hash="0" * 64
                )
            with pytest.raises(TypeError, match="label"):
                _get_profile(db, version_id, label="nominal")

    def test_profile_is_owned_by_one_origin_version(self):
        with Database(":memory:") as db:
            first_version = create_sealed_snapshot(db, name="amp")
            second_version = create_sealed_snapshot(db, name="amp2")

            first = _store_profile(db, first_version, _context())
            second = _store_profile(db, second_version, _context())

        assert second.profile_hash == first.profile_hash
        assert second.profile_id != first.profile_id
        assert first.origin_version_id == first_version
        assert second.origin_version_id == second_version

    def test_profile_store_requires_an_unlocked_sealed_snapshot(self):
        with Database(":memory:") as db:
            building_id = create_building_snapshot(db, version="building")
            with pytest.raises(DatabaseError, match="sealed"):
                _store_profile(db, building_id, _context())

            sealed_id = create_sealed_snapshot(db, version="sealed")
            db.lock_version(sealed_id)
            with pytest.raises(LockedVersionError, match="locked"):
                _store_profile(db, sealed_id, _context())

            assert db.conn.execute(
                "SELECT COUNT(*) FROM characterization_profiles"
            ).fetchone()[0] == 0

    def test_same_digest_with_conflicting_payload_is_rejected_loudly(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            stored = _store_profile(db, version_id, _context())
            db.conn.execute(
                "UPDATE characterization_profiles SET owner_path = ? "
                "WHERE profile_id = ?",
                ("$/corrupt", stored.profile_id),
            )
            db.conn.commit()

            with pytest.raises(EvidenceMismatchError, match="profile"):
                _store_profile(db, version_id, _context())

            assert db.conn.execute(
                "SELECT COUNT(*) FROM characterization_profiles"
            ).fetchone()[0] == 1

    def test_read_recomputes_digest_and_rejects_noncanonical_payload(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            stored = _store_profile(db, version_id, _context())
            db.conn.execute(
                "UPDATE characterization_profiles "
                "SET parameters_json = '[] ' WHERE profile_id = ?",
                (stored.profile_id,),
            )
            db.conn.commit()

            with pytest.raises(EvidenceMismatchError, match="canonical"):
                _get_profile(db,
                    version_id, profile_id=stored.profile_id
                )

    def test_unknown_stored_profile_scheme_fails_before_digest_comparison(self):
        with Database(":memory:") as db:
            version_id = create_sealed_snapshot(db)
            stored = _store_profile(db, version_id, _context())
            db.conn.execute(
                "UPDATE characterization_profiles SET profile_scheme = ? "
                "WHERE profile_id = ?",
                ("silk.profile/v999", stored.profile_id),
            )
            db.conn.commit()

            with pytest.raises(UnsupportedFingerprintSchemeError):
                _get_profile(db,
                    version_id, profile_id=stored.profile_id
                )
