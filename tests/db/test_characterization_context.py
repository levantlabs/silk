"""Unit tests for silk.db.context — the TASK-3 (DB-02-CONTEXT) runtime
characterization-context descriptor.

Covers the §2.1 field contract (owner_path/test_name/parameters/corner/
temperature/voltage) and the §2.2 all-or-nothing factory. Persistence (the
profile digest, DDL, storage) is out of scope — that is TASK-5/6/7.
"""
import copy
import math
import pickle
from dataclasses import FrozenInstanceError

import pytest

from silk.db.context import (
    RESERVED_PVT_PARAMETER_NAMES,
    CharacterizationContext,
    build_characterization_context,
    freeze_parameters,
)
from silk.exceptions import CharacterizationContextError
from silk.simulation.corner import Corner


def _full_kwargs(**overrides):
    kwargs = dict(
        owner_path='$/amp',
        test_name='load_step',
        parameters={'cload_f': 2e-12},
        corner='tt',
        temperature=27.0,
        voltage=1.8,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# CharacterizationContext construction / validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_valid_construction(self):
        ctx = CharacterizationContext(**_full_kwargs())
        assert ctx.owner_path == '$/amp'
        assert ctx.test_name == 'load_step'
        assert ctx.parameters == {'cload_f': 2e-12}
        assert ctx.corner == 'tt'
        assert ctx.temperature == 27.0
        assert ctx.voltage == 1.8

    def test_empty_parameters_valid(self):
        ctx = CharacterizationContext(**_full_kwargs(parameters={}))
        assert ctx.parameters == {}

    def test_corner_enum_reduced_to_string_value(self):
        ctx = CharacterizationContext(**_full_kwargs(corner=Corner.FF))
        assert ctx.corner == 'ff'
        assert isinstance(ctx.corner, str)
        # Reduced to the exact plain str value, not left as an Enum member
        # (Corner IS a str subclass, so this also checks it isn't the enum).
        assert type(ctx.corner) is str

    def test_temperature_int_normalized_to_float(self):
        ctx = CharacterizationContext(**_full_kwargs(temperature=27))
        assert ctx.temperature == 27.0
        assert isinstance(ctx.temperature, float)

    def test_zero_temperature_and_voltage_are_valid(self):
        ctx = CharacterizationContext(**_full_kwargs(temperature=0, voltage=0))
        assert ctx.temperature == 0.0
        assert ctx.voltage == 0.0

    def test_negative_zero_normalized_to_positive_zero(self):
        """Design §2.3: PVT selection compares exact stored numeric values
        with no tolerance. -0.0 == 0.0 under Python's normal equality, but
        math.copysign distinguishes them — an unnormalized negative zero
        could silently mint a second selection key for the same nominal
        point. Both temperature and voltage must always store +0.0."""
        ctx = CharacterizationContext(**_full_kwargs(temperature=-0.0, voltage=-0.0))
        assert math.copysign(1.0, ctx.temperature) == 1.0
        assert math.copysign(1.0, ctx.voltage) == 1.0

    @pytest.mark.parametrize('field_name', ['owner_path', 'test_name'])
    def test_empty_string_rejected(self, field_name):
        with pytest.raises(CharacterizationContextError):
            CharacterizationContext(**_full_kwargs(**{field_name: ''}))

    @pytest.mark.parametrize('field_name', ['owner_path', 'test_name'])
    def test_non_string_rejected(self, field_name):
        with pytest.raises(CharacterizationContextError):
            CharacterizationContext(**_full_kwargs(**{field_name: 123}))

    def test_empty_corner_rejected(self):
        with pytest.raises(CharacterizationContextError):
            CharacterizationContext(**_full_kwargs(corner=''))

    @pytest.mark.parametrize('field_name', ['temperature', 'voltage'])
    def test_bool_rejected_for_real_fields(self, field_name):
        with pytest.raises(CharacterizationContextError):
            CharacterizationContext(**_full_kwargs(**{field_name: True}))

    @pytest.mark.parametrize('field_name', ['temperature', 'voltage'])
    def test_non_numeric_rejected(self, field_name):
        with pytest.raises(CharacterizationContextError):
            CharacterizationContext(**_full_kwargs(**{field_name: 'hot'}))

    @pytest.mark.parametrize('bad', [math.inf, -math.inf, math.nan])
    @pytest.mark.parametrize('field_name', ['temperature', 'voltage'])
    def test_non_finite_rejected(self, field_name, bad):
        with pytest.raises(CharacterizationContextError):
            CharacterizationContext(**_full_kwargs(**{field_name: bad}))

    def test_frozen_rejects_attribute_assignment(self):
        ctx = CharacterizationContext(**_full_kwargs())
        with pytest.raises(FrozenInstanceError):
            ctx.corner = 'ff'


# ---------------------------------------------------------------------------
# Declared-parameter validation (freeze_parameters)
# ---------------------------------------------------------------------------

class TestFreezeParameters:
    def test_non_mapping_rejected(self):
        with pytest.raises(CharacterizationContextError):
            freeze_parameters(['not', 'a', 'mapping'])

    def test_non_string_key_rejected(self):
        with pytest.raises(CharacterizationContextError):
            freeze_parameters({1: 'a'})

    def test_empty_string_key_rejected(self):
        with pytest.raises(CharacterizationContextError):
            freeze_parameters({'': 'a'})

    @pytest.mark.parametrize('reserved', sorted(RESERVED_PVT_PARAMETER_NAMES))
    def test_reserved_pvt_names_rejected(self, reserved):
        with pytest.raises(CharacterizationContextError):
            freeze_parameters({reserved: 1})

    def test_reserved_names_are_exactly_the_documented_three(self):
        assert RESERVED_PVT_PARAMETER_NAMES == {
            'process_corner', 'temperature_c', 'supply_voltage_v',
        }

    def test_snapshot_is_independent_of_caller_mutation(self):
        original = {'cload_f': 2e-12, 'trims': [1, 2, 3]}
        snapshot = freeze_parameters(original)
        original['cload_f'] = 999.0
        original['trims'].append(4)
        assert snapshot == {'cload_f': 2e-12, 'trims': [1, 2, 3]}

    def test_context_parameters_are_snapshotted_too(self):
        original = {'cload_f': 2e-12}
        ctx = CharacterizationContext(**_full_kwargs(parameters=original))
        original['cload_f'] = 999.0
        assert ctx.parameters == {'cload_f': 2e-12}


# ---------------------------------------------------------------------------
# build_characterization_context — all-or-nothing factory
# ---------------------------------------------------------------------------

class TestBuildCharacterizationContext:
    def test_all_none_returns_none(self):
        assert build_characterization_context() is None

    def test_all_six_supplied_builds_context(self):
        ctx = build_characterization_context(**_full_kwargs())
        assert isinstance(ctx, CharacterizationContext)
        assert ctx.owner_path == '$/amp'

    def test_empty_parameters_counts_as_supplied_not_missing(self):
        ctx = build_characterization_context(**_full_kwargs(parameters={}))
        assert ctx is not None
        assert ctx.parameters == {}

    def test_zero_pvt_counts_as_supplied_not_missing(self):
        ctx = build_characterization_context(**_full_kwargs(temperature=0, voltage=0))
        assert ctx is not None

    @pytest.mark.parametrize('missing_field', [
        'owner_path', 'test_name', 'parameters', 'corner', 'temperature', 'voltage',
    ])
    def test_single_missing_field_raises_and_names_it(self, missing_field):
        kwargs = _full_kwargs()
        kwargs[missing_field] = None
        with pytest.raises(CharacterizationContextError, match=missing_field):
            build_characterization_context(**kwargs)

    def test_only_temperature_given_raises(self):
        with pytest.raises(CharacterizationContextError):
            build_characterization_context(temperature=27.0)

    def test_multiple_missing_fields_all_named(self):
        with pytest.raises(CharacterizationContextError) as excinfo:
            build_characterization_context(owner_path='$', test_name='t')
        message = str(excinfo.value)
        assert 'parameters' in message
        assert 'corner' in message
        assert 'temperature' in message
        assert 'voltage' in message

    def test_never_infers_owner_path_or_test_name(self):
        """Omitting owner_path/test_name never falls back to a default like
        '$' or 'default_test' — it is either fully unscoped (all six absent)
        or a loud error, exactly like every other missing field."""
        with pytest.raises(CharacterizationContextError):
            build_characterization_context(
                test_name='t', parameters={}, corner='tt',
                temperature=27.0, voltage=1.8,
            )


# ---------------------------------------------------------------------------
# Pickle / deepcopy safety — CharacterizationContext rides on Module.__dict__,
# which Module.__reduce_ex__ pickles wholesale and __deepcopy__ deep-copies
# per-attribute (silk.faults.FaultCampaign forks via pickle.dumps(module)).
# ---------------------------------------------------------------------------

class TestPickleAndDeepcopySafety:
    def test_pickle_round_trip(self):
        ctx = CharacterizationContext(**_full_kwargs())
        restored = pickle.loads(pickle.dumps(ctx))
        assert restored == ctx

    def test_deepcopy_round_trip(self):
        ctx = CharacterizationContext(**_full_kwargs())
        restored = copy.deepcopy(ctx)
        assert restored == ctx
        assert restored.parameters == ctx.parameters
