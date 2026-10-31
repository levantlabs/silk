"""TASK-403: the operating-conditions vocabulary core (TASK-394 design §2, §4.1, §4.2)."""
import json
import math
import sqlite3
import sys

import pytest

import silk.simulation as sim
from silk.db.context import CharacterizationContext, build_characterization_context
from silk.db.identity import _canonicalize_characterization_profile
from silk.exceptions import CharacterizationContextError
from silk.simulation import (
    CONDITION_VOCABULARY, ConditionState, OperatingConditions, ConditionSweep,
    decode_condition, encode_condition, export_conditions_schema,
)
from silk.simulation.corner import Corner

NONE, SWEEP, FIXED, UNKNOWN = (ConditionState.NONE, ConditionState.SWEEP,
                               ConditionState.FIXED, ConditionState.UNKNOWN)


# --- ConditionState and the vocabulary -------------------------------------

def test_reexported_from_silk_simulation():
    for name in ('ConditionState', 'CONDITION_VOCABULARY', 'encode_condition',
                 'decode_condition', 'export_conditions_schema'):
        assert name in sim.__all__
        assert getattr(sim, name) is getattr(sim.conditions, name)


def test_plain_enum_never_equals_its_wire_word():
    assert not isinstance(NONE, str)
    assert [m.value for m in ConditionState] == ['fixed', 'none', 'unknown', 'sweep']
    assert NONE != 'none' and hash(NONE) != hash('none')
    assert ('spice', NONE) != ('spice', 'none')
    # Every way an unconverted member could leave silk fails loudly.
    with pytest.raises(TypeError):
        json.dumps(NONE)
    with pytest.raises(TypeError):
        float(NONE)
    # Python 3.11 reclassified unsupported parameter types; both reject binding.
    binding_error = (sqlite3.InterfaceError if sys.version_info < (3, 11)
                     else sqlite3.ProgrammingError)
    with pytest.raises(binding_error):
        sqlite3.connect(':memory:').execute('SELECT ?', (NONE,))


def test_vocabulary_states_and_profile_states():
    assert set(CONDITION_VOCABULARY) == {'corner', 'temperature', 'supply'}
    assert CONDITION_VOCABULARY['supply']['states'] == ('fixed', 'none', 'unknown', 'sweep')
    assert CONDITION_VOCABULARY['temperature']['states'] == ('fixed', 'none', 'unknown', 'sweep')
    for fact in ('corner',):
        assert CONDITION_VOCABULARY[fact]['states'] == ('fixed', 'none', 'unknown')
    for fact, entry in CONDITION_VOCABULARY.items():
        assert entry['profile_states'] == ('fixed', 'none')
        assert set(entry['bases']) == set(entry['states']), fact
        for words in entry['bases'].values():
            assert words and all(isinstance(w, str) and w for w in words)


def test_vocabulary_bases_match_design_table():
    bases = {fact: entry['bases'] for fact, entry in CONDITION_VOCABULARY.items()}
    assert {'configured-and-emitted', 'caller-declared'} <= set(bases['corner']['fixed'])
    assert set(bases['corner']['none']) == {
        'proven-no-pdk-device', 'proven-behavioral', 'caller-declared'}
    assert set(bases['temperature']['fixed']) == {
        'configured-and-observed', 'native-and-observed', 'raw-and-observed',
        'backend-observed', 'caller-declared'}
    assert set(bases['temperature']['none']) == {'proven-behavioral', 'caller-declared'}
    assert set(bases['temperature']['sweep']) == {'configured', 'unknown'}
    assert set(bases['supply']['fixed']) == {
        'configured-and-emitted', 'configured-and-driven',
        'declared-and-emitted', 'caller-declared'}
    assert bases['supply']['sweep'] == ('configured', 'unknown')
    assert set(bases['supply']['none']) == {'proven-no-supply', 'caller-declared'}
    for fact in bases:
        assert bases[fact]['unknown'] == ('unknown',)


def test_every_basis_word_has_exactly_one_known_category():
    schema = export_conditions_schema()
    allowed = {'observed', 'configured', 'declared', 'proven',
               'default', 'legacy', 'unknown'}
    assert set(schema['categories']) == allowed
    for fact, entry in schema['facts'].items():
        words = {w for ws in entry['bases'].values() for w in ws}
        assert set(entry['categories']) == words, fact
        assert set(entry['categories'].values()) <= allowed, fact


def test_export_conditions_schema_is_a_json_ready_copy():
    schema = export_conditions_schema()
    assert schema['format_version'] == '1.3'
    assert json.loads(json.dumps(schema)) == schema
    assert schema['wire'] == {'unknown': None,
                              'tagged': {'none': {'state': 'none'}, 'sweep': {'state': 'sweep'}}}
    for fact, entry in CONDITION_VOCABULARY.items():
        exported = schema['facts'][fact]
        assert exported['states'] == list(entry['states'])
        assert exported['profile_states'] == list(entry['profile_states'])
        assert exported['bases'] == {s: list(w) for s, w in entry['bases'].items()}
        assert exported['categories'] == entry['categories']
    schema['facts']['corner']['states'].append('mutated')
    assert 'mutated' not in export_conditions_schema()['facts']['corner']['states']
    assert 'mutated' not in CONDITION_VOCABULARY['corner']['states']


# --- encode / decode --------------------------------------------------------

_ALLOWED = {
    'corner': ['tt', 'none', NONE, None],  # a corner literally named 'none' is a value
    'temperature': [27.0, -40.0, 0.0, NONE, None, SWEEP],
    'supply': [1.8, 0.0, NONE, None, SWEEP],
}


@pytest.mark.parametrize('fact,scalar',
                         [(f, s) for f, values in _ALLOWED.items() for s in values])
def test_round_trip_every_allowed_state(fact, scalar):
    wire = encode_condition(scalar)
    assert json.loads(json.dumps(wire)) == wire
    decoded = decode_condition(fact, json.loads(json.dumps(wire)))
    assert decoded == scalar and type(decoded) is type(scalar)


def test_round_trip_covers_every_state_per_fact():
    def state(v):
        if isinstance(v, ConditionState):
            return v.value
        return 'unknown' if v is None else 'fixed'
    for fact, values in _ALLOWED.items():
        covered = {state(v) for v in values}
        assert covered == set(CONDITION_VOCABULARY[fact]['states']), fact


def test_encode_wire_forms():
    assert encode_condition(NONE) == {'state': 'none'}
    assert encode_condition(SWEEP) == {'state': 'sweep'}
    assert encode_condition(None) is None
    assert encode_condition('tt') == 'tt'
    assert encode_condition(85.0) == 85.0
    tt = encode_condition(Corner.TT)
    assert tt == 'tt' and type(tt) is str


@pytest.mark.parametrize('bad', [FIXED, UNKNOWN])
def test_encode_refuses_states_that_are_not_scalars(bad):
    with pytest.raises(ValueError):
        encode_condition(bad)


@pytest.mark.parametrize('bad', [True, object(), [1.0], {'state': 'none'}])
def test_encode_refuses_other_types(bad):
    with pytest.raises(TypeError):
        encode_condition(bad)


@pytest.mark.parametrize('fact,wire', [
    ('corner', {'state': 'sweep'}),         # corner cannot be swept
    ('corner', {'state': 'fixed'}),         # value and unknown are never tagged
    ('temperature', {'state': 'unknown'}),
    ('temperature', {'state': 'none', 'basis': 'caller-declared'}),
    ('corner', {}),
    ('corner', {'value': None, 'status': 'unavailable'}),
    ('corner', ''),
    ('corner', 1.0),
    ('temperature', 'none'),
    ('temperature', '27'),
    ('supply', True),
    ('supply', [1.8]),
])
def test_decode_refuses_anything_else(fact, wire):
    with pytest.raises(ValueError):
        decode_condition(fact, wire)


def test_decode_refuses_unknown_fact():
    with pytest.raises(ValueError, match='voltage'):
        decode_condition('voltage', 1.8)


def test_decode_widens_ints_to_float():
    assert decode_condition('temperature', 27) == 27.0
    assert type(decode_condition('supply', 2)) is float


# --- OperatingConditions -----------------------------------------------------

def test_positional_construction_keeps_its_meaning():
    snap = OperatingConditions('silk', 'ngspice', 'tt')
    assert snap.corner == 'tt' and snap.corner_kind == 'fixed'
    assert snap.corner_basis == 'unknown'  # inference fills the kind, never the basis
    assert snap.temperature_kind == 'unknown' and snap.supply_kind == 'unknown'
    assert snap.state('corner') is FIXED and snap.scalar('corner') == 'tt'
    assert snap.state('temperature') is UNKNOWN and snap.scalar('temperature') is None

    unknown = OperatingConditions('silk', 'ngspice', None)
    assert unknown.corner_kind == 'unknown' and unknown.scalar('corner') is None


def test_existing_full_positional_and_keyword_constructions():
    sweep = ConditionSweep(-40.0, 125.0, 5.0)
    positional = OperatingConditions('external', None, 'ss', 'fixed', 27.0,
                                     'caller-declared', None, 1.8, 'caller-declared')
    assert positional.temperature_c == 27.0 and positional.supply_voltage_v == 1.8
    assert positional.supply_kind == 'fixed' and positional.corner_kind == 'fixed'
    assert positional.scalar('supply') == 1.8 and positional.scalar('temperature') == 27.0

    swept = OperatingConditions('silk', 'ngspice', 'tt', 'sweep', temperature_sweep=sweep)
    assert swept.state('temperature') is SWEEP and swept.scalar('temperature') is SWEEP

    keyword = OperatingConditions('silk', 'ngspice', 'tt', 'fixed', 85.0, 'backend-observed',
                                  supply_voltage_v=3.3, supply_basis='configured-and-emitted',
                                  supply_source='vdd')
    assert keyword.supply_kind == 'fixed' and keyword.supply_source == 'vdd'


def test_none_kinds_hold_none_values():
    snap = OperatingConditions('behavioral', 'behavioral', None,
                               temperature_kind='none', temperature_basis='proven-behavioral',
                               corner_kind='none', corner_basis='proven-behavioral',
                               supply_kind='none', supply_basis='proven-no-supply')
    assert snap.corner is None and snap.temperature_c is None and snap.supply_voltage_v is None
    for fact in ('corner', 'temperature', 'supply'):
        assert snap.state(fact) is NONE
        assert snap.scalar(fact) is NONE


def test_kind_accepts_condition_state_members_and_stores_wire_words():
    snap = OperatingConditions('silk', 'ngspice', None, corner_kind=NONE,
                               temperature_kind=FIXED, temperature_c=27.0)
    assert snap.corner_kind == 'none' and snap.temperature_kind == 'fixed'
    assert json.loads(json.dumps(snap.to_dict()))['corner_kind'] == 'none'


def test_to_dict_gains_the_three_keys():
    data = OperatingConditions('silk', 'ngspice', 'tt').to_dict()
    assert data['corner_kind'] == 'fixed'
    assert data['corner_basis'] == 'unknown'
    assert data['supply_kind'] == 'unknown'
    assert '_FIELDS' not in data


@pytest.mark.parametrize('kwargs', [
    # value set iff kind fixed
    {'corner': None, 'corner_kind': 'fixed'},
    {'corner': 'tt', 'corner_kind': 'none'},
    {'corner': None, 'temperature_kind': 'fixed'},
    {'corner': None, 'temperature_kind': 'none', 'temperature_c': 27.0},
    {'corner': None, 'temperature_c': 27.0},  # temperature kind is never inferred
    {'corner': None, 'supply_kind': 'fixed'},
    {'corner': None, 'supply_kind': 'none', 'supply_voltage_v': 1.8},
    # sweep iff temperature_sweep
    {'corner': None, 'temperature_kind': 'sweep'},
    {'corner': None, 'temperature_sweep': ConditionSweep(0.0, 1.0, 1.0)},
    # states the fact does not allow
    {'corner': None, 'corner_kind': 'sweep'},
    {'corner': None, 'supply_kind': 'sweep'},
    {'corner': None, 'corner_kind': 'value'},
    # revision 3 A1: value fields never hold a ConditionState member
    {'corner': NONE},
    {'corner': None, 'corner_kind': 'none', 'temperature_kind': 'none',
     'temperature_c': NONE},
    {'corner': None, 'supply_voltage_v': NONE, 'supply_kind': 'none'},
    {'corner': None, 'temperature_kind': 'sweep', 'temperature_c': SWEEP},
])
def test_invalid_combinations_raise(kwargs):
    with pytest.raises(ValueError):
        OperatingConditions('silk', 'ngspice', **kwargs)


def test_state_and_scalar_refuse_unknown_fact():
    snap = OperatingConditions('silk', 'ngspice', 'tt')
    for method in (snap.state, snap.scalar):
        with pytest.raises(ValueError):
            method('voltage')


# --- CharacterizationContext -------------------------------------------------

def _context(**overrides):
    kwargs = dict(owner_path='$', test_name='dc_gain', parameters={},
                  corner='tt', temperature=27.0, voltage=1.8)
    kwargs.update(overrides)
    return CharacterizationContext(**kwargs)


@pytest.mark.parametrize('field', ['corner', 'temperature', 'voltage'])
def test_context_accepts_none_per_fact(field):
    ctx = _context(**{field: NONE})
    assert getattr(ctx, field) is NONE


def test_context_all_none_and_corner_none_is_not_the_corner_named_none():
    ctx = _context(corner=NONE, temperature=NONE, voltage=NONE)
    assert (ctx.corner, ctx.temperature, ctx.voltage) == (NONE, NONE, NONE)
    assert _context(corner='none').corner == 'none'
    assert _context(corner='none') != _context(corner=NONE)
    assert _context(corner=Corner.TT).corner == 'tt'


@pytest.mark.parametrize('field', ['corner', 'temperature', 'voltage'])
@pytest.mark.parametrize('member,why', [(UNKNOWN, 'unknown cannot form a profile'),
                                        (SWEEP, 'sweep cannot form a profile'),
                                        (FIXED, 'not a value')])
def test_context_refuses_other_members_with_reason(field, member, why):
    with pytest.raises(CharacterizationContextError, match=why):
        _context(**{field: member})
    with pytest.raises(CharacterizationContextError, match=why):
        build_characterization_context(**{**dict(owner_path='$', test_name='t', parameters={},
                                                 corner='tt', temperature=27.0, voltage=1.8),
                                          field: member})


def test_build_counts_none_as_given():
    ctx = build_characterization_context(owner_path='$', test_name='t', parameters={},
                                         corner=NONE, temperature=85.0, voltage=NONE)
    assert ctx is not None and ctx.corner is NONE and ctx.voltage is NONE
    assert ctx.temperature == 85.0
    # None is still "not given": all-or-nothing is unchanged.
    with pytest.raises(CharacterizationContextError, match='voltage'):
        build_characterization_context(owner_path='$', test_name='t', parameters={},
                                       corner=NONE, temperature=85.0, voltage=None)
    assert build_characterization_context() is None


def test_context_existing_value_validation_unchanged():
    assert _context(temperature=-0.0).temperature == 0.0
    assert math.copysign(1.0, _context(voltage=-0.0).voltage) == 1.0
    for bad in (True, float('nan'), 'tt'):
        with pytest.raises(CharacterizationContextError):
            _context(voltage=bad)
    with pytest.raises(CharacterizationContextError):
        _context(corner='')


# Digests computed on the pre-TASK-403 tree (a5adf36f) and frozen here: the
# silk.profile/v1 hash of every existing fixed context must not move, and a
# none-bearing context shares the profile of its value-bearing twins.
_STORED_PROFILE_HASHES = [
    (('$', 'dc_gain', {}, 'tt', 27.0, 1.8),
     '652668a5cba10cf721356e90f28477a815b91cd917fa22bc647a22c9d68302f3'),
    (('$/amp1', 'load_step', {'cload_f': 2e-12, 'cycles': 4}, 'ss', -40.0, 1.62),
     '7b585053e6edd5e0d00cd2e48fcd1e5aee8c5a839fadefcde7fbe1449e139e1a'),
    (('$/bias~0', 'ibias', {'mode': 'fast', 'taps': [1, 2, 3]}, 'ff_hv', 125.0, 0.0),
     '8180f7be0787bab3f4a88dcfd447d37075b00d839d0b78e8e464cc1813d760e2'),
    # The frozen design example in tests/db/test_profile_identity.py.
    (('$/amp', 'load_step', {'cload_f': 2e-12, 'enabled': True, 'cycles': 4}, 'tt', 27.0, 1.8),
     'fd6adae669e7684e04ab70830df22a438a58119978c6ffc3fd4f75674c74ba0a'),
]


@pytest.mark.parametrize('args,digest', _STORED_PROFILE_HASHES)
def test_profile_hash_of_fixed_contexts_unchanged(args, digest):
    fingerprint = _canonicalize_characterization_profile(CharacterizationContext(*args)).fingerprint
    assert fingerprint.scheme == 'silk.profile/v1'
    assert fingerprint.digest == digest


@pytest.mark.parametrize('args,digest', _STORED_PROFILE_HASHES)
@pytest.mark.parametrize('pvt', [(NONE, None, None), (None, NONE, None), (None, None, NONE),
                                 (NONE, NONE, NONE)])
def test_none_bearing_context_shares_the_profile_hash(args, digest, pvt):
    owner, test, params, *values = args
    values = [v if p is None else p for v, p in zip(values, pvt)]
    ctx = CharacterizationContext(owner, test, params, *values)
    assert _canonicalize_characterization_profile(ctx).fingerprint.digest == digest
