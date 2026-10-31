"""Unit tests for silk.specs.spec — Measurement, Spec, CalibrationBinding."""
from copy import copy, deepcopy

import pytest

from silk.specs import CalibrationBinding, Measurement, Spec


class _FakeModule:
    """Stand-in for silk.circuit.modules.Module in unit tests."""
    def __init__(self, name: str):
        self.module_name = name
        self._behavioral_params = None


def test_measurement_defaults():
    m = Measurement(source='spice', value=2.3)
    assert m.source == 'spice'
    assert m.value == 2.3
    assert m.passes is None
    assert m.corner is None
    assert m.timestamp == 0.0
    assert m.sim_run_id is None
    assert m.notes == ''
    assert m.temperature is None
    assert m.voltage is None
    assert m.sim_type is None
    assert m.tool_name is None
    assert m.origin_version_id is None


def test_measurement_full():
    m = Measurement(
        source='behavioral',
        value=4.8,
        passes=True,
        corner='tt',
        timestamp=1713000000.0,
        sim_run_id='run-42',
        notes='hand-checked',
        temperature=27.0,
        voltage=1.8,
        sim_type='ac',
        tool_name='ngspice-44',
        origin_version_id='version-7',
    )
    assert m.source == 'behavioral'
    assert m.value == 4.8
    assert m.passes is True
    assert m.corner == 'tt'
    assert m.timestamp == 1713000000.0
    assert m.sim_run_id == 'run-42'
    assert m.notes == 'hand-checked'
    assert m.temperature == 27.0
    assert m.voltage == 1.8
    assert m.sim_type == 'ac'
    assert m.tool_name == 'ngspice-44'
    assert m.origin_version_id == 'version-7'


def _make_spec(**overrides):
    """Construct a Spec without a real Module (pass None as owner for unit tests)."""
    kwargs = dict(
        name='lna_NF',
        module=None,
        metric='noise_figure',
        op='<=',
        target=2.0,
        units='dB',
        description='LNA noise figure at 1 GHz',
    )
    kwargs.update(overrides)
    return Spec(**kwargs)


def test_spec_construction_defaults():
    s = _make_spec()
    assert s.name == 'lna_NF'
    assert s.key == 'lna_NF'
    assert s.module is None
    assert s.metric == 'noise_figure'
    assert s.op == '<='
    assert s.target == 2.0
    assert s.units == 'dB'
    assert s.description == 'LNA noise figure at 1 GHz'
    assert s.parent is None
    assert s.children == []
    assert s.measurements == {}
    assert s.calibration is None
    assert s.declared_at == ''


def test_spec_explicit_key_is_preserved_exactly():
    s = _make_spec(name='Display name', key=' Gain/TT ')

    assert s.name == 'Display name'
    assert s.key == ' Gain/TT '


@pytest.mark.parametrize('key', ['', ' ', '\t\n'])
def test_spec_rejects_empty_or_whitespace_only_key(key):
    with pytest.raises(ValueError, match='key'):
        _make_spec(key=key)


def test_spec_rejects_empty_name_when_it_would_be_the_default_key():
    with pytest.raises(ValueError, match='key'):
        _make_spec(name='')


@pytest.mark.parametrize('target', [None, '2.0', True, 1 + 2j])
def test_spec_rejects_non_real_numeric_target_at_construction(target):
    with pytest.raises(TypeError, match='target must be a real number'):
        _make_spec(target=target)


def test_spec_key_is_last_so_all_existing_positional_fields_keep_their_index():
    children = []
    measurements = {}
    s = Spec(
        'gain', None, 'dc_gain', '>=', 10.0, 'dB', 'description', None,
        children, measurements, None, 'design.py:12',
    )

    assert s.children is children
    assert s.measurements is measurements
    assert s.declared_at == 'design.py:12'
    assert s.key == 'gain'


def test_record_measurement_stores_by_source():
    s = _make_spec()
    m = Measurement(source='spice', value=2.3)
    s.record_measurement(m)
    assert 'spice' in s.measurements
    assert s.measurements['spice'] is m


def test_record_measurement_with_corner_adds_distinct_key():
    s = _make_spec()
    s.record_measurement(Measurement(source='spice', value=2.3))
    s.record_measurement(Measurement(source='spice', value=2.5, corner='ff'))
    assert 'spice' in s.measurements
    assert 'spice@ff' in s.measurements
    assert s.measurements['spice'].value == 2.3
    assert s.measurements['spice@ff'].value == 2.5


def test_measurement_profile_id_defaults_to_none():
    """DB-02/TASK-6 (design SS4.3): profile_id is a new trailing field that
    defaults to None so every pre-DB-02 Measurement() call site (examples,
    tests, the _simulate() live-record path) is unaffected."""
    m = Measurement(source='spice', value=2.3)
    assert m.profile_id is None


def test_record_measurement_unscoped_key_is_unchanged_by_profile_awareness():
    """A Measurement with no profile_id keeps the exact pre-DB-02 legacy
    slot key -- 'source' or 'source@corner' -- so unscoped/legacy restore
    and store_report's own live-measurement lookup are unaffected."""
    s = _make_spec()
    s.record_measurement(Measurement(source='spice', value=2.3))
    s.record_measurement(Measurement(source='spice', value=2.5, corner='ff'))
    assert set(s.measurements) == {'spice', 'spice@ff'}


def test_record_measurement_profile_scoped_key_coexists_with_unscoped():
    """Design SS4.3: 'a scoped slot is at least (profile_id, source, corner,
    temperature, voltage) so two profiles at one source@corner cannot
    overwrite each other.' A profile-scoped Measurement never collides with
    the legacy string key even at the identical source/corner."""
    s = _make_spec()
    legacy = Measurement(source='spice', value=1.0, corner='tt')
    scoped = Measurement(
        source='spice', value=2.0, corner='tt',
        temperature=27.0, voltage=1.8, profile_id='profile-a',
    )
    s.record_measurement(legacy)
    s.record_measurement(scoped)
    assert len(s.measurements) == 2
    assert s.measurements['spice@tt'] is legacy


def test_record_measurement_two_profiles_same_source_corner_coexist():
    """The exact AC5 scenario: two different profiles observed at the same
    source@corner must both survive in the rebuilt Python object instead of
    one silently overwriting the other."""
    s = _make_spec()
    a = Measurement(
        source='spice', value=61.0, corner='tt',
        temperature=27.0, voltage=1.8, profile_id='profile-a',
    )
    b = Measurement(
        source='spice', value=64.0, corner='tt',
        temperature=85.0, voltage=1.62, profile_id='profile-b',
    )
    s.record_measurement(a)
    s.record_measurement(b)
    assert len(s.measurements) == 2
    values = {m.profile_id: m.value for m in s.measurements.values()}
    assert values == {'profile-a': 61.0, 'profile-b': 64.0}


def test_record_measurement_same_profile_and_pvt_overwrites():
    """A re-run at the exact same profile/PVT selection key is the normal
    'latest observation occupies this live slot' overwrite -- unrelated to
    the different-profile coexistence guarantee."""
    s = _make_spec()
    s.record_measurement(Measurement(
        source='spice', value=61.0, corner='tt',
        temperature=27.0, voltage=1.8, profile_id='profile-a',
    ))
    s.record_measurement(Measurement(
        source='spice', value=62.0, corner='tt',
        temperature=27.0, voltage=1.8, profile_id='profile-a',
    ))
    assert len(s.measurements) == 1
    assert next(iter(s.measurements.values())).value == 62.0


def test_latest_returns_most_recent_regardless_of_source():
    s = _make_spec()
    s.record_measurement(Measurement(source='behavioral', value=2.0, timestamp=100.0))
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=200.0))
    latest = s.latest()
    assert latest.source == 'spice'
    assert latest.value == 2.3


def test_latest_filtered_by_source():
    s = _make_spec()
    s.record_measurement(Measurement(source='behavioral', value=2.0, timestamp=100.0))
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=200.0))
    latest_beh = s.latest(source='behavioral')
    assert latest_beh.source == 'behavioral'
    assert latest_beh.value == 2.0


def test_latest_returns_none_when_no_measurements():
    s = _make_spec()
    assert s.latest() is None
    assert s.latest(source='spice') is None


def test_calibration_binding_defaults():
    mod = _FakeModule('lna')
    cb = CalibrationBinding(target_module=mod, param_name='NF')
    assert cb.target_module is mod
    assert cb.param_name == 'NF'
    assert cb.source_filter == frozenset({'spice', 'extraction'})


def test_calibration_binding_custom_filter():
    mod = _FakeModule('lna')
    cb = CalibrationBinding(
        target_module=mod,
        param_name='NF',
        source_filter=frozenset({'spice'}),
    )
    assert cb.source_filter == frozenset({'spice'})
    assert 'behavioral' not in cb.source_filter
    assert 'extraction' not in cb.source_filter


@pytest.mark.parametrize('op,value,target,expected', [
    ('<',  1.9, 2.0, True),
    ('<',  2.0, 2.0, False),
    ('<=', 2.0, 2.0, True),
    ('<=', 2.1, 2.0, False),
    ('>',  2.1, 2.0, True),
    ('>',  2.0, 2.0, False),
    ('>=', 2.0, 2.0, True),
    ('>=', 1.9, 2.0, False),
    ('==', 2.0, 2.0, True),
    ('==', 2.001, 2.0, False),
])
def test_spec_evaluate_operators(op, value, target, expected):
    s = _make_spec(op=op, target=target)
    m = Measurement(source='spice', value=value)
    assert s.evaluate(m) is expected


def test_spec_evaluate_rejects_unknown_op():
    s = _make_spec(op='!=')  # deliberately invalid; evaluate must fail loudly
    with pytest.raises(KeyError):
        s.evaluate(Measurement(source='spice', value=1.0))


@pytest.mark.parametrize('field,value', [
    ('target', 99.0), ('op', '>'), ('metric', 'gain'), ('key', 'other'),
])
@pytest.mark.parametrize('assignment', ['direct', 'setattr'])
def test_spec_rejects_declaration_assignment_without_changing_state(
    field, value, assignment,
):
    s = _make_spec()
    measurement = Measurement(source='spice', value=1.5, passes=True)
    s.record_measurement(measurement)
    before = dict(vars(s))

    with pytest.raises(AttributeError) as error:
        if assignment == 'setattr':
            setattr(s, field, value)
        elif field == 'target':
            s.target = value
        elif field == 'op':
            s.op = value
        elif field == 'metric':
            s.metric = value
        else:
            s.key = value

    message = str(error.value)
    assert field in message
    assert 'declaration' in message and 'rebuild' in message
    assert 'circuit/testbench' in message
    assert vars(s) == before
    assert s.latest() is measurement
    assert s.evaluate(measurement) is True


@pytest.mark.parametrize('field', ['target', 'op', 'metric', 'key'])
@pytest.mark.parametrize('deletion', ['direct', 'delattr'])
def test_spec_rejects_deletion_and_subsequent_reassignment(field, deletion):
    s = _make_spec()
    before = dict(vars(s))

    with pytest.raises(AttributeError, match=field):
        if deletion == 'delattr':
            delattr(s, field)
        elif field == 'target':
            del s.target
        elif field == 'op':
            del s.op
        elif field == 'metric':
            del s.metric
        else:
            del s.key
    with pytest.raises(AttributeError, match=field):
        setattr(s, field, before[field])

    assert vars(s) == before


@pytest.mark.parametrize('field,value', [
    ('name', 'New display'), ('module', _FakeModule('new_owner')),
    ('units', 'V'), ('description', 'New description'),
    ('parent', _make_spec(name='parent')), ('children', []),
    ('measurements', {}), ('calibration', CalibrationBinding(None, 'NF')),
    ('declared_at', 'rebuilt.py:1'),
])
def test_spec_fields_outside_protected_set_remain_assignable(field, value):
    s = _make_spec()
    setattr(s, field, value)
    assert getattr(s, field) is value


@pytest.mark.parametrize('copier', [copy, deepcopy])
@pytest.mark.parametrize('key', [None, 'explicit'])
def test_copied_spec_preserves_declaration_guard_and_measurements(copier, key):
    original = _make_spec(key=key)
    measurement = Measurement(source='spice', value=1.5, passes=True)
    original.record_measurement(measurement)

    copied = copier(original)

    assert copied is not original
    assert (copied.target, copied.op, copied.metric, copied.key) == (
        2.0, '<=', 'noise_figure', key or 'lna_NF',
    )
    assert copied.latest() == measurement
    assert copied.evaluate(copied.latest()) is True
    if copier is copy:
        assert copied.measurements is original.measurements
    else:
        assert copied.measurements is not original.measurements
        assert copied.latest() is not measurement
    for field in ('target', 'op', 'metric', 'key'):
        with pytest.raises(AttributeError, match=field):
            setattr(copied, field, getattr(copied, field))
        with pytest.raises(AttributeError, match=field):
            delattr(copied, field)


def test_spec_unprotected_attribute_deletion_remains_supported():
    s = _make_spec()
    s.runtime_note = 'temporary'
    del s.runtime_note
    assert not hasattr(s, 'runtime_note')
    del s.description
    s.description = 'Replacement description'
    assert s.description == 'Replacement description'
