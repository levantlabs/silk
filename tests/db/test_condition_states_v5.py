"""Schema v5 condition states (TASK-409; design 2026-10-01
operating-conditions-none, §5.1).

Each evidence row stores corner, temperature and voltage as a state plus a
value. Writers derive the state from a widened scalar (a value,
``ConditionState.NONE``, ``None`` for unknown, ``ConditionState.SWEEP`` for
temperature); readers hand the same scalar back; selectors match
state-and-value, so none and unknown (both NULL) never meet.
"""

import json
import sqlite3
import uuid

import pytest

from silk.db import Database
from silk.db.context import CharacterizationContext
from silk.exceptions import EvidenceMismatchError
from silk.simulation.conditions import (
    ConditionState,
    OperatingConditions,
    ConditionSweep,
)
from tests.db.test_characterization_readback import (
    _build_live,
    _current_row,
    _scoped_report,
)
from tests.db_snapshot_helpers import create_sealed_snapshot

NONE = ConditionState.NONE
SWEEP = ConditionState.SWEEP

#: (corner, temperature, voltage) -> expected (corner_state, temperature_state,
#: voltage_state) for every state each fact allows.
ROUND_TRIPS = [
    (('tt', 27.0, 1.8), ('fixed', 'fixed', 'fixed')),
    ((NONE, NONE, NONE), ('none', 'none', 'none')),
    ((None, None, None), ('unknown', 'unknown', 'unknown')),
    (('ss', SWEEP, NONE), ('fixed', 'sweep', 'none')),
    ((NONE, 85.0, None), ('none', 'fixed', 'unknown')),
    (('tt', 27.0, SWEEP), ('fixed', 'fixed', 'sweep')),
]


def _states(db, table, key_column, key):
    return tuple(db.conn.execute(
        f"SELECT corner_state, temperature_state, voltage_state FROM {table} "
        f"WHERE {key_column} = ?", (key,)).fetchone())


def _context(**overrides):
    values = dict(owner_path='$/amp', test_name='load_step',
                  parameters={'cload_f': 2e-12}, corner='tt',
                  temperature=27.0, voltage=1.8)
    values.update(overrides)
    return CharacterizationContext(**values)


# --- writers and readers ---------------------------------------------------

@pytest.mark.parametrize('scalars, states', ROUND_TRIPS)
def test_simulation_result_round_trips_every_state(scalars, states):
    corner, temperature, voltage = scalars
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        sim_id = db.add_simulation_result(
            version, 'gain', value=1.0, corner=corner,
            temperature=temperature, voltage=voltage)
        assert _states(db, 'simulation_results', 'sim_id', sim_id) == states
        record, = db.get_simulation_results(version)
        assert (record.corner, record.temperature, record.voltage) == scalars
        # A plain-Enum member comes back as the member itself, not a string.
        for got, want in zip((record.corner, record.temperature, record.voltage), scalars):
            assert type(got) is type(want)


@pytest.mark.parametrize('scalars, states', ROUND_TRIPS)
def test_spec_measurement_round_trips_every_state(pdk, tmp_path, scalars, states):
    corner, temperature, voltage = scalars
    module, _gain = _build_live(pdk, tmp_path, tmp_path / 'spec.db')
    try:
        version = module.create_circuit_version()
        db = module._tracker.db
        declaration, = db.get_stored_specs(version)
        measurement_id = str(uuid.uuid4())
        db.insert_spec_measurement(
            measurement_id, declaration.spec_id, 'spice', 64.0, corner=corner,
            temperature=temperature, voltage=voltage)
        assert _states(db, 'spec_measurements', 'measurement_id', measurement_id) == states
        stored, = db.get_stored_specs(version, include_measurements=True)
        record, = stored.measurements
        assert (record.corner, record.temperature, record.voltage) == scalars
    finally:
        module.close()


@pytest.mark.parametrize('scalars, states', [
    (s, st) for s, st in ROUND_TRIPS if s[0] is not None
])
def test_current_set_round_trips_every_state(scalars, states):
    corner, temperature, voltage = scalars
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        row = db._snapshot_row(version)
        set_id = db.replace_net_current_set(
            version, corner=corner, temperature=temperature, voltage=voltage,
            structure_scheme=row['structure_scheme'],
            structure_hash=row['structure_hash'], rows=[_current_row(1e-3)])
        assert _states(db, 'net_current_sets', 'current_set_id', set_id) == states
        origin, rows = db.get_net_current_set(version, corner)
        record = db._net_current_set_record(origin, rows)
        assert record.current_set_id == set_id
        assert (record.corner, record.temperature, record.voltage) == scalars


def test_writers_refuse_states_that_are_not_scalars():
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        for kwargs in (
            {'corner': ConditionState.FIXED},
            {'corner': ConditionState.UNKNOWN},
            {'corner': SWEEP},
            {'temperature': ConditionState.FIXED},
            {'temperature': 'hot'},
            {'corner': 1.0},
        ):
            with pytest.raises(TypeError):
                db.add_simulation_result(version, 'gain', value=1.0, **kwargs)
        assert db.conn.execute('SELECT COUNT(*) FROM simulation_results').fetchone()[0] == 0


def test_none_corner_never_equals_a_corner_named_none():
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        db.add_simulation_result(version, 'gain', value=1.0, corner='none')
        db.add_simulation_result(version, 'gain', value=2.0, corner=NONE)
        db.add_simulation_result(version, 'gain', value=3.0, corner=None)
        by_corner = {
            record.value: record.corner
            for record in db.get_simulation_results(version)
        }
        assert by_corner == {1.0: 'none', 2.0: NONE, 3.0: None}
        assert [r.value for r in db.get_simulation_results(version, corner='none')] == [1.0]
        assert [r.value for r in db.get_simulation_results(version, corner=NONE)] == [2.0]
        # latest_only keeps none, unknown and the corner named 'none' apart.
        assert len(db.get_simulation_results(version, latest_only=True)) == 3


# --- CHECK constraints -----------------------------------------------------

def test_checks_tie_each_state_to_its_value():
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        insert = (
            "INSERT INTO simulation_results (sim_id, circuit_version_id, source, "
            "parameter, value, corner, temperature, voltage, corner_state, "
            "temperature_state, voltage_state) VALUES (?, ?, 'user', 'g', 1.0, "
            "?, ?, ?, ?, ?, ?)")
        for values in (
            ('tt', 27.0, 1.8, 'none', 'fixed', 'fixed'),     # value with none
            (None, 27.0, 1.8, 'fixed', 'fixed', 'fixed'),    # fixed without value
            ('tt', 27.0, 1.8, 'fixed', 'fixed', 'sweep'),    # sweep must hold NULL
            ('tt', None, 1.8, 'fixed', 'bogus', 'fixed'),
        ):
            with pytest.raises(sqlite3.IntegrityError):
                db.conn.execute(insert, (str(uuid.uuid4()), version, *values))


@pytest.mark.parametrize('states', [
    ('unknown', 'fixed', 'fixed'),
    ('fixed', 'sweep', 'fixed'),
    ('fixed', 'fixed', 'unknown'),
    ('fixed', 'fixed', 'sweep'),
])
def test_a_profile_row_needs_every_fact_fixed_or_none(states):
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        profile = db.store_characterization_profile(version, _context())
        corner, temperature, voltage = (
            value if state == 'fixed' else None
            for state, value in zip(states, ('tt', 27.0, 1.8))
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO simulation_results (sim_id, circuit_version_id, "
                "source, parameter, value, corner, temperature, voltage, "
                "profile_id, corner_state, temperature_state, voltage_state) "
                "VALUES (?, ?, 'user', 'g', 1.0, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), version, corner, temperature, voltage,
                 profile.profile_id, *states))
        scalars = tuple(
            {'unknown': None, 'sweep': SWEEP}.get(state, value)
            for state, value in zip(states, ('tt', 27.0, 1.8))
        )
        with pytest.raises(ValueError, match='each a value or ConditionState.NONE'):
            db.add_simulation_result(
                version, 'g', value=1.0, corner=scalars[0],
                temperature=scalars[1], voltage=scalars[2],
                profile_id=profile.profile_id)


# --- profiles and exact selection ------------------------------------------

def _store_scoped(db, version, context, *, gain, avg_a):
    profile = db.store_characterization_profile(version, context)
    db.add_simulation_result(
        version, 'gain_dB', value=gain, corner=context.corner,
        temperature=context.temperature, voltage=context.voltage,
        profile_id=profile.profile_id)
    row = db._snapshot_row(version)
    set_id = db.replace_net_current_set(
        version, corner=context.corner, temperature=context.temperature,
        voltage=context.voltage, structure_scheme=row['structure_scheme'],
        structure_hash=row['structure_hash'], context=context,
        rows=[_current_row(avg_a)])
    return profile, set_id


def test_none_bearing_context_selects_only_its_own_evidence():
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        fixed = _context(corner='tt', temperature=85.0, voltage=1.8)
        none = _context(corner=NONE, temperature=85.0, voltage=NONE)
        fixed_profile, fixed_set = _store_scoped(db, version, fixed, gain=10.0, avg_a=1e-3)
        none_profile, none_set = _store_scoped(db, version, none, gain=20.0, avg_a=2e-3)
        # Profile identity excludes PVT, so both contexts share one profile.
        assert none_profile.profile_id == fixed_profile.profile_id

        evidence = db.select_characterization_evidence(version, context=none)
        assert evidence.current_set_id == none_set
        assert (evidence.corner, evidence.temperature, evidence.voltage) == (NONE, 85.0, NONE)
        assert [m.value for m in evidence.metrics] == [20.0]
        evidence = db.select_characterization_evidence(
            version, profile_id=fixed_profile.profile_id,
            corner='tt', temperature=85.0, voltage=1.8)
        assert evidence.current_set_id == fixed_set
        assert [m.value for m in evidence.metrics] == [10.0]

        # The none point has no unknown twin to fall back to.
        with pytest.raises(EvidenceMismatchError, match='not at the requested exact PVT'):
            db.select_characterization_evidence(
                version, profile_id=fixed_profile.profile_id,
                corner=NONE, temperature=85.0, voltage=1.8)
        with pytest.raises(ValueError, match='each a value or ConditionState.NONE'):
            db.select_characterization_evidence(
                version, profile_id=fixed_profile.profile_id,
                corner='tt', temperature=SWEEP, voltage=1.8)
        with pytest.raises(ValueError, match='each a value or ConditionState.NONE'):
            db.select_characterization_evidence(
                version, profile_id=fixed_profile.profile_id,
                corner='tt', temperature=85.0, voltage=SWEEP)


def test_none_corner_current_sets_supersede_and_stay_unique():
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        row = db._snapshot_row(version)
        common = dict(structure_scheme=row['structure_scheme'],
                      structure_hash=row['structure_hash'])
        first = db.replace_net_current_set(
            version, corner=NONE, rows=[_current_row(1e-3)], **common)
        second = db.replace_net_current_set(
            version, corner=NONE, rows=[_current_row(2e-3)], **common)
        tt = db.replace_net_current_set(
            version, corner='tt', rows=[_current_row(3e-3)], **common)
        states = dict(db.conn.execute(
            'SELECT current_set_id, state FROM net_current_sets').fetchall())
        assert states == {first: 'superseded', second: 'current', tt: 'current'}
        assert db.get_net_current_set(version, NONE)[1][0]['avg_a'] == 2e-3
        assert db.get_net_current_set(version, 'tt')[1][0]['avg_a'] == 3e-3

        # The unique index is the backstop: a second current none-corner set
        # collides even though its corner column is NULL.
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO net_current_sets (current_set_id, circuit_version_id, "
                "state, corner, structure_scheme, structure_hash, corner_state, "
                "temperature_state, voltage_state) VALUES (?, ?, 'current', NULL, "
                "?, ?, 'none', 'unknown', 'unknown')",
                (str(uuid.uuid4()), version, row['structure_scheme'],
                 row['structure_hash']))
        with pytest.raises(ValueError, match='corner is required'):
            db.replace_net_current_set(
                version, corner=None, rows=[_current_row(1e-3)], **common)


# --- tracker writers ------------------------------------------------------

def test_store_report_takes_conditions_from_the_snapshot_not_the_label(pdk, tmp_path):
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'report.db')
    try:
        version = module.create_circuit_version()
        report = _scoped_report(gain, _context(), value=64.0)
        report.characterization_context = None
        report.corner = 'tt'          # the run label
        report.operating_conditions = OperatingConditions(
            'silk', 'ngspice', None, corner_kind='none',
            corner_basis='proven-no-pdk-device', temperature_kind='fixed',
            temperature_c=85.0, temperature_basis='configured-and-observed',
            supply_kind='none', supply_basis='proven-no-supply')
        module.store_report(report)
        db = module._tracker.db
        assert {
            (r.parameter, r.corner, r.temperature, r.voltage)
            for r in db.get_simulation_results(version)
        } == {
            ('gain_dB', NONE, 85.0, NONE),
            ('spec_Gain', NONE, 85.0, NONE),
        }
        stored, = db.get_stored_specs(version, include_measurements=True)
        measurement, = stored.measurements
        assert (measurement.corner, measurement.temperature, measurement.voltage) == (
            NONE, 85.0, NONE)
    finally:
        module.close()


def test_store_net_currents_accepts_a_none_corner_and_refuses_none(pdk, tmp_dir, tmp_path):
    from tests.test_net_current_store import _annotate_definition, _build_tb, _fake_report

    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 'currents.db'))
    _annotate_definition(tb)
    with pytest.raises(RuntimeError, match='corner is required'):
        tb.store_net_currents(_fake_report(), corner=None)
    versions = tb.store_net_currents(_fake_report(), corner=NONE)
    db = tb._ensure_tracker().db
    origin, rows = db.get_net_current_set(versions['dut_blk'], NONE)
    assert origin == versions['dut_blk']
    assert {row['corner_state'] for row in rows} == {'none'}
    assert {row['corner'] for row in rows} == {None}
    with pytest.raises(RuntimeError, match='no rows for corner'):
        db.get_net_current_set(versions['dut_blk'], 'tt')


# --- JSON import/export ----------------------------------------------------

def _import_json(db, tmp_path, results, name='states', projection_format=None):
    from silk.db import CircuitImporter

    path = tmp_path / f'{name}.json'
    payload = {'circuits': [{
        'name': name, 'version': '1.0', 'stage': 'dev',
        'transistors': [], 'nets': [],
        'simulation_results': results,
    }]}
    if projection_format is not None:
        payload['projection_format'] = projection_format
    path.write_text(json.dumps(payload))
    version, = CircuitImporter(db).import_from_json(str(path))
    return {r.parameter: (r.corner, r.temperature, r.voltage)
            for r in db.get_simulation_results(version)}


def test_export_carries_states_and_reimports_every_state(tmp_path):
    from silk.db import CircuitExporter

    with Database(':memory:') as db:
        version = create_sealed_snapshot(db, name='states')
        for index, (scalars, _states) in enumerate(ROUND_TRIPS):
            corner, temperature, voltage = scalars
            db.add_simulation_result(
                version, f'p{index}', value=float(index), corner=corner,
                temperature=temperature, voltage=voltage)
        out = tmp_path / 'out.json'
        CircuitExporter(db).export_to_json('states', str(out))
    payload = json.loads(out.read_text())
    assert payload['projection_format'] == 'silk.circuit-projection/v2'
    exported = {
        r['parameter']: r
        for r in payload['circuits'][0]['simulation_results']
    }
    none_row = exported['p1']
    assert none_row['corner'] == {'state': 'none'}
    assert (none_row['corner_state'], none_row['temperature_state'],
            none_row['voltage_state']) == ('none', 'none', 'none')
    assert exported['p3']['temperature'] == {'state': 'sweep'}
    assert exported['p2']['corner'] is None and exported['p2']['corner_state'] == 'unknown'
    assert exported['p0']['corner'] == 'tt' and exported['p0']['corner_state'] == 'fixed'

    with Database(':memory:') as target:
        imported = _import_json(
            target, tmp_path, list(exported.values()), name='states',
            projection_format='silk.circuit-projection/v2')
    assert imported == {
        f'p{index}': scalars for index, (scalars, _s) in enumerate(ROUND_TRIPS)
    }


@pytest.mark.parametrize('projection_format', [None, 'silk.circuit-projection/v1'])
def test_old_export_without_states_imports_per_the_migration_table(
        tmp_path, projection_format):
    sweep_marker = OperatingConditions(
        'silk', 'ngspice', 'tt', temperature_kind='sweep',
        temperature_sweep=ConditionSweep(-40.0, 125.0, 5.0),
    ).to_dict()
    with Database(':memory:') as db:
        imported = _import_json(db, tmp_path, [
            {'parameter': 'fixed', 'value': 1.0, 'corner': 'tt',
             'temperature': 27.0, 'voltage': 1.8},
            {'parameter': 'unknown', 'value': 2.0, 'corner': None,
             'temperature': None, 'voltage': None},
            {'parameter': 'absent', 'value': 3.0},
            {'parameter': 'sweep', 'value': 4.0, 'corner': 'tt',
             'temperature': None, 'voltage': None,
             'conditions': {'silk_operating_conditions': sweep_marker}},
        ], projection_format=projection_format)
    assert imported == {
        'fixed': ('tt', 27.0, 1.8),
        'unknown': (None, None, None),
        'absent': (None, None, None),
        'sweep': ('tt', SWEEP, None),
    }


@pytest.mark.parametrize('result, match', [
    ({'corner': 'tt', 'corner_state': 'none'}, 'corner_state'),
    ({'corner': None, 'corner_state': 'fixed'}, 'corner_state'),
    ({'voltage': {'state': 'sweep'}, 'voltage_state': 'fixed'}, 'voltage_state'),
    ({'corner': {'state': 'bogus'}}, 'corner'),
])
def test_import_refuses_inconsistent_or_invalid_conditions(tmp_path, result, match):
    from silk.exceptions import DatabaseError

    with Database(':memory:') as db:
        with pytest.raises(DatabaseError, match=match):
            _import_json(db, tmp_path, [{'parameter': 'g', 'value': 1.0, **result}])


@pytest.mark.parametrize('result', [
    {'corner': 'tt', 'corner_state': 'fixed'},
    {'corner': {'state': 'none'}},
    {'temperature': {'state': 'sweep'}, 'temperature_state': 'sweep'},
])
def test_a_v1_projection_cannot_carry_condition_states(tmp_path, result):
    from silk.exceptions import DatabaseError

    with Database(':memory:') as db:
        with pytest.raises(DatabaseError, match='circuit-projection/v2'):
            _import_json(db, tmp_path, [{'parameter': 'g', 'value': 1.0, **result}],
                         projection_format='silk.circuit-projection/v1')
        assert db.conn.execute('SELECT COUNT(*) FROM circuit_versions').fetchone()[0] == 0


@pytest.mark.parametrize('projection_format', [
    'silk.circuit-projection/v3', 'silk.circuit-projection', 'other/v2',
])
def test_an_unknown_projection_format_is_refused_before_any_write(
        tmp_path, projection_format):
    from silk.exceptions import DatabaseError

    with Database(':memory:') as db:
        with pytest.raises(DatabaseError, match='Unsupported projection_format'):
            _import_json(db, tmp_path, [{'parameter': 'g', 'value': 1.0}],
                         projection_format=projection_format)
        assert db.conn.execute('SELECT COUNT(*) FROM circuit_versions').fetchone()[0] == 0


def test_database_summary_spells_supply_sweep():
    assert Database._summary_condition_labels('tt', 27, SWEEP) == (
        'tt', ['27°C', 'supply sweep'])
