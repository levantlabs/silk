"""TASK-129: ``store_report`` files an unscoped report under the corner the
run actually used, never the tracker's current/default corner.

Before this fix ``ModuleTestbench.run(corner='ff')`` on an unscoped testbench
followed by ``store_report(report)`` (no ``corner=``) recorded every metric
and spec measurement under the tracker's ``_current_corner`` (``'tt'`` by
default) -- silent wrong evidence. ``SpecReport`` now carries ``corner`` (the
resolved run corner) and ``store_report`` resolves an unscoped report's
corner as: explicit argument, else ``report.corner``, else the tracker
current corner. An explicit argument that disagrees with ``report.corner``
raises, the same way the scoped-context conflict already does.

Scoped (characterization-context) reports are covered unchanged by
tests/db/test_profile_scoped_evidence.py and tests/db/test_spec_restore.py.
"""
from unittest.mock import patch
from numbers import Real

import numpy as np
import pytest
from tests._temperature import mark_simulated

from silk.circuit.modules import Module
from silk.circuit.spec_report import SpecReport
from silk.exceptions import EvidenceMismatchError
from silk.simulation.analysis import Transient
from silk.simulation.corner import Corner
from silk import sources


# ---------------------------------------------------------------------------
# Helpers (mirror tests/test_characterization_context_propagation.py; local so
# this file has no cross-file fixture dependency)
# ---------------------------------------------------------------------------

def _dut(pdk, tmp_path, db_path, module_name='corner_dut'):
    mod = Module(
        'PDK_skywater130', str(tmp_path / module_name), 'dut.cir',
        pdk_details=pdk, module_name=module_name, db_path=str(db_path),
    )
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})
    mod.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
    mod.add(pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))
    return mod


def _wired_test(mod, name='characterize', gain=65.0, *, gain_extractor=None):
    """A testbench with one constant metric and one first-class spec on it,
    ready to run() with ``netlist.simulate`` stubbed out."""
    test = mod.create_test(name)
    test.add_module_under_test({
        'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0',
    })
    test.add_source(sources.VoltageSource(
        'vsup', positive='vdd', negative='0', dc=1.8))
    test.add_source(sources.VoltageSource(
        'vin_src', positive='vin', negative='0', dc=0.9))
    test.add_sim_type(Transient(step=1e-9, stop=1e-8))
    if gain_extractor is None:
        gain_extractor = lambda _ckt: gain
    test.add_metric('gain_dB', gain_extractor)
    test.add_metric('ugf_MHz', lambda _ckt: 12.5)
    test.add_spec('Gain', 'gain_dB', '>=', 60.0, key='gain')
    return test


def _run_without_simulating(test, **kwargs):
    def fake_simulate(*_a, **_kw):
        mark_simulated(test.netlist, _kw.get('corner', 'tt'))
    with patch.object(test.netlist, 'simulate', side_effect=fake_simulate):
        return test.run(**kwargs)


def _stored_corners(mod):
    """(metric-row corners, spec-status-row corners, spec_measurement
    corners) read straight back from the bound database."""
    conn = mod._tracker.db.conn
    metric_rows = {
        (row['parameter'], row['corner'])
        for row in conn.execute(
            "SELECT parameter, corner FROM simulation_results "
            "WHERE parameter NOT LIKE 'spec_%'"
        ).fetchall()
    }
    status_rows = {
        (row['parameter'], row['corner'])
        for row in conn.execute(
            "SELECT parameter, corner FROM simulation_results "
            "WHERE parameter LIKE 'spec_%'"
        ).fetchall()
    }
    measurement_corners = [
        row['corner']
        for row in conn.execute(
            "SELECT corner FROM spec_measurements"
        ).fetchall()
    ]
    return metric_rows, status_rows, measurement_corners


# ---------------------------------------------------------------------------
# AC1: the run corner is recorded on the report and is what gets stored
# ---------------------------------------------------------------------------

def test_run_stamps_the_corner_it_used_on_the_report(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'stamp.db')
    test = _wired_test(mod)

    report = _run_without_simulating(test, corner='ff')

    assert report.corner == 'ff'
    assert report.characterization_context is None  # still unscoped
    mod.close()


def test_run_with_corner_enum_stamps_its_plain_string(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'enum.db')
    test = _wired_test(mod)

    report = _run_without_simulating(test, corner=Corner.SS)

    assert report.corner == 'ss'
    assert type(report.corner) is str
    mod.close()


def test_unscoped_ff_report_is_stored_under_ff_not_the_tracker_default(
    pdk, tmp_path,
):
    """The bug: run(corner='ff') + store_report(report) used to file every
    row under the tracker's current corner ('tt')."""
    mod = _dut(pdk, tmp_path, tmp_path / 'ff.db')
    test = _wired_test(mod)
    mod.create_circuit_version()
    assert mod._tracker._current_corner == 'tt'

    report = _run_without_simulating(test, corner='ff')
    stored = mod.store_report(report)

    assert stored == {'metrics_stored': 2, 'specs_stored': 1, 'skipped': {}}
    metric_rows, status_rows, measurement_corners = _stored_corners(mod)
    assert metric_rows == {('gain_dB', 'ff'), ('ugf_MHz', 'ff')}
    assert status_rows == {('spec_Gain', 'ff')}
    assert measurement_corners == ['ff']
    # Readback through the public query path agrees.
    assert {r.corner for r in mod.query_results(parameter='gain_dB')} == {'ff'}
    # The tracker default was never consulted and is not itself moved.
    assert mod._tracker._current_corner == 'tt'
    mod.close()


def test_default_corner_run_is_stored_under_the_pdk_default_not_hardcoded_tt(
    pdk, tmp_path, monkeypatch,
):
    """run() with no corner resolves the PDK manifest's default_corner; that
    resolved value -- not a literal 'tt', and not the tracker's current
    corner -- is what store_report files the rows under."""
    mod = _dut(pdk, tmp_path, tmp_path / 'default.db')
    test = _wired_test(mod)
    mod.create_circuit_version()
    # Make the PDK default distinguishable from both the literal 'tt' and
    # the tracker's current corner.
    monkeypatch.setattr(test.pdk_details, 'default_corner', 'sf')
    mod.set_corner('fs')

    report = _run_without_simulating(test)
    assert report.corner == 'sf'
    mod.store_report(report)

    metric_rows, status_rows, measurement_corners = _stored_corners(mod)
    assert {corner for _p, corner in metric_rows} == {'sf'}
    assert {corner for _p, corner in status_rows} == {'sf'}
    assert measurement_corners == ['sf']
    mod.close()


# ---------------------------------------------------------------------------
# AC2: an explicit corner may only confirm the run corner
# ---------------------------------------------------------------------------

def test_explicit_corner_that_disagrees_with_the_run_corner_raises(
    pdk, tmp_path,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'conflict.db')
    test = _wired_test(mod)
    version_id = mod.create_circuit_version()
    report = _run_without_simulating(test, corner='ff')

    with pytest.raises(EvidenceMismatchError) as excinfo:
        mod.store_report(report, corner='tt')
    message = str(excinfo.value)
    assert "'tt'" in message and "'ff'" in message
    assert "conflicts" in message

    # Nothing was written.
    conn = mod._tracker.db.conn
    assert conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 0
    mod.close()


def test_explicit_corner_matching_the_run_corner_still_works(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'match.db')
    test = _wired_test(mod)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='ff')

    stored = mod.store_report(report, corner='ff')

    assert stored['metrics_stored'] == 2 and stored['specs_stored'] == 1
    metric_rows, _status_rows, measurement_corners = _stored_corners(mod)
    assert {corner for _p, corner in metric_rows} == {'ff'}
    assert measurement_corners == ['ff']
    mod.close()


def test_explicit_corner_enum_matching_the_run_corner_still_works(
    pdk, tmp_path,
):
    """The scoped path compares Corner enums by value; the run-corner check
    must too, so Corner.FF confirms a report that ran at 'ff'."""
    mod = _dut(pdk, tmp_path, tmp_path / 'match-enum.db')
    test = _wired_test(mod)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='ff')

    mod.store_report(report, corner=Corner.FF)

    metric_rows, _status_rows, measurement_corners = _stored_corners(mod)
    assert {corner for _p, corner in metric_rows} == {'ff'}
    assert measurement_corners == ['ff']
    mod.close()


# ---------------------------------------------------------------------------
# Compatibility: a hand-built report that records no corner keeps the old
# fallback chain (explicit argument, else tracker current corner)
# ---------------------------------------------------------------------------

def test_hand_built_report_without_corner_falls_back_to_tracker_corner(
    pdk, tmp_path,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'handbuilt.db')
    gain = mod.declare_spec('Gain', 'gain_dB', '>=', 60.0, key='gain')
    mod.create_circuit_version()
    mod.set_corner('ss')

    def report():
        r = SpecReport('characterize', 'corner_dut', {'gain_dB': 65.0})
        r.check(gain.name, gain.metric, gain.op, gain.target,
                spec_key=gain.key, owner_path='$')
        return r

    assert report().corner is None
    mod.store_report(report())                 # -> tracker current corner
    mod.store_report(report(), corner='fs')    # -> explicit wins, no conflict

    _metric_rows, _status_rows, measurement_corners = _stored_corners(mod)
    assert sorted(measurement_corners) == ['fs', 'ss']
    mod.close()


# ---------------------------------------------------------------------------
# run_corners / CornerResults keep per-corner identity through the field
# ---------------------------------------------------------------------------

def test_run_corners_reports_each_carry_their_own_corner(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'corners.db')
    test = _wired_test(mod)

    # Patched on the CLASS (plain function, not a Mock instance attribute) so
    # normal method binding passes the ACTUAL netlist instance being
    # simulated -- run_corners() (TASK-301.3) deep-copies each corner's
    # netlist before simulating, so a version closing over test.netlist
    # would mark the wrong object simulated.
    def fake_simulate(netlist, *_a, **_kw):
        mark_simulated(netlist, _kw.get('corner', 'tt'))
    with patch.object(type(test.netlist), 'simulate', fake_simulate):
        results = test.run_corners(corners=('tt', 'ff', 'ss'))

    assert list(results) == ['tt', 'ff', 'ss']
    for corner, report in results.items():
        assert report.corner == corner
        assert report.characterization_context is None
    mod.close()


def test_run_corners_reports_store_under_their_own_corners(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'corners-store.db')
    test = _wired_test(mod)
    mod.create_circuit_version()

    def fake_simulate(netlist, *_a, **_kw):
        mark_simulated(netlist, _kw.get('corner', 'tt'))
    with patch.object(type(test.netlist), 'simulate', fake_simulate):
        results = test.run_corners(corners=('ff', 'ss'))
    for report in results.values():
        mod.store_report(report)

    metric_rows, status_rows, measurement_corners = _stored_corners(mod)
    assert metric_rows == {
        ('gain_dB', 'ff'), ('ugf_MHz', 'ff'),
        ('gain_dB', 'ss'), ('ugf_MHz', 'ss'),
    }
    assert status_rows == {('spec_Gain', 'ff'), ('spec_Gain', 'ss')}
    assert sorted(measurement_corners) == ['ff', 'ss']
    mod.close()


# ---------------------------------------------------------------------------
# API shape: the field is additive and keyword-only
# ---------------------------------------------------------------------------

def test_spec_report_corner_is_additive_keyword_only_and_defaults_to_none():
    import inspect
    params = inspect.signature(SpecReport).parameters
    assert params['corner'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['corner'].default is None
    assert SpecReport('t', 'm', {}).corner is None
    assert SpecReport('t', 'm', {}, corner=Corner.FF).corner == 'ff'


@pytest.mark.parametrize('value', [65+4j, complex(float('nan'), 4),
                                   np.complex128(65+4j)])
@pytest.mark.filterwarnings('error::numpy.exceptions.ComplexWarning')
def test_store_complex_report_skips_numeric_rows_without_losing_evidence(pdk, tmp_path, value):
    mod = _dut(pdk, tmp_path, tmp_path / 'complex.db')
    test = _wired_test(mod, gain=value)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='tt')
    receipt = mod.store_report(report)
    assert report.metrics['gain_dB'] is value
    assert report.results[0].value is value
    assert not report.results[0].passed
    assert 'complex' in receipt['skipped']['gain_dB']
    assert '.real' in receipt['skipped']['gain_dB']
    assert receipt['metrics_stored'] == 1
    assert mod._tracker.db.conn.execute('SELECT COUNT(*) FROM spec_measurements').fetchone()[0] == 0
    mod.close()


@pytest.mark.parametrize('value,other', [(65+4j, 65+5j),
                                         (complex(-0.0, 4), complex(0.0, 4))])
def test_store_complex_report_rejects_changed_component(pdk, tmp_path, value, other):
    mod = _dut(pdk, tmp_path, tmp_path / 'complex-mismatch.db')
    test = _wired_test(mod, gain=value)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='tt')
    report.metrics['gain_dB'] = other
    with pytest.raises(EvidenceMismatchError, match='report metric value'):
        mod.store_report(report)
    mod.close()


@pytest.mark.parametrize('later', [65.0, 72.0])
@pytest.mark.parametrize('transport', [False, True])
@pytest.mark.parametrize('scoped', [False, True])
def test_delayed_report_keeps_original_observation(pdk, tmp_path, later, transport, scoped):
    mod = _dut(pdk, tmp_path, tmp_path / 'delayed.db')
    current_gain = 65.0
    test = _wired_test(mod, gain_extractor=lambda _ckt: current_gain)
    if scoped:
        test.set_temperature(27)
        test.set_supply_voltage(1.8, source='vsup')
    mod.create_circuit_version()
    a = _run_without_simulating(test, corner='tt')
    spec = test._specs[0]['spec']
    original = spec.latest_attempt()
    current_gain = later
    b = _run_without_simulating(test, corner='tt')
    newest = spec.latest_attempt()
    assert newest.attempt_id != original.attempt_id
    assert newest.timestamp != original.timestamp
    mod.store_report(b)
    if transport:
        a = SpecReport.from_dict(a.to_dict())
    for _ in range(2):
        mod.store_report(a)
    rows = mod._tracker.db.conn.execute('SELECT value, timestamp FROM spec_measurements').fetchall()
    assert [(r['value'], r['timestamp']) for r in rows] == [
        (later, newest.timestamp), (65.0, original.timestamp), (65.0, original.timestamp)]
    assert spec.latest_attempt() == newest
    assert spec.latest().timestamp == newest.timestamp
    from silk.circuit.test_placement import evidence_index
    associations = list(evidence_index(mod).projection_associations.values())
    assert [a.observation.attempt_id for a in associations] == [
        newest.attempt_id, original.attempt_id, original.attempt_id]
    assert b.results[0].value == later
    mod.close()


@pytest.mark.parametrize('transport', [False, True])
def test_report_capture_survives_live_mutation_and_outer_rollback(pdk, tmp_path, transport):
    mod = _dut(pdk, tmp_path, tmp_path / 'mutation.db')
    test = _wired_test(mod)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='tt')
    spec = test._specs[0]['spec']
    first = spec.latest_attempt()
    live = next(iter(spec.measurements.values()))
    live.value = 99.0
    live.timestamp = first.timestamp + 500.0
    live.notes = 'later mutation'
    if transport:
        path = tmp_path / 'report.json'
        report.save_json(path)
        report = SpecReport.load_json(path)
    db = mod._tracker.db
    with pytest.raises(RuntimeError, match='rollback'):
        with db.transaction():
            mod.store_report(report)
            raise RuntimeError('rollback')
    assert db.conn.execute('SELECT count(*) FROM spec_measurements').fetchone()[0] == 0
    mod.store_report(report)
    row = db.conn.execute('SELECT value, timestamp, notes FROM spec_measurements').fetchone()
    assert tuple(row) == (65.0, first.timestamp, '')
    assert live.value == 99.0 and live.notes == 'later mutation'
    assert spec.latest_attempt() == first
    mod.close()


@pytest.mark.parametrize('transport', [False, True])
@pytest.mark.parametrize('scoped', [False, True])
def test_absent_observation_never_borrows_live_evidence(pdk, tmp_path, transport, scoped):
    mod = _dut(pdk, tmp_path, tmp_path / 'unknown.db')
    test = _wired_test(mod)
    if scoped:
        test.set_temperature(27)
        test.set_supply_voltage(1.8, source='vsup')
    mod.create_circuit_version()
    original = _run_without_simulating(test, corner='tt')
    data = original.to_dict()
    data['results'][0].pop('observation', None)
    report = SpecReport.from_dict(data) if transport else SpecReport(
        original.test_name, original.module_name, dict(original.metrics), corner='tt',
        characterization_context=original.characterization_context,
        operating_conditions=original.operating_conditions)
    if not transport:
        result = original.results[0]
        report.check(result.name, result.metric_name, result.op, result.target,
                     spec_key=result.spec_key, owner_path=result.owner_path)
    spec = test._specs[0]['spec']
    before = spec.attempt_history()
    mod.store_report(report)
    row = mod._tracker.db.conn.execute(
        'SELECT value, timestamp, sim_run_id, notes FROM spec_measurements').fetchone()
    assert tuple(row) == (65.0, None, None, '')
    assert spec.attempt_history() == before
    mod.close()


@pytest.mark.parametrize('field', ['null', 'identity', 'value', 'timestamp', 'conditions', 'notes'])
def test_present_malformed_observation_is_rejected(pdk, tmp_path, field):
    mod = _dut(pdk, tmp_path, tmp_path / 'malformed.db')
    test = _wired_test(mod)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='tt')
    data = report.to_dict()
    observation = data['results'][0]['observation']
    if field == 'null':
        data['results'][0]['observation'] = None
    elif field == 'identity':
        observation['identity'][1] = 'other'
    elif field == 'conditions':
        observation['attempt']['proved_conditions']['backend'] = 'behavioral'
    else:
        observation['attempt']['measurement'][field] = 123.0
    with pytest.raises(ValueError):
        SpecReport.from_dict(data)
    report.results[0]._observation = None if field == 'null' else ('invalid',)
    with pytest.raises(EvidenceMismatchError, match='observation'):
        mod.store_report(report)
    with pytest.raises(ValueError, match='observation'):
        report.to_dict()
    assert mod._tracker.db.conn.execute('SELECT count(*) FROM spec_measurements').fetchone()[0] == 0
    mod.close()


@pytest.mark.parametrize('field', [
    'attempt_id', 'source', 'timestamp', 'kind', 'reason', 'measurement',
    'test_name', 'report_key', 'failure_stage', 'requested_conditions',
    'proved_conditions', 'aggregate',
])
def test_captured_attempt_requires_recorded_fields(pdk, tmp_path, field):
    mod = _dut(pdk, tmp_path, tmp_path / 'missing-attempt.db')
    test = _wired_test(mod)
    report = _run_without_simulating(test, corner='tt')
    data = report.to_dict()
    del data['results'][0]['observation']['attempt'][field]
    with pytest.raises(ValueError, match='observation.attempt'):
        SpecReport.from_dict(data)
    mod.close()


@pytest.mark.parametrize('field', [
    'source', 'value', 'passes', 'timestamp', 'corner', 'temperature', 'voltage',
    'sim_run_id', 'notes', 'sim_type', 'tool_name',
])
def test_captured_measurement_requires_recorded_fields(pdk, tmp_path, field):
    mod = _dut(pdk, tmp_path, tmp_path / 'missing-measurement.db')
    test = _wired_test(mod)
    report = _run_without_simulating(test, corner='tt')
    data = report.to_dict()
    del data['results'][0]['observation']['attempt']['measurement'][field]
    with pytest.raises(ValueError, match='observation.attempt'):
        SpecReport.from_dict(data)
    mod.close()


@pytest.mark.parametrize('identity', [None, ''])
def test_captured_attempt_identity_cannot_request_creation(pdk, tmp_path, identity):
    mod = _dut(pdk, tmp_path, tmp_path / 'null-attempt.db')
    report = _run_without_simulating(_wired_test(mod), corner='tt')
    data = report.to_dict()
    data['results'][0]['observation']['attempt']['attempt_id'] = identity
    with pytest.raises(ValueError, match='observation.attempt'):
        SpecReport.from_dict(data)
    mod.close()


@pytest.mark.parametrize('value', [float('nan'), 65+4j, np.complex128(65+4j)])
def test_unavailable_captured_observation_round_trip_keeps_existing_storage_policy(pdk, tmp_path, value):
    mod = _dut(pdk, tmp_path, tmp_path / 'unavailable.db')
    test = _wired_test(mod, gain=value)
    mod.create_circuit_version()
    report = _run_without_simulating(test, corner='tt')
    original = test._specs[0]['spec'].latest_attempt()
    restored = SpecReport.from_dict(report.to_dict())
    assert restored.results[0]._observation[-1] == original
    receipt = mod.store_report(restored)
    assert not restored.passed
    rows = mod._tracker.db.conn.execute('SELECT value, timestamp FROM spec_measurements').fetchall()
    if isinstance(value, Real):
        assert len(rows) == 1 and rows[0]['value'] is None
        assert rows[0]['timestamp'] == original.timestamp
    else:
        assert rows == [] and 'gain_dB' in receipt['skipped']
    mod.close()
