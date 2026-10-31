"""TASK-128: ``Module.run_test(..., store=True)`` persists a single test's
report through the existing ``store_report`` contract, opt-in only.

The default (``store=False``) is byte-for-byte the old behaviour: no tracker
is touched, no row is written, and the explicit ``store_report(report)``
path stays available. With ``store=True`` every readiness failure that can be
known before the run (no ``db_path``, no bound version, locked/unsealed
version, design drift) is raised before the simulator starts; nothing creates
or changes a circuit version implicitly. Storage relies on TASK-129's
``report.corner`` (unscoped) / the characterization context (scoped) for
provenance, so the tracker's current corner is never substituted. The
receipt lands on ``report.storage``; a failed save raises
``ReportStorageError`` carrying the completed report.
"""
import sqlite3
from unittest.mock import patch

import pytest
from tests._temperature import mark_simulated

from silk.circuit.modules import Module
from silk.circuit.spec_report import SpecReport
from silk.db import UNSCOPED_PROFILE
from silk.exceptions import (
    DatabaseError,
    EvidenceMismatchError,
    EvidenceNotFoundError,
    LockedVersionError,
    ReportStorageError,
    RunArgumentError,
    SimulationAborted,
)
from silk.simulation.analysis import Transient
from silk import sources


# ---------------------------------------------------------------------------
# Helpers (mirror tests/db/test_store_report_run_corner.py; local on purpose)
# ---------------------------------------------------------------------------

def _dut(pdk, tmp_path, db_path, module_name='store_dut'):
    mod = Module(
        'PDK_skywater130', str(tmp_path / module_name), 'dut.cir',
        pdk_details=pdk, module_name=module_name,
        db_path=None if db_path is None else str(db_path),
    )
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})
    mod.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
    mod.add(pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))
    return mod


def _wired_test(mod, name='characterize', gain=65.0, *, cache=False,
                gain_extractor=None):
    """One constant metric + one first-class spec on it, plus an unattached
    second metric, ready to run() with ``netlist.simulate`` stubbed out."""
    from silk.circuit.module_test import ModuleTestbench
    test = ModuleTestbench(mod, name, cache=cache)
    test.add_module_under_test({
        'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0',
    })
    test.add_source(sources.VoltageSource(
        'vsup', positive='vdd', negative='0', dc=1.8))
    test.add_source(sources.VoltageSource(
        'vin_src', positive='vin', negative='0', dc=0.9))
    test.add_sim_type(Transient(step=1e-9, stop=1e-8))
    test.add_metric(
        'gain_dB', gain_extractor or (lambda _ckt: gain))
    test.add_metric('ugf_MHz', lambda _ckt: 12.5)
    spec = test.add_spec('Gain', 'gain_dB', '>=', 60.0, key='gain')
    return test, spec


class _Simulate:
    """Stand-in for ``netlist.simulate`` that counts calls and marks results
    as loaded (or aborts)."""

    def __init__(self, test, *, abort=False):
        self.test = test
        self.abort = abort
        self.calls = 0

    def __call__(self, *_a, **_kw):
        self.calls += 1
        if self.abort:
            raise SimulationAborted(
                'no data rows', simulator='ngspice',
                reason='Timestep too small', data_rows=0,
            )
        mark_simulated(self.test.netlist, _kw.get('corner', 'tt'),
                       getattr(self, 'temperature', 27),
                       voltage=getattr(self, 'voltage', None))


def _patched(test, **kwargs):
    sim = _Simulate(test, **kwargs)
    return sim, patch.object(test.netlist, 'simulate', side_effect=sim)


def _rows(mod):
    """(metric rows, spec-status rows, spec_measurement corners) straight
    from the bound database."""
    conn = mod._tracker.db.conn
    metric_rows = {
        (r['parameter'], r['corner'], r['value'])
        for r in conn.execute(
            "SELECT parameter, corner, value FROM simulation_results "
            "WHERE parameter NOT LIKE 'spec_%'").fetchall()
    }
    status_rows = [
        (r['parameter'], r['corner'], r['value_text'])
        for r in conn.execute(
            "SELECT parameter, corner, value_text FROM simulation_results "
            "WHERE parameter LIKE 'spec_%' ORDER BY rowid").fetchall()
    ]
    measurement_corners = [
        r['corner'] for r in conn.execute(
            "SELECT corner FROM spec_measurements ORDER BY rowid").fetchall()
    ]
    return metric_rows, status_rows, measurement_corners


def _count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ---------------------------------------------------------------------------
# AC1: default / store=False is untouched; explicit store_report still works
# ---------------------------------------------------------------------------

def test_default_run_test_persists_nothing_and_never_opens_the_tracker(
    pdk, tmp_path,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'default.db')
    test, _spec = _wired_test(mod)
    sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', corner='ff')

    assert isinstance(report, SpecReport)
    assert report.corner == 'ff'
    assert report.storage is None
    assert sim.calls == 1
    # No tracker was created, so no database file work happened at all.
    assert mod._tracker is None
    assert not (tmp_path / 'default.db').exists()


def test_store_false_is_the_same_as_the_default(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'false.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    _sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', corner='ff', store=False)

    assert report.storage is None
    conn = mod._tracker.db.conn
    assert _count(conn, 'simulation_results') == 0
    assert _count(conn, 'spec_measurements') == 0
    mod.close()


def test_store_options_without_store_is_rejected_before_running(
    pdk, tmp_path,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'opts-only.db')
    test, _spec = _wired_test(mod)
    sim, ctx = _patched(test)

    with ctx, pytest.raises(ValueError, match='store_options'):
        mod.run_test('characterize', store_options={'tool_name': 'x'})
    assert sim.calls == 0


def test_explicit_store_report_after_a_plain_run_still_works(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'explicit.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    _sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', corner='ff')
    receipt = mod.store_report(report)

    assert receipt == {'metrics_stored': 2, 'specs_stored': 1, 'skipped': {}}
    # The explicit path leaves the report's receipt attribute alone.
    assert report.storage is None
    _metrics, _status, measurement_corners = _rows(mod)
    assert measurement_corners == ['ff']
    mod.close()


# ---------------------------------------------------------------------------
# AC2/AC3/AC5: store=True persists under the resolved corner with a receipt
# ---------------------------------------------------------------------------

def test_store_true_persists_an_unscoped_ff_run_under_ff(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'ff.db')
    test, spec = _wired_test(mod)
    mod.create_circuit_version()
    assert mod._tracker._current_corner == 'tt'
    sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', corner='ff', store=True)

    assert isinstance(report, SpecReport)
    assert sim.calls == 1
    assert report.corner == 'ff'
    assert report.characterization_context is None
    assert report.storage == {
        'metrics_stored': 2, 'specs_stored': 1, 'skipped': {},
        'cached': False,
    }
    metric_rows, status_rows, measurement_corners = _rows(mod)
    assert metric_rows == {('gain_dB', 'ff', 65.0), ('ugf_MHz', 'ff', 12.5)}
    assert status_rows == [('spec_Gain', 'ff', 'pass (gain_dB=65.0 >= 60.0)')]
    assert measurement_corners == ['ff']
    # Readback through the public paths: metric rows and the declaration's
    # stored measurement both say 'ff', unscoped.
    records = mod.query_results(parameter='gain_dB', profile_id=UNSCOPED_PROFILE)
    assert [(r.corner, r.value) for r in records] == [('ff', 65.0)]
    stored = next(r for r in mod.get_stored_specs() if r.spec_key == 'gain')
    assert [(m.corner, m.value, m.passes) for m in stored.measurements] == [
        ('ff', 65.0, True)
    ]
    # The tracker default was never consulted or moved.
    assert mod._tracker._current_corner == 'tt'
    # The run's own live measurement stands alongside.
    assert spec.latest().value == 65.0
    mod.close()


def test_store_true_default_corner_uses_the_pdk_default_not_the_tracker(
    pdk, tmp_path, monkeypatch,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'default-corner.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    monkeypatch.setattr(test.pdk_details, 'default_corner', 'sf')
    mod.set_corner('fs')
    _sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', store=True)

    assert report.corner == 'sf'
    metric_rows, status_rows, measurement_corners = _rows(mod)
    assert {c for _p, c, _v in metric_rows} == {'sf'}
    assert {c for _p, c, _t in status_rows} == {'sf'}
    assert measurement_corners == ['sf']
    mod.close()


def test_store_true_scoped_run_keeps_its_profile_and_exact_pvt(
    pdk, tmp_path,
):
    """Configured/observed TEMP and declared supply derive the profile/PVT;
    a matching legacy temperature assertion does not configure the run."""
    mod = _dut(pdk, tmp_path, tmp_path / 'scoped.db')
    test, spec = _wired_test(mod)
    mod.create_circuit_version()
    test.set_temperature(85)
    # Two grounded DC sources, so silk will not pick the rail for the author
    # (TASK-158): name it. The stub emits no deck, so it states the supply a
    # real launch would have confirmed from one.
    test.set_supply_voltage(1.8, source='vsup')
    _sim, ctx = _patched(test)
    _sim.temperature = 85
    _sim.voltage = 1.8

    with ctx:
        report = mod.run_test(
            'characterize', corner='ff', temperature=85, voltage=1.8,
            store=True,
        )

    context = report.characterization_context
    assert context is not None
    assert (context.corner, context.temperature, context.voltage) == (
        'ff', 85.0, 1.8)
    assert report.storage['metrics_stored'] == 2
    conn = mod._tracker.db.conn
    profiles = conn.execute(
        "SELECT profile_id, owner_path, test_name "
        "FROM characterization_profiles").fetchall()
    assert len(profiles) == 1
    profile_id = profiles[0]['profile_id']
    assert (profiles[0]['owner_path'], profiles[0]['test_name']) == (
        '$', 'characterize')
    for row in conn.execute(
        "SELECT profile_id, corner, temperature, voltage "
        "FROM simulation_results").fetchall():
        assert row['profile_id'] == profile_id
        assert (row['corner'], row['temperature'], row['voltage']) == (
            'ff', 85.0, 1.8)
    measurement = conn.execute(
        "SELECT profile_id, corner, temperature, voltage "
        "FROM spec_measurements").fetchone()
    assert measurement['profile_id'] == profile_id
    assert (measurement['corner'], measurement['temperature'],
            measurement['voltage']) == ('ff', 85.0, 1.8)
    # Profile-scoped readback selects it exactly; the live Spec carries the
    # profile-keyed measurement store_report promoted (existing bookkeeping).
    records = mod.query_results(parameter='gain_dB', profile_id=profile_id)
    assert [(r.corner, r.temperature, r.voltage, r.value) for r in records] == [
        ('ff', 85.0, 1.8, 65.0)
    ]
    with pytest.raises(EvidenceNotFoundError):
        mod.query_results(parameter='gain_dB', profile_id=UNSCOPED_PROFILE)
    assert {m.profile_id for m in spec.measurements.values()} >= {profile_id}
    mod.close()


def test_store_true_scoped_run_with_a_partial_pvt_is_rejected_before_running(
    pdk, tmp_path,
):
    from silk.exceptions import CharacterizationContextError
    mod = _dut(pdk, tmp_path, tmp_path / 'partial.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    sim, ctx = _patched(test)

    with ctx, pytest.raises(CharacterizationContextError):
        mod.run_test('characterize', corner='ff', temperature=85, store=True)
    assert sim.calls == 0
    assert _count(mod._tracker.db.conn, 'simulation_results') == 0
    mod.close()


@pytest.mark.parametrize('key, value', [
    ('corner', 'tt'), ('temperature', 27.0), ('voltage', 1.8),
])
def test_store_options_may_not_carry_pvt(pdk, tmp_path, key, value):
    """The run is the only source of corner/PVT provenance; a conflicting
    (or even agreeing) override in store_options is rejected up front."""
    mod = _dut(pdk, tmp_path, tmp_path / f'opt-{key}.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    sim, ctx = _patched(test)

    with ctx, pytest.raises(ValueError) as excinfo:
        mod.run_test('characterize', corner='ff', store=True,
                     store_options={key: value})
    assert key in str(excinfo.value)
    assert sim.calls == 0
    assert _count(mod._tracker.db.conn, 'simulation_results') == 0
    mod.close()


def test_store_options_forward_tool_name_units_and_conditions(
    pdk, tmp_path,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'options.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    _sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test(
            'characterize', corner='ss', store=True,
            store_options={
                'tool_name': 'ngspice-44',
                'units': {'ugf_MHz': 'MHz'},
                'sim_type': 'tran',
                'conditions': {'cload_f': 2e-12},
            },
        )

    assert report.storage['metrics_stored'] == 2
    conn = mod._tracker.db.conn
    rows = conn.execute(
        "SELECT parameter, unit, tool_name, sim_type, conditions "
        "FROM simulation_results WHERE parameter = 'ugf_MHz'").fetchall()
    assert len(rows) == 1
    assert tuple(rows[0])[:4] == ('ugf_MHz', 'MHz', 'ngspice-44', 'tran')
    assert 'cload_f' in rows[0]['conditions']
    measurement = conn.execute(
        "SELECT tool_name, sim_type FROM spec_measurements").fetchone()
    assert tuple(measurement) == ('ngspice-44', 'tran')
    mod.close()


# ---------------------------------------------------------------------------
# AC4: readiness failures are raised before the simulator starts, and no
# version is created or changed implicitly
# ---------------------------------------------------------------------------

def test_store_true_without_db_path_fails_before_running(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, None)
    test, _spec = _wired_test(mod)
    sim, ctx = _patched(test)

    with ctx, pytest.raises(RuntimeError) as excinfo:
        mod.run_test('characterize', corner='ff', store=True)
    message = str(excinfo.value)
    assert 'db_path' in message and 'create_circuit_version' in message
    assert sim.calls == 0
    assert mod._tracker is None


def test_store_true_without_a_bound_version_fails_before_running(
    pdk, tmp_path,
):
    db_path = tmp_path / 'noversion.db'
    mod = _dut(pdk, tmp_path, db_path)
    test, _spec = _wired_test(mod)
    sim, ctx = _patched(test)

    with ctx, pytest.raises(DatabaseError) as excinfo:
        mod.run_test('characterize', corner='ff', store=True)
    assert 'create_circuit_version' in str(excinfo.value)
    assert sim.calls == 0
    # The tracker opened the database to check, but created nothing.
    assert mod._tracker.circuit_id is None
    assert _count(mod._tracker.db.conn, 'circuit_versions') == 0
    mod.close()


def test_store_true_on_a_locked_version_fails_before_running(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'locked.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    promoted = mod.promote('stable', lock=True)
    assert mod._tracker.circuit_id == promoted
    sim, ctx = _patched(test)

    with ctx, pytest.raises(LockedVersionError):
        mod.run_test('characterize', corner='ff', store=True)
    assert sim.calls == 0
    conn = mod._tracker.db.conn
    assert _count(conn, 'simulation_results') == 0
    assert mod._tracker.circuit_id == promoted
    assert _count(conn, 'circuit_versions') == 2  # nothing added
    mod.close()


def test_store_true_on_a_drifted_design_fails_before_running(pdk, tmp_path):
    """Editing the live design after create_circuit_version() would make
    store_report reject the report; that is now known before the run."""
    mod = _dut(pdk, tmp_path, tmp_path / 'drift.db')
    test, _spec = _wired_test(mod)
    version_id = mod.create_circuit_version()
    mod.add(pdk.nmos1p8('mn2', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
    sim, ctx = _patched(test)

    with ctx, pytest.raises(EvidenceMismatchError):
        mod.run_test('characterize', corner='ff', store=True)
    assert sim.calls == 0
    conn = mod._tracker.db.conn
    assert _count(conn, 'circuit_versions') == 1
    assert mod._tracker.circuit_id == version_id
    mod.close()


# ---------------------------------------------------------------------------
# AC6: failing specs, extractor errors and aborted runs
# ---------------------------------------------------------------------------

def test_failing_spec_is_stored_as_evidence(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'failing.db')
    test, _spec = _wired_test(mod, gain=50.0)
    mod.create_circuit_version()
    _sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', corner='ff', store=True)

    assert report.passed is False
    assert report.storage == {
        'metrics_stored': 2, 'specs_stored': 1, 'skipped': {},
        'cached': False,
    }
    metric_rows, status_rows, measurement_corners = _rows(mod)
    assert ('gain_dB', 'ff', 50.0) in metric_rows
    assert status_rows == [('spec_Gain', 'ff', 'fail (gain_dB=50.0 >= 60.0)')]
    assert measurement_corners == ['ff']
    stored = next(r for r in mod.get_stored_specs() if r.spec_key == 'gain')
    assert [(m.value, m.passes) for m in stored.measurements] == [(50.0, False)]
    mod.close()


def test_extractor_error_is_skipped_not_fabricated(pdk, tmp_path):
    def broken(_ckt):
        raise RuntimeError('no such signal vout')

    mod = _dut(pdk, tmp_path, tmp_path / 'extractor.db')
    test, spec = _wired_test(mod, gain_extractor=broken)
    mod.create_circuit_version()
    _sim, ctx = _patched(test)

    with ctx:
        report = mod.run_test('characterize', corner='ff', store=True)

    assert report.passed is False
    assert report.storage['metrics_stored'] == 1      # ugf_MHz only
    assert report.storage['specs_stored'] == 1        # the FAIL status row
    assert list(report.storage['skipped']) == ['gain_dB']
    assert 'no such signal' in report.storage['skipped']['gain_dB']
    metric_rows, status_rows, measurement_corners = _rows(mod)
    assert metric_rows == {('ugf_MHz', 'ff', 12.5)}
    assert len(status_rows) == 1
    assert status_rows[0][0] == 'spec_Gain' and status_rows[0][2].startswith('fail')
    assert measurement_corners == []                  # no numeric observation
    assert spec.latest() is None                      # none recorded live either
    mod.close()


def test_aborted_simulation_raises_and_stores_nothing(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'abort.db')
    test, spec = _wired_test(mod)
    mod.create_circuit_version()
    sim, ctx = _patched(test, abort=True)

    with ctx, pytest.raises(SimulationAborted):
        mod.run_test('characterize', corner='ff', store=True)
    assert sim.calls == 1
    conn = mod._tracker.db.conn
    assert _count(conn, 'simulation_results') == 0
    assert _count(conn, 'spec_measurements') == 0
    assert spec.latest() is None
    mod.close()


# ---------------------------------------------------------------------------
# AC5/AC7: a failed save surfaces as ReportStorageError with the report,
# the transaction rolls back, and the run's live measurements stand
# ---------------------------------------------------------------------------

def test_storage_failure_raises_report_storage_error_with_the_report(
    pdk, tmp_path,
):
    mod = _dut(pdk, tmp_path, tmp_path / 'rollback.db')
    test, spec = _wired_test(mod)
    mod.create_circuit_version()
    db = mod._tracker.db
    original_insert = db.insert_spec_measurement

    def failing_insert(*args, **kwargs):
        raise sqlite3.OperationalError('injected: disk I/O error')

    sim, ctx = _patched(test)
    with ctx, patch.object(db, 'insert_spec_measurement', failing_insert), \
            pytest.raises(ReportStorageError) as excinfo:
        mod.run_test('characterize', corner='ff', store=True)

    error = excinfo.value
    assert sim.calls == 1
    assert isinstance(error.__cause__, sqlite3.OperationalError)
    assert 'injected' in str(error)
    # The completed report is not lost, and it carries no receipt.
    report = error.report
    assert isinstance(report, SpecReport)
    assert report.corner == 'ff'
    assert report.metrics['gain_dB'] == 65.0
    assert report.storage is None
    # Every row of the report was rolled back (metrics wrote before the
    # spec measurement raised).
    assert _count(db.conn, 'simulation_results') == 0
    assert _count(db.conn, 'spec_measurements') == 0
    # The live measurement the run recorded is NOT undone.
    assert spec.latest().value == 65.0
    # The database is usable again: the same report stores cleanly.
    db.insert_spec_measurement = original_insert
    assert mod.store_report(report)['specs_stored'] == 1
    mod.close()


def test_delayed_report_and_declaration_guard_through_the_flag(pdk, tmp_path):
    """Later measurements permit delayed storage; declaration drift refuses."""
    mod = _dut(pdk, tmp_path, tmp_path / 'stale.db')
    gains = iter([65.0, 66.0])
    test, _spec = _wired_test(mod, gain_extractor=lambda _c: next(gains))
    mod.create_circuit_version()
    _sim, ctx = _patched(test)

    with ctx:
        old = mod.run_test('characterize', corner='ff')
        new = mod.run_test('characterize', corner='ff', store=True)
    assert new.storage['specs_stored'] == 1
    mod.store_report(old)
    rows = mod._tracker.db.conn.execute(
        'SELECT value FROM spec_measurements ORDER BY rowid').fetchall()
    assert [row[0] for row in rows] == [66.0, 65.0]
    old.results[0].target = 61.0
    with pytest.raises(EvidenceMismatchError):
        mod.store_report(old)
    assert mod._tracker.db.conn.execute('SELECT count(*) FROM spec_measurements').fetchone()[0] == 2
    mod.close()


# ---------------------------------------------------------------------------
# AC8: repeat stores append; a cache hit stores the cached report again and
# says so; batch helpers are untouched
# ---------------------------------------------------------------------------

def test_repeat_store_appends_observations(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'append.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    sim, ctx = _patched(test)

    with ctx:
        first = mod.run_test('characterize', corner='ff', store=True)
        second = mod.run_test('characterize', corner='ff', store=True)

    assert sim.calls == 2
    assert first is not second
    assert first.storage['cached'] is False
    assert second.storage['cached'] is False
    metric_rows, status_rows, measurement_corners = _rows(mod)
    conn = mod._tracker.db.conn
    assert _count(conn, 'simulation_results') == 6   # 2 x (2 metrics + 1 status)
    assert measurement_corners == ['ff', 'ff']
    mod.close()


def test_cache_true_still_stores_a_fresh_report_every_call(
    pdk, tmp_path,
):
    """Formerly test_cache_hit_stores_the_cached_report_again_and_flags_it:
    verified a same-corner repeat call skipped simulation, reused the first
    report, and the receipt said cached=True. The report cache is disabled
    in 0.2.0 (TASK-284): cache=True now only warns at construction and every
    run_test() simulates, so a repeat call behaves exactly like
    test_repeat_store_appends_observations above -- cached is always
    False."""
    mod = _dut(pdk, tmp_path, tmp_path / 'cache.db')
    with pytest.warns(UserWarning, match='TASK-130'):
        test, _spec = _wired_test(mod, cache=True)
    mod.create_circuit_version()
    sim, ctx = _patched(test)

    with ctx:
        first = mod.run_test('characterize', corner='ff', store=True)
        second = mod.run_test('characterize', corner='ff', store=True)
        other = mod.run_test('characterize', corner='ss', store=True)

    assert sim.calls == 3                    # ff, ff, ss -- no cache hit
    assert second is not first
    assert first.storage['cached'] is False
    assert second.storage['cached'] is False
    assert other.storage['cached'] is False
    _metrics, _status, measurement_corners = _rows(mod)
    assert measurement_corners == ['ff', 'ff', 'ss']
    mod.close()


def test_run_all_tests_and_run_corners_do_not_learn_the_flag(pdk, tmp_path):
    mod = _dut(pdk, tmp_path, tmp_path / 'batch.db')
    test, _spec = _wired_test(mod)
    mod.create_circuit_version()
    sim, ctx = _patched(test)

    with ctx:
        # TASK-166: run_all_tests used to fold this into its result map as
        # {'error': "run() got an unexpected keyword argument 'store'"} -- a
        # caller-setup error reported as though a circuit had misbehaved,
        # because Python raises it as a builtin and a builtin cannot carry
        # the ConfigurationError marker. The arguments are now bound to each
        # test's run signature before the call, so it is refused instead.
        with pytest.raises(RunArgumentError, match='store') as refused:
            mod.run_all_tests(recursive=False, store=True)
        with pytest.raises(TypeError):
            test.run_corners(corners=('tt',), store=True)

    # Neither runner learned the flag, and neither ran or persisted anything.
    # RunArgumentError is still a TypeError, so the two refusals have the same
    # shape to any caller that was already handling the run_corners one.
    assert isinstance(refused.value, TypeError)
    assert sim.calls == 0
    assert _count(mod._tracker.db.conn, 'simulation_results') == 0
    mod.close()


# ---------------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------------

def test_run_test_signature_is_keyword_only_and_additive():
    import inspect
    params = inspect.signature(Module.run_test).parameters
    assert params['store'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['store'].default is False
    assert params['store_options'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['store_options'].default is None
    assert params['config'].kind is inspect.Parameter.VAR_KEYWORD
    assert SpecReport('t', 'm', {}).storage is None


def test_report_storage_error_is_a_database_error_carrying_the_report():
    from silk.exceptions import DatabaseError
    report = SpecReport('t', 'm', {})
    error = ReportStorageError('boom', report=report)
    assert isinstance(error, DatabaseError)
    assert error.report is report
    assert ReportStorageError('bare').report is None
