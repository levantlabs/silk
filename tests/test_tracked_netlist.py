"""
Tests for database tracking via Module(db_path=...).
"""

import os
import json
import pytest
import copy
import sqlite3

from silk.db.circuit_db import Database, DesignStage
from silk.circuit.modules import Module
from silk import sources
from silk.exceptions import DatabaseError, LockedVersionError


def _allow_sql(*_):
    """Restore SQL access with a callback supported on Python 3.10 too."""
    return sqlite3.SQLITE_OK


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracked_module(pdk, tmp_dir):
    """Module with tracking via db_path (new API)."""
    m = Module(
        'PDK_skywater130', tmp_dir, 'tracked_test.cir',
        pdk_details=pdk,
        module_name='test_amp',
        db_path=':memory:',
    )
    yield m
    m.close()


@pytest.fixture
def untracked_module(pdk, tmp_dir):
    """Module without tracking (no db_path)."""
    return Module(
        'PDK_skywater130', tmp_dir, 'no_track.cir',
        pdk_details=pdk,
    )


# ===========================================================================
# Module(db_path=...) API
# ===========================================================================


class TestModuleTrackingConstructor:

    def test_tracker_not_created_until_needed(self, tracked_module):
        """Tracker is lazy — not created in __init__."""
        assert tracked_module._tracker is None

    def test_tracker_created_on_first_call(self, tracked_module):
        """First tracking call creates the tracker."""
        tracked_module.create_circuit_version(description='test')
        assert tracked_module._tracker is not None

    def test_no_db_path_raises(self, untracked_module):
        """Calling a tracking method without db_path raises RuntimeError."""
        with pytest.raises(RuntimeError, match="No db_path provided"):
            untracked_module.create_circuit_version(description='test')


class TestModuleCreateCircuitVersion:

    def test_returns_circuit_id(self, tracked_module):
        """create_circuit_version returns a circuit_id string."""
        cid = tracked_module.create_circuit_version(description='test')
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_circuit_appears_in_db(self, tracked_module):
        """Created circuit is retrievable from the database."""
        cid = tracked_module.create_circuit_version(description='test')
        info = tracked_module._tracker.db.get_circuit_by_id(cid)
        assert info is not None
        assert info['circuit_name'] == 'test_amp'

    def test_uses_module_name(self, pdk, tmp_dir):
        """circuit_name in DB comes from module_name."""
        m = Module('PDK_skywater130', tmp_dir, 'name_test.cir',
                   pdk_details=pdk, module_name='my_ldo', db_path=':memory:')
        cid = m.create_circuit_version(description='test')
        info = m._tracker.db.get_circuit_by_id(cid)
        assert info['circuit_name'] == 'my_ldo'
        m.close()

    def test_unnamed_module_derives_from_filename(self, pdk, tmp_dir):
        """Without module_name, circuit_name derives from netlist filename."""
        m = Module('PDK_skywater130', tmp_dir, 'ldo_example.cir',
                   pdk_details=pdk, db_path=':memory:')
        cid = m.create_circuit_version(description='test')
        info = m._tracker.db.get_circuit_by_id(cid)
        assert info['circuit_name'] == 'ldo_example'
        m.close()


class TestModuleStoreResults:

    def test_results_stored_in_db(self, tracked_module):
        """store_results stores parameter names, values, and units."""
        tracked_module.create_circuit_version(description='test')
        results = {
            'gain': (45.2, 'dB'),
            'power': (2.3, 'mW'),
        }
        tracked_module.store_results(results, corner='tt', temperature=27)

        db = tracked_module._tracker.db
        db_results = db.get_simulation_results(tracked_module._tracker.circuit_id)
        assert len(db_results) == 2
        params = {r.parameter: r for r in db_results}
        assert params['gain'].value == 45.2
        assert params['gain'].unit == 'dB'
        assert params['power'].value == 2.3

    def test_results_with_corner_temperature(self, tracked_module):
        """Corner and temperature are stored correctly."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_results({'vout': (1.8, 'V')}, corner='ss', temperature=125)

        db = tracked_module._tracker.db
        db_results = db.get_simulation_results(tracked_module._tracker.circuit_id)
        r = db_results[0]
        assert r.corner == 'ss'
        assert r.temperature == 125.0

    def test_raises_without_circuit_version(self, tracked_module):
        """store_results raises if create_circuit_version not called first."""
        # Force tracker creation by accessing it
        tracked_module._ensure_tracker()
        with pytest.raises(DatabaseError, match="No circuit version created"):
            tracked_module.store_results({'gain': (10, 'dB')})


class TestModuleStoreResult:

    def test_store_single_result(self, tracked_module):
        """store_result stores a single parameter."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_result('bandwidth', 1e6, 'Hz', corner='tt', temperature=27)

        db = tracked_module._tracker.db
        db_results = db.get_simulation_results(tracked_module._tracker.circuit_id)
        assert len(db_results) == 1
        assert db_results[0].parameter == 'bandwidth'
        assert db_results[0].value == 1e6
        assert db_results[0].unit == 'Hz'


class TestModuleStorePassFail:

    def test_pass_fail_recorded(self, tracked_module):
        """store_pass_fail records a pass/fail metric in DB."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_pass_fail('output_range', 'PASS', details='1.0V to 1.6V')

        db = tracked_module._tracker.db
        db_results = db.get_simulation_results(tracked_module._tracker.circuit_id)
        assert len(db_results) == 1
        r = db_results[0]
        assert r.parameter == 'spec_output_range'
        assert 'PASS' in r.value_text


class TestModuleSetCorner:

    def test_updates_current_corner(self, tracked_module):
        """set_corner updates the tracker's internal corner state."""
        tracked_module.set_corner('ff', temperature=85, voltage=1.98)
        t = tracked_module._tracker
        assert t._current_corner == 'ff'
        assert t._current_temp == 85
        assert t._current_voltage == 1.98

    def test_corner_used_in_subsequent_store(self, tracked_module):
        """Corner set via set_corner is used by store_results."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.set_corner('ss', temperature=-40)
        tracked_module.store_results({'gain': (38.0, 'dB')})

        db = tracked_module._tracker.db
        db_results = db.get_simulation_results(tracked_module._tracker.circuit_id)
        assert db_results[0].corner == 'ss'
        assert db_results[0].temperature == -40.0


class TestModuleViewSummary:

    def test_returns_string(self, tracked_module):
        """view_summary returns a summary string."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_results({'gain': (45.2, 'dB')}, corner='tt')
        summary = tracked_module.view_summary()
        assert summary is not None
        assert isinstance(summary, str)


class TestModulePromote:

    def test_promote_stable(self, tracked_module):
        """promote('stable') returns a new circuit ID."""
        tracked_module.create_circuit_version(description='test')
        new_id = tracked_module.promote('stable')
        assert isinstance(new_id, str)
        assert len(new_id) > 0

    def test_promote_invalid_stage(self, tracked_module):
        """promote with invalid stage raises ValueError."""
        tracked_module.create_circuit_version(description='test')
        with pytest.raises(ValueError, match="Invalid stage"):
            tracked_module.promote('nonexistent_stage')

    def test_promote_raises_without_circuit(self, tracked_module):
        """promote raises DatabaseError if no circuit version exists."""
        tracked_module._ensure_tracker()
        with pytest.raises(DatabaseError, match="No circuit version created"):
            tracked_module.promote('stable')


class TestModuleGetCorners:

    def test_get_corner_summary(self, tracked_module):
        """get_corner_summary returns a dictionary."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_results({'gain': (45.0, 'dB')}, corner='tt')
        result = tracked_module.get_corner_summary()
        assert isinstance(result, dict)

    def test_get_corners_for_parameter(self, tracked_module):
        """get_corners_for_parameter returns results across corners."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_results({'gain': (45.0, 'dB')}, corner='tt')
        tracked_module.store_results({'gain': (38.0, 'dB')}, corner='ss')
        results = tracked_module.get_corners_for_parameter('gain')
        assert isinstance(results, list)
        assert len(results) == 2


class TestModuleClose:

    def test_close_clears_tracker(self, pdk, tmp_dir):
        """close() clears the tracker and doesn't raise."""
        m = Module('PDK_skywater130', tmp_dir, 'close_test.cir',
                   pdk_details=pdk, db_path=':memory:')
        m.create_circuit_version(description='test')
        assert m._tracker is not None
        m.close()
        assert m._tracker is None

    def test_close_noop_without_tracker(self, untracked_module):
        """close() on untracked module is a no-op."""
        untracked_module.close()  # should not raise


class TestModuleExportToJson:

    def test_export_creates_json(self, tracked_module, tmp_dir):
        """export_to_json writes a valid JSON file."""
        tracked_module.create_circuit_version(description='test')
        tracked_module.store_results({'gain': (45.0, 'dB')}, corner='tt')
        out = os.path.join(tmp_dir, 'export.json')
        tracked_module.export_to_json(out)
        with open(out) as f:
            data = json.load(f)
        assert 'circuits' in data
        assert data['circuits'][0]['name'] == 'test_amp'

    def test_export_raises_without_circuit(self, tracked_module):
        """export_to_json raises if no circuit version created."""
        tracked_module._ensure_tracker()
        with pytest.raises(RuntimeError, match="No circuit version created"):
            tracked_module.export_to_json('/tmp/test_export.json')


class TestModuleAddTransistor:

    def test_add_transistor(self, tracked_module, monkeypatch):
        """add_transistor calls through to db.add_transistor."""
        tracked_module.create_circuit_version(description='test')
        called = {}
        monkeypatch.setattr(tracked_module._tracker.db, 'add_transistor',
                            lambda **kw: called.update(kw), raising=False)
        tracked_module.add_transistor('M1', 'nmos', width=1e-6, length=0.15e-6,
                                      multiplier=2, model='sky130_fd_pr__nfet_01v8')
        assert called.get('name') == 'M1'
        assert called.get('transistor_type') == 'nmos'


class TestModuleAddNetToDB:

    def test_add_net_to_db(self, tracked_module):
        """A sealed snapshot rejects direct structural mutation."""
        tracked_module.create_circuit_version(description='test')
        with pytest.raises(DatabaseError, match='building'):
            tracked_module.add_net_to_db('vdd', net_type='power')


class TestModuleAutoTrackStructure:

    def test_devices_stored_on_create(self, pdk, tmp_dir):
        """create_circuit_version auto-stores devices from elem_dict."""
        m = Module('PDK_skywater130', tmp_dir, 'autotrack.cir',
                   pdk_details=pdk, module_name='inv', db_path=':memory:')

        mn = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=4.0, l=0.15, m=2)
        mp = pdk.pmos1p8('mp1', d='out', g='in', s='vdd', b='vdd', w=8.0, l=0.15, m=2)
        m.add(mn, mp)

        cid = m.create_circuit_version(description='test')

        summary = m._tracker.db.get_circuit_summary('inv')
        assert len(summary['devices']) == 2
        names = {d['name'] for d in summary['devices']}
        assert names == {'mn1', 'mp1'}

        # Check parameters were stored
        for d in summary['devices']:
            props = json.loads(d['properties']) if d['properties'] else {}
            assert 'w' in props
            assert 'l' in props
            assert 'm' in props

        m.close()

    def test_nets_stored_on_create(self, pdk, tmp_dir):
        """create_circuit_version auto-stores nets from net_dict."""
        m = Module('PDK_skywater130', tmp_dir, 'autotrack.cir',
                   pdk_details=pdk, module_name='inv', db_path=':memory:')

        mn = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=4.0, l=0.15)
        m.add(mn)
        m.create_circuit_version(description='test')

        summary = m._tracker.db.get_circuit_summary('inv')
        net_names = {n['name'] for n in summary['nets']}
        assert 'out' in net_names
        assert 'in' in net_names
        assert 'gnd' in net_names

        m.close()

    def test_net_types_classified(self, pdk, tmp_dir):
        """Net types are classified by name (power, ground, signal)."""
        m = Module('PDK_skywater130', tmp_dir, 'autotrack.cir',
                   pdk_details=pdk, module_name='inv', db_path=':memory:')

        mn = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        mp = pdk.pmos1p8('mp1', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15)
        m.add(mn, mp)
        m.create_circuit_version(description='test')

        summary = m._tracker.db.get_circuit_summary('inv')
        net_types = {n['name']: n['net_type'] for n in summary['nets']}
        assert net_types['vdd'] == 'power'
        assert net_types['gnd'] == 'ground'
        assert net_types['out'] == 'signal'

        m.close()

    def test_device_type_from_class_name(self, pdk, tmp_dir):
        """Device type is derived from class name (nmos1p8, pmos1p8)."""
        m = Module('PDK_skywater130', tmp_dir, 'autotrack.cir',
                   pdk_details=pdk, module_name='inv', db_path=':memory:')

        mn = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        mp = pdk.pmos1p8('mp1', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15)
        m.add(mn, mp)
        m.create_circuit_version(description='test')

        summary = m._tracker.db.get_circuit_summary('inv')
        types = {d['name']: d['type'] for d in summary['devices']}
        assert types['mn1'] == 'nmos1p8'
        assert types['mp1'] == 'pmos1p8'

        m.close()

    def test_empty_circuit_no_crash(self, pdk, tmp_dir):
        """create_circuit_version works on a circuit with no devices."""
        m = Module('PDK_skywater130', tmp_dir, 'empty.cir',
                   pdk_details=pdk, module_name='empty', db_path=':memory:')
        cid = m.create_circuit_version(description='empty')
        summary = m._tracker.db.get_circuit_summary('empty')
        assert len(summary['devices']) == 0
        assert len(summary['nets']) == 0
        m.close()


class TestModuleFullWorkflow:

    def test_create_add_generate_store(self, pdk, tmp_dir, mock_ngspice):
        """Full flow: add devices → create version → store results."""
        m = Module('PDK_skywater130', tmp_dir, 'workflow.cir',
                   pdk_details=pdk, module_name='amp', db_path=':memory:')

        # Add devices BEFORE creating version so they get auto-tracked
        mn = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        m.add(mn)
        assert 'mn1' in m.elem_dict

        m.add_source(sources.VoltageSource('vdd_src', positive='vdd', negative='gnd', dc=1.8))

        cid = m.create_circuit_version(description='full flow test')
        assert cid is not None

        # Verify the device and DUT-owned source were auto-stored.
        summary = m._tracker.db.get_circuit_summary('amp')
        assert {row['name']: row['type'] for row in summary['devices']} == {
            'mn1': 'nmos1p8', 'vdd_src': 'silk_source',
        }

        # Store results
        m.store_results({
            'gain': (45.2, 'dB'),
            'power': (2.3, 'mW'),
        }, corner='tt', temperature=27)

        # Verify results in DB
        db_results = m._tracker.db.get_simulation_results(cid)
        assert len(db_results) == 2
        params = {r.parameter for r in db_results}
        assert params == {'gain', 'power'}

        m.close()


class TestRerunIdempotence:
    """2026-08-23: create_circuit_version is get-or-create keyed by structure
    hash. Re-running an unchanged script reuses its row (before: minted a
    duplicate same-label row per session — and crashed once duplicate labels
    were rejected; caught live by examples/ldo_design.py on its second run)."""

    def test_same_structure_reuses_the_row(self, tracked_module):
        c1 = tracked_module.create_circuit_version(description='run 1')
        c2 = tracked_module.create_circuit_version(description='run 2')
        assert c1 == c2
        n = tracked_module._tracker.db.conn.execute(
            "SELECT COUNT(*) FROM circuit_versions").fetchone()[0]
        assert n == 1

    def test_changed_structure_gets_a_new_row(self, tracked_module):
        from silk.elements import elements
        c1 = tracked_module.create_circuit_version(description='run 1')
        tracked_module.add(
            elements.resistor('r_new', p='vin', n='0', value=123.0))
        c2 = tracked_module.create_circuit_version(description='run 2')
        assert c1 != c2


class TestPromoteHardening:
    """2026-08-23 G15 review: promote() is rerun-safe (get-or-create at the
    same target stage), rebinds the tracker to the promoted row, and the lock
    it applies actually protects subsequent writes — before the rebind, every
    post-promote write silently targeted the stale, UNLOCKED pre-promotion
    row, so promote(lock=True) protected a row nothing wrote to again."""

    def test_promote_rebinds_tracker_to_promoted_row(self, tracked_module):
        c1 = tracked_module.create_circuit_version(description='v')
        new_id = tracked_module.promote('stable')
        t = tracked_module._tracker
        assert t.circuit_id == new_id
        assert new_id != c1
        assert t.stage == DesignStage.STABLE
        assert t.version == '1.0.stable'

    def test_write_after_locked_promote_raises(self, tracked_module):
        tracked_module.create_circuit_version(description='v')
        tracked_module.promote('stable')
        with pytest.raises(LockedVersionError, match='locked'):
            tracked_module.store_results({'gain_dB': (1.0, 'dB')}, corner='tt')

    def test_promote_lock_false_stays_writable(self, tracked_module):
        tracked_module.create_circuit_version(description='v')
        tracked_module.promote('stable', lock=False)
        tracked_module.store_results({'gain_dB': (1.0, 'dB')}, corner='tt')

    def test_promote_rerun_reuses_row(self, pdk, tmp_dir, tmp_path):
        """The quickstart §12 flow, twice against the same db file — the
        duplicate-label rejection used to crash the second promote()."""
        db_path = str(tmp_path / 'promote.db')
        ids = []
        for _ in range(2):
            m = Module('PDK_skywater130', tmp_dir, 'p.cir', pdk_details=pdk,
                       module_name='promo_ckt', db_path=db_path)
            m.create_circuit_version(description='v')
            ids.append(m.promote('stable'))
            m.close()
        assert ids[0] == ids[1]

    def test_label_collision_at_other_stage_raises(self, tracked_module):
        source_id = tracked_module.create_circuit_version(description='v')
        t = tracked_module._ensure_tracker()
        identity = t.compute_snapshot_identity_for(tracked_module)
        t.db.create_circuit('test_amp', '1.0.stable',
                            stage=DesignStage.TAPEOUT,
                            identity=identity,
                            parent_version_id=source_id)
        with pytest.raises(DatabaseError, match='target label'):
            tracked_module.promote('stable')

    def test_structure_store_rejects_locked_version(self, tracked_module):
        """The raw device INSERT/UPDATE in store_structure bypasses the
        guarded add_device — the explicit lock check covers it."""
        cid = tracked_module.create_circuit_version(description='v')
        t = tracked_module._tracker
        t.db.lock_circuit('test_amp', '1.0')
        from silk.db.structure_tracker import CircuitStructureTracker
        st = CircuitStructureTracker(t.db, cid)
        structure = st.extract_structure(tracked_module)
        h, mh = st.compute_structure_hash(structure)
        with pytest.raises(LockedVersionError, match='locked'):
            st.store_structure(structure, h, mh)


# ===========================================================================
# store_report — one-call SpecReport → db bridge
# ===========================================================================

class TestStoreReport:
    """store_report() bridges silk.circuit.spec_report.SpecReport (what
    ModuleTestbench.run() returns) into store_results/store_pass_fail in one
    call, resolving units from the metric-name suffix convention."""

    def _report(self):
        from silk.circuit.spec_report import SpecReport
        return SpecReport('amp_test', 'test_amp', {
            'gain_dB': 65.0,
            'ugf_MHz': 12.5,
            'some_metric': 3.0,
            'offset_extraction': 'Error: node not found',
        })

    def test_metrics_stored_with_suffix_units(self, tracked_module):
        tracked_module.create_circuit_version(description='v')
        report = self._report()
        result = tracked_module.store_report(report, corner='tt')

        assert result['metrics_stored'] == 3
        assert result['specs_stored'] == 0
        assert result['skipped'] == {
            'offset_extraction': "non-numeric value: 'Error: node not found'"
        }

        t = tracked_module._tracker
        rows = {r[0]: r[1] for r in t.db.conn.execute(
            "SELECT parameter, unit FROM simulation_results "
            "WHERE circuit_version_id = ?", (t.circuit_id,)).fetchall()}
        assert rows['gain_dB'] == 'dB'
        assert rows['ugf_MHz'] == 'MHz'
        assert rows['some_metric'] == ''
        assert 'offset_extraction' not in rows

    def test_explicit_units_override_suffix(self, tracked_module):
        tracked_module.create_circuit_version(description='v')
        report = self._report()
        tracked_module.store_report(report, corner='tt',
                                    units={'gain_dB': 'decibels'})

        t = tracked_module._tracker
        row = t.db.conn.execute(
            "SELECT unit FROM simulation_results "
            "WHERE circuit_version_id = ? AND parameter = 'gain_dB'",
            (t.circuit_id,)).fetchone()
        assert row[0] == 'decibels'

    def test_spec_results_stored_pass_and_fail(self, tracked_module):
        gain_spec = tracked_module.declare_spec(
            'gain spec', 'gain_dB', '>', 60.0, key='gain',
        )
        ugf_spec = tracked_module.declare_spec(
            'ugf spec', 'ugf_MHz', '>', 100.0, key='ugf',
        )
        tracked_module.create_circuit_version(description='v')
        report = self._report()
        report.check(
            'gain spec', 'gain_dB', '>', 60.0,
            spec_key=gain_spec.key, owner_path='$',
        )   # passes: 65 > 60
        report.check(
            'ugf spec', 'ugf_MHz', '>', 100.0,
            spec_key=ugf_spec.key, owner_path='$',
        )    # fails: 12.5 > 100

        result = tracked_module.store_report(report, corner='tt')
        assert result['specs_stored'] == 2

        t = tracked_module._tracker
        rows = t.db.conn.execute(
            "SELECT parameter, value_text FROM simulation_results "
            "WHERE circuit_version_id = ? AND parameter LIKE 'spec_%'",
            (t.circuit_id,)).fetchall()
        by_param = {r[0]: r[1] for r in rows}
        assert by_param['spec_gain spec'].startswith('pass')
        assert by_param['spec_ugf spec'].startswith('fail')

    def test_no_spec_results_stores_none(self, tracked_module):
        tracked_module.create_circuit_version(description='v')
        report = self._report()
        result = tracked_module.store_report(report, corner='tt')

        t = tracked_module._tracker
        rows = t.db.conn.execute(
            "SELECT parameter FROM simulation_results "
            "WHERE circuit_version_id = ? AND parameter LIKE 'spec_%'",
            (t.circuit_id,)).fetchall()
        assert rows == []
        assert result['specs_stored'] == 0

    def test_no_circuit_version_raises(self, tracked_module):
        report = self._report()
        with pytest.raises(DatabaseError, match='No circuit version created'):
            tracked_module.store_report(report)

    def test_write_after_locked_promote_raises(self, tracked_module):
        tracked_module.create_circuit_version(description='v')
        tracked_module.promote('stable')
        report = self._report()
        with pytest.raises(LockedVersionError, match='locked'):
            tracked_module.store_report(report, corner='tt')


# TASK-511: each tracker write has the same applicability precondition.
_TRACKER_WRITES = [
    ('store_results', lambda t: t.store_results({'x': (1., 'V')})),
    ('empty_results', lambda t: t.store_results({})),
    ('store_result', lambda t: t.store_result('x', 1., 'V')),
    ('store_pass_fail', lambda t: t.store_pass_fail('gain', 'PASS')),
    ('store_report', lambda t: t.store_report(None)),
    ('add_transistor', lambda t: t.add_transistor('m1', 'nmos', 1., 1.)),
    ('add_net', lambda t: t.add_net('extra')),
    ('promote', lambda t: t.promote('stable', lock=False)),
    ('require_report_storage_ready', lambda t: t.require_report_storage_ready()),
]


@pytest.mark.parametrize('name,write', _TRACKER_WRITES, ids=[x[0] for x in _TRACKER_WRITES])
@pytest.mark.parametrize('change', ['structure', 'specs'])
def test_tracker_write_rejects_stale_identity(tracked_module, name, write, change):
    from silk.exceptions import EvidenceMismatchError
    m = tracked_module
    m.create_circuit_version()
    t = m._tracker
    if change == 'structure':
        m.add_net('changed')
    else:
        m.declare_spec('gain', 'gain', '>=', 1., key='gain')
    before = list(t.db.conn.iterdump())
    with pytest.raises(EvidenceMismatchError):
        write(t)
    assert list(t.db.conn.iterdump()) == before


@pytest.mark.parametrize('name,write', _TRACKER_WRITES, ids=[x[0] for x in _TRACKER_WRITES])
def test_tracker_write_requires_bound_version(tracked_module, name, write):
    with pytest.raises(DatabaseError, match='create_circuit_version'):
        write(tracked_module._ensure_tracker())


def test_pass_fail_preserves_observation_metadata(tracked_module):
    m = tracked_module
    m.create_circuit_version()
    m.set_corner('ff', temperature=90, voltage=2.)
    m._tracker.store_pass_fail('gain', 'PASS', 'good', source='user', conditions={'tag': 'x'})
    row = m._tracker.db.conn.execute('SELECT * FROM simulation_results').fetchone()
    assert row['parameter'] == 'spec_gain'
    assert row['value_text'] == 'PASS (good)'
    assert row['source'] == 'user'
    for key in ('value', 'unit', 'corner', 'temperature', 'voltage', 'tool_name', 'sim_type'):
        assert row[key] is None
    assert json.loads(row['conditions']) == {'tag': 'x'}


def _binding(t):
    return t.circuit_id, t.stage, t.version


def _change_binding(t, operation):
    if operation == 'promote':
        return t.promote('stable', new_version='reviewed', lock=False)
    if operation == 'ensure':
        return t.ensure_version(t._module)
    return t.create_circuit_version()


@pytest.mark.parametrize('operation', ['promote', 'create', 'ensure'])
@pytest.mark.parametrize('reuse', [False, True])
def test_binding_rollback_restores_previous_state(tracked_module, operation, reuse):
    m = tracked_module
    t = m._ensure_tracker()
    t.create_circuit_version()
    old = _binding(t)
    if operation != 'promote':
        m.add_net('changed')
    if reuse:
        # A second tracker prepares the target without changing this binding.
        from silk.db import CircuitTracker
        other = CircuitTracker(m, ':memory:')
        other.db.close()
        other.db = t.db
        other.circuit_id, other.stage, other.version = old
        target = _change_binding(other, operation)
    with pytest.raises(ValueError, match='abort'):
        with t.db.transaction():
            new = _change_binding(t, operation)
            assert t.circuit_id == new != old[0]
            t.store_result('during', 1., 'V')
            raise ValueError('abort')
    assert _binding(t) == old
    row = t.db.conn.execute('SELECT version_id FROM circuit_versions WHERE version_id=?', (new,)).fetchone()
    assert (row is not None) == reuse
    if reuse:
        assert new == target
    if operation != 'promote':
        m.remove_net('changed')
    t.store_result('after', 2., 'V')
    assert t.db.get_simulation_results(old[0])[0].parameter == 'after'


@pytest.mark.parametrize('operation', ['create', 'ensure'])
def test_initial_binding_rollback_restores_unbound_tuple(tracked_module, operation):
    t = tracked_module._ensure_tracker()
    old = _binding(t)
    with pytest.raises(ValueError):
        with t.db.transaction():
            _change_binding(t, operation)
            raise ValueError('abort')
    assert _binding(t) == old
    assert t.db.conn.execute('SELECT COUNT(*) FROM circuit_versions').fetchone()[0] == 0
    _change_binding(t, operation)
    t.store_result('after', 1., 'V')


def test_grouped_locked_promotion_rejects_following_write(tracked_module):
    m = tracked_module
    old = m.create_circuit_version()
    t = m._tracker
    with t.db.transaction():
        new = t.promote('stable', new_version='locked')
        assert _binding(t) == (new, DesignStage.STABLE, 'locked')
        with pytest.raises(LockedVersionError):
            t.store_result('wrong_target', 1., 'V')
    assert t.circuit_id == new
    assert not t.db.get_simulation_results(old)


@pytest.mark.parametrize('operation', ['create', 'ensure'])
def test_grouped_creation_writes_to_new_version(tracked_module, operation):
    m = tracked_module
    old = m.create_circuit_version()
    t = m._tracker
    m.add_net('changed')
    with t.db.transaction():
        new = _change_binding(t, operation)
        assert t.circuit_id == new != old
        t.store_result('new_only', 1., 'V')
    assert t.circuit_id == new
    assert t.db.get_simulation_results(new, include_ancestors=False)[0].parameter == 'new_only'
    assert not t.db.get_simulation_results(old)


def test_binding_nested_rollback_restores_only_inner_change(tracked_module):
    m = tracked_module
    m.create_circuit_version()
    t = m._tracker
    with t.db.transaction():
        stable = t.promote('stable', lock=False)
        outer = _binding(t)
        with pytest.raises(ValueError):
            with t.db.transaction():
                t.promote('post_layout', lock=False)
                raise ValueError('inner')
        assert _binding(t) == outer
        t.store_result('kept', 1., 'V')
    assert t.circuit_id == stable


def test_multiple_bindings_rollback_in_reverse_order(tracked_module):
    m = tracked_module
    m.create_circuit_version()
    t = m._tracker
    old = _binding(t)
    with pytest.raises(ValueError):
        with t.db.transaction():
            t.promote('stable', lock=False)
            with t.db.transaction():
                t.promote('post_layout', lock=False)
            raise ValueError('outer')
    assert _binding(t) == old
    t.store_result('after', 1., 'V')


def test_reused_binding_uses_stored_stage_and_version(tracked_module):
    t = tracked_module._ensure_tracker()
    vid = t.create_circuit_version()
    t.version = 'requested-label'
    with t.db.transaction():
        assert t.create_circuit_version() == vid
        assert _binding(t) == (vid, DesignStage.DEV, '1.0')
    assert _binding(t) == (vid, DesignStage.DEV, '1.0')


@pytest.mark.parametrize('failure', ['commit', 'release', 'rollback', 'poison'])
def test_binding_restored_after_sql_failure(tracked_module, failure):
    t = tracked_module._ensure_tracker()
    t.create_circuit_version()
    old = _binding(t)

    def deny(op, arg, *_):
        blocked = ((failure == 'commit' and op == sqlite3.SQLITE_TRANSACTION and arg == 'COMMIT')
                   or (failure == 'release' and op == sqlite3.SQLITE_SAVEPOINT and arg == 'RELEASE')
                   or (failure == 'rollback' and op == sqlite3.SQLITE_TRANSACTION and arg == 'ROLLBACK')
                   or (failure == 'poison' and op == sqlite3.SQLITE_SAVEPOINT and arg == 'ROLLBACK'))
        return sqlite3.SQLITE_DENY if blocked else sqlite3.SQLITE_OK

    try:
        with pytest.raises((sqlite3.DatabaseError, DatabaseError)):
            with t.db.transaction():
                if failure == 'poison':
                    with pytest.raises(ValueError):
                        with t.db.transaction():
                            t.promote('stable', lock=False)
                            t.db.conn.set_authorizer(deny)
                            raise ValueError('poison inner rollback')
                    assert _binding(t) == old
                elif failure == 'release':
                    with t.db.transaction():
                        t.promote('stable', lock=False)
                        t.db.conn.set_authorizer(deny)
                else:
                    t.promote('stable', lock=False)
                    t.db.conn.set_authorizer(deny)
                    if failure == 'rollback':
                        raise ValueError('abort')
        assert _binding(t) == old
    finally:
        t.db.conn.set_authorizer(_allow_sql)
        # A denied SQL rollback leaves SQL live; explicitly recover the connection.
        t.db.conn.rollback()
    assert t.db.conn.execute('SELECT COUNT(*) FROM circuit_versions').fetchone()[0] == 1
    t.store_result('after', 1., 'V')


def test_committed_binding_not_undone_by_after_commit_failure(tracked_module):
    t = tracked_module._ensure_tracker()
    t.create_circuit_version()

    def fail():
        raise ValueError('post commit')

    with pytest.raises(ValueError, match='post commit'):
        with t.db.transaction():
            new = t.promote('stable', lock=False)
            t.db._after_commit(fail)
    assert _binding(t) == (new, DesignStage.STABLE, '1.0.stable')
    with pytest.raises(ValueError):
        with t.db.transaction():
            raise ValueError('later')
    assert t.circuit_id == new
    t.store_result('committed', 1., 'V')
