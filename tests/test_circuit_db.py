"""
Tests for circuit_db.Database and DesignStage.

All tests use in-memory SQLite via the in_memory_db and db_with_circuit fixtures.
"""

import json

import pytest

from silk.db.circuit_db import Database, DesignStage
from silk.exceptions import DatabaseError
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
    structure_with_net,
)


# ---------------------------------------------------------------------------
# DesignStage enum
# ---------------------------------------------------------------------------

class TestDesignStage:

    def test_dev_value(self):
        """DesignStage.DEV has value 'dev'."""
        assert DesignStage.DEV.value == 'dev'

    def test_stable_value(self):
        """DesignStage.STABLE has value 'stable'."""
        assert DesignStage.STABLE.value == 'stable'

    def test_post_layout_value(self):
        """DesignStage.POST_LAYOUT has value 'post_layout'."""
        assert DesignStage.POST_LAYOUT.value == 'post_layout'

    def test_tapeout_value(self):
        """DesignStage.TAPEOUT has value 'tapeout'."""
        assert DesignStage.TAPEOUT.value == 'tapeout'

    def test_all_stages_present(self):
        """All four design stages exist in the enum."""
        members = [s.name for s in DesignStage]
        assert set(members) == {'DEV', 'STABLE', 'POST_LAYOUT', 'TAPEOUT'}


# ---------------------------------------------------------------------------
# create_circuit
# ---------------------------------------------------------------------------

class TestCreateCircuit:

    def test_returns_string_id(self, in_memory_db):
        """create_circuit returns a string circuit_id."""
        cid = create_building_snapshot(in_memory_db, name='amp', version='1.0')
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_circuit_appears_in_db(self, in_memory_db):
        """Created circuit is retrievable from the database."""
        cid = create_building_snapshot(in_memory_db, name='amp', version='1.0')
        info = in_memory_db.get_circuit_by_id(cid)
        assert info is not None
        assert info['circuit_name'] == 'amp'
        assert info['version'] == '1.0'
        assert info['stage'] == 'dev'

    def test_circuit_with_description(self, in_memory_db):
        """Circuit description is stored correctly."""
        cid = create_building_snapshot(
            in_memory_db,
            name='ldo', version='2.0', stage=DesignStage.STABLE,
            description='Low dropout regulator'
        )
        info = in_memory_db.get_circuit_by_id(cid)
        assert info['description'] == 'Low dropout regulator'


# ---------------------------------------------------------------------------
# get_circuit_by_id (aliased as get_circuit_info in the task)
# ---------------------------------------------------------------------------

class TestGetCircuitInfo:

    def test_returns_dict_with_expected_keys(self, db_with_circuit):
        """get_circuit_by_id returns dict with standard keys."""
        db, circuit_id = db_with_circuit
        info = db.get_circuit_by_id(circuit_id)
        expected_keys = {'version_id', 'circuit_name', 'version', 'stage',
                         'created_at', 'netlist_path', 'description', 'metadata'}
        assert expected_keys.issubset(set(info.keys()))

    def test_returns_none_for_invalid_id(self, in_memory_db):
        """get_circuit_by_id returns None for nonexistent ID."""
        assert in_memory_db.get_circuit_by_id('nonexistent-uuid') is None

    def test_alias_matches(self, db_with_circuit):
        """get_circuit_by_version_id is an alias that returns same result."""
        db, circuit_id = db_with_circuit
        assert db.get_circuit_by_version_id(circuit_id) == db.get_circuit_by_id(circuit_id)


# ---------------------------------------------------------------------------
# add_simulation_result
# ---------------------------------------------------------------------------

class TestAddSimulationResult:

    def test_returns_sim_id(self, db_with_circuit):
        """add_simulation_result returns a sim_id string."""
        db, cid = db_with_circuit
        sid = db.add_simulation_result(circuit_version_id=cid, parameter='gain', value=42.5, unit='dB')
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_result_appears_in_db(self, db_with_circuit):
        """Stored simulation result is retrievable."""
        db, cid = db_with_circuit
        db.add_simulation_result(circuit_version_id=cid, parameter='gain', value=42.5, unit='dB')
        results = db.get_simulation_results(cid)
        assert len(results) >= 1
        assert any(r.parameter == 'gain' for r in results)

    def test_result_with_corner_temperature_voltage(self, db_with_circuit):
        """Corner, temperature, and voltage are stored correctly."""
        db, cid = db_with_circuit
        db.add_simulation_result(
            circuit_version_id=cid, parameter='power', value=2.3, unit='mW',
            corner='ff', temperature=85.0, voltage=1.98
        )
        results = db.get_simulation_results(cid, parameter='power')
        assert len(results) == 1
        r = results[0]
        assert r.corner == 'ff'
        assert r.temperature == 85.0
        assert r.voltage == 1.98

    def test_multiple_results_for_same_circuit(self, db_with_circuit):
        """Multiple simulation results for same circuit are all retrievable."""
        db, cid = db_with_circuit
        db.add_simulation_result(circuit_version_id=cid, parameter='gain', value=42.5, unit='dB')
        db.add_simulation_result(circuit_version_id=cid, parameter='bandwidth', value=1e6, unit='Hz')
        db.add_simulation_result(circuit_version_id=cid, parameter='power', value=2.3, unit='mW')
        results = db.get_simulation_results(cid)
        assert len(results) == 3

    def test_multiple_corners_stored_independently(self, db_with_circuit):
        """Same parameter at different corners stored as separate records."""
        db, cid = db_with_circuit
        for corner in ['tt', 'ff', 'ss']:
            db.add_simulation_result(
                circuit_version_id=cid, parameter='gain', value=42.5,
                unit='dB', corner=corner
            )
        results = db.get_simulation_results(cid, parameter='gain')
        assert len(results) == 3
        corners = {r.corner for r in results}
        assert corners == {'tt', 'ff', 'ss'}

    def test_corner_filter_returns_only_matching_corner(self, db_with_circuit):
        """Filtering by corner returns only results for that corner."""
        db, cid = db_with_circuit
        for corner in ['tt', 'ff', 'ss']:
            db.add_simulation_result(
                circuit_version_id=cid, parameter='gain', value=42.5,
                unit='dB', corner=corner
            )
        results = db.get_simulation_results(cid, corner='tt')
        assert len(results) == 1
        assert results[0].corner == 'tt'


# ---------------------------------------------------------------------------
# insert_or_get_module
# ---------------------------------------------------------------------------

class TestInsertOrGetModule:

    def test_first_call_inserts(self, in_memory_db):
        """First call with a new hash creates a module and returns its ID."""
        mid = in_memory_db.insert_or_get_module(
            module_name='inv', module_path='top.inv',
            module_hash='abc123', device_count=2
        )
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_second_call_returns_same_id(self, in_memory_db):
        """Second call with same hash returns the existing module_id."""
        mid1 = in_memory_db.insert_or_get_module(
            module_name='inv', module_path='top.inv',
            module_hash='abc123', device_count=2
        )
        mid2 = in_memory_db.insert_or_get_module(
            module_name='inv', module_path='top.inv',
            module_hash='abc123', device_count=2
        )
        assert mid1 == mid2

    def test_different_hash_creates_new(self, in_memory_db):
        """Different hash creates a new module with different ID."""
        mid1 = in_memory_db.insert_or_get_module(
            module_name='inv', module_path='top.inv',
            module_hash='hash_a', device_count=2
        )
        mid2 = in_memory_db.insert_or_get_module(
            module_name='inv_v2', module_path='top.inv',
            module_hash='hash_b', device_count=3
        )
        assert mid1 != mid2


# ---------------------------------------------------------------------------
# link_module_to_circuit
# ---------------------------------------------------------------------------

class TestLinkModuleToCircuit:

    def test_link_created(self, db_with_building_circuit):
        """link_module_to_circuit creates a junction record in DB."""
        db, cid = db_with_building_circuit
        mid = db.insert_or_get_module(
            module_name='buffer', module_path='top.buf',
            module_hash='buf_hash', device_count=4
        )
        db.link_module_to_circuit(cid, mid, instance_name='buf_0')
        modules = db.get_circuit_modules(cid)
        assert len(modules) == 1
        assert modules[0]['module_name'] == 'buffer'
        assert modules[0]['instance_name'] == 'buf_0'

    def test_multiple_links(self, db_with_building_circuit):
        """Multiple modules can be linked to the same circuit."""
        db, cid = db_with_building_circuit
        mid1 = db.insert_or_get_module('mod_a', 'top.a', 'ha', 2)
        mid2 = db.insert_or_get_module('mod_b', 'top.b', 'hb', 3)
        db.link_module_to_circuit(cid, mid1, instance_name='a_0')
        db.link_module_to_circuit(cid, mid2, instance_name='b_0')
        modules = db.get_circuit_modules(cid)
        assert len(modules) == 2


# ---------------------------------------------------------------------------
# db_with_circuit fixture
# ---------------------------------------------------------------------------

class TestDBWithCircuit:

    def test_circuit_id_is_valid(self, db_with_circuit):
        """db_with_circuit fixture provides a valid circuit_id."""
        db, circuit_id = db_with_circuit
        assert isinstance(circuit_id, str)
        assert len(circuit_id) > 0

    def test_circuit_id_reusable(self, db_with_circuit):
        """circuit_id from fixture can be used to query the database."""
        db, circuit_id = db_with_circuit
        info = db.get_circuit_by_id(circuit_id)
        assert info is not None
        assert info['circuit_name'] == 'test_inverter'


# ---------------------------------------------------------------------------
# get_circuits_using_module
# ---------------------------------------------------------------------------

class TestGetCircuitsUsingModule:

    def test_returns_circuits_using_module(self, db_with_building_circuit):
        """Returns circuits that use a module with the given hash."""
        db, cid = db_with_building_circuit
        mid = db.insert_or_get_module('buf', 'top.buf', 'hash_xyz', 2)
        db.link_module_to_circuit(cid, mid, instance_name='buf_0')
        circuits = db.get_circuits_using_module('hash_xyz')
        assert len(circuits) == 1
        assert circuits[0]['version_id'] == cid
        assert circuits[0]['circuit_name'] == 'test_inverter'

    def test_returns_empty_for_unused_hash(self, in_memory_db):
        """Returns empty list when no circuit uses the module hash."""
        assert db.get_circuits_using_module('nonexistent') == [] if (db := in_memory_db) else True

    def test_multiple_circuits_using_same_module(self, in_memory_db):
        """Multiple circuits using the same module are all returned."""
        db = in_memory_db
        cid1 = create_building_snapshot(db, name='amp1', version='1.0')
        cid2 = create_building_snapshot(db, name='amp2', version='1.0')
        mid = db.insert_or_get_module('shared', 'top.shared', 'shared_hash', 3)
        db.link_module_to_circuit(cid1, mid, instance_name='s0')
        db.link_module_to_circuit(cid2, mid, instance_name='s0')
        circuits = db.get_circuits_using_module('shared_hash')
        assert len(circuits) == 2
        names = {c['circuit_name'] for c in circuits}
        assert names == {'amp1', 'amp2'}


# ---------------------------------------------------------------------------
# compare_circuit_modules
# ---------------------------------------------------------------------------

class TestCompareCircuitModules:

    def test_identical_modules(self, in_memory_db):
        """Two circuits with the same module show it as unchanged."""
        db = in_memory_db
        cid1 = create_building_snapshot(db, name='c1', version='1.0')
        cid2 = create_building_snapshot(db, name='c2', version='1.0')
        mid = db.insert_or_get_module('mod', 'top.mod', 'h1', 2)
        db.link_module_to_circuit(cid1, mid, instance_name='m0')
        db.link_module_to_circuit(cid2, mid, instance_name='m0')
        result = db.compare_circuit_modules(cid1, cid2)
        assert len(result['unchanged']) == 1
        assert len(result['added']) == 0
        assert len(result['removed']) == 0

    def test_added_module(self, in_memory_db):
        """Module in circuit 2 but not circuit 1 shows as added."""
        db = in_memory_db
        cid1 = create_building_snapshot(db, name='c1', version='1.0')
        cid2 = create_building_snapshot(db, name='c2', version='1.0')
        mid = db.insert_or_get_module('mod_new', 'top.new', 'h_new', 1)
        db.link_module_to_circuit(cid2, mid, instance_name='m0')
        result = db.compare_circuit_modules(cid1, cid2)
        assert len(result['added']) == 1
        assert len(result['removed']) == 0

    def test_removed_module(self, in_memory_db):
        """Module in circuit 1 but not circuit 2 shows as removed."""
        db = in_memory_db
        cid1 = create_building_snapshot(db, name='c1', version='1.0')
        cid2 = create_building_snapshot(db, name='c2', version='1.0')
        mid = db.insert_or_get_module('mod_old', 'top.old', 'h_old', 1)
        db.link_module_to_circuit(cid1, mid, instance_name='m0')
        result = db.compare_circuit_modules(cid1, cid2)
        assert len(result['removed']) == 1
        assert len(result['added']) == 0


# ---------------------------------------------------------------------------
# add_device / add_net
# ---------------------------------------------------------------------------

class TestAddDevice:

    def test_add_device_returns_id(self, db_with_building_circuit):
        """add_device returns a device_id string."""
        db, cid = db_with_building_circuit
        did = db.add_device(cid, 'M1', 'nmos', model='sky130_nfet', width=10e-6, length=180e-9)
        assert isinstance(did, str) and len(did) > 0

    def test_add_device_without_properties(self, db_with_building_circuit):
        """add_device works without extra properties."""
        db, cid = db_with_building_circuit
        did = db.add_device(cid, 'R1', 'resistor')
        assert isinstance(did, str)

    def test_add_transistor_returns_id(self, db_with_building_circuit):
        """add_transistor returns a device_id and stores transistor data."""
        db, cid = db_with_building_circuit
        did = db.add_transistor(cid, 'M2', 'nmos', width=5e-6, length=180e-9, multiplier=2)
        assert isinstance(did, str) and len(did) > 0


class TestAddNet:

    def test_add_net_returns_id(self, db_with_building_circuit):
        """add_net returns a net_id string."""
        db, cid = db_with_building_circuit
        nid = db.add_net(cid, 'vdd', net_type='power')
        assert isinstance(nid, str) and len(nid) > 0

    def test_add_net_without_properties(self, db_with_building_circuit):
        """add_net works without type or properties."""
        db, cid = db_with_building_circuit
        nid = db.add_net(cid, 'sig_out')
        assert isinstance(nid, str)


# ---------------------------------------------------------------------------
# get_latest (various branches)
# ---------------------------------------------------------------------------

class TestGetLatest:

    def test_get_latest_with_stage(self, in_memory_db):
        """get_latest with stage filter returns correct circuit."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        result = db.get_latest('amp', DesignStage.DEV)
        assert result is not None
        assert result['circuit_name'] == 'amp'

    def test_get_latest_without_stage(self, in_memory_db):
        """get_latest without stage returns the most recent version."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        result = db.get_latest('amp')
        assert result is not None
        assert result['circuit_name'] == 'amp'

    def test_get_latest_returns_none_for_unknown(self, in_memory_db):
        """get_latest returns None for nonexistent circuit."""
        assert in_memory_db.get_latest('nonexistent') is None


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------

class TestGetVersion:

    def test_get_version_found(self, in_memory_db):
        """get_version returns circuit info for existing name+version."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        result = db.get_version('amp', '1.0')
        assert result is not None
        assert result['version'] == '1.0'

    def test_get_version_not_found(self, in_memory_db):
        """get_version returns None for nonexistent version."""
        assert in_memory_db.get_version('amp', '99.0') is None


# ---------------------------------------------------------------------------
# promote_circuit / lock_circuit
# ---------------------------------------------------------------------------

class TestPromoteCircuit:

    def test_promote_success(self, in_memory_db):
        """promote_circuit creates a new version in the target stage."""
        db = in_memory_db
        source_id = create_sealed_snapshot(db, name='amp', version='1.0')
        new_id = db.promote_circuit(source_id, DesignStage.STABLE)
        info = db.get_circuit_by_id(new_id)
        assert info['stage'] == 'stable'
        assert db._snapshot_row(new_id)['parent_version_id'] == source_id

    def test_promote_no_source_raises(self, in_memory_db):
        """promote_circuit rejects a missing source UUID."""
        with pytest.raises(DatabaseError, match="does not exist"):
            in_memory_db.promote_circuit('ghost', DesignStage.STABLE)


class TestLockCircuit:

    def test_lock_success(self, in_memory_db):
        """lock_circuit locks an existing circuit version."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        db.lock_circuit('amp', '1.0')
        # Verify via raw SQL
        cursor = db.conn.cursor()
        cursor.execute("SELECT is_locked FROM circuit_versions WHERE circuit_name='amp'")
        assert cursor.fetchone()[0] == 1

    def test_lock_not_found_raises(self, in_memory_db):
        """lock_circuit raises ValueError for nonexistent circuit."""
        with pytest.raises(ValueError, match="not found"):
            in_memory_db.lock_circuit('ghost', '1.0')


# ---------------------------------------------------------------------------
# get_all_circuit_versions
# ---------------------------------------------------------------------------

class TestGetAllCircuitVersions:

    def test_returns_all_versions(self, in_memory_db):
        """Returns all versions of a circuit."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        create_sealed_snapshot(
            db, name='amp', version='2.0', stage=DesignStage.STABLE
        )
        versions = db.get_all_circuit_versions('amp')
        assert len(versions) == 2

    def test_returns_empty_for_unknown(self, in_memory_db):
        """Returns empty list for nonexistent circuit."""
        assert in_memory_db.get_all_circuit_versions('ghost') == []

    def test_invalid_order_by_defaults(self, in_memory_db):
        """Invalid order_by falls back to created_at."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        versions = db.get_all_circuit_versions('amp', order_by='invalid')
        assert len(versions) == 1


# ---------------------------------------------------------------------------
# get_most_simulated_circuit
# ---------------------------------------------------------------------------

class TestGetMostSimulatedCircuit:

    def test_returns_circuit_with_most_runs(self, in_memory_db):
        """Returns the circuit version with the highest run_count in metadata."""
        db = in_memory_db
        first_id = create_sealed_snapshot(
            db, name='amp', version='1.0', metadata={'run_count': 5},
            structure=structure_with_net('out_v1'),
        )
        second_id = create_sealed_snapshot(
            db, name='amp', version='2.0', metadata={'run_count': 15},
            structure=structure_with_net('out_v2'),
        )
        db.conn.execute(
            "UPDATE circuit_versions SET metadata = ? WHERE version_id = ?",
            (json.dumps({'run_count': 5}), first_id),
        )
        db.conn.execute(
            "UPDATE circuit_versions SET metadata = ? WHERE version_id = ?",
            (json.dumps({'run_count': 15}), second_id),
        )
        db.conn.commit()
        result = db.get_most_simulated_circuit('amp')
        assert result is not None
        assert result['metadata']['run_count'] == 15

    def test_returns_none_for_unknown(self, in_memory_db):
        """Returns None for nonexistent circuit name."""
        assert in_memory_db.get_most_simulated_circuit('ghost') is None

    def test_no_metadata_defaults_zero(self, in_memory_db):
        """Circuits without run_count metadata default to 0."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        result = db.get_most_simulated_circuit('amp')
        assert result is not None


# ---------------------------------------------------------------------------
# create_release
# ---------------------------------------------------------------------------

class TestCreateRelease:

    def test_create_empty_release(self, in_memory_db):
        """Creates a release with no circuits."""
        db = in_memory_db
        rid = db.create_release('v1.0-release', DesignStage.STABLE, notes='First release')
        assert isinstance(rid, str) and len(rid) > 0

    def test_create_release_with_circuits(self, in_memory_db):
        """Creates a release with circuit contents."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0', stage=DesignStage.STABLE)
        create_sealed_snapshot(db, name='ldo', version='1.0', stage=DesignStage.STABLE)
        rid = db.create_release('v1.0-release', DesignStage.STABLE, circuit_names=['amp', 'ldo'])
        assert isinstance(rid, str) and len(rid) > 0

    def test_release_with_missing_circuit_skips(self, in_memory_db):
        """Release creation skips circuits that don't exist at that stage."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0', stage=DesignStage.STABLE)
        rid = db.create_release('v1.0-release', DesignStage.STABLE, circuit_names=['amp', 'ghost'])
        assert isinstance(rid, str)


# ---------------------------------------------------------------------------
# list_circuits
# ---------------------------------------------------------------------------

class TestListCircuits:

    def test_list_all(self, in_memory_db):
        """list_circuits without filter returns all circuits."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        create_sealed_snapshot(db, name='ldo', version='1.0', stage=DesignStage.STABLE)
        result = db.list_circuits()
        assert len(result) == 2

    def test_list_by_stage(self, in_memory_db):
        """list_circuits with stage filter returns only matching circuits."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        create_sealed_snapshot(db, name='ldo', version='1.0', stage=DesignStage.STABLE)
        result = db.list_circuits(stage=DesignStage.DEV)
        assert len(result) == 1
        assert result[0]['circuit_name'] == 'amp'


# ---------------------------------------------------------------------------
# get_circuit_summary
# ---------------------------------------------------------------------------

class TestGetCircuitSummary:

    def test_summary_with_all_data(self, in_memory_db):
        """Summary includes devices, nets, modules, and simulation results."""
        db = in_memory_db
        cid = create_building_snapshot(db, name='amp', version='1.0')
        db.add_device(cid, 'M1', 'nmos', model='nfet')
        db.add_net(cid, 'vdd', net_type='power')
        mid = db.insert_or_get_module('buf', 'top.buf', 'hbuf', 1)
        db.link_module_to_circuit(cid, mid, instance_name='b0')
        db.store_spec_declarations(cid, ())
        db.seal_circuit_version(cid)
        db.add_simulation_result(cid, parameter='gain', value=40.0, unit='dB')
        summary = db.get_circuit_summary('amp', '1.0')
        assert summary is not None
        assert len(summary['devices']) == 1
        assert len(summary['nets']) == 1
        assert len(summary['modules']) == 1
        assert len(summary['simulation_results']) == 1

    def test_summary_latest_version(self, in_memory_db):
        """Summary without version uses latest."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        summary = db.get_circuit_summary('amp')
        assert summary is not None
        assert summary['circuit']['version'] == '1.0'

    def test_summary_returns_none_for_unknown(self, in_memory_db):
        """get_circuit_summary returns None for nonexistent circuit."""
        assert in_memory_db.get_circuit_summary('ghost') is None

    def test_summary_net_with_no_properties_reads_as_not_inferred(self, in_memory_db):
        """TASK-112. A net added with no 'inferred' property at all -- a
        manually declared net, or a row stored before TASK-112 gave
        add_net's properties an 'inferred' key any way to reach the DB --
        must not be mistaken for an inferred one. Pre-fix, get_circuit_summary
        did not even select the properties column, so this distinction did
        not exist at all; the net's role provenance was simply dropped."""
        db = in_memory_db
        cid = create_building_snapshot(db, name='amp', version='1.0')
        db.add_net(cid, 'vdd', net_type='power')
        db.store_spec_declarations(cid, ())
        db.seal_circuit_version(cid)
        summary = db.get_circuit_summary('amp')
        net = summary['nets'][0]
        assert net['name'] == 'vdd'
        assert net['net_type'] == 'power'
        assert net['role_inferred'] is False

    def test_summary_net_with_inferred_property_reads_back_true(self, in_memory_db):
        """TASK-112. add_net's **properties already accepted an 'inferred'
        key before this fix -- the key parse_spice_netlist's in-memory
        record uses (TASK-97) -- but get_circuit_summary's net SELECT
        never fetched the properties column, so the flag was written and
        then silently lost on readback."""
        db = in_memory_db
        cid = create_building_snapshot(db, name='amp', version='1.0')
        db.add_net(cid, 'vdd', net_type='power', inferred=True)
        db.store_spec_declarations(cid, ())
        db.seal_circuit_version(cid)
        summary = db.get_circuit_summary('amp')
        net = summary['nets'][0]
        assert net['role_inferred'] is True

    def test_summary_net_accepts_role_inferred_property_spelling(self, in_memory_db):
        """TASK-112. A re-imported export projection writes the flag back
        under the 'role_inferred' key (matching this method's own field
        name, see CircuitExporter.export_to_json) rather than
        parse_spice_netlist's 'inferred' -- the two writer conventions
        devices.properties already has for width/w, length/l, multiplier/m.
        get_circuit_summary must read this spelling too."""
        db = in_memory_db
        cid = create_building_snapshot(db, name='amp', version='1.0')
        db.add_net(cid, 'vdd', net_type='power', role_inferred=True)
        db.store_spec_declarations(cid, ())
        db.seal_circuit_version(cid)
        summary = db.get_circuit_summary('amp')
        net = summary['nets'][0]
        assert net['role_inferred'] is True


# ---------------------------------------------------------------------------
# update_simulation_summary / get_simulation_summary
# ---------------------------------------------------------------------------

class TestSimulationSummary:

    def test_empty_summary(self, db_with_circuit):
        """update_simulation_summary returns empty string when no results."""
        db, cid = db_with_circuit
        summary = db.update_simulation_summary(cid)
        assert summary == ""

    def test_single_result_summary(self, db_with_circuit):
        """Summary for single result includes parameter name and value."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=42.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        summary = db.update_simulation_summary(cid)
        assert 'gain' in summary
        assert 'SIMULATION SUMMARY' in summary

    def test_multi_corner_summary(self, db_with_circuit):
        """Summary for multi-corner results shows range."""
        db, cid = db_with_circuit
        for corner, val in [('tt', 42.0), ('ss', 38.0), ('ff', 46.0)]:
            db.add_simulation_result(cid, parameter='gain', value=val, unit='dB',
                                     corner=corner, temperature=27.0, voltage=1.8)
        summary = db.update_simulation_summary(cid)
        assert '3 corners' in summary
        assert '38' in summary
        assert '46' in summary

    def test_summary_with_pass_fail_status(self, db_with_circuit):
        """Summary includes pass/fail status results."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain_check', value_text='PASS')
        summary = db.update_simulation_summary(cid)
        assert 'STATUS' in summary
        assert 'PASS' in summary

    def test_get_simulation_summary_after_explicit_update(self, db_with_circuit):
        """A read renders the evidence after an explicit summary update."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=42.0, unit='dB', corner='tt')
        db.update_simulation_summary(cid)
        cached = db.get_simulation_summary(cid)
        assert 'gain' in cached

    def test_get_simulation_summary_generates_if_missing(self, db_with_circuit):
        """get_simulation_summary auto-generates if not cached."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='bw', value=1e6, unit='Hz', corner='tt')
        result = db.get_simulation_summary(cid)
        assert 'bw' in result

    def test_summary_no_conditions(self, db_with_circuit):
        """Summary works when temperature and voltage are None."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='offset', value=0.5, unit='mV', corner='tt')
        summary = db.update_simulation_summary(cid)
        assert 'offset' in summary

    def test_summary_single_result_no_value(self, db_with_circuit):
        """Summary handles a single result where value is None (value_text only)."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='status', value=None, value_text='OK', corner='tt')
        summary = db.update_simulation_summary(cid)
        assert 'STATUS' in summary


# ---------------------------------------------------------------------------
# add_parasitic
# ---------------------------------------------------------------------------

class TestAddParasitic:

    def test_add_parasitic_returns_id(self, db_with_building_circuit):
        """add_parasitic returns a parasitic_id string."""
        db, cid = db_with_building_circuit
        nid = db.add_net(cid, 'out', net_type='signal')
        db.store_spec_declarations(cid, ())
        db.seal_circuit_version(cid)
        pid = db.add_parasitic(cid, nid, 'capacitance', 1.5e-15, 'F',
                               from_node='out', to_node='gnd', extraction_tool='magic')
        assert isinstance(pid, str) and len(pid) > 0


# ---------------------------------------------------------------------------
# get_latest_circuit / get_latest_circuit_by_version
# ---------------------------------------------------------------------------

class TestLatestCircuitAliases:

    def test_get_latest_circuit_alias(self, in_memory_db):
        """get_latest_circuit is an alias for get_latest."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        result = db.get_latest_circuit('amp')
        assert result is not None
        assert result['circuit_name'] == 'amp'

    def test_get_latest_circuit_by_version(self, in_memory_db):
        """get_latest_circuit_by_version returns highest version."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        create_sealed_snapshot(db, name='amp', version='2.0', stage=DesignStage.STABLE)
        result = db.get_latest_circuit_by_version('amp')
        assert result is not None
        assert result['version'] == '2.0'

    def test_get_latest_circuit_by_version_none(self, in_memory_db):
        """get_latest_circuit_by_version returns None for unknown circuit."""
        assert in_memory_db.get_latest_circuit_by_version('ghost') is None


# ---------------------------------------------------------------------------
# latest_at_stage
# ---------------------------------------------------------------------------

class TestLatestAtStage:

    def test_returns_latest_of_each(self, in_memory_db):
        """Returns latest version of each circuit at the given stage."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        create_sealed_snapshot(db, name='ldo', version='1.0')
        result = db.latest_at_stage(DesignStage.DEV)
        assert len(result) == 2
        names = {c['circuit_name'] for c in result}
        assert names == {'amp', 'ldo'}

    def test_building_only_left_out_never_none(self, in_memory_db):
        """A circuit with only building snapshots is omitted, never None."""
        db = in_memory_db
        create_building_snapshot(db, name='wip', version='1.0')
        create_sealed_snapshot(db, name='done', version='1.0')
        result = db.latest_at_stage(DesignStage.DEV)
        assert None not in result
        assert [c['circuit_name'] for c in result] == ['done']

    def test_mixed_circuit_returns_newest_sealed(self, in_memory_db):
        """A circuit with a sealed and a newer building snapshot lists the sealed."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0')
        create_building_snapshot(db, name='amp', version='2.0')
        result = db.latest_at_stage(DesignStage.DEV)
        assert [(c['circuit_name'], c['version']) for c in result] == [('amp', '1.0')]

    def test_selects_newest_sealed_at_the_stage(self, in_memory_db):
        """Two sealed DEV snapshots, a newer building DEV one and a sealed
        snapshot at another stage: the newest sealed DEV one is selected."""
        db = in_memory_db
        create_sealed_snapshot(db, name='amp', version='1.0',
                               structure=structure_with_net('a'))
        newest = create_sealed_snapshot(db, name='amp', version='2.0',
                                        structure=structure_with_net('b'))
        create_building_snapshot(db, name='amp', version='3.0',
                                 structure=structure_with_net('c'))
        create_sealed_snapshot(db, name='amp', version='4.0',
                               stage=DesignStage.STABLE,
                               structure=structure_with_net('d'))
        result = db.latest_at_stage(DesignStage.DEV)
        assert [c['version_id'] for c in result] == [newest]
        assert result[0]['stage'] == 'dev'

    def test_old_name_removed(self, in_memory_db):
        assert not hasattr(in_memory_db, 'get_all_circuits_at_stage')

    def test_empty_stage(self, in_memory_db):
        """Returns empty list when no circuits at the stage."""
        assert in_memory_db.latest_at_stage(DesignStage.TAPEOUT) == []


# ---------------------------------------------------------------------------
# get_corners_for_parameter
# ---------------------------------------------------------------------------

class TestGetCornersForParameter:

    def test_returns_corner_results(self, db_with_circuit):
        """Returns results across corners for a parameter."""
        db, cid = db_with_circuit
        for corner in ['tt', 'ss', 'ff']:
            db.add_simulation_result(cid, parameter='gain', value=40.0, unit='dB', corner=corner)
        results = db.get_corners_for_parameter(cid, 'gain')
        assert len(results) == 3

    def test_empty_for_unknown_param(self, db_with_circuit):
        """Returns empty list for nonexistent parameter."""
        db, cid = db_with_circuit
        assert db.get_corners_for_parameter(cid, 'nonexistent') == []


# ---------------------------------------------------------------------------
# get_corner_summary
# ---------------------------------------------------------------------------

class TestGetCornerSummary:

    def test_corner_summary_with_conditions(self, db_with_circuit):
        """Corner summary groups results by corner with conditions."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=42.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        db.add_simulation_result(cid, parameter='gain', value=38.0, unit='dB',
                                 corner='ss', temperature=85.0, voltage=1.62)
        summary = db.get_corner_summary(cid)
        assert 'tt' in summary
        assert 'ss' in summary

    def test_corner_summary_without_conditions(self, db_with_circuit):
        """Corner summary handles results with no temperature/voltage."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=42.0, unit='dB', corner='tt')
        summary = db.get_corner_summary(cid)
        assert 'tt' in summary

    def test_corner_summary_empty(self, db_with_circuit):
        """Corner summary is empty dict when no results exist."""
        db, cid = db_with_circuit
        assert db.get_corner_summary(cid) == {}

    def test_corner_summary_null_corner(self, db_with_circuit):
        """Corner summary handles null corner as 'nominal'."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=42.0, unit='dB')
        summary = db.get_corner_summary(cid)
        assert 'nominal' in summary


# ---------------------------------------------------------------------------
# update_simulation_summary
# ---------------------------------------------------------------------------

class TestUpdateSimulationSummary:

    def test_empty_results_returns_empty_string(self, db_with_circuit):
        """update_simulation_summary returns empty string when no results exist."""
        db, cid = db_with_circuit
        result = db.update_simulation_summary(cid)
        assert result == ""

    def test_single_result_produces_summary(self, db_with_circuit):
        """update_simulation_summary with one result produces a formatted summary."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=42.5, unit='dB', corner='tt')
        summary = db.update_simulation_summary(cid)
        assert 'gain' in summary
        assert '42.5' in summary

    def test_multi_corner_shows_range(self, db_with_circuit):
        """update_simulation_summary with multiple corners shows min-to-max range."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=40.0, unit='dB', corner='ss')
        db.add_simulation_result(cid, parameter='gain', value=45.0, unit='dB', corner='ff')
        db.add_simulation_result(cid, parameter='gain', value=42.5, unit='dB', corner='tt')
        summary = db.update_simulation_summary(cid)
        assert 'gain' in summary
        assert 'to' in summary  # shows range "40 to 45"

    def test_summary_includes_status_text(self, db_with_circuit):
        """update_simulation_summary includes value_text (pass/fail) in summary."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain_check', value=None,
                                 value_text='PASS', unit=None)
        summary = db.update_simulation_summary(cid)
        assert 'STATUS' in summary
        assert 'PASS' in summary

    def test_get_simulation_summary_renders_without_cache(self, db_with_circuit):
        """get_simulation_summary generates summary when none cached."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='bw', value=1e6, unit='Hz')
        # Read current records even when no stored summary exists.
        result = db.get_simulation_summary(cid)
        assert 'bw' in result

    def test_multi_corner_with_conditions_shows_corner_detail(self, db_with_circuit):
        """Multi-corner summary shows each corner's individual value."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='power', value=2.3, unit='mW',
                                 corner='tt', temperature=27.0, voltage=1.8)
        db.add_simulation_result(cid, parameter='power', value=3.1, unit='mW',
                                 corner='ff', temperature=85.0, voltage=1.98)
        summary = db.update_simulation_summary(cid)
        assert 'power' in summary
        assert 'tt' in summary or 'ff' in summary


# ---------------------------------------------------------------------------
# 2026-08-24 fidelity fixes: stale summary cache, duplicate-observation
# mislabeling, get_corners_for_parameter latest_only, tracker default
# sim_type/tool_name.
# ---------------------------------------------------------------------------

class TestSummaryCacheInvalidation:
    """add_simulation_result must invalidate the cached summary column —
    before the fix, a store after the first read left view_summary()
    permanently stale (verified: stored 99 dB after reading a 30 dB summary,
    still printed 30 dB)."""

    def test_cache_invalidated_on_new_store(self, db_with_circuit):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB', corner='tt')
        first = db.get_simulation_summary(cid)
        assert '30 dB' in first

        db.add_simulation_result(cid, parameter='gain', value=99.0, unit='dB', corner='tt')
        second = db.get_simulation_summary(cid)
        assert '99 dB' in second
        assert '30 dB' not in second

    def test_cache_columns_cleared_immediately_after_store(self, db_with_circuit):
        """The cache columns are NULLed by the store itself, not lazily by
        the next read — verify directly against the row."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB')
        db.update_simulation_summary(cid)
        row = db.conn.execute(
            "SELECT simulation_summary FROM circuit_versions WHERE version_id = ?",
            (cid,)).fetchone()
        assert row[0] is not None  # cached

        db.add_simulation_result(cid, parameter='gain', value=99.0, unit='dB')
        row = db.conn.execute(
            "SELECT simulation_summary, last_summary_update FROM circuit_versions "
            "WHERE version_id = ?", (cid,)).fetchone()
        assert row[0] is None
        assert row[1] is None


class TestSummaryDuplicateObservations:
    """Two re-runs of the same (corner, temperature, voltage) are one
    corner with two observations, not '2 corners'."""

    def test_duplicate_observation_counts_as_one_corner(self, db_with_circuit):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        db.add_simulation_result(cid, parameter='gain', value=99.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        summary = db.update_simulation_summary(cid)
        assert '1 corner' in summary
        assert '2 corners' not in summary
        assert '(latest of 2)' in summary

    def test_duplicate_observation_range_uses_latest_only(self, db_with_circuit):
        """The header value must reflect the latest observation (99), never
        the superseded one (30)."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        db.add_simulation_result(cid, parameter='gain', value=99.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        summary = db.update_simulation_summary(cid)
        assert '99 dB' in summary
        assert '30 dB' not in summary

    def test_distinct_corners_still_counted_correctly_with_a_duplicate(self, db_with_circuit):
        """One corner re-run twice plus one genuinely different corner is
        2 distinct corners, not 3 rows."""
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB', corner='tt')
        db.add_simulation_result(cid, parameter='gain', value=32.0, unit='dB', corner='tt')
        db.add_simulation_result(cid, parameter='gain', value=45.0, unit='dB', corner='ff')
        summary = db.update_simulation_summary(cid)
        assert '2 corners' in summary
        assert '3 corners' not in summary
        assert '(latest of 2)' in summary


class TestGetCornersForParameterLatestOnly:
    """get_corners_for_parameter is append-only by default; latest_only
    collapses duplicate (parameter, unit, corner, temperature, voltage) observations."""

    def test_default_returns_every_observation(self, db_with_circuit):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        db.add_simulation_result(cid, parameter='gain', value=99.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        results = db.get_corners_for_parameter(cid, 'gain')
        assert len(results) == 2

    def test_latest_only_collapses_to_one_row(self, db_with_circuit):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        db.add_simulation_result(cid, parameter='gain', value=99.0, unit='dB',
                                 corner='tt', temperature=27.0, voltage=1.8)
        results = db.get_corners_for_parameter(cid, 'gain', latest_only=True)
        assert len(results) == 1
        assert results[0]['value'] == 99.0

    def test_latest_only_keeps_distinct_corners_separate(self, db_with_circuit):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, parameter='gain', value=30.0, unit='dB', corner='tt')
        db.add_simulation_result(cid, parameter='gain', value=45.0, unit='dB', corner='ff')
        results = db.get_corners_for_parameter(cid, 'gain', latest_only=True)
        assert len(results) == 2


class _FakeModule:
    """Minimal empty module used to exercise tracker result defaults."""
    module_name = 'fake_tracked_circuit'
    elem_dict = {}
    net_dict = {}
    passives_dict = {}
    module_dict = {}


class TestTrackerResultDefaults:
    """CircuitTracker.store_results/store_result/store_report must not
    silently mislabel non-ngspice/non-tran results — defaults are None,
    matching Database.add_simulation_result (2026-08-24 fix)."""

    def _tracker(self, tmp_path):
        from silk.db.tracking import CircuitTracker
        t = CircuitTracker(_FakeModule(), str(tmp_path / "trk.db"))
        t.circuit_id = create_sealed_snapshot(
            t.db,
            name=t.circuit_name,
            version=t.version,
            stage=t.stage,
            created_by=t.created_by,
        )
        return t

    def _row(self, t, parameter):
        return t.db.conn.execute(
            "SELECT sim_type, tool_name FROM simulation_results WHERE parameter = ?",
            (parameter,)).fetchone()

    def test_store_results_default_sim_type_and_tool_name_are_null(self, tmp_path):
        t = self._tracker(tmp_path)
        t.store_results({'gain': (40.0, 'dB')})
        row = self._row(t, 'gain')
        assert row[0] is None
        assert row[1] is None

    def test_store_results_explicit_sim_type_lands(self, tmp_path):
        t = self._tracker(tmp_path)
        t.store_results({'noise': (1.2e-9, 'V/rtHz')}, sim_type='ac', tool_name='xyce')
        row = self._row(t, 'noise')
        assert row[0] == 'ac'
        assert row[1] == 'xyce'

    def test_store_result_default_sim_type_and_tool_name_are_null(self, tmp_path):
        t = self._tracker(tmp_path)
        t.store_result('bw', 1e6, 'Hz')
        row = self._row(t, 'bw')
        assert row[0] is None
        assert row[1] is None

    def test_store_result_explicit_sim_type_lands(self, tmp_path):
        t = self._tracker(tmp_path)
        t.store_result('bw', 1e6, 'Hz', sim_type='ac', tool_name='xyce')
        row = self._row(t, 'bw')
        assert row[0] == 'ac'
        assert row[1] == 'xyce'


class TestSummaryObservationIdentity:
    """TASK-512: summaries and latest reads must preserve stored evidence."""

    @staticmethod
    def _parameter_lines(summary, parameter):
        return [line.strip() for line in summary.splitlines()
                if line.startswith(f'{parameter:20s}:')]

    def test_tracker_summary_preserves_parameter_units(self, tmp_path):
        from silk import Module
        from silk.db import CircuitTracker

        mod = Module(None, str(tmp_path / 'units'), 'units.cir',
                     pdk_details=None, module_name='units')
        tracker = CircuitTracker(mod, str(tmp_path / 'units.db'))
        try:
            tracker.create_circuit_version()
            tracker.store_result('voltage', 1, 'V', corner='tt')
            tracker.store_result('voltage', 1000, 'mV', corner='tt')
            assert self._parameter_lines(tracker.view_summary(), 'voltage') == [
                'voltage             : 1 V (tt, 1 corner)',
                'voltage             : 1e+03 mV (tt, 1 corner)',
            ]
        finally:
            tracker.db.close()

    def test_tracker_summary_renders_unavailable_corner(self, tmp_path):
        from silk import Module
        from silk.db import CircuitTracker

        mod = Module(None, str(tmp_path / 'missing'), 'missing.cir',
                     pdk_details=None, module_name='missing')
        tracker = CircuitTracker(mod, str(tmp_path / 'missing.db'))
        try:
            tracker.create_circuit_version()
            tracker.store_result('gain', 1.0, 'dB', corner='tt')
            tracker.store_result('gain', float('nan'), 'dB', corner='ss')
            summary = tracker.view_summary()
            assert '1 to 1 dB (2 corners)' in summary
            assert any('ss' in line and 'unavailable dB' in line
                       for line in summary.splitlines())
        finally:
            tracker.db.close()

    def test_summary_units_keep_independent_latest_values_and_ranges(self, db_with_circuit):
        db, cid = db_with_circuit
        for unit, values in [('V', (1, 2, 3)), ('mV', (1000, 2000, 3000)),
                             (None, (4, 5, 6)), ('', (7, 8, 9))]:
            for corner, value in zip(('tt', 'tt', 'ss'), values):
                db.add_simulation_result(cid, 'voltage', value=value, unit=unit,
                                         corner=corner, timestamp='2026-10-08 00:00:00')
        summary = db.get_simulation_summary(cid)
        assert set(self._parameter_lines(summary, 'voltage')) == {
            'voltage             : 2 to 3 V (2 corners)',
            'voltage             : 2e+03 to 3e+03 mV (2 corners)',
            'voltage             : 5 to 6 (2 corners)',
            'voltage             : 8 to 9 (2 corners)',
        }
        assert summary.count('(latest of 2)') == 4
        assert 'latest of 8' not in summary

    @pytest.mark.parametrize('values, expected', [
        ([('tt', float('nan'))], 'unavailable dB (tt, 1 corner)'),
        ([('tt', 1), ('ss', float('nan'))], '1 to 1 dB (2 corners)'),
        ([('tt', float('nan')), ('ss', float('nan'))], 'unavailable dB (2 corners)'),
        ([('tt', 60), ('tt', float('nan')), ('ss', 2)], '2 to 2 dB (2 corners)'),
    ])
    def test_summary_unavailable_values(self, db_with_circuit, values, expected):
        db, cid = db_with_circuit
        for corner, value in values:
            db.add_simulation_result(cid, 'gain', value=value, unit='dB', corner=corner)
        summary = db.get_simulation_summary(cid)
        assert expected in summary
        assert '60 dB' not in summary
        unavailable_corners = {corner for corner, value in values if value != value}
        for corner in unavailable_corners:
            assert any(corner in line and 'unavailable dB' in line
                       for line in summary.splitlines())

    def test_summary_text_status_preserves_values_and_units(self, db_with_circuit):
        db, cid = db_with_circuit
        for unit, text in [('V', 'PASS'), ('mV', 'FAIL'), ('V', 'RECHECK')]:
            db.add_simulation_result(cid, 'check', value_text=text, unit=unit, corner='tt')
        summary = db.get_simulation_summary(cid)
        numeric, statuses = summary.split('=== STATUS ===')
        assert 'check' not in numeric
        assert self._parameter_lines(statuses, 'check') == [
            'check               : PASS V',
            'check               : RECHECK V',
            'check               : FAIL mV',
        ]
        assert 'unavailable' not in summary

    @pytest.mark.parametrize('entrypoint', ['update', 'read', 'cached', 'locked'])
    def test_summary_entrypoints_preserve_units_and_unavailable_values(self, db_with_circuit, entrypoint):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, 'voltage', value=1, unit='V', corner='tt')
        db.add_simulation_result(cid, 'voltage', value=1000, unit='mV', corner='tt')
        db.add_simulation_result(cid, 'gain', value=float('nan'), unit='dB', corner='ss')
        before = [tuple(row) for row in db.conn.execute('SELECT * FROM simulation_results')]
        schema = list(db.conn.execute('SELECT * FROM sqlite_master'))
        if entrypoint == 'cached':
            db.update_simulation_summary(cid)
        if entrypoint == 'locked':
            db.lock_version(cid)
        read = db.update_simulation_summary if entrypoint == 'update' else db.get_simulation_summary
        summary = read(cid)
        assert '1 V (tt, 1 corner)' in summary
        assert '1e+03 mV (tt, 1 corner)' in summary
        assert 'unavailable dB (ss, 1 corner)' in summary
        assert before == [tuple(row) for row in db.conn.execute('SELECT * FROM simulation_results')]
        assert schema == list(db.conn.execute('SELECT * FROM sqlite_master'))

    def test_summary_condition_states_remain_distinct_with_unavailable_values(self, db_with_circuit):
        from silk.simulation.conditions import ConditionState

        db, cid = db_with_circuit
        for corner, temp in [(None, None), (ConditionState.NONE, None),
                             ('tt', ConditionState.NONE), ('tt', ConditionState.SWEEP)]:
            db.add_simulation_result(cid, 'gain', value=float('nan'), unit='dB',
                                     corner=corner, temperature=temp)
        summary = db.get_simulation_summary(cid)
        assert 'unavailable dB (4 corners)' in summary
        assert summary.count('unavailable dB') == 5
        for label in ('nominal', 'none', 'temperature none', 'temperature sweep'):
            assert label in summary
        assert len(db.get_corners_for_parameter(cid, 'gain', latest_only=True)) == 4

    @pytest.mark.parametrize('locked', [False, True])
    def test_summary_ignores_old_cached_text(self, db_with_circuit, locked):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, 'voltage', value=1, unit='V', corner='tt')
        db.add_simulation_result(cid, 'voltage', value=1000, unit='mV', corner='tt')
        db.conn.execute('UPDATE circuit_versions SET simulation_summary = ?, '
                        'last_summary_update = ? WHERE version_id = ?',
                        ('obsolete: voltage 1000 V', '2020-01-01 00:00:00', cid))
        db.conn.commit()
        if locked:
            db.lock_version(cid)
        before = list(db.conn.execute('SELECT * FROM circuit_versions'))
        observations = list(db.conn.execute('SELECT * FROM simulation_results'))
        summary = db.get_simulation_summary(cid)
        assert 'obsolete' not in summary
        assert '1 V (tt, 1 corner)' in summary
        assert '1e+03 mV (tt, 1 corner)' in summary
        assert before == list(db.conn.execute('SELECT * FROM circuit_versions'))
        assert observations == list(db.conn.execute('SELECT * FROM simulation_results'))

    def test_summary_read_does_not_populate_cache(self, db_with_circuit):
        db, cid = db_with_circuit
        db.add_simulation_result(cid, 'gain', value=42, unit='dB')
        summary = db.get_simulation_summary(cid)
        assert '42 dB' in summary
        assert tuple(db.conn.execute('SELECT simulation_summary, last_summary_update '
                                     'FROM circuit_versions WHERE version_id = ?', (cid,)).fetchone()) == (None, None)
        updated = db.update_simulation_summary(cid)
        cache = db.conn.execute('SELECT simulation_summary, last_summary_update '
                                'FROM circuit_versions WHERE version_id = ?', (cid,)).fetchone()
        assert cache[0] == updated
        assert cache[1] is not None
        with pytest.raises(DatabaseError):
            db.get_simulation_summary('not-a-snapshot')

    def test_latest_only_preserves_unit_series(self, db_with_circuit):
        db, cid = db_with_circuit
        for unit, old, latest in [('V', 1, 2), ('mV', 1000, 2000), (None, 3, 4), ('', 5, 6)]:
            for value in (old, latest):
                db.add_simulation_result(cid, 'voltage', value=value, unit=unit, corner='tt')
        assert len(db.get_corners_for_parameter(cid, 'voltage')) == 8
        rows = db.get_corners_for_parameter(cid, 'voltage', latest_only=True)
        assert {row['unit']: row['value'] for row in rows} == {
            'V': 2, 'mV': 2000, None: 4, '': 6,
        }
        assert len(rows) == 4

    @pytest.mark.parametrize('text_last', [True, False])
    def test_latest_only_and_summary_share_numeric_text_identity(self, db_with_circuit, text_last):
        db, cid = db_with_circuit
        observations = [{'value': 60}, {'value_text': 'Error: extractor failed'}]
        if not text_last:
            observations.reverse()
        for observation in observations:
            db.add_simulation_result(cid, 'gain', unit='dB', corner='tt',
                                     temperature=27, voltage=1.8,
                                     timestamp='2026-10-08 00:00:00', **observation)
        row, = db.get_corners_for_parameter(cid, 'gain', latest_only=True)
        assert row['value'] == ('Error: extractor failed' if text_last else 60)
        summary = db.get_simulation_summary(cid)
        numeric, statuses = summary.split('=== STATUS ===')
        assert 'Error: extractor failed dB' in statuses
        if text_last:
            assert '60' not in summary
            assert 'gain' not in numeric
        else:
            assert '60 dB' in numeric
            assert 'latest of 2' in numeric

    def test_latest_only_and_summary_share_timestamp_precedence(self, db_with_circuit):
        db, cid = db_with_circuit
        for value, timestamp in [(60, '2026-10-08 00:00:00'),
                                 (70, '2026-10-09 00:00:00'),
                                 (80, '2026-10-08 00:00:00')]:
            db.add_simulation_result(cid, 'gain', value=value, unit='dB', corner='tt', timestamp=timestamp)
        row, = db.get_corners_for_parameter(cid, 'gain', latest_only=True)
        assert row['value'] == 70
        summary = db.get_simulation_summary(cid)
        assert '70 dB (tt, 1 corner) (latest of 3)' in summary
        assert '80 dB' not in summary
