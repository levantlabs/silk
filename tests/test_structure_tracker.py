"""
Tests for pynetlist.db.structure_tracker.CircuitStructureTracker.

Uses in-memory SQLite via Database to avoid touching disk.
PyNetlist instances are replaced with lightweight SimpleNamespace mocks.
"""

import pytest
from types import SimpleNamespace

from silk.db.circuit_db import Database, DesignStage
from silk.db.identity import (
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprint_spec_set_records,
    _fingerprint_structure,
)
from silk.db.models import Fingerprint, SnapshotIdentity
from silk.db.structure_tracker import CircuitStructureTracker
from silk.exceptions import (
    CircuitError,
    DatabaseError,
    EvidenceMismatchError,
    UnsupportedFingerprintSchemeError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_netlist(elem_dict=None, net_dict=None, module_dict=None):
    """Build a minimal netlist-like object accepted by extract_structure."""
    nl = SimpleNamespace()
    nl.elem_dict = elem_dict if elem_dict is not None else {}
    nl.net_dict = net_dict if net_dict is not None else {}
    nl.module_dict = module_dict if module_dict is not None else {}
    return nl


def make_device(device_type='nmos', parameters=None, nets=None):
    """Build a minimal device object."""
    dev = SimpleNamespace()
    dev.deviceType = device_type
    dev.parameters = parameters if parameters is not None else {'w': 1e-6, 'l': 0.15e-6}
    dev.nets = nets if nets is not None else {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'gnd'}
    return dev


def make_module_obj(elem_dict=None, net_dict=None, module_name='my_module'):
    """Build a minimal Module-like object."""
    mod = SimpleNamespace()
    mod.elem_dict = elem_dict if elem_dict is not None else {}
    mod.net_dict = net_dict if net_dict is not None else {}
    mod.module_name = module_name
    mod.pin_list = {}
    mod.parameters = {}
    return mod


def empty_structure():
    return {
        'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [],
        'passives': [], 'modules': {},
    }


def identity_for(structure):
    structure_fingerprint, _module_fingerprints = _fingerprint_structure(
        structure
    )
    return SnapshotIdentity(
        structure=structure_fingerprint,
        spec_set=_fingerprint_spec_set_records(()),
    )


def tracker_for_structure(db, structure, *, debug=False):
    """Create a building snapshot whose declared identity matches structure."""
    circuit_id = db.create_circuit(
        name='test_amp',
        version='1.0',
        stage=DesignStage.DEV,
        description='test circuit',
        identity=identity_for(structure),
    )
    return CircuitStructureTracker(db, circuit_id, debug=debug)


def set_cached_module_hashes(tracker, hashes):
    """Set the legacy digest view and its authoritative scheme tags."""
    tracker._last_module_hashes = dict(hashes)
    tracker._last_module_fingerprints = {
        name: Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, digest)
        for name, digest in hashes.items()
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    d = Database(':memory:')
    yield d
    d.close()


@pytest.fixture
def circuit_id(db):
    return db.create_circuit(
        name='test_amp',
        version='1.0',
        stage=DesignStage.DEV,
        description='test circuit',
        identity=identity_for(empty_structure()),
    )


@pytest.fixture
def tracker(db, circuit_id):
    return CircuitStructureTracker(db, circuit_id)


@pytest.fixture
def tracker_no_id(db):
    return CircuitStructureTracker(db, None)


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------

class TestInit:

    def test_no_circuit_id_hash_is_none(self, db):
        """Tracker with circuit_id=None starts with no stored hash."""
        t = CircuitStructureTracker(db, None)
        assert t.get_structure_hash() is None

    def test_no_circuit_id_run_count_zero(self, db):
        """Tracker with circuit_id=None starts with run count 0."""
        t = CircuitStructureTracker(db, None)
        assert t.get_run_count() == 0

    def test_with_circuit_id_does_not_crash(self, db, circuit_id):
        """Initializing with a real circuit_id does not raise."""
        t = CircuitStructureTracker(db, circuit_id)
        assert t is not None

    def test_with_circuit_id_hash_is_none_initially(self, db, circuit_id):
        """Fresh circuit has no structure hash stored."""
        t = CircuitStructureTracker(db, circuit_id)
        assert t.get_structure_hash() is None

    def test_debug_true_does_not_crash(self, db, circuit_id):
        """debug=True initializes without errors."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        assert t is not None

    def test_circuit_id_stored(self, db, circuit_id):
        """circuit_id attribute is stored correctly."""
        t = CircuitStructureTracker(db, circuit_id)
        assert t.circuit_id == circuit_id


# ---------------------------------------------------------------------------
# extract_structure tests
# ---------------------------------------------------------------------------

class TestExtractStructure:

    def test_empty_netlist_returns_dict(self, tracker):
        """extract_structure on empty netlist returns a dict."""
        nl = make_mock_netlist()
        result = tracker.extract_structure(nl)
        assert isinstance(result, dict)

    def test_empty_netlist_has_required_keys(self, tracker):
        """Result always contains transistors and nets keys."""
        nl = make_mock_netlist()
        result = tracker.extract_structure(nl)
        assert 'transistors' in result
        assert 'nets' in result
        assert 'hierarchy' in result
        assert 'modules' in result

    def test_empty_netlist_transistors_list_empty(self, tracker):
        """No elements means empty transistors list."""
        nl = make_mock_netlist()
        result = tracker.extract_structure(nl)
        assert result['transistors'] == []

    def test_empty_netlist_nets_list_empty(self, tracker):
        """No nets means empty nets list."""
        nl = make_mock_netlist()
        result = tracker.extract_structure(nl)
        assert result['nets'] == []

    def test_one_device_type_d_appears_in_transistors(self, tracker):
        """A dict-based type='d' element is added to transistors list."""
        dev = make_device('nmos')
        nl = make_mock_netlist(
            elem_dict={'M1': {'device': dev, 'type': 'd'}}
        )
        result = tracker.extract_structure(nl)
        assert len(result['transistors']) == 1
        assert result['transistors'][0]['name'] == 'M1'

    def test_device_type_field_captured(self, tracker):
        """Device type attribute is captured from deviceType."""
        dev = make_device('pmos')
        nl = make_mock_netlist(
            elem_dict={'M2': {'device': dev, 'type': 'd'}}
        )
        result = tracker.extract_structure(nl)
        assert result['transistors'][0]['type'] == 'pmos'

    def test_device_parameters_captured(self, tracker):
        """Device parameters dict is captured."""
        dev = make_device(parameters={'w': 2e-6, 'l': 0.18e-6})
        nl = make_mock_netlist(
            elem_dict={'M3': {'device': dev, 'type': 'd'}}
        )
        result = tracker.extract_structure(nl)
        params = result['transistors'][0]['parameters']
        assert params.get('w') == 2e-6
        assert params.get('l') == 0.18e-6

    def test_device_nets_captured(self, tracker):
        """Device net connections are captured."""
        dev = make_device(nets={'d': 'vout', 'g': 'vin', 's': 'gnd', 'b': 'gnd'})
        nl = make_mock_netlist(
            elem_dict={'M4': {'device': dev, 'type': 'd'}}
        )
        result = tracker.extract_structure(nl)
        nets = result['transistors'][0]['nets']
        assert nets.get('d') == 'vout'

    def test_net_dict_captured(self, tracker):
        """Nets from net_dict are included in nets list."""
        nl = make_mock_netlist(net_dict={'vout': {}, 'gnd': {}})
        result = tracker.extract_structure(nl)
        net_names = [n['name'] for n in result['nets']]
        assert 'vout' in net_names
        assert 'gnd' in net_names

    def test_net_type_power_classified(self, tracker):
        """Net named vdd is classified as power."""
        nl = make_mock_netlist(net_dict={'vdd': {}})
        result = tracker.extract_structure(nl)
        vdd_net = result['nets'][0]
        assert vdd_net['type'] == 'power'

    def test_net_type_ground_classified(self, tracker):
        """Net named gnd is classified as ground."""
        nl = make_mock_netlist(net_dict={'gnd': {}})
        result = tracker.extract_structure(nl)
        gnd_net = result['nets'][0]
        assert gnd_net['type'] == 'ground'

    def test_net_type_signal_classified(self, tracker):
        """Net named vout is classified as signal."""
        nl = make_mock_netlist(net_dict={'vout': {}})
        result = tracker.extract_structure(nl)
        sig_net = result['nets'][0]
        assert sig_net['type'] == 'signal'

    def test_module_type_m_goes_to_hierarchy(self, tracker):
        """A dict-based type='m' element is added to hierarchy list."""
        mod = make_module_obj()
        nl = make_mock_netlist(
            elem_dict={'U1': {'device': mod, 'type': 'm'}}
        )
        result = tracker.extract_structure(nl)
        assert len(result['hierarchy']) == 1
        assert result['hierarchy'][0]['instance_name'] == 'U1'

    def test_multiple_devices_all_captured(self, tracker):
        """Multiple devices are all captured without duplicates."""
        dev1 = make_device('nmos')
        dev2 = make_device('pmos')
        nl = make_mock_netlist(
            elem_dict={
                'M1': {'device': dev1, 'type': 'd'},
                'M2': {'device': dev2, 'type': 'd'},
            }
        )
        result = tracker.extract_structure(nl)
        assert len(result['transistors']) == 2

    def test_module_dict_entry_populates_modules(self, tracker):
        """module_dict entry (dict with 'device') populates structure['modules']."""
        mod = make_module_obj()
        nl = make_mock_netlist(
            module_dict={'amp_core': {'device': mod, 'defined': True}}
        )
        result = tracker.extract_structure(nl)
        assert 'amp_core' in result['modules']

    def test_module_dict_entry_has_hash(self, tracker):
        """Each module in structure['modules'] has a hash key."""
        mod = make_module_obj()
        nl = make_mock_netlist(
            module_dict={'amp_core': {'device': mod, 'defined': True}}
        )
        result = tracker.extract_structure(nl)
        assert 'hash' in result['modules']['amp_core']

    def test_debug_mode_extract_does_not_crash(self, db, circuit_id):
        """extract_structure with debug=True prints but does not raise."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        nl = make_mock_netlist(
            elem_dict={'M1': {'device': make_device(), 'type': 'd'}},
            net_dict={'vout': {}}
        )
        result = t.extract_structure(nl)
        assert isinstance(result, dict)

    def test_module_devices_recursively_extracted(self, tracker):
        """Devices inside a module elem_dict are recursively extracted."""
        inner_dev = make_device('nmos')
        mod = make_module_obj(
            elem_dict={'Minner': {'device': inner_dev, 'type': 'd'}}
        )
        nl = make_mock_netlist(
            elem_dict={'U1': {'device': mod, 'type': 'm'}}
        )
        result = tracker.extract_structure(nl)
        names = [d['name'] for d in result['transistors']]
        assert 'U1.Minner' in names


# ---------------------------------------------------------------------------
# compute_structure_hash tests
# ---------------------------------------------------------------------------

class TestComputeStructureHash:

    def _simple_structure(self):
        return {
            'transistors': [{'name': 'M1', 'type': 'nmos', 'parameters': {'w': 1e-6}, 'nets': {}}],
            'nets': [{'name': 'vout', 'type': 'signal'}],
            'hierarchy': [],
            'parameters': [],
            'modules': {}
        }

    def test_returns_tuple(self, tracker):
        """compute_structure_hash returns a 2-tuple."""
        s = self._simple_structure()
        result = tracker.compute_structure_hash(s)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hash_is_string(self, tracker):
        """First element of tuple is a string."""
        s = self._simple_structure()
        h, _ = tracker.compute_structure_hash(s)
        assert isinstance(h, str)

    def test_module_hashes_is_dict(self, tracker):
        """Second element is a dict."""
        s = self._simple_structure()
        _, mh = tracker.compute_structure_hash(s)
        assert isinstance(mh, dict)

    def test_deterministic_same_input(self, tracker):
        """Same structure always produces the same hash."""
        s = self._simple_structure()
        h1, _ = tracker.compute_structure_hash(s)
        h2, _ = tracker.compute_structure_hash(s)
        assert h1 == h2

    def test_different_structures_produce_different_hashes(self, tracker):
        """Different structures produce different hashes."""
        s1 = self._simple_structure()
        s2 = self._simple_structure()
        s2['transistors'][0]['parameters']['w'] = 99e-6
        h1, _ = tracker.compute_structure_hash(s1)
        h2, _ = tracker.compute_structure_hash(s2)
        assert h1 != h2

    def test_order_independent_transistors(self, tracker):
        """Transistors in different order produce the same hash."""
        t1 = {'name': 'M1', 'type': 'nmos', 'parameters': {}, 'nets': {}}
        t2 = {'name': 'M2', 'type': 'pmos', 'parameters': {}, 'nets': {}}
        s_a = {'transistors': [t1, t2], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        s_b = {'transistors': [t2, t1], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h_a, _ = tracker.compute_structure_hash(s_a)
        h_b, _ = tracker.compute_structure_hash(s_b)
        assert h_a == h_b

    def test_empty_structure_hash_stable(self, tracker):
        """Empty structure always hashes to the same value."""
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h1, _ = tracker.compute_structure_hash(s)
        h2, _ = tracker.compute_structure_hash(s)
        assert h1 == h2

    def test_adding_transistor_changes_hash(self, tracker):
        """Adding a transistor changes the structure hash."""
        base = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        with_t = {
            'transistors': [{'name': 'M1', 'type': 'nmos', 'parameters': {}, 'nets': {}}],
            'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}
        }
        h_base, _ = tracker.compute_structure_hash(base)
        h_with, _ = tracker.compute_structure_hash(with_t)
        assert h_base != h_with

    def test_module_hashes_included_in_overall(self, tracker):
        """Module hashes are extracted into the returned module_hashes dict."""
        s = {
            'transistors': [],
            'nets': [],
            'hierarchy': [],
            'parameters': [],
            'modules': {'core': {'hash': 'abc123', 'devices': [], 'nets': [], 'submodules': {}}}
        }
        _, mh = tracker.compute_structure_hash(s)
        assert 'core' in mh
        assert mh['core'] == 'abc123'


# ---------------------------------------------------------------------------
# has_structure_changed tests
# ---------------------------------------------------------------------------

class TestHasStructureChanged:

    def test_no_prior_hash_returns_true(self, tracker_no_id):
        """has_structure_changed returns True when no hash is stored."""
        assert tracker_no_id.has_structure_changed('somehash') is True

    def test_debug_unchanged_does_not_crash(self, db, circuit_id):
        """has_structure_changed with debug=True, unchanged hash, prints without error (lines 705-710)."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = t.compute_structure_hash(s)
        t.store_structure(s, h, mh)
        # Should print "Structure unchanged" debug message
        result = t.has_structure_changed(h)
        assert result is False

    def test_debug_changed_does_not_crash(self, db, circuit_id):
        """has_structure_changed with debug=True, changed hash, prints without error."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = t.compute_structure_hash(s)
        t.store_structure(s, h, mh)
        result = t.has_structure_changed('completelydifferenthash')
        assert result is True

    def test_same_hash_returns_false_after_store(self, tracker):
        """has_structure_changed returns False when hash matches stored hash."""
        structure = {
            'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}
        }
        h, mh = tracker.compute_structure_hash(structure)
        tracker.store_structure(structure, h, mh)
        assert tracker.has_structure_changed(h) is False

    def test_different_hash_returns_true_after_store(self, tracker):
        """has_structure_changed returns True when hash differs from stored."""
        structure = {
            'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}
        }
        h, mh = tracker.compute_structure_hash(structure)
        tracker.store_structure(structure, h, mh)
        assert tracker.has_structure_changed('completelydifferenthash') is True

    def test_unknown_stored_structure_scheme_is_never_compared(
        self, db, circuit_id
    ):
        db.conn.execute(
            """
            UPDATE circuit_versions SET structure_scheme = 'silk.structure/v999'
            WHERE version_id = ?
            """,
            (circuit_id,),
        )
        db.conn.commit()

        with pytest.raises(
            UnsupportedFingerprintSchemeError, match='silk.structure/v999'
        ):
            CircuitStructureTracker(db, circuit_id)


# ---------------------------------------------------------------------------
# store_structure / get_structure_hash tests
# ---------------------------------------------------------------------------

class TestStoreStructure:

    def _empty_structure(self):
        return {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}

    def test_stored_hash_is_retrievable(self, tracker):
        """After store_structure, get_structure_hash returns the stored hash."""
        s = self._empty_structure()
        h, mh = tracker.compute_structure_hash(s)
        tracker.store_structure(s, h, mh)
        assert tracker.get_structure_hash() == h

    def test_store_with_device(self, db, tracker):
        """store_structure with a transistor does not raise."""
        dev = make_device()
        nl = make_mock_netlist(elem_dict={'M1': {'device': dev, 'type': 'd'}})
        t = tracker
        s = t.extract_structure(nl)
        h, mh = t.compute_structure_hash(s)
        t = tracker_for_structure(db, s)
        t.store_structure(s, h, mh)  # Should not raise
        assert t.get_structure_hash() == h

    def test_store_with_net(self, db, tracker):
        """store_structure with a net does not raise."""
        nl = make_mock_netlist(net_dict={'vout': {}})
        s = tracker.extract_structure(nl)
        h, mh = tracker.compute_structure_hash(s)
        t = tracker_for_structure(db, s)
        t.store_structure(s, h, mh)
        assert t.get_structure_hash() == h

    def test_store_with_module_hashes_none(self, tracker):
        """store_structure with module_hashes=None extracts from structure."""
        s = self._empty_structure()
        h, _ = tracker.compute_structure_hash(s)
        tracker.store_structure(s, h, None)  # module_hashes=None
        assert tracker.get_structure_hash() == h

    def test_store_updates_last_structure_hash_cache(self, tracker):
        """After store_structure, internal cache _last_structure_hash matches."""
        s = self._empty_structure()
        h, mh = tracker.compute_structure_hash(s)
        tracker.store_structure(s, h, mh)
        assert tracker._last_structure_hash == h

    def test_store_with_modules_stores_module_in_db(self, db, tracker):
        """store_structure with modules dict stores the module via DB (lines 810-857)."""
        mod = make_module_obj(
            elem_dict={'M1': {'device': make_device(), 'type': 'd'}},
            net_dict={'out': {}, 'gnd': {}},
        )
        nl = make_mock_netlist(
            module_dict={'amp_core': {'device': mod, 'defined': True}}
        )
        s = tracker.extract_structure(nl)
        h, mh = tracker.compute_structure_hash(s)
        t = tracker_for_structure(db, s)
        t.store_structure(s, h, mh)  # Should store the module in DB
        assert t.get_structure_hash() == h

    def test_unknown_persisted_module_scheme_is_rejected_on_reload(self, db):
        mod = make_module_obj(
            elem_dict={'M1': {'device': make_device(), 'type': 'd'}},
            net_dict={'out': {}, 'gnd': {}},
        )
        structure = CircuitStructureTracker(db, None).extract_structure(
            make_mock_netlist(
                module_dict={'amp_core': {'device': mod, 'defined': True}}
            )
        )
        fingerprint, module_fingerprints = _fingerprint_structure(structure)
        tracker = tracker_for_structure(db, structure)
        tracker.store_structure(
            structure,
            fingerprint.digest,
            {
                path: value.digest
                for path, value in module_fingerprints.items()
            },
        )
        db.seal_circuit_version(tracker.circuit_id)
        db.conn.execute(
            "UPDATE modules SET module_hash_scheme = 'silk.structure/v999'"
        )
        db.conn.commit()

        with pytest.raises(
            UnsupportedFingerprintSchemeError, match='silk.structure/v999'
        ):
            CircuitStructureTracker(db, tracker.circuit_id)

    def test_module_passives_survive_structure_storage(self, db, tracker):
        mod = make_module_obj()
        mod.passives_dict = {
            'r_load': SimpleNamespace(
                value=1_000.0, pos='out', neg='0'
            )
        }
        nl = make_mock_netlist(
            module_dict={'core': {'device': mod, 'defined': True}}
        )
        s = tracker.extract_structure(nl)
        assert s['modules']['core']['passives'] == [{
            'name': 'r_load',
            'kind': 'SimpleNamespace',
            'value': 1_000.0,
            'pos': 'out',
            'neg': '0',
        }]
        h, mh = tracker.compute_structure_hash(s)
        t = tracker_for_structure(db, s)
        t.store_structure(s, h, mh)

        stored = db.get_circuit_modules(t.circuit_id)[0]['structure']
        assert stored['passives'] == s['modules']['core']['passives']

    def test_forged_module_hash_is_rejected(self, db):
        s = empty_structure()
        s['modules'] = {
            'core': {
                'hash': 'forged',
                'devices': [],
                'nets': [],
                'submodules': {},
                'passives': [],
                'device_count': 0,
            }
        }
        t = tracker_for_structure(db, s)
        h, mh = t.compute_structure_hash(s)

        with pytest.raises(EvidenceMismatchError, match='module hash'):
            t.store_structure(s, h, mh)

    def test_store_with_module_hashes_none_and_modules_present(self, db, tracker):
        """store_structure with None module_hashes extracts them from structure dict (lines 727-728)."""
        mod = make_module_obj()
        nl = make_mock_netlist(
            module_dict={'core': {'device': mod, 'defined': True}}
        )
        s = tracker.extract_structure(nl)
        h, _ = tracker.compute_structure_hash(s)
        # Pass None for module_hashes when modules are non-empty
        t = tracker_for_structure(db, s)
        t.store_structure(s, h, None)  # Should extract module_hashes from structure
        assert t.get_structure_hash() == h

    def test_store_with_modules_debug_mode(self, db, circuit_id):
        """store_structure with modules and debug=True prints without crashing."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        mod = make_module_obj()
        nl = make_mock_netlist(
            module_dict={'core': {'device': mod, 'defined': True}}
        )
        s = t.extract_structure(nl)
        h, mh = t.compute_structure_hash(s)
        t = tracker_for_structure(db, s, debug=True)
        t.store_structure(s, h, mh)
        assert t.get_structure_hash() == h

    def test_store_existing_device_is_rejected(self, db, tracker):
        """Structure rows are insert-only within a building snapshot."""
        dev = make_device('nmos')
        nl = make_mock_netlist(elem_dict={'M1': {'device': dev, 'type': 'd'}})
        s = tracker.extract_structure(nl)
        h, mh = tracker.compute_structure_hash(s)
        t = tracker_for_structure(db, s)

        # First store
        t.store_structure(s, h, mh)

        with pytest.raises(DatabaseError, match='insert-only'):
            t.store_structure(s, h, mh)


# ---------------------------------------------------------------------------
# increment_run_count / get_run_count tests
# ---------------------------------------------------------------------------

class TestRunCount:

    def test_initial_run_count_is_zero(self, tracker_no_id):
        """Run count starts at 0 when no circuit_id."""
        assert tracker_no_id.get_run_count() == 0

    def test_increment_increases_by_one(self, tracker):
        """increment_run_count increases count by 1."""
        # First store structure so metadata row exists
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = tracker.compute_structure_hash(s)
        tracker.store_structure(s, h, mh)
        # After store, run_count is reset to 1
        before = tracker.get_run_count()
        tracker.increment_run_count()
        assert tracker.get_run_count() == before + 1

    def test_increment_twice(self, tracker):
        """Calling increment_run_count twice gives count + 2."""
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = tracker.compute_structure_hash(s)
        tracker.store_structure(s, h, mh)
        before = tracker.get_run_count()
        tracker.increment_run_count()
        tracker.increment_run_count()
        assert tracker.get_run_count() == before + 2

    def test_increment_run_count_debug_mode(self, db, circuit_id):
        """increment_run_count with debug=True prints without crashing (lines 871-883)."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = t.compute_structure_hash(s)
        t.store_structure(s, h, mh)
        before = t.get_run_count()
        t.increment_run_count()
        assert t.get_run_count() == before + 1

    def test_store_resets_run_count_to_one(self, tracker):
        """store_structure resets internal run count to 1."""
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = tracker.compute_structure_hash(s)
        tracker.store_structure(s, h, mh)
        assert tracker.get_run_count() == 1


# ---------------------------------------------------------------------------
# Hash change detection end-to-end tests
# ---------------------------------------------------------------------------

class TestHashChangeDetection:

    def test_hash_changes_when_transistor_added(self, tracker):
        """Adding a transistor produces a different hash."""
        base = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        dev = make_device()
        nl_with = make_mock_netlist(elem_dict={'M1': {'device': dev, 'type': 'd'}})
        s_with = tracker.extract_structure(nl_with)

        h_base, _ = tracker.compute_structure_hash(base)
        h_with, _ = tracker.compute_structure_hash(s_with)
        assert h_base != h_with

    def test_hash_stable_across_two_empty_extractions(self, tracker):
        """Two extractions of the same empty netlist produce equal hashes."""
        nl = make_mock_netlist()
        s1 = tracker.extract_structure(nl)
        s2 = tracker.extract_structure(nl)
        h1, _ = tracker.compute_structure_hash(s1)
        h2, _ = tracker.compute_structure_hash(s2)
        assert h1 == h2

    def test_full_workflow_change_detected(self, tracker):
        """Full workflow: store empty structure, then changed structure is detected."""
        empty = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h_empty, mh = tracker.compute_structure_hash(empty)
        tracker.store_structure(empty, h_empty, mh)

        dev = make_device()
        nl = make_mock_netlist(elem_dict={'M1': {'device': dev, 'type': 'd'}})
        s_new = tracker.extract_structure(nl)
        h_new, _ = tracker.compute_structure_hash(s_new)

        assert tracker.has_structure_changed(h_new) is True

    def test_full_workflow_no_change_detected(self, db, tracker):
        """Full workflow: store structure, then same structure shows no change."""
        dev = make_device()
        nl = make_mock_netlist(
            elem_dict={'M1': {'device': dev, 'type': 'd'}},
            net_dict={'vout': {}, 'gnd': {}}
        )
        s = tracker.extract_structure(nl)
        h, mh = tracker.compute_structure_hash(s)
        t = tracker_for_structure(db, s)
        t.store_structure(s, h, mh)

        # Extract again from same netlist
        s2 = tracker.extract_structure(nl)
        h2, _ = tracker.compute_structure_hash(s2)
        assert t.has_structure_changed(h2) is False


# ---------------------------------------------------------------------------
# _load_last_structure: reload stored hash from metadata (lines 66-72)
# ---------------------------------------------------------------------------

class TestGetModuleChanges:

    def test_get_module_changes_returns_last_hashes(self, tracker):
        """get_module_changes() returns the _last_module_hashes dict (line 1203)."""
        tracker._last_module_hashes = {'m1': 'hash1', 'm2': 'hash2'}
        result = tracker.get_module_changes()
        assert result == {'m1': 'hash1', 'm2': 'hash2'}

    def test_get_module_changes_empty_initially(self, tracker_no_id):
        """get_module_changes() returns empty dict initially."""
        assert tracker_no_id.get_module_changes() == {}


class TestLoadLastStructure:

    def test_reload_hash_after_new_tracker(self, db, circuit_id):
        """A new tracker for the same circuit_id reloads the stored structure hash."""
        t1 = CircuitStructureTracker(db, circuit_id)
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = t1.compute_structure_hash(s)
        t1.store_structure(s, h, mh)
        db.store_spec_declarations(circuit_id, ())
        db.seal_circuit_version(circuit_id)

        # Create a second tracker for the same circuit — it should load the hash
        t2 = CircuitStructureTracker(db, circuit_id)
        assert t2.get_structure_hash() == h

    def test_update_run_count_with_malformed_json(self, db, circuit_id):
        """_update_run_count handles malformed existing JSON silently (lines 1039-1040)."""
        # First store a valid structure to set up metadata
        t = CircuitStructureTracker(db, circuit_id)
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = t.compute_structure_hash(s)
        t.store_structure(s, h, mh)

        # Now corrupt the metadata
        cursor = db.conn.cursor()
        cursor.execute(
            "UPDATE circuit_versions SET metadata = ? WHERE version_id = ?",
            ('INVALID JSON', circuit_id)
        )
        db.conn.commit()

        # increment_run_count calls _update_run_count — should not crash
        t.increment_run_count()
        assert t.get_run_count() == 2

    def test_malformed_json_metadata_does_not_crash(self, db, circuit_id):
        """_load_last_structure with malformed JSON in metadata silently ignores it (lines 71-72)."""
        # Inject malformed JSON into circuit metadata
        cursor = db.conn.cursor()
        cursor.execute(
            "UPDATE circuit_versions SET metadata = ? WHERE version_id = ?",
            ('NOT VALID JSON {{{', circuit_id)
        )
        db.conn.commit()

        # Creating a new tracker should not raise even with malformed JSON
        t = CircuitStructureTracker(db, circuit_id)
        assert t.get_structure_hash() is None  # Failed to load → None

    def test_reload_run_count_after_new_tracker(self, db, circuit_id):
        """A new tracker for the same circuit_id reloads the run count from metadata."""
        t1 = CircuitStructureTracker(db, circuit_id)
        s = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        h, mh = t1.compute_structure_hash(s)
        t1.store_structure(s, h, mh)
        t1.increment_run_count()  # run_count is now 2

        t2 = CircuitStructureTracker(db, circuit_id)
        assert t2.get_run_count() == 2


# ---------------------------------------------------------------------------
# _is_pymodule and _is_transistor helper methods (lines 298, 310-323)
# ---------------------------------------------------------------------------

class TestHelperMethods:

    def test_is_pymodule_with_elem_dict(self, tracker):
        """_is_pymodule returns True for object with elem_dict."""
        obj = make_module_obj()
        assert tracker._is_pymodule(obj) is True

    def test_is_pymodule_with_module_names(self, tracker):
        """_is_pymodule returns True for object with module_names attribute."""
        from types import SimpleNamespace
        obj = SimpleNamespace()
        obj.module_names = ['module']
        assert tracker._is_pymodule(obj) is True

    def test_is_pymodule_false_for_plain_device(self, tracker):
        """_is_pymodule returns False for a plain device object."""
        dev = make_device()
        # Plain device has no elem_dict or module_names → False
        assert tracker._is_pymodule(dev) is False

    def test_is_transistor_with_deviceType(self, tracker):
        """_is_transistor returns True for object with deviceType."""
        dev = make_device('nmos')
        assert tracker._is_transistor(dev) is True

    def test_is_transistor_with_pin_list(self, tracker):
        """_is_transistor returns True for object with pin_list."""
        from types import SimpleNamespace
        obj = SimpleNamespace()
        obj.pin_list = {}
        assert tracker._is_transistor(obj) is True

    def test_is_transistor_false_for_unknown(self, tracker):
        """_is_transistor returns False for objects without device attributes."""
        from types import SimpleNamespace
        obj = SimpleNamespace()  # No deviceType, parameters, pin_list, params, or nets
        assert tracker._is_transistor(obj) is False

    def test_is_transistor_false_for_primitives(self, tracker):
        """_is_transistor returns False for strings, ints, None."""
        assert tracker._is_transistor("hello") is False
        assert tracker._is_transistor(42) is False
        assert tracker._is_transistor(None) is False
        assert tracker._is_transistor([1, 2, 3]) is False

    def test_is_transistor_with_params_and_nets(self, tracker):
        """_is_transistor returns True for silk AnalogElement (has params + nets)."""
        from types import SimpleNamespace
        obj = SimpleNamespace()
        obj.params = {'w': 1.0, 'l': 0.15}
        obj.nets = {'d': 'out', 'g': 'in'}
        assert tracker._is_transistor(obj) is True


# ---------------------------------------------------------------------------
# _extract_transistor_info: various attribute formats (lines 348-412)
# ---------------------------------------------------------------------------

class TestExtractTransistorInfo:

    def test_device_type_from_device_type_attr(self, tracker):
        """_extract_transistor_info uses device.device_type when deviceType absent."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.device_type = 'resistor'
        dev.parameters = {}
        dev.nets = {}
        info = tracker._extract_transistor_info('R1', dev, None)
        assert info is not None
        assert info['type'] == 'resistor'

    def test_device_type_from_pdk_device(self, tracker):
        """_extract_transistor_info uses pdk_device as type when no other type attr."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.pdk_device = 'sky130_nfet'
        dev.parameters = {}
        dev.nets = {}
        info = tracker._extract_transistor_info('M1', dev, None)
        assert info is not None
        assert info['type'] == 'sky130_nfet'

    def test_device_model_attr_captured(self, tracker):
        """_extract_transistor_info captures model attribute."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.model = 'sky130_nfet'
        dev.parameters = {}
        dev.nets = {}
        info = tracker._extract_transistor_info('M1', dev, None)
        assert info is not None
        assert info.get('model') == 'sky130_nfet'

    def test_parameters_dict_format(self, tracker):
        """_extract_transistor_info handles parameters dict directly."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.deviceType = 'resistor'
        dev.parameters = {'r': 10e3}
        dev.nets = {'p': 'vout', 'n': 'gnd'}
        info = tracker._extract_transistor_info('R1', dev, None)
        assert info['parameters']['r'] == 10e3

    def test_parameters_with_value_attr(self, tracker):
        """_extract_transistor_info unpacks parameters with .value attribute."""
        from types import SimpleNamespace

        class ParamVal:
            value = 5e3

        dev = SimpleNamespace()
        dev.deviceType = 'resistor'
        dev.parameters = {'r': ParamVal()}
        dev.nets = {}
        info = tracker._extract_transistor_info('R1', dev, None)
        assert info['parameters']['r'] == 5e3

    def test_nets_dict_format(self, tracker):
        """_extract_transistor_info handles nets dict directly."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.deviceType = 'nmos'
        dev.nets = {'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'}
        dev.parameters = {}
        info = tracker._extract_transistor_info('M1', dev, None)
        assert info['nets']['d'] == 'out'

    def test_device_type_from_type_attr(self, tracker):
        """_extract_transistor_info uses device.type when no deviceType/device_type/pdk_device."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.type = 'capacitor'
        dev.parameters = {}
        dev.nets = {}
        info = tracker._extract_transistor_info('C1', dev, None)
        assert info is not None
        assert info['type'] == 'capacitor'

    def test_exception_in_extract_raises_with_device_context(self, tracker):
        """An extraction error must not produce a partial structure identity."""
        class BrokenDevice:
            @property
            def deviceType(self):
                raise RuntimeError("broken!")
            parameters = {}
            nets = {}

        with pytest.raises(CircuitError, match="broken.*broken!"):
            tracker._extract_transistor_info('broken', BrokenDevice(), None)

    @pytest.mark.parametrize('variable', [False, True])
    def test_param_list_values_render_with_owner_units(self, tracker, variable):
        """Numeric and Variable values use actual legacy constraints."""
        from silk import load_pdk
        from silk.circuit.variable import Variable
        from silk.pdk.basepdk import ParameterConstraint

        value = Variable('width', value=2e-6) if variable else 2e-6
        dev = SimpleNamespace(
            deviceType='nmos', pin_list={},
            param_list={'w': ParameterConstraint(minval='0', maxval='-1', value=value,
                                                 pdk_unit='u', quantity='length')},
        )
        owner = SimpleNamespace(pdk_details=load_pdk(
            'sky130', user_units='si', library_location='/fake/lib'))
        info = tracker._extract_transistor_info('M1', dev, owner)
        assert info['parameters']['w'] == '2'
        assert dev.param_list['w'].value is value

    def test_param_list_without_constraint_metadata_raises(self, tracker):
        """Malformed legacy parameters cannot silently enter an identity."""
        dev = SimpleNamespace(deviceType='nmos', param_list={'m': 1}, pin_list={})
        with pytest.raises(CircuitError, match="M1.*m.*PDK unit metadata"):
            tracker._extract_transistor_info('M1', dev, None)

    def test_param_list_exception_with_debug(self, db, circuit_id):
        """Debug mode must also reject a partial parameter payload."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)

        class BrokenParam:
            @property
            def value(self):
                raise RuntimeError("broken param!")

        dev = SimpleNamespace(deviceType='nmos', param_list={'bad': BrokenParam()}, pin_list={})
        with pytest.raises(CircuitError, match="M1.*bad.*broken param"):
            t._extract_transistor_info('M1', dev, None)

    def test_pdk_device_captured_as_model(self, tracker):
        """_extract_transistor_info uses pdk_device as model when no model attr."""
        from types import SimpleNamespace
        dev = SimpleNamespace()
        dev.pdk_device = 'sky130_fd_pr__nfet_01v8'
        dev.parameters = {}
        dev.nets = {}
        info = tracker._extract_transistor_info('M1', dev, None)
        assert info.get('model') == 'sky130_fd_pr__nfet_01v8'


# ---------------------------------------------------------------------------
# _compare_module_hashes (lines 658-685)
# ---------------------------------------------------------------------------

class TestExtractModuleStructure:

    def test_module_with_submodule_gets_hash(self, tracker):
        """_extract_module_structure with a submodule computes its hash (lines 495-509)."""
        inner_dev = make_device('nmos')
        sub = make_module_obj(
            elem_dict={'Mn': {'device': inner_dev, 'type': 'd'}},
            net_dict={'out': {}, 'gnd': {}},
            module_name='subcell',
        )
        outer = make_module_obj(
            elem_dict={'Usub': {'device': sub, 'type': 'm'}},
            net_dict={'vdd': {}, 'gnd': {}},
            module_name='top_cell',
        )

        mod_struct = tracker._extract_module_structure(outer, 'top')
        assert 'Usub' in mod_struct['submodules']
        # The submodule hash should be a non-empty string
        assert isinstance(mod_struct['submodules']['Usub']['hash'], str)
        assert len(mod_struct['submodules']['Usub']['hash']) > 0
        assert mod_struct['submodules']['Usub']['ports'] == {}


class TestCompareModuleHashes:

    def test_new_module_shows_as_changed(self, tracker):
        """New module (not in previous hashes) shows as changed."""
        modules = {'amp_core': {'hash': 'newhash', 'devices': []}}
        changes = tracker._compare_module_hashes(modules)
        assert 'amp_core' in changes
        assert changes['amp_core'] is True  # new module → changed

    def test_unchanged_module_shows_not_changed(self, tracker):
        """Module with same hash as stored shows as unchanged."""
        # Simulate stored module hash
        set_cached_module_hashes(tracker, {'amp_core': 'existinghash'})
        modules = {'amp_core': {'hash': 'existinghash'}}
        changes = tracker._compare_module_hashes(modules)
        assert changes['amp_core'] is False

    def test_changed_module_detected(self, tracker):
        """Module with different hash shows as changed."""
        set_cached_module_hashes(tracker, {'amp_core': 'oldhash'})
        modules = {'amp_core': {'hash': 'newhash'}}
        changes = tracker._compare_module_hashes(modules)
        assert changes['amp_core'] is True

    def test_deleted_module_shows_as_changed(self, tracker):
        """Module present in stored hashes but not in current (alongside another current module) shows as changed."""
        set_cached_module_hashes(
            tracker, {'old_module': 'oldhash', 'remaining': 'rhash'}
        )
        # old_module is absent but remaining is still present (non-empty current_modules required)
        modules = {'remaining': {'hash': 'rhash'}}
        changes = tracker._compare_module_hashes(modules)
        assert changes.get('old_module') is True

    def test_unknown_cached_module_scheme_is_never_compared(self, tracker):
        tracker._last_module_fingerprints = {
            'amp_core': Fingerprint('silk.structure/v999', 'samehash')
        }
        modules = {'amp_core': {'hash': 'samehash'}}

        with pytest.raises(
            UnsupportedFingerprintSchemeError, match='silk.structure/v999'
        ):
            tracker._compare_module_hashes(modules)

    def test_empty_modules_returns_empty_dict(self, tracker):
        """_compare_module_hashes with empty current modules returns {}."""
        set_cached_module_hashes(tracker, {})
        changes = tracker._compare_module_hashes({})
        assert changes == {}

    def test_debug_mode_compare_does_not_crash(self, db, circuit_id):
        """_compare_module_hashes with debug=True prints without error."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        set_cached_module_hashes(t, {'m': 'old'})
        changes = t._compare_module_hashes({'m': {'hash': 'new'}})
        assert 'm' in changes

    def test_debug_mode_new_module(self, db, circuit_id):
        """_compare_module_hashes with debug=True on new module prints 'NEW' message."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        set_cached_module_hashes(t, {})
        changes = t._compare_module_hashes({'new_mod': {'hash': 'abc'}})
        assert changes['new_mod'] is True

    def test_debug_mode_deleted_module(self, db, circuit_id):
        """_compare_module_hashes with debug=True on deleted module prints 'DELETED' message."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        set_cached_module_hashes(
            t, {'old_m': 'oldhash', 'other': 'hash2'}
        )
        changes = t._compare_module_hashes({'other': {'hash': 'hash2'}})
        assert changes.get('old_m') is True


# ---------------------------------------------------------------------------
# _extract_parameters (lines 555-565)
# ---------------------------------------------------------------------------

class TestExtractModuleInfo:

    def test_module_with_parameters_extracted(self, tracker):
        """_extract_module_info extracts module parameters (lines 443-446)."""
        from types import SimpleNamespace

        class Param:
            value = 10.0

        mod = SimpleNamespace()
        mod.module_name = 'test_mod'
        mod.pin_list = {}
        mod.parameters = {'gain': Param(), 'bias': 5.0}
        info = tracker._extract_module_info('U1', mod, None)
        assert info is not None
        assert info['parameters']['gain'] == 10.0
        assert info['parameters']['bias'] == 5.0

    def test_module_exception_raises(self, tracker):
        """A failed instance extraction cannot yield a partial fingerprint."""
        class BrokenModule:
            @property
            def module_name(self):
                raise RuntimeError("broken module!")

        with pytest.raises(CircuitError, match='U_bad.*broken module'):
            tracker._extract_module_info('U_bad', BrokenModule(), None)

    def test_module_exception_debug_mode(self, db, circuit_id):
        """Debug mode preserves the same fail-loud contract."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)

        class BrokenModule:
            @property
            def module_name(self):
                raise RuntimeError("debug error!")

        with pytest.raises(CircuitError, match='U_debug.*debug error'):
            t._extract_module_info('U_debug', BrokenModule(), None)


class TestExtractParameters:

    def test_no_parameters_attr_returns_empty(self, tracker):
        """_extract_parameters returns [] when module has no parameters attr."""
        from types import SimpleNamespace
        mod = SimpleNamespace()  # no parameters attr
        result = tracker._extract_parameters('amp', mod)
        assert result == []

    def test_extracts_plain_value(self, tracker):
        """_extract_parameters extracts plain values directly."""
        from types import SimpleNamespace
        mod = SimpleNamespace()
        mod.parameters = {'gain': 10.0}
        result = tracker._extract_parameters('amp', mod)
        assert len(result) == 1
        assert result[0]['name'] == 'gain'
        assert result[0]['value'] == 10.0

    def test_extracts_value_attr(self, tracker):
        """_extract_parameters unpacks .value from parameter objects."""
        from types import SimpleNamespace

        class Param:
            value = 42.5

        mod = SimpleNamespace()
        mod.parameters = {'bw': Param()}
        result = tracker._extract_parameters('amp', mod)
        assert result[0]['value'] == 42.5


# _extract_from_module: old-style (non-dict) elements (lines 263-293)
# ---------------------------------------------------------------------------

class TestExtractFromModuleOldStyle:

    def test_non_dict_module_element_extracted(self, tracker):
        """_extract_from_module handles non-dict elem_dict entries (old-style)."""
        from types import SimpleNamespace

        # Old-style: elem_dict values are direct objects, not dicts
        inner_dev = make_device('pmos')
        # Don't put a 'type' wrapper — put the raw device
        mod = make_module_obj(elem_dict={'Mp': inner_dev})

        structure = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        tracker._extract_from_module(mod, structure, prefix='U1', processed=set())
        # The device has deviceType, so _is_transistor → True → should be extracted
        names = [d['name'] for d in structure['transistors']]
        assert 'U1.Mp' in names

    def test_non_dict_device_in_top_level_netlist(self, tracker):
        """extract_structure handles non-dict (old-style) device at top level."""
        dev = make_device('nmos')
        # Old-style: elem_dict value is a direct device object, not a dict
        nl = make_mock_netlist(elem_dict={'M_old': dev})
        result = tracker.extract_structure(nl)
        names = [d['name'] for d in result['transistors']]
        assert 'M_old' in names

    def test_non_dict_module_at_top_level(self, tracker):
        """extract_structure handles non-dict (old-style) module at top level."""
        mod = make_module_obj()
        nl = make_mock_netlist(elem_dict={'U_old': mod})
        result = tracker.extract_structure(nl)
        hier_names = [h['instance_name'] for h in result['hierarchy']]
        assert 'U_old' in hier_names

    def test_non_dict_device_debug_mode(self, db, circuit_id):
        """extract_structure with debug=True handles non-dict device without crashing."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        dev = make_device('nmos')
        nl = make_mock_netlist(elem_dict={'M_debug': dev})
        result = t.extract_structure(nl)
        assert any(d['name'] == 'M_debug' for d in result['transistors'])

    def test_extract_from_module_deduplicates_nets(self, tracker):
        """_extract_from_module doesn't add duplicate nets to structure['nets'] (lines 289-293)."""
        # First add 'gnd' to the structure nets manually
        structure = {
            'transistors': [],
            'nets': [{'name': 'gnd', 'type': 'ground'}],  # already present
            'hierarchy': [], 'parameters': [], 'modules': {}
        }
        dev = make_device('nmos')
        mod = make_module_obj(
            elem_dict={'M1': {'device': dev, 'type': 'd'}},
            net_dict={'gnd': {}, 'out': {}},  # 'gnd' is already in structure
        )
        tracker._extract_from_module(mod, structure, prefix='U1', processed=set(), processed_nets={'gnd'})
        # 'gnd' should not be duplicated; 'out' should be added
        net_names = [n['name'] for n in structure['nets']]
        assert net_names.count('gnd') == 1
        assert 'out' in net_names

    def test_extract_from_module_debug_with_dict_device(self, db, circuit_id):
        """_extract_from_module with debug=True and dict-style device prints debug (lines 225-266)."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        inner_dev = make_device('nmos')
        mod = make_module_obj(
            elem_dict={'M1': {'device': inner_dev, 'type': 'd'}},
            net_dict={'out': {}},
        )
        structure = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        t._extract_from_module(mod, structure, prefix='U1', processed=set())
        assert any(d['name'] == 'U1.M1' for d in structure['transistors'])

    def test_extract_from_module_debug_nested_module(self, db, circuit_id):
        """_extract_from_module with debug=True and nested module prints debug (line 251)."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        inner_dev = make_device('nmos')
        inner_mod = make_module_obj(
            elem_dict={'Mn': {'device': inner_dev, 'type': 'd'}}
        )
        outer_mod = make_module_obj(
            elem_dict={'Usub': {'device': inner_mod, 'type': 'm'}}
        )
        structure = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        t._extract_from_module(outer_mod, structure, prefix='TOP', processed=set())
        names = [d['name'] for d in structure['transistors']]
        assert 'TOP.Usub.Mn' in names

    def test_extract_from_module_no_elem_dict_debug(self, db, circuit_id):
        """_extract_from_module with debug=True and module with no elem_dict (line 283)."""
        t = CircuitStructureTracker(db, circuit_id, debug=True)
        from types import SimpleNamespace
        # Module with no elem_dict
        mod = SimpleNamespace()
        mod.net_dict = {'gnd': {}}  # has net_dict
        structure = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        t._extract_from_module(mod, structure, prefix='U_empty', processed=set())
        # Should add nets from net_dict
        net_names = [n['name'] for n in structure['nets']]
        assert 'gnd' in net_names

    def test_non_dict_module_in_module_recurses(self, tracker):
        """_extract_from_module recurses into non-dict Module sub-elements."""
        from types import SimpleNamespace

        inner_dev = make_device('nmos')
        inner_mod = make_module_obj(elem_dict={'Mn': inner_dev})

        # Old-style: elem_dict value is a Module-like object directly
        outer_mod = make_module_obj(elem_dict={'U2': inner_mod})

        structure = {'transistors': [], 'nets': [], 'hierarchy': [], 'parameters': [], 'modules': {}}
        tracker._extract_from_module(outer_mod, structure, prefix='TOP', processed=set())
        names = [d['name'] for d in structure['transistors']]
        assert 'TOP.U2.Mn' in names
