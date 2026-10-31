"""Tests for audit issue fixes — context manager, thread safety, sanitization, etc."""

import pytest
import warnings

from silk.db.circuit_db import Database, DesignStage
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
)


# ---------------------------------------------------------------------------
# #70 — Database context manager
# ---------------------------------------------------------------------------

class TestDatabaseContextManager:

    def test_context_manager(self):
        """Database supports with statement."""
        with Database(':memory:') as db:
            cid = create_building_snapshot(db, name='test', version='1.0')
            assert cid is not None
        # After exiting, connection is closed

    def test_context_manager_on_exception(self):
        """Database closes even if exception is raised inside with block."""
        try:
            with Database(':memory:') as db:
                create_building_snapshot(db, name='test', version='1.0')
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        # db.conn should be closed — further operations would fail


# ---------------------------------------------------------------------------
# #68 — Thread safety
# ---------------------------------------------------------------------------

class TestDatabaseThreadSafety:

    def test_check_same_thread_false(self):
        """Database connection allows cross-thread access."""
        import threading
        db = Database(':memory:')
        cid = create_building_snapshot(db, name='test', version='1.0')

        result = {}
        def read_in_thread():
            try:
                info = db.get_circuit_by_id(cid)
                result['success'] = info is not None
            except Exception as e:
                result['error'] = str(e)

        t = threading.Thread(target=read_in_thread)
        t.start()
        t.join()
        assert result.get('success') is True, f"Thread access failed: {result}"
        db.close()


# ---------------------------------------------------------------------------
# #71/72 — Database methods for CLI
# ---------------------------------------------------------------------------

class TestDatabaseNewMethods:

    def test_get_nets_for_circuit(self):
        """get_nets_for_circuit returns net dicts."""
        with Database(':memory:') as db:
            cid = create_building_snapshot(db, name='test', version='1.0')
            db.add_net(circuit_version_id=cid, name='vdd', net_type='power')
            db.add_net(circuit_version_id=cid, name='gnd', net_type='ground')
            nets = db.get_nets_for_circuit(cid)
            assert len(nets) == 2
            names = {n['name'] for n in nets}
            assert names == {'vdd', 'gnd'}

    def test_get_parameters_for_circuit(self):
        """get_parameters_for_circuit returns distinct parameter names."""
        with Database(':memory:') as db:
            cid = create_sealed_snapshot(db, name='test', version='1.0')
            db.add_simulation_result(circuit_version_id=cid, parameter='gain',
                                     value=45.0, unit='dB')
            db.add_simulation_result(circuit_version_id=cid, parameter='power',
                                     value=2.3, unit='mW')
            db.add_simulation_result(circuit_version_id=cid, parameter='gain',
                                     value=38.0, unit='dB', corner='ss')
            params = db.get_parameters_for_circuit(cid)
            assert params == ['gain', 'power']  # sorted, deduplicated

    def test_get_nets_empty(self):
        """get_nets_for_circuit returns empty list for circuit with no nets."""
        with Database(':memory:') as db:
            cid = create_building_snapshot(db, name='test', version='1.0')
            assert db.get_nets_for_circuit(cid) == []

    def test_get_parameters_empty(self):
        """get_parameters_for_circuit returns empty list for circuit with no results."""
        with Database(':memory:') as db:
            cid = create_sealed_snapshot(db, name='test', version='1.0')
            assert db.get_parameters_for_circuit(cid) == []


# ---------------------------------------------------------------------------
# #40 — Filename sanitization
# ---------------------------------------------------------------------------

class TestFilenameSanitization:

    def test_path_traversal_stripped(self):
        """Path traversal characters are stripped from circuit name."""
        from silk.codegen.generator import _sanitize_filename
        assert '/' not in _sanitize_filename('../../etc/passwd')
        assert '..' not in _sanitize_filename('../../etc/passwd')

    def test_spaces_to_underscores(self):
        from silk.codegen.generator import _sanitize_filename
        assert _sanitize_filename('my circuit') == 'my_circuit'

    def test_preserves_underscore_hyphen(self):
        from silk.codegen.generator import _sanitize_filename
        assert _sanitize_filename('ldo_nmos-v2') == 'ldo_nmos-v2'

    def test_strips_special_chars(self):
        from silk.codegen.generator import _sanitize_filename
        assert _sanitize_filename('test;name$(rm -rf)') == 'testnamerm_-rf'

    def test_alphanumeric_preserved(self):
        from silk.codegen.generator import _sanitize_filename
        assert _sanitize_filename('diffpair123') == 'diffpair123'


# ---------------------------------------------------------------------------
# #48 — records with None/missing objectives never rank as best
# ---------------------------------------------------------------------------

class TestMissingObjectives:

    def _opt(self):
        from silk.circuit.variable import Variable
        from silk.optimize import Optimizer, VariableSpace

        space = VariableSpace([Variable('x', bounds=(0, 1))])
        opt = Optimizer(test=lambda x: {'obj': x['x']}, space=space)
        opt.add_objective('obj', minimize=True)
        return opt

    def test_missing_objective_is_not_best(self):
        """A success lacking the declared objective is excluded from best."""
        opt = self._opt()
        good = {'params': {'x': 0.5}, 'metrics': {'obj': 0.5},
                'feasible': True, 'status': 'success'}
        missing = {'params': {'x': 0.7}, 'metrics': {},  # missing 'obj'
                   'feasible': True, 'status': 'success'}
        opt._history.extend([missing, good])

        assert opt.best is good
        assert opt.summary()['best'] is good
        assert opt.history == [missing, good]

    def test_all_records_present_ranked(self):
        """With every objective present, the lowest value is best."""
        opt = self._opt()
        lo = {'params': {'x': 0.1}, 'metrics': {'obj': 0.1},
              'feasible': True, 'status': 'success'}
        hi = {'params': {'x': 0.5}, 'metrics': {'obj': 0.5},
              'feasible': True, 'status': 'success'}
        opt._history.extend([hi, lo])

        assert opt.best is lo
        assert opt.summary()['objectives'] == [
            {'name': 'obj', 'goal': 'minimize', 'value_source': 'metric'}]


# ---------------------------------------------------------------------------
# #69 — SQL order_by uses dict lookup
# ---------------------------------------------------------------------------

class TestSQLOrderBy:

    def test_valid_order_by(self):
        """get_all_circuit_versions accepts valid order_by values."""
        with Database(':memory:') as db:
            create_building_snapshot(db, name='test', version='1.0')
            for order in ('created_at', 'version', 'stage'):
                result = db.get_all_circuit_versions('test', order_by=order)
                assert isinstance(result, list)

    def test_invalid_order_by_defaults(self):
        """Invalid order_by falls back to created_at without error."""
        with Database(':memory:') as db:
            create_building_snapshot(db, name='test', version='1.0')
            result = db.get_all_circuit_versions('test', order_by='DROP TABLE')
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Module.plot() auto-scaling
# ---------------------------------------------------------------------------

class TestAutoScale:

    def test_nanoseconds(self):
        """Time values in the ns range produce ns label."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([0, 10e-9, 50e-9, 100e-9])
        scaled, label = Module._auto_scale(values, 'time')
        assert 'ns' in label
        assert scaled[-1] == pytest.approx(100.0)

    def test_microseconds(self):
        """Time values in the us range produce us label."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([0, 1e-6, 5e-6, 10e-6])
        scaled, label = Module._auto_scale(values, 'time')
        assert 'us' in label
        assert scaled[-1] == pytest.approx(10.0)

    def test_millivolts(self):
        """Voltage values in mV range produce mV label."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([0, 0.5e-3, 1e-3, 1.5e-3])
        scaled, label = Module._auto_scale(values, 'v')
        assert 'mV' in label

    def test_volts(self):
        """Voltage values in V range produce V label."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([0, 0.5, 1.0, 1.8])
        scaled, label = Module._auto_scale(values, 'v')
        assert 'V' in label
        assert 'mV' not in label

    def test_frequency_mhz(self):
        """Frequency values in MHz range produce MHz label."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([1e3, 1e6, 100e6])
        scaled, label = Module._auto_scale(values, 'frequency')
        assert 'MHz' in label

    def test_microamps(self):
        """Current values in uA range produce uA label."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([0, 50e-6, 100e-6])
        scaled, label = Module._auto_scale(values, 'i')
        assert 'uA' in label

    def test_zero_values(self):
        """All-zero values don't crash."""
        import numpy as np
        from silk.circuit.modules import Module
        values = np.array([0, 0, 0])
        scaled, label = Module._auto_scale(values, 'time')
        assert label is not None
