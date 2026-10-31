"""Runtime root ownership survives persistent owner-path evidence workflows."""
import pytest

from silk import Module
from tests._env import needs_ngspice
from tests.circuit.test_selected_testbench import leaf, root


@needs_ngspice
@pytest.mark.parametrize('voltages', [(1, 2), (2, 2)])
def test_repeated_spec_versions_store_and_restore_each_owner(tmp_path, voltages):
    child, tb = leaf(tmp_path)
    spec = tb.add_spec('level limit', 'level', '>=', 1.5)
    top = Module(None, str(tmp_path), 'top.cir', None, db_path=str(tmp_path / 'evidence.db'))
    from silk.specs import Measurement
    parent_spec = top.declare_spec('parent budget', 'budget', '>=', 0)
    parent_spec.record_measurement(Measurement('user', 3))
    parents = []
    try:
        for name, voltage in zip(('low', 'high'), voltages):
            carrier = root(tmp_path)
            carrier.set_supply_voltage(voltage)
            carrier.set_temperature(27)
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(child, 'cell', out='out', vss='0')
            top.add_module(carrier, name, nets={})
            parents.append(carrier)
        reports = list(top.run_all_tests().values())
        top.create_circuit_version()
        for report in reports:
            top.store_report(report)
        spec.measurements.clear()
        top.restore_spec_measurements()
        outcomes = top.reconcile_specs().outcomes
        parent = next(o for o in outcomes if o.spec is parent_spec)
        children = [o for o in outcomes if o.spec is spec]
        assert parent.status == 'PASS' and parent.measurement.value == 3
        assert [o.measurement.value for o in children] == pytest.approx(voltages)
        assert [o.status for o in children] == [('PASS' if v >= 1.5 else 'FAIL') for v in voltages]
    finally:
        top.close()
        child.close()
        for carrier in parents:
            carrier.close()


@needs_ngspice
@pytest.mark.parametrize('finish', ['commit', 'rollback', 'rejected_commit'])
def test_delayed_publication_keeps_staged_root_after_selection_changes(tmp_path, finish):
    import sqlite3
    from silk.circuit.test_placement import execution_scope
    child, tb = leaf(tmp_path)
    spec = tb.add_spec('level limit', 'level', '>=', 1.5)
    roots = [Module(None, str(tmp_path), name + '.cir', None,
                    db_path=str(tmp_path / (name + '.db'))) for name in ('a', 'b')]
    try:
        reports = []
        for top, voltage in zip(roots, (1, 2)):
            top.set_supply_voltage(voltage)
            top.set_temperature(27)
            top.add_module(child, 'x1', out='out', vss='0')
            reports.append(next(iter(top.run_all_tests().values())))
            top.create_circuit_version()
        a, b = roots
        before = dict(spec.measurements)
        db = a._tracker.db
        transaction = db.transaction()
        transaction.__enter__()
        a.store_report(reports[0])
        assert spec.measurements == before
        def deny_commit(operation, argument, *_):
            if operation == sqlite3.SQLITE_TRANSACTION and argument == 'COMMIT':
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        try:
            with execution_scope(b.list_test_entries()[0]):
                if finish == 'rollback':
                    error = RuntimeError('rollback')
                    transaction.__exit__(RuntimeError, error, None)
                elif finish == 'rejected_commit':
                    db.conn.set_authorizer(deny_commit)
                    with pytest.raises(sqlite3.DatabaseError):
                        transaction.__exit__(None, None, None)
                else:
                    transaction.__exit__(None, None, None)
        finally:
            db.conn.set_authorizer(None)
        profiles = [m for m in spec.measurements.values() if m.profile_id is not None]
        if finish == 'commit':
            assert len(profiles) == 1
            measurement = profiles[0]
            assert measurement._owner_scope[0]() is a.__dict__['_test_evidence']
            assert measurement._owner_scope[1] == '$/x1'
            assert measurement.value == 1
        else:
            assert spec.measurements == before
        assert a.reconcile_specs().outcomes[0].status == 'FAIL'
        assert b.reconcile_specs().outcomes[0].status == 'PASS'
    finally:
        for top in roots:
            top.close()
        child.close()


def test_current_restore_keeps_each_owner_off_shared_net(tmp_path):
    from types import SimpleNamespace
    child, _ = leaf(tmp_path)
    child.declare_spec('current', 'net.out.peak_current', '<=', 3)
    top = Module(None, str(tmp_path), 'top.cir', None, db_path=str(tmp_path / 'current.db'))
    for name in ('a', 'b'):
        top.add_module(child, name, out=name, vss='0')
    child.net_dict['out'].peak_current = 99
    try:
        top.create_circuit_version()
        tracker = top._tracker
        for name, value in [('a', 2), ('b', 5)]:
            row = SimpleNamespace(net_name='out', pin_id='', avg_a=value,
                                  peak_a=value, rms_a=value)
            staged = tracker._stage_current_restore([row], force=True, owner_path='$/'+name)
            tracker._apply_current_restore(staged, owner_path='$/'+name)
        outcomes = top.reconcile_specs().outcomes
        assert [o.measurement.value for o in outcomes] == [2, 5]
        assert child.net_dict['out'].peak_current == 99
    finally:
        top.close()
        child.close()
