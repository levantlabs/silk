"""store_net_currents — tb-level driver over annotated definition copies."""
import sqlite3

import pytest

from silk.circuit.modules import Module
from silk.circuit.modules_results import CurrentAnnotationReport
from silk.db.circuit_db import Database
from silk.db.context import CharacterizationContext
from silk.db.identity import STRUCTURE_FINGERPRINT_SCHEME
from silk.exceptions import LockedVersionError


def _fake_report(
    *, results_file='/tmp/rawfile.raw', analysis='transient', window=(1e-6,)
):
    return CurrentAnnotationReport(
        results_file=results_file, analysis=analysis, window=window)


def _build_dut(pdk, tmp_dir, r2=1000.0, db_path=None):
    from silk.elements import elements
    from silk.circuit.circuit_elements import Port
    m = Module('PDK_skywater130', tmp_dir, 'dut_blk.cir',
               pdk_details=pdk, module_name='dut_blk', db_path=db_path)
    m.define_module_ports({'A': Port.analog_input(), 'B': Port.analog_output()})
    m.add_net('mid')
    m.add(elements.resistor('r1', p='A', n='mid', value=1000.0))
    m.add(elements.resistor('r2', p='mid', n='B', value=r2))
    return m


def _build_tb(pdk, tmp_dir, db_path=None, dut=None):
    tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk,
                db_path=db_path)
    tb.add_net('x')
    tb.add_module(
        dut or _build_dut(pdk, tmp_dir),
        'dut',
        nets={'A': 'x', 'B': '0'},
    )
    return tb


def _annotate_definition(tb):
    """Hand-populate the definition copy the way annotate_currents would:
    'mid' fully annotated (aggregate + pins); port nets 'A'/'B' partial
    (pin rows only, aggregates None)."""
    defn = tb.module_dict['dut_blk']['device']
    for net_name, pin_avg in (('mid', 1e-3), ('A', 1e-3), ('B', -1e-3)):
        net = defn.net_dict[net_name]
        for b in net.pins:
            b.avg_current = pin_avg if b.instance == 'r1' else -pin_avg
            b.rms_current = abs(pin_avg)
            b.peak_current = abs(pin_avg)
            b.current_vector = f'i(v.x.{b.instance})'
            b._current_source = 'spice'
    mid = defn.net_dict['mid']
    mid.avg_current = 1e-3      # ½·Σ of the two 1 mA pins
    mid.rms_current = 1e-3
    mid.peak_current = 1e-3
    mid._current_source = 'spice'
    return defn


def test_store_writes_aggregate_and_pin_rows(pdk, tmp_dir, tmp_path):
    db_path = str(tmp_path / 't.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    versions = tb.store_net_currents(_fake_report(), corner='tt')
    assert 'dut_blk' in versions
    tracker = tb._ensure_tracker()
    rows = tracker.db.conn.execute(
        "SELECT nc.net_name, nc.pin_id, nc.avg_a, ncs.corner, "
        "ncs.structure_hash, ncs.results_file, ncs.analysis, ncs.tstart "
        "FROM net_currents AS nc "
        "JOIN net_current_sets AS ncs ON ncs.current_set_id = nc.current_set_id"
    ).fetchall()
    by = {(r[0], r[1]): r for r in rows}
    assert ('mid', '') in by                      # aggregate row
    assert by[('mid', '')][2] == pytest.approx(1e-3)
    assert ('mid', 'r1.n') in by or ('mid', 'r1.p') in by   # pin rows exist
    assert ('A', '') not in by                    # partial net: NO aggregate row
    assert any(k[0] == 'A' and k[1] != '' for k in by)          # but pin rows yes
    assert all(r[3] == 'tt' and r[5] == '/tmp/rawfile.raw' for r in rows)
    assert len({r[4] for r in rows}) == 1         # one structure_hash throughout
    # provenance: every row carries the report's analysis type and window start
    assert all(r[6] == 'transient' for r in rows)
    assert all(r[7] == pytest.approx(1e-6) for r in rows)


def test_store_context_and_origin_survive_close_reopen(
    pdk, tmp_dir, tmp_path
):
    db_path = str(tmp_path / 'context.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    versions = tb.store_net_currents(
        _fake_report(
            results_file='/tmp/context.raw',
            analysis='transient',
            window=(2e-6,),
        ),
        corner='ff',
        temperature=125.0,
        voltage=1.62,
        tool_name='ngspice-44',
    )
    origin_version_id = versions['dut_blk']
    tb.close()

    with Database(db_path) as reopened:
        selected_origin, rows = reopened.get_net_current_set(
            origin_version_id, 'ff'
        )
        version_hash = reopened._snapshot_row(origin_version_id)['structure_hash']

    assert selected_origin == origin_version_id
    assert rows
    assert all(row['origin_version_id'] == origin_version_id for row in rows)
    assert all(row['timestamp'] is not None for row in rows)
    assert {
        (
            row['corner'], row['temperature'], row['voltage'],
            row['tool_name'], row['analysis'], row['tstart'],
            row['results_file'], row['structure_scheme'],
            row['structure_hash'],
        )
        for row in rows
    } == {
        (
            'ff', 125.0, 1.62, 'ngspice-44', 'transient', 2e-6,
            '/tmp/context.raw', STRUCTURE_FINGERPRINT_SCHEME, version_hash,
        )
    }
    assert {row['source'] for row in rows} == {'spice'}


def test_restore_replaces_complete_corner_set(pdk, tmp_dir, tmp_path):
    db_path = str(tmp_path / 'replace.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    defn = _annotate_definition(tb)
    versions = tb.store_net_currents(_fake_report(), corner='tt')
    version_id = versions['dut_blk']
    tb.store_net_currents(_fake_report(), corner='ss')

    for binding in defn.net_dict['B'].pins:
        binding.avg_current = None
        binding.rms_current = None
        binding.peak_current = None
    tb.store_net_currents(_fake_report(), corner='tt')

    rows = tb._ensure_tracker().db.conn.execute(
        """
        SELECT nc.net_name, nc.pin_id, ncs.corner, ncs.state
        FROM net_currents AS nc
        JOIN net_current_sets AS ncs ON ncs.current_set_id = nc.current_set_id
        WHERE ncs.circuit_version_id = ?
        """,
        (version_id,),
    ).fetchall()
    assert rows
    assert not any(
        row['net_name'] == 'B'
        and row['corner'] == 'tt'
        and row['state'] == 'current'
        for row in rows
    )
    assert any(
        row['net_name'] == 'B'
        and row['corner'] == 'tt'
        and row['state'] == 'superseded'
        for row in rows
    )
    assert any(
        row['net_name'] == 'B'
        and row['corner'] == 'ss'
        and row['state'] == 'current'
        for row in rows
    )


def test_failed_grouped_restore_preserves_prior_complete_set(
    pdk, tmp_dir, tmp_path, monkeypatch
):
    db_path = str(tmp_path / 'rollback.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    defn = _annotate_definition(tb)
    versions = tb.store_net_currents(_fake_report(), corner='tt')
    version_id = versions['dut_blk']
    tracker = tb._ensure_tracker()
    before = tracker.db.conn.execute(
        """
        SELECT nc.net_name, nc.pin_id, nc.avg_a, nc.rms_a, nc.peak_a, nc.source
        FROM net_currents AS nc
        JOIN net_current_sets AS ncs ON ncs.current_set_id = nc.current_set_id
        WHERE ncs.circuit_version_id = ? AND ncs.corner = 'tt'
        ORDER BY nc.net_name, nc.pin_id
        """,
        (version_id,),
    ).fetchall()

    original_collect = tracker._collect_current_rows

    def collect_with_bad_second_row(module):
        rows = original_collect(module)
        rows.insert(
            1,
            ('mid', 'injected.pin', 9.0, 9.0, 9.0, 'invalid', None),
        )
        return rows

    defn.net_dict['mid'].avg_current = 2e-3
    monkeypatch.setattr(
        tracker, '_collect_current_rows', collect_with_bad_second_row
    )
    with pytest.raises(ValueError, match='source'):
        tb.store_net_currents(_fake_report(), corner='tt')

    after = tracker.db.conn.execute(
        """
        SELECT nc.net_name, nc.pin_id, nc.avg_a, nc.rms_a, nc.peak_a, nc.source
        FROM net_currents AS nc
        JOIN net_current_sets AS ncs ON ncs.current_set_id = nc.current_set_id
        WHERE ncs.circuit_version_id = ? AND ncs.corner = 'tt'
        ORDER BY nc.net_name, nc.pin_id
        """,
        (version_id,),
    ).fetchall()
    assert [tuple(row) for row in after] == [tuple(row) for row in before]


def test_failed_first_store_rolls_back_new_snapshot(
    pdk, tmp_dir, tmp_path, monkeypatch
):
    db_path = str(tmp_path / 'first-store-rollback.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    tracker = tb._ensure_tracker()
    original_collect = tracker._collect_current_rows

    def collect_with_duplicate_row(module):
        rows = original_collect(module)
        rows.insert(1, rows[0])
        return rows

    monkeypatch.setattr(
        tracker, '_collect_current_rows', collect_with_duplicate_row
    )
    with pytest.raises(sqlite3.IntegrityError):
        tb.store_net_currents(_fake_report(), corner='tt')

    assert tracker.db.conn.execute(
        "SELECT COUNT(*) FROM circuit_versions WHERE circuit_name = 'dut_blk'"
    ).fetchone()[0] == 0
    assert tracker.db.conn.execute(
        "SELECT COUNT(*) FROM net_currents"
    ).fetchone()[0] == 0
    assert tracker.db.conn.execute(
        "SELECT COUNT(*) FROM modules"
    ).fetchone()[0] == 0


def test_store_requires_corner(pdk, tmp_dir, tmp_path):
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 't.db'))
    _annotate_definition(tb)
    with pytest.raises(RuntimeError, match='corner is required'):
        tb.store_net_currents(_fake_report(), corner=None)


def test_store_reuses_version_and_supersedes_selected_set(pdk, tmp_dir, tmp_path):
    db_path = str(tmp_path / 't.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    v1 = tb.store_net_currents(_fake_report(), corner='tt')
    v2 = tb.store_net_currents(_fake_report(), corner='tt')
    assert v1 == v2                               # same version resolved, not a new one
    tracker = tb._ensure_tracker()
    n = tracker.db.conn.execute("SELECT COUNT(*) FROM net_currents").fetchone()[0]
    n_versions = tracker.db.conn.execute(
        "SELECT COUNT(*) FROM circuit_versions WHERE circuit_name='dut_blk'"
    ).fetchone()[0]
    assert n_versions == 1
    current_rows = tracker.db.conn.execute(
        """
        SELECT COUNT(*) FROM net_currents AS nc
        JOIN net_current_sets AS ncs
          ON ncs.current_set_id = nc.current_set_id
        WHERE ncs.state = 'current'
        """
    ).fetchone()[0]
    # Replacement preserves one complete current set and retains immutable
    # historical rows under superseded parent IDs.
    tb.store_net_currents(_fake_report(), corner='tt')
    assert tracker.db.conn.execute(
        "SELECT COUNT(*) FROM net_currents"
    ).fetchone()[0] > n
    assert tracker.db.conn.execute(
        """
        SELECT COUNT(*) FROM net_currents AS nc
        JOIN net_current_sets AS ncs
          ON ncs.current_set_id = nc.current_set_id
        WHERE ncs.state = 'current'
        """
    ).fetchone()[0] == current_rows


def test_store_nothing_annotated_raises(pdk, tmp_dir, tmp_path):
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 't.db'))
    with pytest.raises(RuntimeError, match='no annotated definition'):
        tb.store_net_currents(_fake_report(), corner='tt')


def test_store_skips_db_sourced_rows(pdk, tmp_dir, tmp_path):
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 't.db'))
    defn = _annotate_definition(tb)
    for b in defn.net_dict['mid'].pins:
        b._current_source = 'db'                  # loaded, not measured
    defn.net_dict['mid']._current_source = 'db'
    tb.store_net_currents(_fake_report(), corner='tt')
    tracker = tb._ensure_tracker()
    rows = tracker.db.conn.execute(
        "SELECT net_name FROM net_currents WHERE net_name='mid'").fetchall()
    assert rows == []                             # laundering prevented

def test_store_uses_generated_module_characterization_context(
    pdk, tmp_dir, tmp_path
):
    db_path = str(tmp_path / 'scoped-context.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    tb.characterization_context = CharacterizationContext(
        owner_path='$',
        test_name='max_load',
        parameters={'cload_f': 2e-12},
        corner='tt',
        temperature=85.0,
        voltage=1.62,
    )

    versions = tb.store_net_currents(_fake_report(), corner='tt')
    version_id = versions['dut_blk']
    tracker = tb._ensure_tracker()
    profile = tracker.db.get_characterization_profile(
        version_id, context=tb.characterization_context
    )
    parent = tracker.db.conn.execute(
        """
        SELECT profile_id, corner, temperature, voltage
        FROM net_current_sets
        WHERE circuit_version_id = ? AND state = 'current'
        """,
        (version_id,),
    ).fetchone()

    assert tuple(parent) == (
        profile.profile_id,
        'tt',
        85.0,
        1.62,
    )


def test_store_requires_db(pdk, tmp_dir):
    tb = _build_tb(pdk, tmp_dir, db_path=None)
    _annotate_definition(tb)
    with pytest.raises(RuntimeError, match='db_path'):
        tb.store_net_currents(_fake_report(), corner='tt')


def test_store_rejects_fabricated_context_before_database_mutation(
    pdk, tmp_dir, tmp_path
):
    tb = _build_tb(
        pdk,
        tmp_dir,
        db_path=str(tmp_path / 'fabricated-context.db'),
    )
    _annotate_definition(tb)
    tb.characterization_context = {
        'owner_path': '$',
        'test_name': 'invented',
    }

    with pytest.raises(TypeError, match='CharacterizationContext'):
        tb.store_net_currents(_fake_report(), corner='tt')

    tracker = tb._ensure_tracker()
    assert tracker.db.conn.execute(
        "SELECT COUNT(*) FROM circuit_versions WHERE circuit_name = 'dut_blk'"
    ).fetchone()[0] == 0
    assert tracker.db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0


def test_store_adopts_explicit_db_path(pdk, tmp_dir, tmp_path):
    tb = _build_tb(pdk, tmp_dir, db_path=None)
    _annotate_definition(tb)
    versions = tb.store_net_currents(_fake_report(), corner='tt',
                                     db_path=str(tmp_path / 'adopt.db'))
    assert 'dut_blk' in versions


def test_store_db_path_mismatch_raises(pdk, tmp_dir, tmp_path):
    bound = str(tmp_path / 'bound.db')
    other = str(tmp_path / 'other.db')
    tb = _build_tb(pdk, tmp_dir, db_path=bound)
    _annotate_definition(tb)
    with pytest.raises(ValueError) as excinfo:
        tb.store_net_currents(_fake_report(), corner='tt', db_path=other)
    assert bound in str(excinfo.value) and other in str(excinfo.value)


def test_store_db_path_same_path_is_noop(pdk, tmp_dir, tmp_path):
    bound = str(tmp_path / 'bound.db')
    tb = _build_tb(pdk, tmp_dir, db_path=bound)
    _annotate_definition(tb)
    versions = tb.store_net_currents(_fake_report(), corner='tt', db_path=bound)
    assert 'dut_blk' in versions


def test_store_rejects_locked_version(pdk, tmp_dir, tmp_path):
    """2026-08-23: the net_currents upsert is raw SQL (bypasses the guarded
    Database write methods), so store_net_currents enforces the lock itself."""
    db_path = str(tmp_path / 't.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    versions = tb.store_net_currents(_fake_report(), corner='tt')
    tracker = tb._ensure_tracker()
    row = tracker.db.conn.execute(
        "SELECT circuit_name, version FROM circuit_versions WHERE version_id = ?",
        (versions['dut_blk'],)).fetchone()
    tracker.db.lock_circuit(row['circuit_name'], row['version'])
    with pytest.raises(LockedVersionError, match='locked'):
        tb.store_net_currents(_fake_report(), corner='tt')


@pytest.mark.parametrize('temperature,voltage', [(85.0, None), (None, 1.8), (None, None)])
def test_current_snapshot_prevents_defaults_and_late_relabeling(
    pdk, tmp_dir, tmp_path, temperature, voltage,
):
    from silk.exceptions import EvidenceMismatchError
    from silk.simulation import OperatingConditions
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 'snapshot.db'))
    _annotate_definition(tb)
    tb.set_corner('ss', temperature=-40, voltage=5)
    tb.operating_conditions = OperatingConditions(
        'silk', 'ngspice', 'ff', 'fixed' if temperature is not None else 'unknown',
        temperature, 'backend-observed' if temperature is not None else 'unknown',
        supply_voltage_v=voltage,
    )
    tb.store_net_currents(_fake_report(), corner='ff')
    conn = tb._tracker.db.conn
    before = list(conn.iterdump())
    row = conn.execute('SELECT corner, temperature, voltage, profile_id FROM net_current_sets').fetchone()
    assert tuple(row) == ('ff', temperature, voltage, None)
    for kwargs in ({'corner': 'ss'}, {'corner': 'ff', 'temperature': 27},
                   {'corner': 'ff', 'voltage': 5}):
        with pytest.raises(EvidenceMismatchError):
            tb.store_net_currents(_fake_report(), **kwargs)
        assert list(conn.iterdump()) == before
    tb.close()


@pytest.mark.parametrize('change', ['structure', 'specs'])
def test_current_store_rejects_stale_bound_parent(pdk, tmp_dir, tmp_path, change):
    from silk.exceptions import EvidenceMismatchError
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 'stale.db'))
    _annotate_definition(tb)
    tb.create_circuit_version()
    if change == 'structure':
        tb.add_net('changed')
    else:
        tb.declare_spec('gain', 'gain', '>=', 1., key='gain')
    before = list(tb._tracker.db.conn.iterdump())
    try:
        with pytest.raises(EvidenceMismatchError):
            tb.store_net_currents(_fake_report(), corner='tt')
        assert list(tb._tracker.db.conn.iterdump()) == before
    finally:
        tb.close()


@pytest.mark.parametrize('change', ['structure', 'specs'])
def test_current_store_validates_each_definition_target(pdk, tmp_dir, tmp_path, monkeypatch, change):
    from silk.exceptions import EvidenceMismatchError
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 'target.db'))
    defn = _annotate_definition(tb)
    original = tb.store_net_currents(_fake_report(), corner='tt')['dut_blk']
    if change == 'structure':
        defn.add_net('changed')
    else:
        defn.declare_spec('gain', 'gain', '>=', 1., key='gain')
    t = tb._tracker
    before = list(t.db.conn.iterdump())
    # Fault at the resolution boundary: storage must independently check its target.
    monkeypatch.setattr(t, 'resolve_version', lambda *a, **kw: original)
    try:
        with pytest.raises(EvidenceMismatchError):
            tb.store_net_currents(_fake_report(), corner='tt')
        assert list(t.db.conn.iterdump()) == before
        assert t.circuit_id is None
    finally:
        tb.close()


def test_current_store_validation_failure_rolls_back_all_definitions(pdk, tmp_dir, tmp_path, monkeypatch):
    import copy
    from silk.exceptions import EvidenceMismatchError
    tb = _build_tb(pdk, tmp_dir, db_path=str(tmp_path / 'atomic.db'))
    defn = _annotate_definition(tb)
    old = tb.store_net_currents(_fake_report(), corner='tt')['dut_blk']
    second = copy.deepcopy(defn)
    second.module_name = 'second'
    second.add_net('different')
    tb.add_module(second, 'second_instance', nets={'A': 'x', 'B': '0'})
    defn.add_net('new_snapshot')
    t = tb._tracker
    resolve = t.resolve_version
    monkeypatch.setattr(t, 'resolve_version', lambda name, identity, **kw:
                        old if name == 'second' else resolve(name, identity, **kw))
    before = list(t.db.conn.iterdump())
    try:
        with pytest.raises(EvidenceMismatchError):
            tb.store_net_currents(_fake_report(), corner='tt')
        assert list(t.db.conn.iterdump()) == before
    finally:
        tb.close()
