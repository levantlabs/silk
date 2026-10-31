"""load_net_currents — round-trip onto a fresh build, hash-verified."""
import pytest

from silk.db.circuit_db import Database
from silk.exceptions import (
    EvidenceMismatchError,
    UnsupportedFingerprintSchemeError,
)

# Reuse the store-test builders (same-module import keeps them single-source).
from tests.test_net_current_store import (
    _build_dut, _build_tb, _annotate_definition, _fake_report)


def _stored_db(pdk, tmp_dir, tmp_path, corner='tt'):
    db_path = str(tmp_path / 't.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    tb.store_net_currents(_fake_report(), corner=corner)
    return db_path


def test_round_trip_aggregates_and_pins(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    n = fresh.load_net_currents('tt', db_path=db_path)
    assert n > 0
    mid = fresh.net_dict['mid']
    assert mid.avg_current == pytest.approx(1e-3)
    assert mid._current_source == 'db'
    loaded_pins = [b for b in mid.pins if b.avg_current is not None]
    assert loaded_pins and all(b._current_source == 'db' for b in loaded_pins)


def test_partial_net_stays_none(pdk, tmp_dir, tmp_path):
    """Port nets stored pin rows only — the aggregate must round-trip as None,
    never be recomputed from pins (the under-sizing guard)."""
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    fresh.load_net_currents('tt', db_path=db_path)
    a = fresh.net_dict['A']
    assert a.avg_current is None                          # NOT 1e-3, NOT 0.0
    assert any(b.avg_current is not None for b in a.pins) # pins restored


def test_missing_version_is_loud(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    changed = _build_dut(pdk, tmp_dir, r2=4700.0)         # different structure
    with pytest.raises(RuntimeError, match='[Nn]o stored version'):
        changed.load_net_currents('tt', db_path=db_path)


def test_missing_corner_lists_available(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path, corner='tt')
    fresh = _build_dut(pdk, tmp_dir)
    with pytest.raises(RuntimeError, match='tt'):
        fresh.load_net_currents('ss', db_path=db_path)


def test_stale_row_hash_is_loud(pdk, tmp_dir, tmp_path):
    """Set-level hash guards against mutated version metadata.

    Schema v2 moved structure_hash from each net_currents row to its
    net_current_sets parent (design §6.3): staleness is now a per-SET check,
    not per-row — every row of one set necessarily shares the same parent.
    """
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE net_current_sets SET structure_hash='TAMPERED'")
    conn.commit(); conn.close()
    with pytest.raises(RuntimeError, match='stale'):
        fresh.load_net_currents('tt', db_path=db_path)


def test_stale_hash_does_not_affect_other_corners_set(pdk, tmp_dir, tmp_path):
    """The stale check is per-SET: tampering the 'tt' set's structure_hash
    must not affect an unrelated 'ss' set's own current_set_id row.

    Schema v1 had this granularity at the row level (structure_hash was
    stored per net_currents row); v2 moves it to the shared
    net_current_sets parent, so the smallest unit that can go stale
    independently is now one complete selected set, not one row within it —
    this is the v2-equivalent regression for the retired
    ``test_stale_single_row_hash_is_loud``.
    """
    import sqlite3
    db_path = str(tmp_path / 't.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    tb.store_net_currents(_fake_report(), corner='tt')
    tb.store_net_currents(_fake_report(), corner='ss')
    tb.close()

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE net_current_sets SET structure_hash='TAMPERED' "
        "WHERE corner = 'tt'"
    )
    conn.commit(); conn.close()

    fresh_tt = _build_dut(pdk, tmp_dir)
    with pytest.raises(RuntimeError, match='stale'):
        fresh_tt.load_net_currents('tt', db_path=db_path)

    fresh_ss = _build_dut(pdk, tmp_dir)
    n = fresh_ss.load_net_currents('ss', db_path=db_path)
    assert n > 0
    assert fresh_ss.net_dict['mid'].avg_current == pytest.approx(1e-3)


def test_unknown_row_scheme_fails_before_mutation(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE net_current_sets SET structure_scheme = 'silk.structure/v999'
        WHERE current_set_id = (
            SELECT current_set_id FROM net_current_sets
            ORDER BY current_set_id LIMIT 1
        )
        """
    )
    conn.commit(); conn.close()

    with pytest.raises(
        UnsupportedFingerprintSchemeError, match='silk.structure/v999'
    ):
        fresh.load_net_currents('tt', db_path=db_path)
    assert fresh.net_dict['mid'].avg_current is None


def test_unknown_version_scheme_fails_before_mutation(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    import sqlite3
    conn = sqlite3.connect(db_path)
    version_id = conn.execute(
        "SELECT circuit_version_id FROM net_current_sets LIMIT 1"
    ).fetchone()[0]
    conn.execute(
        """
        UPDATE circuit_versions SET structure_scheme = 'silk.structure/v999'
        WHERE version_id = ?
        """,
        (version_id,),
    )
    conn.commit(); conn.close()

    with pytest.raises(
        UnsupportedFingerprintSchemeError, match='silk.structure/v999'
    ):
        fresh.load_net_currents('tt', db_path=db_path)
    assert fresh.net_dict['mid'].avg_current is None


def test_spec_only_successor_loads_ancestor_and_query_reports_origin(
    pdk, tmp_dir, tmp_path
):
    db_path = str(tmp_path / 'spec-successor.db')
    source_dut = _build_dut(pdk, tmp_dir)
    source_dut.declare_spec(
        'Gain target', 'gain_dB', '>=', 60.0, key='gain'
    )
    tb = _build_tb(pdk, tmp_dir, db_path=db_path, dut=source_dut)
    _annotate_definition(tb)
    source_id = tb.store_net_currents(
        _fake_report(), corner='tt'
    )['dut_blk']
    tb.close()

    successor = _build_dut(pdk, tmp_dir, db_path=db_path)
    successor.declare_spec(
        'Gain target', 'gain_dB', '>=', 65.0, key='gain'
    )
    successor_id = successor.create_circuit_version(
        parent_version_id=source_id
    )

    with Database(db_path) as reopened:
        origin_id, rows = reopened.get_net_current_set(successor_id, 'tt')
    assert origin_id == source_id
    assert rows and all(row['origin_version_id'] == source_id for row in rows)

    assert successor.load_net_currents('tt') > 0
    assert successor.net_dict['mid'].avg_current == pytest.approx(1e-3)
    successor.close()


def test_bound_version_current_set_precedes_compatible_ancestor(
    pdk, tmp_dir, tmp_path
):
    db_path = str(tmp_path / 'local-precedence.db')
    source_dut = _build_dut(pdk, tmp_dir)
    source_dut.declare_spec(
        'Gain target', 'gain_dB', '>=', 60.0, key='gain'
    )
    source_tb = _build_tb(pdk, tmp_dir, db_path=db_path, dut=source_dut)
    _annotate_definition(source_tb)
    source_id = source_tb.store_net_currents(
        _fake_report(), corner='tt'
    )['dut_blk']
    source_tb.close()

    successor = _build_dut(pdk, tmp_dir, db_path=db_path)
    successor.declare_spec(
        'Gain target', 'gain_dB', '>=', 65.0, key='gain'
    )
    successor_id = successor.create_circuit_version(
        parent_version_id=source_id
    )
    successor.close()

    local_dut = _build_dut(pdk, tmp_dir)
    local_dut.declare_spec(
        'Gain target', 'gain_dB', '>=', 65.0, key='gain'
    )
    local_tb = _build_tb(pdk, tmp_dir, db_path=db_path, dut=local_dut)
    local_definition = _annotate_definition(local_tb)
    local_definition.net_dict['mid'].avg_current = 2e-3
    assert local_tb.store_net_currents(
        _fake_report(results_file='/tmp/local.raw'), corner='tt'
    )['dut_blk'] == successor_id
    local_tb.close()

    with Database(db_path) as reopened:
        origin_id, rows = reopened.get_net_current_set(successor_id, 'tt')

    aggregate = next(
        row for row in rows
        if row['net_name'] == 'mid' and row['pin_id'] == ''
    )
    assert origin_id == successor_id
    assert aggregate['origin_version_id'] == successor_id
    assert aggregate['avg_a'] == pytest.approx(2e-3)


def test_structural_successor_rejects_ancestor_before_mutation(
    pdk, tmp_dir, tmp_path
):
    db_path = str(tmp_path / 'structure-successor.db')
    tb = _build_tb(pdk, tmp_dir, db_path=db_path)
    _annotate_definition(tb)
    source_id = tb.store_net_currents(
        _fake_report(), corner='tt'
    )['dut_blk']
    tb.close()

    successor = _build_dut(pdk, tmp_dir, r2=4700.0, db_path=db_path)
    successor.create_circuit_version(parent_version_id=source_id)

    with pytest.raises(EvidenceMismatchError, match='structure'):
        successor.load_net_currents('tt')
    assert successor.net_dict['mid'].avg_current is None
    assert all(
        binding.avg_current is None
        for binding in successor.net_dict['mid'].pins
    )
    successor.close()


def test_precedence_user_and_spice_need_force(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    mid = fresh.net_dict['mid']
    mid.avg_current = 42.0
    mid._current_source = 'user'
    fresh.load_net_currents('tt', db_path=db_path)
    assert mid.avg_current == 42.0                        # preserved
    fresh.load_net_currents('tt', db_path=db_path, force=True)
    assert mid.avg_current == pytest.approx(1e-3)         # forced


def test_annotate_would_overwrite_db(pdk, tmp_dir, tmp_path):
    """Documented direction: 'db' is overwritable by anything fresher —
    load itself may overwrite previously-loaded 'db' values without force."""
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    fresh.load_net_currents('tt', db_path=db_path)
    mid = fresh.net_dict['mid']
    mid.avg_current = 42.0
    mid._current_source = 'db'                             # simulate a stale 'db' value
    fresh.load_net_currents('tt', db_path=db_path)          # no force needed
    assert mid.avg_current == pytest.approx(1e-3)            # 'db' was overwritten, not skipped


def test_reload_clears_db_rows_removed_from_complete_set(
    pdk, tmp_dir, tmp_path
):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    fresh.load_net_currents('tt', db_path=db_path)
    mid = fresh.net_dict['mid']
    removed_pin = next(
        binding for binding in mid.pins
        if binding._current_source == 'db'
    )
    removed_pin_id = f'{removed_pin.instance}.{removed_pin.port}'

    # A value without a corresponding DB row is not DB-owned and must survive,
    # even when force allows matching stored rows to replace user evidence.
    user_net = fresh.net_dict['B']
    user_net.avg_current = 42.0
    user_net.rms_current = 43.0
    user_net.peak_current = 44.0
    user_net._current_source = 'user'

    tracker = fresh._ensure_tracker()
    tracker.db.conn.execute(
        """
        DELETE FROM net_currents
        WHERE net_name = 'mid' AND (pin_id = '' OR pin_id = ?)
          AND current_set_id IN (
              SELECT current_set_id FROM net_current_sets WHERE corner = 'tt'
          )
        """,
        (removed_pin_id,),
    )
    tracker.db.conn.commit()

    fresh.load_net_currents('tt', force=True)

    assert (mid.avg_current, mid.rms_current, mid.peak_current) == (
        None, None, None
    )
    assert mid._current_source is None
    assert (
        removed_pin.avg_current,
        removed_pin.rms_current,
        removed_pin.peak_current,
        removed_pin.current_vector,
        removed_pin._current_source,
    ) == (None, None, None, None, None)
    assert (
        user_net.avg_current,
        user_net.rms_current,
        user_net.peak_current,
        user_net._current_source,
    ) == (42.0, 43.0, 44.0, 'user')


def test_load_db_path_mismatch_raises(pdk, tmp_dir, tmp_path):
    bound = str(tmp_path / 'bound.db')
    other = str(tmp_path / 'other.db')
    tb = _build_tb(pdk, tmp_dir, db_path=bound)
    with pytest.raises(ValueError) as excinfo:
        tb.load_net_currents('tt', db_path=other)
    assert bound in str(excinfo.value) and other in str(excinfo.value)


def test_load_db_path_same_path_is_noop(pdk, tmp_dir, tmp_path):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    fresh.load_net_currents('tt', db_path=db_path)
    n = fresh.load_net_currents('tt', db_path=db_path)      # same path again: no-op guard, not a raise
    assert n > 0


def test_bad_pin_id_row_is_atomic(pdk, tmp_dir, tmp_path):
    """Pre-validation prevents partial application: a hash-valid row whose
    pin_id matches no live binding must raise before ANY row is applied —
    a midway-invalid row leaves the module untouched."""
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    import sqlite3
    import uuid
    conn = sqlite3.connect(db_path)
    current_set_id = conn.execute(
        "SELECT current_set_id FROM net_current_sets WHERE corner = 'tt' LIMIT 1"
    ).fetchone()[0]
    conn.execute(
        """INSERT INTO net_currents
           (current_id, current_set_id, net_name, pin_id, avg_a, rms_a,
            peak_a, source, current_vector)
           VALUES (?, ?, 'mid', 'bogus.pin', 1e-3, 1e-3, 1e-3, 'spice', NULL)""",
        (str(uuid.uuid4()), current_set_id))
    conn.commit(); conn.close()
    with pytest.raises(RuntimeError, match='matches no binding'):
        fresh.load_net_currents('tt', db_path=db_path)
    assert fresh.net_dict['mid'].avg_current is None       # nothing applied


# Schema v2 declares net_currents.avg_a REAL NOT NULL (design §6.3), so a
# raw SQL UPDATE can no longer plant a NULL there to reach the application
# check — the DB itself now rejects it (see
# test_null_avg_a_rejected in test_net_currents_schema.py). Only the
# wrong-type ('corrupt') case remains reachable through storage.
@pytest.mark.parametrize('corrupt_value', ['corrupt'])
def test_corrupt_numeric_row_is_rejected_before_live_mutation(
    pdk, tmp_dir, tmp_path, corrupt_value
):
    db_path = _stored_db(pdk, tmp_dir, tmp_path)
    fresh = _build_dut(pdk, tmp_dir)
    fresh.load_net_currents('tt', db_path=db_path)
    before = fresh.net_dict['mid'].avg_current

    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE net_currents SET avg_a = ?
        WHERE net_name = 'mid' AND pin_id = ''
          AND current_set_id IN (
              SELECT current_set_id FROM net_current_sets WHERE corner = 'tt'
          )
        """,
        (corrupt_value,),
    )
    conn.commit(); conn.close()

    with pytest.raises(EvidenceMismatchError, match='avg_a.*real number'):
        fresh.load_net_currents('tt')
    assert fresh.net_dict['mid'].avg_current == before
