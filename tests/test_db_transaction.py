"""Database.transaction() — atomic multi-write grouping (review-followup G).

Per-call writes auto-commit by default; inside a transaction() block they defer
to a single commit (or roll back on error), so a mid-sequence failure can't
leave partial state.
"""
import sqlite3

import pytest

from silk.db.circuit_db import Database
from silk.exceptions import DatabaseError, LockedVersionError
from tests.db_snapshot_helpers import (
    create_building_snapshot,
    create_sealed_snapshot,
    structure_with_net,
)


def _allow_sql(*_):
    """Restore SQL access; set_authorizer(None) requires Python 3.11+."""
    return sqlite3.SQLITE_OK


def _db(tmp_path):
    db = Database(str(tmp_path / "t.db"))
    cid = create_building_snapshot(db, name="amp", version="1.0")
    return db, cid


def _device_count(db, cid):
    return db.conn.execute(
        "SELECT COUNT(*) FROM devices WHERE circuit_version_id=?", (cid,)
    ).fetchone()[0]


def test_transaction_commits_all_on_success(tmp_path):
    db, cid = _db(tmp_path)
    with db.transaction():
        assert db.conn.in_transaction is True
        db.add_device(cid, "M1", "nmos")
        db.add_device(cid, "M2", "pmos")
    assert _device_count(db, cid) == 2
    assert db._txn_depth == 0


def test_transaction_rolls_back_on_error(tmp_path):
    """A failure mid-sequence leaves NO partial state."""
    db, cid = _db(tmp_path)
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.add_device(cid, "M1", "nmos")
            db.add_device(cid, "M2", "pmos")
            raise RuntimeError("boom after both writes")
    assert _device_count(db, cid) == 0      # both writes rolled back
    assert db._txn_depth == 0               # depth restored after rollback


def test_foreign_keys_remain_enabled_after_reopen(tmp_path):
    path = tmp_path / "foreign-keys.db"
    with Database(path) as db:
        assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    with Database(path) as reopened:
        assert reopened.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_individual_call_is_committed(tmp_path):
    """Outside a transaction, each write auto-commits — visible to a second
    connection (proves it's a real commit, not just same-connection visibility)."""
    path = str(tmp_path / "t.db")
    db = Database(path)
    cid = create_building_snapshot(db, name="amp", version="1.0")
    db.add_device(cid, "M1", "nmos")
    other = sqlite3.connect(path)
    n = other.execute(
        "SELECT COUNT(*) FROM devices WHERE circuit_version_id=?", (cid,)
    ).fetchone()[0]
    other.close()
    assert n == 1


def test_nested_transaction_commits_once(tmp_path):
    """Re-entrant: the inner block does not commit; the outer one does."""
    db, cid = _db(tmp_path)
    with db.transaction():
        db.add_device(cid, "M1", "nmos")
        with db.transaction():
            db.add_device(cid, "M2", "pmos")
            assert db._txn_depth == 2
        assert db._txn_depth == 1          # inner exit did not commit/reset
    assert db._txn_depth == 0
    assert _device_count(db, cid) == 2


def test_latest_by_version_is_numeric_not_lexicographic(tmp_path):
    """'1.9' sorted ABOVE '1.10' before 2026-08-23: the SQL ORDER BY on the
    TEXT version column is lexicographic, so the method whose whole job is
    'highest version' returned the wrong row whenever a version crossed a
    digit-count boundary (1.9/1.10, 9.x/10.x). Ordering now happens in
    Python with per-segment numeric comparison."""
    from silk.db.circuit_db import DesignStage
    db = Database(str(tmp_path / "v.db"))
    create_sealed_snapshot(
        db, name="vertest", version="1.9",
        structure=structure_with_net("v19"),
    )
    create_sealed_snapshot(
        db, name="vertest", version="1.10",
        structure=structure_with_net("v110"),
    )
    create_sealed_snapshot(
        db, name="vertest", version="1.2",
        structure=structure_with_net("v12"),
    )
    assert db.get_latest_circuit_by_version("vertest")["version"] == "1.10"


class TestIntegrityHardening:
    """Foreign keys and snapshot write locks are enforced.

    Duplicate circuit-name and version labels are allowed; those labels do
    not define snapshot identity.
    """

    def _db(self, tmp_path):
        from silk.db.circuit_db import DesignStage
        db = Database(str(tmp_path / "h.db"))
        vid = create_building_snapshot(db, name="amp", version="1.0")
        return db, vid

    def test_foreign_keys_enforced(self, tmp_path):
        db, _ = self._db(tmp_path)
        assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(DatabaseError, match="does not exist"):
            db.add_device("no-such-version-id", "M1", "nmos")

    def test_locked_circuit_rejects_writes_and_unlock_restores(self, tmp_path):
        db, vid = self._db(tmp_path)
        db.add_device(vid, "M1", "nmos")            # unlocked: fine
        db.lock_circuit("amp", "1.0")
        for write in (lambda: db.add_device(vid, "M2", "nmos"),
                      lambda: db.add_net(vid, "vout"),
                      lambda: db.add_simulation_result(vid, "gain", value=40.0),
                      ):
            with pytest.raises(LockedVersionError, match="locked"):
                write()
        db.unlock_circuit("amp", "1.0")
        db.add_device(vid, "M2", "nmos")            # symmetric escape hatch

    def test_duplicate_human_version_label_is_allowed(self, tmp_path):
        db, _ = self._db(tmp_path)
        second = create_building_snapshot(
            db,
            name="amp",
            version="1.0",
            structure=structure_with_net("changed"),
        )
        assert second is not None
        with pytest.raises(DatabaseError, match="ambiguous"):
            db.get_version("amp", "1.0")


def test_commit_actions_wait_for_outer_commit_and_do_not_replay(tmp_path):
    db, cid = _db(tmp_path)
    other = sqlite3.connect(db.db_path)
    published = []

    def publish(label):
        # A different connection proves the SQL commit precedes the action.
        assert other.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 2
        assert not db.conn.in_transaction
        assert db._txn_depth == 0
        published.append(label)

    try:
        with db.transaction():
            db.add_device(cid, "M1", "nmos")
            db._after_commit(lambda: publish("first"))
            with db.transaction():
                db.add_device(cid, "M2", "pmos")
                db._after_commit(lambda: publish("second"))
            assert published == []
            assert other.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 0
        assert published == ["first", "second"]
        with db.transaction():
            pass
        assert published == ["first", "second"]
    finally:
        other.close()
        db.close()


def test_action_error_after_commit_does_not_roll_back_or_replay(tmp_path):
    db, cid = _db(tmp_path)
    statements = []
    published = []
    db.conn.set_trace_callback(statements.append)

    def fail_after_commit():
        published.append("failed")
        raise RuntimeError("live recording failed")

    try:
        with pytest.raises(RuntimeError, match="live recording failed"):
            with db.transaction():
                db.add_device(cid, "M1", "nmos")
                db._after_commit(fail_after_commit)
                db._after_commit(lambda: published.append("abandoned"))

        assert "COMMIT" in statements
        assert "ROLLBACK" not in statements
        assert not db.conn.in_transaction
        assert db._txn_depth == 0
        other = sqlite3.connect(db.db_path)
        try:
            assert other.execute("SELECT COUNT(*) FROM devices").fetchone()[0] == 1
        finally:
            other.close()
        with db.transaction():
            db.add_device(cid, "M2", "pmos")
            db._after_commit(lambda: published.append("recovered"))
        assert published == ["failed", "recovered"]
        assert _device_count(db, cid) == 2
    finally:
        db.close()


# ---------------------------------------------------------------------------
# TASK-171: every nested transaction() scope is its own rollback boundary.
#
# Before the fix, transaction() nested by depth counting alone: an inner scope
# had no boundary, so a caller that CAUGHT the inner failure inside an
# enclosing transaction left the failed save's earlier rows sitting in the
# outer transaction, to be committed at depth zero.
# ---------------------------------------------------------------------------


def _committed_devices(db):
    """Device names visible to a SECOND connection -- the real commit
    boundary, not this connection's uncommitted view."""
    other = sqlite3.connect(db.db_path)
    try:
        return sorted(
            row[0] for row in other.execute("SELECT name FROM devices").fetchall()
        )
    finally:
        other.close()


def test_nested_rollback_undoes_its_scope_and_its_deeper_scopes(tmp_path):
    """AC1/AC5: an exception escaping a nested scope undoes everything written
    under it -- including a deeper scope that already exited successfully --
    and nothing outside it. No nested scope commits early."""
    db, cid = _db(tmp_path)
    try:
        with db.transaction():
            db.add_device(cid, "before", "nmos")
            with pytest.raises(RuntimeError, match="inner boom"):
                with db.transaction():
                    db.add_device(cid, "inner", "nmos")
                    with db.transaction():
                        db.add_device(cid, "deeper", "pmos")
                    # The deeper scope exited cleanly; it is still undone.
                    assert _device_count(db, cid) == 3
                    raise RuntimeError("inner boom")
            assert _device_count(db, cid) == 1          # same-connection view
            assert _committed_devices(db) == []          # nothing committed yet
            db.add_device(cid, "after", "pmos")
        assert _committed_devices(db) == ["after", "before"]
        assert db._txn_depth == 0
    finally:
        db.close()


def test_caught_store_results_failure_commits_no_rows_from_that_batch(tmp_path):
    """AC2: the task's public reproducer. A late failure inside store_results()
    whose TypeError the outer caller catches commits none of that batch's rows;
    sibling saves before and after it still commit."""
    from silk import Module
    from silk.db import CircuitTracker

    folder = tmp_path / "audit"
    folder.mkdir()
    module = Module(
        None, str(folder), "audit.cir", pdk_details=None, module_name="audit",
    )
    path = folder / "audit.db"
    tracker = CircuitTracker(module, path)
    try:
        tracker.create_circuit_version()
        with tracker.db.transaction():
            tracker.store_results({"before": (1.0, "V")})
            with pytest.raises(TypeError, match="must be a real number"):
                tracker.store_results({
                    "gain": (60.0, "dB"),
                    "bandwidth": (object(), "Hz"),
                })
            tracker.store_results({"after": (2.0, "V")})

        reader = sqlite3.connect(path)
        try:
            committed = sorted(
                reader.execute(
                    "SELECT parameter FROM simulation_results"
                ).fetchall()
            )
        finally:
            reader.close()
        assert committed == [("after",), ("before",)]
    finally:
        tracker.close()


def test_standalone_failed_multi_write_save_still_leaves_no_rows(tmp_path):
    """AC4 control A: store_results() on its own was already atomic and
    stays atomic -- the fix must not regress the non-nested path."""
    from silk import Module
    from silk.db import CircuitTracker

    folder = tmp_path / "standalone"
    folder.mkdir()
    module = Module(
        None, str(folder), "audit.cir", pdk_details=None, module_name="audit",
    )
    path = folder / "audit.db"
    tracker = CircuitTracker(module, path)
    try:
        tracker.create_circuit_version()
        with pytest.raises(TypeError, match="must be a real number"):
            tracker.store_results({
                "gain": (60.0, "dB"),
                "bandwidth": (object(), "Hz"),
            })
        tracker.store_results({"solo": (3.0, "V")})     # commits before return
        reader = sqlite3.connect(path)
        try:
            assert reader.execute(
                "SELECT parameter FROM simulation_results"
            ).fetchall() == [("solo",)]
        finally:
            reader.close()
    finally:
        tracker.close()


def test_exception_escaping_the_outermost_scope_rolls_back_the_group(tmp_path):
    """AC4 control B: successful nested scopes give no protection to a group
    whose own exception escapes -- the whole group is still discarded."""
    db, cid = _db(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="outer boom"):
            with db.transaction():
                with db.transaction():
                    db.add_device(cid, "M1", "nmos")
                with db.transaction():
                    db.add_device(cid, "M2", "pmos")
                raise RuntimeError("outer boom")
        assert _device_count(db, cid) == 0
        assert _committed_devices(db) == []
        assert db._txn_depth == 0
        assert not db.conn.in_transaction
    finally:
        db.close()


def test_rolled_back_scope_discards_only_its_own_pending_actions(tmp_path):
    """AC6/AC7: three nesting levels. The intermediate scope fails after a
    deeper scope already registered an action and returned successfully; the
    outer scope catches and continues. Only the surviving actions run, once,
    in their original order."""
    db, cid = _db(tmp_path)
    published = []
    try:
        with db.transaction():
            db.add_device(cid, "kept-early", "nmos")
            db._after_commit(lambda: published.append("outer-early"))
            with pytest.raises(RuntimeError, match="intermediate boom"):
                with db.transaction():                      # intermediate
                    db.add_device(cid, "dropped-mid", "pmos")
                    db._after_commit(lambda: published.append("intermediate"))
                    with db.transaction():                  # inner, succeeds
                        db.add_device(cid, "dropped-inner", "nmos")
                        db._after_commit(lambda: published.append("inner"))
                    raise RuntimeError("intermediate boom")
            assert published == []
            db.add_device(cid, "kept-late", "pmos")
            db._after_commit(lambda: published.append("outer-late"))
        assert published == ["outer-early", "outer-late"]
        assert _committed_devices(db) == ["kept-early", "kept-late"]
        # Nothing from the rolled-back scope leaks into a later transaction.
        with db.transaction():
            db.add_device(cid, "later", "nmos")
        assert published == ["outer-early", "outer-late"]
    finally:
        db.close()


def test_nested_rollback_leaves_the_connection_usable(tmp_path):
    """AC9: normal nested rollback restores usable transaction bookkeeping --
    proven by a later successful operation on the same connection."""
    db, cid = _db(tmp_path)
    try:
        with pytest.raises(RuntimeError):
            with db.transaction():
                db.add_device(cid, "M1", "nmos")
                raise RuntimeError("boom")
        assert db._txn_depth == 0
        assert not db.conn.in_transaction
        with db.transaction():
            with db.transaction():
                db.add_device(cid, "M2", "pmos")
        assert _committed_devices(db) == ["M2"]
        db.add_device(cid, "M3", "nmos")                 # auto-commit path
        assert _committed_devices(db) == ["M2", "M3"]
    finally:
        db.close()


def test_denied_commit_rolls_back_everything_and_publishes_nothing(tmp_path):
    """AC9: a deterministic SQL commit failure at the outermost scope rolls
    back all SQL -- including successfully released nested scopes -- and
    neither publishes nor retains the pending actions for replay."""
    db, cid = _db(tmp_path)
    published = []

    def deny_commit(operation, argument, *_):
        if operation == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            with db.transaction():
                with db.transaction():
                    db.add_device(cid, "M1", "nmos")
                    db._after_commit(lambda: published.append("inner"))
                db._after_commit(lambda: published.append("outer"))
                db.conn.set_authorizer(deny_commit)
        db.conn.set_authorizer(_allow_sql)
        assert published == []
        assert _device_count(db, cid) == 0
        assert _committed_devices(db) == []
        assert db._txn_depth == 0
        assert not db.conn.in_transaction
        with db.transaction():
            db.add_device(cid, "M2", "pmos")
        assert published == []                       # no replay
        assert _committed_devices(db) == ["M2"]
    finally:
        db.conn.set_authorizer(_allow_sql)
        db.close()


def test_action_error_after_commit_is_unchanged_by_nesting(tmp_path):
    """AC10: post-commit action errors keep their existing behaviour under the
    new boundary -- committed SQL is not reported as rolled back, the error
    propagates, and abandoned actions never replay."""
    db, cid = _db(tmp_path)
    statements = []
    published = []
    db.conn.set_trace_callback(statements.append)

    def fail_after_commit():
        published.append("failed")
        raise RuntimeError("live recording failed")

    try:
        with pytest.raises(RuntimeError, match="live recording failed"):
            with db.transaction():
                with db.transaction():
                    db.add_device(cid, "M1", "nmos")
                    db._after_commit(fail_after_commit)
                db._after_commit(lambda: published.append("abandoned"))

        assert "COMMIT" in statements
        assert "ROLLBACK" not in statements
        assert _committed_devices(db) == ["M1"]
        assert db._txn_depth == 0
        with db.transaction():
            db.add_device(cid, "M2", "pmos")
            db._after_commit(lambda: published.append("recovered"))
        assert published == ["failed", "recovered"]
    finally:
        db.conn.set_trace_callback(None)
        db.close()


# ---------------------------------------------------------------------------
# TASK-171 review follow-up: SQLite failing *during savepoint handling* must
# not become a way for a failed scope to reach the outer commit. Both cases
# below need SQLite itself to refuse the savepoint statement, which the
# authorizer injects deterministically.
# ---------------------------------------------------------------------------


def _deny_savepoint(kind):
    """Authorizer that refuses one savepoint verb: BEGIN, RELEASE or ROLLBACK."""
    def authorize(operation, verb, *_):
        if operation == sqlite3.SQLITE_SAVEPOINT and verb == kind:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return authorize


def test_refused_release_cleans_up_like_a_failure_inside_the_scope(tmp_path):
    """A nested scope whose RELEASE SQLite refuses did not merge into the
    enclosing transaction. Catching that error must not let the scope's rows
    or its deferred live updates commit at the outer commit anyway."""
    db, cid = _db(tmp_path)
    published = []
    try:
        with db.transaction():
            db.add_device(cid, "outer", "nmos")
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                with db.transaction():
                    db.add_device(cid, "inner", "pmos")
                    db._after_commit(lambda: published.append("inner"))
                    db.conn.set_authorizer(_deny_savepoint("RELEASE"))
            db.conn.set_authorizer(_allow_sql)
            assert _device_count(db, cid) == 1        # inner row already gone
            db.add_device(cid, "after", "nmos")
        assert _committed_devices(db) == ["after", "outer"]
        assert published == []
        assert db._txn_depth == 0
    finally:
        db.conn.set_authorizer(_allow_sql)
        db.close()


def test_unconfirmed_savepoint_rollback_refuses_to_commit_the_group(tmp_path):
    """If SQLite will not confirm the nested ROLLBACK TO, the scope's rows may
    still be live. The outermost context must discard the whole group and say
    so, rather than commit rows nothing proved were undone."""
    db, cid = _db(tmp_path)
    published = []
    try:
        with pytest.raises(DatabaseError, match="could not be confirmed") as caught:
            with db.transaction():
                db.add_device(cid, "outer", "nmos")
                db._after_commit(lambda: published.append("outer"))
                with pytest.raises(RuntimeError, match="inner boom"):
                    with db.transaction():
                        db.add_device(cid, "inner", "pmos")
                        db.conn.set_authorizer(_deny_savepoint("ROLLBACK"))
                        raise RuntimeError("inner boom")
                db.conn.set_authorizer(_allow_sql)
        assert isinstance(caught.value.__cause__, sqlite3.Error)
        assert _committed_devices(db) == []      # including the outer's own row
        assert published == []
        assert db._txn_depth == 0
        assert not db.conn.in_transaction

        # The poison does not outlive the group that raised it.
        db.conn.set_authorizer(_allow_sql)
        with db.transaction():
            db.add_device(cid, "later", "nmos")
            db._after_commit(lambda: published.append("later"))
        assert _committed_devices(db) == ["later"]
        assert published == ["later"]
    finally:
        db.conn.set_authorizer(_allow_sql)
        db.close()


def test_confirmed_rollback_with_a_refused_release_still_commits_siblings(
    tmp_path
):
    """ROLLBACK TO succeeded, so the scope's rows ARE gone; only the savepoint
    stack entry survives. Names are never reused, so the group stays
    committable -- this must not be poisoned like an unconfirmed rollback."""
    db, cid = _db(tmp_path)
    try:
        with db.transaction():
            db.add_device(cid, "outer", "nmos")
            with pytest.raises(RuntimeError, match="inner boom"):
                with db.transaction():
                    db.add_device(cid, "inner", "pmos")
                    db.conn.set_authorizer(_deny_savepoint("RELEASE"))
                    raise RuntimeError("inner boom")
            db.conn.set_authorizer(_allow_sql)
            assert _device_count(db, cid) == 1
            db.add_device(cid, "after", "pmos")
        assert _committed_devices(db) == ["after", "outer"]
    finally:
        db.conn.set_authorizer(_allow_sql)
        db.close()


def _abort_the_transaction(db, cid):
    """Interrupt a write mid-statement. SQLite rolls the WHOLE explicit
    transaction back automatically (an interrupted INSERT/UPDATE/DELETE inside
    one cannot be undone in part), so ``conn.in_transaction`` goes False while
    silk still believes it is inside a group."""
    calls = {"n": 0}

    def handler():
        calls["n"] += 1
        return 1 if calls["n"] > 20 else 0

    db.conn.set_progress_handler(handler, 20)
    try:
        db.conn.execute(
            "WITH RECURSIVE c(i) AS ("
            "  SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i < 200000"
            ") INSERT INTO devices(device_id, circuit_version_id, name, type) "
            "SELECT 'bulk'||i, ?, 'bulk'||i, 'nmos' FROM c",
            (cid,),
        )
    except sqlite3.Error:
        pass
    finally:
        db.conn.set_progress_handler(None, 0)
    assert not db.conn.in_transaction         # the abort actually happened
    assert db._txn_depth > 0                  # ... and silk has not noticed


def test_nested_scope_is_refused_when_sqlite_discarded_the_transaction(
    tmp_path
):
    """A SAVEPOINT issued outside a transaction OPENS one, and releasing that
    outermost savepoint COMMITS it. So a nested save entered after SQLite
    discarded the enclosing transaction used to commit on the spot, before the
    outer block exited, and survive the rollback that block then reported."""
    db, cid = _db(tmp_path)
    published = []
    try:
        with pytest.raises(DatabaseError, match="enclosing SQL transaction is gone"):
            with db.transaction():
                db.add_device(cid, "outer", "nmos")
                _abort_the_transaction(db, cid)
                with db.transaction():
                    db.add_device(cid, "early", "nmos")
                    db._after_commit(lambda: published.append("early"))
        assert _committed_devices(db) == []       # nothing slipped out early
        assert published == []
        assert db._txn_depth == 0
        assert db._unsafe_group is None
        with db.transaction():                    # connection still usable
            db.add_device(cid, "later", "nmos")
        assert _committed_devices(db) == ["later"]
    finally:
        db.conn.set_progress_handler(None, 0)
        db.close()


def test_caught_nested_entry_refusal_still_fails_the_group(tmp_path):
    """Catching the refusal must not buy the group a clean exit: whatever it
    wrote before SQLite's abort is already gone, so reporting success would be
    a lie."""
    db, cid = _db(tmp_path)
    try:
        with pytest.raises(DatabaseError, match="was not committed") as caught:
            with db.transaction():
                db.add_device(cid, "outer", "nmos")
                _abort_the_transaction(db, cid)
                with pytest.raises(DatabaseError, match="Cannot open a nested"):
                    with db.transaction():
                        db.add_device(cid, "early", "nmos")
                # Swallowed -- the outer block exits normally from here.
        assert isinstance(caught.value.__cause__, DatabaseError)
        assert _committed_devices(db) == []
        assert db._txn_depth == 0
        assert db._unsafe_group is None
    finally:
        db.conn.set_progress_handler(None, 0)
        db.close()


def test_nested_scope_is_refused_on_an_already_poisoned_group(tmp_path):
    """The other half of the guard: once a nested rollback went unconfirmed,
    no further nested scope may be opened on that group either."""
    db, cid = _db(tmp_path)
    try:
        with pytest.raises(DatabaseError, match="rollback could not be confirmed"):
            with db.transaction():
                db.add_device(cid, "outer", "nmos")
                with pytest.raises(RuntimeError, match="inner boom"):
                    with db.transaction():
                        db.add_device(cid, "poisoning", "pmos")
                        db.conn.set_authorizer(_deny_savepoint("ROLLBACK"))
                        raise RuntimeError("inner boom")
                db.conn.set_authorizer(_allow_sql)
                with pytest.raises(DatabaseError, match="Cannot open a nested"):
                    with db.transaction():
                        db.add_device(cid, "after-poison", "nmos")
        assert _committed_devices(db) == []
        assert db._txn_depth == 0
        assert db._unsafe_group is None
    finally:
        db.conn.set_authorizer(_allow_sql)
        db.close()


# TASK-511: SQL transaction scopes own restoration of live state too.
def test_undo_actions_require_transaction(tmp_path):
    db, _ = _db(tmp_path)
    try:
        with pytest.raises(RuntimeError, match='transaction'):
            db._after_rollback(lambda: None)
    finally:
        db.close()


def test_undo_actions_follow_nested_scope_ownership(tmp_path):
    db, _ = _db(tmp_path)
    restored = []
    try:
        with pytest.raises(ValueError):
            with db.transaction():
                db._after_rollback(lambda: restored.append('outer'))
                with pytest.raises(ValueError):
                    with db.transaction():
                        db._after_rollback(lambda: restored.append('inner'))
                        with db.transaction():
                            db._after_rollback(lambda: restored.append('deep'))
                        raise ValueError('inner')
                assert restored == ['deep', 'inner']
                with db.transaction():
                    db._after_rollback(lambda: restored.append('sibling'))
                raise ValueError('outer')
        assert restored == ['deep', 'inner', 'sibling', 'outer']
        with db.transaction():
            pass
        assert restored == ['deep', 'inner', 'sibling', 'outer']
    finally:
        db.close()


def test_undo_actions_discarded_before_after_commit_callbacks(tmp_path):
    db, _ = _db(tmp_path)
    restored = []

    def fail():
        raise ValueError('after commit')

    try:
        with pytest.raises(ValueError, match='after commit'):
            with db.transaction():
                db._after_rollback(lambda: restored.append('committed'))
                db._after_commit(fail)
        with pytest.raises(ValueError):
            with db.transaction():
                raise ValueError('later rollback')
        assert restored == []
    finally:
        db.close()


@pytest.mark.parametrize('failure', ['commit', 'release', 'rollback', 'poison'])
def test_undo_actions_restore_on_sql_failure(tmp_path, failure):
    db, cid = _db(tmp_path)
    restored, published = [], []

    def deny(op, arg, *_):
        blocked = ((failure == 'commit' and op == sqlite3.SQLITE_TRANSACTION and arg == 'COMMIT')
                   or (failure == 'release' and op == sqlite3.SQLITE_SAVEPOINT and arg == 'RELEASE')
                   or (failure == 'rollback' and op == sqlite3.SQLITE_TRANSACTION and arg == 'ROLLBACK')
                   or (failure == 'poison' and op == sqlite3.SQLITE_SAVEPOINT and arg == 'ROLLBACK'))
        return sqlite3.SQLITE_DENY if blocked else sqlite3.SQLITE_OK

    try:
        with pytest.raises((sqlite3.DatabaseError, DatabaseError)):
            with db.transaction():
                db._after_rollback(lambda: restored.append('outer'))
                db._after_commit(lambda: published.append('outer'))
                if failure in ('poison', 'release'):
                    try:
                        with db.transaction():
                            db.add_device(cid, 'temporary', 'nmos')
                            db._after_rollback(lambda: restored.append('inner'))
                            db.conn.set_authorizer(deny)
                            if failure == 'poison':
                                raise ValueError('inner')
                    except ValueError:
                        assert failure == 'poison'
                        assert restored == ['inner']
                else:
                    db.add_device(cid, 'temporary', 'nmos')
                    db.conn.set_authorizer(deny)
                    if failure == 'rollback':
                        raise ValueError('abort')
        assert restored == (['inner', 'outer'] if failure in ('poison', 'release') else ['outer'])
        assert published == []
        db.conn.set_authorizer(_allow_sql)
        db.conn.rollback()
        assert _device_count(db, cid) == 0
        before = restored[:]
        with db.transaction():
            db.add_device(cid, 'after', 'nmos')
        assert restored == before
        assert _device_count(db, cid) == 1
    finally:
        db.conn.set_authorizer(_allow_sql)
        db.conn.rollback()
        db.close()
