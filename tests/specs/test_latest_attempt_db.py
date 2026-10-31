"""Existing SQL transactions hydrate/promote attempts without inventing runs."""
import pytest
from silk.specs import Measurement
from tests.db.test_profile_scoped_evidence import _build, _context, _report, _record_original
from tests.specs.test_latest_attempt import failure


def test_restore_deduplicates_without_clearing_newer_failure(pdk, tmp_path):
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    _record_original(spec)
    mod.store_report(_report((spec, 65.), context=_context()))
    mod.restore_spec_measurements()
    count = len(spec.attempt_history())
    failure(spec, timestamp=500.)
    mod.restore_spec_measurements()
    mod.restore_spec_measurements()
    assert len(spec.attempt_history()) == count + 1
    row = mod.reconcile_specs().outcomes[0]
    assert row.status == 'FAIL'
    assert row.last_successful_attempt.timestamp == 123.


def test_profile_promotion_preserves_attempt_identity_and_rollback(pdk, tmp_path):
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = _record_original(spec)
    first, = spec.attempt_history()
    with pytest.raises(RuntimeError, match='rollback'):
        with mod._tracker.db.transaction():
            mod.store_report(_report((spec, 65.), context=_context()))
            raise RuntimeError('rollback')
    assert spec.attempt_history() == (first,)
    from silk.circuit.test_placement import evidence_index
    assert evidence_index(mod).projection_associations == {}
    mod.store_report(_report((spec, 65.), context=_context()))
    promoted, = spec.attempt_history()
    assert promoted.attempt_id == first.attempt_id
    assert promoted.timestamp == first.timestamp
    assert promoted == first
    assert promoted.measurement.tool_name == 'ngspice'
    assert promoted.measurement.profile_id is None
    slots = [m for m in spec.measurements.values() if m.profile_id is not None]
    assert len(slots) == 1
    row = mod._tracker.db.conn.execute('SELECT profile_id FROM spec_measurements').fetchone()
    assert slots[0].profile_id == row[0]
    assert first.measurement.profile_id is None
    assert original.value == 65.


def test_conflicting_promotion_is_rejected_before_commit(pdk, tmp_path):
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = _record_original(spec)
    # A mutable live value can agree with a report but disagree with the
    # already-published immutable observation. Storage must reject atomically.
    original.value = 99.
    before = list(mod._tracker.db.conn.iterdump())
    with pytest.raises(ValueError, match='conflict'):
        mod.store_report(_report((spec, 99.), context=_context()))
    assert list(mod._tracker.db.conn.iterdump()) == before
    assert spec.attempt_history()[0].measurement.value == 65.


def test_grouped_projections_restore_without_rewriting_observation(pdk, tmp_path, monkeypatch):
    from silk.circuit.test_placement import evidence_index
    from tests.db.test_characterization_readback import _current_row
    mod, gain, settling = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = _record_original(gain)
    first, = gain.attempt_history()
    a, b = _context(parameters={'amplitude': .1}), _context(parameters={'amplitude': .2})
    db = mod._tracker.db
    published = []
    for spec in (gain, settling):
        publish = spec._publish_measurement
        def observe(m, attempt, scope, publish=publish, key=spec.key):
            assert not db.conn.in_transaction
            publish(m, attempt, scope)
            published.append((key, m.tool_name))
        monkeypatch.setattr(spec, '_publish_measurement', observe)
    with db.transaction():
        for context, value, tool in ((a, 1.5e-6, 'first'), (b, 1.2e-6, 'second'), (a, 1e-6, 'last')):
            mod.store_report(_report((gain, 65.), (settling, value), context=context), tool_name=tool)
            assert gain.attempt_history() == (first,)
            assert published == []
    assert published == [('gain', 'first'), ('settling', 'first'), ('gain', 'second'),
                         ('settling', 'second'), ('gain', 'last'), ('settling', 'last')]
    assert db.conn.execute('SELECT count(*) FROM spec_measurements').fetchone()[0] == 6
    assert gain.attempt_history() == (first,)
    assert original.tool_name == 'ngspice'
    assert {m.tool_name for m in gain.measurements.values() if m.profile_id} == {'second', 'last'}
    version = db._snapshot_row(mod._tracker.circuit_id)
    for context in (a, b):
        db.replace_net_current_set(mod._tracker.circuit_id, corner='tt', temperature=27., voltage=1.8,
            structure_scheme=version['structure_scheme'], structure_hash=version['structure_hash'],
            context=context, rows=[_current_row(.001)], analysis='transient')
    failure(gain, timestamp=500.)
    history = gain.attempt_history()
    for _ in range(2):
        mod.restore_spec_measurements()
        for context in (a, b):
            mod.restore_characterization_evidence(context=context, force=True)
        assert gain.attempt_history() == history
        row = mod.reconcile_specs().outcomes[0]
        assert row.status == 'FAIL'
        assert row.last_successful_attempt == first
        assert 'first' not in str(row.latest_attempt.as_dict())
    assert len(evidence_index(mod).projection_associations) == 6


def test_staged_enrichment_conflict_rolls_back_and_does_not_poison_sibling(pdk, tmp_path):
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = Measurement('spice', 65., corner='tt', timestamp=123., temperature=27., voltage=1.8)
    spec.record_measurement(original)
    first, = spec.attempt_history()
    db = mod._tracker.db
    with db.transaction():
        with pytest.raises(RuntimeError, match='discard'):
            with db.transaction():
                original.tool_name = 'discarded'
                mod.store_report(_report((spec, 65.), context=_context()))
                raise RuntimeError('discard')
        original.tool_name = 'accepted'
        mod.store_report(_report((spec, 65.), context=_context()))
        original.tool_name = 'conflicting'
        with pytest.raises(ValueError, match='conflict'):
            mod.store_report(_report((spec, 65.), context=_context()))
        assert spec.attempt_history() == (first,)
    assert db.conn.execute('SELECT count(*) FROM spec_measurements').fetchone()[0] == 1
    assert spec.attempt_history()[0].measurement.tool_name == 'accepted'
    assert len(spec.attempt_history()) == 1
    from silk.circuit.test_placement import evidence_index
    assert len(evidence_index(mod).projection_associations) == 1


def test_fresh_hydration_does_not_invent_attempt_associations(pdk, tmp_path):
    from silk.circuit.test_placement import evidence_index
    path = tmp_path / 'hydrate.db'
    mod, spec, _ = _build(pdk, tmp_path, path)
    mod.create_circuit_version()
    _record_original(spec)
    mod.store_report(_report((spec, 65.), context=_context()), tool_name='storage tool')
    failure(spec)
    mod.close()
    rebuilt, spec, _ = _build(pdk, tmp_path, path)
    rebuilt.create_circuit_version()
    rebuilt.restore_spec_measurements()
    assert spec.latest().value == 65.
    assert spec.attempt_history() == ()
    assert evidence_index(rebuilt).projection_associations == {}


def test_projection_link_is_not_transportable_evidence(pdk, tmp_path):
    import pickle
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    _record_original(spec)
    mod.store_report(_report((spec, 65.), context=_context()))
    projected = next(m for m in spec.measurements.values() if m.profile_id)
    assert hasattr(projected, '_projection_observation')
    restored = pickle.loads(pickle.dumps(projected))
    assert not hasattr(restored, '_projection_observation')
    assert restored.profile_id == projected.profile_id
    assert restored.value == projected.value


def test_projection_restore_accepts_sql_numeric_normalization(pdk, tmp_path):
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = _record_original(spec)
    original.timestamp, original.temperature = 123, 27
    first, = spec.attempt_history()
    mod.store_report(_report((spec, 65.), context=_context()))
    mod.restore_spec_measurements()
    assert spec.attempt_history() == (first,)
    assert spec.latest().timestamp == 123.


def _publish_interleaved(spec, original, method):
    from silk.specs import SpecAttempt
    if method == 'record_measurement':
        spec.record_measurement(original)
    else:
        spec.record_attempt(SpecAttempt(
            attempt_id=original._attempt_id, source=original.source,
            timestamp=original.timestamp,
            measurement=Measurement('spice', original.value, passes=True,
                                    timestamp=original.timestamp,
                                    tool_name=original.tool_name,
                                    sim_type=original.sim_type)))


@pytest.mark.parametrize('scoped', [False, True])
@pytest.mark.parametrize('method', ['record_measurement', 'record_attempt'])
def test_live_publication_conflict_refuses_outer_commit(pdk, tmp_path, monkeypatch, scoped, method):
    from silk.circuit.test_placement import evidence_index
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = Measurement('spice', 65., corner='tt', timestamp=123.,
                           temperature=27., voltage=1.8)
    spec.record_measurement(original)
    db = mod._tracker.db
    before_sql = list(db.conn.iterdump())
    commits = []
    commit = db._maybe_commit
    def observe_commit():
        if db._txn_depth == 0:
            commits.append(True)
        commit()
    monkeypatch.setattr(db, '_maybe_commit', observe_commit)
    with pytest.raises(ValueError, match='conflict'):
        with db.transaction():
            original.tool_name = 'staged'
            mod.store_report(_report((spec, 65.), context=_context() if scoped else None))
            original.tool_name = 'later-live'
            _publish_interleaved(spec, original, method)
            live_history = spec.attempt_history()
    assert commits == []  # Reject before SQL commit, never in its callbacks.
    assert list(db.conn.iterdump()) == before_sql
    assert spec.attempt_history() == live_history
    assert spec.latest_attempt().measurement.tool_name == 'later-live'
    assert len(spec.measurements) == 1
    assert evidence_index(mod).projection_associations == {}
    assert db._pending_spec_attempts == {}


@pytest.mark.parametrize('scoped', [False, True])
@pytest.mark.parametrize('method', ['record_measurement', 'record_attempt'])
def test_live_publication_enrichment_preserves_all_known_facts(pdk, tmp_path, scoped, method):
    from silk.circuit.test_placement import evidence_index
    mod, spec, _ = _build(pdk, tmp_path, ':memory:')
    mod.create_circuit_version()
    original = Measurement('spice', 65., corner='tt', timestamp=123.,
                           temperature=27., voltage=1.8)
    spec.record_measurement(original)
    first = spec.latest_attempt()
    with mod._tracker.db.transaction():
        original.tool_name = 'staged'
        mod.store_report(_report((spec, 65.), context=_context() if scoped else None))
        original.tool_name = None
        original.sim_type = 'ac'
        _publish_interleaved(spec, original, method)
    assert spec.latest_attempt().measurement.sim_type == 'ac'
    association, = evidence_index(mod).projection_associations.values()
    assert association.observation.measurement.tool_name == 'staged'
    assert association.observation.measurement.sim_type == 'ac'
    assert first.measurement.tool_name is None and first.measurement.sim_type is None
    mod.restore_spec_measurements()
    latest, = spec.attempt_history()
    assert latest.attempt_id == first.attempt_id and latest.timestamp == first.timestamp
    assert latest.measurement.tool_name == 'staged'
    assert latest.measurement.sim_type == 'ac'
