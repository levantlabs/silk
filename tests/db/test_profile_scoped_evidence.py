"""Profile-scoped metrics and spec measurements (TASK-6 / DB-02-EVIDENCE).

``store_report()`` threads ``report.characterization_context`` into
the same atomic transaction that writes metric rows, compatibility-status
rows, and first-class spec measurements, deriving/reusing the profile via
``store_characterization_profile``. Bare ``store_results()``/``store_result()``
stay explicitly unscoped -- no ``conditions`` dict can fabricate a profile.
Queries and restore accept a 3-state ``profile_id`` filter, and live
``Spec.measurements`` becomes profile-aware so two profiles observed at the
same source@corner coexist instead of overwriting.
"""

import sqlite3
from contextlib import nullcontext

import pytest

from silk.circuit.modules import Module
from silk.circuit.spec_report import SpecReport
from silk.db import UNSCOPED_PROFILE
from silk.db.context import CharacterizationContext
from silk.exceptions import EvidenceMismatchError, EvidenceNotFoundError
from silk.specs import Measurement


def _allow_sql(*_):
    """Restore SQL access; set_authorizer(None) requires Python 3.11+."""
    return sqlite3.SQLITE_OK


def _build(pdk, tmp_path, db_path, *, module_name="profile_evidence"):
    module = Module(
        "PDK_skywater130",
        str(tmp_path / module_name),
        f"{module_name}.cir",
        pdk_details=pdk,
        module_name=module_name,
        db_path=str(db_path),
    )
    module.add_net("out")
    gain = module.declare_spec(
        "Gain", "gain_dB", ">=", 60.0, units="dB", key="gain"
    )
    settling = module.declare_spec(
        "Settling time", "settling_s", "<=", 2e-6, units="s", key="settling"
    )
    return module, gain, settling


def _context(**overrides):
    kwargs = {
        "owner_path": "$",
        "test_name": "characterize",
        "parameters": {},
        "corner": "tt",
        "temperature": 27.0,
        "voltage": 1.8,
    }
    kwargs.update(overrides)
    return CharacterizationContext(**kwargs)


def _report(*observations, context=None, source="spice"):
    metrics = {spec.metric: value for spec, value in observations}
    report = SpecReport(
        "characterize", "profile_evidence", metrics,
        owner_path="$",
        parameters=dict(context.parameters) if context is not None else {},
        characterization_context=context,
    )
    for spec, _value in observations:
        report.check(
            spec.name, spec.metric, spec.op, spec.target,
            spec_key=spec.key, owner_path="$",
        )
        # Fixtures that supply original observations carry them explicitly.
        # Other hand-built reports deliberately have unknown observation facts.
        original = spec.measurements.get(f'{source}@tt')
        if original is not None and original.profile_id is None and original.value == _value:
            from silk.specs.attempt import prepare_measurement
            report.results[-1]._observation = (
                '$', spec.key, spec.metric, spec.op, spec.target,
                prepare_measurement(spec, original)[0])
    return report


def _record_original(gain):
    measurement = Measurement(
        source="spice", value=65.0, passes=True, corner="tt",
        timestamp=123.0, sim_run_id="run-1", notes="original observation",
        temperature=27.0, voltage=1.8, sim_type="ac", tool_name="ngspice",
    )
    gain.record_measurement(measurement)
    return measurement


def _evidence_counts(conn):
    return tuple(
        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "characterization_profiles", "simulation_results", "spec_measurements"
        )
    )


def _assert_measurements_unchanged(spec, before):
    assert spec.measurements.keys() == before.keys()
    for key, measurement in before.items():
        assert spec.measurements[key] is measurement


# ---------------------------------------------------------------------------
# AC1: store_report threads profile identity + exact PVT atomically
# ---------------------------------------------------------------------------

def test_scoped_store_report_writes_matching_profile_id_everywhere(
    pdk, tmp_path
):
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "scoped.db")
    version_id = module.create_circuit_version()
    context = _context(parameters={"vin_amplitude_v": 0.1})

    result = module.store_report(_report((gain, 65.0), context=context))
    assert result == {"metrics_stored": 1, "specs_stored": 1, "skipped": {}}

    db = module._tracker.db
    profiles = db.conn.execute(
        "SELECT profile_id, owner_path, test_name FROM characterization_profiles"
    ).fetchall()
    assert len(profiles) == 1
    profile_id = profiles[0]["profile_id"]
    assert profiles[0]["owner_path"] == "$"
    assert profiles[0]["test_name"] == "characterize"

    sim_rows = db.conn.execute(
        "SELECT profile_id, corner, temperature, voltage "
        "FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchall()
    assert len(sim_rows) == 2  # the metric row + the spec_ compatibility row
    for row in sim_rows:
        assert row["profile_id"] == profile_id
        assert (row["corner"], row["temperature"], row["voltage"]) == (
            "tt", 27.0, 1.8,
        )

    measurement_row = db.conn.execute(
        "SELECT profile_id FROM spec_measurements"
    ).fetchone()
    assert measurement_row["profile_id"] == profile_id

    # Re-storing an equivalent scoped report reuses the SAME profile row
    # (idempotent derivation) rather than minting a second one.
    module.store_report(_report((gain, 66.0), context=context))
    assert db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 1
    module.close()


def test_unscoped_store_report_is_unaffected(pdk, tmp_path):
    """A bare (no characterization_context) report keeps writing
    profile_id=NULL, exactly as before DB-02 (AC3/AC6)."""
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "unscoped.db")
    module.create_circuit_version()
    module.store_report(_report((gain, 65.0)), corner="tt")

    db = module._tracker.db
    assert db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0
    profile_ids = {
        row["profile_id"]
        for row in db.conn.execute(
            "SELECT profile_id FROM simulation_results"
        ).fetchall()
    }
    assert profile_ids == {None}
    assert db.conn.execute(
        "SELECT profile_id FROM spec_measurements"
    ).fetchone()["profile_id"] is None
    module.close()


def test_scoped_store_report_rejects_conflicting_explicit_pvt(pdk, tmp_path):
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "conflict.db")
    version_id = module.create_circuit_version()
    context = _context()

    with pytest.raises(EvidenceMismatchError, match="conflicts"):
        module.store_report(
            _report((gain, 65.0), context=context), temperature=85.0
        )

    db = module._tracker.db
    assert db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0] == 0
    module.close()


@pytest.mark.parametrize("grouped", [False, True])
def test_scoped_store_report_rolls_back_profile_on_injected_failure(
    pdk, tmp_path, monkeypatch, grouped
):
    """AC1: the profile row itself is part of the all-or-nothing write --
    a mid-report failure must not leave a newly-minted profile behind."""
    module, gain, settling = _build(
        pdk, tmp_path, tmp_path / "rollback.db"
    )
    version_id = module.create_circuit_version()
    original = _record_original(gain)
    context = _context()
    report = _report((gain, 65.0), (settling, 1e-6), context=context)

    db = module._tracker.db
    real_insert = db.insert_spec_measurement
    calls = 0

    def fail_before_second_measurement(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected measurement failure")
        return real_insert(*args, **kwargs)

    monkeypatch.setattr(
        db, "insert_spec_measurement", fail_before_second_measurement
    )
    with pytest.raises(RuntimeError, match="injected measurement failure"):
        with db.transaction() if grouped else nullcontext():
            module.store_report(report)

    assert db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 0
    _assert_measurements_unchanged(gain, {"spice@tt": original})
    assert settling.measurements == {}
    monkeypatch.setattr(db, "insert_spec_measurement", real_insert)
    module.store_report(
        _report((gain, 65.0), context=_context(parameters={"recovery": True}))
    )
    assert len(gain.measurements) == 2
    assert gain.measurements["spice@tt"] is original
    assert settling.measurements == {}
    module.close()


# ---------------------------------------------------------------------------
# AC6: bare store_results()/store_result() remain explicitly unscoped
# ---------------------------------------------------------------------------

def test_store_results_never_fabricates_a_profile_from_conditions(
    pdk, tmp_path
):
    module, _gain, _settling = _build(
        pdk, tmp_path, tmp_path / "bare-results.db"
    )
    module.create_circuit_version()
    # A conditions dict that looks exactly like a profile descriptor must
    # still never create or imply one.
    module.store_results(
        {"gain_dB": (65.0, "dB")},
        corner="tt", temperature=27.0, voltage=1.8,
        conditions={
            "process_corner": "tt",
            "temperature_c": 27.0,
            "supply_voltage_v": 1.8,
            "test_name": "characterize",
        },
    )
    module.store_result("bandwidth_MHz", 12.0, "MHz", corner="tt")

    db = module._tracker.db
    assert db.conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0
    profile_ids = {
        row["profile_id"]
        for row in db.conn.execute(
            "SELECT profile_id FROM simulation_results"
        ).fetchall()
    }
    assert profile_ids == {None}
    module.close()


def test_public_store_results_signature_has_no_profile_kwarg():
    """Bare store_results() gets no new profile_id knob at all (design
    SS4.3: 'Bare store_results() remains unscoped')."""
    import inspect
    signature = inspect.signature(Module.store_results)
    assert "profile_id" not in signature.parameters


# ---------------------------------------------------------------------------
# AC2/AC5: two profiles at one corner -- query/restore filter + live coexist
# ---------------------------------------------------------------------------

def test_two_profiles_same_corner_coexist_and_are_queryable(pdk, tmp_path):
    module, gain, _settling = _build(
        pdk, tmp_path, tmp_path / "two-profiles.db"
    )
    module.create_circuit_version()
    context_a = _context(parameters={"vin_amplitude_v": 0.1})
    context_b = _context(parameters={"vin_amplitude_v": 0.2})

    module.store_report(_report((gain, 61.0), context=context_a))
    module.store_report(_report((gain, 64.0), context=context_b))

    db = module._tracker.db
    profile_rows = db.conn.execute(
        "SELECT profile_id FROM characterization_profiles ORDER BY created_at"
    ).fetchall()
    assert len(profile_rows) == 2
    profile_a_id, profile_b_id = (row["profile_id"] for row in profile_rows)
    assert profile_a_id != profile_b_id

    # Live Spec.measurements: both profiles coexist -- neither overwrote the
    # other even though both are 'spice@tt' by source/corner (AC5).
    values_by_profile = {
        m.profile_id: m.value for m in gain.measurements.values()
        if m.profile_id is not None
    }
    assert values_by_profile == {profile_a_id: 61.0, profile_b_id: 64.0}

    # query_results: no filter returns both; an exact profile filters to one.
    all_records = module.query_results(parameter="gain_dB")
    assert {r.profile_id for r in all_records} == {profile_a_id, profile_b_id}
    only_a = module.query_results(parameter="gain_dB", profile_id=profile_a_id)
    assert [r.value for r in only_a] == [61.0]
    # Both stored gain_dB rows are scoped -- an unscoped-only filter finds
    # nothing (the existing "not found" reporting, unaffected by DB-02).
    with pytest.raises(EvidenceNotFoundError):
        module.query_results(parameter="gain_dB", profile_id=UNSCOPED_PROFILE)

    # get_stored_specs: both measurements show up, clearly marked by
    # profile_id; a profile filter narrows to one.
    unfiltered = next(
        r for r in module.get_stored_specs() if r.spec_key == "gain"
    )
    assert {m.profile_id for m in unfiltered.measurements} == {
        profile_a_id, profile_b_id,
    }
    filtered = next(
        r for r in module.get_stored_specs(profile_id=profile_b_id)
        if r.spec_key == "gain"
    )
    assert [m.value for m in filtered.measurements] == [64.0]
    module.close()


def test_restore_rebuilds_both_profiles_without_overwrite(pdk, tmp_path):
    db_path = tmp_path / "restore-two-profiles.db"
    module, gain, _settling = _build(pdk, tmp_path, db_path)
    version_id = module.create_circuit_version()
    context_a = _context(parameters={"vin_amplitude_v": 0.1})
    context_b = _context(parameters={"vin_amplitude_v": 0.2})
    module.store_report(_report((gain, 61.0), context=context_a))
    module.store_report(_report((gain, 64.0), context=context_b))
    db = module._tracker.db
    profile_a_id, profile_b_id = (
        row["profile_id"]
        for row in db.conn.execute(
            "SELECT profile_id FROM characterization_profiles "
            "ORDER BY created_at"
        ).fetchall()
    )
    module.close()

    rebuilt, rebuilt_gain, _settling = _build(pdk, tmp_path, db_path)
    assert rebuilt.create_circuit_version() == version_id

    assert rebuilt.restore_spec_measurements() == 2
    values_by_profile = {
        m.profile_id: m.value for m in rebuilt_gain.measurements.values()
    }
    assert values_by_profile == {profile_a_id: 61.0, profile_b_id: 64.0}

    rebuilt_gain.measurements.clear()
    assert rebuilt.restore_spec_measurements(profile_id=profile_a_id) == 1
    assert [
        m.value for m in rebuilt_gain.measurements.values()
    ] == [61.0]

    with pytest.raises(EvidenceNotFoundError):
        rebuilt.restore_spec_measurements(profile_id=UNSCOPED_PROFILE)
    rebuilt.close()


# ---------------------------------------------------------------------------
# AC3: DB-01 restore validation / promotion-origin behavior untouched
# ---------------------------------------------------------------------------

def test_profile_scoping_does_not_disturb_unscoped_restore(pdk, tmp_path):
    """The pre-existing unscoped restore-and-select-latest-per-slot behavior
    (test_spec_restore.py) is unaffected by an interleaved scoped report to
    the same spec: an UNSCOPED_PROFILE-filtered restore still finds exactly
    the unscoped observation, keyed exactly as before DB-02."""
    db_path = tmp_path / "mixed.db"
    module, gain, _settling = _build(pdk, tmp_path, db_path)
    version_id = module.create_circuit_version()
    module.store_report(_report((gain, 61.0)), corner="tt")
    module.store_report(
        _report((gain, 64.0), context=_context(parameters={"x": 1}))
    )
    module.close()

    rebuilt, rebuilt_gain, _settling = _build(pdk, tmp_path, db_path)
    assert rebuilt.create_circuit_version() == version_id
    assert rebuilt.restore_spec_measurements(
        profile_id=UNSCOPED_PROFILE
    ) == 1
    assert set(rebuilt_gain.measurements) == {"spice@tt"}
    assert rebuilt_gain.measurements["spice@tt"].value == 61.0
    assert rebuilt_gain.measurements["spice@tt"].profile_id is None
    rebuilt.close()


# ---------------------------------------------------------------------------
# Low-level Database validation (add_simulation_result / insert_spec_measurement)
# ---------------------------------------------------------------------------

def test_add_simulation_result_rejects_unknown_profile_id(pdk, tmp_path):
    module, _gain, _settling = _build(
        pdk, tmp_path, tmp_path / "bad-profile.db"
    )
    version_id = module.create_circuit_version()
    db = module._tracker.db

    with pytest.raises(EvidenceMismatchError, match="not a stored"):
        db.add_simulation_result(
            version_id, "gain", value=1.0,
            corner="tt", temperature=27.0, voltage=1.8,
            profile_id="does-not-exist",
        )
    module.close()


def test_add_simulation_result_profile_requires_complete_pvt(pdk, tmp_path):
    module, _gain, _settling = _build(
        pdk, tmp_path, tmp_path / "incomplete-pvt.db"
    )
    version_id = module.create_circuit_version()
    db = module._tracker.db
    profile = db.store_characterization_profile(version_id, _context())

    with pytest.raises(ValueError, match="corner, temperature, and voltage"):
        db.add_simulation_result(
            version_id, "gain", value=1.0,
            corner="tt", temperature=27.0, voltage=None,
            profile_id=profile.profile_id,
        )
    module.close()


def test_query_results_profile_filter_rejects_wrong_type(pdk, tmp_path):
    module, _gain, _settling = _build(
        pdk, tmp_path, tmp_path / "bad-filter-type.db"
    )
    module.create_circuit_version()
    with pytest.raises(TypeError, match="profile_id filter"):
        module.query_results(parameter="gain_dB", profile_id=123)
    module.close()


def test_insert_spec_measurement_rejects_unknown_profile_id(pdk, tmp_path):
    module, gain, _settling = _build(
        pdk, tmp_path, tmp_path / "bad-measurement-profile.db"
    )
    module.create_circuit_version()
    bindings = module._tracker._validated_live_spec_bindings()
    _live_spec, declaration = bindings[("$", "gain")]
    db = module._tracker.db

    with pytest.raises(EvidenceMismatchError, match="not a stored"):
        db.insert_spec_measurement(
            "meas-1", declaration.spec_id, "spice", 65.0,
            corner="tt", temperature=27.0, voltage=1.8,
            profile_id="does-not-exist",
        )
    module.close()


def test_grouped_reports_publish_in_order_only_after_commit(
    pdk, tmp_path, monkeypatch
):
    module, gain, settling = _build(pdk, tmp_path, tmp_path / "group.db")
    module.create_circuit_version()
    original = _record_original(gain)
    db = module._tracker.db
    context_a = _context(parameters={"amplitude": 0.1})
    context_b = _context(parameters={"amplitude": 0.2})
    other = sqlite3.connect(db.db_path)
    published = []

    def observe_recording(spec):
        record = spec._publish_measurement

        def publish(measurement, attempt, scope):
            assert not db.conn.in_transaction
            assert _evidence_counts(other) == (2, 12, 6)
            record(measurement, attempt, scope)
            published.append((spec.key, measurement))

        monkeypatch.setattr(spec, "_publish_measurement", publish)

    observe_recording(gain)
    observe_recording(settling)
    try:
        with db.transaction():
            for context, value, tool in (
                (context_a, 1.5e-6, "first"),
                (context_b, 1.2e-6, "second"),
                (context_a, 1e-6, "last"),
            ):
                with db.transaction():
                    result = module.store_report(
                        _report((gain, 65.0), (settling, value), context=context),
                        tool_name=tool,
                    )
                assert result == {
                    "metrics_stored": 2, "specs_stored": 2, "skipped": {}
                }
                assert published == []
                _assert_measurements_unchanged(gain, {"spice@tt": original})
                assert settling.measurements == {}
                assert _evidence_counts(other) == (0, 0, 0)
            assert _evidence_counts(db.conn) == (2, 12, 6)

        assert [(key, m.value, m.tool_name) for key, m in published] == [
            ("gain", 65.0, "first"), ("settling", 1.5e-6, "first"),
            ("gain", 65.0, "second"), ("settling", 1.2e-6, "second"),
            ("gain", 65.0, "last"), ("settling", 1e-6, "last"),
        ]
        rows = db.conn.execute(
            "SELECT s.spec_key, m.* FROM spec_measurements m "
            "JOIN specs s ON s.spec_id = m.spec_id ORDER BY m.rowid"
        ).fetchall()
        for (key, measurement), row in zip(published, rows, strict=True):
            assert key == row["spec_key"]
            for field in (
                "source", "value", "passes", "corner", "timestamp", "sim_run_id",
                "notes", "temperature", "voltage", "sim_type", "tool_name",
                "profile_id",
            ):
                expected = 0.0 if field == 'timestamp' and row[field] is None else row[field]
                assert getattr(measurement, field) == expected
            assert measurement.origin_version_id is None
        first, second, last = (published[i][1] for i in (0, 2, 4))
        assert first.profile_id == last.profile_id != second.profile_id
        assert last.timestamp == original.timestamp
        assert last.sim_run_id == original.sim_run_id
        assert last.notes == original.notes
        assert last.sim_type == original.sim_type
        for spec, offset in ((gain, 0), (settling, 1)):
            assert spec.measurements[
                ("profile", first.profile_id, "spice", "tt", 27.0, 1.8)
            ] is published[4 + offset][1]
            assert spec.measurements[
                ("profile", second.profile_id, "spice", "tt", 27.0, 1.8)
            ] is published[2 + offset][1]
        assert gain.measurements["spice@tt"] is original
        with db.transaction():
            pass
        assert len(published) == 6
    finally:
        other.close()
        module.close()


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("failure", ["outer_exception", "sql_commit"])
def test_aborted_group_preserves_live_measurements_and_discards_actions(
    pdk, tmp_path, monkeypatch, preexisting, failure
):
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "abort-group.db")
    module.create_circuit_version()
    original = _record_original(gain)
    db = module._tracker.db
    context_a = _context(parameters={"amplitude": 0.1})
    context_b = _context(parameters={"amplitude": 0.2})
    if preexisting:
        module.store_report(_report((gain, 65.0), context=context_a))
    before = dict(gain.measurements)
    counts_before = _evidence_counts(db.conn)
    published = []
    record = gain._publish_measurement

    def publish(measurement, attempt, scope):
        record(measurement, attempt, scope)
        published.append(measurement)

    def deny_commit(operation, argument, *_):
        if operation == sqlite3.SQLITE_TRANSACTION and argument == "COMMIT":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    monkeypatch.setattr(gain, "_publish_measurement", publish)
    error = RuntimeError if failure == "outer_exception" else sqlite3.DatabaseError
    message = (
        "later grouped operation failed"
        if failure == "outer_exception" else "not authorized"
    )
    try:
        with pytest.raises(error, match=message):
            with db.transaction():
                for context, tool in (
                    (context_a, "first"), (context_b, "second"), (context_a, "last")
                ):
                    module.store_report(
                        _report((gain, 65.0), context=context), tool_name=tool
                    )
                    assert published == []
                    _assert_measurements_unchanged(gain, before)
                if failure == "outer_exception":
                    raise RuntimeError("later grouped operation failed")
                # Reject the real SQLite COMMIT; ROLLBACK remains allowed.
                db.conn.set_authorizer(deny_commit)
        db.conn.set_authorizer(_allow_sql)
        assert published == []
        _assert_measurements_unchanged(gain, before)
        assert gain.measurements["spice@tt"] is original
        assert _evidence_counts(db.conn) == counts_before
        assert db._txn_depth == 0
        assert not db.conn.in_transaction
        other = sqlite3.connect(db.db_path)
        try:
            assert _evidence_counts(other) == counts_before
        finally:
            other.close()

        # Reuse the same connection without replaying the abandoned group.
        module.store_report(
            _report((gain, 65.0), context=_context(parameters={"recovery": True})),
            tool_name="recovered",
        )
        assert [m.tool_name for m in published] == ["recovered"]
        for key, measurement in before.items():
            assert gain.measurements[key] is measurement
        assert len(gain.measurements) == len(before) + 1
        assert _evidence_counts(db.conn) == tuple(
            count + added for count, added in zip(counts_before, (1, 2, 1))
        )
    finally:
        db.conn.set_authorizer(_allow_sql)
        module.close()


@pytest.mark.parametrize("scoped", [False, True])
def test_standalone_report_publishes_before_return_only_when_scoped(
    pdk, tmp_path, scoped
):
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "standalone.db")
    module.create_circuit_version()
    original = _record_original(gain)
    try:
        result = module.store_report(
            _report((gain, 65.0), context=_context() if scoped else None)
        )
        assert result == {"metrics_stored": 1, "specs_stored": 1, "skipped": {}}
        assert gain.measurements["spice@tt"] is original
        other = sqlite3.connect(module._tracker.db.db_path)
        try:
            assert _evidence_counts(other) == (int(scoped), 2, 1)
            profile_id = other.execute(
                "SELECT profile_id FROM spec_measurements"
            ).fetchone()[0]
        finally:
            other.close()
        if scoped:
            assert len(gain.measurements) == 2
            assert gain.measurements[
                ("profile", profile_id, "spice", "tt", 27.0, 1.8)
            ].value == 65.0
        else:
            assert profile_id is None
            _assert_measurements_unchanged(gain, {"spice@tt": original})
    finally:
        module.close()


def test_caught_report_failure_registers_no_partial_live_updates(
    pdk, tmp_path, monkeypatch
):
    module, gain, settling = _build(pdk, tmp_path, tmp_path / "caught.db")
    module.create_circuit_version()
    original = _record_original(gain)
    db = module._tracker.db
    real_insert = db.insert_spec_measurement
    calls = 0

    def fail_before_second_measurement(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected measurement failure")
        return real_insert(*args, **kwargs)

    try:
        with db.transaction():
            with monkeypatch.context() as patch:
                patch.setattr(
                    db, "insert_spec_measurement", fail_before_second_measurement
                )
                with pytest.raises(RuntimeError, match="injected measurement failure"):
                    module.store_report(
                        _report((gain, 65.0), (settling, 1e-6), context=_context())
                    )
            _assert_measurements_unchanged(gain, {"spice@tt": original})
            module.store_report(
                _report(
                    (gain, 65.0),
                    context=_context(parameters={"successful": True}),
                ),
                tool_name="successful",
            )
        # Each transaction()-wrapped save is its own rollback boundary
        # (TASK-171): the caught failure takes its SQL rows with it as well as
        # its live writes, and only the successful sibling report commits.
        scoped = [m for m in gain.measurements.values() if m.profile_id is not None]
        assert [m.tool_name for m in scoped] == ["successful"]
        assert gain.measurements["spice@tt"] is original
        assert settling.measurements == {}
        other = sqlite3.connect(db.db_path)
        try:
            assert _evidence_counts(other) == (1, 2, 1)
            assert other.execute(
                "SELECT tool_name FROM spec_measurements"
            ).fetchall() == [("successful",)]
        finally:
            other.close()
    finally:
        module.close()


# ---------------------------------------------------------------------------
# TASK-171: a caught store_report() failure inside an enclosing transaction
# leaves none of that report's SQL evidence, not just none of its live updates.
# ---------------------------------------------------------------------------


def _deny_spec_measurement_insert(operation, table, *_):
    """Authorizer that fails the LAST write store_report() makes -- the
    spec-measurement insert -- after its profile/metric/status rows are in."""
    if operation == sqlite3.SQLITE_INSERT and table == "spec_measurements":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def test_caught_report_sql_failure_leaves_none_of_its_evidence(pdk, tmp_path):
    """AC3/AC8: a real, late SQLite write failure in store_report() that the
    outer caller catches inside an enclosing transaction commits no profile,
    metric, compatibility-status or spec-measurement row from that report.
    Pre-existing evidence and successful sibling saves are untouched."""
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "caught-sql.db")
    module.create_circuit_version()
    original = _record_original(gain)
    db = module._tracker.db

    # Pre-existing, already-committed evidence from an earlier standalone save.
    module.store_report(
        _report((gain, 65.0), context=_context(parameters={"preexisting": True})),
        tool_name="preexisting",
    )
    counts_before = _evidence_counts(db.conn)
    assert counts_before == (1, 2, 1)
    before = dict(gain.measurements)

    try:
        with db.transaction():
            db.conn.set_authorizer(_deny_spec_measurement_insert)
            with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                module.store_report(
                    _report(
                        (gain, 65.0),
                        context=_context(parameters={"doomed": True}),
                    ),
                    tool_name="doomed",
                )
            db.conn.set_authorizer(_allow_sql)
            # Same-connection view already shows the failed report undone.
            assert _evidence_counts(db.conn) == counts_before
            module.store_report(
                _report(
                    (gain, 65.0),
                    context=_context(parameters={"survivor": True}),
                ),
                tool_name="survivor",
            )

        other = sqlite3.connect(db.db_path)
        try:
            assert _evidence_counts(other) == tuple(
                count + added for count, added in zip(counts_before, (1, 2, 1))
            )
            tools = sorted(
                row[0] for row in other.execute(
                    "SELECT tool_name FROM spec_measurements"
                ).fetchall()
            )
        finally:
            other.close()
        assert tools == ["preexisting", "survivor"]
        # Pre-existing live measurements keep their exact objects; only the
        # surviving report added one.
        assert gain.measurements["spice@tt"] is original
        for key, measurement in before.items():
            assert gain.measurements[key] is measurement
        assert len(gain.measurements) == len(before) + 1
        assert sorted(
            m.tool_name for m in gain.measurements.values()
            if m.profile_id is not None
        ) == ["preexisting", "survivor"]
    finally:
        db.conn.set_authorizer(_allow_sql)
        module.close()


def test_three_level_rollback_keeps_only_surviving_report_actions(
    pdk, tmp_path
):
    """AC6/AC7: store_report() succeeds in an inner scope, the enclosing
    intermediate scope fails afterwards, and the outer scope catches and
    continues. The inner report's SQL and its pending live update are both
    discarded; the outer scope's own reports survive, in order, once."""
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "three-level.db")
    module.create_circuit_version()
    original = _record_original(gain)
    db = module._tracker.db
    published = []
    record = gain._publish_measurement

    def publish(measurement, attempt, scope):
        record(measurement, attempt, scope)
        published.append(measurement)

    gain._publish_measurement = publish
    try:
        with db.transaction():                                   # outer
            module.store_report(
                _report((gain, 65.0), context=_context(parameters={"n": 1})),
                tool_name="outer-early",
            )
            with pytest.raises(RuntimeError, match="intermediate boom"):
                with db.transaction():                           # intermediate
                    module.store_report(                         # inner scope
                        _report(
                            (gain, 65.0), context=_context(parameters={"n": 2})
                        ),
                        tool_name="dropped",
                    )
                    raise RuntimeError("intermediate boom")
            assert published == []
            module.store_report(
                _report((gain, 65.0), context=_context(parameters={"n": 3})),
                tool_name="outer-late",
            )
            assert published == []
        assert [m.tool_name for m in published] == ["outer-early", "outer-late"]

        other = sqlite3.connect(db.db_path)
        try:
            assert _evidence_counts(other) == (2, 4, 2)
            assert sorted(
                row[0] for row in other.execute(
                    "SELECT tool_name FROM spec_measurements"
                ).fetchall()
            ) == ["outer-early", "outer-late"]
        finally:
            other.close()
        assert gain.measurements["spice@tt"] is original
        # Nothing from the rolled-back scope leaks into a later transaction.
        with db.transaction():
            pass
        assert len(published) == 2
    finally:
        gain._publish_measurement = record
        module.close()


def test_failed_spec_is_stored_evidence_not_a_rollback(pdk, tmp_path):
    """AC11: a valid measurement that misses its target is evidence. Spec
    failure is never an exception and never triggers the rollback boundary --
    including inside an enclosing transaction."""
    module, gain, _settling = _build(pdk, tmp_path, tmp_path / "failed-spec.db")
    module.create_circuit_version()
    db = module._tracker.db
    context = _context(parameters={"failing": True})
    try:
        with db.transaction():
            result = module.store_report(
                _report((gain, 55.0), context=context), tool_name="failing"
            )
        assert result == {"metrics_stored": 1, "specs_stored": 1, "skipped": {}}

        other = sqlite3.connect(db.db_path)
        try:
            assert _evidence_counts(other) == (1, 2, 1)
            value, passes = other.execute(
                "SELECT value, passes FROM spec_measurements"
            ).fetchone()
            assert (value, bool(passes)) == (55.0, False)
            status = other.execute(
                "SELECT value_text FROM simulation_results "
                "WHERE parameter = 'spec_Gain'"
            ).fetchone()[0]
        finally:
            other.close()
        assert status.startswith("fail (")
        stored = [
            m for m in gain.measurements.values() if m.profile_id is not None
        ]
        assert [(m.value, m.passes) for m in stored] == [(55.0, False)]
    finally:
        module.close()
