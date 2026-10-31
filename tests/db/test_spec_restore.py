"""Atomic first-class Spec measurement storage and live restore."""

import inspect

import pytest

from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.spec_report import SpecReport
from silk.db import Database
from silk.db.schema import REQUIRED_COLUMNS
from silk.exceptions import (
    CircuitError,
    DuplicateNameError,
    EvidenceMismatchError,
    UnsupportedFingerprintSchemeError,
)
from silk.specs import Measurement, Spec


_DEFAULT_KEY = object()


def _build(
    pdk,
    tmp_path,
    db_path,
    *,
    display_name="Readable gain",
    key="gain",
):
    module = Module(
        "PDK_skywater130",
        str(tmp_path / "spec-restore"),
        "spec_restore.cir",
        pdk_details=pdk,
        module_name="spec_restore",
        db_path=str(db_path),
    )
    module.add_net("out")
    key_arg = {} if key is _DEFAULT_KEY else {"key": key}
    gain = module.declare_spec(
        display_name,
        "gain_dB",
        ">=",
        60.0,
        units="dB",
        **key_arg,
    )
    settling = module.declare_spec(
        "Settling time",
        "settling_s",
        "<=",
        2e-6,
        units="s",
        key="settling",
    )
    return module, gain, settling


def _report(*observations):
    metrics = {spec.metric: value for spec, value in observations}
    report = SpecReport("characterize", "spec_restore", metrics)
    for spec, _value in observations:
        report.check(
            spec.name,
            spec.metric,
            spec.op,
            spec.target,
            spec_key=spec.key,
            owner_path="$",
        )
    return report


def _record(
    spec,
    value,
    *,
    source="spice",
    corner="tt",
    timestamp=10.0,
    sim_run_id="run-1",
    notes="nominal characterization",
    temperature=27.0,
    voltage=1.8,
    sim_type="ac",
    tool_name="ngspice-44",
    passes=None,
):
    measurement = Measurement(
        source=source,
        value=value,
        passes=passes,
        corner=corner,
        timestamp=timestamp,
        sim_run_id=sim_run_id,
        notes=notes,
        temperature=temperature,
        voltage=voltage,
        sim_type=sim_type,
        tool_name=tool_name,
    )
    if passes is None:
        measurement.passes = spec.evaluate(measurement)
    spec.record_measurement(measurement)
    return measurement


def _store_one(module, spec, value=65.0, *, captured=True, **measurement_kwargs):
    measurement = _record(spec, value, **measurement_kwargs)
    report = _report((spec, value))
    if captured:
        report.results[0]._observation = (
            '$', spec.key, spec.metric, spec.op, spec.target, spec.latest_attempt())
        report = SpecReport.from_dict(report.to_dict())
    result = module.store_report(
        report,
        corner=measurement.corner,
        temperature=measurement.temperature,
        voltage=measurement.voltage,
        sim_type=measurement.sim_type,
        tool_name=measurement.tool_name,
        source=measurement.source,
    )
    return result


def test_schema_v1_requires_complete_spec_measurement_context():
    expected = {"temperature", "voltage", "sim_type", "tool_name"}
    with Database(":memory:") as db:
        actual = {
            row["name"]
            for row in db.conn.execute("PRAGMA table_info(spec_measurements)")
        }

    assert expected <= actual
    assert expected <= REQUIRED_COLUMNS["spec_measurements"]


@pytest.mark.parametrize('captured', [False, True])
def test_report_round_trip_rebuild_restore_and_reconcile(pdk, tmp_path, captured):
    db_path = tmp_path / "round-trip.db"
    module, gain, _unmeasured = _build(pdk, tmp_path, db_path)
    version_id = module.create_circuit_version()
    result = _store_one(module, gain, captured=captured)

    assert result == {
        "metrics_stored": 1,
        "specs_stored": 1,
        "skipped": {},
    }
    db = module._tracker.db
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 1
    assert db.conn.execute(
        "SELECT COUNT(*) FROM simulation_results"
    ).fetchone()[0] == 2
    module.close()

    rebuilt, rebuilt_gain, rebuilt_unmeasured = _build(
        pdk, tmp_path, db_path
    )
    assert rebuilt.create_circuit_version() == version_id
    records = rebuilt.get_stored_specs()

    assert [record.spec_key for record in records] == ["gain", "settling"]
    by_key = {record.spec_key: record for record in records}
    assert by_key["settling"].measurements == ()
    stored = by_key["gain"].measurements[0]
    assert stored.origin_version_id == version_id
    assert stored.source == "spice"
    assert stored.value == 65.0
    assert stored.corner == "tt"
    assert stored.timestamp == (10.0 if captured else None)
    assert stored.sim_run_id == ("run-1" if captured else None)
    assert stored.notes == ("nominal characterization" if captured else "")
    assert stored.temperature == 27.0
    assert stored.voltage == 1.8
    assert stored.sim_type == "ac"
    assert stored.tool_name == "ngspice-44"
    assert stored.passes is True

    assert rebuilt.restore_spec_measurements() == 1
    restored = rebuilt_gain.measurements["spice@tt"]
    assert restored.value == 65.0
    assert restored.passes is True
    assert restored.origin_version_id == version_id
    assert restored.timestamp == (10.0 if captured else 0.0)
    assert restored.sim_run_id == stored.sim_run_id
    assert restored.notes == stored.notes
    assert rebuilt_gain.attempt_history() == ()
    assert restored.temperature == 27.0
    assert restored.voltage == 1.8
    assert restored.sim_type == "ac"
    assert restored.tool_name == "ngspice-44"
    assert rebuilt_unmeasured.measurements == {}

    reconciliation = rebuilt.reconcile_specs(prefer_source="spice")
    assert [outcome.status for outcome in reconciliation.outcomes] == [
        "PASS",
        "UNMEASURED",
    ]
    if captured:
        # A later save with unknown observation time sorts before known evidence.
        rebuilt.store_report(_report((rebuilt_gain, 66.0)), corner='tt',
                             temperature=27.0, voltage=1.8,
                             sim_type='ac', tool_name='ngspice-44')
        history = next(r for r in rebuilt.get_stored_specs() if r.spec_key == 'gain')
        assert [(m.value, m.timestamp) for m in history.measurements] == [
            (66.0, None), (65.0, 10.0)]
        assert rebuilt.restore_spec_measurements() == 1
        assert rebuilt_gain.measurements['spice@tt'].value == 65.0
        assert rebuilt_gain.measurements['spice@tt'].timestamp == 10.0
    rebuilt.close()


@pytest.mark.parametrize('explicit', [False, True])
def test_store_report_without_observation_uses_explicit_context_only(
    pdk, tmp_path, explicit
):
    module, gain, _settling = _build(
        pdk, tmp_path, tmp_path / "context-fallback.db"
    )
    module.create_circuit_version()
    _record(gain, 65.0)

    arguments = (dict(temperature=85.0, voltage=1.2, sim_type='tran',
                      tool_name='explicit-tool') if explicit else {})
    module.store_report(_report((gain, 65.0)), **arguments)

    db = module._tracker.db
    result_contexts = db.conn.execute(
        """
        SELECT temperature, voltage, sim_type, tool_name
        FROM simulation_results
        ORDER BY parameter
        """
    ).fetchall()
    measurement_context = db.conn.execute(
        """
        SELECT temperature, voltage, sim_type, tool_name
        FROM spec_measurements
        """
    ).fetchone()
    expected = (85.0, 1.2, 'tran', 'explicit-tool') if explicit else (None,) * 4
    assert [tuple(row) for row in result_contexts] == [expected, expected]
    assert tuple(measurement_context) == expected
    row = db.conn.execute('SELECT timestamp, sim_run_id, notes FROM spec_measurements').fetchone()
    assert tuple(row) == (None, None, '')
    module.close()


def test_generated_nested_report_uses_canonical_owner_path(
    pdk, tmp_path, monkeypatch
):
    child = Module(
        "PDK_skywater130",
        str(tmp_path / "nested-child"),
        "child.cir",
        pdk_details=pdk,
        module_name="child",
    )
    tb = ModuleTestbench(child, name="unit")
    tb.add_metric("gain_dB", lambda _ckt: 65.0)
    tb.add_spec("Child gain", "gain_dB", ">=", 60.0, key="gain")

    middle = Module(
        "PDK_skywater130",
        str(tmp_path / "nested-middle"),
        "middle.cir",
        pdk_details=pdk,
        module_name="middle",
    )
    middle.add_module(child, "u1", nets={})

    top = Module(
        "PDK_skywater130",
        str(tmp_path / "nested-top"),
        "top.cir",
        pdk_details=pdk,
        module_name="nested_top",
        db_path=str(tmp_path / "nested.db"),
    )
    top.add_module(middle, "mid", nets={})
    stored_middle = top._element_mgr.elem_dict["mid"]["device"]
    stored_child = stored_middle._element_mgr.elem_dict["u1"]["device"]
    stored_tb = stored_child._tests["unit"]
    stored_tb._create_netlist()
    stored_tb._module_added = True

    def _fake_simulate(corner="tt"):
        # No-op stand-in for the simulator; also flips sim_results_loaded
        # (TASK-115: run() calls the public evaluate(), whose "nothing
        # loaded" guard would otherwise reject this never-really-simulated
        # netlist) so the extraction tail still runs on gain_dB's constant.
        stored_tb.netlist.sim_results_loaded = True

    monkeypatch.setattr(stored_tb.netlist, "simulate", _fake_simulate)

    reports = top.run_all_tests(recursive=True, corner="tt")
    report = reports["child.unit"]
    assert report.results[0].owner_path == "$/mid/u1"
    assert report.results[0].spec_key == "gain"

    top.create_circuit_version()
    assert top.store_report(report, corner="tt")["specs_stored"] == 1
    nested = next(
        record
        for record in top.get_stored_specs()
        if record.owner_path == "$/mid/u1" and record.spec_key == "gain"
    )
    assert [measurement.value for measurement in nested.measurements] == [65.0]
    top.close()


def test_extractor_error_spec_keeps_skip_and_compatibility_behavior(
    pdk, tmp_path
):
    module, gain, settling = _build(
        pdk, tmp_path, tmp_path / "extractor-error.db"
    )
    version_id = module.create_circuit_version()
    _record(settling, 1e-6)
    report = SpecReport(
        "characterize",
        "spec_restore",
        {
            "gain_dB": "Error: extractor failed",
            "settling_s": 1e-6,
        },
    )
    report.check(
        gain.name,
        gain.metric,
        gain.op,
        gain.target,
        spec_key=gain.key,
        owner_path="$",
    )
    report.check(
        settling.name,
        settling.metric,
        settling.op,
        settling.target,
        spec_key=settling.key,
        owner_path="$",
    )

    result = module.store_report(report, corner="tt")

    assert result == {
        "metrics_stored": 1,
        "specs_stored": 2,
        "skipped": {
            "gain_dB": "non-numeric value: 'Error: extractor failed'"
        },
    }
    db = module._tracker.db
    assert db.conn.execute(
        """
        SELECT COUNT(*) FROM simulation_results
        WHERE circuit_version_id = ? AND parameter LIKE 'spec_%'
        """,
        (version_id,),
    ).fetchone()[0] == 2
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 1
    module.close()


def test_store_report_rolls_back_metrics_statuses_and_measurements(
    pdk, tmp_path, monkeypatch
):
    module, gain, settling = _build(
        pdk, tmp_path, tmp_path / "atomic.db"
    )
    version_id = module.create_circuit_version()
    _record(gain, 65.0)
    _record(settling, 1e-6)
    report = _report((gain, 65.0), (settling, 1e-6))
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
        module.store_report(report, corner="tt", source="spice")

    assert db.conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 0
    module.close()


def test_cached_pass_disagreement_is_rejected_on_store(pdk, tmp_path):
    module, gain, _settling = _build(
        pdk, tmp_path, tmp_path / "bad-store-cache.db"
    )
    version_id = module.create_circuit_version()
    _record(gain, 65.0)
    report = _report((gain, 65.0))
    report.results[0].passed = False

    with pytest.raises(EvidenceMismatchError, match="cached|pass"):
        module.store_report(report)

    db = module._tracker.db
    assert db.conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 0
    module.close()


def test_cached_pass_disagreement_is_rejected_on_historical_read(
    pdk, tmp_path
):
    module, gain, _settling = _build(
        pdk, tmp_path, tmp_path / "bad-read-cache.db"
    )
    version_id = module.create_circuit_version()
    _store_one(module, gain)
    db = module._tracker.db
    db.conn.execute("UPDATE spec_measurements SET passes = 0")
    db.conn.commit()

    with pytest.raises(EvidenceMismatchError, match="cached|pass"):
        db.get_stored_specs(version_id)
    module.close()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("structure", EvidenceMismatchError),
        ("target", EvidenceMismatchError),
        ("missing", EvidenceMismatchError),
        ("extra", EvidenceMismatchError),
        ("duplicate", DuplicateNameError),
        ("bad_parent", CircuitError),
        ("unsupported_scheme", UnsupportedFingerprintSchemeError),
        ("silk.structure/v1", UnsupportedFingerprintSchemeError),
        ("silk.structure/v2", UnsupportedFingerprintSchemeError),
    ],
)
def test_restore_rejects_live_or_stored_identity_drift_before_mutation(
    pdk, tmp_path, mutation, expected_error
):
    db_path = tmp_path / f"mismatch-{mutation.replace('/', '-')}.db"
    original, original_gain, _settling = _build(pdk, tmp_path, db_path)
    original.create_circuit_version()
    _store_one(original, original_gain)
    original.close()

    rebuilt, gain, _settling = _build(pdk, tmp_path, db_path)
    rebuilt.create_circuit_version()
    existing = Measurement(
        source="behavioral", value=62.0, timestamp=1.0, notes="keep me"
    )
    gain.record_measurement(existing)
    before = dict(gain.measurements)

    if mutation == "structure":
        rebuilt.add_net("late_edit")
    elif mutation == "target":
        # Deliberate corruption: restore must still reject identity drift even
        # when a caller bypasses the public declaration guard.
        object.__setattr__(gain, "target", 70.0)
    elif mutation == "missing":
        rebuilt._specs.remove(gain)
    elif mutation == "extra":
        rebuilt.declare_spec("Extra", "extra", ">", 0.0, key="extra")
    elif mutation == "duplicate":
        rebuilt._specs.append(
            Spec("Duplicate", rebuilt, "gain_dB", ">=", 60.0, key="gain")
        )
    elif mutation == "bad_parent":
        gain.parent = Spec("External", rebuilt, "x", ">", 0.0)
    elif mutation == "unsupported_scheme":
        db = rebuilt._tracker.db
        db.conn.execute(
            "UPDATE circuit_versions SET spec_scheme = 'silk.spec/v999' "
            "WHERE version_id = ?",
            (rebuilt._tracker.circuit_id,),
        )
        db.conn.commit()

    elif mutation.startswith("silk.structure/"):
        db = rebuilt._tracker.db
        db.conn.execute(
            "UPDATE circuit_versions SET structure_scheme = ? WHERE version_id = ?",
            (mutation, rebuilt._tracker.circuit_id),
        )
        db.conn.commit()

    stored_before = list(rebuilt._tracker.db.conn.iterdump())
    with pytest.raises(expected_error) as error:
        rebuilt.restore_spec_measurements()
    if mutation.startswith("silk.structure/"):
        assert "rebuild" in str(error.value)
        assert "new database" in str(error.value)
    assert list(rebuilt._tracker.db.conn.iterdump()) == stored_before

    assert gain.measurements == before
    assert gain.measurements["behavioral"] is existing
    rebuilt.close()


def test_explicit_key_display_rename_restores_and_keeps_live_name(
    pdk, tmp_path
):
    db_path = tmp_path / "explicit-rename.db"
    original, gain, _settling = _build(
        pdk, tmp_path, db_path, display_name="Old display", key="gain"
    )
    version_id = original.create_circuit_version()
    _store_one(original, gain)
    original.close()

    rebuilt, renamed, _settling = _build(
        pdk, tmp_path, db_path, display_name="New display", key="gain"
    )
    assert rebuilt.create_circuit_version() == version_id
    assert rebuilt.restore_spec_measurements() == 1
    assert renamed.name == "New display"
    assert renamed.measurements["spice@tt"].value == 65.0
    rebuilt.close()


def test_default_key_display_rename_does_not_restore(pdk, tmp_path):
    db_path = tmp_path / "default-rename.db"
    original, gain, _settling = _build(
        pdk,
        tmp_path,
        db_path,
        display_name="Old display",
        key=_DEFAULT_KEY,
    )
    original.create_circuit_version()
    _store_one(original, gain)
    original.close()

    rebuilt, renamed, _settling = _build(
        pdk,
        tmp_path,
        db_path,
        display_name="New display",
        key=_DEFAULT_KEY,
    )
    rebuilt.create_circuit_version()
    with pytest.raises(EvidenceMismatchError, match="incompatible"):
        rebuilt.restore_spec_measurements()
    assert renamed.measurements == {}
    rebuilt.close()


def test_history_keeps_reruns_and_restore_selects_latest_per_slot(
    pdk, tmp_path
):
    db_path = tmp_path / "reruns.db"
    module, gain, _settling = _build(pdk, tmp_path, db_path)
    version_id = module.create_circuit_version()
    observations = [
        (61.0, "spice", "tt", 10.0),
        (64.0, "spice", "ff", 12.0),
        (63.0, "behavioral", "tt", 15.0),
        (62.0, "spice", "tt", 20.0),
    ]
    for index, (value, source, corner, timestamp) in enumerate(observations):
        _store_one(
            module,
            gain,
            value,
            source=source,
            corner=corner,
            timestamp=timestamp,
            sim_run_id=f"run-{index}",
            tool_name="silk" if source == "behavioral" else "ngspice-44",
        )
    module.close()

    rebuilt, rebuilt_gain, _settling = _build(pdk, tmp_path, db_path)
    assert rebuilt.create_circuit_version() == version_id
    gain_history = next(
        record
        for record in rebuilt.get_stored_specs()
        if record.spec_key == "gain"
    )
    assert [m.value for m in gain_history.measurements] == [
        61.0,
        64.0,
        63.0,
        62.0,
    ]

    assert rebuilt.restore_spec_measurements(
        source="spice",
        corner="tt",
        temperature=27.0,
        voltage=1.8,
        tool_name="ngspice-44",
    ) == 1
    assert set(rebuilt_gain.measurements) == {"spice@tt"}
    assert rebuilt_gain.measurements["spice@tt"].value == 62.0

    rebuilt_gain.measurements.clear()
    assert rebuilt.restore_spec_measurements() == 3
    assert rebuilt_gain.measurements["spice@tt"].value == 62.0
    assert rebuilt_gain.measurements["spice@ff"].value == 64.0
    assert rebuilt_gain.measurements["behavioral@tt"].value == 63.0
    rebuilt.close()


def test_store_report_rejects_result_without_declaration_identity(
    pdk, tmp_path
):
    module, gain, _settling = _build(
        pdk, tmp_path, tmp_path / "missing-result-identity.db"
    )
    version_id = module.create_circuit_version()
    _record(gain, 65.0)
    report = SpecReport("characterize", "spec_restore", {"gain_dB": 65.0})
    report.check(gain.name, gain.metric, gain.op, gain.target)

    with pytest.raises(EvidenceMismatchError, match="identity|spec_key"):
        module.store_report(report)

    db = module._tracker.db
    assert db.conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE circuit_version_id = ?",
        (version_id,),
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0] == 0
    module.close()


def test_module_exposes_exact_restore_surface():
    assert str(inspect.signature(Module.restore_spec_measurements)) == (
        "(self, *, source=None, corner=None, temperature=None, voltage=None, "
        "tool_name=None, profile_id=None, include_ancestors=True)"
    )


@pytest.mark.parametrize('field', [None, 'sigma'])
@pytest.mark.parametrize('value', [float('nan'), 50.0, 65.0])
def test_calibration_report_storage_rebuild_restore(pdk, tmp_path, field, value):
    import math
    from silk.circuit.report import generate_report
    from silk.specs import CalibrationBinding
    from silk.simulation.variation import Normal

    path = tmp_path / 'calibration.db'
    module, gain, settling = _build(pdk, tmp_path, path)
    original = Normal(0, 1) if field else 1.0
    module._behavioral_params = {'gain': original, 'settling': 0.0}
    gain.calibration = CalibrationBinding(module, 'gain', field=field)
    settling.calibration = CalibrationBinding(module, 'settling')
    module.create_circuit_version()
    if math.isnan(value):
        _record(gain, 65.0, timestamp=5.0)
        module.store_report(_report((gain, 65.0)), corner='tt')
    _record(gain, value)
    _record(settling, 1e-6)
    live = module.reconcile_specs()
    assert live.outcomes[0].calibrated is (not math.isnan(value))
    assert live.outcomes[0].status == ('PASS' if value >= 60 else 'FAIL')
    if math.isnan(value):
        assert module._behavioral_params['gain'] is original
    else:
        assert (module._behavioral_params['gain'].sigma if field else
                module._behavioral_params['gain']) == value
    module.store_report(_report((gain, value), (settling, 1e-6)), corner='tt')
    module.close()

    rebuilt, gain, settling = _build(pdk, tmp_path, path)
    original = Normal(0, 1) if field else 1.0
    rebuilt._behavioral_params = {'gain': original, 'settling': 0.0}
    gain.calibration = CalibrationBinding(rebuilt, 'gain', field=field)
    settling.calibration = CalibrationBinding(rebuilt, 'settling')
    rebuilt.create_circuit_version()
    assert rebuilt.restore_spec_measurements() == 2
    report = generate_report(rebuilt)
    outcome = report.spec_reconciliation.outcomes[0]
    assert outcome.calibrated is (not math.isnan(value))
    assert outcome.status == ('PASS' if value >= 60 else 'FAIL')
    assert rebuilt._behavioral_params['settling'] == 1e-6
    if math.isnan(value):
        assert rebuilt._behavioral_params['gain'] is original
        assert math.isnan(outcome.measurement.value)
        assert outcome.measurement.passes is False
        assert len(report.spec_reconciliation.calibrations_applied) == 1
        assert 'settling' in report.spec_reconciliation.calibrations_applied[0]
    else:
        assert len(report.spec_reconciliation.calibrations_applied) == 2
        assert (rebuilt._behavioral_params['gain'].sigma if field else
                rebuilt._behavioral_params['gain']) == value
    rebuilt.close()
