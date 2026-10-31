"""Aggregate profile-scoped readback and restore (TASK-8 / DB-02-READBACK)."""

from dataclasses import FrozenInstanceError
import inspect

import pytest

import silk.exceptions as silk_exceptions
from silk.circuit.modules import Module
from silk.circuit.spec_report import SpecReport
from silk.db import Database, DesignStage
from silk.db.context import CharacterizationContext
from silk.exceptions import (
    EvidenceMismatchError,
    EvidenceNotFoundError,
    UnsupportedFingerprintSchemeError,
)
from tests.db_snapshot_helpers import (
    create_sealed_snapshot,
    structure_with_net,
)
from silk.specs import Measurement


def _context(*, test_name="load_step", parameters=None, **overrides):
    values = {
        "owner_path": "$/amp",
        "test_name": test_name,
        "parameters": {"cload_f": 2e-12} if parameters is None else parameters,
        "corner": "tt",
        "temperature": 27.0,
        "voltage": 1.8,
    }
    values.update(overrides)
    return CharacterizationContext(**values)


def _current_row(avg_a):
    return {
        "net_name": "out",
        "pin_id": "",
        "avg_a": avg_a,
        "rms_a": avg_a * 1.1,
        "peak_a": avg_a * 2.0,
        "source": "spice",
        "current_vector": None,
    }


def _store_selected_context(db, version_id, context, *, gain, avg_a):
    profile = db.store_characterization_profile(version_id, context)
    db.add_simulation_result(
        version_id,
        "gain_dB",
        value=gain,
        unit="dB",
        source="spice",
        sim_type="ac",
        corner=context.corner,
        temperature=context.temperature,
        voltage=context.voltage,
        tool_name="ngspice-44",
        profile_id=profile.profile_id,
    )
    version = db._snapshot_row(version_id)
    current_set_id = db.replace_net_current_set(
        version_id,
        corner=context.corner,
        temperature=context.temperature,
        voltage=context.voltage,
        structure_scheme=version["structure_scheme"],
        structure_hash=version["structure_hash"],
        context=context,
        rows=[_current_row(avg_a)],
        analysis="transient",
        tool_name="ngspice-44",
    )
    return profile, current_set_id


def _build_live(pdk, tmp_path, db_path, *, target=60.0):
    module = Module(
        "PDK_skywater130",
        str(tmp_path / "characterization-readback"),
        "characterization_readback.cir",
        pdk_details=pdk,
        module_name="characterization_readback",
        db_path=str(db_path),
    )
    module.add_net("out")
    gain = module.declare_spec(
        "Gain", "gain_dB", ">=", target, units="dB", key="gain"
    )
    return module, gain


def _scoped_report(gain, context, value=64.0):
    report = SpecReport(
        "load_step",
        "characterization_readback",
        {"gain_dB": value},
        owner_path="$/amp",
        parameters=dict(context.parameters),
        characterization_context=context,
    )
    report.check(
        gain.name,
        gain.metric,
        gain.op,
        gain.target,
        spec_key=gain.key,
        owner_path="$",
    )
    return report


def _store_live_context(module, gain, context, *, avg_a=1e-3):
    version_id = module.create_circuit_version()
    module.store_report(_scoped_report(gain, context))
    profile = module._tracker.db.resolve_characterization_profile(
        version_id, context=context
    )
    version = module._tracker.db._snapshot_row(version_id)
    current_set_id = module._tracker.db.replace_net_current_set(
        version_id,
        corner=context.corner,
        temperature=context.temperature,
        voltage=context.voltage,
        structure_scheme=version["structure_scheme"],
        structure_hash=version["structure_hash"],
        context=context,
        rows=[_current_row(avg_a)],
    )
    return version_id, profile, current_set_id


def test_list_profiles_returns_compatible_lineage_nearest_first():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        source_load = db.store_characterization_profile(source_id, _context())
        source_startup = db.store_characterization_profile(
            source_id,
            _context(test_name="startup", parameters={"slew_v_s": 1e6}),
        )
        target_id = db.promote_circuit(
            source_id, DesignStage.STABLE, lock=False
        )
        target_load = db.store_characterization_profile(target_id, _context())

        records = db.list_characterization_profiles(target_id)

    assert [record.profile_id for record in records] == [
        target_load.profile_id,
        source_load.profile_id,
        source_startup.profile_id,
    ]
    assert [record.origin_version_id for record in records] == [
        target_id,
        source_id,
        source_id,
    ]


def test_resolve_profile_uses_exact_id_or_nearest_complete_descriptor():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        source = db.store_characterization_profile(source_id, _context())
        target_id = db.promote_circuit(
            source_id, DesignStage.STABLE, lock=False
        )
        target = db.store_characterization_profile(target_id, _context())

        by_id = db.resolve_characterization_profile(
            target_id, profile_id=source.profile_id
        )
        by_descriptor = db.resolve_characterization_profile(
            target_id,
            context=_context(corner="ss", temperature=125.0, voltage=1.62),
        )

    assert by_id == source
    assert by_descriptor == target


def test_resolve_profile_distinguishes_missing_from_incompatible_lineage():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        stored = db.store_characterization_profile(source_id, _context())
        incompatible_id = create_sealed_snapshot(
            db,
            name="amp",
            version="changed",
            parent_version_id=source_id,
            structure=structure_with_net("changed"),
        )

        with pytest.raises(EvidenceMismatchError, match="profile.*structure"):
            db.resolve_characterization_profile(
                incompatible_id, profile_id=stored.profile_id
            )
        with pytest.raises(EvidenceNotFoundError, match="profile"):
            db.resolve_characterization_profile(
                incompatible_id,
                profile_id="00000000-0000-4000-8000-000000000000",
            )


def test_profile_listing_validates_stored_scheme_before_returning_records():
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db, name="amp")
        stored = db.store_characterization_profile(version_id, _context())
        db.conn.execute(
            "UPDATE characterization_profiles SET profile_scheme = ? "
            "WHERE profile_id = ?",
            ("silk.profile/v999", stored.profile_id),
        )
        db.conn.commit()

        with pytest.raises(UnsupportedFingerprintSchemeError):
            db.list_characterization_profiles(version_id)


def test_select_exact_context_reports_stable_ids_and_true_lineage_origin():
    with Database(":memory:") as db:
        source_id = create_sealed_snapshot(db, name="amp")
        context = _context(parameters={"mode": "nominal"})
        profile, current_set_id = _store_selected_context(
            db, source_id, context, gain=64.0, avg_a=1e-3
        )
        target_id = db.promote_circuit(
            source_id, DesignStage.STABLE, lock=False
        )

        selected = db.select_characterization_evidence(
            target_id,
            profile_id=profile.profile_id,
            corner="tt",
            temperature=27.0,
            voltage=1.8,
        )

    assert selected.target_version_id == target_id
    assert selected.origin_version_id == source_id
    assert selected.profile_id == profile.profile_id
    assert selected.current_set_id == current_set_id
    assert selected.profile == profile
    assert selected.current_set.origin_version_id == source_id
    assert selected.current_set.profile_id == profile.profile_id
    assert selected.current_set.current_set_id == current_set_id
    assert [row.avg_a for row in selected.current_set.currents] == [
        pytest.approx(1e-3)
    ]
    assert [record.value for record in selected.metrics] == [64.0]
    assert selected.specs == ()
    with pytest.raises(FrozenInstanceError):
        selected.profile_id = "changed"


def test_select_context_never_crosses_profiles_or_pvt_points():
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db, name="amp")
        nominal = _context(parameters={"mode": "nominal"})
        alternate = _context(parameters={"mode": "alternate"})
        hot = _context(
            parameters={"mode": "nominal"},
            temperature=85.0,
            voltage=1.62,
        )
        nominal_profile, nominal_set = _store_selected_context(
            db, version_id, nominal, gain=61.0, avg_a=1e-3
        )
        _alternate_profile, _alternate_set = _store_selected_context(
            db, version_id, alternate, gain=72.0, avg_a=2e-3
        )
        hot_profile, hot_set = _store_selected_context(
            db, version_id, hot, gain=58.0, avg_a=3e-3
        )
        assert hot_profile.profile_id == nominal_profile.profile_id

        selected = db.select_characterization_evidence(
            version_id, context=nominal
        )
        selected_hot = db.select_characterization_evidence(
            version_id, context=hot
        )

    assert selected.profile_id == nominal_profile.profile_id
    assert selected.current_set_id == nominal_set
    assert [record.value for record in selected.metrics] == [61.0]
    assert [row.avg_a for row in selected.current_set.currents] == [
        pytest.approx(1e-3)
    ]
    assert selected_hot.current_set_id == hot_set
    assert [record.value for record in selected_hot.metrics] == [58.0]
    assert [row.avg_a for row in selected_hot.current_set.currents] == [
        pytest.approx(3e-3)
    ]


def test_select_context_distinguishes_missing_from_known_wrong_pvt():
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db, name="amp")
        context = _context()
        profile = db.store_characterization_profile(version_id, context)

        with pytest.raises(EvidenceNotFoundError, match="selected.*evidence"):
            db.select_characterization_evidence(
                version_id, context=context
            )

        _store_selected_context(
            db, version_id, context, gain=64.0, avg_a=1e-3
        )
        with pytest.raises(EvidenceMismatchError, match="PVT"):
            db.select_characterization_evidence(
                version_id,
                profile_id=profile.profile_id,
                corner="tt",
                temperature=85.0,
                voltage=1.62,
            )


def test_select_context_rejects_ambiguous_or_incomplete_selectors():
    ambiguity_error = getattr(
        silk_exceptions, "EvidenceAmbiguityError", None
    )
    assert ambiguity_error is not None, (
        "TASK-8 requires a distinct typed ambiguity error"
    )
    with Database(":memory:") as db:
        version_id = create_sealed_snapshot(db, name="amp")
        context = _context()
        profile, _current_set = _store_selected_context(
            db, version_id, context, gain=64.0, avg_a=1e-3
        )

        with pytest.raises(ambiguity_error, match="exactly one"):
            db.select_characterization_evidence(
                version_id,
                profile_id=profile.profile_id,
                context=context,
            )
        with pytest.raises(ValueError, match="corner.*temperature.*voltage"):
            db.select_characterization_evidence(
                version_id,
                profile_id=profile.profile_id,
                corner="tt",
            )


def test_rebuilt_module_lists_selects_and_atomically_restores_one_context(
    pdk, tmp_path
):
    db_path = tmp_path / "aggregate-restore.db"
    context = _context(parameters={"mode": "nominal"})
    original, gain = _build_live(pdk, tmp_path, db_path)
    version_id, profile, current_set_id = _store_live_context(
        original, gain, context
    )
    original.close()

    rebuilt, rebuilt_gain = _build_live(pdk, tmp_path, db_path)
    assert rebuilt.create_circuit_version() == version_id
    assert [
        record.profile_id for record in rebuilt.list_characterization_profiles()
    ] == [profile.profile_id]
    assert rebuilt.resolve_characterization_profile(
        context=context
    ) == profile
    selected = rebuilt.select_characterization_evidence(context=context)
    restored = rebuilt.restore_characterization_evidence(context=context)

    assert selected.profile_id == restored.profile_id == profile.profile_id
    assert selected.current_set_id == restored.current_set_id == current_set_id
    assert restored.origin_version_id == version_id
    assert restored.target_version_id == version_id
    assert restored.measurement_count == 1
    assert restored.current_count == 1
    restored_measurement = next(iter(rebuilt_gain.measurements.values()))
    assert restored_measurement.value == 64.0
    assert restored_measurement.profile_id == profile.profile_id
    assert restored_measurement.origin_version_id == version_id
    assert rebuilt.net_dict["out"].avg_current == pytest.approx(1e-3)
    assert rebuilt.net_dict["out"]._current_source == "db"
    rebuilt.close()


def test_aggregate_restore_validates_late_current_binding_before_any_mutation(
    pdk, tmp_path
):
    db_path = tmp_path / "atomic-late-current.db"
    context = _context(parameters={"mode": "nominal"})
    original, gain = _build_live(pdk, tmp_path, db_path)
    version_id, profile, current_set_id = _store_live_context(
        original, gain, context
    )
    original._tracker.db.conn.execute(
        "UPDATE net_currents SET net_name = 'ghost' WHERE current_set_id = ?",
        (current_set_id,),
    )
    original._tracker.db.conn.commit()
    original.close()

    rebuilt, rebuilt_gain = _build_live(pdk, tmp_path, db_path)
    assert rebuilt.create_circuit_version() == version_id
    keep = Measurement(source="user", value=62.0, notes="keep")
    rebuilt_gain.record_measurement(keep)
    rebuilt.net_dict["out"].avg_current = 9e-3
    rebuilt.net_dict["out"]._current_source = "user"
    before_measurements = dict(rebuilt_gain.measurements)

    with pytest.raises(EvidenceMismatchError, match="stored net.*ghost"):
        rebuilt.restore_characterization_evidence(
            profile_id=profile.profile_id,
            corner="tt",
            temperature=27.0,
            voltage=1.8,
            force=True,
        )

    assert rebuilt_gain.measurements == before_measurements
    assert rebuilt.net_dict["out"].avg_current == pytest.approx(9e-3)
    assert rebuilt.net_dict["out"]._current_source == "user"
    rebuilt.close()


def test_module_readback_apis_are_keyword_only_after_self():
    expected = {
        "list_characterization_profiles": ["include_ancestors"],
        "resolve_characterization_profile": [
            "profile_id", "context", "include_ancestors"
        ],
        "select_characterization_evidence": [
            "profile_id", "context", "corner", "temperature", "voltage",
            "include_ancestors",
        ],
        "restore_characterization_evidence": [
            "profile_id", "context", "corner", "temperature", "voltage",
            "include_ancestors", "force",
        ],
    }
    for method_name, keyword_names in expected.items():
        method = getattr(Module, method_name, None)
        assert method is not None, f"Module.{method_name} is missing"
        parameters = inspect.signature(method).parameters
        assert list(parameters)[1:] == keyword_names
        assert all(
            parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            for name in keyword_names
        )


@pytest.mark.parametrize('historical_scheme', ['silk.structure/v1', 'silk.structure/v2'])
def test_historical_profile_evidence_restore_preserves_database_and_live_state(
    pdk, tmp_path, historical_scheme,
):
    db_path = tmp_path / 'historical-context.db'
    context = _context(parameters={'mode': 'nominal'})
    original, gain = _build_live(pdk, tmp_path, db_path)
    version_id, profile, current_set_id = _store_live_context(original, gain, context)
    db = original._tracker.db
    db.conn.execute('UPDATE circuit_versions SET structure_scheme = ?', (historical_scheme,))
    db.conn.execute('UPDATE net_current_sets SET structure_scheme = ?', (historical_scheme,))
    db.conn.commit()
    stored = list(db.conn.iterdump())
    original.close()

    rebuilt, rebuilt_gain = _build_live(pdk, tmp_path, db_path)
    keep = Measurement(source='user', value=62.0, notes='keep')
    rebuilt_gain.record_measurement(keep)
    rebuilt.net_dict['out'].avg_current = 9e-3
    rebuilt.net_dict['out']._current_source = 'user'
    before_measurements = dict(rebuilt_gain.measurements)
    try:
        with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild.*new database'):
            rebuilt.create_circuit_version()
        # Even a caller bypassing the snapshot-binding guard cannot restore
        # historical evidence. Exercise the independent read/restore guard.
        rebuilt._ensure_tracker().circuit_id = version_id
        with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild.*new database'):
            rebuilt.restore_characterization_evidence(profile_id=profile.profile_id)
        assert rebuilt_gain.measurements == before_measurements
        assert rebuilt_gain.measurements['user'] is keep
        assert rebuilt.net_dict['out'].avg_current == 9e-3
        assert rebuilt.net_dict['out']._current_source == 'user'
        with Database(str(db_path)) as historical:
            assert list(historical.conn.iterdump()) == stored
            assert historical.conn.execute(
                'SELECT structure_scheme FROM circuit_versions WHERE version_id = ?',
                (version_id,),
            ).fetchone()[0] == historical_scheme
            assert historical.conn.execute(
                'SELECT avg_a FROM net_currents WHERE current_set_id = ?',
                (current_set_id,),
            ).fetchone()[0] == 1e-3
    finally:
        rebuilt.close()


@pytest.mark.parametrize('unattributed', [False, True])
def test_restore_characterization_refuses_swept_attribution_atomically(pdk, tmp_path, unattributed):
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'swept.db')
    context = _context(parameters={'mode': 'nominal'})
    version_id = module.create_circuit_version()
    db = module._tracker.db
    profile = db.store_characterization_profile(version_id, context)
    version = db._snapshot_row(version_id)
    current_set = db.replace_net_current_set(version_id, corner=context.corner,
        temperature=context.temperature, voltage=context.voltage,
        structure_scheme=version['structure_scheme'], structure_hash=version['structure_hash'],
        context=context, rows=[_current_row(1e-3)])
    if unattributed:
        module.store_report(_scoped_report(gain, context))
    for trim in (1, 3):
        report = _scoped_report(gain, context)
        report.parameters['silk_run_variables'] = {'trim': trim}
        module.store_report(report)
    assert module._tracker.db.conn.execute('SELECT COUNT(*) FROM spec_measurements').fetchone()[0] == (3 if unattributed else 2)
    module.net_dict['out'].avg_current = 9e-3
    module.net_dict['out']._current_source = 'user'
    before = dict(gain.measurements)
    from silk.circuit.test_placement import evidence_index
    index = evidence_index(module)
    observations = repr(index)
    selected = module.select_characterization_evidence(context=context)
    assert selected.current_set_id == current_set
    assert selected.profile_id == profile.profile_id
    assert selected.deck_check.state != 'stale'
    with pytest.raises(EvidenceMismatchError, match='Variable|variable|attribution|ambig'):
        module.restore_characterization_evidence(context=context)
    assert gain.measurements == before
    assert repr(index) == observations
    assert module.net_dict['out'].avg_current == 9e-3
    assert module.net_dict['out']._current_source == 'user'
    module.close()


@pytest.mark.parametrize('aggregate', [False, True])
@pytest.mark.parametrize('damage', ['missing', 'wrong_id', 'wrong_source', 'null_id', 'malformed_id'])
def test_spec_restore_requires_exact_swept_measurement_link(pdk, tmp_path, damage, aggregate):
    import json
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'linked.db')
    version_id = module.create_circuit_version()
    context = _context(parameters={'mode': 'nominal'})
    report = _scoped_report(gain, context)
    report.parameters['silk_run_variables'] = {'trim': 3}
    module.store_report(report)
    db = module._tracker.db
    version = db._snapshot_row(version_id)
    db.replace_net_current_set(version_id, corner=context.corner, temperature=context.temperature,
        voltage=context.voltage, structure_scheme=version['structure_scheme'],
        structure_hash=version['structure_hash'], context=context, rows=[_current_row(1e-3)])
    row = db.conn.execute("SELECT sim_id, conditions FROM simulation_results WHERE parameter LIKE 'spec_%'").fetchone()
    marker = json.loads(row['conditions'])
    if damage == 'missing':
        del marker['silk_spec_measurement']
    elif damage == 'wrong_id':
        marker['silk_spec_measurement']['measurement_id'] = 'missing-id'
    elif damage in ('null_id', 'malformed_id'):
        marker['silk_spec_measurement']['measurement_id'] = None if damage == 'null_id' else []
    else:
        db.conn.execute("UPDATE simulation_results SET source = 'behavioral' WHERE sim_id = ?", (row['sim_id'],))
    db.conn.execute('UPDATE simulation_results SET conditions = ? WHERE sim_id = ?',
                    (json.dumps(marker), row['sim_id']))
    db.conn.commit()
    before = dict(gain.measurements)
    from silk.circuit.test_placement import evidence_index
    observations = repr(evidence_index(module))
    module.net_dict['out'].avg_current = 9e-3
    with pytest.raises(EvidenceMismatchError, match='attribution|Variable'):
        if aggregate:
            module.restore_characterization_evidence(context=context)
        else:
            module.restore_spec_measurements()
    assert gain.measurements == before
    assert repr(evidence_index(module)) == observations
    assert module.net_dict['out'].avg_current == 9e-3
    module.close()


@pytest.mark.parametrize('extra', [{}, {'mode': 'μ/%~', 'flag': True, 'gain': 1.25, 'count': 2}])
def test_swept_report_rows_keep_coordinate_identity_after_storage(pdk, tmp_path, extra):
    from silk import Variable
    knob = Variable('trim', value=7)
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'coordinates.db')
    module.create_circuit_version()
    context = _context(parameters={'mode': 'nominal'})
    reports = []
    for value in (1, 3):
        knob.value = value
        report = _scoped_report(gain, context)
        report.parameters['silk_run_variables'] = {'trim': knob.value, **extra}
        reports.append(report)
    knob.value = 7
    for report in reports:
        module.store_report(report)
    for latest in (False, True):
        rows = module.query_results(parameter='gain_dB', latest_only=latest)
        assert [row.conditions['silk_run_variables'] for row in rows] == [{'trim': 1, **extra}, {'trim': 3, **extra}]
        statuses = module.query_results(parameter='spec_Gain', latest_only=latest)
        assert len(statuses) == 2
        links = [row.conditions['silk_spec_measurement'] for row in statuses]
        assert len({link['measurement_id'] for link in links}) == 2
    module.store_report(reports[0])
    assert len(module.query_results(parameter='gain_dB', latest_only=True)) == 2
    assert len(module.query_results(parameter='spec_Gain', latest_only=True)) == 2
    before = dict(gain.measurements)
    with pytest.raises(EvidenceMismatchError, match='attribution|Variable'):
        module.restore_spec_measurements()
    assert gain.measurements == before and knob.value == 7
    module.close()


@pytest.mark.parametrize('aggregate', [False, True])
@pytest.mark.parametrize('ancestor', [False, True])
@pytest.mark.parametrize('same_spec', [False, True])
def test_swept_mixed_report_restores_only_numeric_measurements(pdk, tmp_path, aggregate, ancestor, same_spec):
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'mixed.db')
    unavailable = module.declare_spec('Unavailable', 'missing', '>=', 1.0, key='missing')
    version_id = module.create_circuit_version()
    context = _context()
    report = _scoped_report(gain, context)
    report.check(unavailable.name, unavailable.metric, unavailable.op, unavailable.target,
                 spec_key=unavailable.key, owner_path='$')
    report.parameters['silk_run_variables'] = {'trim': 3}
    module.store_report(report)
    if same_spec:
        missing = _scoped_report(gain, context, value=None)
        missing.parameters['silk_run_variables'] = {'trim': 5}
        module.store_report(missing)
    db = module._tracker.db
    version = db._snapshot_row(version_id)
    db.replace_net_current_set(version_id, corner=context.corner, temperature=context.temperature,
        voltage=context.voltage, structure_scheme=version['structure_scheme'],
        structure_hash=version['structure_hash'], context=context, rows=[_current_row(1e-3)])
    if ancestor:
        module.promote('stable', lock=False)
    gain.measurements.clear()
    if aggregate:
        module.restore_characterization_evidence(context=context)
    else:
        assert module.restore_spec_measurements() == 1
    assert [m.value for m in gain.measurements.values()] == [64.0]
    assert not unavailable.measurements
    statuses = module.query_results(parameter='spec_Unavailable')
    assert len(statuses) == 1 and statuses[0].value_text is not None
    module.close()


@pytest.mark.parametrize('ancestor', [False, True])
def test_swept_links_use_declaration_identity_with_duplicate_display_names(pdk, tmp_path, ancestor):
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'names.db')
    margin = module.declare_spec('Gain', 'margin_dB', '>=', 20.0, key='margin')
    origin = module.create_circuit_version()
    context = _context()
    report = _scoped_report(gain, context)
    report.metrics['margin_dB'] = 25.0
    report.check(margin.name, margin.metric, margin.op, margin.target,
                 spec_key=margin.key, owner_path='$')
    report.parameters['silk_run_variables'] = {'trim': 3}
    module.store_report(report)
    if ancestor:
        module.promote('stable', lock=False)
    gain.measurements.clear()
    margin.measurements.clear()
    assert module.restore_spec_measurements() == 2
    assert [m.value for m in gain.measurements.values()] == [64.0]
    assert [m.value for m in margin.measurements.values()] == [25.0]
    assert all(m.origin_version_id == origin
               for spec in (gain, margin) for m in spec.measurements.values())
    module.close()


def test_swept_restore_legitimate_context_filters_select_one_point(pdk, tmp_path):
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'filters.db')
    module.create_circuit_version()
    for trim, temperature, test_name in [(1, 27.0, 'nominal'), (3, 85.0, 'hot')]:
        context = _context(test_name=test_name, temperature=temperature)
        report = _scoped_report(gain, context, value=64.0 + trim)
        report.parameters['silk_run_variables'] = {'trim': trim}
        module.store_report(report)
    profiles = module.list_characterization_profiles()
    for profile in profiles:
        gain.measurements.clear()
        assert module.restore_spec_measurements(profile_id=profile.profile_id) == 1
        restored, = gain.measurements.values()
        assert restored.profile_id == profile.profile_id
    gain.measurements.clear()
    assert module.restore_spec_measurements(temperature=27.0) == 1
    assert next(iter(gain.measurements.values())).value == 65.0
    module.close()


@pytest.mark.parametrize('aggregate', [False, True])
def test_ordinary_measurement_survives_measurement_free_swept_status(pdk, tmp_path, aggregate):
    module, gain = _build_live(pdk, tmp_path, tmp_path / 'ordinary-and-swept.db')
    context = _context()
    _, _, current_set = _store_live_context(module, gain, context)
    report = _scoped_report(gain, context, value=None)
    report.parameters['silk_run_variables'] = {'trim': 5}
    module.store_report(report)
    assert module._tracker.db.conn.execute(
        'SELECT COUNT(*) FROM spec_measurements').fetchone()[0] == 1
    gain.measurements.clear()
    if aggregate:
        evidence = module.restore_characterization_evidence(context=context)
        assert evidence.current_set_id == current_set
    else:
        assert module.restore_spec_measurements() == 1
    measurement, = gain.measurements.values()
    assert measurement.value == 64.0
    statuses = module.query_results(parameter='spec_Gain')
    assert len(statuses) == 2
    assert sum('silk_run_variables' in (row.conditions or {}) for row in statuses) == 1
    module.close()
