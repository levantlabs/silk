"""Producing-launch deck checks through a built design, without a simulator."""
from dataclasses import replace
import gc
from pathlib import Path
import warnings

import pytest

from silk import Module, sources
from silk.elements import elements
from silk.exceptions import EvidenceMismatchError
from silk.signals.rawfile import read_rawfile
from silk.simulation import Transient, ConditionState
from silk.simulation.backend import NgspiceBackend, RawResult


@pytest.fixture(autouse=True)
def _retain_objects_from_prior_tests():
    # Keep pre-existing garbage alive until this test's warning assertions
    # have finished. In particular, SQLite connections abandoned by another
    # test must not emit ResourceWarnings inside our catch_warnings blocks.
    # Objects created by this test (including silk's API calls) are not held,
    # so warnings from resources they abandon remain visible to assertions.
    prior_objects = gc.get_objects()
    yield
    del prior_objects


@pytest.mark.parametrize('api', ['history', 'latest', 'selection', 'spec_restore', 'aggregate_restore'])
@pytest.mark.parametrize('state', ['unwired', 'missing_netlist', 'wired', 'stale'])
def test_fresh_rebuild_readback_preserves_bench_configuration(aggregate, tmp_path, api, state):
    original, _, _, profile = aggregate
    folder = tmp_path / 'fresh'
    db = original._tracker.db.db_path
    if state == 'unwired':
        fresh = Module(None, str(folder), 'dut.cir', None, module_name='dut', db_path=db)
        fresh.define_module_ports({'out': 'out', 'vss': 'ground'})
        fresh.add(elements.resistor('r', p='out', n='vss', value=1000))
        tb = fresh.create_test('gain')
        tb.add_metric('out_V', lambda c: float(c.load_signal('out').y[-1]))
    else:
        fresh, tb = build(folder, db)
        if state == 'missing_netlist':
            tb.netlist = None
        elif state == 'stale':
            tb.add(elements.capacitor('extra', p='out', n='0', value=1e-12))
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    fresh.set_temperature(27)
    fresh.set_supply_voltage(1, source='stim')
    try:
        assert fresh.create_circuit_version() == original._tracker.circuit_id
        assert not spec.measurements
        fresh.net_dict['out'].peak_current = 99
        netlist = tb.netlist
        added = tb._module_added
        modules = [fresh] + ([] if netlist is None else [netlist])
        before = _emission_state(modules)
        files = set(folder.rglob('*'))
        kwargs = dict(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
        calls = {
            'history': lambda: fresh.query_results(parameter='out_V'),
            'latest': lambda: fresh.query_results(parameter='out_V', latest_only=True),
            'selection': lambda: fresh.select_characterization_evidence(**kwargs),
            'spec_restore': lambda: fresh.restore_spec_measurements(profile_id=profile),
            'aggregate_restore': lambda: fresh.restore_characterization_evidence(**kwargs, force=True),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if state == 'stale' and api in {'latest', 'spec_restore', 'aggregate_restore'}:
                with pytest.raises(EvidenceMismatchError):
                    calls[api]()
                assert not spec.measurements
                assert fresh.net_dict['out'].peak_current == 99
            else:
                result = calls[api]()
                expected = ('not checked' if state in {'unwired', 'missing_netlist'} else
                            'stale' if state == 'stale' else 'checked')
                if api in {'history', 'latest'}:
                    assert result[0].deck_check.state == expected
                elif api in {'selection', 'aggregate_restore'}:
                    evidence = result if api == 'selection' else result.evidence
                    assert all(row.deck_check.state == expected for row in evidence.metrics)
                    assert evidence.specs[0].measurements[0].deck_check.state == expected
                else:
                    assert result == 1
                if api in {'spec_restore', 'aggregate_restore'}:
                    assert spec.measurements
                if api == 'aggregate_restore':
                    assert fresh.net_dict['out'].peak_current == 2
        warns = (state in {'unwired', 'missing_netlist'} or
                 api in {'selection', 'aggregate_restore'} and state != 'stale' or
                 api == 'selection' and state == 'stale')
        assert len(caught) == int(warns)
        if state in {'unwired', 'missing_netlist'}:
            assert 'registered but not wired in this session; wire it to check' in str(caught[0].message)
        assert tb.netlist is netlist and tb._module_added == added
        assert _emission_state(modules) == before
        assert set(folder.rglob('*')) == files
    finally:
        fresh.close()


def build(folder, db, dc=1):
    dut = Module(None, str(folder), 'dut.cir', None, module_name='dut', db_path=str(db))
    dut.define_module_ports({'out': 'out', 'vss': 'ground'})
    dut.add(elements.resistor('r', p='out', n='vss', value=1000))
    tb = dut.create_test('gain')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('stim', positive='out', negative='0', dc=dc))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('out_V', lambda c: float(c.load_signal('out').y[-1]))
    return dut, tb


@pytest.fixture
def launch(monkeypatch):
    # Replace external work only: backend.run retains data and completion gates.
    def fake(self, files, **kwargs):
        path = Path(files.rawfile_path)
        path.write_text('Title: test\nDate: today\nPlotname: Transient Analysis\n'
                        'Flags: real\nNo. Variables: 2\nNo. Points: 2\nVariables:\n'
                        '0\ttime\ttime\n1\tv(out)\tvoltage\n'
                        'Values:\n0\t0\n\t1\n1\t1e-8\n\t1\n')
        return RawResult(read_rawfile(str(path)), str(path), files), None, None
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', fake)


@pytest.fixture
def stored(tmp_path, launch):
    dut, tb = build(tmp_path / 'a', tmp_path / 'design.db')
    dut.create_circuit_version()
    dut.run_test('gain', store=True)
    yield dut, tb
    dut.close()


def test_changed_stimulus_flags_history_and_refuses_latest(stored):
    dut, tb = stored
    tb.netlist.sig_sources_dict['stim'].dc = 2
    row, = dut.query_results(parameter='out_V')
    assert row.value == 1
    assert row.deck_check.state == 'stale'
    with pytest.raises(EvidenceMismatchError, match='gain.*netlist.*run_test'):
        dut.query_results(parameter='out_V', latest_only=True)


def test_identical_rebuild_in_another_folder_checks(stored, tmp_path):
    dut, _ = stored
    other, _ = build(tmp_path / 'b', tmp_path / 'design.db')
    try:
        other.create_circuit_version()
        row, = other.query_results(parameter='out_V', latest_only=True)
        assert row.deck_check.state == 'checked'
        raw, = other._tracker.db.get_simulation_results(other._tracker.circuit_id)
        assert raw.deck_check is None
        assert row == raw
    finally:
        other.close()


def test_prepare_b_cannot_relabel_a(stored):
    dut, tb = stored
    report = tb.evaluate()
    before = report.provenance
    tb.netlist.sig_sources_dict['stim'].dc = 2
    tb.prepare()
    dut.store_report(report)
    assert report.provenance == before
    rows = dut.query_results(parameter='out_V')
    assert all(row.deck_check.state == 'stale' for row in rows)
    # Re-evaluation of the loaded A after B preparation also retains A.
    again = tb.evaluate()
    assert again.provenance.normalized_deck_sha256 == before.normalized_deck_sha256


def test_manual_rows_warn_once_and_raw_history_does_not(stored):
    dut, _ = stored
    dut.store_result('manual_a', 3, 'V')
    dut.store_result('manual_b', 4, 'V')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rows = dut.query_results()
    assert len(caught) == 1
    assert all(r.deck_check.state == 'not checked' for r in rows if r.parameter.startswith('manual'))
    with warnings.catch_warnings(record=True) as caught:
        dut._tracker.db.get_simulation_results(dut._tracker.circuit_id)
    assert not caught


def test_old_and_new_digests_reuse_emission_not_verdict(stored, monkeypatch):
    dut, tb = stored
    tb.netlist.sig_sources_dict['stim'].dc = 2
    dut.run_test('gain', store=True)
    from silk.circuit import modules_netlist
    real = modules_netlist._emit_deck_text
    count = []
    def emit(*args, **kwargs):
        count.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(modules_netlist, '_emit_deck_text', emit)
    rows = dut.query_results(parameter='out_V')
    assert [r.deck_check.state for r in rows] == ['stale', 'checked']
    assert len(count) == 1
    dut.query_results(parameter='out_V', latest_only=True)
    assert len(count) == 2
    tb.netlist.sig_sources_dict['stim'].dc = 3
    with pytest.raises(EvidenceMismatchError):
        dut.query_results(parameter='out_V', latest_only=True)
    assert len(count) == 3


def test_missing_test_is_usable_not_checked(stored):
    dut, _ = stored
    del dut._tests['gain']
    with pytest.warns(UserWarning, match='not checked.*gain'):
        row, = dut.query_results(parameter='out_V', latest_only=True)
    assert row.deck_check.state == 'not checked'


def test_provenance_retains_launch_label_when_stored_corner_is_none(stored):
    dut, _ = stored
    row, = dut._tracker.db.get_simulation_results(dut._tracker.circuit_id)
    record = dut._tracker.db.get_run_provenance(row.provenance_id)
    assert record.launch_corner_label == 'tt'
    assert record.producing_test == {'owner_path': '$', 'test_name': 'gain'}
    assert record.normalized_deck_sha256.startswith('silk.normalized-deck/v1:')


def test_spec_restore_stale_is_atomic(tmp_path, launch):
    dut, tb = build(tmp_path, tmp_path / 'spec.db')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    try:
        dut.create_circuit_version()
        dut.run_test('gain', store=True)
        spec.measurements.clear()
        tb.netlist.sig_sources_dict['stim'].dc = 2
        with pytest.raises(EvidenceMismatchError, match='gain'):
            dut.restore_spec_measurements()
        assert not spec.measurements
    finally:
        dut.close()


def test_queries_leave_sources_settings_netlister_and_evidence_untouched(stored):
    dut, tb = stored
    module = tb.netlist
    source = module.sig_sources_dict['stim']
    before = dict(vars(module.netlister))
    options = dict(module.simulator_options)
    original = source.dc
    dut.set_supply_voltage(2, source='stim')
    row, = dut.query_results(parameter='out_V')
    assert row.deck_check.state == 'stale'
    assert module.sig_sources_dict['stim'] is source
    assert source.dc == original
    assert module.simulator_options == options
    assert vars(module.netlister) == before
    # A malformed current supply is a typed error, not unknown history.
    dut.set_supply_voltage(2, source='missing')
    from silk.exceptions import SimulationConfigurationError
    with pytest.raises(SimulationConfigurationError):
        dut.query_results(parameter='out_V')
    assert module.sig_sources_dict['stim'] is source and source.dc == original


@pytest.mark.parametrize('kind', ['manual', 'legacy', 'external', 'scheme'])
def test_not_checked_launch_facts_never_borrow_current_preparation(stored, kind):
    dut, _ = stored
    row, = dut._tracker.db.get_simulation_results(dut._tracker.circuit_id)
    provenance = dut._tracker.db.get_run_provenance(row.provenance_id)
    changes = {
        'manual': {'normalized_deck_sha256': None, 'producing_test': None},
        'legacy': {'scheme': 'silk.provenance/v2', 'normalized_deck_sha256': None},
        'external': {'run_origin': 'external-rawfile'},
        'scheme': {'normalized_deck_sha256': 'future-deck/v100:' + '0' * 64},
    }
    altered = replace(provenance, **changes[kind])
    provenance_id = dut._tracker.db.store_run_provenance(altered)
    dut._tracker.db.add_simulation_result(dut._tracker.circuit_id, 'other', value=9,
                                         source='spice', provenance_id=provenance_id)
    with pytest.warns(UserWarning) as caught:
        other, = dut.query_results(parameter='other', latest_only=True)
    assert len(caught) == 1
    assert other.deck_check.state == 'not checked'


@pytest.mark.parametrize('alter', ['custom', 'stochastic', 'mc', 'behavioral'])
def test_excluded_current_bench_is_not_checked(stored, alter):
    dut, tb = stored
    if alter == 'custom':
        tb.netlist.add_custom_line('* opaque comment')
    elif alter == 'stochastic':
        tb.netlist.add_source(sources.BehavioralVoltage('random', 'noise', '0', 'gauss(1)'))
    elif alter == 'mc':
        tb.set_monte_carlo(2, seed=1)
    else:
        dut.simulator = 'behavioral'
    with pytest.warns(UserWarning):
        row, = dut.query_results(parameter='out_V', latest_only=True)
    assert row.deck_check.state == 'not checked'


def test_many_metric_and_spec_rows_emit_once(tmp_path, launch, monkeypatch):
    dut, tb = build(tmp_path, tmp_path / 'many.db')
    for index in range(6):
        tb.add_metric(f'copy{index}', lambda c: 1.)
        tb.add_spec(f'limit{index}', f'copy{index}', '>=', .5)
    try:
        dut.create_circuit_version()
        dut.run_test('gain', store=True)
        from silk.circuit import modules_netlist
        emit = modules_netlist._emit_deck_text
        calls = []
        monkeypatch.setattr(modules_netlist, '_emit_deck_text',
                            lambda *a, **kw: (calls.append(1), emit(*a, **kw))[1])
        assert all(r.deck_check.state == 'checked' for r in dut.query_results())
        assert len(calls) == 1
        assert dut.restore_spec_measurements() == 6
        assert len(calls) == 2
    finally:
        dut.close()


def test_failed_launch_cannot_publish_digest(stored, monkeypatch):
    dut, tb = stored
    from silk.exceptions import SimulationAborted
    def abort(*args, **kwargs):
        raise SimulationAborted('incomplete synthetic launch', simulator='ngspice')
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', abort)
    with pytest.raises(SimulationAborted):
        tb.run()
    from silk.simulation.provenance import from_module
    assert from_module(tb.netlist).normalized_deck_sha256 is None
    assert tb.netlist._launch_record is None


def test_accepted_op_is_checked_without_completion_plan(tmp_path, launch):
    from silk.simulation.analysis import Op
    dut, tb = build(tmp_path, tmp_path / 'op.db')
    tb.netlist.sim_analysis_dict.clear()
    tb.add_sim_type(Op())
    try:
        dut.create_circuit_version()
        dut.run_test('gain', store=True)
        assert tb.netlist._run_files.completion is None
        row, = dut.query_results(parameter='out_V')
        assert row.deck_check.state == 'checked'
    finally:
        dut.close()


@pytest.fixture
def aggregate(tmp_path, launch, monkeypatch):
    monkeypatch.setattr(NgspiceBackend, 'read_ambient_temperatures', lambda *a: (27.,))
    dut, tb = build(tmp_path, tmp_path / 'aggregate.db')
    dut.set_temperature(27)
    dut.set_supply_voltage(1, source='stim')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    dut.create_circuit_version()
    report = dut.run_test('gain', store=True)
    tracker = dut._tracker
    profile = tracker.db.resolve_characterization_profile(tracker.circuit_id,
                                                          context=report.characterization_context)
    identity = tracker.compute_snapshot_identity_for(dut)
    tracker.db.replace_net_current_set(tracker.circuit_id,
        corner=report.characterization_context.corner,
        temperature=27, voltage=1, structure_scheme=identity.structure.scheme,
        structure_hash=identity.structure.digest, context=report.characterization_context,
        rows=[dict(net_name='out', pin_id='', avg_a=1., peak_a=2., rms_a=1.,
                   source='spice', current_vector=None)])
    yield dut, tb, spec, profile.profile_id
    dut.close()


def test_aggregate_members_check_but_currents_always_warn(aggregate, monkeypatch):
    dut, tb, spec, profile = aggregate
    from silk.circuit import modules_netlist
    emit = modules_netlist._emit_deck_text
    calls = []
    monkeypatch.setattr(modules_netlist, '_emit_deck_text',
                        lambda *a, **kw: (calls.append(1), emit(*a, **kw))[1])
    with pytest.warns(UserWarning, match='current annotations') as caught:
        selected = dut.select_characterization_evidence(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
    assert len(caught) == 1
    assert selected.deck_check.state == 'not checked'
    assert all(row.deck_check.state == 'checked' for row in selected.metrics)
    assert selected.specs[0].measurements[0].deck_check.state == 'checked'
    assert selected.current_set.deck_check.state == 'not checked'
    raw = dut._tracker.db.select_characterization_evidence(dut._tracker.circuit_id, profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
    assert selected == raw
    assert len(calls) == 1
    with pytest.warns(UserWarning, match='current annotations') as caught:
        restored = dut.restore_characterization_evidence(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
    assert len(caught) == 1
    assert restored.measurement_count == 1 and restored.current_count == 1
    assert len(calls) == 2


def test_stale_aggregate_returns_history_but_restore_is_atomic(aggregate):
    dut, tb, spec, profile = aggregate
    dut.set_supply_voltage(2, source='stim')
    with pytest.warns(UserWarning) as caught:
        selected = dut.select_characterization_evidence(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
    assert len(caught) == 1
    assert selected.deck_check.state == 'stale'
    spec.measurements.clear()
    dut.net_dict['out'].peak_current = 99
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(EvidenceMismatchError, match='gain'):
            dut.restore_characterization_evidence(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1, force=True)
    assert not caught
    assert not spec.measurements
    assert dut.net_dict['out'].peak_current == 99


def test_unknown_spec_restore_is_usable_with_one_warning(aggregate):
    dut, tb, spec, profile = aggregate
    # Simulate old evidence without provenance: preserve all row identities.
    dut._tracker.db.conn.execute('UPDATE spec_measurements SET provenance_id = NULL')
    spec.measurements.clear()
    with pytest.warns(UserWarning, match='not checked') as caught:
        assert dut.restore_spec_measurements(profile_id=profile) == 1
    assert len(caught) == 1
    assert any(m.value == 1 and m.passes for m in spec.measurements.values())


def test_exact_placements_and_launch_corners_emit_separately(tmp_path, launch, monkeypatch):
    child, tb = build(tmp_path / 'child', tmp_path / 'unused.db')
    top = Module(None, str(tmp_path / 'root'), 'root.cir', None, db_path=str(tmp_path / 'root.db'))
    top.add_module(child, 'a', out='a', vss='0')
    top.add_module(child, 'b', out='b', vss='0')
    try:
        top.create_circuit_version()
        for instance in ('a', 'b'):
            for corner in ('tt', 'ss'):
                top.store_report(top.get_test('gain', instance=instance).run(corner=corner))
        from silk.circuit import modules_netlist
        emit = modules_netlist._emit_deck_text
        calls = []
        monkeypatch.setattr(modules_netlist, '_emit_deck_text',
                            lambda *a, **kw: (calls.append(1), emit(*a, **kw))[1])
        with warnings.catch_warnings(record=True) as caught:
            rows = top.query_results(parameter='out_V')
        assert len(rows) == 4
        assert all(r.deck_check.state == 'checked' for r in rows)
        assert len(calls) == 4
        assert not caught
        # Lookup must not register or prune selected evidence.
        from silk.circuit.test_placement import evidence_index
        index = evidence_index(top)
        before = (dict(index.records), dict(index.selectors), dict(index.annotations))
        top.query_results(parameter='out_V')
        assert (index.records, index.selectors, index.annotations) == before
        # A standalone child cannot stand in for its missing root producer.
        child.create_circuit_version()
        for row in rows:
            record = top._tracker.db.get_run_provenance(row.provenance_id)
            pid = child._tracker.db.store_run_provenance(record)
            child._tracker.db.add_simulation_result(child._tracker.circuit_id, 'other', value=1,
                                                    source='spice', provenance_id=pid)
        with pytest.warns(UserWarning):
            unknown = child.query_results(parameter='other')
        assert all(r.deck_check.state == 'not checked' for r in unknown)
    finally:
        top.close()
        child.close()


def test_checked_facts_never_leak_into_silk_exports(stored, tmp_path):
    from silk.db import DeckCheck, CircuitExporter
    dut, _ = stored
    row, = dut.query_results(parameter='out_V')
    assert isinstance(row.deck_check, DeckCheck)
    exporter = CircuitExporter(dut._tracker.db)
    json_path, csv_path = tmp_path / 'export.json', tmp_path / 'export.csv'
    exporter.export_to_json('dut', str(json_path), include_results=True)
    exporter.export_results_to_csv('dut', str(csv_path))
    assert 'deck_check' not in json_path.read_text()
    assert 'deck_check' not in csv_path.read_text()
    provenance = dut._tracker.db.get_run_provenance(row.provenance_id)
    assert 'deck_check' not in provenance.to_dict()


def test_matching_digest_does_not_hide_backend_change(stored):
    dut, _ = stored
    row, = dut._tracker.db.get_simulation_results(dut._tracker.circuit_id)
    provenance = dut._tracker.db.get_run_provenance(row.provenance_id)
    # Even a coincidentally equal deck cannot certify another backend.
    pid = dut._tracker.db.store_run_provenance(replace(provenance, simulator='xyce'))
    dut._tracker.db.add_simulation_result(dut._tracker.circuit_id, 'other', value=1,
                                         source='spice', provenance_id=pid)
    flagged, = dut.query_results(parameter='other')
    assert flagged.deck_check.state == 'stale'


def test_dut_owned_source_changes_still_refuse_identity(stored):
    dut, _ = stored
    dut.add_source(sources.VoltageSource('bias', positive='out', negative='vss', dc=2))
    with pytest.raises(EvidenceMismatchError):
        dut.query_results(parameter='out_V')


def test_lookup_does_not_mutate_dut_netlister(stored):
    dut, _ = stored
    # A caller's emission state must survive a lookup of its enclosing bench.
    dut.netlister._terminal_current_method = 'caller-state'
    dut.netlister._ammeter_wrappers = {'caller': 'value'}
    before = dict(vars(dut.netlister))
    dut.query_results(parameter='out_V')
    assert vars(dut.netlister) == before


def test_invalid_current_backend_capture_preserves_typed_error(stored):
    dut, _ = stored
    from silk.exceptions import NotSupportedError
    dut.simulator = 'xyce'
    dut.save_device_params = True
    with pytest.raises(NotSupportedError, match='requires Op'):
        dut.query_results(parameter='out_V', latest_only=True)


def test_stale_restore_suppresses_emission_advisories(stored):
    dut, _ = stored
    dut.save_device_params = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        dut.query_results(parameter='out_V')
    assert not caught  # This is flagged history, not an explicit generation.


def test_stale_spec_restore_has_no_advisory_or_partial_state(tmp_path, launch):
    dut, tb = build(tmp_path, tmp_path / 'quiet.db')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    try:
        dut.create_circuit_version()
        dut.run_test('gain', store=True)
        dut.save_device_params = True
        tb.netlist.sig_sources_dict['stim'].dc = 2
        spec.measurements.clear()
        before = dict(vars(dut.netlister))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(EvidenceMismatchError):
                dut.restore_spec_measurements()
        assert not caught
        assert not spec.measurements
        assert vars(dut.netlister) == before
    finally:
        dut.close()


def test_readonly_enumeration_leaves_native_graph_registries_untouched(stored):
    dut, _ = stored
    from silk.db._deck_check import CheckContext
    graph = dut._element_mgr
    mapping = graph.module_dict
    # Direct enumeration must not replace/rebuild caller-owned registry entries.
    mapping['sentinel'] = {'device': dut, 'numberInstantiated': 0}
    before = list(mapping.items())
    try:
        CheckContext(dut._tracker)
        assert graph.module_dict is mapping
        assert list(mapping.items()) == before
    finally:
        del mapping['sentinel']


def test_invalid_sweep_source_is_a_request_error(stored):
    dut, tb = stored
    from silk.simulation import DC
    from silk.exceptions import SweepSourceError
    tb.netlist.sim_analysis_dict.clear()
    tb.add_sim_type(DC(source='missing', start=0, stop=1, step=.1))
    with pytest.raises(SweepSourceError, match='missing'):
        dut.query_results(parameter='out_V')


def test_undeclared_device_probe_requires_no_preflight_at_lookup(tmp_path, pdk, launch, monkeypatch):
    dut = Module('sky130', str(tmp_path), 'device.cir', pdk_details=pdk,
                 module_name='device', db_path=str(tmp_path / 'device.db'))
    dut.define_module_ports({'out': 'out', 'vss': 'ground'})
    dut.add(pdk.nmos1p8('m1', d='out', g='out', s='vss', b='vss', w=1, l=.35))
    tb = dut.create_test('gain')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('stim', positive='out', negative='0', dc=1))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('out_V', lambda c: 1.)
    try:
        dut.create_circuit_version()
        dut.run_test('gain', store=True)
        # This parameter is outside the catalog's declared OP data. Actual
        # prepare would refuse an auxiliary SPICE solve for it.
        tb.probe('m1:cgso')
        from silk.circuit import spice_netlister
        def forbid(*args, **kwargs):
            raise AssertionError('lookup launched a probe preflight')
        monkeypatch.setattr(spice_netlister.PySpiceNetlister, '_ngspice_param_preflight', forbid)
        with pytest.warns(UserWarning, match='preflight'):
            row, = dut.query_results(parameter='out_V', latest_only=True)
        assert row.deck_check.state == 'not checked'
    finally:
        dut.close()


def test_external_rawfile_evaluation_never_borrows_prepared_digest(stored):
    dut, tb = stored
    report = tb.evaluate(rawfile=tb.results_file)
    assert report.provenance.run_origin == 'external-rawfile'
    assert report.provenance.normalized_deck_sha256 is None
    dut.store_report(report)
    with pytest.warns(UserWarning) as caught:
        rows = dut.query_results(parameter='out_V', latest_only=True)
    # External conditions form a distinct latest bucket from the silk run.
    assert len(caught) == 1
    assert sorted(row.deck_check.state for row in rows) == ['checked', 'not checked']


def test_incomplete_data_fails_gate_and_publishes_no_identity(stored, monkeypatch):
    dut, tb = stored
    launch = NgspiceBackend._launch_prepared
    def short(self, files, **kwargs):
        result = launch(self, files, **kwargs)
        raw = result[0]
        raw.native.plots[0].vectors[0].data[-1] = 2e-9
        return result
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', short)
    from silk.exceptions import SimulationAborted
    with pytest.raises(SimulationAborted):
        tb.run()
    from silk.simulation.provenance import from_module
    assert from_module(tb.netlist).normalized_deck_sha256 is None


def test_stale_rerun_command_uses_recorded_launch_label(stored):
    dut, tb = stored
    dut.run_test('gain', corner='ss', store=True)
    tb.netlist.sig_sources_dict['stim'].dc = 2
    with pytest.raises(EvidenceMismatchError, match="corner='ss'"):
        dut.query_results(parameter='out_V', latest_only=True)


def test_failed_temperature_assertion_preserves_unapplied_supply(stored):
    dut, tb = stored
    dut.set_supply_voltage(2, source='stim')
    source = tb.netlist.sig_sources_dict['stim']
    original = source.dc
    from silk.exceptions import CharacterizationContextError
    # TEMP has not been independently configured, so this assertion refuses.
    with pytest.warns(DeprecationWarning, match='deprecated'):
        with pytest.raises(CharacterizationContextError):
            tb.run(temperature=27)
    assert source.dc == original


def test_producing_backend_is_launch_backend_not_later_selection(stored, monkeypatch):
    dut, tb = stored
    launch = NgspiceBackend._launch_prepared
    def switch_after_launch(self, files, **kwargs):
        result = launch(self, files, **kwargs)
        tb.netlist.simulator = 'xyce'
        return result
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', switch_after_launch)
    report = tb.run()
    assert report.provenance.simulator == 'ngspice'


def test_uncheckable_launch_retains_logical_producer_for_warning(stored):
    dut, tb = stored
    tb.netlist.add_source(sources.BehavioralVoltage('random', 'noise', '0', 'gauss(1)'))
    report = dut.run_test('gain', store=True)
    assert report.provenance.normalized_deck_sha256 is None
    assert report.provenance.producing_test == {'owner_path': '$', 'test_name': 'gain'}
    with pytest.warns(UserWarning, match='gain'):
        rows = dut.query_results(parameter='out_V', latest_only=True)
    assert all(row.deck_check.state == 'not checked' for row in rows)


def _stored_bias_pair(tmp_path, monkeypatch, *, variants=False, sine=False):
    monkeypatch.setattr(NgspiceBackend, 'read_ambient_temperatures', lambda *a: (27.,))
    dut, tb = build(tmp_path / 'dut', tmp_path / 'bias.db')
    helpers = []
    for index in range(2):
        helper = Module(None, str(tmp_path / f'bias{index}'), 'bias.cir', None,
                        module_name='bias', variants=variants)
        helper.define_module_ports({'n': 'out', 'vss': 'ground'})
        stimulus = (dict(sine=sources.Sine(0, 1, 1e3)) if sine else
                    dict(dc=1 + index if variants else 1))
        helper.add_source(sources.VoltageSource('vb', positive='n', negative='vss',
                                                **stimulus))
        helper.add(elements.resistor('rl', p='n', n='vss', value=1000))
        tb.add_module(helper, f'bias{index}', n=f'nb{index}', vss='0')
        helpers.append(helper)
    dut.set_temperature(27)
    dut.set_supply_voltage(1, source='stim')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    dut.create_circuit_version()
    report = dut.run_test('gain', store=True)
    tracker = dut._tracker
    profile = tracker.db.resolve_characterization_profile(tracker.circuit_id,
                                                           context=report.characterization_context)
    identity = tracker.compute_snapshot_identity_for(dut)
    tracker.db.replace_net_current_set(tracker.circuit_id,
        corner=report.characterization_context.corner, temperature=27, voltage=1,
        structure_scheme=identity.structure.scheme, structure_hash=identity.structure.digest,
        context=report.characterization_context,
        rows=[dict(net_name='out', pin_id='', avg_a=1., peak_a=2., rms_a=1.,
                   source='spice', current_vector=None)])
    return dut, tb, spec, profile.profile_id, helpers


def _emission_state(modules):
    from copy import deepcopy
    # Read the native mappings directly; public properties can refresh them.
    return [(deepcopy(vars(module.netlister)),
             {name: {key: (set(value) if isinstance(value, set) else value)
                     for key, value in entry.items()}
              for name, entry in module._element_mgr.module_dict.items()},
             {name: entry.get('module_name') for name, entry in module._element_mgr.elem_dict.items()})
            for module in modules]


@pytest.mark.parametrize('api', ['history', 'latest', 'selection', 'spec_restore', 'aggregate_restore'])
def test_sibling_definition_collision_refuses_all_checked_reads_without_mutation(tmp_path, launch, monkeypatch, api):
    from silk.exceptions import CircuitError
    dut, tb, spec, profile, helpers = _stored_bias_pair(tmp_path, monkeypatch)
    try:
        modules = [dut, tb.netlist, *helpers]
        before = _emission_state(modules)
        row, = dut.query_results(parameter='out_V', latest_only=True)
        assert row.deck_check.state == 'checked'  # Identical bodies are valid.
        assert _emission_state(modules) == before
        identity = dut._tracker.compute_snapshot_identity_for(dut)
        helpers[1].sig_sources_dict['vb'].dc = 2
        assert dut._tracker.compute_snapshot_identity_for(dut) == identity
        with pytest.raises(CircuitError, match='different.*circuits'):
            tb.prepare()
        spec.measurements.clear()
        dut.net_dict['out'].peak_current = 99
        before = _emission_state(modules)
        aggregate_kwargs = dict(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
        calls = {
            'history': lambda: dut.query_results(parameter='out_V'),
            'latest': lambda: dut.query_results(parameter='out_V', latest_only=True),
            'selection': lambda: dut.select_characterization_evidence(**aggregate_kwargs),
            'spec_restore': lambda: dut.restore_spec_measurements(profile_id=profile),
            'aggregate_restore': lambda: dut.restore_characterization_evidence(**aggregate_kwargs, force=True),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(CircuitError, match='different.*circuits'):
                calls[api]()
        assert not caught
        assert not spec.measurements
        assert dut.net_dict['out'].peak_current == 99
        assert _emission_state(modules) == before
    finally:
        dut.close()


def _custom_bias_waves(helpers, renderer):
    from silk.sources.waveforms import Sine

    class CustomSine(Sine):
        def to_spice(self):
            self.calls += 1
            if renderer == 'raising':
                raise RuntimeError('custom renderer must not be called')
            return super().to_spice()

    waves = []
    for helper in helpers:
        wave = CustomSine(0, 1, 1e3)
        wave.calls = 0
        helper.sig_sources_dict['vb'].sine = wave
        waves.append(wave)
    return waves


@pytest.mark.parametrize('override,callback', [
    (override, callback) for override in ('class', 'instance')
    for callback in ('counting', 'raising', 'rendering')
] + [('inherited', 'counting')])
def test_custom_source_selector_normalization_never_calls_it(tmp_path, launch, monkeypatch, override, callback):
    from silk.circuit.test_execution import normalized_deck_text
    from silk.db._deck_check import _emission_view
    from tests._selector_control import install_selector
    dut, tb, _, _, helpers = _stored_bias_pair(tmp_path, monkeypatch, sine=True)
    try:
        wave = install_selector(helpers[0].sig_sources_dict['vb'], override, callback)
        if override == 'inherited':
            install_selector(helpers[1].sig_sources_dict['vb'], override, callback)
        before = _emission_state([dut, tb.netlist, *helpers])
        deck, reason = normalized_deck_text(_emission_view(tb.netlist), 'tt')
        if override == 'inherited':
            assert deck is not None and reason is None
        else:
            assert deck is None and 'custom waveform selector' in reason
        assert (wave.selector_calls, wave.renderer_calls, wave.vampl) == (0, 0, 1)
        assert _emission_state([dut, tb.netlist, *helpers]) == before
    finally:
        dut.close()


@pytest.mark.parametrize('override', ['class', 'instance'])
@pytest.mark.parametrize('callback', ['counting', 'raising', 'rendering'])
@pytest.mark.parametrize('api', ['history', 'latest', 'selection', 'spec_restore', 'aggregate_restore'])
def test_custom_source_selector_public_calls_use_warned_history(tmp_path, launch, monkeypatch, override, callback, api):
    from tests._selector_control import install_selector
    dut, tb, spec, profile, helpers = _stored_bias_pair(tmp_path, monkeypatch, sine=True)
    try:
        row, = dut.query_results(parameter='out_V', latest_only=True)
        assert row.deck_check.state == 'checked'
        wave = install_selector(helpers[0].sig_sources_dict['vb'], override, callback)
        before = _emission_state([dut, tb.netlist, *helpers])
        spec.measurements.clear()
        kwargs = dict(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
        calls = {
            'history': lambda: dut.query_results(parameter='out_V'),
            'latest': lambda: dut.query_results(parameter='out_V', latest_only=True),
            'selection': lambda: dut.select_characterization_evidence(**kwargs),
            'spec_restore': lambda: dut.restore_spec_measurements(profile_id=profile),
            'aggregate_restore': lambda: dut.restore_characterization_evidence(**kwargs, force=True),
        }
        with pytest.warns(UserWarning, match='not checked') as caught:
            result = calls[api]()
        assert len(caught) == 1
        if api in {'history', 'latest'}:
            assert all(row.deck_check.state == 'not checked' for row in result)
        elif api in {'selection', 'aggregate_restore'}:
            evidence = result if api == 'selection' else result.evidence
            assert all(row.deck_check.state == 'not checked' for row in evidence.metrics)
            assert evidence.specs[0].measurements[0].deck_check.state == 'not checked'
        else:
            assert result == 1
        if api in {'spec_restore', 'aggregate_restore'}:
            assert spec.measurements
        assert (wave.selector_calls, wave.renderer_calls, wave.vampl) == (0, 0, 1)
        assert _emission_state([dut, tb.netlist, *helpers]) == before
    finally:
        dut.close()


@pytest.mark.parametrize('renderer', ['counting', 'raising'])
def test_opaque_bias_normalization_declines_before_definition_rendering(tmp_path, launch, monkeypatch, renderer):
    from silk.circuit.test_execution import normalized_deck_text
    from silk.db._deck_check import _emission_view
    dut, tb, _, _, helpers = _stored_bias_pair(tmp_path, monkeypatch, sine=True)
    try:
        waves = _custom_bias_waves(helpers, renderer)
        before = _emission_state([dut, tb.netlist, *helpers])
        deck, reason = normalized_deck_text(_emission_view(tb.netlist), 'tt')
        assert deck is None
        assert 'custom waveform' in reason
        assert [wave.calls for wave in waves] == [0, 0]
        assert _emission_state([dut, tb.netlist, *helpers]) == before
    finally:
        dut.close()


@pytest.mark.parametrize('renderer', ['counting', 'raising'])
@pytest.mark.parametrize('api', ['history', 'latest', 'selection', 'spec_restore', 'aggregate_restore'])
def test_opaque_bias_public_lookups_warn_without_rendering(tmp_path, launch, monkeypatch, renderer, api):
    dut, tb, spec, profile, helpers = _stored_bias_pair(tmp_path, monkeypatch, sine=True)
    try:
        row, = dut.query_results(parameter='out_V', latest_only=True)
        assert row.deck_check.state == 'checked'
        waves = _custom_bias_waves(helpers, renderer)
        before = _emission_state([dut, tb.netlist, *helpers])
        spec.measurements.clear()
        kwargs = dict(profile_id=profile, corner=ConditionState.NONE, temperature=27, voltage=1)
        calls = {
            'history': lambda: dut.query_results(parameter='out_V'),
            'latest': lambda: dut.query_results(parameter='out_V', latest_only=True),
            'selection': lambda: dut.select_characterization_evidence(**kwargs),
            'spec_restore': lambda: dut.restore_spec_measurements(profile_id=profile),
            'aggregate_restore': lambda: dut.restore_characterization_evidence(**kwargs, force=True),
        }
        with pytest.warns(UserWarning, match='not checked') as caught:
            result = calls[api]()
        assert len(caught) == 1
        if api in {'history', 'latest'}:
            assert all(row.deck_check.state == 'not checked' for row in result)
        elif api in {'selection', 'aggregate_restore'}:
            evidence = result if api == 'selection' else result.evidence
            assert evidence.deck_check.state == 'not checked'
            assert evidence.specs[0].measurements[0].deck_check.state == 'not checked'
        else:
            assert result == 1
        if api in {'spec_restore', 'aggregate_restore'}:
            assert spec.measurements
        assert [wave.calls for wave in waves] == [0, 0]
        assert _emission_state([dut, tb.netlist, *helpers]) == before
    finally:
        dut.close()


def test_variant_bias_definitions_keep_valid_normalized_bytes_and_isolation(tmp_path, launch, monkeypatch):
    from silk.circuit.test_execution import normalized_deck_digest
    from silk.db._deck_check import _emission_view
    dut, tb, _, _, helpers = _stored_bias_pair(tmp_path, monkeypatch, variants=True)
    try:
        before = _emission_state([dut, tb.netlist, *helpers])
        row, = dut.query_results(parameter='out_V', latest_only=True)
        assert row.deck_check.state == 'checked'
        assert _emission_state([dut, tb.netlist, *helpers]) == before
        tb.prepare()
        digest, reason = normalized_deck_digest(_emission_view(tb.netlist), 'tt')
        assert reason is None
        assert digest == tb.netlist._prepared_deck_record['normalized_deck_sha256']
        helpers[1].sig_sources_dict['vb'].dc = 3
        tb.prepare()  # Distinct variant bodies are valid, not a collision.
        before = _emission_state([dut, tb.netlist, *helpers])
        row, = dut.query_results(parameter='out_V')
        assert row.deck_check.state == 'stale'
        assert _emission_state([dut, tb.netlist, *helpers]) == before
    finally:
        dut.close()


@pytest.mark.parametrize('temps,volts', [([27, 85], None), (None, [.8, 1.8]),
                                       ([27, 85], [.8, 1.8]), ([85], [1.8])])
def test_current_grid_rebuild_checks_and_restores(tmp_path, grid_launch, temps, volts):
    db = tmp_path / 'grid.db'
    def configure(folder):
        dut, tb = build(folder, db)
        spec = tb.add_spec('level', 'out_V', '>=', .5)
        if temps is not None:
            dut.set_temperatures(temps)
        if volts is not None:
            dut.set_supply_voltages(volts, source='stim')
        else:
            dut.set_supply_voltage(1, source='stim')
        dut.create_circuit_version()
        return dut, tb, spec
    dut, tb, _ = configure(tmp_path / 'original')
    try:
        reports = tb.run()
        for report in reports.values():
            dut.store_report(report)
        fresh, ftb, spec = configure(tmp_path / 'rebuilt')
        try:
            before = _emission_state([fresh, ftb.netlist])
            files = set(tmp_path.rglob('*'))
            rows = fresh.query_results(parameter='out_V', latest_only=True)
            assert len(rows) == len(list(reports.values()))
            assert all(row.deck_check.state == 'checked' for row in rows)
            for row in rows:
                selectors = dict(profile_id=row.profile_id, temperature=row.temperature,
                                 voltage=row.voltage)
                selected = fresh.query_results(parameter='out_V', latest_only=True, **selectors)
                assert len(selected) == 1 and selected[0] == row
                spec.measurements.clear()
                assert fresh.restore_spec_measurements(**selectors) == 1
                measurement, = spec.measurements.values()
                assert measurement.value == row.value
                assert measurement.temperature == row.temperature
                assert measurement.voltage == row.voltage
                assert measurement.profile_id == row.profile_id
            spec.measurements.clear()
            assert fresh.restore_spec_measurements() == len(list(reports.values()))
            assert spec.measurements
            assert _emission_state([fresh, ftb.netlist]) == before
            assert set(tmp_path.rglob('*')) == files
        finally:
            fresh.close()
    finally:
        dut.close()


@pytest.fixture
def grid_launch(launch, monkeypatch):
    def ambient(self, raw, log):
        token = raw.run_files.directory.split('temperature-')[1].split('/')[0]
        return (27. if token == 'unset' else float(token),)
    monkeypatch.setattr(NgspiceBackend, 'read_ambient_temperatures', ambient)


def _store_grid(dut, tb):
    dut.create_circuit_version()
    reports = tb.run()
    assert not reports.failed
    for report in reports.values():
        dut.store_report(report)
    return reports


def test_grid_point_membership_cache_and_atomic_restore(tmp_path, grid_launch, monkeypatch):
    dut, tb = build(tmp_path, tmp_path / 'membership.db')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    tb.set_temperatures([27, 85]).set_supply_voltages([.8, 1.8], source='stim')
    try:
        _store_grid(dut, tb)
        from silk.circuit import modules_netlist
        emit = modules_netlist._emit_deck_text
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return emit(*args, **kwargs)
        monkeypatch.setattr(modules_netlist, '_emit_deck_text', counted)
        tb.set_temperatures([85, 27, 100])
        assert all(r.deck_check.state == 'checked' for r in dut.query_results())
        assert len(calls) == 6  # Metric and spec-status rows share each point emission.
        tb.set_temperatures([27])
        rows = dut.query_results(parameter='out_V')
        assert [r.deck_check.state for r in rows] == ['checked', 'checked', 'stale', 'stale']
        spec.measurements.clear()
        with pytest.raises(EvidenceMismatchError, match='for report in .*store_report'):
            dut.restore_spec_measurements()
        assert not spec.measurements
        with pytest.raises(EvidenceMismatchError):
            dut.query_results(parameter='out_V', latest_only=True)
        assert dut.restore_spec_measurements(temperature=27) == 2
        # A remaining configuration emitting the same deck can certify a row,
        # independent of the old point-list identity.
        tb.set_temperatures(None).set_temperature(27)
        assert all(r.deck_check.state == 'checked'
                   for r in dut.query_results(parameter='out_V', temperature=27))
    finally:
        dut.close()


@pytest.mark.parametrize('change', ['temperature', 'supply', 'source', 'analysis', 'option', 'backend', 'circuit'])
def test_grid_real_deck_changes_stale(tmp_path, grid_launch, change):
    dut, tb = build(tmp_path, tmp_path / 'change.db')
    tb.set_temperatures([27])
    tb.set_supply_voltage(1, source='stim')
    tb.add_source(sources.VoltageSource('other', positive='other', negative='0', dc=2))
    try:
        _store_grid(dut, tb)
        if change == 'temperature':
            tb.set_temperatures([85])
        elif change == 'supply':
            tb.set_supply_voltage(2, source='stim')
        elif change == 'source':
            tb.set_supply_voltage(1, source='other')
        elif change == 'analysis':
            tb.netlist.sim_analysis_dict.clear()
            tb.add_sim_type(Transient(step=2e-9, stop=10e-9))
        elif change == 'option':
            tb.set_simulator_options(reltol=.01)
        elif change == 'backend':
            dut.simulator = 'xyce'
        else:
            tb.add(elements.resistor('extra', p='out', n='0', value=20))
        row, = dut.query_results(parameter='out_V')
        assert row.deck_check.state == 'stale'
    finally:
        dut.close()


@pytest.mark.parametrize('observed', [(), (27.,)])
def test_supply_grid_never_injects_observed_default_temperature(tmp_path, launch, monkeypatch, observed):
    monkeypatch.setattr(NgspiceBackend, 'read_ambient_temperatures', lambda *a: observed)
    dut, tb = build(tmp_path, tmp_path / 'default.db')
    tb.set_supply_voltages([.8, 1.8], source='stim')
    try:
        _store_grid(dut, tb)
        assert all(r.deck_check.state == 'checked' for r in dut.query_results())
        # Explicitly configuring the observed default changes the emitted deck.
        tb.set_temperature(27)
        assert all(r.deck_check.state == 'stale' for r in dut.query_results())
    finally:
        dut.close()


def test_nested_grid_placement_cache_and_no_live_point_application(tmp_path, grid_launch, monkeypatch):
    from contextlib import contextmanager
    from silk.circuit import module_test, modules_netlist
    from silk.circuit.test_placement import evidence_index
    child, tb = build(tmp_path / 'child', tmp_path / 'child.db')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    top = Module(None, str(tmp_path / 'root'), 'root.cir', None, db_path=str(tmp_path / 'root.db'))
    parents = []
    try:
        for name, temps, volts in [('low', [27, 85], [.8]), ('high', [85], [1.8])]:
            parent = Module(None, str(tmp_path / name), name + '.cir', None)
            parent.define_module_ports({})
            parent.set_temperatures(temps).set_supply_voltages(volts, source='stim')
            parent.define_module_name(name)
            parent.create_test('unused')
            parent.add_module(child, 'cell', out='out', vss='0')
            top.add_module(parent, name, nets={})
            parents.append(parent)
        top.create_circuit_version()
        for name in ('low', 'high'):
            reports = top.get_test('gain', instance=name + '.cell').run(corner='ss')
            for report in reports.values():
                top.store_report(report)
        modules = [top, child, tb.netlist, *parents]
        before = _emission_state(modules)
        options = dict(tb.netlist.simulator_options)
        condition_names = ('_temperature_explicit', '_temperature_basis', '_temperature_list',
                           '_supply_explicit', '_supply_voltage', '_supply_source', '_supply_list')
        unset = object()
        conditions = tuple(getattr(tb.netlist, name, unset) for name in condition_names)
        source = tb.netlist.sig_sources_dict['stim']
        source_state = dict(vars(source))
        test_state = dict(vars(tb))
        index = evidence_index(top)
        evidence = (dict(index.records), dict(index.selectors), dict(index.annotations))
        files = set(tmp_path.rglob('*'))
        planned = module_test._planned_conditions
        @contextmanager
        def isolated(test, run):
            assert tuple(getattr(tb.netlist, name, unset) for name in condition_names) == conditions
            assert test is not tb
            assert test.netlist is not tb.netlist
            assert test.netlist.sig_sources_dict['stim'] is not source
            assert dict(vars(source)) == source_state
            with planned(test, run):
                assert dict(vars(source)) == source_state
                yield
        monkeypatch.setattr(module_test, '_planned_conditions', isolated)
        emit = modules_netlist._emit_deck_text
        calls = []
        def counted(*args, **kwargs):
            calls.append(1)
            return emit(*args, **kwargs)
        monkeypatch.setattr(modules_netlist, '_emit_deck_text', counted)
        rows = top.query_results(parameter='out_V', latest_only=True)
        assert len(rows) == 3
        assert all(r.deck_check.state == 'checked' for r in rows)
        assert len(calls) == 3
        spec.measurements.clear()
        assert top.restore_spec_measurements() == 3
        assert _emission_state(modules) == before
        assert dict(vars(source)) == source_state
        assert tb.netlist.simulator_options == options
        assert tuple(getattr(tb.netlist, name, unset) for name in condition_names) == conditions
        assert vars(tb) == test_state
        assert (index.records, index.selectors, index.annotations) == evidence
        assert set(tmp_path.rglob('*')) == files
        parents[0].set_supply_voltages([1.8], source='stim')
        rows = top.query_results(parameter='out_V')
        assert sorted(r.deck_check.state for r in rows) == ['checked', 'stale', 'stale']
        spec.measurements.clear()
        refused_state = _emission_state(modules)
        with pytest.raises(EvidenceMismatchError):
            top.restore_spec_measurements()
        assert not spec.measurements
        assert _emission_state(modules) == refused_state
        assert (index.records, index.selectors, index.annotations) == evidence
        assert tuple(getattr(tb.netlist, name, unset) for name in condition_names) == conditions
        assert tb.netlist.simulator_options == options
        assert vars(tb) == test_state
        assert set(tmp_path.rglob('*')) == files
        assert dict(vars(source)) == source_state
    finally:
        top.close()
        child.close()
        for parent in parents:
            parent.close()


@pytest.mark.parametrize('clear', ['temperature', 'supply'])
def test_grid_explicit_clear_overrides_inherited_list(tmp_path, grid_launch, clear):
    dut, tb = build(tmp_path, tmp_path / 'clear.db')
    dut.set_temperatures([27, 85]).set_supply_voltages([.8, 1.8], source='stim')
    try:
        if clear == 'temperature':
            tb.set_temperatures(None).set_temperature(40)
        else:
            tb.set_supply_voltages(None).set_supply_voltage(1.2, source='stim')
        reports = _store_grid(dut, tb)
        assert len(list(reports.values())) == 2
        assert all(r.deck_check.state == 'checked' for r in dut.query_results())
        # Changes to the cleared inherited axis cannot invalidate this bench.
        if clear == 'temperature':
            dut.set_temperatures([100])
        else:
            dut.set_supply_voltages([3], source='stim')
        assert all(r.deck_check.state == 'checked' for r in dut.query_results())
    finally:
        dut.close()


def test_stale_grid_recovery_runs_plan_and_stores_reports(tmp_path, grid_launch):
    dut, tb = build(tmp_path, tmp_path / 'recovery.db')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    tb.set_temperatures([27]).set_supply_voltages([.8, 1.8], source='stim')
    try:
        _store_grid(dut, tb)
        tb.set_simulator_options(reltol=.01)
        row = dut.query_results(parameter='out_V')[0]
        assert row.deck_check.state == 'stale'
        recovery = row.deck_check.reason.split('rerun it: ', 1)[1]
        exec(recovery, {'dut': dut})
        assert all(r.deck_check.state == 'checked'
                   for r in dut.query_results(parameter='out_V', latest_only=True))
        # Historical stale observations must not prevent a matching latest restore.
        assert any(r.deck_check.state == 'stale' for r in dut.query_results(parameter='out_V'))
        spec.measurements.clear()
        assert dut.restore_spec_measurements() == 2
        # A newer stale observation is refused rather than falling back.
        tb.set_simulator_options(reltol=.02)
        with pytest.raises(EvidenceMismatchError):
            dut.query_results(parameter='out_V', latest_only=True)
    finally:
        dut.close()


def test_grid_invalid_source_keeps_error_and_live_state(tmp_path, grid_launch):
    from silk.exceptions import SimulationConfigurationError
    dut, tb = build(tmp_path, tmp_path / 'invalid.db')
    tb.set_temperatures([27]).set_supply_voltages([1], source='stim')
    try:
        _store_grid(dut, tb)
        tb.set_supply_voltages([2], source='missing')
        before = _emission_state([dut, tb.netlist])
        source = tb.netlist.sig_sources_dict['stim']
        options = dict(tb.netlist.simulator_options)
        files = set(tmp_path.rglob('*'))
        with pytest.raises(SimulationConfigurationError, match='missing'):
            dut.query_results(parameter='out_V')
        assert _emission_state([dut, tb.netlist]) == before
        assert tb.netlist.sig_sources_dict['stim'] is source and source.dc == 1
        assert tb.netlist.simulator_options == options
        assert set(tmp_path.rglob('*')) == files
    finally:
        dut.close()


@pytest.mark.parametrize('api', ['history', 'restore'])
@pytest.mark.parametrize('configuration', ['valid', 'supply', 'sweep'])
def test_monte_carlo_exclusion_preserves_validation_and_atomicity(
        tmp_path, launch, monkeypatch, api, configuration):
    from silk.circuit.module_test import ModuleTestbench
    from silk.circuit.test_placement import evidence_index
    from silk.exceptions import SimulationConfigurationError, SweepSourceError
    from silk.simulation import DC

    dut, tb = build(tmp_path, tmp_path / 'mc-validation.db')
    spec = tb.add_spec('level', 'out_V', '>=', .5)
    try:
        dut.create_circuit_version()
        dut.run_test('gain', store=True)
        spec.measurements.clear()
        tb.set_monte_carlo(2, seed=1, output_mode='new')
        if configuration == 'supply':
            tb.set_supply_voltage(2, source='missing')
        elif configuration == 'sweep':
            tb.netlist.sim_analysis_dict.clear()
            tb.add_sim_type(DC(source='missing', start=0, stop=1, step=.1))
        def forbid(*args, **kwargs):
            raise AssertionError('Monte Carlo lookup must not plan or prepare output')
        monkeypatch.setattr(ModuleTestbench, '_plan_runs', forbid)
        monkeypatch.setattr(ModuleTestbench, '_prepare_run', forbid)
        source = tb.netlist.sig_sources_dict['stim']
        source_state = dict(vars(source))
        options = dict(tb.netlist.simulator_options)
        bench_state = dict(vars(tb))
        conditions = {name: value for name, value in vars(tb.netlist).items()
                      if name.startswith(('_temperature', '_supply'))}
        before = _emission_state([dut, tb.netlist])
        index = evidence_index(dut)
        evidence = (dict(index.records), dict(index.selectors), dict(index.annotations))
        raw = dut._tracker.db.get_simulation_results(dut._tracker.circuit_id)
        provenance = [dut._tracker.db.get_run_provenance(row.provenance_id) for row in raw]
        files = set(tmp_path.rglob('*'))
        call = (lambda: dut.query_results(parameter='out_V', latest_only=True)
                if api == 'history' else dut.restore_spec_measurements())
        if configuration == 'valid':
            with pytest.warns(UserWarning, match='Monte Carlo test'):
                result = call()
            if api == 'history':
                assert result[0].deck_check.state == 'not checked'
            else:
                assert result == 1
        else:
            error = SimulationConfigurationError if configuration == 'supply' else SweepSourceError
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with pytest.raises(error, match='missing'):
                    call()
            assert not caught
            assert not spec.measurements
        assert _emission_state([dut, tb.netlist]) == before
        assert vars(tb) == bench_state
        assert tb.netlist.sig_sources_dict['stim'] is source
        assert vars(source) == source_state
        assert tb.netlist.simulator_options == options
        assert {name: value for name, value in vars(tb.netlist).items()
                if name.startswith(('_temperature', '_supply'))} == conditions
        assert (index.records, index.selectors, index.annotations) == evidence
        assert dut._tracker.db.get_simulation_results(dut._tracker.circuit_id) == raw
        assert [dut._tracker.db.get_run_provenance(row.provenance_id) for row in raw] == provenance
        assert set(tmp_path.rglob('*')) == files
    finally:
        dut.close()


def test_variable_list_plan_is_not_checked_without_planning(stored, monkeypatch):
    from silk import Variable
    from silk.circuit.module_test import ModuleTestbench
    dut, tb = stored
    knob = Variable('rail', value=1)
    source = tb.netlist.sig_sources_dict['stim']
    source.dc = knob
    tb.set_variable_values(knob, [1, 2])
    def forbid(*args, **kwargs):
        raise AssertionError('Variable plans must be excluded before planning or emission')
    monkeypatch.setattr(ModuleTestbench, '_plan_runs', forbid)
    before = _emission_state([dut, tb.netlist])
    files = set(Path(dut.dir_folder).rglob('*'))
    with pytest.warns(UserWarning, match='Variable|variable'):
        rows = dut.query_results(parameter='out_V', latest_only=True)
    assert rows and all(row.deck_check.state == 'not checked' for row in rows)
    assert all('Variable' in row.deck_check.reason or 'variable' in row.deck_check.reason for row in rows)
    assert _emission_state([dut, tb.netlist]) == before
    assert set(Path(dut.dir_folder).rglob('*')) == files
    assert source.dc is knob and knob.value == 1
