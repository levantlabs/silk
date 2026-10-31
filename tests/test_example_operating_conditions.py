"""Real sky130 regressions: simulator conditions must match persisted PVT.

These tests deliberately read TEMP from ngspice's log and VDD from its
rawfile before querying the metric/spec/current evidence at that point.
The core configured/observed run() contract is covered separately.
"""
import os
import numpy as np
import pytest

from silk.db import Database, EvidenceMismatchError
from silk.examples import design_loop_walkthrough as dl
from tests._env import needs_ngspice, needs_pdk
from tests._example_conditions import assert_simulator_temperature

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


def _observed_point(test, context, temperature, voltage):
    actual_temp, tnom = assert_simulator_temperature(
        os.path.join(test.netlist.dir_folder, test.netlist.netlistname), temperature)
    assert tnom == 27.0, 'operating TEMP must not change model reference TNOM'
    supply = np.asarray(test.netlist.load_signal('vdd').y)
    assert supply.size > 1
    np.testing.assert_allclose(supply, voltage, rtol=0, atol=1e-12)
    assert context == test.netlist.characterization_context
    assert context.temperature == actual_temp
    assert context.voltage == pytest.approx(float(supply[-1]))
    return actual_temp, float(supply[-1])


def _stored_point(history, version, context, observed, metrics):
    # Select by the independently observed operating point, then inspect
    # every evidence family's own fields and measured values.
    profile = history.get_characterization_profile(version, context=context)
    evidence = history.select_characterization_evidence(
        version, profile_id=profile.profile_id, corner=context.corner,
        temperature=observed[0], voltage=observed[1])
    measurements = [m for spec in evidence.specs for m in spec.measurements]
    assert evidence.metrics and measurements and evidence.current_set.currents
    for row in (*evidence.metrics, *measurements, evidence.current_set):
        assert row.profile_id == profile.profile_id
        assert row.corner == context.corner
        assert row.temperature == observed[0]
        assert row.voltage == pytest.approx(observed[1])
    for name, value in metrics.items():
        rows = [r for r in evidence.metrics if r.parameter == name]
        assert rows
        assert all(r.value == pytest.approx(value) for r in rows)
    for spec in evidence.specs:
        for measurement in spec.measurements:
            assert measurement.value == pytest.approx(metrics[spec.metric])
            assert measurement.passes
    assert any(abs(row.peak_a) > 0 for row in evidence.current_set.currents)
    return evidence


@pytest.mark.parametrize('temperatures', [(27.0, 85.0), (35.0, 65.0)],
                         ids=['default-intent', 'overridden-intent'])
def test_walkthrough_applies_intent_and_persists_actual_point(tmp_path, temperatures):
    intent = dict(dl.INTENT, nominal_temperature_c=temperatures[0],
                  hot_temperature_c=temperatures[1])
    knobs = dl.size_from_intent(intent)
    db_path = str(tmp_path / 'conditions.db')
    mod, tests = dl.build_sampling_buffer(
        knobs, intent=intent, db_path=db_path, work_dir=str(tmp_path))
    mod.set_output_options(mode='overwrite')
    dl.wire_pre_layout_tests(tests, knobs, intent)
    version, contexts, metrics, report = dl.characterize_and_store(
        mod, tests, knobs, db_path, str(tmp_path), intent=intent)
    statuses = {o.spec.key: o.status for o in report.spec_reconciliation.outcomes}
    assert statuses == {'settle_light': 'PASS', 'settle_heavy': 'PASS',
                        'gain': 'PASS', 'bw_pre_layout': 'PASS',
                        'bw_post_layout': 'UNMEASURED'}
    with Database(db_path) as history:
        for name, temp, voltage, metric in (
                (dl.T_STEP_LIGHT, temperatures[0], intent['vdd_v'], 't_settle_light_s'),
                (dl.T_STEP_HEAVY, temperatures[1], intent['hot_vdd_v'], 't_settle_heavy_s')):
            test = tests[name]
            assert test.netlist.save_device_currents
            point = _observed_point(test, contexts[name], temp, voltage)
            _stored_point(history, version, contexts[name], point,
                          {metric: metrics[metric]})
    assert_simulator_temperature(os.path.join(tests[dl.T_AC_PRE].netlist.dir_folder, tests[dl.T_AC_PRE].netlist.netlistname),
                                 temperatures[0])


def test_walkthrough_repeated_run_replaces_temperature(tmp_path):
    knobs = dl.size_from_intent()
    mod, tests = dl.build_sampling_buffer(knobs, work_dir=str(tmp_path))
    mod.set_output_options(mode='overwrite')
    dl.wire_pre_layout_tests(tests, knobs)
    test = tests[dl.T_STEP_HEAVY]
    try:
        # Return to the original point too: neither direction can retain
        # a stale option, and the ordinary current-disabled path also runs.
        for temperature, currents in ((85.0, True), (41.0, False), (85.0, True)):
            report = dl.run_test(test, corner='tt', temperature=temperature,
                                 voltage=1.62, currents=currents)
            assert report.passed, report.summary()
            _observed_point(test, report.characterization_context, temperature, 1.62)
    finally:
        for test in tests.values():
            if test.netlist is not None:
                test.netlist.close()
        mod.close()


@pytest.mark.parametrize('temperatures', [(27.0, 85.0), (35.0, 65.0)],
                         ids=['default-points', 'overridden-points'])
def test_current_persistence_applies_and_replaces_temperature(tmp_path, temperatures):
    # This example loads its PDK on import; keep that behind the skip marker.
    from silk.examples import net_current_persistence_example as cp

    db_path = str(tmp_path / 'currents.db')
    dut = cp.build_current_sink(db_path=db_path)
    dut.dir_folder = str(tmp_path)
    dut.set_output_options(mode='overwrite')
    dut.save_device_currents = True
    dut.terminal_current_method = 'ammeter_wrapper'
    params = {'tstart_s': 1e-6}
    nominal = cp.declare_test(dut, test_name='characterize_nominal',
                              spec=cp.SPEC_NOMINAL, parameters=params)
    hot = cp.declare_test(dut, test_name='characterize_hot_corner',
                          spec=cp.SPEC_HOT, parameters=params)
    cp.wire_test(nominal, voltage=1.8)
    cp.wire_test(hot, voltage=1.62)
    dut.create_circuit_version('operating condition regression')
    observations = []
    try:
        for test, temp, voltage in (
                (nominal, temperatures[0], 1.8), (hot, temperatures[1], 1.62),
                (hot, 41.0, 1.62), (hot, temperatures[1], 1.62)):
            version, context = cp.run_and_store(
                dut, test, db_path, test_name=test.test_name, parameters=params,
                temperature=temp, voltage=voltage)
            point = _observed_point(test, context, temp, voltage)
            observations.append((context, point, cp._vout_headroom(test.netlist)))
        assert all(o.status == 'PASS' for o in dut.reconcile_specs().outcomes)
    finally:
        nominal.netlist.close()
        hot.netlist.close()
        dut.close()
    with Database(db_path) as history:
        for context, point, value in observations:
            _stored_point(history, version, context, point, {'vout_headroom_v': value})


def test_current_persistence_main_restores_unscoped_supply_evidence(
    tmp_path, monkeypatch, capsys,
):
    # Import behind the dependency markers, as above.
    from silk.examples import net_current_persistence_example as cp

    # Output now resolves through silk.examples.example_output_dir()
    # ($SILK_EXAMPLES_OUT, else ./silk-output) instead of the example's
    # own file location — TASK-243.
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    module_class = cp.Module
    modules = []

    def isolated_module(pdk_name, dir_folder, netlistname, **kwargs):
        # main() resets its demo database; block/testbench files also need
        # isolation because build_current_sink() normally uses shared /tmp.
        if dir_folder == '/tmp':
            dir_folder = str(tmp_path / f'block_{len(modules)}')
        module = module_class(pdk_name, dir_folder, netlistname, **kwargs)
        modules.append(module)
        return module

    monkeypatch.setattr(cp, 'Module', isolated_module)
    db_path = str(tmp_path / 'net_current_persistence' / 'em_currents.db')
    try:
        # Exercise the actual final section, including its explicit storage
        # voltage assertion, as well as the earlier scoped workflow.
        cp.main()
        output = capsys.readouterr().out
        assert 'reloaded via load_net_currents(): out carries' in output
        assert 'Done. See docs/database.md' in output

        tb, = [module for module in modules if not module.module_name]
        conditions = tb.operating_conditions
        assert conditions.supply_voltage_v == 1.8
        assert conditions.supply_basis == 'configured-and-emitted'
        assert conditions.supply_source == 'vdd'
        assert conditions.temperature_c == 27.0
        assert tb.characterization_context is None
        assert not tb.list_tests()
        supply = np.asarray(tb.load_signal('vdd').y)
        assert supply.size > 1
        np.testing.assert_allclose(supply, 1.8, rtol=0, atol=1e-12)
        assert conditions.temperature_basis == 'backend-observed'
        definition = tb.module_dict['current_sink']['device']
        measured_peak = definition.net_dict['out'].peak_current
        assert measured_peak > 0

        # main() has closed its connections. Read the unscoped evidence from
        # a new connection and compare it with the actual simulated current.
        with Database(db_path) as history:
            current_set, = history.conn.execute(
                'SELECT circuit_version_id, profile_id, corner, temperature, voltage '
                'FROM net_current_sets WHERE profile_id IS NULL'
            ).fetchall()
            version = current_set['circuit_version_id']
            assert tuple(current_set)[1:] == (None, 'tt', 27.0, 1.8)
            assert history.conn.execute(
                'SELECT COUNT(*) FROM characterization_profiles '
                'WHERE circuit_version_id = ?', (version,),
            ).fetchone()[0] == 0
            origin, rows = history.get_net_current_set(version, 'tt')
            assert origin == version
            assert rows
            saved_out, = [row for row in rows
                          if row['net_name'] == 'out' and row['pin_id'] == '']
            assert saved_out['peak_a'] == pytest.approx(measured_peak)
            before = list(history.conn.iterdump())

        fresh = cp.build_current_sink()
        assert not fresh.sim_results_loaded
        assert fresh.net_dict['out'].peak_current is None
        fresh.load_net_currents('tt', db_path=db_path)
        assert fresh.net_dict['out'].peak_current == pytest.approx(saved_out['peak_a'])
        assert not fresh.sim_results_loaded
        assert fresh.characterization_context is None

        report = tb.annotate_currents(tstart=1e-6)
        with pytest.raises(EvidenceMismatchError, match='voltage'):
            tb.store_net_currents(report, corner='tt', db_path=db_path,
                                  temperature=27.0, voltage=1.7)
        with Database(db_path) as history:
            assert list(history.conn.iterdump()) == before
    finally:
        for module in modules:
            for test in module._tests.values():
                if test.netlist is not None:
                    test.netlist.close()
            module.close()
