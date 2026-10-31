"""TASK-499: typed failures, run records and per-test device-parameter capture."""
import json
import os

import pytest

from silk import Module, sources
from silk.circuit import TestFailure
from silk.circuit.circuit_elements import PortDir
from silk.exceptions import SimulationAborted
from silk.simulation import Transient, analysis
from tests._env import needs_ngspice, needs_pdk
from tests.circuit.test_selected_testbench import leaf, root


# --- TestFailure: no simulator needed ------------------------------------

def test_testfailure_is_a_dict_equal_to_the_old_shape():
    failure = TestFailure.from_exception(ValueError('boom'))
    assert failure == {'error': 'boom'}
    assert set(failure) == {'error'}
    assert failure['error'] == 'boom'
    assert json.dumps(failure) == '{"error": "boom"}'
    assert failure.error_type == 'ValueError'
    assert failure.error_class == 'builtins.ValueError'
    assert failure.details == {}
    assert failure.run is None
    assert isinstance(failure.exception, ValueError)
    assert failure.exception_is_original


def test_testfailure_details_for_simulation_aborted():
    exc = SimulationAborted('aborted', simulator='ngspice', reason='Timestep too small',
                            data_rows=0, log_path='/x/output.log')
    failure = TestFailure.from_exception(exc)
    assert failure.error_type == 'SimulationAborted'
    assert failure.details == {'simulator': 'ngspice', 'reason': 'Timestep too small',
                               'data_rows': 0, 'log_path': '/x/output.log'}


def test_testfailure_pickles():
    import pickle
    failure = pickle.loads(pickle.dumps(TestFailure.from_exception(ValueError('boom'))))
    assert type(failure) is TestFailure and failure == {'error': 'boom'}
    assert failure.error_type == 'ValueError'


def test_results_codec_round_trips_a_testfailure(tmp_path):
    from silk.circuit._results_json import results_data as dump_results, restore_results
    exc = SimulationAborted('aborted', simulator='ngspice', reason='r', data_rows=0,
                            log_path='/x/output.log')
    results = {'m.t': TestFailure.from_exception(exc)}
    data = dump_results(results)
    assert json.loads(json.dumps(data))
    restored = restore_results(json.loads(json.dumps(data)))
    failure = restored['m.t']
    assert isinstance(failure, TestFailure)
    assert failure == {'error': 'aborted'}
    assert failure.error_type == 'SimulationAborted'
    assert failure.error_class == 'silk.exceptions.SimulationAborted'
    assert failure.details['log_path'] == '/x/output.log'
    assert failure.exception is None


def test_legacy_plain_error_dict_still_loads():
    from silk.circuit._results_json import results_data as dump_results, restore_results
    restored = restore_results(dump_results({'m.t': {'error': 'old'}}))
    assert restored['m.t'] == {'error': 'old'}


def test_unpicklable_exception_keeps_original_identity():
    from silk.circuit.module_test import _RunOutcome
    import pickle

    class Local(Exception):
        pass
    err = Local('local failure')
    err.handle = lambda: None            # not picklable
    outcome = pickle.loads(pickle.dumps(_RunOutcome(error=err)))
    failure = TestFailure.from_outcome(outcome)
    assert failure.error_type == 'Local'
    assert failure.error_class.endswith('Local')
    assert failure == {'error': 'local failure'}
    assert not failure.exception_is_original


# --- save_device_params: tri-state ----------------------------------------

def test_save_device_params_is_tristate(tmp_path):
    mod, tb = leaf(tmp_path)
    try:
        assert tb.save_device_params is None
        tb.save_device_params = True
        tb.save_device_params = False
        tb.save_device_params = None
        with pytest.raises(TypeError, match='save_device_params'):
            tb.save_device_params = 'yes'
        other = mod.create_test('other', save_device_params=True)
        assert other.save_device_params is True
        assert mod.save_device_params is False
    finally:
        mod.close()


# --- run records ------------------------------------------------------------

@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_report_names_deck_log_and_launch(tmp_path, workers):
    mod, tb = leaf(tmp_path)
    try:
        report = next(iter(mod.run_all_tests(n_workers=workers).values()))
        run = report.run
        assert os.path.isabs(run.deck_path) and os.path.isfile(run.deck_path)
        assert run.deck_path.endswith('.cir')
        assert run.rawfile_path == report.rawfile_path
        assert os.path.isdir(run.directory)
        assert run.launch_id and report.launch_id == run.launch_id
        again = next(iter(mod.run_all_tests(n_workers=workers).values()))
        assert again.launch_id != report.launch_id
    finally:
        mod.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_shared_placements_share_one_launch_id(tmp_path, workers):
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    try:
        for index in range(3):
            top.add_module(child, f'u{index}', out=f'n{index}', vss='0')
        reports = list(top.run_all_tests(n_workers=workers).values())
        assert len({r.launch_id for r in reports}) == 1
        assert reports[0].launch_id
        assert len({r.run.deck_path for r in reports}) == 1
    finally:
        top.close()
        child.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_simulator_failure_keeps_type_details_and_files(tmp_path, workers):
    mod, tb = leaf(tmp_path)
    # Two ideal sources forcing one node to different voltages: singular.
    tb.add_source(sources.VoltageSource('rail2', positive='out', negative='0', dc=1.0))
    try:
        results = mod.run_all_tests(n_workers=workers)
        failure = next(iter(results.values()))
        assert type(failure) is TestFailure
        assert 'error' in failure and set(failure) == {'error'}
        assert failure.error_type in {'SimulationAborted', 'ConvergenceError'}
        assert issubclass(type(failure.exception), SimulationAborted)
        assert failure.details['simulator'] == 'ngspice'
        assert failure.run is not None
        assert os.path.isfile(failure.run.deck_path)
        assert failure.run.log_path == failure.details['log_path']
        assert failure.run.launch_id
    finally:
        mod.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_plain_error_is_distinguishable(tmp_path, monkeypatch, workers):
    from silk.circuit.module_test import ModuleTestbench

    def broken(self, *args, **kwargs):
        raise ValueError('evaluation broke')
    monkeypatch.setattr(ModuleTestbench, 'evaluate', broken)
    mod, tb = leaf(tmp_path)
    try:
        failure = next(iter(mod.run_all_tests(n_workers=workers).values()))
        assert isinstance(failure, TestFailure)
        assert failure == {'error': 'evaluation broke'}
        assert failure.error_type == 'ValueError'
        assert failure.details == {}
        assert failure.exception_is_original
        # The launch itself succeeded, so its files are known.
        assert failure.run is not None and os.path.isfile(failure.run.deck_path)
    finally:
        mod.close()


# --- per-test device-parameter capture ---------------------------------------

def _two_bench_mos(tmp_path):
    from silk.pdk import load_pdk
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                             's': PortDir.GROUND, 'b': PortDir.GROUND})
    dut.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.5))
    tbs = []
    for name in ('with_params', 'without_params'):
        tb = dut.create_test(name)
        tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
        tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
        tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
        tb.add_sim_type(analysis.Op())
        tbs.append(tb)
    return dut, tbs


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('workers', [1, 2])
def test_capture_for_one_test_only(tmp_path, workers):
    dut, (on, off) = _two_bench_mos(tmp_path)
    on.save_device_params = True
    try:
        results = dut.run_all_tests(n_workers=workers)
        by_name = {key.rsplit('.', 1)[-1]: report for key, report in results.items()}
        assert len(by_name['with_params'].op_table().rows) == 1
        with pytest.raises(Exception, match='save_device_params'):
            by_name['without_params'].op_table()
        assert dut.save_device_params is False
        assert off.save_device_params is None
    finally:
        dut.close()


@needs_ngspice
@needs_pdk
def test_explicit_false_overrides_a_true_dut(tmp_path):
    dut, (on, off) = _two_bench_mos(tmp_path)
    dut.save_device_params = True
    off.save_device_params = False
    try:
        results = dut.run_all_tests()
        by_name = {key.rsplit('.', 1)[-1]: report for key, report in results.items()}
        assert len(by_name['with_params'].op_table().rows) == 1          # inherits True
        with pytest.raises(Exception, match='save_device_params'):
            by_name['without_params'].op_table()                  # explicit False
    finally:
        dut.close()


# --- Run-record serialization and failure-handling checks -------------------

def _record(**over):
    from silk.circuit import RunRecord
    values = dict(launch_id='abc', deck_path='/a/d.cir', log_path='/a/o.log',
                  rawfile_path='/a/r.raw', noise_path=None, directory='/a')
    values.update(over)
    return RunRecord(**values)


def test_spec_report_json_carries_run_and_old_files_load():
    from silk.circuit import SpecReport
    report = SpecReport('t', 'm', {})
    report.run = _record()
    wire = report.to_dict()
    assert wire['run']['deck_path'] == '/a/d.cir' and wire['launch_id'] == 'abc'
    restored = SpecReport.from_dict(json.loads(json.dumps(wire)))
    assert restored.run == report.run and restored.launch_id == 'abc'
    old = dict(wire)
    del old['run'], old['launch_id']
    assert SpecReport.from_dict(old).run is None
    bare = SpecReport('t', 'm', {}).to_dict()          # no record: keys not written
    assert 'run' not in bare and 'launch_id' not in bare
    bad = dict(wire, launch_id='other')
    with pytest.raises(ValueError, match='launch_id'):
        SpecReport.from_dict(bad)


def test_spec_report_run_survives_a_results_collection():
    from silk.circuit import SpecReport
    from silk.circuit._results_json import results_data, restore_results
    report = SpecReport('t', 'm', {})
    report.run = _record()
    restored = restore_results(json.loads(json.dumps(results_data({'k': report}))))
    assert restored['k'].run == report.run


@pytest.mark.parametrize('make', [
    lambda: SimulationAborted('aborted', simulator='ngspice', reason='r',
                              data_rows=0, log_path='/x/o.log'),
    lambda: __import__('silk.exceptions', fromlist=['x']).ConvergenceError(
        'did not converge', simulator='ngspice', matched_lines=['x'], reason='r',
        data_rows=None, log_path=None),
])
def test_testfailure_pickles_with_silk_exceptions(make):
    import pickle
    failure = TestFailure.from_exception(make())
    restored = pickle.loads(pickle.dumps(failure))
    assert restored == failure
    assert restored.error_type == failure.error_type
    assert restored.details == failure.details
    assert type(restored.exception) is type(failure.exception)
    assert restored.exception.simulator == 'ngspice'
    assert restored.exception_is_original


def test_testfailure_pickle_with_unpicklable_attribute_uses_a_flagged_surrogate():
    import pickle
    err = ValueError('plain')
    err.handle = lambda: None
    restored = pickle.loads(pickle.dumps(TestFailure.from_exception(err)))
    assert restored == {'error': 'plain'} and restored.error_type == 'ValueError'
    assert not restored.exception_is_original
    assert type(restored.exception) is ValueError
    assert not hasattr(restored.exception, 'handle')           # a stand-in, flagged


def test_testfailure_pickle_can_drop_the_exception(monkeypatch):
    import pickle
    from silk.circuit import test_failure
    monkeypatch.setattr(test_failure, 'pack_exception', lambda error: (None, True))
    exc = SimulationAborted('aborted', simulator='ngspice', reason='r', data_rows=0,
                            log_path='/x/o.log')
    failure = TestFailure.from_exception(exc)
    failure.run = _record()
    restored = pickle.loads(pickle.dumps(failure))
    assert restored.exception is None and not restored.exception_is_original
    assert restored == {'error': 'aborted'} and restored.error_type == 'SimulationAborted'
    assert restored.details == failure.details and restored.run == failure.run


def test_testfailure_json_round_trip_with_a_run_record():
    from silk.circuit._results_json import results_data, restore_results
    failure = TestFailure.from_exception(SimulationAborted(
        'aborted', simulator='ngspice', reason='r', data_rows=0, log_path='/a/o.log'))
    failure.run = _record()
    wire = json.loads(json.dumps(results_data({'k': failure})))
    assert wire['schema_version'] == 2
    restored = restore_results(wire)['k']
    assert restored == failure and restored.run == failure.run
    assert restored.exception is None and not restored.exception_is_original
    assert restored.details == failure.details and restored.error_class == failure.error_class


def test_repeated_capture_keeps_one_launch_identity():
    from types import SimpleNamespace
    from silk.circuit.test_execution import RunEvidence
    record = _record(launch_id='one-launch')
    result = SimpleNamespace(netlist=SimpleNamespace(), corner_str='tt', conditions={},
                             run_id='one-launch', run=record)
    first, second = RunEvidence.capture(result), RunEvidence.capture(result)
    assert first.run_id == second.run_id == 'one-launch'
    spec = SimpleNamespace(invocation=(), cache_key=None)
    assert first.result(spec).run is record and second.result(spec).run_id == 'one-launch'


def test_same_class_lossy_fallback_is_not_original():
    from silk.circuit.module_test import _RunOutcome
    import pickle
    err = ValueError('lossy')
    err.handle = lambda: None                     # same importable class survives
    outcome = pickle.loads(pickle.dumps(_RunOutcome(error=err)))
    assert type(outcome.error) is ValueError and not hasattr(outcome.error, 'handle')
    failure = TestFailure.from_outcome(outcome)
    assert failure.error_class == 'builtins.ValueError'
    assert not failure.exception_is_original
    clean = pickle.loads(pickle.dumps(_RunOutcome(error=ValueError('fine'))))
    assert TestFailure.from_outcome(clean).exception_is_original


def test_failure_node_validation_and_version(tmp_path):
    from silk.circuit._results_json import results_data
    good = results_data({'k': TestFailure.from_exception(ValueError('x'))})
    assert good['schema_version'] == 2
    assert results_data({'k': {'error': 'x'}})['schema_version'] == 1
    extra = TestFailure.from_exception(ValueError('x'))
    extra['more'] = 1
    with pytest.raises((TypeError, ValueError), match='error'):
        results_data({'k': extra})
    gone = TestFailure.from_exception(ValueError('x'))
    del gone['error']
    with pytest.raises((TypeError, ValueError), match='error'):
        results_data({'k': gone})


def test_selected_handle_setter_reaches_the_authored_bench(tmp_path):
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    try:
        top.add_module(child, 'u0', out='n0', vss='0')
        handle = top.get_test('step', instance='u0')
        handle.save_device_params = True
        assert tb.save_device_params is True and handle.save_device_params is True
        with pytest.raises(TypeError, match='save_device_params'):
            handle.save_device_params = 'yes'
        assert tb.save_device_params is True
        handle.save_device_params = None
        assert tb.save_device_params is None
    finally:
        top.close()
        child.close()


def _behavioral_bench(pdk, tmp_path, fail=False):
    from silk.circuit.modules import Module
    from silk.metrics import final_value
    dut = Module('PDK_skywater130', str(tmp_path / 'dut'), 'dut.cir', pdk_details=pdk,
                 module_name='dut', db_path=str(tmp_path / 'runs.db'))
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    from silk.sources import VoltageControlledVoltageSource
    dut.add_source(VoltageControlledVoltageSource(
        'gain', out_pos='vout', out_neg='gnd', ctrl_pos='vin', ctrl_neg='gnd', gain=2))

    def model(ports, t, params):
        if fail:
            raise RuntimeError('model broke')
        ports.vout.v = 2 * ports.vin.v
    dut.set_behavioral_model(model, execution='vector')
    dut.simulator = 'behavioral'
    tb = dut.create_test('transfer')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(sources.VoltageSource('input', positive='vin', negative='gnd', dc=0.5))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('final_V', final_value, sig=tb.probe('vout'))
    return dut, tb


@pytest.mark.parametrize('workers', [1, 2])
def test_behavioral_run_records_its_own_files(pdk, tmp_path, workers):
    dut, tb = _behavioral_bench(pdk, tmp_path)
    try:
        report = next(iter(dut.run_all_tests(n_workers=workers).values()))
        run = report.run
        assert run is not None and run.launch_id
        assert run.deck_path is None and run.log_path is None
        assert run.rawfile_path == os.path.abspath(report.rawfile_path)
        assert os.path.isfile(run.rawfile_path)
        assert os.path.isdir(run.directory)
    finally:
        dut.close()


@needs_ngspice
def test_failed_run_in_new_mode_keeps_its_own_files(tmp_path):
    mod, tb = leaf(tmp_path)
    tb.add_source(sources.VoltageSource('rail2', positive='out', negative='0', dc=1.0))
    mod.set_output_options(mode='new')
    try:
        first = next(iter(mod.run_all_tests().values()))
        second = next(iter(mod.run_all_tests().values()))
        assert first.run.directory != second.run.directory
        assert os.path.isfile(first.run.deck_path) and os.path.isfile(second.run.deck_path)
        assert second.run.log_path == second.details['log_path']
        assert os.path.exists(second.run.log_path)
    finally:
        mod.close()


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('workers', [1, 2])
def test_selected_handle_capture_reaches_recursive_runs(tmp_path, workers):
    dut, (on, off) = _two_bench_mos(tmp_path)
    top = root(tmp_path)
    try:
        top.add_module(dut, 'u0', d='vd', g='vg', s='0', b='0')
        entries = {e.test_name: e for e in top.list_test_entries()}
        entries['with_params'].test.save_device_params = True
        results = top.run_all_tests(n_workers=workers)
        by_name = {key.rsplit('.', 1)[-1]: report for key, report in results.items()}
        assert len(by_name['with_params'].op_table().rows) == 1
        with pytest.raises(Exception, match='save_device_params'):
            by_name['without_params'].op_table()
        assert dut.save_device_params is False
    finally:
        top.close()
        dut.close()


# --- non-vacuous run-record controls (ruling 499d) ------------------------------

def _out_voltage(log_path):
    """The solved ``out`` node voltage printed in an ngspice log."""
    import re
    text = open(log_path, errors='replace').read()
    match = re.search(r'^out\s+(\S+)\s*$', text, re.M)
    assert match, text
    return float(match.group(1))


def _count_launches(monkeypatch, counter_file):
    """Append one line per real ngspice launch; visible from forked workers."""
    import subprocess
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        if isinstance(argv, (list, tuple)) and len(argv) > 2 and '-r' in argv \
                and str(argv[0]).endswith('ngspice'):
            with open(counter_file, 'a') as handle:
                handle.write('launch\n')
        return real_run(argv, *args, **kwargs)
    monkeypatch.setattr(subprocess, 'run', run)
    return lambda: (len(open(counter_file).read().split())
                    if os.path.exists(counter_file) else 0)


@needs_ngspice
def test_behavioral_after_a_real_spice_run_in_new_mode_records_only_its_own_files(pdk, tmp_path):
    dut, tb = _behavioral_bench(pdk, tmp_path)
    try:
        dut.simulator = 'ngspice'
        spice = tb.run()
        assert spice.run.deck_path and os.path.isfile(spice.run.deck_path)
        assert spice.run.log_path and os.path.isfile(spice.run.log_path)
        dut.simulator = 'behavioral'
        tb.netlist.simulator = 'behavioral'
        dut.set_output_options(mode='new')
        behavioral = tb.run()
        run = behavioral.run
        assert run.deck_path is None and run.log_path is None and run.noise_path is None
        assert os.path.isfile(run.rawfile_path)
        assert run.rawfile_path != spice.run.rawfile_path
        assert run.directory != spice.run.directory
        assert run.launch_id != spice.run.launch_id
    finally:
        dut.close()


def test_behavioral_failures_in_new_mode_record_their_own_distinct_files(pdk, tmp_path):
    dut, tb = _behavioral_bench(pdk, tmp_path, fail=True)
    dut.set_output_options(mode='new')
    try:
        first = next(iter(dut.run_all_tests().values()))
        second = next(iter(dut.run_all_tests().values()))
        for failure in (first, second):
            assert isinstance(failure, TestFailure) and failure.run is not None
            assert failure.run.deck_path is None and failure.run.log_path is None
            assert failure.run.rawfile_path.endswith('.raw')
            assert failure.run.directory == os.path.dirname(failure.run.rawfile_path)
        assert first.run.directory != second.run.directory
        assert first.run.launch_id != second.run.launch_id
    finally:
        dut.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_staging_refusal_after_success_launches_nothing(tmp_path, monkeypatch, workers):
    mod, tb = leaf(tmp_path)
    mod.set_output_options(mode='overwrite')
    count = _count_launches(monkeypatch, str(tmp_path / 'launches.txt'))
    try:
        ok = next(iter(mod.run_all_tests(n_workers=workers).values()))
        assert count() == 1
        raw = ok.run.rawfile_path
        os.remove(raw)
        os.mkdir(raw)                              # the output path is now a directory
        failure = next(iter(mod.run_all_tests(n_workers=workers).values()))
        assert isinstance(failure, TestFailure)
        assert 'not launched' in failure['error'].lower()
        assert failure.run is None
        assert count() == 1                        # zero launches for the refused attempt
    finally:
        mod.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_shared_placements_launch_once_and_expose_the_public_log(tmp_path, monkeypatch, workers):
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    count = _count_launches(monkeypatch, str(tmp_path / 'launches.txt'))
    try:
        for index in range(3):
            top.add_module(child, f'u{index}', out=f'n{index}', vss='0')
        reports = list(top.run_all_tests(n_workers=workers).values())
        assert len(reports) == 3
        assert count() == 1
        ids = {r.launch_id for r in reports}
        assert len(ids) == 1 and ids != {None}
        run = reports[0].run
        assert run.log_path is not None and os.path.isfile(run.log_path)
        assert _out_voltage(run.log_path) == pytest.approx(0.5)     # the run's own log
        assert os.path.isfile(run.deck_path) and os.path.isfile(run.rawfile_path)
        assert 'dc 0.5' in open(run.deck_path).read().lower() or \
            '0.5' in open(run.deck_path).read()
        # a second invocation is a new launch with a new identity
        again = list(top.run_all_tests(n_workers=workers).values())
        assert count() == 2 and {r.launch_id for r in again} != ids
    finally:
        top.close()
        child.close()
    # Private generation artifacts have their own lifetime (tests/test_run_generations.py);
    # the public files the record names stay after the owners close.
    assert os.path.isfile(run.deck_path) and os.path.isfile(run.log_path)
    assert not run.deck_path.split(os.sep)[-1].startswith('.silk-')


@needs_ngspice
@needs_pdk
def test_corner_plan_reports_each_have_their_own_run_record(tmp_path):
    dut, (on, off) = _two_bench_mos(tmp_path)
    on.set_corners(['tt', 'ss'])
    try:
        results = dut.run_all_tests()
        corners = next(r for key, r in results.items() if key.endswith('with_params'))
        reports = list(corners.reports.values())
        assert len(reports) == 2
        runs = [r.run for r in reports]
        assert all(run is not None and os.path.isfile(run.deck_path) for run in runs)
        assert len({run.launch_id for run in runs}) == 2
        assert len({run.directory for run in runs}) == 2
    finally:
        dut.close()


@needs_ngspice
@needs_pdk
def test_selected_handle_run_reconfiguration_and_evaluate_only_readback(tmp_path):
    dut, (on, off) = _two_bench_mos(tmp_path)
    top = root(tmp_path)
    try:
        top.add_module(dut, 'u0', d='vd', g='vg', s='0', b='0')
        handle = top.get_test('with_params', instance='u0')
        handle.save_device_params = True
        report = handle.run()                                   # direct selected run()
        assert len(report.op_table().rows) == 1
        assert len(handle.evaluate().op_table().rows) == 1      # evaluate-only readback
        handle.save_device_params = False                       # reconfigure
        with pytest.raises(Exception, match='save_device_params'):
            handle.evaluate().op_table()
        assert 'M1' not in str(handle.run().op_devices or '')
        with pytest.raises(Exception, match='save_device_params'):
            handle.run().op_table()
        handle.save_device_params = None                        # inherit again (DUT False)
        with pytest.raises(Exception, match='save_device_params'):
            handle.run().op_table()
        dut.save_device_params = True
        assert len(handle.run().op_table().rows) == 1
    finally:
        top.close()
        dut.close()




def _deck_dc(deck_path):
    """The DC value the deck gives source ``rail``."""
    import re
    match = re.search(r'^Vrail out 0 DC (\S+)\s*$', open(deck_path).read(), re.M)
    assert match, open(deck_path).read()
    return float(match.group(1))


def _raw_out(rawfile_path):
    """The last ``v(out)`` value in a rawfile."""
    from silk.signals.rawfile import read_rawfile
    vectors = {v.name: v for v in read_rawfile(rawfile_path).vectors}
    return float(vectors['v(out)'].data[-1])


def _snapshot(run):
    return (open(run.deck_path, 'rb').read(), open(run.rawfile_path, 'rb').read(),
            open(run.log_path, 'rb').read())


@needs_ngspice
def test_file_lifetime_reads_the_files_the_records_name(tmp_path):
    mod, tb = leaf(tmp_path)
    mod.set_output_options(mode='overwrite')
    try:
        first = next(iter(mod.run_all_tests().values())).run
        assert _deck_dc(first.deck_path) == 0.5 and _raw_out(first.rawfile_path) == 0.5
        assert _out_voltage(first.log_path) == pytest.approx(0.5)
        tb.netlist.sig_sources_dict['rail'].dc = 2
        second = next(iter(mod.run_all_tests().values())).run
        # overwrite: the reused public deck, rawfile and log hold the second attempt
        assert (second.deck_path, second.rawfile_path, second.log_path) == \
            (first.deck_path, first.rawfile_path, first.log_path)
        assert second.launch_id != first.launch_id
        assert _deck_dc(second.deck_path) == 2 and _raw_out(second.rawfile_path) == 2
        assert _out_voltage(second.log_path) == pytest.approx(2)
        mod.set_output_options(mode='new')
        tb.netlist.sig_sources_dict['rail'].dc = 3
        kept = next(iter(mod.run_all_tests().values())).run
        before = _snapshot(kept)
        assert _deck_dc(kept.deck_path) == 3 and _raw_out(kept.rawfile_path) == 3
        tb.netlist.sig_sources_dict['rail'].dc = 4
        later = next(iter(mod.run_all_tests().values())).run
        # new mode: distinct files, each with its own values, the first untouched
        assert len({kept.deck_path, later.deck_path}) == 2
        assert len({kept.rawfile_path, later.rawfile_path}) == 2
        assert len({kept.log_path, later.log_path}) == 2
        assert _deck_dc(later.deck_path) == 4 and _raw_out(later.rawfile_path) == 4
        assert _out_voltage(later.log_path) == pytest.approx(4)
        assert _snapshot(kept) == before
        assert _deck_dc(kept.deck_path) == 3 and _raw_out(kept.rawfile_path) == 3
    finally:
        mod.close()


@needs_ngspice
def test_first_failed_log_survives_the_next_failure(tmp_path):
    mod, tb = leaf(tmp_path)
    tb.add_source(sources.VoltageSource('rail2', positive='out', negative='0', dc=1.0))
    mod.set_output_options(mode='new')
    try:
        first = next(iter(mod.run_all_tests().values()))
        text = open(first.run.log_path, errors='replace').read()
        assert 'Timestep too small' in text and 'singular matrix' in text
        second = next(iter(mod.run_all_tests().values()))
        assert second.run.log_path != first.run.log_path
        assert open(first.run.log_path, errors='replace').read() == text
        assert 'singular matrix' in open(second.run.log_path, errors='replace').read()
    finally:
        mod.close()


def _mos_factory(folder, capture):
    """Same-named, otherwise identical benches that differ only in capture."""
    from silk.pdk import load_pdk
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(folder), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                             's': PortDir.GROUND, 'b': PortDir.GROUND})
    dut.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.5))
    tb = dut.create_test('op', save_device_params=capture)
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1.0))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.0))
    tb.add_sim_type(analysis.Op())
    return dut


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('captures,launches', [((True, False), 2), ((True, True), 1),
                                              ((False, False), 1)])
def test_equivalent_benches_share_only_when_capture_requests_match(
        tmp_path, monkeypatch, workers, captures, launches):
    first = _mos_factory(tmp_path / 'a', captures[0])
    second = _mos_factory(tmp_path / 'b', captures[1])
    top = root(tmp_path)
    count = _count_launches(monkeypatch, str(tmp_path / 'launches.txt'))
    try:
        top.add_module(first, 'a', d='vd', g='vg', s='0', b='0')
        top.add_module(second, 'b', d='vd', g='vg', s='0', b='0')
        reports = list(top.run_all_tests(n_workers=workers).values())
        assert len(reports) == 2
        assert count() == launches
        assert (len({r.launch_id for r in reports}) == launches)
        for report, captured in zip(reports, captures):
            if captured:
                assert len(report.op_table().rows) == 1
            else:
                with pytest.raises(Exception, match='save_device_params'):
                    report.op_table()
    finally:
        top.close()
        first.close()
        second.close()
