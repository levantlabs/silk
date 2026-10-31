"""TASK-448.7: the same testbench workflow on behavioral evidence."""
import os
from pathlib import Path
import warnings
from uuid import uuid4

import pytest

from silk import PortDir, sources
from silk.circuit.modules import Module
from silk.exceptions import CircuitError, EvidenceMismatchError, NotSupportedError
from silk.metrics import final_value
from silk.simulation import AC, Transient
from silk.simulation.conditions import ConditionState
from tests._env import needs_ngspice


_CALL_LOGS = {}


def bench(pdk, tmp_path, *, transistor=False, name="dut", child=None):
    dut = Module('PDK_skywater130', str(tmp_path / 'dut'), 'dut.cir',
                 pdk_details=pdk, module_name=name, db_path=str(tmp_path / 'runs.db'))
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    if transistor:
        dut.add(pdk.nmos1p8('m', d='vout', g='vin', s='gnd', b='gnd', w=1, l=0.15))
    else:
        from silk.sources import VoltageControlledVoltageSource
        dut.add_source(VoltageControlledVoltageSource(
            'gain', out_pos='vout', out_neg='gnd',
            ctrl_pos='vin', ctrl_neg='gnd', gain=2))
    # Each copied model retains its bench's external observation destination.
    bench_id = uuid4().hex
    calls = _CALL_LOGS[bench_id] = []
    def model(ports, t, params):
        _CALL_LOGS[bench_id].append(os.getpid())
        ports.vout.v = 2 * ports.vin.v
    dut.set_behavioral_model(model, execution='vector')
    dut.simulator = 'behavioral'
    if child is not None:
        dut.add_module(child, 'spice_child', nets={'vin': 'vin', 'vout': 'vout'})
    tb = dut.create_test('transfer')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(sources.VoltageSource('input', positive='vin', negative='gnd', dc=0.5))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('final_V', final_value, sig=tb.probe('vout'))
    tb.add_metric('difference_V', final_value, sig=tb.probe('vout') - tb.probe('vin'))
    tb.add_spec('Output', 'final_V', '>=', 0.9, key='output')
    return dut, tb, calls


@pytest.mark.parametrize('copied', [False, True])
def test_each_bench_keeps_its_external_log(pdk, tmp_path, copied):
    from types import SimpleNamespace
    from silk.simulation._behavioral_models import _clone

    child, _, child_log = bench(pdk, tmp_path / 'spice', name='spice_dut')
    child.simulator = 'ngspice'
    _, _, root_log = bench(pdk, tmp_path / 'behavioral', child=child)
    model = _clone(child._behavioral_model) if copied else child._behavioral_model
    ports = SimpleNamespace(vin=SimpleNamespace(v=.5), vout=SimpleNamespace(v=None))
    model(ports, 0, {})
    assert child_log == [os.getpid()]
    assert root_log == []
    assert ports.vout.v == 1.0


def check_report(report):
    assert report.metrics == pytest.approx({'final_V': 1.0, 'difference_V': 0.5})
    assert report.passed
    assert report.operating_conditions.origin == 'silk'
    assert report.operating_conditions.backend == 'behavioral'
    assert report.operating_conditions.scalar('corner') is ConditionState.NONE
    assert report.operating_conditions.scalar('temperature') is ConditionState.NONE
    assert 'silk, behavioral' in report.summary()
    assert Path(report.rawfile_path).is_file()


@pytest.mark.parametrize("manual", [False, True])
def test_run_and_evaluate_keep_evidence_and_measurement_source(pdk, tmp_path, manual):
    dut, tb, calls = bench(pdk, tmp_path)
    if manual:
        tb.netlist.simulator = 'behavioral'
        tb.netlist.simulate()
        check_report(tb.evaluate())
    else:
        check_report(tb.run())
    previous = tb.netlist.operating_conditions
    dut.simulator = 'ngspice'  # selection cannot relabel evidence already in memory
    check_report(tb.evaluate())
    assert tb.netlist.operating_conditions is previous
    spec = dut._specs[0]
    assert spec.latest(source='behavioral').corner is ConditionState.NONE
    assert spec.latest(source='spice') is None
    assert calls == [os.getpid()]


@pytest.mark.parametrize('workers', [None, 1, 2, 'all'])
def test_run_all_tests_keeps_closures_in_parent(pdk, tmp_path, workers):
    dut, tb, calls = bench(pdk, tmp_path)
    results = dut.run_all_tests(n_workers=workers)
    check_report(next(iter(results.values())))
    assert calls == [os.getpid()]


@pytest.mark.parametrize('workers', [1, 2])
def test_corners_of_modelled_transistors_warn_once_and_run_in_parent(pdk, tmp_path, workers):
    dut, tb, calls = bench(pdk, tmp_path, transistor=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        reports = tb.run_corners(['tt', 'ff'], n_workers=workers)
    identical = [w for w in caught if 'identical' in str(w.message)]
    assert len(identical) == 1
    for report in reports.values():
        check_report(report)
    assert {r.corner for r in reports.values()} == {'tt', 'ff'}
    assert len({r.rawfile_path for r in reports.values()}) == 2
    assert calls == [os.getpid(), os.getpid()]


def test_evaluate_refuses_ac_declaration_over_behavioral_memory(pdk, tmp_path):
    dut, tb, _ = bench(pdk, tmp_path)
    tb.netlist.simulator = 'behavioral'
    tb.netlist.simulate()
    tb.netlist.sim_analysis_dict.clear()
    tb.add_sim_type(AC('dec', 10, 1, 1e6))
    with pytest.raises(NotSupportedError, match='AC|ac'):
        tb.evaluate()


def test_store_report_uses_evidence_and_refuses_contradiction(pdk, tmp_path):
    dut, tb, _ = bench(pdk, tmp_path)
    report = tb.run()
    dut.create_circuit_version()
    dut.simulator = 'ngspice'
    dut.store_report(report)
    conn = dut._tracker.db.conn
    for table in ('simulation_results', 'spec_measurements'):
        assert {r[0] for r in conn.execute(f'SELECT source FROM {table}')} == {'behavioral'}
    with pytest.raises(EvidenceMismatchError, match='source'):
        dut.store_report(report, source='spice')
    dut.close()


def test_viewer_accepts_testbench_results_file(pdk, tmp_path):
    from silk.signals.view.emit import emit_view
    _, tb, _ = bench(pdk, tmp_path)
    tb.run()
    result = emit_view(tb.results_file, tmp_path / 'view.html')
    assert Path(result.html_path).is_file()


@pytest.mark.slow
@needs_ngspice
def test_same_test_object_has_same_metric_and_spec_names_on_both_backends(pdk, tmp_path):
    dut, tb, _ = bench(pdk, tmp_path)
    behavioral = tb.run()
    dut.simulator = 'ngspice'
    spice = tb.run()
    assert set(spice.metrics) == set(behavioral.metrics) == {'final_V', 'difference_V'}
    assert [r.name for r in spice.results] == [r.name for r in behavioral.results] == ['Output']
    assert spice.metrics == pytest.approx(behavioral.metrics)
    assert dut._specs[0].latest(source='spice') is not None
    assert dut._specs[0].latest(source='behavioral') is not None
    dut.create_circuit_version()
    with pytest.raises(EvidenceMismatchError, match='source'):
        dut.store_report(spice, source='behavioral')
    dut.store_report(spice)
    assert {r.source for r in dut.query_results(parameter='final_V')} == {'spice'}
    dut.close()


def test_declared_corners_in_run_all_tests_keep_models(pdk, tmp_path):
    dut, tb, calls = bench(pdk, tmp_path, transistor=True)
    tb.set_corners(['tt', 'ff'])
    with pytest.warns(UserWarning, match='identical') as caught:
        results = dut.run_all_tests(n_workers=2)
    assert len(caught) == 1
    reports = results['dut.transfer']
    for report in reports.values():
        check_report(report)
    assert calls == [os.getpid(), os.getpid()]


@pytest.mark.slow
@needs_ngspice
def test_mixed_tree_runs_spice_and_behavioral_tests(pdk, tmp_path):
    child, _, spice_calls = bench(pdk, tmp_path / 'spice', name='spice_dut')
    child.simulator = 'ngspice'
    root, _, calls = bench(pdk, tmp_path / 'behavioral', child=child)
    results = root.run_all_tests(n_workers=2)
    reports = list(results.values())
    assert len(reports) == 2
    by_backend = {r.operating_conditions.backend: r for r in reports}
    check_report(by_backend['behavioral'])
    assert by_backend['ngspice'].metrics == pytest.approx({'final_V': 1.0, 'difference_V': 0.5})
    assert calls == [os.getpid()]
    assert spice_calls == []


def test_behavioral_memory_needs_no_analysis_declaration(pdk, tmp_path):
    _, tb, _ = bench(pdk, tmp_path)
    loaded = tb.run()
    tb.netlist.sim_analysis_dict.clear()
    again = tb.evaluate()
    assert again.metrics == loaded.metrics
    assert again.operating_conditions.origin == 'silk'
    assert again.operating_conditions.backend == 'behavioral'


def test_task458_rename_after_test_placement_runs(pdk, tmp_path):
    """TASK-458 preserves rename-after-create_test ordering."""
    dut, tb, _ = bench(pdk, tmp_path)
    dut.module_name = 'renamed_dut'
    report = tb.run()
    assert report.sim_results.elem_dict['dut']['device'] is dut
    assert tb.report_key == 'renamed_dut.transfer'



def test_get_test_reads_own_run_all_tests_evidence(pdk, tmp_path):
    dut, created, _ = bench(pdk, tmp_path)
    report = dut.run_all_tests()['dut.transfer']
    retrieved = dut.get_test('transfer')
    assert retrieved is created
    assert retrieved.results_file == report.rawfile_path
    assert Path(retrieved.results_file).is_file()
    assert final_value(retrieved.probe('vout')) == pytest.approx(1.0)
    assert final_value(retrieved.probe('vout')) == report.metrics['final_V']
    reread = retrieved.evaluate(rawfile=retrieved.results_file)
    assert reread.metrics == report.metrics
    assert reread.rawfile_path == report.rawfile_path
