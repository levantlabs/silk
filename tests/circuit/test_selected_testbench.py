"""Selected test handles preserve authored benches and hierarchy ownership."""
import re
from pathlib import Path

import pytest

from silk import Module, sources
from silk.elements import elements
from silk.exceptions import ConfigurationError
from silk.simulation import Transient
from tests._env import needs_ngspice


def leaf(folder, name='leaf', *, waveform=None, wired=True):
    mod = Module(None, str(folder), name + '.cir', None, module_name=name)
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('r', p='out', n='vss', value=1000))
    tb = mod.create_test('step')
    if wired:
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
        tb.add_source(sources.VoltageSource('rail', positive='out', negative='0', dc=0.5, waveform=waveform))
        tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('level', lambda c: float(c.load_signal('out').y[-1]))
    return mod, tb


def root(folder):
    return Module(None, str(folder), 'top.cir', None)


def test_root_entry_is_original_bench(tmp_path):
    mod, tb = leaf(tmp_path)
    try:
        assert mod.get_test('step') is tb
        assert mod.get_test('step', instance='') is tb
        assert mod.list_test_entries()[0].test is tb
        assert mod.list_test_entries()[0].test is mod.list_test_entries()[0].test
        assert not hasattr(mod.list_test_entries()[0], 'as_placement')
    finally:
        mod.close()


def test_selected_handles_are_stable_and_do_not_select_each_other(tmp_path):
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    try:
        top.add_module(mod, 'one', out='a', vss='0')
        top.add_module(mod, 'two', out='b', vss='0')
        a = top.get_test('step', instance='one')
        b = top.get_test('step', instance='two')
        entries = top.list_test_entries()
        assert entries[0].test is a
        assert entries[1].test is b
        assert top.get_test('step', instance='one') is a
        assert a is not b
        assert a.owner_path == '$/one'
        assert b.owner_path == '$/two'
        assert tb.owner_path == '$'
        assert a.netlist is b.netlist is tb.netlist
        assert tb.results_file is None
        with pytest.raises(ConfigurationError, match='one.*two'):
            top.get_test('step', instance='missing')
        with pytest.raises(ConfigurationError, match='step'):
            top.get_test('missing', instance='one')
    finally:
        top.close()
        mod.close()


@needs_ngspice
def test_serial_root_preserves_instance_wrappers_and_netlist_identity(tmp_path):
    mod, tb = leaf(tmp_path)
    netlist = tb.netlist
    original_run = tb.run
    original_simulator = netlist.run_simulator
    calls = []

    def run(*args, **kwargs):
        assert tb.netlist is netlist
        calls.append('run')
        return original_run(*args, **kwargs)

    def simulate(*args, **kwargs):
        assert tb.netlist is netlist
        calls.append('simulate')
        return original_simulator(*args, **kwargs)

    tb.run = run
    netlist.run_simulator = simulate
    try:
        report, = mod.run_all_tests().values()
        assert not isinstance(report, dict), report
        assert report.metrics['level'] == pytest.approx(0.5)
        assert calls == ['run', 'simulate']
        assert mod.get_test('step').results_file == report.rawfile_path
        assert Path(tb.results_file).is_file()
        assert tb.report_key == mod.list_test_entries()[0].report_key
        tb.evaluate(rawfile=tb.results_file, reason='external runner')
        assert tb.report_key == mod.list_test_entries()[0].report_key
    finally:
        mod.close()


@needs_ngspice
def test_inherited_supplies_do_not_change_authored_bench_or_direct_run(tmp_path):
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    parents = []
    try:
        for name, voltage in [('low', 1.0), ('high', 2.0)]:
            carrier = Module(None, str(tmp_path), name + '.cir', None)
            carrier.set_supply_voltage(voltage)
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(mod, 'cell', out='out', vss='0')
            top.add_module(carrier, name, nets={})
            parents.append(carrier)
        authored = tb.netlist
        before = (authored.dir_folder, authored.netlistname, authored._output_mode,
                  dict(authored.simulator_options), authored._supply_voltage,
                  authored._supply_source, authored.sig_sources_dict['rail'].dc)
        reports = top.run_all_tests()
        assert [r.metrics['level'] for r in reports.values()] == pytest.approx([1, 2])
        after = (authored.dir_folder, authored.netlistname, authored._output_mode,
                 dict(authored.simulator_options), authored._supply_voltage,
                 authored._supply_source, authored.sig_sources_dict['rail'].dc)
        assert after == before
        a = top.get_test('step', instance='low.cell')
        b = top.get_test('step', instance='high.cell')
        assert a.probe('out').y[-1] == pytest.approx(1.0)
        assert b.probe('out').y[-1] == pytest.approx(2.0)
        assert tb.run().metrics['level'] == pytest.approx(0.5)
    finally:
        top.close()
        mod.close()
        for carrier in parents:
            carrier.close()


@needs_ngspice
@pytest.mark.parametrize('placed', [False, True])
def test_single_test_keeps_main_output_location(tmp_path, placed):
    mod, tb = leaf(tmp_path)
    mod.set_output_options(mode='overwrite')
    top = root(tmp_path) if placed else mod
    top.set_output_options(mode='overwrite')
    try:
        expected = Path(tb.prepare()).parent
        if placed:
            top.add_module(mod, 'only', out='out', vss='0')
        report, = top.run_all_tests().values()
        assert Path(report.rawfile_path).parent == expected
    finally:
        top.close()
        mod.close()


def test_evidence_index_is_lazy_private_transient_and_collectable(tmp_path):
    import copy
    import gc
    import pickle
    import weakref
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    top.add_module(mod, 'only', out='out', vss='0')
    assert '_test_evidence' not in top.__dict__
    assert '_test_evidence' not in dir(Module)
    handle = top.get_test('step', instance='only')
    index = top.__dict__['_test_evidence']
    ref = weakref.ref(index)
    assert '_test_evidence' not in copy.deepcopy(top).__dict__
    from silk.simulation._worker_pickle import dump_worker_netlist
    assert '_test_evidence' not in pickle.loads(dump_worker_netlist(top)).__dict__
    top.close()
    del index, top
    gc.collect()
    assert ref() is None
    with pytest.raises(ConfigurationError, match='root'):
        handle.run()
    mod.close()


@needs_ngspice
def test_overwrite_two_roots_keep_fresh_probe_readback(tmp_path):
    mod, tb = leaf(tmp_path)
    roots = [root(tmp_path), root(tmp_path)]
    try:
        for top, voltage in zip(roots, (1.0, 2.0)):
            top.set_supply_voltage(voltage)
            top.add_module(mod, 'x1', out='out', vss='0')
            top.run_all_tests()
        handles = [top.get_test('step', instance='x1') for top in roots]
        assert [h.probe('out').y[-1] for h in handles] == pytest.approx([1, 2])
        assert [h.probe('out').corner(next(iter(h.sweep_corners))).y[-1]
                for h in handles] == pytest.approx([1, 2])
    finally:
        for top in roots:
            top.close()
        mod.close()


@needs_ngspice
def test_selected_rerun_does_not_replace_sibling_waveform(tmp_path):
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    try:
        for name in ('x1', 'x2'):
            top.add_module(mod, name, out=name, vss='0')
        tb.netlist.sig_sources_dict['rail'].dc = 1.0
        top.run_all_tests()
        a = top.get_test('step', instance='x1')
        b = top.get_test('step', instance='x2')
        tb.netlist.sig_sources_dict['rail'].dc = 2.0
        report = a.run()
        assert report.metrics['level'] == pytest.approx(2.0)
        assert a.report is report
        assert a.metrics == report.metrics
        assert a.probe('out').y[-1] == pytest.approx(2.0)
        assert b.probe('out').y[-1] == pytest.approx(1.0)
    finally:
        top.close()
        mod.close()


@needs_ngspice
def test_selected_evaluate_reuses_its_own_memory(tmp_path):
    mod, tb = leaf(tmp_path)
    roots = [root(tmp_path), root(tmp_path)]
    try:
        for top, voltage in zip(roots, (1.0, 2.0)):
            top.set_supply_voltage(voltage)
            top.add_module(mod, 'x1', out='out', vss='0')
            top.run_all_tests()
        assert roots[0].get_test('step', instance='x1').evaluate().metrics['level'] == pytest.approx(1)
    finally:
        for top in roots:
            top.close()
        mod.close()


@needs_ngspice
def test_original_readback_refuses_multiple_distinct_runs(tmp_path):
    from silk.exceptions import UnboundSignalError
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    carriers = []
    try:
        for name, voltage in [('a', 1.0), ('b', 2.0)]:
            carrier = root(tmp_path)
            carrier.set_supply_voltage(voltage)
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(mod, 'cell', out='out', vss='0')
            top.add_module(carrier, name, nets={})
            carriers.append(carrier)
        top.run_all_tests()
        with pytest.raises(UnboundSignalError, match='get_test.*a.cell.*b.cell'):
            _ = tb.probe('out').y
        with pytest.raises(UnboundSignalError, match='get_test'):
            _ = tb._metric_values
        tb.run()
        assert tb.probe('out').y[-1] == pytest.approx(.5)
    finally:
        top.close()
        mod.close()
        for carrier in carriers:
            carrier.close()


@needs_ngspice
def test_failed_selected_run_drops_previous_readback(tmp_path, monkeypatch):
    from silk.exceptions import SimulationError, UnboundSignalError
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    top.add_module(mod, 'a', out='out', vss='0')
    selected = top.get_test('step', instance='a')
    try:
        selected.run()
        def fail(*args, **kwargs):
            from silk.circuit.modules_sim import _begin_result_replacement
            _begin_result_replacement(tb.netlist)
            raise SimulationError('injected failure')
        monkeypatch.setattr(tb.netlist, 'simulate', fail)
        with pytest.raises(SimulationError, match='injected'):
            selected.run()
        assert selected.results_file is None
        with pytest.raises(UnboundSignalError):
            _ = selected.probe('out').y
    finally:
        top.close()
        mod.close()


@needs_ngspice
def test_listing_prunes_removed_selection_and_evidence(tmp_path):
    import gc
    import weakref
    mod, tb = leaf(tmp_path)
    tb.add_spec('level', 'level', '>=', 0)
    top = root(tmp_path)
    top.add_module(mod, 'a', out='out', vss='0')
    selected = top.get_test('step', instance='a')
    report = selected.run()
    evidence = weakref.ref(report.sim_results)
    del report
    try:
        del top._element_mgr.elem_dict['a']
        assert top.list_test_entries() == ()
        gc.collect()
        assert evidence() is None
        assert not tb._specs[0]['spec'].measurements
        with pytest.raises(ConfigurationError, match='removed or replaced'):
            selected.run()
    finally:
        top.close()
        mod.close()


@pytest.mark.parametrize('placed', [False, True])
def test_evaluation_uses_recorded_backend_after_authored_selection_restored(tmp_path, monkeypatch, placed):
    from types import SimpleNamespace
    from silk.circuit.module_test import _RunSimResult
    from silk.circuit.test_placement import execution_scope
    from silk.simulation.conditions import OperatingConditions
    from silk.simulation.backend import SimulatorBackend
    child, tb = leaf(tmp_path)
    top = root(tmp_path) if placed else child
    if placed:
        top.add_module(child, 'x1', out='out', vss='0')
    entry = top.list_test_entries()[0]
    conditions = OperatingConditions('silk', 'xyce', None)
    result = _RunSimResult(SimpleNamespace(operating_conditions=conditions), 'tt', conditions, ())
    def evaluate(*args, **kwargs):
        reader = tb._result_netlist()
        assert reader.simulator == 'xyce'
        assert type(reader.backend) is type(SimulatorBackend.get('xyce'))
        assert tb.netlist.simulator == 'ngspice'
        return SimpleNamespace()
    monkeypatch.setattr(tb, 'evaluate', evaluate)
    try:
        with execution_scope(entry, executing=True):
            tb._evaluate_run(result)
    finally:
        top.close()
        if placed:
            child.close()


@pytest.fixture
def recorded_selection(tmp_path, monkeypatch):
    """Real selected readers and decks, with only simulator execution replaced."""
    from tests._mock_rawfile import write_complete_rawfile
    child, tb = leaf(tmp_path / 'child')
    top = root(tmp_path / 'top')
    top.add_module(child, 'a', out='a', vss='0')
    top.add_module(child, 'b', out='b', vss='0')
    selected = top.get_test('step', instance='a')
    raw = tmp_path / 'recorded.raw'
    write_complete_rawfile(raw, 'recorded\n.tran 1n 10n\n.end\n', values={'v(out)': 1.25})
    launches = []
    backend = tb.netlist.backend

    def launch(_backend, netlist_path, results_dir, **kwargs):
        launches.append(netlist_path)
        path = Path(results_dir) / 'synthetic.raw'
        write_complete_rawfile(path, Path(netlist_path).read_text(), values={'v(out)': 2.5})
        result = backend.load_raw(str(path))
        temperature = re.search(r'\btemp=([^\s]+)', Path(netlist_path).read_text())
        if temperature:
            result.ambient_temperatures = (float(temperature[1]),)
        return result

    monkeypatch.setattr(type(backend), 'run', launch)
    report = selected.evaluate(rawfile=raw)
    try:
        yield selected, tb, top, raw, report, launches
    finally:
        top.close()
        child.close()


def assert_recorded_evidence(selected, report, probe):
    assert selected.report is report
    assert selected.metrics == {'level': 1.25}
    assert Path(selected.results_file) == Path(report.rawfile_path)
    assert list(selected.sweep_corners) == ['tt']
    assert probe.y[-1] == 1.25
    assert probe.corner('tt').y[-1] == 1.25
    assert selected._result_netlist().load_signal('out').y[-1] == 1.25


@pytest.mark.parametrize('method,kwargs,error', [
    ('evaluate', {'temperature': 'nonsense'}, TypeError),
    ('run', {'temperature': 'nonsense'}, TypeError),
    ('run', {'unexpected': True}, TypeError),
    ('evaluate', {'unexpected': True}, TypeError),
    ('run', {'run_label': '../outside'}, ValueError),
    ('run_monte_carlo', {'n': 0}, ValueError),
    ('prepare', {'temperature': 'nonsense'}, TypeError),
    ('run_corners', {'corners': 1}, TypeError),
])
def test_selected_refusal_preserves_evidence(recorded_selection, method, kwargs, error):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    if method == 'evaluate':
        kwargs = dict(kwargs, rawfile=raw)
    with pytest.raises(error):
        getattr(selected, method)(**kwargs)
    assert_recorded_evidence(selected, report, probe)
    assert launches == []


def test_selected_failed_explicit_replacement_drops_evidence(recorded_selection, tmp_path):
    from silk.exceptions import SimulationError, UnboundSignalError
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    with pytest.raises(SimulationError):
        selected.evaluate(rawfile=tmp_path / 'missing.raw')
    assert selected.report is None
    assert selected.metrics == {}
    assert selected.results_file is None
    with pytest.raises(UnboundSignalError):
        _ = probe.y
    assert report.metrics['level'] == 1.25


@pytest.mark.parametrize('method', ['run', 'evaluate'])
def test_selected_success_replaces_all_readback(recorded_selection, method):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    if method == 'evaluate':
        from tests._mock_rawfile import write_complete_rawfile
        write_complete_rawfile(raw, 'replacement\n.tran 1n 10n\n.end\n',
                               values={'v(out)': 2.5})
        replacement = selected.evaluate(rawfile=raw, corner='ff')
        label = 'ff'
    else:
        replacement = selected.run()
        label = 'tt'
    assert replacement is not report
    assert selected.report is replacement
    assert selected.metrics == {'level': 2.5}
    assert list(selected.sweep_corners) == [label]
    assert probe.y[-1] == 2.5
    assert probe.corner(label).y[-1] == 2.5
    assert report.metrics['level'] == 1.25
    assert report.sim_results.load_signal('out').y[-1] == 1.25


@pytest.mark.parametrize('started', [False, None, True])
@pytest.mark.parametrize('runner', ['run', 'plan', 'corners', 'parent', 'parent_plan'])
def test_selected_outcome_boundary_and_collection_routing(
        recorded_selection, monkeypatch, started, runner):
    from silk.circuit.module_test import _RunOutcome
    from silk.circuit.spec_report import CornerRunError
    from silk.exceptions import SimulationError, UnboundSignalError
    from silk.simulation.executor import RunOutcome
    selected, tb, top, raw, report, launches = recorded_selection
    sibling = top.get_test('step', instance='b')
    sibling_report = sibling.evaluate(rawfile=raw)
    probe = selected.probe('out')

    def refuse(spec):
        if started:
            from silk.circuit.modules_sim import _begin_result_replacement
            _begin_result_replacement(spec.netlist)
        raise SimulationError('boundary control')

    def worker_outcomes(tasks, **kwargs):
        return [RunOutcome(index=i, run_dir=task.run_dir,
                           error='worker connection lost' if started is None else None,
                           value=None if started is None else _RunOutcome(
                               error=SimulationError('boundary control'),
                               replacement_started=started))
                for i, task in enumerate(tasks)]

    monkeypatch.setattr(tb, '_simulate_run', refuse)
    if runner != 'run':
        monkeypatch.setattr('silk.simulation.executor.run_tasks', worker_outcomes)
    if runner in {'plan', 'parent_plan'}:
        tb.set_corners(['ss'])
    if runner.startswith('parent'):
        results = top.run_all_tests(n_workers=2)
        assert all('error' in value for value in results.values())
    elif runner == 'corners':
        with pytest.raises(CornerRunError):
            selected.run_corners(['ss'])
    else:
        with pytest.raises(SimulationError):
            selected.run()
    if started:
        assert selected.report is None
        assert selected.metrics == {}
        assert selected.results_file is None
        with pytest.raises(UnboundSignalError):
            _ = probe.y
    else:
        assert_recorded_evidence(selected, report, probe)
    if not runner.startswith('parent') or not started:
        assert sibling.report is sibling_report
        assert sibling.probe('out').y[-1] == 1.25
    assert report.metrics == {'level': 1.25}
    assert report.sim_results.load_signal('out').y[-1] == 1.25
    assert launches == []


def test_selected_two_roots_and_sibling_are_isolated(recorded_selection, tmp_path):
    from silk.exceptions import SimulationError
    selected, tb, top, raw, report, launches = recorded_selection
    child = tb.module
    other = root(tmp_path / 'other')
    other.add_module(child, 'a', out='out', vss='0')
    sibling = top.get_test('step', instance='b')
    foreign = other.get_test('step', instance='a')
    try:
        sibling_report = sibling.evaluate(rawfile=raw)
        foreign_report = foreign.evaluate(rawfile=raw)
        with pytest.raises(TypeError):
            selected.run(temperature='nonsense')
        selected.run()
        assert sibling.report is sibling_report
        assert foreign.report is foreign_report
        assert sibling.probe('out').y[-1] == foreign.probe('out').y[-1] == 1.25
        with pytest.raises(SimulationError):
            selected.evaluate(rawfile=raw.parent / 'missing.raw')
        assert sibling.report is sibling_report
        assert foreign.report is foreign_report
        assert sibling.probe('out').y[-1] == foreign.probe('out').y[-1] == 1.25
    finally:
        other.close()


def test_selected_new_waveform_metric_error_does_not_restore_old_evidence(recorded_selection):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')

    def fail(circuit):
        raise ValueError('invalid extraction window')

    tb.add_metric('broken', fail)
    replacement = selected.run()
    assert replacement.metric_errors['broken']
    assert selected.report is replacement
    assert selected.metrics['level'] == 2.5
    assert probe.y[-1] == 2.5
    assert report.metrics == {'level': 1.25}


def test_selected_prepare_keeps_evidence(recorded_selection):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    assert Path(selected.prepare()).is_file()
    assert_recorded_evidence(selected, report, probe)
    assert launches == []


@pytest.mark.parametrize('runner', ['plan', 'corners'])
def test_selected_partial_collection_retains_only_new_successes(
        recorded_selection, monkeypatch, runner):
    from silk.circuit.module_test import _RunOutcome, _simulate_run_outcome
    from silk.circuit.spec_report import CornerRunError
    from silk.exceptions import SimulationError
    from silk.simulation.executor import RunOutcome
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')

    def mixed_outcomes(tasks, **kwargs):
        outcomes = []
        for i, task in enumerate(tasks):
            spec, = task.args
            value = (_simulate_run_outcome(spec) if spec.corner_str == 'tt' else
                     _RunOutcome(error=SimulationError('replacement failed'),
                                 replacement_started=True))
            outcomes.append(RunOutcome(index=i, run_dir=task.run_dir, value=value))
        return outcomes

    monkeypatch.setattr('silk.simulation.executor.run_tasks', mixed_outcomes)
    if runner == 'plan':
        tb.set_corners(['tt', 'ss'])
        results = selected.run()
    else:
        with pytest.raises(CornerRunError) as caught:
            selected.run_corners(['tt', 'ss'])
        results = caught.value.results
    assert list(results.reports) == ['tt']
    assert selected.metrics == {'level': 2.5}
    assert probe.y[-1] == 2.5
    assert probe.corner('tt').y[-1] == 2.5
    assert list(selected.sweep_corners) == ['tt']


def test_selected_declared_plan_preparation_refusal_keeps_corner_access(recorded_selection):
    from silk.exceptions import SimulationError
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    selected.set_corners(['ss'])
    with pytest.raises(SimulationError, match='every declared run failed'):
        selected.run(temperature='nonsense')
    assert_recorded_evidence(selected, report, probe)
    assert launches == []


@pytest.mark.parametrize('empty', [[], iter(())], ids=['list', 'iterator'])
def test_selected_empty_corners_keep_whole_evidence(recorded_selection, empty):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    result = selected.run_corners(empty)
    assert not result.reports
    assert_recorded_evidence(selected, report, probe)
    assert selected._mc_samples is None
    assert launches == []


@pytest.mark.parametrize('parent_plan', [False, True], ids=['single', 'plan'])
def test_parent_fail_fast_reconciles_later_replacement(
        recorded_selection, monkeypatch, parent_plan):
    from silk.circuit.module_test import _RunOutcome
    from silk.exceptions import SimulationConfigurationError, SimulationError, UnboundSignalError
    from silk.simulation.executor import RunOutcome
    selected, tb, top, raw, report, launches = recorded_selection
    later = top.get_test('step', instance='b')
    held = later.evaluate(rawfile=raw)
    top.add_module(tb.module, 'c', out='c', vss='0')
    unknown = top.get_test('step', instance='c')
    unknown_report = unknown.evaluate(rawfile=raw)
    probe = later.probe('out')
    refusal = SimulationConfigurationError('early refusal')
    if parent_plan:
        tb.set_corners(['tt'])

    def outcomes(tasks, **kwargs):
        # Keep placements as distinct consumers of already executed work.
        values = [_RunOutcome(error=refusal, replacement_started=False),
                  _RunOutcome(error=SimulationError('replacement failed'), replacement_started=True),
                  _RunOutcome(error=RuntimeError('unknown transport'), replacement_started=None)]
        return [RunOutcome(index=i, run_dir=task.run_dir, value=value)
                for i, (task, value) in enumerate(zip(tasks, values))]

    monkeypatch.setattr('silk.circuit.test_execution.equivalence', lambda spec: None)
    monkeypatch.setattr('silk.simulation.executor.run_tasks', outcomes)
    with pytest.raises(SimulationConfigurationError) as caught:
        top.run_all_tests(n_workers=2)
    assert caught.value is refusal
    assert_recorded_evidence(selected, report, selected.probe('out'))
    assert later.report is None
    assert later.metrics == {}
    assert later.results_file is None
    assert list(later.sweep_corners) == []
    with pytest.raises(UnboundSignalError):
        _ = probe.y
    assert held.sim_results.load_signal('out').y[-1] == 1.25
    assert_recorded_evidence(unknown, unknown_report, unknown.probe('out'))
    assert launches == []


@pytest.mark.parametrize('order', ['failure_first', 'refusal_first'])
def test_selected_plan_fail_fast_reconciles_entire_batch(
        recorded_selection, monkeypatch, order):
    from silk.circuit.module_test import _RunOutcome
    from silk.exceptions import SimulationConfigurationError, SimulationError, UnboundSignalError
    from silk.simulation.executor import RunOutcome
    selected, tb, top, raw, report, launches = recorded_selection
    sibling = top.get_test('step', instance='b')
    sibling_report = sibling.evaluate(rawfile=raw)
    probe = selected.probe('out')
    selected.set_corners(['tt', 'ss'])
    refusal = SimulationConfigurationError('batch refusal')

    def outcomes(tasks, **kwargs):
        values = [_RunOutcome(error=SimulationError('replacement failed'), replacement_started=True),
                  _RunOutcome(error=refusal, replacement_started=False)]
        if order == 'refusal_first':
            values.reverse()
        return [RunOutcome(index=i, run_dir=task.run_dir, value=value)
                for i, (task, value) in enumerate(zip(tasks, values))]

    monkeypatch.setattr('silk.simulation.executor.run_tasks', outcomes)
    with pytest.raises(SimulationConfigurationError) as caught:
        selected.run()
    assert caught.value is refusal
    assert selected.report is None
    assert selected.metrics == {}
    assert selected.results_file is None
    assert list(selected.sweep_corners) == []
    with pytest.raises(UnboundSignalError):
        _ = probe.y
    assert_recorded_evidence(sibling, sibling_report, sibling.probe('out'))
    assert report.sim_results.load_signal('out').y[-1] == 1.25
    assert launches == []


@pytest.mark.parametrize('wired', [False, True])
@pytest.mark.parametrize('method,kwargs,fragment', [
    ('set_temperature', {'temperature_c': 85}, 'temp=85.0'),
    ('set_supply_voltage', {'volts': 1.2, 'source': 'rail'}, 'Vrail out 0 DC 1.2'),
    ('set_simulator_options', {'reltol': 1e-4}, 'reltol=0.0001'),
])
def test_selected_setters_persist_on_shared_bench(tmp_path, wired, method, kwargs, fragment):
    child, tb = leaf(tmp_path / 'child', wired=wired)
    top, other = root(tmp_path / 'top'), root(tmp_path / 'other')
    try:
        top.set_temperature(50)
        top.set_supply_voltage(1.8, source='rail')
        child.set_simulator_options(reltol=1e-3, abstol=1e-9)
        top.add_module(child, 'a', out='a', vss='0')
        top.add_module(child, 'b', out='b', vss='0')
        other.add_module(child, 'c', out='c', vss='0')
        selected = top.get_test('step', instance='a')
        getattr(selected, method)(**kwargs)
        if not wired:
            tb.add_module_under_test({'out': 'out', 'vss': '0'})
            tb.add_source(sources.VoltageSource('rail', positive='out', negative='0', dc=0.5))
            tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
        for handle in (selected, top.get_test('step', instance='b'),
                       other.get_test('step', instance='c'), tb):
            assert handle.netlist is tb.netlist
            assert fragment in Path(handle.prepare()).read_text()
        deck = Path(selected.prepare()).read_text()
        assert 'abstol=1e-09' in deck
        if method != 'set_temperature':
            assert 'temp=50.0' in deck
        if method != 'set_simulator_options':
            assert 'reltol=0.001' in deck
        foreign = other.get_test('step', instance='c')
        foreign.set_temperature(95)
        assert tb.netlist.simulator_options['temp'] == 95
        assert 'temp=95.0' in Path(selected.prepare()).read_text()
    finally:
        other.close()
        top.close()
        child.close()


@pytest.mark.parametrize('method,kwargs', [
    ('set_corner', {'corner': 'tt'}),
    ('set_corners', {'corners': ['tt']}),
    ('set_temperature', {'temperature_c': 85}),
    ('set_supply_voltage', {'volts': 1.2, 'source': 'rail'}),
    ('set_simulator_options', {'reltol': 1e-4}),
])
def test_selected_fluent_calls_keep_placement(recorded_selection, method, kwargs):
    selected, tb, top, raw, prior, launches = recorded_selection
    top.set_temperature(50)
    top.set_supply_voltage(1.8, source='rail')
    returned = getattr(selected, method)(**kwargs)
    assert returned is selected
    top.set_output_options(mode='overwrite')
    chained = Path(returned.prepare()).read_bytes()
    assert chained == Path(selected.prepare()).read_bytes()
    report = returned.run()
    # A run with declared corners returns its existing report collection.
    if method == 'set_corners':
        assert selected.report is report
        report = report.reports['tt']
    else:
        assert selected.report is report
    assert report.sim_results.operating_conditions.temperature_c == (85 if method == 'set_temperature' else 50)
    unchained = selected.run()
    if method == 'set_corners':
        unchained = unchained.reports['tt']
    assert unchained.sim_results.operating_conditions == report.sim_results.operating_conditions
    assert unchained.owner_path == report.owner_path == '$/a'
    assert selected.probe('out').y[-1] == 2.5
    assert top.get_test('step', instance='b').report is None


def _condition_state(netlist):
    return (netlist._simulator, netlist.netlister, dict(netlist.simulator_options),
            netlist._inherited_simulator_option_keys, netlist._temperature_basis,
            netlist._supply_voltage, netlist._supply_source,
            tuple((name, source.dc) for name, source in netlist.sig_sources_dict.items()
                  if hasattr(source, 'dc')))


@pytest.mark.parametrize('outer_executes', [False, True])
@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('same_entry', [False, True])
def test_selected_nested_execution_restores_conditions(
        recorded_selection, monkeypatch, tmp_path, request, outer_executes, fail, same_entry):
    from silk.circuit.test_placement import execution_scope, current_placement
    import silk.circuit.module_test as module_test
    selected, tb, top, raw, report, launches = recorded_selection
    top.set_temperature(50)
    top.set_supply_voltage(1.8, source='rail')
    other = root(tmp_path / 'nested-other')
    request.addfinalizer(other.close)
    other.set_temperature(70)
    other.set_supply_voltage(2.2, source='rail')
    other.add_module(tb.module, 'c', out='c', vss='0')
    inner = selected if same_entry else other.get_test('step', instance='c')
    # Persistent caller edit precedes both temporary scopes.
    selected.set_simulator_options(reltol=1e-4)
    assert tb.netlist.simulator_options['reltol'] == 1e-4
    before = _condition_state(tb.netlist)
    configure = module_test._configure_conditions

    def configured(*args, **kwargs):
        result = configure(*args, **kwargs)
        assert current_placement(tb).executing
        assert tb.netlist.simulator_options['temp'] == (50 if same_entry else 70)
        assert tb.netlist.sig_sources_dict['rail'].dc == (1.8 if same_entry else 2.2)
        if fail:
            raise RuntimeError('after condition configuration')
        return result

    monkeypatch.setattr(module_test, '_configure_conditions', configured)
    with execution_scope(selected._entry(), executing=outer_executes):
        outer = current_placement(tb)
        if outer_executes:
            configure(tb, tb.netlist)
        outer_state = _condition_state(tb.netlist)
        if fail:
            with pytest.raises(RuntimeError, match='after condition configuration'):
                inner.prepare()
        else:
            inner.prepare()
        assert current_placement(tb) is outer
        # An executing outer selection owns cleanup; a nested binding restores it.
        assert _condition_state(tb.netlist) == outer_state
        assert_recorded_evidence(selected, report, selected.probe('out'))
    assert current_placement(tb) is None
    assert _condition_state(tb.netlist) == before
    assert_recorded_evidence(selected, report, selected.probe('out'))


@pytest.mark.parametrize('method,kwargs', [
    ('set_temperature', {'temperature_c': 85}),
    ('set_temperature', {'temperature_c': None}),
    ('set_supply_voltage', {'volts': 1.2, 'source': 'rail'}),
    ('set_supply_voltage', {'volts': None}),
    ('set_simulator_options', {'reltol': 1e-4}),
])
def test_selected_setters_leave_retained_evidence(recorded_selection, method, kwargs):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    reader = selected._result_netlist()
    results = selected.results_file
    getattr(selected, method)(**kwargs)
    assert_recorded_evidence(selected, report, probe)
    assert selected._result_netlist() is reader
    assert selected.results_file == results
    assert launches == []


@pytest.mark.parametrize('method,kwargs,error,match', [
    ('set_temperature', {'temperature_c': True}, TypeError, 'Temperature'),
    ('set_temperature', {'temperature_c': -274}, ValueError, '273.15'),
    ('set_simulator_options', {'temp': True}, TypeError, 'Temperature'),
    ('set_supply_voltage', {'volts': None, 'source': 'rail'}, ValueError, 'source'),
])
def test_selected_invalid_setters_preserve_state(recorded_selection, method, kwargs, error, match):
    selected, tb, top, raw, report, launches = recorded_selection
    before = _condition_state(tb.netlist)
    with pytest.raises(error, match=match):
        getattr(selected, method)(**kwargs)
    assert _condition_state(tb.netlist) == before
    assert_recorded_evidence(selected, report, selected.probe('out'))


def test_selected_clearing_and_source_choice(recorded_selection):
    selected, tb, top, raw, report, launches = recorded_selection
    tb.add_source(sources.VoltageSource('aux', positive='aux', negative='0', dc=0.7))
    top.set_temperature(50)
    top.set_supply_voltage(1.8, source='rail')
    selected.set_temperature(85)
    selected.set_supply_voltage(1.2, source='aux')
    deck = Path(selected.prepare()).read_text()
    assert 'Vaux aux 0 DC 1.2' in deck
    assert 'Vrail out 0 DC 0.5' in deck
    selected.set_temperature(None)
    selected.set_supply_voltage(None)
    deck = Path(selected.prepare()).read_text()
    assert 'temp=' not in deck
    assert 'Vaux aux 0 DC 0.7' in deck
    assert 'Vrail out 0 DC 0.5' in deck
    assert_recorded_evidence(selected, report, selected.probe('out'))


def test_selected_native_temperature_clearing_masks_inheritance(recorded_selection):
    selected, tb, top, raw, report, launches = recorded_selection
    top.set_temperature(50)
    selected.set_simulator_options(temp=85)
    assert 'temp=85' in Path(selected.prepare()).read_text()
    selected.set_simulator_options(temp=None)
    assert 'temp=' not in Path(selected.prepare()).read_text()
    assert 'temp' not in tb.netlist.simulator_options
    assert_recorded_evidence(selected, report, selected.probe('out'))


@pytest.mark.parametrize('empty_kind', ['list', 'tuple', 'iterator', 'exhausted'])
@pytest.mark.parametrize('retained', [(), ('tt',), ('tt', 'ff')])
@pytest.mark.parametrize('placed', [False, True], ids=['authored', 'selected'])
def test_empty_corner_request_preserves_evidence(
        recorded_selection, tmp_path, monkeypatch, empty_kind, retained, placed):
    from silk.circuit.module_test import ModuleTestbench
    from silk.circuit.spec_report import CornerResults
    from silk.exceptions import UnboundSignalError
    selected, target, top, _, _, launches = recorded_selection
    mod, authored = leaf(tmp_path / 'direct')
    bench = (selected if retained else top.get_test(
        'step', instance='b')) if placed else authored
    assert placed or type(bench) is ModuleTestbench
    try:
        reports = bench.run_corners(retained).reports if retained else {}
        report_readers = {key: value.sim_results for key, value in reports.items()}
        probe = bench.probe('out')
        table = bench.sweep_corners
        generations = getattr(bench, '_sweep_generations', None)
        metrics = dict(getattr(bench, '_metric_values', {}))
        reader = bench._result_netlist()
        path = bench.results_file
        report = bench.report if placed else None
        files = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        before = {corner: probe.corner(corner).y.copy() for corner in retained}
        if len(retained) == 2:
            with pytest.raises(UnboundSignalError):
                _ = probe.y
        elif retained:
            assert probe.y[-1] == 2.5
        target = target if placed else authored

        def refuse_prepare(*args, **kwargs):
            pytest.fail('empty request prepared a run')
        monkeypatch.setattr(target, '_prepare_run', refuse_prepare)
        launches.clear()
        empty = {'list': [], 'tuple': (), 'iterator': iter(()),
                 'exhausted': iter(('tt',))}[empty_kind]
        if empty_kind == 'exhausted':
            next(empty)
        result = bench.run_corners(empty)
        assert isinstance(result, CornerResults)
        assert result.reports == {}
        assert result.passed is True
        assert bench.sweep_corners == table
        assert getattr(bench, '_sweep_generations', None) is generations
        assert getattr(bench, '_metric_values', {}) == metrics
        assert bench._result_netlist() is reader
        assert bench.results_file == path
        assert (bench.report if placed else None) is report
        assert launches == []
        assert {p: p.read_bytes() for p in files} == files
        for corner in retained:
            assert reports[corner].sim_results is report_readers[corner]
            assert reports[corner].metrics == {'level': 2.5}
            assert (probe.corner(corner).y == before[corner]).all()
        if len(retained) == 2:
            with pytest.raises(UnboundSignalError):
                _ = probe.y
        elif retained:
            assert probe.y[-1] == 2.5
        else:
            with pytest.raises(UnboundSignalError):
                _ = probe.y
        with pytest.raises(ValueError):
            bench.run_corners([], n_workers=0)
    finally:
        mod.close()


@pytest.mark.parametrize('placed', [False, True], ids=['authored', 'selected'])
@pytest.mark.parametrize('failed', [('ss',), ('tt', 'ss')], ids=['partial', 'all'])
def test_nonempty_corner_failures_keep_publication_rules(
        recorded_selection, tmp_path, monkeypatch, placed, failed):
    from silk.circuit.spec_report import CornerRunError
    selected, tb, _, _, _, launches = recorded_selection
    mod, direct = leaf(tmp_path / 'direct')
    bench = selected if placed else direct
    target = tb if placed else direct
    try:
        bench.run_corners(['ff'])
        prepare = target._prepare_run
        def prepare_or_refuse(corner, **kwargs):
            if corner in failed:
                raise ValueError('corner preparation refused')
            return prepare(corner, **kwargs)
        monkeypatch.setattr(target, '_prepare_run', prepare_or_refuse)
        with pytest.raises(CornerRunError) as caught:
            bench.run_corners(['tt', 'ss'])
        assert list(caught.value.results.reports) == ([] if len(failed) == 2 else ['tt'])
        assert list(bench.sweep_corners) == (
            ['ff'] if placed and len(failed) == 2 else
            [] if len(failed) == 2 else ['tt'])
    finally:
        mod.close()


@pytest.mark.parametrize('explicit_none', [False, True], ids=['omitted', 'none'])
def test_authored_default_corners_still_prepare(recorded_selection, tmp_path, monkeypatch, explicit_none):
    from silk.circuit.spec_report import CornerRunError
    mod, tb = leaf(tmp_path / 'defaults')
    prepared = []
    def refuse(corner, **kwargs):
        prepared.append(corner)
        raise ValueError('default corner control')
    monkeypatch.setattr(tb, '_prepare_run', refuse)
    try:
        with pytest.raises(CornerRunError):
            tb.run_corners(**({'corners': None} if explicit_none else {}))
        assert prepared == ['tt', 'ff', 'ss', 'sf', 'fs']
    finally:
        mod.close()


@needs_ngspice
@pytest.mark.parametrize('n_workers', [None, 2])
def test_condition_grids_resolve_per_selected_placement_and_restore(tmp_path, n_workers):
    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    parents = []
    try:
        for name, voltages in [('low', [1.0, 1.1]), ('high', [2.0, 2.1])]:
            carrier = root(tmp_path)
            carrier.set_supply_voltages(voltages, source='rail')
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(mod, 'cell', out='out', vss='0')
            top.add_module(carrier, name, nets={})
            parents.append(carrier)
        before = (dict(tb.netlist.simulator_options), tb.netlist.sig_sources_dict['rail'].dc)
        collections = list(top.run_all_tests(n_workers=n_workers).values())
        assert [list(result.reports) for result in collections] == [
            [('tt', None, 1.0), ('tt', None, 1.1)],
            [('tt', None, 2.0), ('tt', None, 2.1)]]
        assert [[r.metrics['level'] for r in result.reports.values()]
                for result in collections] == [[1.0, 1.1], [2.0, 2.1]]
        assert (dict(tb.netlist.simulator_options), tb.netlist.sig_sources_dict['rail'].dc) == before
        low = top.get_test('step', instance='low.cell')
        high = top.get_test('step', instance='high.cell')
        assert low.probe('out').point(('tt',None,1.0)).y[-1] == pytest.approx(1.0)
        assert high.probe('out').point(('tt',None,2.0)).y[-1] == pytest.approx(2.0)
        import matplotlib.pyplot as plt
        import numpy as np
        try:
            for selected, values, name in [(low, [1.0, 1.1], 'low.cell'),
                                           (high, [2.0, 2.1], 'high.cell')]:
                ax = selected.probe('out').plot()
                assert len(ax.lines) == 2
                for line, value in zip(ax.lines, values):
                    np.testing.assert_allclose(line.get_ydata(), value)
                    assert name in line.get_label()
                    assert str(('tt', None, value)) in line.get_label()
                plt.close('all')
        finally:
            plt.close('all')
    finally:
        top.close()
        mod.close()
        for parent in parents:
            parent.close()


@pytest.mark.parametrize('name', [
    'corner', 'monte_carlo', 'owner_path', 'report_key', 'results_file', 'sweep_corners',
])
def test_selected_property_read_only_fresh(tmp_path, name):
    child, tb = leaf(tmp_path / 'child', wired=False)
    top = root(tmp_path / 'top')
    try:
        top.add_module(child, 'a', out='a', vss='0')
        selected = top.get_test('step', instance='a')
        before = dict(tb.__dict__)
        for target in (tb, selected):
            with pytest.raises(AttributeError) as caught:
                setattr(target, name, 'shadow')
            assert name in str(caught.value)
            assert name not in target.__dict__
        assert tb.__dict__ == before
        assert not top._test_evidence.records
        assert not list(tmp_path.rglob('*.cir'))
    finally:
        top.close()
        child.close()


@pytest.mark.parametrize('name', [
    'corner', 'monte_carlo', 'owner_path', 'report_key', 'results_file', 'sweep_corners',
])
def test_selected_property_read_only_retains_evidence(recorded_selection, name):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    reader = selected._result_netlist()
    files = {p: p.read_bytes() for p in Path(top.dir_folder).parent.rglob('*') if p.is_file()}
    with pytest.raises(AttributeError) as caught:
        setattr(selected, name, 'shadow')
    assert name in str(caught.value)
    assert name not in selected.__dict__
    assert_recorded_evidence(selected, report, probe)
    assert selected._result_netlist() is reader
    assert files == {p: p.read_bytes() for p in Path(top.dir_folder).parent.rglob('*') if p.is_file()}
    assert launches == []


def test_selected_property_corners_shared_validated_and_retained(recorded_selection):
    from silk.exceptions import SimulationConfigurationError, UnknownCornerError
    selected, tb, top, raw, report, launches = recorded_selection
    sibling = top.get_test('step', instance='b')
    probe = selected.probe('out')
    reader = selected._result_netlist()
    files = {p: p.read_bytes() for p in raw.parent.rglob('*') if p.is_file()}
    selected.corners = ['ss', 'tt']
    for handle in (selected, sibling, tb):
        assert handle.corners == ('ss', 'tt')
    for invalid, error in [(1, TypeError), ('tt', TypeError),
                           ([''], UnknownCornerError)]:
        with pytest.raises(error):
            selected.corners = invalid
        assert tb.corners == ('ss', 'tt')
    selected.corners = ['ss', 'ss']
    assert selected.corners == sibling.corners == tb.corners == ('ss', 'ss')
    with pytest.raises(SimulationConfigurationError, match='Repeated|repeated'):
        selected.run()
    selected.corners = None
    assert selected.corners is sibling.corners is tb.corners is None
    assert 'corners' not in selected.__dict__
    assert_recorded_evidence(selected, report, probe)
    assert selected._result_netlist() is reader
    assert files == {p: p.read_bytes() for p in raw.parent.rglob('*') if p.is_file()}
    assert launches == []
    selected.corners = ['ss', 'tt']
    results = selected.run()
    assert list(results.reports) == ['ss', 'tt']
    assert len(launches) == 2
    assert selected.owner_path == '$/a'
    assert tb.owner_path == '$'


def test_selected_property_save_device_params_route(recorded_selection):
    selected, tb, top, raw, report, launches = recorded_selection
    probe = selected.probe('out')
    for value in (None, True, False):
        selected.save_device_params = value
        assert tb.save_device_params is value
        assert selected.save_device_params is value
        with pytest.raises(TypeError):
            selected.save_device_params = 'yes'
        assert tb.save_device_params is value
    assert 'save_device_params' not in selected.__dict__
    assert_recorded_evidence(selected, report, probe)
    assert launches == []


def test_selected_property_inherited_descriptor_scope_and_copy(tmp_path):
    from copy import deepcopy
    from silk.circuit.module_test import ModuleTestbench

    class PublicProperty(property):
        pass

    class ParentBench(ModuleTestbench):
        @PublicProperty
        def forbidden(self):
            raise AssertionError('property discovery invoked a getter')

        @PublicProperty
        def note(self):
            raise AssertionError('assignment invoked a getter')

        @note.setter
        def note(self, value):
            self._note = (value, self.owner_path)

    class ChildBench(ParentBench):
        pass

    child, tb = leaf(tmp_path / 'child', wired=False)
    carrier = root(tmp_path / 'carrier')
    carrier.define_module_name('carrier')
    carrier.define_module_ports({})
    top = root(tmp_path / 'top')
    try:
        tb.__class__ = ChildBench
        carrier.add_module(child, 'cell', out='out', vss='0')
        top.add_module(carrier, 'outer', nets={})
        selected = top.get_test('step', instance='outer.cell')
        with pytest.raises(AttributeError, match='forbidden'):
            selected.forbidden = 1
        assert not top._test_evidence.records
        selected.note = 3
        assert tb._note == (3, '$/outer/cell')
        assert tb.owner_path == '$'
        assert 'note' not in selected.__dict__
        selected.flag = 4
        selected._private = 5
        assert selected.flag == 4 and selected._private == 5
        assert 'flag' not in tb.__dict__
        copied = deepcopy(selected)
        copied.note = 6
        assert copied._authored._note == (6, '$/outer/cell')
        assert tb._note == (3, '$/outer/cell')
        copied._root().close()
    finally:
        top.close()
        carrier.close()
        child.close()


def test_selected_property_corners_unknown_pdk_corner(tmp_path):
    from silk.exceptions import UnknownCornerError
    from silk.pdk import load_pdk
    child = Module('sky130', str(tmp_path / 'child'), 'leaf.cir', load_pdk('sky130'),
                   module_name='leaf')
    child.define_module_ports({})
    tb = child.create_test('step')
    top = root(tmp_path / 'top')
    try:
        top.add_module(child, 'a', nets={})
        selected = top.get_test('step', instance='a')
        selected.corners = ['ss']
        with pytest.raises(UnknownCornerError):
            selected.corners = ['unknown']
        assert tb.corners == selected.corners == ('ss',)
        assert not list(tmp_path.rglob('*.cir'))
    finally:
        top.close()
        child.close()


@pytest.mark.parametrize('nested', [False, True])
def test_corner_expression_keeps_selected_readers_separate(tmp_path, nested):
    import numpy as np
    from tests._mock_rawfile import write_complete_rawfile
    child, authored = leaf(tmp_path / 'child')
    top = root(tmp_path / 'top')
    middle = root(tmp_path / 'middle')
    middle.define_module_name('middle')
    middle.define_module_ports({})
    try:
        parent = middle if nested else top
        parent.add_module(child, 'one', out='a', vss='0')
        parent.add_module(child, 'two', out='b', vss='0')
        if nested:
            top.add_module(middle, 'outer', nets={})
        prefix = 'outer.' if nested else ''
        benches = [top.get_test('step', instance=prefix + name) for name in ('one', 'two')]
        reports = []
        for bench, value, name in zip(benches, (1.0, 9.0), ('first', 'second')):
            raw = tmp_path / (name + '.raw')
            write_complete_rawfile(raw, 'test\n.tran 1n 10n\n.end\n', values={'v(out)': value})
            reports.append(bench.evaluate(corner='tt', rawfile=raw))
        readers = [bench._result_netlist() for bench in benches]
        files = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
        a, b = [bench.probe('out') for bench in benches]
        for left, right, difference in [(a, b, -8), (b, a, 8)]:
            np.testing.assert_allclose((left.corner('tt') - right.corner('tt')).y, difference)
            np.testing.assert_allclose((left + right).corner('tt').y, 10)
            np.testing.assert_allclose((left - right).corner('tt').y, difference)
        cache = {}
        first = benches[0]._corner_netlist('tt', cache)
        assert benches[0]._corner_netlist('tt', cache) is first
        assert benches[1]._corner_netlist('tt', cache) is not first
        for bench, report, reader in zip(benches, reports, readers):
            assert bench.report is report
            assert bench._result_netlist() is reader
        assert {p: p.read_bytes() for p in files} == files
        benches[0].evaluate(corner='tt', rawfile=tmp_path / 'second.raw')
        np.testing.assert_allclose((a + b).corner('tt').y, 18)
    finally:
        top.close()
        middle.close()
        child.close()


def test_reference_first_plot_keeps_selected_placement(tmp_path, monkeypatch):
    import numpy as np
    import matplotlib.pyplot as plt
    from tests._mock_rawfile import write_complete_rawfile
    child, authored = leaf(tmp_path / 'child')
    top = root(tmp_path / 'top')
    try:
        top.add_module(child, 'one', out='a', vss='0')
        top.add_module(child, 'two', out='b', vss='0')
        selected = [top.get_test('step', instance=name) for name in ('one', 'two')]
        for bench, value in zip(selected, [1, 9]):
            raw = tmp_path / f'{value}.raw'
            write_complete_rawfile(raw, 'test\n.tran 1n 10n\n.end\n',
                                   values={'v(out)': value})
            bench.evaluate(corner='tt', rawfile=raw)
        monkeypatch.setattr(type(authored), '_simulate_run', staticmethod(
            lambda *a, **k: pytest.fail('plotting launched a simulation')))
        for bench, expected, name in zip(selected, [3, 11], ['one', 'two']):
            probe = bench.probe('out', analysis='tran')
            reference = probe.corner('tt') * 0 + 2
            expression = reference + probe
            ax = expression.plot()
            assert len(ax.lines) == 1
            np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
            assert name in ax.lines[0].get_label()
            np.testing.assert_allclose(ax.lines[0].get_ydata(), expression.corner('tt').y)
            plt.close('all')
    finally:
        plt.close('all')
        top.close()
        child.close()


@pytest.mark.parametrize('form', ['decorated', 'plain', 'expression', 'transform'])
@pytest.mark.parametrize('foreign_root', [False, True, 'same_path'])
def test_shared_metric_refuses_pinned_foreign_placement(tmp_path, form, foreign_root):
    from silk.exceptions import MetricBindingError
    from silk.metrics import average
    from tests.test_probe_node_paths import load

    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    other = root(tmp_path / 'other') if foreign_root else top
    top.add_module(mod, 'one', out='a', vss='0')
    second_path = 'one' if foreign_root == 'same_path' else 'two'
    other.add_module(mod, second_path, out='b', vss='0')
    one = top.get_test('step', instance='one')
    two = other.get_test('step', instance=second_path)
    calls = []

    def plain(sig):
        calls.append(True)
        return float(sig.y[-1])

    tb.add_metric('before', lambda c: calls.append('before') or 0.)
    pinned = one.probe('out')
    value = ((pinned + 0) if form == 'expression' else
             pinned.window(1., 2.) if form == 'transform' else pinned)
    fn = average if form == 'decorated' else plain
    # Controls measure the current placement rather than retained owner data.
    tb.add_metric('local', average, sig=tb.probe('out'))
    tb.add_metric('named', type='average', signal='out')
    first = load(tb, tmp_path, ['v(out)'], filename='owner.raw')
    second = load(tb, tmp_path, ['v(unused)', 'v(out)'], filename='sibling.raw')
    control = two.evaluate(rawfile=second)
    assert control.metrics['local'] == control.metrics['named'] == 2.25
    one.add_metric('pinned', fn, sig=value)
    one.add_spec('small', 'pinned', '<', 2.)
    retained = one.evaluate(rawfile=first)
    assert retained.metrics['pinned'] < 2.
    calls.clear()
    with pytest.raises(TypeError):
        two.add_metric('foreign', fn, sig=value)
    with pytest.raises(MetricBindingError) as exc:
        two.evaluate(rawfile=second)
    assert '$/one' in str(exc.value) and f'$/{second_path}' in str(exc.value)
    if foreign_root:
        assert f'{id(top):#x}' in str(exc.value)
        assert f'{id(other):#x}' in str(exc.value)
    assert 'authored/local probe' in str(exc.value)
    assert not calls  # No extractor may run before all ownership is checked.
    assert one.report is retained
    assert retained.metrics['pinned'] < 2.
    assert pinned.y[-1] == 1.5
    assert two.report is None
    top.close()
    if other is not top:
        other.close()
    mod.close()


def test_campaign_without_placement_refuses_selected_probe(tmp_path):
    from silk.circuit.test_placement import campaign_evaluation
    from silk.exceptions import MetricBindingError
    from silk.faults.evaluation import CampaignRun, _run_reader
    from silk.metrics import average
    from tests.test_probe_node_paths import load

    mod, tb = leaf(tmp_path)
    top = root(tmp_path)
    top.add_module(mod, 'one', out='a', vss='0')
    one = top.get_test('step', instance='one')
    path = load(tb, tmp_path, ['v(out)'])
    one.add_metric('pinned', average, sig=one.probe('out'))
    retained = one.evaluate(rawfile=path)
    run = CampaignRun('ngspice', path, 'tt', _run_reader(tb, 'ngspice'))
    with campaign_evaluation(tb, run):
        with pytest.raises(MetricBindingError, match='authored/local probe'):
            tb.evaluate(rawfile=path)
    assert one.report is retained
    assert retained.metrics['pinned'] == 1.25
    top.close()
    mod.close()


@pytest.mark.parametrize('form', ['callable', 'decorated', 'plain', 'declarative'])
def test_selected_metric_handle_reads_its_placement(recorded_selection, tmp_path, form):
    from silk.metrics import average
    from tests._mock_rawfile import write_complete_rawfile
    selected, tb, top, raw, _, _ = recorded_selection
    if form == 'callable':
        handle = selected.add_metric('reading', lambda c: float(c.load_signal('out').y[-1]))
    elif form == 'declarative':
        handle = selected.add_metric('reading', type='average', signal='out')
    else:
        def plain(sig):
            return float(sig.y[-1])
        handle = selected.add_metric('reading', average if form == 'decorated' else plain,
                                     sig=tb.probe('out'))
    with pytest.raises(RuntimeError, match='run.*first'):
        _ = handle.value
    first = handle.evaluate(rawfile=raw)
    assert first.owner_path == '$/a'
    sibling = top.get_test('step', instance='b')
    other = tmp_path / 'sibling.raw'
    write_complete_rawfile(other, 'sibling\n.tran 1n 10n\n.end\n', values={'v(out)': 9.})
    sibling.evaluate(rawfile=other)
    assert selected.metrics['reading'] == handle.value == 1.25
    assert (2 * abs(-handle) + handle / 2).value == 3.125
    assert handle.evaluate().owner_path == '$/a'
    assert sibling.metrics['reading'] == 9.


def test_selected_metric_handle_derived_recipe_and_fluent_run(recorded_selection):
    selected, tb, top, raw, _, _ = recorded_selection
    top.set_temperature(85)
    top.set_supply_voltage(1.8, source='rail')
    handle = selected.add_metric('reading', lambda c: float(c.load_signal('out').y[-1]))
    derived = handle.add_metric('twice', handle * 2)
    derived.add_spec('large', derived, '>', 3.)
    assert derived.set_simulator_options(reltol=1e-4) is selected
    report = derived.run()
    assert report.owner_path == '$/a'
    assert report.operating_conditions.temperature_c == 85
    assert report.operating_conditions.supply_voltage_v == 1.8
    assert report.metrics['twice'] == derived.value == 5.
    assert report.passed
    assert 'temp=85.0' in Path(selected.prepare()).read_text()
    assert 'Vrail out 0 DC 1.8' in Path(selected.prepare()).read_text()
    sibling = top.get_test('step', instance='b')
    assert not sibling.evaluate(rawfile=raw).passed
    assert sibling.metrics['twice'] == 2.5
    assert derived.value == 5.
    recipe = next(m for m in tb._metrics if m['name'] == 'twice')
    assert recipe == {'name': 'twice', 'expression': ('*', ('metric', 'reading'), ('number', 2))}
    assert derived.add_custom_line('* selected handle') is selected


@pytest.mark.parametrize('foreign', ['owner', 'root', 'test', 'authored'])
def test_selected_metric_handle_rejects_foreign_ownership(tmp_path, monkeypatch, foreign):
    child, tb = leaf(tmp_path / 'child', wired=False)
    top, other = root(tmp_path / 'top'), root(tmp_path / 'other')
    try:
        top.add_module(child, 'one', out='a', vss='0')
        top.add_module(child, 'two', out='b', vss='0')
        other.add_module(child, 'one', out='a', vss='0')
        one = top.get_test('step', instance='one')
        h = one.add_metric('first', abs)
        child.create_test('other')
        bench = {'owner': top.get_test('step', instance='two'),
                 'root': other.get_test('step', instance='one'),
                 'test': top.get_test('other', instance='one'),
                 'authored': tb}[foreign]
        second = bench.add_metric('second', abs)
        if foreign == 'test':
            # Equal definitions are still distinct authored test owners.
            monkeypatch.setattr(type(tb), '__eq__', lambda self, other: True, raising=False)
        with pytest.raises(ValueError, match='same testbench'):
            _ = h + second
        with pytest.raises(ValueError, match='testbench'):
            bench.add_metric('bad', h + 1)
        with pytest.raises(ValueError, match='testbench'):
            bench.add_spec('bad', h, '>', 0)
        if foreign == 'authored':
            # Authored recipes remain reusable in selected shared definitions.
            one.add_metric('shared', second + 1)
            one.add_spec('shared spec', second, '>', 0)
    finally:
        other.close()
        top.close()
        child.close()


def test_selected_metric_handle_repeated_nested_selection(tmp_path):
    child, tb = leaf(tmp_path / 'child', wired=False)
    carrier = root(tmp_path / 'carrier')
    carrier.define_module_name('carrier')
    carrier.define_module_ports({})
    top = root(tmp_path / 'top')
    try:
        carrier.add_module(child, 'cell', out='out', vss='0')
        top.add_module(carrier, 'outer', nets={})
        selected = top.get_test('step', instance='outer.cell')
        first = selected.add_metric('first', abs)
        repeated = top.get_test('step', instance='outer.cell')
        second = repeated.add_metric('second', abs)
        combined = repeated.add_metric('combined', first + second)
        repeated.add_spec('combined spec', combined, '>', 0)
        assert combined.add_custom_line('* nested') is selected
        assert combined._bench.owner_path == '$/outer/cell'
    finally:
        top.close()
        carrier.close()
        child.close()


@pytest.mark.parametrize('handle_first', [False, True])
def test_selected_metric_handle_copy_and_pickle_contract(tmp_path, handle_first):
    import copy
    import gc
    import pickle
    import weakref
    import numpy as np
    from silk.metrics import average
    from silk.signals import Signal
    from silk.circuit.test_placement import execution_scope
    child = root(tmp_path / 'child')
    child.define_module_name('child')
    child.define_module_ports({})
    tb = child.create_test('read')
    top = root(tmp_path / 'top')
    top.add_module(child, 'one', nets={})
    selected = top.get_test('read', instance='one')
    handle = selected.add_metric('reading', average,
                                 sig=Signal('s', 'V', 'time', 'V', np.array([0., 1.]),
                                            np.array([2., 6.])))
    pair = (handle, top) if handle_first else (top, handle)
    copied = copy.deepcopy(pair)
    copied_handle, copied_root = copied if handle_first else reversed(copied)
    try:
        assert copied_handle._bench is copied_root.get_test('read', instance='one')
        with pytest.raises(RuntimeError, match='run.*first'):
            _ = copied_handle.value
        with execution_scope(copied_handle._bench._entry()):
            copied_handle._bench._authored._metric_values = {'reading': 7.}
        assert copied_handle.value == 7.
        for value in (selected, (top, selected), handle, pair):
            with pytest.raises(TypeError, match='weakref') as error:
                pickle.dumps(value)
            assert 'SelectedTestbench' in str(error.value)
        del value
        lone = copy.deepcopy(handle)
        gc.collect()
        with pytest.raises(ConfigurationError, match='root.*no longer exists'):
            _ = lone.value
        root_ref = weakref.ref(top)
        top.close()
        del pair, top, selected
        gc.collect()
        assert root_ref() is None
        with pytest.raises(ConfigurationError, match='root.*no longer exists'):
            _ = handle.value
    finally:
        copied_root.close()
        child.close()


def test_selected_metric_handle_removed_selection_refuses_use(recorded_selection):
    selected, tb, top, _, _, _ = recorded_selection
    handle = selected.add_metric('reading', abs)
    del top._element_mgr.elem_dict['a']
    with pytest.raises(ConfigurationError, match='removed or replaced'):
        _ = handle.value
    with pytest.raises(ConfigurationError, match='removed or replaced'):
        handle.evaluate()


def test_selected_metric_handle_campaign_registration_refuses(recorded_selection):
    from silk.circuit.test_placement import campaign_evaluation
    from silk.faults.evaluation import CampaignRun, _run_reader
    selected, tb, _, raw, _, _ = recorded_selection
    handle = selected.add_metric('reading', abs)
    run = CampaignRun('ngspice', str(raw), 'tt', _run_reader(tb, 'ngspice'))
    with campaign_evaluation(tb, run):
        with pytest.raises(ValueError, match='testbench'):
            tb.add_metric('foreign', handle)
        with pytest.raises(ValueError, match='testbench'):
            tb.add_spec('foreign', handle, '>', 0)
