"""One hierarchy invocation shares proven simulations, never evaluations."""
import pytest

from silk import Module
from tests._env import needs_ngspice, needs_pdk
from tests.circuit.test_selected_testbench import leaf, root


@needs_ngspice
@pytest.mark.parametrize('count', [2, 15])
@pytest.mark.parametrize('workers', [1, 2])
def test_identical_placements_launch_once(tmp_path, monkeypatch, count, workers):
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    for index in range(count):
        top.add_module(child, f'u{index}', out=f'n{index}', vss='0')
    from silk.simulation import _worker_pickle
    copies = []
    original_copy = _worker_pickle.copy_simulation_netlist
    def copy_netlist(module):
        copies.append(1)
        return original_copy(module)
    monkeypatch.setattr(_worker_pickle, 'copy_simulation_netlist', copy_netlist)
    launches = []
    if workers == 1:
        original = tb.netlist.run_simulator
        def run(*args, **kwargs):
            launches.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(tb.netlist, 'run_simulator', run)
    else:
        from silk.simulation import executor
        original = executor.run_tasks
        def tasks(items, **kwargs):
            items = list(items)
            launches.extend(items)
            return original(items, **kwargs)
        monkeypatch.setattr(executor, 'run_tasks', tasks)
    try:
        reports = top.run_all_tests(n_workers=workers)
        assert len(launches) == 1
        if workers == 2:
            assert len(copies) == 1
        assert len(reports) == count
        assert len({id(r) for r in reports.values()}) == count
        assert [r.metrics['level'] for r in reports.values()] == pytest.approx([.5] * count)
        assert len({r.rawfile_path for r in reports.values()}) == 1
        tb.netlist.sig_sources_dict['rail'].dc = 2
        assert top.run_all_tests(n_workers=workers)[next(iter(reports))].metrics['level'] == pytest.approx(2)
        assert len(launches) == 2
    finally:
        top.close()
        child.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_nonadjacent_supply_groups_keep_run_evidence(tmp_path, monkeypatch, workers):
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    parents = []
    for index, voltage in enumerate((1, 2, 1)):
        carrier = Module(None, str(tmp_path), f'carrier{index}.cir', None)
        carrier.set_supply_voltage(voltage)
        carrier.define_module_name('carrier')
        carrier.define_module_ports({})
        carrier.add_module(child, 'cell', out='out', vss='0')
        top.add_module(carrier, f'u{index}', nets={})
        parents.append(carrier)
    launches = []
    if workers == 1:
        original = tb.netlist.run_simulator
        def run(*args, **kwargs):
            launches.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(tb.netlist, 'run_simulator', run)
    else:
        from silk.simulation import executor
        original = executor.run_tasks
        def tasks(items, **kwargs):
            items = list(items)
            launches.extend(items)
            return original(items, **kwargs)
        monkeypatch.setattr(executor, 'run_tasks', tasks)
    try:
        reports = list(top.run_all_tests(n_workers=workers).values())
        assert len(launches) == 2
        assert [r.metrics['level'] for r in reports] == pytest.approx([1, 2, 1])
        assert [e.test.probe('out').y[-1] for e in top.list_test_entries()] == pytest.approx([1, 2, 1])
        assert reports[0].rawfile_path == reports[2].rawfile_path
        assert reports[0].sim_results._launch_record == reports[2].sim_results._launch_record
        assert reports[0].sim_results._prepared_deck_record == reports[2].sim_results._prepared_deck_record
    finally:
        top.close()
        child.close()
        for carrier in parents:
            carrier.close()


@needs_ngspice
def test_shared_group_keeps_main_output_location(tmp_path):
    from pathlib import Path
    child, tb = leaf(tmp_path)
    child.set_output_options(mode='overwrite')
    expected = Path(tb.prepare()).parent
    top = root(tmp_path)
    top.set_output_options(mode='overwrite')
    try:
        for name in ('x1', 'x2'):
            top.add_module(child, name, out=name, vss='0')
        reports = list(top.run_all_tests().values())
        assert {Path(report.rawfile_path).parent for report in reports} == {expected}
    finally:
        top.close()
        child.close()


@needs_ngspice
@pytest.mark.parametrize('difference', [None, 'source', 'resistance', 'analysis', 'options', 'save'])
def test_factory_benches_group_only_equal_emitted_inputs(tmp_path, monkeypatch, difference):
    from silk.simulation import Transient
    first, a = leaf(tmp_path / 'a')
    second, b = leaf(tmp_path / 'b')
    top = root(tmp_path)
    top.add_module(first, 'a', out='a', vss='0')
    top.add_module(second, 'b', out='b', vss='0')
    if difference == 'source':
        b.netlist.sig_sources_dict['rail'].dc = 2
    elif difference == 'resistance':
        second.elem_dict['r']['device'].value = 2000
    elif difference == 'analysis':
        b.add_sim_type(Transient(step=2e-9, stop=20e-9))
    elif difference == 'options':
        b.set_simulator_options(reltol=.003)
    elif difference == 'save':
        b.netlist.set_save_list(['out'])
    launches = []
    for tb in (a, b):
        original = tb.netlist.run_simulator
        def run(*args, _original=original, **kwargs):
            launches.append(1)
            return _original(*args, **kwargs)
        monkeypatch.setattr(tb.netlist, 'run_simulator', run)
    try:
        reports = top.run_all_tests()
        assert all(not isinstance(r, dict) for r in reports.values()), reports
        assert len(launches) == (1 if difference is None else 2)
    finally:
        top.close()
        first.close()
        second.close()


def test_behavioral_closure_is_not_assumed_pure(tmp_path):
    # External global side effects still require independent launches.
    global calls
    from tests.circuit.test_module_references import placement_fixture
    top, tb, _ = placement_fixture(tmp_path)
    calls = []
    def model(ports, t, params):
        calls.append(1)
        ports.p.v = t * 0 + len(calls)
    tb.module.set_behavioral_model(model, execution='vector')
    try:
        reports = list(top.run_all_tests().values())
        assert [r.metrics['level'] for r in reports] == [1, 2]
    finally:
        top.close()
        tb.module.close()


@needs_ngspice
def test_new_mode_group_hit_preserves_representative_artifact(tmp_path):
    from pathlib import Path
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    top.set_output_options(mode='new')
    for name in ('a', 'b'):
        top.add_module(child, name, out=name, vss='0')
    try:
        reports = list(top.run_all_tests().values())
        assert reports[0].rawfile_path == reports[1].rawfile_path
        assert Path(reports[0].rawfile_path).is_file()
        assert len(list(tmp_path.rglob('*.raw'))) == 1
    finally:
        top.close()
        child.close()


@needs_ngspice
@pytest.mark.parametrize('own_temperature', [None, 40.0])
def test_temperature_inheritance_separates_runs_and_own_setting_wins(
        tmp_path, monkeypatch, own_temperature):
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    carriers = []
    for name, temperature in [('cool', 27), ('hot', 85)]:
        carrier = root(tmp_path)
        carrier.set_temperature(temperature)
        carrier.define_module_name('carrier')
        carrier.define_module_ports({})
        carrier.add_module(child, 'cell', out='out', vss='0')
        top.add_module(carrier, name, nets={})
        carriers.append(carrier)
    if own_temperature is not None:
        tb.set_temperature(own_temperature)
    launches = []
    original = tb.netlist.run_simulator
    def run(*args, **kwargs):
        launches.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(tb.netlist, 'run_simulator', run)
    try:
        reports = list(top.run_all_tests().values())
        assert all(not isinstance(report, dict) for report in reports), reports
        assert len(launches) == (2 if own_temperature is None else 1)
        expected = [27, 85] if own_temperature is None else [40, 40]
        assert [r.operating_conditions.temperature_c for r in reports] == expected
    finally:
        top.close()
        child.close()
        for carrier in carriers:
            carrier.close()


@needs_ngspice
def test_stochastic_source_never_shares_a_launch(tmp_path, monkeypatch):
    from silk.sources.waveforms import TransientNoise
    child, tb = leaf(tmp_path, waveform=TransientNoise(na=.01, nt=1e-9))
    top = root(tmp_path)
    for name in ('a', 'b'):
        top.add_module(child, name, out=name, vss='0')
    launches = []
    original = tb.netlist.run_simulator
    def run(*args, **kwargs):
        launches.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(tb.netlist, 'run_simulator', run)
    try:
        reports = top.run_all_tests()
        assert all(not isinstance(report, dict) for report in reports.values()), reports
        assert len(launches) == 2
    finally:
        top.close()
        child.close()


@needs_ngspice
@pytest.mark.parametrize('directive', ['include', 'lib'])
@pytest.mark.parametrize('absolute', [False, True])
def test_include_dependencies_are_not_assumed_equivalent(
        tmp_path, monkeypatch, directive, absolute):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    modules = [leaf(tmp_path / name) for name in ('a', 'b')]
    # Opaque custom text conservatively retains the PDK include. An empty
    # local library satisfies that declaration without requiring installed models.
    from types import SimpleNamespace
    library = tmp_path / 'empty.lib'
    library.write_text('.lib tt\n.param declared=1\n.endl tt\n')
    pdk = SimpleNamespace(library_location=str(library), corners={}, default_corner='tt')
    top = root(tmp_path)
    launches = []
    try:
        for name, (child, tb) in zip(('a', 'b'), modules):
            top.add_module(child, name, out=name, vss='0')
            tb.netlist.pdk_details = pdk
            tb.netlist.netlister.pdk_details = pdk
            directory = Path(tb.prepare()).parent if absolute else tmp_path
            dependency = directory / ('parts.' + directive)
            body = '.param extra=1\n'
            dependency.write_text('.lib local\n' + body + '.endl local\n'
                                  if directive == 'lib' else body)
            path = str(dependency) if absolute else dependency.name
            tb.add_custom_line(f'.{directive} "{path}"' + (' local' if directive == 'lib' else ''))
            original = tb.netlist.run_simulator
            def run(*args, _original=original, **kwargs):
                launches.append(1)
                return _original(*args, **kwargs)
            monkeypatch.setattr(tb.netlist, 'run_simulator', run)
        reports = top.run_all_tests()
        assert all(not isinstance(report, dict) for report in reports.values()), reports
        assert len(launches) == 2
    finally:
        top.close()
        for child, _ in modules:
            child.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_shared_run_keeps_consumer_metric_failures_separate(tmp_path, workers):
    first, a = leaf(tmp_path / 'a')
    second, b = leaf(tmp_path / 'b')
    def failing_metric(netlist):
        raise ValueError('consumer-only failure')
    a.add_metric('extra', failing_metric)
    b.add_metric('extra', lambda netlist: 7)
    top = root(tmp_path)
    top.add_module(first, 'a', out='a', vss='0')
    top.add_module(second, 'b', out='b', vss='0')
    try:
        reports = list(top.run_all_tests(n_workers=workers).values())
        assert reports[0].rawfile_path == reports[1].rawfile_path
        assert 'consumer-only failure' in reports[0].metrics['extra']
        assert reports[1].metrics['extra'] == 7
    finally:
        top.close()
        first.close()
        second.close()


def test_shared_failed_launch_is_attributed_to_every_consumer(tmp_path, monkeypatch):
    from silk.exceptions import SimulationError
    child, tb = leaf(tmp_path)
    top = root(tmp_path)
    for name in ('a', 'b'):
        top.add_module(child, name, out=name, vss='0')
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise SimulationError('injected launch failure')
    monkeypatch.setattr(tb.netlist, 'simulate', fail)
    try:
        results = top.run_all_tests()
        assert len(calls) == 1
        assert len(results) == 2
        assert all('injected launch failure' in value['error'] for value in results.values())
        assert all(entry.test.results_file is None for entry in top.list_test_entries())
    finally:
        top.close()
        child.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_declared_corner_groups_share_launch_and_keep_selected_reports(tmp_path, monkeypatch, workers):
    from silk.simulation import executor
    child, tb = leaf(tmp_path)
    tb.set_corners(['tt'])
    top = root(tmp_path)
    for name in ('x1', 'x2'):
        top.add_module(child, name, out=name, vss='0')
    launches = []
    original = Module.run_simulator
    if workers == 1:
        def launch(module, *args, **kwargs):
            launches.append(1)
            return original(module, *args, **kwargs)
        monkeypatch.setattr(Module, 'run_simulator', launch)
    else:
        execute = executor.run_tasks
        def launch(tasks, **kwargs):
            tasks = list(tasks)
            launches.extend(tasks)
            return execute(tasks, **kwargs)
        monkeypatch.setattr(executor, 'run_tasks', launch)
    try:
        reports = top.run_all_tests(n_workers=workers)
        assert len(launches) == 1
        for entry in top.list_test_entries():
            assert entry.test.report is reports[entry.report_key]
            assert entry.test.probe('out').corner('tt').y[-1] == pytest.approx(.5)
    finally:
        top.close()
        child.close()


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('workers', [1, 2])
def test_monte_carlo_placements_launch_every_sample(tmp_path, monkeypatch, workers):
    from silk import load_pdk, sources
    from silk.simulation import Transient, executor
    pdk = load_pdk('sky130')
    child = Module('sky130', str(tmp_path), 'diode.cir', pdk, module_name='diode')
    child.define_module_ports({'out': 'inout', 'vss': 'ground'})
    child.add(pdk.nmos1p8('m1', d='out', g='out', s='vss', b='vss', w=5, l=2))
    tb = child.create_test('step')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.CurrentSource('bias', positive='0', negative='out', dc=20e-6))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    tb.add_metric('level', lambda c: float(c.load_signal('out').y[-1]))
    tb.set_monte_carlo(n=2)
    top = root(tmp_path)
    for name in ('x1', 'x2'):
        top.add_module(child, name, out=name, vss='0')
    launches = []
    if workers == 1:
        original = Module.run_simulator
        def launch(module, *args, **kwargs):
            launches.append(1)
            return original(module, *args, **kwargs)
        monkeypatch.setattr(Module, 'run_simulator', launch)
    else:
        original = executor.run_tasks
        def launch(tasks, **kwargs):
            tasks = list(tasks)
            launches.extend(tasks)
            return original(tasks, **kwargs)
        monkeypatch.setattr(executor, 'run_tasks', launch)
    try:
        reports = top.run_all_tests(n_workers=workers)
        assert len(launches) == 4
        paths = set()
        for entry in top.list_test_entries():
            campaign = reports[entry.report_key]
            for key, report in campaign.reports.items():
                assert entry.test.probe('out').sample(key).y[-1] == pytest.approx(report.metrics['level'])
                paths.add(report.rawfile_path)
        assert len(paths) == 4
    finally:
        top.close()
        child.close()


@pytest.mark.parametrize('workers', [1, 2])
def test_submodule_banners_precede_each_owners_tests(tmp_path, monkeypatch, capsys, workers):
    from types import SimpleNamespace
    top = root(tmp_path)
    child, first = leaf(tmp_path)
    second = child.create_test('other')
    for tb in (first, second):
        monkeypatch.setattr(tb, 'run', lambda **kwargs: SimpleNamespace())
        monkeypatch.setattr(tb, '_prepare_run', lambda **kwargs: SimpleNamespace())
    top.add_module(child, 'left', out='left', vss='0')
    top.add_module(child, 'right', out='right', vss='0')
    try:
        assert top.run_all_tests(recursive=False, n_workers=workers) == {}
        assert 'Testing submodule:' not in capsys.readouterr().out
        top.run_all_tests(n_workers=workers)
        output = capsys.readouterr().out
        for name in ('left', 'right'):
            banner = f'--- Testing submodule: {name} ---'
            assert output.count(banner) == 1
            assert output.index(banner) < output.index(f'Running test: $/{name}:leaf.step')
    finally:
        top.close()
        child.close()


@needs_ngspice
def test_distinct_root_tests_keep_wrappers_and_own_run_files(tmp_path, monkeypatch):
    from pathlib import Path
    from silk import sources
    from silk.elements import elements
    from silk.simulation import Transient

    dut = Module(None, str(tmp_path), 'rc.cir', None, module_name='rc')
    dut.define_module_ports({'a': 'in', 'y': 'out', 'g': 'ground'})
    dut.add(elements.resistor('r1', p='a', n='y', value=1e3))
    dut.add(elements.capacitor('c1', p='y', n='g', value=1e-12))
    dut.set_output_options(mode='overwrite')
    calls = {}
    folders = {}
    for name, metric in (('t_max', lambda c: float(c.load_signal('y').y.max())),
                         ('t_final', lambda c: float(c.load_signal('y').y[-1]))):
        tb = dut.create_test(name)
        tb.add_module_under_test({'a': 'a', 'y': 'y', 'g': '0'})
        tb.add_source(sources.VoltageSource('vin', positive='a', negative='0', dc=0,
                      pulse=sources.Pulse(v1=0, v2=1, td=1e-9, tr=1e-10,
                                          tf=1e-10, pw=5e-9, per=10e-9)))
        tb.add_sim_type(Transient(1e-11, 20e-9))
        tb.add_metric('v', metric)
        tb.add_spec(name + '_nonnegative', 'v', '>=', 0)
        folders[name] = Path(tb.prepare()).parent
        original_run, original_simulator = tb.run, tb.netlist.run_simulator
        def run(*args, _name=name, _original=original_run, **kwargs):
            calls.setdefault(_name, []).append('run')
            return _original(*args, **kwargs)
        def simulate(*args, _name=name, _original=original_simulator, **kwargs):
            calls.setdefault(_name, []).append('run_simulator')
            return _original(*args, **kwargs)
        monkeypatch.setattr(tb, 'run', run)
        monkeypatch.setattr(tb.netlist, 'run_simulator', simulate)
    try:
        reports = dut.run_all_tests()
        assert calls == {name: ['run', 'run_simulator'] for name in folders}
        assert len({report.rawfile_path for report in reports.values()}) == 2
        for name, folder in folders.items():
            report = reports[f'rc.{name}']
            assert report.passed
            assert Path(report.rawfile_path).parent == folder
            assert dut.get_test(name).netlist._run_files
    finally:
        dut.close()


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_different_named_tests_share_only_their_own_placements(tmp_path, monkeypatch, workers):
    from silk import sources
    from silk.simulation import Transient
    child, first = leaf(tmp_path)
    other = child.create_test('other')
    other.add_module_under_test({'out': 'out', 'vss': '0'})
    other.add_source(sources.VoltageSource('rail', positive='out', negative='0', dc=.5))
    other.add_sim_type(Transient(step=1e-9, stop=10e-9))
    other.add_metric('level', lambda c: float(c.load_signal('out').y[-1]))
    top = root(tmp_path)
    top.add_module(child, 'one', out='a', vss='0')
    top.add_module(child, 'two', out='b', vss='0')
    launches = []
    if workers == 1:
        original = Module.run_simulator
        def launch(module, *args, **kwargs):
            launches.append(module)
            return original(module, *args, **kwargs)
        monkeypatch.setattr(Module, 'run_simulator', launch)
    else:
        from silk.simulation import executor
        original = executor.run_tasks
        def launch(tasks, **kwargs):
            tasks = list(tasks)
            launches.extend(tasks)
            return original(tasks, **kwargs)
        monkeypatch.setattr(executor, 'run_tasks', launch)
    try:
        reports = top.run_all_tests(n_workers=workers)
        assert len(launches) == 2
        assert len(reports) == 4
        assert [report.metrics['level'] for report in reports.values()] == pytest.approx([.5] * 4)
        step_paths = {report.rawfile_path for name, report in reports.items() if name.endswith('.step')}
        other_paths = {report.rawfile_path for name, report in reports.items() if name.endswith('.other')}
        assert len(step_paths) == len(other_paths) == 1
        assert step_paths.isdisjoint(other_paths)
    finally:
        top.close()
        child.close()
