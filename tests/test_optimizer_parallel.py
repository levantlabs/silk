"""Optimizer.sweep(n_workers=) — parallel design points (TASK-301.5).

Prepare in the parent, simulate through the shared executor
(``silk.simulation.executor``), evaluate in the parent, in point order —
same shape TASK-301.3 uses for ``run_corners()``. PDK-free RC decks below
need no PDK install, only ngspice, and DO request genuine multi-worker
pooling (``n_workers>1``): pickling the ideal ``elements.resistor``/
``capacitor`` classes used here is safe now that TASK-312.3 fixed
``silk.elements.factory``'s "last registration wins" registry for ideal
elements (first registration -- the package singleton built at import
time -- wins instead, so a later PDK load can no longer invalidate them;
this was TASK-301.2's own documented caveat, and TASK-301.3's own
``run_corners(n_workers=4)`` PDK-free test already relies on the same fix).
The one `@pytest.mark.slow` class at the bottom repeats every claim on a
real sky130 NMOS DUT instead, and also reports a measured speedup.
"""
import glob
import os

import pytest

from silk import Port, load_pdk, sources
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.elements import elements
from silk.exceptions import SimulationError
from silk.optimize import Optimizer, VariableSpace
from silk.optimize import optimizer as optimizer_module
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk


def _build_variable_lowpass_tb(tmp_dir, name, r_value=1000.0):
    """A PDK-free RC low-pass whose resistor is a swept `Variable`.

    Driven by a pulse (not a bare DC source): a constant source settles to
    its DC operating point instantly, before any transient step ever runs,
    so an early-time sample would be R-insensitive and useless as an
    optimizer objective. The pulse's rising edge is what actually exercises
    RC charging, and an early sample distinguishes the R values a sweep
    would compare (TASK-301.5 evidence).
    """
    dut = Module(None, tmp_dir, f'{name}.cir', pdk_details=None,
                 module_name='rc_lowpass_' + name)
    dut.define_module_ports({'in': 'in', 'out': 'out', 'vss': 'ground'})
    r = Variable('r', value=r_value, bounds=(200.0, 20000.0))
    dut.add(elements.resistor('r1', p='in', n='out', value=r))
    dut.add(elements.capacitor('c1', p='out', n='vss', value=1e-9))
    tb = dut.create_test(name)
    tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vss': '0'})
    pulse = sources.Pulse(v1=0, v2=1, tr=1e-10, tf=1e-10, pw=1e-6, per=2e-6)
    tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0',
                                        dc=0.0, pulse=pulse))
    tb.add_sim_type(Transient(step=1e-9, stop=100e-9))
    tb.add_metric('vout_early', lambda ckt: float(ckt.load_signal('vout', 'V').y[20]))
    tb.add_metric('r_seen', lambda ckt, v=r: v.value)
    tb.add_spec('settles', 'vout_early', '>', -1e9)
    return dut, tb, r


def _build_optimizer(tmp_dir, name):
    dut, tb, r = _build_variable_lowpass_tb(tmp_dir, name)
    opt = Optimizer(tb, VariableSpace([r]))
    opt.add_objective('vout_early', minimize=True)
    opt.add_constraint('vout_early', min=-1e9)
    return opt, dut, tb, r


@needs_ngspice
class TestSerialAndParallelIdentical:
    """AC #1: same points give identical results and best, serial vs
    GENUINE multi-worker pooling, on the PDK-free deck (safe since
    TASK-312.3 -- see the module docstring)."""

    def test_metrics_and_best_match(self, tmp_path):
        opt_serial, _, tb_serial, _ = _build_optimizer(str(tmp_path / 'serial'), 'serial')
        opt_par, _, tb_par, _ = _build_optimizer(str(tmp_path / 'par'), 'par')

        # 'grid' sampling is deterministic (np.linspace over bounds), so two
        # independently constructed VariableSpace objects with the same bounds
        # yield the identical point sequence.
        records_serial = opt_serial.sweep(n=4, method='grid', corner='tt', n_workers=None)
        records_par = opt_par.sweep(n=4, method='grid', corner='tt', n_workers=2)

        assert len(records_serial) == len(records_par) == 4
        for rs, rp in zip(records_serial, records_par):
            assert rs['params'] == rp['params']
            assert rs['status'] == rp['status'] == 'success'
            assert rs['metrics'] == rp['metrics']
            assert rs['feasible'] == rp['feasible']

        best_serial, best_par = opt_serial.best, opt_par.best
        assert best_serial is not None and best_par is not None
        assert best_serial['params'] == best_par['params']
        assert best_serial['metrics'] == best_par['metrics']

    def test_more_workers_than_points_still_matches_serial(self, tmp_path):
        opt_serial, _, _, _ = _build_optimizer(str(tmp_path / 'serial'), 'serial')
        opt_par, _, _, _ = _build_optimizer(str(tmp_path / 'par'), 'par')

        records_serial = opt_serial.sweep(n=3, method='grid', n_workers=1)
        records_par = opt_par.sweep(n=3, method='grid', n_workers=8)

        assert [r['metrics'] for r in records_serial] == [r['metrics'] for r in records_par]


@needs_ngspice
class TestPerPointOutputFolders:
    """AC #2: every point gets its own folder, serial and parallel alike."""

    @pytest.mark.parametrize('n_workers', [None, 2])
    def test_each_point_has_its_own_folder_no_shared_files(self, tmp_path, n_workers):
        opt, dut, tb, r = _build_optimizer(str(tmp_path), 'folders')
        opt.sweep(n=4, method='grid', n_workers=n_workers)

        run_dirs = sorted(glob.glob(os.path.join(dut.dir_folder, '*~run~point_*')))
        assert len(run_dirs) == 4, run_dirs
        cir_files = set()
        raw_files = set()
        for d in run_dirs:
            cirs = glob.glob(os.path.join(d, 'run-*', '*.cir'))
            raws = glob.glob(os.path.join(d, 'run-*', '*.raw'))
            assert len(cirs) == 1, (d, cirs)
            assert len(raws) == 1, (d, raws)
            cir_files.add(os.path.realpath(cirs[0]))
            raw_files.add(os.path.realpath(raws[0]))
        # No two points share a .cir or .raw path.
        assert len(cir_files) == 4
        assert len(raw_files) == 4


@needs_ngspice
class TestLiveNetlistIdentity:
    """AC #2: the live module ends in the same state as after a serial run."""

    def test_identity_preserved_and_reflects_last_point(self, tmp_path):
        opt, dut, tb, r = _build_optimizer(str(tmp_path), 'identity')
        live_netlist = tb.netlist
        # Genuine pooling: result.netlist is a worker's unpickled copy,
        # not tb.netlist itself, so this exercises ModuleTestbench.
        # _evaluate_run's real state-adoption path, not just its no-op
        # (result.netlist is self.netlist) branch.
        records = opt.sweep(n=3, method='grid', n_workers=2)

        # Never replaced by a worker's unpickled copy.
        assert tb.netlist is live_netlist

        # Reflects the LAST point run, exactly as a lone test.run() on that
        # point would leave it: metrics extracted directly from the live
        # object's own state agree with the last record's metrics.
        last = records[-1]
        assert tb.netlist.sim_results_loaded is True
        assert float(tb.netlist.load_signal('vout', 'V').y[20]) == last['metrics']['vout_early']

        # The live object stays usable: evaluate() on it (no new run) must
        # not raise and must reproduce the same report evaluate() would
        # give right after an ordinary serial run() of that last point.
        report = tb.evaluate('tt')
        assert report.metrics['vout_early'] == last['metrics']['vout_early']


@needs_ngspice
class TestWorkerFailureIsolation:
    """AC #3: one point's simulate-step failure is that point's error only."""

    def test_one_bad_point_does_not_abort_the_rest(self, tmp_path, monkeypatch):
        opt, dut, tb, r = _build_optimizer(str(tmp_path), 'failure')
        original = optimizer_module.ModuleTestbench._simulate_run
        calls = {'n': 0}

        def flaky(spec):
            calls['n'] += 1
            if calls['n'] == 2:
                raise SimulationError('injected failure for point 2')
            return original(spec)

        # Monkeypatched in the parent process: only meaningful for the
        # serial in-process path (n_workers=None), where _run_design_point
        # runs in THIS process. A pool worker re-imports the module fresh
        # and would not see this patch — the failure-capture mechanism
        # itself (silk.simulation.executor.run_tasks + _run_design_point)
        # is the same code regardless of n_workers, so this proves it once,
        # in-process, rather than needing a cross-process injection.
        monkeypatch.setattr(optimizer_module.ModuleTestbench, '_simulate_run',
                            staticmethod(flaky))

        records = opt.sweep(n=4, method='grid', n_workers=None)

        assert len(records) == 4
        statuses = [r['status'] for r in records]
        assert statuses.count('simulation_failure') == 1
        assert statuses.count('success') == 3
        failed = [r for r in records if r['status'] == 'simulation_failure']
        assert failed[0]['error_type'] == 'SimulationError'
        assert 'injected failure' in failed[0]['error']


@needs_ngspice
class TestMaxFailuresChunking:
    """max_failures on the new (ModuleTestbench) path: chunk-granularity
    early stop -- exact per-point for n_workers=None/1 (chunk size 1),
    coarser for real parallelism (chunk size = n_workers), documented on
    sweep()."""

    def test_serial_stops_at_exact_count(self, tmp_path, monkeypatch):
        opt, dut, tb, r = _build_optimizer(str(tmp_path), 'maxfail')

        def always_fails(spec):
            raise SimulationError('always fails')

        monkeypatch.setattr(optimizer_module.ModuleTestbench, '_simulate_run',
                            staticmethod(always_fails))

        records = opt.sweep(n=10, method='grid', n_workers=None, max_failures=3)
        assert len(records) == 3
        assert all(rec['status'] == 'simulation_failure' for rec in records)


class TestCustomTestRejectsRealParallel:
    """Custom (non-ModuleTestbench) tests cannot be split across workers."""

    def test_n_workers_greater_than_one_raises(self):
        class MockTest:
            def run(self, corner='tt'):
                return {'metrics': {'g': 1.0}}

        w = Variable('w', value=1.0, bounds=(0.5, 2.0))
        opt = Optimizer(MockTest(), VariableSpace([w]))
        with pytest.raises(TypeError):
            opt.sweep(n=2, method='grid', n_workers=4)

    def test_n_workers_none_is_unaffected(self):
        class MockTest:
            def run(self, corner='tt'):
                return {'metrics': {'g': 1.0}}

        w = Variable('w', value=1.0, bounds=(0.5, 2.0))
        opt = Optimizer(MockTest(), VariableSpace([w]))
        records = opt.sweep(n=3, method='grid', n_workers=None)
        assert len(records) == 3
        assert all(r['status'] == 'success' for r in records)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
class TestRealPdkParallelSweep:
    """Same identity claim again, this time on a real sky130 NMOS DUT under
    genuine multi-worker pooling, and with a real measured speedup -- the
    fast tests above already prove correctness AND real pooling on a
    PDK-free deck (safe since TASK-312.3), so this class exists for the
    realistic device + timing evidence, not to cover a gap in them."""

    def test_serial_and_parallel_match_on_real_pdk(self, tmp_path):
        pdk = load_pdk('sky130')

        def build(subdir):
            width = Variable('w_n', value=2.0, bounds=(1.0, 8.0))
            out = str(tmp_path / subdir)
            os.makedirs(out, exist_ok=True)
            dut = Module('PDK_skywater130', out, 'nmos.cir',
                        pdk_details=pdk, module_name='nmos')
            dut.define_module_ports({
                'D': Port.analog_output(), 'G': Port.analog_input(),
                'S': Port.ground(),
            })
            with dut:
                pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=width, l=0.5)
            tb = dut.create_test('current')
            tb.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
            emitted_dut = tb.netlist.module_dict['nmos']['device']
            emitted_dut.update_element('m1', ['w'], [width])
            tb.add_source(sources.VoltageSource(
                'drain', positive='drain', negative='0', dc=0.8))
            tb.add_source(sources.VoltageSource(
                'gate', positive='gate', negative='0', dc=1.0))
            tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
            tb.add_metric('drain_current_A',
                          lambda ckt: -float(ckt.load_signal('vdrain', 'I').y[-1]))
            tb.add_spec('conducts', 'drain_current_A', '>', 0.0)
            opt = Optimizer(tb, VariableSpace([width]))
            opt.add_objective('drain_current_A', maximize=True)
            opt.add_constraint('drain_current_A', min=0.0)
            return opt, dut, tb

        opt_serial, dut_serial, tb_serial = build('serial')
        opt_par, dut_par, tb_par = build('par')
        live_netlist = tb_par.netlist

        import time
        t0 = time.time()
        records_serial = opt_serial.sweep(n=4, method='grid', corner='tt', n_workers=None)
        t_serial = time.time() - t0

        t0 = time.time()
        records_par = opt_par.sweep(n=4, method='grid', corner='tt', n_workers=4)
        t_par = time.time() - t0

        assert [r['metrics'] for r in records_serial] == [r['metrics'] for r in records_par]
        assert opt_serial.best['params'] == opt_par.best['params']
        print(f'\nTASK-301.5 real-PDK sweep(n=4): serial {t_serial:.2f}s, '
              f'n_workers=4 {t_par:.2f}s (speedup {t_serial / t_par:.2f}x)')

        # AC #2, with real multi-worker pickling: never replaced by a
        # worker's copy, and each point still gets its own folder.
        assert tb_par.netlist is live_netlist
        last = records_par[-1]
        assert float(tb_par.netlist.load_signal(
            'vdrain', 'I').y[-1]) == -last['metrics']['drain_current_A']
        run_dirs = sorted(glob.glob(os.path.join(dut_par.dir_folder, '*~run~point_*')))
        assert len(run_dirs) == 4, run_dirs
        raw_files = {os.path.realpath(glob.glob(os.path.join(d, 'run-*', '*.raw'))[0])
                    for d in run_dirs}
        assert len(raw_files) == 4


# TASK-509 regressions are dependency-free, including the worker cases.
from tests.test_run_split import replacement_backend, _replacement_bench


@pytest.mark.parametrize('api', ['evaluate', 'serial', 'parallel'])
@pytest.mark.parametrize('declaration', ['mc', 'corners', 'single_corner', 'both'])
def test_optimizer_refuses_declared_collection_before_preparation(
        tmp_path, monkeypatch, api, declaration):
    from silk.exceptions import NotSupportedError
    tb = _replacement_bench(tmp_path)
    x = Variable('x', value=1, bounds=(1, 2))
    opt = Optimizer(tb, VariableSpace([x]))
    if declaration in ('mc', 'both'):
        tb.set_monte_carlo(3)
    if declaration in ('corners', 'single_corner', 'both'):
        tb.set_corners(['tt'] if declaration == 'single_corner' else ['tt', 'ss'])
    def forbidden(*args, **kwargs):
        pytest.fail('collection reached preparation or simulation')
    monkeypatch.setattr(tb, '_prepare_run', forbidden)
    monkeypatch.setattr(tb, '_simulate_run', forbidden)
    with pytest.raises(NotSupportedError) as caught:
        if api == 'evaluate':
            opt.evaluate({'x': 2})
        else:
            opt.sweep(2, method='grid', n_workers=2 if api == 'parallel' else 1)
    assert ('set_monte_carlo' if declaration == 'mc' else 'set_corners') in str(caught.value)
    assert opt.history == []
    assert x.value == 1
    assert not list(tmp_path.rglob('*.cir'))


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('mode', ['before', 'after'])
def test_failed_final_point_matches_serial_live_state(
        tmp_path, replacement_backend, workers, mode):
    tb = _replacement_bench(tmp_path, mode)
    x = Variable('vg', value=1, bounds=(1, 2))
    tb.netlist.sig_sources_dict['vg'].dc = x
    live = tb.netlist
    opt = Optimizer(tb, VariableSpace([x]))
    records = opt.sweep(2, method='grid', n_workers=workers)
    assert [r['status'] for r in records] == ['success', 'simulation_failure']
    assert tb.netlist is live
    assert tb.netlist.sig_sources_dict['vg'].dc is x
    assert x.value == 2
    assert tb.netlist.sim_results_loaded is (mode == 'before')
    if mode == 'before':
        assert tb.netlist.load_signal('vout', 'V').y[-1] == 1.25
        assert 'point_0' in tb.netlist.results_file
    else:
        for name in ('_raw', 'simResultsAll', 'results_file', 'dataFile'):
            assert getattr(tb.netlist, name) is None


@pytest.mark.parametrize('workers', [1, 2])
def test_prepare_failure_preserves_evidence_and_final_parameters(
        tmp_path, replacement_backend, monkeypatch, workers):
    tb = _replacement_bench(tmp_path)
    x = Variable('vg', value=1, bounds=(1, 2))
    tb.netlist.sig_sources_dict['vg'].dc = x
    original = tb._prepare_run
    def prepare(*args, **kwargs):
        if x.value == 2:
            raise SimulationError('preparation failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(tb, '_prepare_run', prepare)
    opt = Optimizer(tb, VariableSpace([x]))
    records = opt.sweep(2, method='grid', n_workers=workers)
    assert [r['status'] for r in records] == ['success', 'simulation_failure']
    assert x.value == 2
    assert tb.netlist.sim_results_loaded
    assert tb.netlist.load_signal('vout', 'V').y[-1] == 1.25


def test_sweep_uses_testbench_planner(tmp_path, replacement_backend, monkeypatch):
    tb = _replacement_bench(tmp_path)
    opt = Optimizer(tb, VariableSpace([Variable('x', value=1, bounds=(1, 2))]))
    calls = []
    original = tb._plan_runs
    def plan(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(tb, '_plan_runs', plan)
    assert all(r['status'] == 'success' for r in opt.sweep(2, method='grid'))
    assert len(calls) >= 2


@pytest.mark.parametrize('workers', [1, 2])
def test_custom_run_override_is_not_bypassed(tmp_path, workers):
    from silk.circuit.module_test import ModuleTestbench
    class Custom(ModuleTestbench):
        def run(self, corner=None):
            return {'metrics': {'custom': 42}}
    tb = _replacement_bench(tmp_path)
    tb.__class__ = Custom
    opt = Optimizer(tb, VariableSpace([Variable('x', value=1, bounds=(1, 2))]))
    if workers == 2:
        with pytest.raises(TypeError, match='run'):
            opt.sweep(2, method='grid', n_workers=workers)
    else:
        assert [r['metrics'] for r in opt.sweep(2, method='grid')] == [
            {'custom': 42}, {'custom': 42}]


def test_cleared_declarations_allow_scalar_optimizer(tmp_path, replacement_backend):
    tb = _replacement_bench(tmp_path)
    tb.set_monte_carlo(3).set_corners(['tt'])
    tb.set_monte_carlo(None).set_corners(None)
    opt = Optimizer(tb, VariableSpace([Variable('x', value=1, bounds=(1, 2))]))
    assert opt.evaluate({'x': 1})['status'] == 'success'
    assert all(r['status'] == 'success' for r in opt.sweep(2, method='grid'))


def test_unknown_executor_failure_preserves_evidence(tmp_path, replacement_backend, monkeypatch):
    from silk.simulation.executor import RunOutcome
    tb = _replacement_bench(tmp_path)
    tb.run('tt')
    raw = tb.netlist._raw
    x = Variable('x', value=1, bounds=(1, 2))
    opt = Optimizer(tb, VariableSpace([x]))
    monkeypatch.setattr(optimizer_module, 'run_tasks', lambda tasks, **kw: [
        RunOutcome(index=i, run_dir=t.run_dir, error='transport failure')
        for i, t in enumerate(tasks)])
    records = opt.sweep(2, method='grid', n_workers=2)
    assert [r['status'] for r in records] == ['unexpected_error'] * 2
    assert x.value == 2
    assert tb.netlist._raw is raw
    assert tb.netlist.sim_results_loaded


@pytest.mark.parametrize('workers', [1, 2])
def test_parameter_failure_and_success_order(tmp_path, replacement_backend, monkeypatch, workers):
    tb = _replacement_bench(tmp_path)
    x = Variable('vg', value=1, bounds=(1, 2))
    tb.netlist.sig_sources_dict['vg'].dc = x
    space = VariableSpace([x])
    monkeypatch.setattr(space, 'sample', lambda *a, **kw: [{'vg': 2}, {'vg': 3}])
    records = Optimizer(tb, space).sweep(2, n_workers=workers)
    assert [r['status'] for r in records] == ['success', 'parameter_error']
    assert x.value == 2
    assert tb.netlist.sim_results_loaded


@pytest.mark.parametrize('workers', [1, 2])
def test_success_after_failure_and_metric_error_retain_new_data(
        tmp_path, replacement_backend, monkeypatch, workers):
    tb = _replacement_bench(tmp_path, 'after')
    x = Variable('vg', value=1, bounds=(1, 2))
    tb.netlist.sig_sources_dict['vg'].dc = x
    space = VariableSpace([x])
    monkeypatch.setattr(space, 'sample', lambda *a, **kw: [{'vg': 2}, {'vg': 1}])
    def bad_metric(ckt):
        raise ValueError('metric failed')
    tb.add_metric('bad', bad_metric)
    records = Optimizer(tb, space).sweep(2, n_workers=workers)
    assert [r['status'] for r in records] == ['simulation_failure', 'metric_error']
    assert x.value == 1
    assert tb.netlist.sim_results_loaded
    assert tb.netlist.load_signal('vout', 'V').y[-1] == 1.25


def test_report_interpretation_errors_are_not_simulation_failures(tmp_path, monkeypatch):
    from silk.circuit.spec_report import SpecReport
    tb = _replacement_bench(tmp_path)
    tb.add_metric('not_orderable', lambda ckt: 1j)
    opt = Optimizer(tb, VariableSpace([Variable('x', value=1, bounds=(1, 2))]))
    opt.add_constraint('not_orderable', min=0)
    report = SpecReport(test_name='op', module_name='amp',
                        metrics={'gain': 10, 'not_orderable': 1j})
    monkeypatch.setattr(tb, '_prepare_run', lambda *args, **kwargs: report)
    # A scalar report has completed. Its invalid ordering is a caller error,
    # just as it is in evaluate(), not a failure of the simulation attempt.
    with pytest.raises((TypeError, ValueError)):
        opt.sweep(1)


@pytest.fixture
def nested_optimizer_bench(tmp_path, request, monkeypatch):
    """Two placements ensure preflight keeps the selected ancestor context."""
    tb = _replacement_bench(tmp_path / 'dut')
    dut = tb.module
    middle = Module(None, str(tmp_path / 'middle'), 'middle.cir', None,
                    module_name='middle')
    middle.define_module_ports({'d': 'out', 'g': 'in', 's': 'ground'})
    middle.add_module(dut, 'u1', d='d', g='g', s='s')
    top = Module(None, str(tmp_path / 'top'), 'top.cir', None)
    top.add_module(middle, 'u0', d='out', g='in', s='0')
    for module in (dut, middle, top):
        request.addfinalizer(module.close)
    from pathlib import Path
    import re
    from tests._mock_rawfile import write_complete_rawfile
    backend = tb.netlist.backend

    def launch(_backend, netlist_path, results_dir, **kwargs):
        deck = Path(netlist_path).read_text()
        path = Path(results_dir) / 'synthetic.raw'
        write_complete_rawfile(path, deck, values={'v(vout)': 1.25})
        result = backend.load_raw(str(path))
        temperature = re.search(r'\btemp=([^\s]+)', deck)
        if temperature:
            result.ambient_temperatures = (float(temperature[1]),)
        return result

    monkeypatch.setattr(type(backend), 'run', launch)
    selected = top.get_test('op', instance='u0.u1')
    assert top.list_test_entries()[0].test is selected
    knob = Variable('vg', value=1, bounds=(1, 3))
    tb.netlist.sig_sources_dict['vg'].dc = knob
    return tb, dut, top, selected, knob


def _declare_optimizer_collection(target, declaration):
    if declaration in ('mc', 'all'):
        target.set_monte_carlo(2)
    if declaration in ('corners', 'single_corner', 'all'):
        target.set_corners(['tt'] if declaration == 'single_corner' else ['tt', 'ss'])
    if declaration in ('temperature', 'axes', 'all'):
        target.set_temperatures([27, 85])
    if declaration in ('supply', 'axes', 'all'):
        target.set_supply_voltages([1.0, 1.8], source='vd')


@pytest.mark.parametrize('api', ['evaluate', 'serial', 'parallel'])
@pytest.mark.parametrize('route', ['authored', 'selected', 'inherited'])
@pytest.mark.parametrize('declaration', [
    'mc', 'corners', 'single_corner', 'temperature', 'supply', 'axes', 'all'])
@pytest.mark.parametrize('retained', [False, True], ids=['fresh', 'retained'])
def test_optimizer_collection_refusal_preserves_selected_evidence(
        tmp_path, monkeypatch, nested_optimizer_bench,
        api, route, declaration, retained):
    from silk.exceptions import NotSupportedError
    tb, dut, top, selected, knob = nested_optimizer_bench
    bench = tb if route == 'authored' else selected
    opt = Optimizer(bench, VariableSpace([knob]))
    report = bench.run('tt') if retained else None
    reader = bench._result_netlist()
    raw = getattr(reader, '_raw', None)
    if retained:
        assert report.sim_results is not None
        assert reader.load_signal('vout', 'V').y[-1] == 1.25
    # All declarations happen after both the selector and optimizer exist.
    # Inherited grids must be read through the selected receiver, not tb.
    if route == 'inherited':
        if declaration in ('mc', 'corners', 'single_corner', 'all'):
            _declare_optimizer_collection(tb, declaration)
        if declaration in ('temperature', 'axes', 'all'):
            top.set_temperatures([27, 85])
        if declaration in ('supply', 'axes', 'all'):
            top.set_supply_voltages([1.0, 1.8], source='vd')
    else:
        _declare_optimizer_collection(tb, declaration)
    paths = {p: p.read_bytes() if p.is_file() else None
             for p in tmp_path.rglob('*')}

    def forbidden(*args, **kwargs):
        pytest.fail('collection reached preparation or simulation')
    monkeypatch.setattr(tb, '_prepare_run', forbidden)
    monkeypatch.setattr(tb, '_simulate_run', forbidden)
    with pytest.raises(NotSupportedError, match='Optimizer requires one report per point'):
        if api == 'evaluate':
            opt.evaluate({'vg': 2})
        else:
            opt.sweep(2, method='grid', n_workers=2 if api == 'parallel' else 1)
    assert knob.value == 1
    assert opt.history == []
    assert bench._result_netlist() is reader
    assert getattr(reader, '_raw', None) is raw
    if route != 'authored':
        assert selected.report is report
    if retained:
        assert report.sim_results is not None
        assert reader.load_signal('vout', 'V').y[-1] == 1.25
    assert {p: p.read_bytes() if p.is_file() else None
            for p in tmp_path.rglob('*')} == paths


@pytest.mark.parametrize('axis', ['temperature', 'supply'])
@pytest.mark.parametrize('clear_at', ['test', 'owner'])
def test_selected_optimizer_clear_masks_ancestor_list(
        nested_optimizer_bench, axis, clear_at):
    tb, dut, top, selected, knob = nested_optimizer_bench
    top.set_temperature(50)
    top.set_supply_voltage(1.2, source='vd')
    opt = Optimizer(selected, VariableSpace([knob]))
    setter = 'set_temperatures' if axis == 'temperature' else 'set_supply_voltages'
    kwargs = {} if axis == 'temperature' else {'source': 'vd'}
    getattr(top, setter)([27, 85] if axis == 'temperature' else [1.0, 1.8], **kwargs)
    from silk.exceptions import NotSupportedError
    with pytest.raises(NotSupportedError):
        opt.evaluate({'vg': 2})
    getattr(tb if clear_at == 'test' else dut, setter)(None)
    result = opt.evaluate({'vg': 2})
    assert result['status'] == 'success', result['error']
    assert all(r['status'] == 'success' for r in opt.sweep(2, method='grid'))
    report = selected.report
    assert report.owner_path == '$/u0/u1'
    conditions = report.sim_results.operating_conditions
    assert conditions.temperature_c == 50
    assert conditions.supply_voltage_v == 1.2


def test_selected_scalar_optimizer_keeps_serial_and_parallel_contract(
        nested_optimizer_bench):
    tb, dut, top, selected, knob = nested_optimizer_bench
    top.set_temperature(50)
    top.set_supply_voltage(1.2, source='vd')
    opt = Optimizer(selected, VariableSpace([knob]))
    result = opt.evaluate({'vg': 2})
    assert result['status'] == 'success', result['error']
    assert all(r['status'] == 'success' for r in opt.sweep(2, method='grid'))
    report = selected.report
    reader = selected._result_netlist()
    history = list(opt.history)
    value = knob.value
    with pytest.raises(TypeError, match='ModuleTestbench'):
        opt.sweep(2, method='grid', n_workers=2)
    assert knob.value == value
    assert opt.history == history
    assert selected.report is report
    assert selected._result_netlist() is reader
    assert report.owner_path == '$/u0/u1'
    assert report.sim_results.operating_conditions.temperature_c == 50
    assert report.sim_results.operating_conditions.supply_voltage_v == 1.2
