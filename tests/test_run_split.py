"""ModuleTestbench.run() = prepare -> simulate -> evaluate (TASK-301.2).

No behaviour change is the whole point of the split: run() itself must still
produce the same decks, rawfiles, SpecReports and Spec measurements it did
before -- proved here on a PDK-free RC deck, so it needs neither a PDK
install nor a mocked simulator. The riskiest claim -- that ``_simulate_run``
is genuinely self-free and can run from a pickled netlist in a real worker
process -- gets its own proof at the bottom, on a real (slow) sky130 NMOS
DUT instead: see that test's docstring for why the PDK-free deck above
cannot stand in for it.
"""
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import pytest

from silk import Port, load_pdk, sources
from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench, _RunSimResult, _RunSpec
from silk.elements import elements
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk


def _build_lowpass_tb(tmp_dir, name='lowpass', *, db_path=None):
    """A PDK-free RC low-pass: no PDK model, so it needs no PDK install.

    ``create_test`` is called on the DUT itself (the ordinary shape — see
    ``tests/test_dut_must_be_subcircuit.py``): the DUT is instantiated as a
    subcircuit inside the testbench netlist ``create_test`` builds.
    """
    dut = Module(None, tmp_dir, f'{name}.cir', pdk_details=None,
                 module_name='rc_lowpass_' + name, db_path=db_path)
    dut.define_module_ports({'in': 'in', 'out': 'out', 'vss': 'ground'})
    dut.add(elements.resistor('r1', p='in', n='out', value=1000.0))
    dut.add(elements.capacitor('c1', p='out', n='vss', value=1e-9))
    tb = dut.create_test(name)
    tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1.0))
    tb.add_sim_type(Transient(step=1e-9, stop=200e-9))
    tb.add_metric('vout_final',
                 lambda ckt: float(ckt.load_signal('vout', 'V').y[-1]))
    tb.add_spec('rises', 'vout_final', '>', 0.0)
    return dut, tb


def _mask_date(raw):
    """Blank the rawfile ``Date:`` header line.

    The simulator stamps it with one-second resolution, so two otherwise
    identical runs differ whenever they straddle a second boundary (TASK-424:
    the intermittent failure, likelier under load because the runs stretch).
    """
    return re.sub(rb'^Date: [^\n]*\n', b'Date: <masked>\n', raw, count=1,
                  flags=re.MULTILINE)


@needs_ngspice
class TestRunSplitNoBehaviourChange:
    def test_run_is_prepare_then_simulate_then_evaluate(self, tmp_path):
        dut, tb = _build_lowpass_tb(str(tmp_path))
        try:
            report = tb.run(corner='tt')
            assert report.metrics['vout_final'] > 0.0
            assert report.results[0].passed is True
        finally:
            dut.close()

    def test_prepare_returns_a_run_spec_on_a_miss(self, tmp_path):
        dut, tb = _build_lowpass_tb(str(tmp_path))
        try:
            spec = tb._prepare_run('tt')
            assert isinstance(spec, _RunSpec)
            assert spec.corner_str == 'tt'
            assert spec.netlist is tb.netlist
            assert spec.test_name == tb.test_name
        finally:
            dut.close()

    def test_split_run_matches_run_bit_for_bit(self, tmp_path):
        """prepare -> simulate -> evaluate, called by hand, must match run()."""
        # Same bench name on both sides -- only the output directory differs
        # -- so the emitted deck/rawfile Title lines match byte for byte too.
        dut_a, tb_a = _build_lowpass_tb(str(tmp_path / 'a'), name='same')
        dut_b, tb_b = _build_lowpass_tb(str(tmp_path / 'b'), name='same')
        try:
            report_direct = tb_a.run(corner='tt')

            spec = tb_b._prepare_run('tt')
            result = ModuleTestbench._simulate_run(spec)
            assert isinstance(result, _RunSimResult)
            report_split = tb_b._evaluate_run(result)

            assert report_split.metrics == report_direct.metrics
            assert (report_split.results[0].passed
                    == report_direct.results[0].passed)
            with open(tb_a.results_file, 'rb') as fa, \
                 open(tb_b.results_file, 'rb') as fb:
                assert _mask_date(fa.read()) == _mask_date(fb.read())
        finally:
            dut_a.close()
            dut_b.close()


def _build_nmos_tb(tmp_dir, pdk, name):
    """A real sky130 NMOS DUT -- the shape ``FaultCampaign`` already proves
    picklable, and unlike the ideal RC deck above it cannot collide with
    another test's device-class registration (see the test below).
    """
    dut = Module('PDK_skywater130', tmp_dir, f'{name}.cir',
                 pdk_details=pdk, module_name='nmos_' + name)
    dut.define_module_ports({'D': Port.analog_output(), 'G': Port.analog_input(),
                             'S': Port.ground()})
    with dut:
        pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=2.0, l=0.5)
    tb = dut.create_test(name)
    tb.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
    tb.add_source(sources.VoltageSource('drain', positive='drain', negative='0', dc=0.8))
    tb.add_source(sources.VoltageSource('gate', positive='gate', negative='0', dc=1.0))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.add_metric('drain_current_A',
                 lambda ckt: -float(ckt.load_signal('vdrain', 'I').y[-1]))
    tb.add_spec('conducts', 'drain_current_A', '>', 0.0)
    return dut, tb


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_simulate_run_executes_from_a_pickled_copy_in_a_separate_process(tmp_path):
    """AC #2 (and the slow sky130 leg of AC #1): _simulate_run must work on
    a pickled netlist, in a real worker process -- not merely be self-free
    in principle.

    A real sky130 NMOS DUT, not the ideal-element RC deck above:
    ``elements.resistor``/``capacitor`` share their class *name* with
    whatever PDK-specific device a YAML also happens to call 'resistor'
    (last registration wins in ``silk.elements.factory``'s pickle registry,
    by design -- see its docstring), so an ideal-element deck can pickle
    against a stale class once another test in the same process has loaded
    a PDK. A PDK MOS device has no such collision — this is the same shape
    ``FaultCampaign`` already pickles.
    """
    pdk = load_pdk('sky130')
    dut_ref, tb_ref = _build_nmos_tb(str(tmp_path / 'ref'), pdk, 'ref')
    dut_worker, tb_worker = _build_nmos_tb(str(tmp_path / 'worker'), pdk, 'worker')
    try:
        report_ref = tb_ref.run(corner='tt')

        spec = tb_worker._prepare_run('tt')
        # Prove it round-trips pickle before crossing a real process boundary.
        spec = pickle.loads(pickle.dumps(spec))
        with ProcessPoolExecutor(max_workers=1) as pool:
            result = pool.submit(ModuleTestbench._simulate_run, spec).result()
        assert isinstance(result, _RunSimResult)
        report_worker = tb_worker._evaluate_run(result)

        assert report_worker.metrics == report_ref.metrics
        assert report_worker.results[0].passed == report_ref.results[0].passed
    finally:
        dut_ref.close()
        dut_worker.close()


class TestPublicPrepare:
    """External runners get run's live settings without recording a run."""

    @pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
    def test_live_settings_without_simulating(self, tmp_path, monkeypatch, simulator):
        from pathlib import Path
        dut, tb = _build_lowpass_tb(str(tmp_path))
        try:
            dut.simulator = simulator
            dut.set_temperature(85.0)
            dut.set_simulator_options(reltol=1e-4)
            dut.set_supply_voltage(1.8, source='vin')
            dut.save_device_currents = True
            dut.terminal_current_method = 'ammeter_wrapper'
            dut.save_device_params = False
            evidence = object()
            tb._metric_values = {'previous': evidence}
            tb._cache = {'previous': evidence}
            tb._sweep_table = {'previous': 'old.raw'}
            tb._cache_enabled = True

            def forbidden(*args, **kwargs):
                pytest.fail('prepare must not simulate, evaluate or consult the cache')

            monkeypatch.setattr(tb.netlist, 'simulate', forbidden)
            monkeypatch.setattr(tb, 'evaluate', forbidden)
            monkeypatch.setattr(tb, '_fingerprint', forbidden)
            path = tb.prepare(temperature=85.0, voltage=1.8, run_label='external')
            deck = Path(path).read_text()
            assert 'reltol=0.0001' in deck.lower()
            if simulator == 'ngspice':
                assert '.options temp=85.0 reltol=0.0001' in deck
            assert tb.netlist.simulator == simulator
            assert tb.netlist.save_device_currents is True
            assert tb.netlist.terminal_current_method == 'ammeter_wrapper'
            assert tb.netlist.save_device_params is False
            assert tb._metric_values == {'previous': evidence}
            assert tb._cache == {'previous': evidence}
            assert tb._sweep_table == {'previous': 'old.raw'}
            assert not list(tmp_path.rglob('*.raw'))
        finally:
            dut.close()

    @pytest.mark.parametrize('case', ['behavioral', 'Monte Carlo', 'multi-deck'])
    def test_refuses_deckless_or_multiple_runs_before_writing(self, tmp_path, case):
        from silk.exceptions import NotSupportedError
        dut, tb = _build_lowpass_tb(str(tmp_path))
        try:
            if case == 'behavioral':
                dut.simulator = 'behavioral'
            elif case == 'Monte Carlo':
                tb.set_monte_carlo(2)
            else:
                tb.set_corners(['tt', 'ff'])
            with pytest.raises(NotSupportedError, match=case):
                tb.prepare()
            assert not list(tmp_path.rglob('*.cir'))
        finally:
            dut.close()

    def test_new_mode_returns_concrete_unique_deck(self, tmp_path):
        from pathlib import Path
        dut, tb = _build_lowpass_tb(str(tmp_path))
        try:
            dut.set_output_options(mode='new')
            first = Path(tb.prepare(run_label='external'))
            second = Path(tb.prepare(run_label='external'))
            assert first.is_file() and second.is_file()
            assert first.parent != second.parent
            assert first.parent.parent == second.parent.parent
        finally:
            dut.close()


@needs_ngspice
@pytest.mark.parametrize('declared', [False, True])
def test_public_prepare_matches_run_deck(tmp_path, declared):
    _assert_public_prepare_matches_run(tmp_path, 'ngspice', declared)


@pytest.mark.slow
@pytest.mark.requires_xyce
def test_public_prepare_matches_xyce_run_deck(tmp_path):
    from silk import settings
    import shutil
    if not shutil.which(settings.get('SILK_XYCE_BIN')):
        pytest.skip('needs configured Xyce')
    _assert_public_prepare_matches_run(tmp_path, 'xyce', False)


def _assert_public_prepare_matches_run(tmp_path, simulator, declared):
    from pathlib import Path
    dut, tb = _build_lowpass_tb(str(tmp_path))
    try:
        dut.simulator = simulator
        dut.set_output_options(mode='overwrite')
        dut.set_temperature(85.0)
        dut.set_simulator_options(reltol=1e-4)
        dut.set_supply_voltage(1.8, source='vin')
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        dut.save_device_params = False
        if declared:
            tb.set_corners(['tt', 'ff'])
        kwargs = dict(corner='tt', temperature=85.0, voltage=1.8,
                      run_label='external')
        path = Path(tb.prepare(**kwargs))
        before = path.read_bytes()
        if declared:
            assert path.parent.name == 'tt'
        tb.run(**kwargs)
        assert Path(tb.netlist._run_files.netlist_path) == path
        assert path.read_bytes() == before
        # Preparing again preserves the evidence and existing output bytes.
        metrics = dict(tb._metric_values)
        measurements = [dict(item['spec'].measurements)
                        for item in tb._specs if item.get('spec') is not None]
        conditions = tb.netlist.operating_conditions
        sweep_table = dict(tb._sweep_table)
        rawpath = Path(tb.results_file)
        raw = rawpath.read_bytes()
        tb.prepare(**kwargs)
        assert tb._metric_values == metrics
        assert [dict(item['spec'].measurements) for item in tb._specs
                if item.get('spec') is not None] == measurements
        assert tb.netlist.operating_conditions is conditions
        assert tb._sweep_table == sweep_table
        assert rawpath.read_bytes() == raw
    finally:
        dut.close()


@pytest.mark.slow
@needs_pdk
@pytest.mark.parametrize('simulator', [
    pytest.param('ngspice', marks=needs_ngspice),
    pytest.param('xyce', marks=pytest.mark.requires_xyce),
])
@pytest.mark.parametrize('capture', ['currents', 'params'])
def test_prepare_transistor_capture_matches_run(tmp_path, simulator, capture):
    from pathlib import Path
    from silk import settings
    from silk.simulation import Op
    import shutil
    if simulator == 'xyce' and not shutil.which(settings.get('SILK_XYCE_BIN')):
        pytest.skip('needs configured Xyce')
    dut, tb = _build_nmos_tb(str(tmp_path), load_pdk('sky130'), 'capture')
    try:
        dut.simulator = simulator
        dut.save_device_currents = capture == 'currents'
        dut.save_device_params = capture == 'params'
        dut.terminal_current_method = 'ammeter_wrapper'
        if capture == 'params':
            tb.netlist.sim_analysis_dict.clear()
            tb.add_sim_type(Op())
        path = Path(tb.prepare())
        deck = path.read_bytes()
        if capture == 'params':
            assert b'gm' in deck.lower()
        else:
            assert b'Vamm_d d d_i 0' in deck
        tb.run()
        assert path.read_bytes() == deck
    finally:
        dut.close()


def test_public_prepare_rejects_unknown_pdk_corner(tmp_path):
    dut, tb = _build_nmos_tb(str(tmp_path), load_pdk('sky130'), 'invalid')
    try:
        with pytest.raises(ValueError, match='unknown corner'):
            tb.prepare(corner='not_a_corner')
        assert not list(tmp_path.rglob('*.cir'))
    finally:
        dut.close()


def test_public_prepare_leaves_database_unchanged(tmp_path):
    import sqlite3
    from contextlib import closing
    db_path = str(tmp_path / 'evidence.db')
    dut, tb = _build_lowpass_tb(str(tmp_path), db_path=db_path)
    try:
        with closing(sqlite3.connect(db_path)) as connection:
            before = list(connection.iterdump())
            tb.prepare()
            assert list(connection.iterdump()) == before
    finally:
        dut.close()


@pytest.mark.parametrize('simulator,param', [
    ('ngspice', 'gm'), ('ngspice', 'cgso'), ('xyce', 'gm'),
])
def test_prepare_never_launches_device_parameter_preflight(
        tmp_path, monkeypatch, simulator, param):
    from pathlib import Path
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.exceptions import NotSupportedError
    dut, tb = _build_nmos_tb(str(tmp_path), load_pdk('sky130'), 'preflight')
    try:
        dut.simulator = simulator
        tb.probe(f'm1:{param}')

        def forbidden(*args, **kwargs):
            pytest.fail('prepare must not run or consult device-parameter preflight')

        monkeypatch.setattr(PySpiceNetlister, '_ngspice_param_preflight', forbidden)
        if param == 'cgso':
            with pytest.raises(NotSupportedError, match='m1:cgso.*preflight'):
                tb.prepare()
            assert not list(tmp_path.rglob('*.cir'))
        else:
            assert Path(tb.prepare()).is_file()
    finally:
        dut.close()


@needs_ngspice
def test_public_prepare_uses_each_selection_and_matches_its_run(tmp_path):
    from pathlib import Path
    dut, tb = _build_lowpass_tb(str(tmp_path), name='placed')
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.set_output_options(mode='overwrite')
    parents = []
    try:
        for name, voltage, temperature in [('one', 0.75, 25.0), ('two', 1.5, 85.0)]:
            parent = Module(None, str(tmp_path), name + '.cir', None)
            parent.set_supply_voltage(voltage, source='vin')
            parent.set_temperature(temperature)
            parent.define_module_name('carrier')
            parent.define_module_ports({})
            parent.add_module(dut, 'cell', nets={'in': 'in', 'out': 'out', 'vss': '0'})
            top.add_module(parent, name, nets={})
            parents.append(parent)
        entries = top.list_test_entries()
        prepared = {}
        for entry, voltage, temperature in zip(entries, (0.75, 1.5), (25.0, 85.0)):
            selected = entry.test
            path = Path(selected.prepare(run_label='external'))
            deck = path.read_bytes()
            assert f'temp={temperature}'.encode() in deck
            source_line = next(line for line in deck.decode().splitlines() if line.startswith('Vvin '))
            assert str(voltage) in source_line
            assert selected.owner_path == entry.owner_path
            assert selected.report_key == entry.report_key
            assert selected.results_file is None
            prepared[entry.path] = (deck, voltage)
        assert not list(tmp_path.rglob('*.raw'))
        measured = []
        for entry in reversed(top.list_test_entries()):
            selected = entry.test
            deck, voltage = prepared[entry.path]
            path = Path(selected.prepare(run_label='external'))
            assert path.read_bytes() == deck
            report = selected.run(run_label='external')
            assert path.read_bytes() == deck
            assert report.owner_path == entry.owner_path
            measured.append(report.metrics['vout_final'] / voltage)
            rawpath = Path(selected.results_file)
            raw = rawpath.read_bytes()
            selected.prepare(run_label='external')
            assert rawpath.read_bytes() == raw
            assert selected._metric_values == report.metrics
        assert measured[0] == pytest.approx(measured[1])
        assert tb.netlist.sig_sources_dict['vin'].dc == 1.0
        assert not tb.netlist._temperature_explicit
        assert not tb.netlist._supply_explicit
    finally:
        top.close()
        dut.close()
        for parent in parents:
            parent.close()


@pytest.mark.parametrize('edit', ['source', 'add_source', 'analysis', 'options'])
def test_selected_prepare_reads_authored_bench_edits(tmp_path, edit):
    from pathlib import Path
    dut, tb = _build_lowpass_tb(str(tmp_path))
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_module(dut, 'one', nets={'in': 'in', 'out': 'out', 'vss': '0'})
    selected = top.get_test(tb.test_name, instance='one')
    source = tb.netlist.sig_sources_dict['vin']
    try:
        selected.prepare()
        if edit == 'source':
            source.dc = 2.0
        elif edit == 'add_source':
            tb.add_source(sources.VoltageSource('extra', positive='extra', negative='0', dc=0.3))
        elif edit == 'analysis':
            tb.add_sim_type(Transient(step=2e-9, stop=400e-9))
        else:
            tb.set_simulator_options(reltol=3e-4)
        text = Path(selected.prepare()).read_text()
        assert selected.netlist is tb.netlist
        if edit == 'source':
            assert 'Vvin vin 0 DC 2\n' in text
        elif edit == 'add_source':
            assert 'Vextra extra 0 DC 0.3' in text
        elif edit == 'analysis':
            assert re.search(r'tran\s+2n\s+400n', text)
        else:
            assert 'reltol=0.0003' in text
        source.dc = 4.0
        assert 'Vvin vin 0 DC 4\n' in Path(selected.prepare()).read_text()
    finally:
        top.close()
        dut.close()


# TASK-509: exercise the real replacement boundaries, with no simulator process.
@pytest.fixture
def replacement_backend(monkeypatch):
    from pathlib import Path
    from unittest.mock import patch
    from silk.exceptions import SimulationError
    from tests._mock_rawfile import write_complete_rawfile

    original = Module.run_simulator

    def run(net, *args, **kwargs):
        bias = net.sig_sources_dict.get('vg')
        value = getattr(getattr(bias, 'dc', None), 'value', None)
        fail = value == 2 or net._last_requested_corner == 'ss'
        mode = getattr(net, '_test_failure_mode', 'none')
        backend = net.backend

        def launch(_backend, netlist_path, results_dir, **options):
            if fail and mode == 'after':
                raise SimulationError('injected replacement failure')
            path = Path(results_dir) / 'synthetic.raw'
            write_complete_rawfile(path, Path(netlist_path).read_text(), values={'v(vout)': 1.25})
            return backend.load_raw(str(path))

        with patch.object(type(backend), 'run', launch):
            if fail and mode == 'before':
                with patch('silk.circuit.modules_netlist._require_fresh_deck',
                           side_effect=SimulationError('injected pre-launch refusal')):
                    return original(net, *args, **kwargs)
            return original(net, *args, **kwargs)

    original_simulate = Module.simulate

    def simulate(net, corner=None, **kwargs):
        net._last_requested_corner = corner
        return original_simulate(net, corner=corner, **kwargs)

    monkeypatch.setattr(Module, 'run_simulator', run)
    monkeypatch.setattr(Module, 'simulate', simulate)


def _replacement_bench(path, mode='none'):
    from tests.test_monte_carlo import _bench
    tb = _bench(path, 'sky130', None)
    tb.netlist._test_failure_mode = mode
    return tb


@pytest.mark.parametrize('runner', ['run', 'corners', 'plan', 'module', 'module_plan'])
@pytest.mark.parametrize('mode', ['before', 'after'])
@pytest.mark.parametrize('workers', [1, 2])
def test_run_consumers_preserve_replacement_boundary(
        tmp_path, replacement_backend, runner, mode, workers):
    from silk.circuit.spec_report import CornerRunError
    from silk.exceptions import SimulationError
    tb = _replacement_bench(tmp_path, mode)
    tb.run('tt')
    earlier = tb.netlist.results_file
    assert tb.netlist.sim_results_loaded
    if runner == 'run':
        with pytest.raises(SimulationError):
            tb.run('ss')
    elif runner == 'corners':
        with pytest.raises(CornerRunError) as caught:
            tb.run_corners(['tt', 'ss'], n_workers=workers)
        assert list(caught.value.results.reports) == ['tt']
        earlier = caught.value.results['tt'].rawfile_path
    elif runner == 'plan':
        tb.set_corners(['ss'])
        with pytest.raises(SimulationError):
            tb._run_plan(tb._plan_runs(), n_workers=workers)
    else:
        if runner == 'module_plan':
            tb.set_corners(['ss'])
        reports = tb.module.run_all_tests(corner='ss', n_workers=workers)
        assert 'error' in next(iter(reports.values()))
    assert tb.netlist.sim_results_loaded is (mode == 'before')
    if mode == 'before':
        assert tb.netlist.results_file == earlier
        assert tb.netlist.load_signal('vout', 'V').y[-1] == 1.25
    else:
        assert tb.netlist._raw is None
        assert tb.netlist.simResultsAll is None
        assert tb.netlist.results_file is None
        assert tb.netlist.dataFile is None


@pytest.mark.parametrize('mode', ['before', 'after'])
def test_outcome_transports_replacement_fact(tmp_path, replacement_backend, mode):
    from silk.circuit import module_test
    tb = _replacement_bench(tmp_path, mode)
    tb.run('tt')
    for corner, expected in [('ss', mode == 'after'), ('tt', True),
                             ('ss', mode == 'after')]:
        outcome = module_test._simulate_run_outcome(tb._prepare_run(corner))
        transported = pickle.loads(pickle.dumps(outcome))
        assert transported.replacement_started is expected
        assert (transported.error is not None) is (corner == 'ss')


def test_module_backend_tracks_replacement_boundary(tmp_path, monkeypatch):
    from silk.circuit import module_test
    from silk.exceptions import NotSupportedError, SimulationError
    _, tb = _build_lowpass_tb(str(tmp_path))
    tb.netlist.simulator = 'behavioral'
    spec = tb._prepare_run('tt')
    tb.netlist.simulator_options['temp'] = 25
    refused = module_test._simulate_run_outcome(spec)
    assert isinstance(refused.error, NotSupportedError)
    assert refused.replacement_started is False
    tb.netlist.simulator_options.pop('temp')
    def fail(*args, **kwargs):
        raise SimulationError('module backend failed')
    monkeypatch.setattr(type(tb.netlist.backend), 'run_module', fail)
    failed = module_test._simulate_run_outcome(spec)
    assert isinstance(failed.error, SimulationError)
    assert failed.replacement_started is True


def test_unknown_outcome_does_not_invalidate_current_evidence(tmp_path, replacement_backend):
    from silk.circuit import module_test
    tb = _replacement_bench(tmp_path)
    tb.run('tt')
    raw = tb.netlist._raw
    unknown = module_test._RunOutcome(error=RuntimeError('transport failed'),
                                     replacement_started=None)
    with pytest.raises(RuntimeError, match='transport failed'):
        tb._consume_run_outcome(unknown)
    assert tb.netlist._raw is raw
    assert tb.netlist.sim_results_loaded


@pytest.mark.parametrize('error_name', ['SimulationAborted', 'ConvergenceError'])
def test_outcome_roundtrip_preserves_structured_simulation_errors(error_name):
    from silk.circuit.module_test import _RunOutcome
    from silk import exceptions
    error_type = getattr(exceptions, error_name)
    kwargs = {'simulator': 'ngspice', 'reason': 'no rows', 'data_rows': 0}
    if error_name == 'ConvergenceError':
        kwargs['matched_lines'] = ['timestep too small']
    error = error_type('backend failed', **kwargs)
    outcome = _RunOutcome(error=error, replacement_started=True)
    received = pickle.loads(pickle.dumps(outcome))
    assert type(received.error) is error_type
    assert received.error.__dict__ == error.__dict__
    assert str(received.error) == str(error)
    assert received.replacement_started is True


def test_direct_run_preserves_instance_simulation_hook(tmp_path, monkeypatch):
    from silk.exceptions import SimulationError
    tb = _replacement_bench(tmp_path)
    def injected(spec):
        raise SimulationError('instance hook used')
    monkeypatch.setattr(tb, '_simulate_run', injected)
    with pytest.raises(SimulationError, match='instance hook used'):
        tb.run('tt')


@pytest.mark.parametrize('kind', ['file', 'unicode', 'local', 'attribute', 'unicode_attribute', 'group'])
def test_outcome_error_transport_is_safe_and_keeps_diagnostic(kind):
    from silk.circuit.module_test import _RunOutcome
    if kind == 'file':
        error = FileNotFoundError(2, 'missing', '/tmp/sim.raw')
    elif kind == 'unicode':
        error = UnicodeDecodeError('utf8', b'\xff', 0, 1, 'bad')
    elif kind in ('local', 'group'):
        class LocalBackendError(ValueError):
            pass
        error = LocalBackendError('backend refused')
        if kind == 'group':
            import builtins
            if not hasattr(builtins, 'ExceptionGroup'):
                pytest.skip('ExceptionGroup requires Python 3.11')
            error = ExceptionGroup('backend failed', [error])
    elif kind == 'unicode_attribute':
        error = UnicodeDecodeError('utf8', b'\xff', 0, 1, 'bad')
        error.callback = lambda: None
    else:
        error = ValueError('backend failed')
        error.callback = lambda: None
    outcome = _RunOutcome(error=error, replacement_started=True)
    received = pickle.loads(pickle.dumps(outcome))
    assert received.replacement_started is True
    assert str(received.error) == str(error)
    assert received.error_type == type(error).__name__
    if kind in ('file', 'unicode'):
        assert type(received.error) is type(error)
        assert received.error.args == error.args
    if kind == 'file':
        assert received.error.filename == error.filename


def test_in_process_module_walk_preserves_instance_simulation_hook(tmp_path, monkeypatch):
    from silk.exceptions import SimulationError
    tb = _replacement_bench(tmp_path)
    tb.netlist.simulator = 'behavioral'
    def injected(spec):
        raise SimulationError('instance hook used')
    monkeypatch.setattr(tb, '_simulate_run', injected)
    result = tb.module.run_all_tests(n_workers=2)
    assert next(iter(result.values())) == {'error': 'instance hook used'}


@pytest.mark.parametrize('selected', [False, True])
def test_prepare_ngspice_target(tmp_path, monkeypatch, selected):
    from pathlib import Path
    from silk.exceptions import NotSupportedError
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    dut, tb = _build_lowpass_tb(str(tmp_path))
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_module(dut, 'one', nets={'in': 'in', 'out': 'out', 'vss': '0'})
    bench = top.get_test(tb.test_name, instance='one') if selected else tb
    dut.omit_device_internal_nodes = True
    top.omit_device_internal_nodes = True
    try:
        path = bench.prepare(ngspice_version='45.2')
        assert 'ngspice 45.2' in Path(path).read_text()
        with pytest.raises(NotSupportedError, match='save_nointernals'):
            bench.prepare()
        for backend in ('behavioral', 'xyce'):
            dut.simulator = backend
            with pytest.raises(ValueError, match='ngspice'):
                bench.prepare(ngspice_version='45.2')
    finally:
        dut.close()


@pytest.mark.parametrize('options', [{}, {'ngspice_version': None}])
def test_prepare_default_keeps_old_generation_hook_signature(tmp_path, monkeypatch, options):
    from pathlib import Path
    dut, tb = _build_lowpass_tb(str(tmp_path))
    original = tb.netlist.generate_netlist

    def old_signature(corner=None, subckt=None, fault_set=None,
                      manifest=None, seed=0):
        return original(corner, subckt, fault_set, manifest, seed)

    monkeypatch.setattr(tb.netlist, 'generate_netlist', old_signature)
    try:
        assert Path(tb.prepare(**options)).is_file()
    finally:
        dut.close()
