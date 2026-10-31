"""Managed output replacement preserves leased generations, never stale public data."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from silk.exceptions import SimulationError
from silk.simulation.backend import SimulatorBackend
from silk.simulation.run_files import RunFiles
from silk.signals.rawfile import read_rawfile


def _patch_simulator_launch(monkeypatch, launch, simulator='ngspice'):
    """Mock deck launches without intercepting dependency/tool discovery."""
    from silk.simulation.runner import ngspice_executable
    from silk.simulation.xyce_runner import _xyce_executable

    executable = ngspice_executable() if simulator == 'ngspice' else _xyce_executable()
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        if (isinstance(argv, (list, tuple)) and len(argv) > 1
                and argv[0] == executable
                and ('-r' in argv if simulator == 'ngspice'
                     else str(argv[1]).endswith('.cir'))):
            return launch(argv, *args, **kwargs)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', run)


def raw_bytes(value):
    return (b'Title: generation\nPlotname: Transient Analysis\nFlags: real\n'
            b'No. Variables: 3\nNo. Points: 2\nVariables:\n'
            b'0 time time\n1 v(out) voltage\n2 v(unread) voltage\nBinary:\n'
            + np.array([[0, value, value + 10], [1, value, value + 10]], dtype='<f8').tobytes())


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_failed_replacement_keeps_unread_old_generation(tmp_path, monkeypatch, simulator):
    backend = SimulatorBackend.get(simulator)
    files = RunFiles.resolve(str(tmp_path / 'test.cir'), str(tmp_path), log=simulator == 'ngspice')
    Path(files.netlist_path).write_text('* deck\n.end\n')
    values = iter([3, None])
    def launch(argv, **kwargs):
        value = next(values)
        if value is None:
            raise OSError('launch failed')
        if simulator == 'ngspice':
            target = argv[argv.index('-r') + 1]
        else:
            text = Path(argv[1]).read_text()
            target = text.split('file="')[1].split('"')[0]
        Path(target).write_bytes(raw_bytes(value))
        return SimpleNamespace(returncode=0, stdout='End of Xyce', stderr='')
    if simulator == 'xyce':
        Path(files.netlist_path).write_text(f'* deck\n.print tran format=raw file="{files.rawfile_path}" V(out)\n.end\n')
    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    a = backend.run(files.netlist_path, files.directory, run_files=files)
    assert a.native.vectors[2]._data is None
    with pytest.raises(SimulationError, match='launch'):
        backend.run(files.netlist_path, files.directory, run_files=files)
    assert not Path(files.rawfile_path).exists()
    assert_array_equal(a.native.vectors[2].data, [13, 13])


def test_backend_external_loading_is_lazy(tmp_path):
    path = tmp_path / 'external.raw'
    path.write_bytes(raw_bytes(4))
    raw = SimulatorBackend.get('ngspice').load_raw(str(path))
    assert all(v._data is None for v in raw.native.vectors)
    assert_array_equal(raw.native.vectors[1].data, [4, 4])


@pytest.mark.parametrize('unsafe', ['rawfile_path', 'temperature_path'])
def test_all_targets_preflight_preserves_old_public(tmp_path, unsafe):
    from silk.simulation.generation import begin
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True, temperature=True)
    Path(files.netlist_path).write_text('* deck\n.end\n')
    Path(files.rawfile_path).write_bytes(raw_bytes(5))
    target = Path(getattr(files, unsafe))
    if target.exists():
        target.unlink()
    target.mkdir()
    with pytest.raises(SimulationError, match='safely clear'):
        begin(files)
    if unsafe != 'rawfile_path':
        assert Path(files.rawfile_path).read_bytes() == raw_bytes(5)
    assert not list(tmp_path.glob('.silk-generation-*'))


def test_intervening_public_file_is_not_overwritten(tmp_path):
    from silk.simulation.generation import begin, offer, attach, accept
    from silk.simulation.backend import RawResult
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(7))
    raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    Path(files.rawfile_path).write_bytes(b'intervening writer')
    with pytest.raises(SimulationError, match='intervening'):
        accept(raw)
    assert Path(files.rawfile_path).read_bytes() == b'intervening writer'
    assert not gen.accepted


def test_worker_result_retains_generation_without_transporting_arrays(tmp_path):
    import pickle
    from silk.circuit.modules import Module
    from silk.circuit.module_test import _RunSimResult
    from silk.simulation.generation import begin, offer, attach
    from silk.simulation.backend import RawResult
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(9))
    raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    raw.native.vectors[0].data
    from silk.pdk import load_pdk
    mod = Module('sky130', str(tmp_path), str(tmp_path / 'deck.cir'), load_pdk('sky130'))
    mod._raw, mod.simResultsAll = raw, raw.native
    result = _RunSimResult(mod, 'tt', None, ())
    restored = pickle.loads(pickle.dumps(result))
    assert restored.netlist._raw is not None
    assert restored.netlist._raw._generation is gen
    assert restored.netlist.simResultsAll is None
    assert restored.netlist._raw.native is None
    from silk.simulation.generation import accept
    accept(restored.netlist._raw)
    assert all(v._data is None for v in restored.netlist._raw.native.vectors)
    assert_array_equal(restored.netlist._raw.native.vectors[2].data, [19, 19])


def test_object_view_uses_retained_generation_after_replacement(tmp_path):
    from silk import Module
    from silk.circuit.view_nodes import _capture_addresses, _view_node_context
    from silk.simulation.generation import begin, offer, attach, accept
    from silk.simulation.backend import RawResult
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(11))
    raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    accept(raw)
    mod = Module(None, str(tmp_path), 'deck.cir', None)
    mod.add_net('out')
    mod.add_net('unread')
    mod._raw, mod.results_file = raw, files.rawfile_path
    raw._view_addresses = _capture_addresses(mod)
    begin(files)  # B fails before writing anything; A must remain viewable.
    context = _view_node_context(mod)
    assert context is not None
    assert_array_equal(context.source.vectors[2].data, [21, 21])
    assert Path(mod.view(open=False, inline=True)).is_file()


def test_corner_reader_uses_held_generation_with_fresh_cache(tmp_path):
    from silk.simulation.generation import begin, offer, attach, accept
    from silk.simulation.backend import RawResult
    from silk.circuit.module_test import _CornerRawfile
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(12))
    raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    accept(raw)
    held = _CornerRawfile('tt', files.rawfile_path, SimulatorBackend.get('ngspice'), generation=gen)
    begin(files)
    assert_array_equal(held._raw_result().native.vectors[2].data, [22, 22])
    assert raw.native.vectors[2]._data is None


def test_direct_runner_retains_unread_generation(tmp_path, monkeypatch):
    from silk.simulation.runner import NgspiceRunner
    deck = tmp_path / 'direct.cir'
    deck.write_text('* deck\n.end\n')
    calls = iter([15, None])
    def launch(argv, **kwargs):
        value = next(calls)
        if value is None:
            raise OSError('launch failed')
        Path(argv[argv.index('-r') + 1]).write_bytes(raw_bytes(value))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    _patch_simulator_launch(monkeypatch, launch)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    runner = NgspiceRunner(str(deck), str(tmp_path))
    runner.run()
    a = runner.load()
    with pytest.raises(SimulationError):
        runner.run()
    assert_array_equal(a.vectors[2].data, [25, 25])


def _backend_worker(run_dir, files, simulator):
    from silk.circuit.module_test import _RunSimResult
    raw = SimulatorBackend.get(simulator).run(files.netlist_path, files.directory, run_files=files)
    netlist = SimpleNamespace(_raw=raw, simResultsAll=raw.native)
    return _RunSimResult(netlist, 'tt', None, ())


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_worker_offers_until_parent_accepts(tmp_path, monkeypatch, simulator):
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import accept
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=simulator == 'ngspice')
    Path(files.netlist_path).write_text(f'* deck\n.print tran format=raw file="{files.rawfile_path}" V(out)\n.end\n')
    def launch(argv, **kwargs):
        target = (argv[argv.index('-r') + 1] if simulator == 'ngspice' else
                  Path(argv[1]).read_text().split('file="')[1].split('"')[0])
        Path(target).write_bytes(raw_bytes(17))
        return SimpleNamespace(returncode=0, stdout='End of Xyce', stderr='')
    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    monkeypatch.setattr('silk.simulation._ngspice_compat.warn_version_advice', lambda *a: None)
    result, = run_tasks([RunTask(_backend_worker, str(tmp_path), args=(files, simulator), simulator=simulator)], n_workers=2)
    assert result.error is None, result.error
    raw = result.value.netlist._raw
    assert not raw._generation.accepted
    assert not Path(files.rawfile_path).exists()
    assert raw.native is None
    accept(raw)
    assert raw._generation.accepted
    assert_array_equal(read_rawfile(files.rawfile_path).vectors[2].data, [27, 27])
    assert_array_equal(raw.native.vectors[2].data, [27, 27])


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('deny_log_move', [False, True])
def test_parent_accept_refusal_publishes_launched_offer_log(
        tmp_path, monkeypatch, workers, simulator, deny_log_move):
    from silk.simulation import generation
    from silk.simulation.executor import RunTask, run_tasks
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text(
        f'* deck\n.print tran format=raw file="{files.rawfile_path}" V(out)\n.end\n')

    def launch(argv, **kwargs):
        if simulator == 'ngspice':
            target = argv[argv.index('-r') + 1]
            Path(argv[argv.index('-o') + 1]).write_text('launched attempt log')
        else:
            target = Path(argv[1]).read_text().split('file="')[1].split('"')[0]
            kwargs['stdout'].write('launched attempt log\nEnd of Xyce\n')
        Path(target).write_bytes(raw_bytes(23))
        return SimpleNamespace(returncode=0, stdout='End of Xyce', stderr='')

    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    monkeypatch.setattr('silk.simulation._ngspice_compat.warn_version_advice', lambda *a: None)
    accept = generation.accept
    # Delay serial acceptance too, so both paths exercise the same parent seam.
    monkeypatch.setattr(generation, 'accept', lambda raw: raw)
    outcome, = run_tasks([RunTask(_backend_worker, str(tmp_path), args=(files, simulator),
                                  simulator=simulator)], n_workers=workers)
    assert outcome.error is None, outcome.error
    raw = outcome.value.netlist._raw
    assert raw._generation.launched
    assert not Path(files.logfile_path).exists()
    Path(files.rawfile_path).write_text('intervening raw output')
    if deny_log_move:
        original_replace = generation.os.replace

        def replace(source, target):
            if source == raw._generation.staged.logfile_path and target == files.logfile_path:
                raise PermissionError('log move denied')
            return original_replace(source, target)

        monkeypatch.setattr(generation.os, 'replace', replace)
    with pytest.raises(SimulationError, match='intervening output') as error:
        accept(raw)
    if deny_log_move:
        assert error.value.log_path == raw._generation.staged.logfile_path
        assert 'log move denied' in str(error.value)
        assert not Path(files.logfile_path).exists()
        assert not getattr(raw._generation, 'cleanup_unavailable', False)
    else:
        assert error.value.log_path == files.logfile_path
    assert Path(files.rawfile_path).read_text() == 'intervening raw output'
    assert 'launched attempt log' in Path(error.value.log_path).read_text()
    assert list(tmp_path.glob('.silk-generation-*/rawfile.raw'))
    assert not (tmp_path / 'noise.csv').exists()
    assert not (tmp_path / 'xyce_temperature.csv').exists()
    assert raw._generation.ident not in generation._pending


@pytest.mark.parametrize('mode', ['delayed', 'in-progress', 'stopped', 'bad-key',
                                 'idle', 'late-dispatch', 'partial-frame', 'stalled-auth',
                                 'during-shutdown', 'slow-request', 'slow-auth'])
def test_coordinator_survives_expired_clients_and_shutdown_is_bounded(mode):
    import os
    import sys
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name('_generation_coordinator_driver.py')), mode],
        env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_real_generation_request_payloads_stay_small(tmp_path, monkeypatch, simulator):
    import pickle
    from multiprocessing.reduction import ForkingPickler
    from silk.simulation import generation
    from tests.test_run_files import _module

    module = _module(tmp_path, simulator)
    gen = None
    requests = {}
    invoke = generation._invoke

    def record(method, *args):
        if method in ('begin', 'offer'):
            wire = ForkingPickler.dumps((method, args))
            decoded_method, decoded_args = pickle.loads(wire)
            assert (decoded_method, decoded_args) == (method, args)
            requests[method] = len(wire)
            # Actual paths/completion/manifest metadata, with substantial room
            # below the receiver's 1 MiB cap; no waveform/deck payload belongs here.
            assert 0 < len(wire) < 16 * 1024
        return invoke(method, *args)

    try:
        module.generate_netlist()
        files = module._run_files
        assert files.completion is not None
        assert files.completion.requests
        monkeypatch.setattr(generation, '_invoke', record)
        with generation.attempt_scope('wire-size-control'):
            gen = generation.begin(files)
            gen.launched = True
            Path(gen.staged.rawfile_path).write_bytes(raw_bytes(5))
            Path(gen.staged.logfile_path).write_text('actual offered diagnostic')
            offered = generation.offer(gen)
        assert offered.manifest['rawfile_path'] is not None
        assert offered.manifest['logfile_path'] is not None
        assert set(requests) == {'begin', 'offer'}
        print(f'{simulator} coordinator request bytes: {requests}')
    finally:
        generation.abandon(gen)
        module.close()


def _control_worker(run_dir, files, simulator):
    from silk.circuit.module_test import _RunOutcome
    from silk.circuit.modules_sim import _track_result_replacement, _begin_result_replacement
    from silk.simulation.generation import replacement_callback
    owner = SimpleNamespace()
    with _track_result_replacement(owner) as started:
        try:
            with replacement_callback(lambda: _begin_result_replacement(owner)):
                result = _backend_worker(run_dir, files, simulator)
        except Exception as exc:
            return _RunOutcome(error=exc, replacement_started=started[0])
        return _RunOutcome(result=result, replacement_started=started[0])


def _failure_log_worker(run_dir, files, simulator, direct):
    try:
        if direct:
            from silk.simulation.runner import NgspiceRunner
            from silk.simulation.xyce_runner import XyceRunner
            cls = NgspiceRunner if simulator == 'ngspice' else XyceRunner
            runner = cls(files.netlist_path, files.directory, run_files=files)
            runner.run()
            runner.load()
        else:
            SimulatorBackend.get(simulator).run(
                files.netlist_path, files.directory, run_files=files)
    except BaseException as exc:
        return type(exc).__name__, getattr(exc, 'log_path', None), str(exc)
    raise AssertionError('Expected the attempt to fail')


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('direct', [False, True], ids=['backend', 'runner'])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('failure,expected', [
    ('timeout', 'SimulationError'), ('nonzero', 'SimulationError'),
    ('convergence', 'ConvergenceError'), ('abort', 'SimulationAborted'),
    ('empty', 'SimulationAborted'), ('cancel', 'KeyboardInterrupt'),
    ('sigterm', 'SystemExit'),
])
def test_failed_attempt_publishes_only_its_log(
        tmp_path, monkeypatch, workers, direct, simulator, failure, expected):
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import begin, abandon
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path),
                             log=True, noise=True, temperature=True)
    Path(files.netlist_path).write_text(
        f'* deck\n.print tran format=raw file="{files.rawfile_path}" V(out)\n.end\n')
    for kind in ('rawfile_path', 'noise_path', 'temperature_path', 'logfile_path'):
        Path(getattr(files, kind)).write_text('previous attempt')
    diagnostic = ('numerically singular matrix\n' if failure == 'convergence'
                  else f'attempt diagnostic: {failure}\n')

    def launch(argv, **kwargs):
        if simulator == 'ngspice':
            log = Path(argv[argv.index('-o') + 1])
            raw = Path(argv[argv.index('-r') + 1])
            log.write_text(diagnostic)
        else:
            stream = kwargs['stdout']
            assert Path(stream.name).parent.name.startswith('.silk-generation-')
            assert not Path(files.logfile_path).exists()
            stream.write(diagnostic)
            stream.flush()
            raw = Path(Path(argv[1]).read_text().split('file="')[1].split('"')[0])
        (raw.parent / 'noise.csv').write_text('partial noise')
        (raw.parent / 'xyce_temperature.csv').write_text('partial temperature')
        if failure != 'abort':
            raw.write_bytes(raw_bytes(3))
        if failure == 'empty':
            raw.write_text('Title: empty\nPlotname: Transient Analysis\nFlags: real\n'
                           'No. Variables: 1\nNo. Points: 0\nVariables:\n'
                           '0 time time\nValues:\n')
            # Xyce validates noise separately before checking rawfile rows.
            (raw.parent / 'noise.csv').write_text('FREQ,ONOISE,INOISE\n1,2,3\n')
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(argv, 1)
        if failure == 'cancel':
            raise KeyboardInterrupt()
        if failure == 'sigterm':
            # This matrix catches SystemExit deliberately; real worker exit and
            # its watchdog are exercised by the isolated process-kill tests.
            monkeypatch.setattr('silk.simulation.executor._arm_worker_exit_watchdog', lambda: None)
            from silk.simulation.executor import _worker_sigterm_handler
            _worker_sigterm_handler(15, None)
        return SimpleNamespace(returncode=1 if failure in ('nonzero', 'convergence') else 0,
                               stdout='End of Xyce\n', stderr='')

    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    monkeypatch.setattr('silk.simulation._ngspice_compat.warn_version_advice', lambda *a: None)
    outcome, = run_tasks([RunTask(_failure_log_worker, str(tmp_path),
                                  args=(files, simulator, direct), simulator=simulator)],
                         n_workers=workers)
    assert outcome.error is None, outcome.error
    error_type, log_path, message = outcome.value
    assert error_type == expected
    if failure not in ('cancel', 'sigterm'):
        assert log_path == files.logfile_path
        assert files.logfile_path in message
    assert diagnostic in Path(files.logfile_path).read_text()
    assert 'previous attempt' not in Path(files.logfile_path).read_text()
    for kind in ('rawfile_path', 'noise_path', 'temperature_path'):
        assert not Path(getattr(files, kind)).exists()
    assert list(tmp_path.glob('.silk-*.cir'))
    assert list(tmp_path.glob('.silk-generation-*'))
    # Refusal keeps the published diagnostic; the next begun attempt clears it.
    Path(files.rawfile_path).mkdir()
    with pytest.raises(SimulationError) as refusal:
        begin(files)
    assert refusal.value.log_path is None
    assert diagnostic in Path(files.logfile_path).read_text()
    Path(files.rawfile_path).rmdir()
    abandon(begin(files))
    assert not Path(files.logfile_path).exists()


def test_simulation_error_optional_log_path():
    import pickle
    assert SimulationError('before launch').log_path is None
    error = SimulationError('failed', log_path='output.log')
    assert pickle.loads(pickle.dumps(error)).log_path == 'output.log'


def test_failure_rpc_timeout_preserves_private_log_and_original_error(tmp_path, monkeypatch):
    from silk.simulation import generation
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = generation.begin(files)
    gen.launched = True
    Path(gen.staged.logfile_path).write_text('failed attempt evidence')
    error = SimulationError('original failure')
    requests = []

    def unavailable(address, key, method, args):
        requests.append(method)
        raise TimeoutError('silent coordinator')

    try:
        with monkeypatch.context() as patch:
            patch.setattr(generation, '_owner', -1)
            patch.setattr(generation, '_endpoint', (('127.0.0.1', 1), b'key'))
            patch.setattr(generation, '_failure_rpc', unavailable)
            generation.publish_failure_log(gen, error)
            generation.publish_failure_log(gen, error)
            generation.abandon(gen)
        assert requests == ['publish_failure_log']
        assert error.log_path == gen.staged.logfile_path
        assert 'original failure' in str(error)
        assert error.log_path in str(error)
        assert Path(error.log_path).read_text() == 'failed attempt evidence'
        assert not Path(files.logfile_path).exists()
    finally:
        gen.cleanup_unavailable = False
        generation.abandon(gen)


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('direct', [False, True], ids=['backend', 'runner'])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('cancel', [False, True], ids=['failure', 'cancel'])
@pytest.mark.parametrize('problem', ['intervening', 'unsafe', 'missing'])
def test_log_publication_problem_keeps_original_failure_and_abandons(
        tmp_path, monkeypatch, workers, direct, simulator, cancel, problem):
    from silk.simulation import generation
    from silk.simulation.executor import RunTask, run_tasks
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text('* deck\n.end\n')
    abandoned = tmp_path / 'abandoned'
    original_abandon = generation._abandon

    def record_abandon(ident):
        with abandoned.open('a') as stream:
            stream.write(ident + '\n')
        original_abandon(ident)

    def launch(argv, **kwargs):
        if simulator == 'ngspice':
            staged = Path(argv[argv.index('-o') + 1])
            staged.write_text('this attempt failed')
        else:
            stream = kwargs['stdout']
            stream.write('this attempt failed')
            stream.flush()
            staged = Path(stream.name)
        if problem == 'intervening':
            Path(files.logfile_path).write_text('other writer')
        elif problem == 'unsafe':
            Path(files.logfile_path).mkdir()
        else:
            staged.unlink()
        if cancel:
            raise KeyboardInterrupt('original cancellation')
        raise SimulationError('original simulation failure')

    monkeypatch.setattr(generation, '_abandon', record_abandon)
    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    monkeypatch.setattr('silk.simulation._ngspice_compat.warn_version_advice', lambda *a: None)
    outcome, = run_tasks([RunTask(_failure_log_worker, str(tmp_path),
                                  args=(files, simulator, direct), simulator=simulator)],
                         n_workers=workers)
    assert outcome.error is None, outcome.error
    error_type, log_path, message = outcome.value
    assert error_type == ('KeyboardInterrupt' if cancel else 'SimulationError')
    if cancel:
        assert message == 'original cancellation'
    else:
        assert 'original simulation failure' in message
        if problem == 'missing':
            assert log_path is None
            assert 'no diagnostic log was written' in message
        else:
            assert 'log publication refused:' in message
            assert f'log retained at {log_path}' in message
            assert Path(log_path).read_text() == 'this attempt failed'
            assert Path(log_path).parent.name.startswith('.silk-generation-')
    assert not Path(files.rawfile_path).exists()
    if problem == 'intervening':
        assert Path(files.logfile_path).read_text() == 'other writer'
    elif problem == 'unsafe':
        assert Path(files.logfile_path).is_dir()
    else:
        assert not Path(files.logfile_path).exists()
        assert not list(tmp_path.glob('.silk-generation-*/output.log'))
    assert len(abandoned.read_text().splitlines()) == 1
    assert all(ident not in generation._pending for ident in abandoned.read_text().splitlines())


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('replacement', ['unsafe', 'failure', 'success'])
def test_replacement_controls_both_execution_paths(tmp_path, monkeypatch, workers, simulator, replacement):
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import accept
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True, temperature=True)
    Path(files.netlist_path).write_text(f'* deck\n.print tran format=raw file="{files.rawfile_path}" V(out)\n.end\n')
    control = tmp_path / 'control'
    launched = tmp_path / 'launches'
    control.write_text('A')
    def launch(argv, **kwargs):
        with launched.open('a') as stream:
            stream.write('launch\n')
        mode = control.read_text()
        if mode == 'failure':
            raise OSError('injected launch failure')
        target = (argv[argv.index('-r') + 1] if simulator == 'ngspice' else
                  Path(argv[1]).read_text().split('file="')[1].split('"')[0])
        Path(target).write_bytes(raw_bytes(31 if mode == 'A' else 41))
        return SimpleNamespace(returncode=0, stdout='End of Xyce', stderr='')
    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    monkeypatch.setattr('silk.simulation._ngspice_compat.warn_version_advice', lambda *a: None)
    a = SimulatorBackend.get(simulator).run(files.netlist_path, files.directory, run_files=files)
    assert a.native.vectors[2]._data is None
    control.write_text(replacement)
    if replacement == 'unsafe':
        Path(files.temperature_path).mkdir()  # Last cleanup target, after raw/log.
    outcome, = run_tasks([RunTask(_control_worker, str(tmp_path), args=(files, simulator), simulator=simulator)], n_workers=workers)
    assert outcome.error is None, outcome.error
    attempt = outcome.value
    assert attempt.replacement_started is (replacement != 'unsafe')
    if replacement == 'unsafe':
        assert attempt.error is not None
        assert launched.read_text().splitlines() == ['launch']
        assert_array_equal(read_rawfile(files.rawfile_path).vectors[1].data, [31, 31])
    elif replacement == 'failure':
        assert attempt.error is not None
        assert not Path(files.rawfile_path).exists()
    else:
        assert attempt.error is None
        accept(attempt.result.netlist._raw)
        assert_array_equal(read_rawfile(files.rawfile_path).vectors[1].data, [41, 41])
    assert_array_equal(a.native.vectors[2].data, [41, 41])


def test_replacement_through_parent_symlink_preserves_old_reader(tmp_path):
    from silk.simulation.generation import begin, offer, attach, accept
    from silk.simulation.backend import RawResult
    actual = tmp_path / 'actual'
    actual.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(actual, target_is_directory=True)
    files = RunFiles.resolve(str(actual / 'deck.cir'), str(actual))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(51))
    a = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    accept(a)
    begin(RunFiles.resolve(str(alias / 'deck.cir'), str(alias)))
    assert_array_equal(a.native.vectors[2].data, [61, 61])


def _missing_offer_worker(run_dir, files, simulator):
    result = _backend_worker(run_dir, files, simulator)
    Path(result.netlist._raw._generation.resolve('rawfile_path')).unlink()
    return result


def test_missing_offer_survives_worker_unpickle_then_refuses_parent_accept(tmp_path, monkeypatch):
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import accept
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text('* deck\n.end\n')
    def launch(argv, **kwargs):
        Path(argv[argv.index('-r') + 1]).write_bytes(raw_bytes(18))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    _patch_simulator_launch(monkeypatch, launch)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    result, = run_tasks([RunTask(_missing_offer_worker, str(tmp_path), args=(files, 'ngspice'))], n_workers=2)
    assert result.error is None, result.error
    assert result.value.netlist._raw.native is None
    with pytest.raises(SimulationError, match='unavailable retained artifact'):
        accept(result.value.netlist._raw)
    assert not Path(files.rawfile_path).exists()


def test_lazy_completion_rejects_short_grid_even_when_endpoint_matches(tmp_path, monkeypatch):
    from dataclasses import replace
    from hashlib import sha256
    from silk.exceptions import SimulationAborted
    from silk.simulation.completion import CompletionPlan, CompletionRequest
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    deck = '* certified\n.ac lin 4 1 4\n.end\n'
    Path(files.netlist_path).write_text(deck)
    files = replace(files, completion=CompletionPlan('ngspice', sha256(deck.encode()).digest(),
                    (CompletionRequest('ac', 1, 4, 4, 4, 1, 'lin'),)))
    data = (b'Title: truncated\nPlotname: AC Analysis\nFlags: real\n'
            b'No. Variables: 2\nNo. Points: 2\nVariables:\n'
            b'0 frequency frequency\n1 v(out) voltage\nBinary:\n'
            + np.array([[1, 2], [4, 2]], dtype='<f8').tobytes())
    def launch(argv, **kwargs):
        Path(argv[argv.index('-r') + 1]).write_bytes(data)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    _patch_simulator_launch(monkeypatch, launch)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    with pytest.raises(SimulationAborted, match='point'):
        SimulatorBackend.get('ngspice').run(files.netlist_path, files.directory, run_files=files)
    assert not Path(files.rawfile_path).exists()


def test_relative_generation_paths_survive_cwd_and_release(tmp_path, monkeypatch):
    import gc
    from silk.simulation.generation import begin, offer, attach, accept
    from silk.simulation.backend import RawResult
    monkeypatch.chdir(tmp_path)
    Path('relative').mkdir()
    files = RunFiles.resolve('relative/deck.cir', 'relative')
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(21))
    raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    accept(raw)
    private = Path(gen.directory)
    monkeypatch.chdir(tmp_path.parent)
    assert raw.source_path == 'relative/rawfile.raw'
    assert_array_equal(raw.native.vectors[2].data, [31, 31])
    del raw, gen
    gc.collect()
    assert not private.exists()


def test_held_direct_runner_reload_reads_its_own_schema(tmp_path, monkeypatch):
    from silk.simulation.runner import NgspiceRunner
    deck = tmp_path / 'direct.cir'
    deck.write_text('* deck\n.end\n')
    calls = iter([raw_bytes(23), raw_bytes(24).replace(b'v(unread)', b'v(otherx)')])
    def launch(argv, **kwargs):
        Path(argv[argv.index('-r') + 1]).write_bytes(next(calls))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    _patch_simulator_launch(monkeypatch, launch)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    a = NgspiceRunner(str(deck), str(tmp_path))
    a.run()
    a.load()
    b = NgspiceRunner(str(deck), str(tmp_path))
    b.run()
    b.load()
    held = a.load()
    assert held.vectors[2].name == 'v(unread)'
    assert_array_equal(held.vectors[2].data, [33, 33])


def _cancelled_worker(run_dir, files, stage):
    import threading
    from silk.simulation.generation import begin, offer
    gen = begin(files)
    if stage == 'offer':
        Path(gen.staged.rawfile_path).write_bytes(raw_bytes(91))
        offer(gen)
    Path(run_dir, 'cancel-ready').write_text(gen.ident)
    threading.Event().wait(30)


@pytest.mark.parametrize('stage', ['begin', 'offer'])
def test_cancelled_worker_clears_parent_current_but_keeps_held_reader(tmp_path, monkeypatch, stage):
    import multiprocessing.pool
    import time
    from silk import Module
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import begin, offer, attach, accept, register_owner
    from silk.simulation.backend import RawResult
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(81))
    held = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    accept(held)
    owner = Module(None, str(tmp_path), 'deck.cir', None)
    owner._raw, owner.results_file = held, files.rawfile_path
    register_owner(owner, held)
    def interrupt_when_ready(result, timeout=None):
        until = time.monotonic() + 5
        while not (tmp_path / 'cancel-ready').exists():
            if time.monotonic() > until:
                raise AssertionError('worker did not reach cancellation boundary')
            time.sleep(.01)
        raise KeyboardInterrupt
    monkeypatch.setattr(multiprocessing.pool.ApplyResult, 'get', interrupt_when_ready)
    with pytest.raises(KeyboardInterrupt):
        run_tasks([RunTask(_cancelled_worker, str(tmp_path), args=(files, stage))], n_workers=2)
    assert owner._raw is None
    assert owner.results_file is None
    assert not Path(files.rawfile_path).exists()
    assert_array_equal(held.native.vectors[2].data, [91, 91])


def test_relative_output_cards_are_redirected_to_staging(tmp_path, monkeypatch):
    from silk.simulation.generation import begin
    monkeypatch.chdir(tmp_path)
    Path('relative').mkdir()
    files = RunFiles.resolve('relative/deck.cir', 'relative')
    Path(files.netlist_path).write_text('* deck\n.control\nwrite relative/rawfile.raw\n.endc\n.end\n')
    gen = begin(files)
    text = Path(gen.staged.netlist_path).read_text()
    assert f'write {gen.staged.rawfile_path}\n' in text
    assert 'write relative/rawfile.raw' not in text


def _independent_generation(run_dir, transfer):
    import gc
    import pickle
    from silk.simulation.generation import begin, offer, attach, accept
    from silk.simulation.backend import RawResult
    from silk.circuit.module_test import _RunSimResult
    files = RunFiles.resolve(str(Path(run_dir, 'deck.cir')), run_dir)
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(71))
    raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path), offer(gen))
    accept(raw)
    if transfer:
        return _RunSimResult(SimpleNamespace(_raw=raw, simResultsAll=raw.native), 'tt', None, ())
    duplicate = pickle.loads(pickle.dumps(gen))
    assert duplicate is gen
    directory, deck = gen.directory, gen.staged.netlist_path
    del duplicate, gen, raw
    gc.collect()
    return not Path(directory).exists(), not Path(deck).exists()


def test_independent_child_generic_pickle_keeps_cleanup_authority(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as pool:
        assert pool.submit(_independent_generation, str(tmp_path), False).result(timeout=10) == (True, True)


def test_explicit_independent_child_result_handoff_survives_sender_exit(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from silk.simulation.generation import accept, begin
    with ProcessPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_independent_generation, str(tmp_path), True).result(timeout=10)
    raw = result.netlist._raw
    assert raw.native is None
    accept(raw)
    assert Path(raw._generation.directory).is_dir()
    begin(raw.run_files)
    assert_array_equal(raw.native.vectors[2].data, [81, 81])


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('managed', [False, True])
def test_missing_managed_draw_companion_never_borrows_public_file(tmp_path, simulator, managed):
    from silk.circuit.module_test import _read_sample_draws, _attach_xyce_draws
    from silk.exceptions import MonteCarloError
    from silk.simulation.backend import RawResult
    from silk.simulation.generation import begin, offer, attach, accept

    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text('* deck\n.end\n')
    raw = None
    if managed:
        gen = begin(files)
        Path(gen.staged.rawfile_path).write_bytes(raw_bytes(83))
        raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True),
                               files.rawfile_path), offer(gen))
        accept(raw)
        kind = 'logfile_path' if simulator == 'ngspice' else 'draws_path'
        assert gen.manifest[kind] is None

    # This file arrived after the managed generation was closed. It is valid
    # for an external caller, but cannot fill a hole in a managed manifest.
    if simulator == 'ngspice':
        Path(files.logfile_path).write_text('unrelated later log\n')
    else:
        Path(files.directory, Path(files.netlist_path).name + '.res').write_text('STEP\n1\n')
    netlist = SimpleNamespace(simulator=simulator, elem_dict={},
                              pdk_details=SimpleNamespace(statistical=SimpleNamespace(record={})),
                              _run_files=files, _raw=raw)
    report = SimpleNamespace()

    def read_draws():
        if simulator == 'ngspice':
            return _read_sample_draws(netlist)
        _attach_xyce_draws(report, netlist, SimpleNamespace(record_draws=True))
        return report._mc_draws

    if managed:
        error = SimulationError if simulator == 'ngspice' else MonteCarloError
        with pytest.raises(error, match='record_draws:'):
            read_draws()
        assert not hasattr(report, '_mc_draws')
    else:
        assert read_draws() == {}


def _lost_begin_reply_worker(run_dir, spec):
    from silk.circuit.module_test import _simulate_run_outcome
    from silk.simulation import generation

    invoke = generation._invoke

    def lose_reply(method, *args):
        result = invoke(method, *args)
        if method == 'begin':
            raise EOFError('injected loss after coordinator completed begin')
        return result

    generation._invoke = lose_reply
    try:
        return _simulate_run_outcome(spec, simulate=lambda prepared: generation.begin(prepared.files))
    finally:
        generation._invoke = invoke


def test_parent_reconciles_replacement_when_worker_loses_begin_reply(tmp_path):
    from silk.circuit.module_test import _planned_outcome
    from silk.simulation.backend import RawResult
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import begin, offer, attach, accept

    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(87))
    held = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True),
                            files.rawfile_path), offer(gen))
    accept(held)
    prepared = SimpleNamespace(attempt_id='lost-begin-reply', attempt_context={},
                               netlist=SimpleNamespace(), files=files)
    outer, = run_tasks([RunTask(_lost_begin_reply_worker, str(tmp_path), args=(prepared,))],
                      n_workers=2)
    assert outer.error is None
    assert isinstance(outer.value.error, EOFError)
    assert outer.value.replacement_started is False  # The worker missed the callback.
    reconciled = _planned_outcome(outer, prepared)
    assert reconciled.replacement_started is True  # The coordinator knows it happened.
    assert not Path(files.rawfile_path).exists()
    assert_array_equal(held.native.vectors[2].data, [97, 97])


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('refuse', [False, True])
def test_direct_runner_repeat_before_load_reconciles_pending_metadata(tmp_path, monkeypatch, simulator, refuse):
    from silk.simulation import generation
    from silk.simulation.runner import NgspiceRunner
    from silk.simulation.xyce_runner import XyceRunner

    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text(
        f'* deck\n.print tran format=raw file="{files.rawfile_path}" V(out)\n.end\n')

    def launch(argv, **kwargs):
        target = (argv[argv.index('-r') + 1] if simulator == 'ngspice' else
                  Path(argv[1]).read_text().split('file="')[1].split('"')[0])
        Path(target).write_bytes(raw_bytes(89))
        return SimpleNamespace(returncode=0, stdout='End of Xyce', stderr='')

    _patch_simulator_launch(monkeypatch, launch, simulator)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    runner_type = NgspiceRunner if simulator == 'ngspice' else XyceRunner
    runner = runner_type(files.netlist_path, files.directory, run_files=files)
    runner.run()
    first = runner._generation
    assert first.ident in generation._pending
    if refuse:
        # A is still staged. A refused B must leave A available to load.
        Path(files.rawfile_path).mkdir()
        with pytest.raises(SimulationError, match='safely clear'):
            runner.run()
        assert runner._generation is first
        assert first.ident in generation._pending
        Path(files.rawfile_path).rmdir()
        loaded = runner.load()
        assert_array_equal(loaded.vectors[2].data, [99, 99])
        assert first.accepted
        assert first.ident not in generation._pending
        return
    runner.run()
    assert runner._generation.ident != first.ident
    assert first.ident not in generation._pending
    assert Path(first.staged.rawfile_path).is_file()  # Abandon metadata, retain disk evidence.
    runner.load()
    assert runner._generation.ident not in generation._pending


def test_coordinator_queues_worker_connections_before_serving():
    import socket
    from silk.simulation import generation

    # All pool workers can reach begin together while the service is still
    # handling another request. Queue the burst before allowing any accept,
    # so insufficient capacity fails with a bounded socket timeout.
    with generation.worker_coordinator():
        connections = []
        try:
            address, _key = generation._endpoint
            for _ in range(8):
                connections.append(socket.create_connection(address, timeout=0.2))
        finally:
            for connection in connections:
                connection.close()


def _concurrent_offer_worker(run_dir, files, rendezvous, value):
    import os
    import time
    from silk.simulation.generation import begin, offer

    Path(rendezvous, str(os.getpid())).touch()
    deadline = time.monotonic() + 5
    while len(list(Path(rendezvous).iterdir())) < 8:
        if time.monotonic() > deadline:
            raise TimeoutError('workers did not reach the generation rendezvous')
        time.sleep(0.01)
    gen = begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(value))
    return offer(gen)


def test_coordinator_accepts_simultaneous_worker_generation_offers(tmp_path, monkeypatch):
    import multiprocessing
    import multiprocessing.pool
    from silk.simulation.backend import RawResult
    from silk.simulation.executor import RunTask, run_tasks
    from silk.simulation.generation import attach, accept

    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip('concurrent worker offers require fork workers')
    rendezvous = tmp_path / 'ready'
    rendezvous.mkdir()
    tasks = []
    for value in range(8):
        directory = tmp_path / str(value)
        directory.mkdir()
        files = RunFiles.resolve(str(directory / 'deck.cir'), str(directory))
        Path(files.netlist_path).write_text('* deck\n.end\n')
        tasks.append(RunTask(_concurrent_offer_worker, str(directory),
                             args=(files, str(rendezvous), value)))
    original_get = multiprocessing.pool.ApplyResult.get

    def bounded_get(result, timeout=None):
        return original_get(result, timeout=10)

    monkeypatch.setattr(multiprocessing.pool.ApplyResult, 'get', bounded_get)
    for value, outcome in enumerate(run_tasks(tasks, n_workers=8)):
        assert outcome.error is None, outcome.error
        gen = outcome.value
        assert not gen.accepted
        assert not Path(gen.public.rawfile_path).exists()
        raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True),
                               gen.public.rawfile_path), gen)
        accept(raw)
        assert_array_equal(raw.native.vectors[2].data, [value + 10] * 2)


def test_long_public_deck_name_supports_staging_and_retained_replacement(tmp_path):
    from silk.simulation.backend import RawResult
    from silk.simulation.generation import begin, offer, attach, accept

    deck = tmp_path / ('a' * 240 + '.cir')
    text = '* long public name\n.include "relative.inc"\n.end\n'
    deck.write_text(text)
    (tmp_path / 'relative.inc').write_text('* included file\n')
    files = RunFiles.resolve(str(deck), str(tmp_path))
    held = []
    for value in (93, 94):
        gen = begin(files)
        assert Path(gen.staged.netlist_path).parent == deck.parent
        assert Path(gen.staged.netlist_path).read_text() == text
        assert gen.public.netlist_path == str(deck)
        Path(gen.staged.rawfile_path).write_bytes(raw_bytes(value))
        raw = attach(RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True),
                               files.rawfile_path), offer(gen))
        accept(raw)
        held.append(raw)
    assert deck.read_text() == text
    assert_array_equal(held[0].native.vectors[2].data, [103, 103])
    assert_array_equal(held[1].native.vectors[2].data, [104, 104])


@pytest.mark.parametrize('workers', [1, 2])
def test_parent_accept_refusal_reaches_the_testbench_failure_record(
        tmp_path, monkeypatch, workers):
    """TASK-499 P2-R1: with the log retained privately, the failure's run record
    names that log (the exception's own path) and keeps the launch identity."""
    from silk.circuit import TestFailure
    from silk.circuit.module_test import ModuleTestbench
    from silk.simulation import generation
    from tests._env import _ngspice_available as ngspice_available
    from tests.circuit.test_selected_testbench import leaf
    if not ngspice_available():
        pytest.skip('needs ngspice')
    mod, tb = leaf(tmp_path)
    real_accept = generation.accept
    real_evaluate = ModuleTestbench._evaluate_run
    real_replace = generation.os.replace
    seen = {}

    def evaluate(self, result):
        seen['launch_id'] = result.run.launch_id
        return real_evaluate(self, result)

    def accept(raw):
        gen = getattr(raw, '_generation', None)
        if gen is not None and not gen.accepted:
            Path(gen.targets.rawfile_path).write_text('intervening raw output')
        return real_accept(raw)

    def replace(source, target):
        if (str(source).endswith('output.log') and '.silk-generation-' in str(source)
                and str(target).endswith('/output.log')
                and '.silk-generation-' not in str(target)):
            raise PermissionError('log move denied')
        return real_replace(source, target)

    monkeypatch.setattr(ModuleTestbench, '_evaluate_run', evaluate)
    monkeypatch.setattr(generation, 'accept', accept)
    monkeypatch.setattr(generation.os, 'replace', replace)
    try:
        failure = next(iter(mod.run_all_tests(n_workers=workers).values()))
        assert isinstance(failure, TestFailure)
        assert 'intervening output' in failure['error']
        retained = failure.exception.log_path
        public_log = str(Path(failure.run.directory) / 'output.log')
        assert '.silk-generation-' in retained and Path(retained).exists()
        assert failure.run is not None
        assert failure.run.log_path == retained
        assert failure.run.launch_id
        if workers == 2:                 # parent-side acceptance (worker dispatch)
            assert 'launch_id' in seen, 'the refusal must reach the parent evaluator'
            assert failure.run.launch_id == seen['launch_id']
        assert failure.run.deck_path and failure.run.rawfile_path
        assert not Path(public_log).exists()
        assert Path(retained).read_text().strip()
    finally:
        mod.close()


@pytest.mark.parametrize('workers', [1, 2])
def test_evaluation_error_keeps_the_successful_run_record(tmp_path, monkeypatch, workers):
    """The counterpart: an ordinary evaluation error keeps the launch's own record."""
    from silk.circuit.module_test import ModuleTestbench
    from tests._env import _ngspice_available as ngspice_available
    from tests.circuit.test_selected_testbench import leaf
    if not ngspice_available():
        pytest.skip('needs ngspice')
    mod, tb = leaf(tmp_path)
    seen = {}

    def evaluate(self, result):
        seen['run'] = result.run
        raise ValueError('evaluation broke')
    monkeypatch.setattr(ModuleTestbench, '_evaluate_run', evaluate)
    try:
        failure = next(iter(mod.run_all_tests(n_workers=workers).values()))
        assert failure.run == seen['run']
    finally:
        mod.close()
