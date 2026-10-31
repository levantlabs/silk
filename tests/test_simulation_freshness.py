"""Replacement attempts cannot reuse earlier files, waveforms or evidence."""

from dataclasses import replace
from pathlib import Path
import os
import pickle
import re
import subprocess
import warnings
from unittest.mock import Mock

import numpy as np
import pytest

from tests._mock_rawfile import write_complete_rawfile, complete_noise_csv

from silk.exceptions import CharacterizationContextError, SimulationAborted, SimulationError
from silk.faults import FaultSet
from silk.faults.runner import run_one
from silk.elements import elements
from silk.simulation import AC, Noise, Transient
from silk.simulation.backend import SimulatorBackend
from silk.simulation.run_files import RunFiles
from silk.simulation.runner import NgspiceRunner
from silk.simulation.xyce_runner import XyceRunner
from tests._env import needs_ngspice
from tests.test_run_files import _module, _NOISE_TEXT
from tests.test_testbench_output import _load


def _raw(value=1.8, *, points=2, plot='Transient Analysis', stop=1e-9):
    return (
        f'Title: freshness\nDate: d\nPlotname: {plot}\nFlags: real\n'
        f'No. Variables: 2\nNo. Points: {points}\n'
        'Variables:\n\t0\ttime\ttime\n\t1\tv(out)\tvoltage\nValues:\n'
        + ''.join(f'{i}\t{i * stop / max(points - 1, 1)}\n\t{value}\n' for i in range(points))
    )


def _launched_files(files, argv, kwargs):
    """A simulator writes command-line/print destinations, including staging."""
    if '-b' in argv:
        return replace(files, netlist_path=argv[-1],
                       rawfile_path=argv[argv.index('-r') + 1],
                       logfile_path=argv[argv.index('-o') + 1])
    deck = Path(argv[1]).read_text()
    outputs = dict(re.findall(r'format=(raw|csv) file="?([^\s"]+)', deck))
    return replace(files, netlist_path=argv[1],
                   logfile_path=kwargs['stdout'].name,
                   rawfile_path=outputs.get('raw') if files.rawfile_path else None,
                   noise_path=outputs.get('csv') if files.noise_path else None)


def _external_deck(files):
    """A minimal external input declaring the mock simulator's outputs."""
    return ('* external deck retained verbatim\n'
            + (f'.print tran format=raw file={files.rawfile_path} V(out)\n'
               if files.rawfile_path else '')
            + (f'.print noise format=csv file={files.noise_path} FREQ ONOISE INOISE\n'
               if files.noise_path else '') + '.end\n')


def _success(files, *, value=1.8):
    def launch(argv, **kwargs):
        for path in (files.rawfile_path, files.noise_path, files.logfile_path):
            if path:
                assert not Path(path).exists(), f'previous output survived: {path}'
        targets = _launched_files(files, argv, kwargs)
        deck_path = Path(targets.netlist_path)
        deck = deck_path.read_text() if deck_path.exists() else ''
        if targets.rawfile_path:
            write_complete_rawfile(targets.rawfile_path, deck,
                                   simulator='ngspice' if '-b' in argv else 'xyce', values={'v(out)': value})
        if targets.noise_path:
            Path(targets.noise_path).write_text(complete_noise_csv(deck))
        if targets.logfile_path:
            Path(targets.logfile_path).write_text('Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n')
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce\n', stderr='')
    return launch


def _failure(files, mode):
    def launch(argv, **kwargs):
        for path in (files.rawfile_path, files.noise_path, files.logfile_path):
            if path:
                assert not Path(path).exists()
        targets = _launched_files(files, argv, kwargs)
        if mode == 'timeout':
            Path(targets.rawfile_path).write_text(_raw(0.9))  # partial output remains diagnostic
            raise subprocess.TimeoutExpired(argv, 0.01)
        if mode == 'launch':
            raise PermissionError('cannot execute test simulator')
        if mode == 'malformed':
            Path(targets.rawfile_path).write_text('this is not a rawfile\n')
        if mode == 'empty':
            Path(targets.rawfile_path).write_text(_raw(points=0))
        if mode == 'constants':
            Path(targets.rawfile_path).write_text(_raw(plot='constants'))
        return subprocess.CompletedProcess(
            argv, 1 if mode == 'nonzero' else 0, stdout='End of Xyce\n', stderr='')
    return launch


def _assert_invalid(mod):
    assert mod._raw is mod.simResultsAll is None
    assert mod.results_file is mod.dataFile is None
    assert mod.characterization_context is None
    assert mod._last_sim_corner is None
    assert not mod.sim_results_loaded
    with pytest.raises(SimulationError):
        mod.load_signal('out')
    with pytest.raises(SimulationError):
        mod.read_raw_file()


@pytest.mark.parametrize('linked_parent', [False, True])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_only_concrete_outputs_are_removed_at_launch(tmp_path, monkeypatch, simulator,
                                                    linked_parent, mock_ngspice_version):
    if linked_parent:
        alias = tmp_path / 'alias'
        alias.symlink_to(tmp_path, target_is_directory=True)
        tmp_path = alias
    deck = tmp_path / 'user-deck.cir'
    deck.write_text('* explicit deck is never rewritten\n.end\n')
    files = RunFiles(str(tmp_path), str(deck), str(tmp_path / 'testrawfile123.rawfile'),
                     str(tmp_path / 'exact.log'),
                     str(tmp_path / 'exact.csv') if simulator == 'xyce' else None)
    if simulator == 'xyce':
        deck.write_text(_external_deck(files))
    for path in (files.rawfile_path, files.logfile_path, files.noise_path):
        if path:
            Path(path).write_text('old generated output')
    keep = [deck, *(tmp_path / name for name in (
        'notes.txt', 'evidence.db', 'rawfile.raw', 'unrelated.raw', 'output.log',
        'unrelated.log', 'noise.csv'))]
    for path in keep[1:]:
        path.write_text(f'keep {path.name}')
    sibling = tmp_path / 'sibling-run'
    sibling.mkdir()
    keep.append(sibling / 'rawfile.raw')
    keep[-1].write_text('sibling result')
    before = {p: p.read_bytes() for p in keep}
    launch = _success(files)

    def checked(argv, **kwargs):
        assert {p: p.read_bytes() for p in keep} == before
        return launch(argv, **kwargs)

    monkeypatch.setattr(subprocess, 'run', checked)
    raw = SimulatorBackend.get(simulator).run(str(deck), str(tmp_path), run_files=files)
    assert raw.native.npoints == 2
    assert {p: p.read_bytes() for p in keep} == before


@needs_ngspice
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_real_ngspice_first_run_and_rerun_through_symlinked_parent(tmp_path, mode):
    real = tmp_path / 'project'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    output = alias / 'run'
    mod = _module(output)
    mod.set_output_options(mode=mode)
    assert not any(output.iterdir())

    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 1)
    first = Path(mod.results_file)
    previous = first.read_bytes()
    assert first.is_relative_to(output)
    assert Path(mod._run_files.logfile_path).is_file()
    keep = output / 'notes.txt'
    keep.write_text('preserve user data')

    mod.sig_sources_dict['vin'].dc = 2
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 2)
    second = Path(mod.results_file)
    assert second.is_relative_to(output)
    assert Path(mod._run_files.logfile_path).is_file()
    assert keep.read_text() == 'preserve user data'
    if mode == 'overwrite':
        assert second == first
        assert first.read_bytes() != previous
    else:
        assert second != first
        assert second.parent.parent == first.parent.parent == output
        assert first.read_bytes() == previous


@pytest.mark.parametrize('linked_parent', [False, True])
@pytest.mark.parametrize('unsafe', ['directory', 'symlink', 'broken-symlink',
                                  'fifo', 'netlist', 'hardlink', 'log-directory'])
@pytest.mark.parametrize('runner_class', [NgspiceRunner, XyceRunner])
def test_unsafe_cleanup_refuses_launch(tmp_path, monkeypatch, unsafe, runner_class,
                                      linked_parent, mock_ngspice_version):
    output = tmp_path / 'outputs'
    output.mkdir()
    deck = output / 'deck.cir'
    deck.write_text('preserve netlist')
    if linked_parent:
        alias = tmp_path / 'alias'
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    victim = tmp_path / 'user-file'
    victim.write_text('preserve user data')
    raw = output / 'rawfile.raw'
    log = output / 'output.log'
    log.write_text('old diagnostic')
    if unsafe == 'directory':
        raw.mkdir()
        (raw / 'keep').write_text('keep directory contents')
    elif unsafe in ('symlink', 'broken-symlink'):
        raw.symlink_to(victim if unsafe == 'symlink' else tmp_path / 'absent')
    elif unsafe == 'fifo':
        if not hasattr(os, 'mkfifo'):
            pytest.skip('requires os.mkfifo')
        os.mkfifo(raw)
    elif unsafe == 'netlist':
        raw = output / deck.name
    elif unsafe == 'hardlink':
        os.link(deck, raw)
    elif unsafe == 'log-directory':
        raw.write_text('old result')
        log.unlink()
        log.mkdir()
    files = RunFiles(str(output), str(deck), str(raw), str(log))
    launch = Mock()
    monkeypatch.setattr(subprocess, 'run', launch)
    runner = runner_class(str(deck), str(tmp_path), run_files=files)
    with pytest.raises(SimulationError, match='Cannot safely clear.*Simulator was not launched'):
        runner.run()
    launch.assert_not_called()
    assert deck.read_text() == 'preserve netlist'
    assert victim.read_text() == 'preserve user data'
    assert raw.exists() or raw.is_symlink()
    assert log.exists()


@pytest.mark.parametrize('runner_class', [NgspiceRunner, XyceRunner])
def test_cleanup_permission_failure_is_typed_and_prevents_launch(tmp_path, monkeypatch, runner_class, mock_ngspice_version):
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    Path(files.netlist_path).write_text('* cleanup permission test\n.end\n')
    Path(files.rawfile_path).write_text('old result')
    launch = Mock()
    monkeypatch.setattr(subprocess, 'run', launch)
    unlink = os.unlink

    def denied(path, *args, **kwargs):
        if path == files.rawfile_path:
            raise PermissionError('injected permission failure')
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, 'unlink', denied)
    with pytest.raises(SimulationError, match='permissions.*Simulator was not launched'):
        runner_class(files.netlist_path, files.directory, run_files=files).run()
    launch.assert_not_called()
    assert Path(files.rawfile_path).read_text() == 'old result'


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('mode', ['missing', 'timeout', 'nonzero', 'launch',
                                 'malformed', 'empty', 'constants'])
def test_module_failure_invalidates_live_data_and_old_reload_paths(tmp_path, monkeypatch, mode, manual, mock_ngspice_version):
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    files = mod._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 1.8)
    # Include the legacy explicit-file fallback in the state being invalidated.
    older = tmp_path / 'older.raw'
    older.write_text(_raw(3.3))
    mod.dataFile = str(older)
    Path(files.logfile_path).write_text('Fatal error from the PREVIOUS run\n')
    monkeypatch.setattr(subprocess, 'run', _failure(files, mode))
    for _ in range(2):
        with pytest.raises(SimulationError) as error:
            if manual:
                mod.generate_netlist()
                mod.run_simulator()
            else:
                mod.simulate(corner='ff')
        if mode in ('missing', 'empty', 'constants'):
            assert isinstance(error.value, SimulationAborted)
        assert 'PREVIOUS' not in str(error.value)
        _assert_invalid(mod)
        assert older.read_text() == _raw(3.3)
    # The same object recovers normally and publishes fresh values.
    monkeypatch.setattr(subprocess, 'run', _success(files, value=0.9))
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 0.9)


@pytest.mark.parametrize('runner_class', [NgspiceRunner, XyceRunner])
@pytest.mark.parametrize('mode', ['missing', 'timeout', 'nonzero', 'launch',
                                 'malformed', 'empty', 'constants'])
def test_reused_runner_cannot_return_cached_or_partial_results(tmp_path, monkeypatch, runner_class, mode, mock_ngspice_version):
    files = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path),
                             log=runner_class is NgspiceRunner)
    Path(files.netlist_path).write_text(_external_deck(files))
    runner = runner_class(files.netlist_path, files.directory, run_files=files)
    monkeypatch.setattr(subprocess, 'run', _success(files))
    runner.run()
    assert runner.load().npoints == 2
    assert runner.results is not None
    monkeypatch.setattr(subprocess, 'run', _failure(files, mode))
    with pytest.raises(SimulationError):
        runner.run()
        runner.load()
    assert runner.results is None
    with pytest.raises(SimulationError, match='last attempt failed'):
        runner.load()


@pytest.mark.parametrize('invalid', ['corner', 'label', 'context', 'generation'])
def test_testbench_preflight_keeps_previous_files_results_and_context(tmp_path, monkeypatch, invalid, mock_ngspice_version):
    tb = _load(tmp_path)._tests['level']
    # Configure the supply the run declares, so the declaration is true of
    # the deck and each variant below fails on the input it is testing.
    tb.set_supply_voltage(1.8)
    tb.netlist.generate_netlist()
    monkeypatch.setattr(subprocess, 'run', _success(tb.netlist._run_files))
    report = tb.run(voltage=1.8)
    raw = tb.netlist._raw
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    launch = Mock()
    monkeypatch.setattr(subprocess, 'run', launch)
    kwargs = {'voltage': 1.8}
    if invalid == 'corner':
        kwargs['corner'] = 'not-a-corner'
    elif invalid == 'label':
        kwargs['run_label'] = '../bad'
    elif invalid == 'context':
        kwargs['temperature'] = 80
    else:
        monkeypatch.setattr(tb.netlist.netlister, 'generate_complete_netlist',
                            Mock(side_effect=ValueError('injected generation failure')))
    with pytest.raises((ValueError, SimulationError, CharacterizationContextError)):
        tb.run(**kwargs)
    launch.assert_not_called()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert tb.netlist._raw is raw
    assert tb.netlist.sim_results_loaded
    assert tb.netlist.characterization_context == report.characterization_context
    assert tb.evaluate().metrics == report.metrics


@pytest.mark.parametrize('store', [False, True])
def test_failed_test_run_preserves_reports_measurements_and_committed_db(tmp_path, monkeypatch, store, mock_ngspice_version):
    mod = _load(tmp_path, db_path=str(tmp_path / 'evidence.db'))
    tb = mod._tests['level']
    mod.create_circuit_version()
    try:
        tb.set_supply_voltage(1.8)
        tb.netlist.generate_netlist()
        files = tb.netlist._run_files
        monkeypatch.setattr(subprocess, 'run', _success(files))
        first = mod.run_test('level', store=True, voltage=1.8)
        spec = tb._specs[0]['spec']
        measurements = dict(spec.measurements)
        db_before = list(mod._tracker.db.conn.iterdump())
        first_metrics = dict(first.metrics)
        extract = Mock(side_effect=AssertionError('failed attempts must not measure'))
        tb._metrics[0]['extractor'] = extract
        monkeypatch.setattr(subprocess, 'run', _failure(files, 'missing'))
        tb.set_temperature(80)
        for _ in range(2):
            with pytest.raises(SimulationAborted):
                mod.run_test('level', store=store, temperature=80, voltage=1.8)
            with pytest.raises(SimulationError, match='no simulation results'):
                tb.evaluate()
            _assert_invalid(tb.netlist)
            assert first.metrics == first_metrics
            assert first.characterization_context.voltage == 1.8
            assert first.sim_results is tb.netlist
            assert spec.measurements == measurements
            assert list(mod._tracker.db.conn.iterdump()) == db_before
        extract.assert_not_called()
        assert tb.results_file is None
    finally:
        mod.close()


@pytest.mark.parametrize('entry', ['all_tests', 'corners'])
def test_aggregate_failure_uses_existing_error_channel(tmp_path, monkeypatch, entry, mock_ngspice_version):
    mod = _load(tmp_path)
    tb = mod._tests['level']
    tb.netlist.generate_netlist()
    files = tb.netlist._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    tb.run()
    if entry == 'all_tests':
        monkeypatch.setattr(subprocess, 'run', _failure(files, 'missing'))
        result = mod.run_all_tests()
        assert len(result) == 1
        assert 'error' in next(iter(result.values()))
    else:
        # run_corners() (TASK-301.3) always gives 'tt' its own fresh
        # <testbench dir>/tt/ folder, distinct from the plain run() above,
        # so there is no PREVIOUS output at this corner's path for
        # _failure()'s existence check to assert against — pass it a
        # path-less stand-in and rely on the no-rawfile injection alone.
        # A failing corner no longer aborts run_corners() immediately
        # (AC #3): it is reported as that corner's error, wrapped in
        # CornerRunError, after every corner has been attempted.
        from silk.circuit.spec_report import CornerRunError
        no_prior_files = replace(files, rawfile_path=None, noise_path=None,
                                 logfile_path=None)
        monkeypatch.setattr(subprocess, 'run', _failure(no_prior_files, 'missing'))
        with pytest.raises(CornerRunError) as excinfo:
            tb.run_corners(['tt'])
        assert 'SimulationAborted' in excinfo.value.errors['tt']
    _assert_invalid(tb.netlist)


def test_explicit_readback_never_cleans_or_prepares(tmp_path, monkeypatch, mock_ngspice_version):
    """Formerly test_cache_and_explicit_readback_never_clean_or_prepare: also
    built the testbench with cache=True and asserted a same-corner run()
    reused the first report without preparing or launching anything. The
    report cache is disabled in 0.2.0 (TASK-284), so that half no longer
    applies -- this keeps the explicit-readback-never-launches coverage."""
    tb = _load(tmp_path)._tests['level']
    tb.netlist.generate_netlist()
    files = tb.netlist._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    tb.run()
    saved = tb.results_file
    forbidden = Mock(side_effect=AssertionError('must not prepare or launch'))
    monkeypatch.setattr(RunFiles, 'prepare', forbidden)
    monkeypatch.setattr(RunFiles, 'clear_previous_outputs', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    Path(saved).write_text(_raw(0.9))
    assert tb.evaluate(rawfile=saved).metrics['voltage'] == pytest.approx(0.9)
    RunFiles.resolve('deck', str(tmp_path / 'unused'))
    assert not (tmp_path / 'unused').exists()
    forbidden.assert_not_called()


def test_cache_true_still_prepares_and_launches_every_run(tmp_path, monkeypatch, mock_ngspice_version):
    """The report cache is disabled in 0.2.0 (TASK-284): a cache=True
    testbench prepares, cleans and launches on every run(), exactly like
    cache=False, instead of reusing an earlier report. ``_success``'s
    launch callback asserts the expected prior output was already removed
    before each launch (TASK-144), so a cache hit -- which would skip that
    entirely -- would fail this the second time through just as loudly as a
    missing prepare/clean would."""
    with pytest.warns(UserWarning, match='TASK-130'):
        tb = _load(tmp_path, cache=True)._tests['level']
    tb.netlist.generate_netlist()
    files = tb.netlist._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    first = tb.run()
    second = tb.run()
    assert second is not first


@pytest.mark.parametrize('analyses', [
    [Transient(1e-9, 1e-8)],
    [Noise('out', '0', 'vin', 'dec', 1, 1, 10)],
])
def test_xyce_removes_actual_output_set_and_rejects_no_output(tmp_path, monkeypatch, analyses, mock_ngspice_version):
    mod = _module(tmp_path, 'xyce', analyses)
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    files = mod._run_files
    unrelated = (tmp_path / 'noise.csv' if files.noise_path is None
                 else tmp_path / 'rawfile.raw' if files.rawfile_path is None
                 else tmp_path / 'unrelated.csv')
    unrelated.write_text('unrelated data')
    monkeypatch.setattr(subprocess, 'run', _success(files))
    mod.simulate()
    if files.noise_path:
        np.testing.assert_allclose(mod.load_signal('noise_out').y, [2e-5, 1e-5])
    monkeypatch.setattr(subprocess, 'run', _failure(files, 'missing'))
    with pytest.raises(SimulationAborted):
        mod.simulate()
    _assert_invalid(mod)
    assert unrelated.read_text() == 'unrelated data'


@pytest.mark.parametrize('contents,error', [
    ('"FREQ","ONOISE","INOISE"\n', SimulationAborted),
    ('garbage\n', SimulationError),
    ('"FREQ","ONOISE","INOISE"\n1,broken,3\n', SimulationError),
    ('"FREQ","ONOISE","INOISE"\n1,2\n', SimulationError),
])
def test_xyce_rejects_unusable_fresh_noise_before_publishing(tmp_path, monkeypatch, contents, error, mock_ngspice_version):
    mod = _module(tmp_path, 'xyce', [Noise('out', '0', 'vin', 'dec', 1, 1, 10)])
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    files = mod._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    mod.simulate()

    def launch(argv, **kwargs):
        assert not Path(files.noise_path).exists()
        Path(_launched_files(files, argv, kwargs).noise_path).write_text(contents)
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce\n')

    monkeypatch.setattr(subprocess, 'run', launch)
    with pytest.raises(error):
        mod.simulate()
    _assert_invalid(mod)


@pytest.mark.parametrize('noise_only', [False, True])
def test_direct_xyce_backend_clears_both_legacy_candidates(tmp_path, monkeypatch, noise_only, mock_ngspice_version):
    backend = SimulatorBackend.get('xyce')
    files = backend._resolve_run_files(str(tmp_path / 'deck.cir'), str(tmp_path))
    produced = replace(files, rawfile_path=None) if noise_only else replace(files, noise_path=None)
    Path(files.netlist_path).write_text(_external_deck(files))
    for p in (files.rawfile_path, files.noise_path):
        Path(p).write_text('old result')

    def launch(argv, **kwargs):
        assert not Path(files.rawfile_path).exists()
        assert not Path(files.noise_path).exists()
        return _success(produced)(argv, **kwargs)

    monkeypatch.setattr(subprocess, 'run', launch)
    raw = backend.run(files.netlist_path, files.directory)
    assert (raw.native is None) == noise_only
    monkeypatch.setattr(subprocess, 'run', _failure(files, 'missing'))
    with pytest.raises(SimulationAborted):
        backend.run(files.netlist_path, files.directory)


@pytest.mark.parametrize('archive', [False, True])
def test_fault_worker_direct_backend_rejects_stale_results(tmp_path, monkeypatch, archive, mock_ngspice_version):
    mod = _module(tmp_path / 'source')
    mod.set_output_options(mode='overwrite')
    run_dir = tmp_path / 'worker'
    run_dir.mkdir()
    files = RunFiles.resolve(str(run_dir / 'netlist.cir'), str(run_dir), log=True)
    monkeypatch.setattr(subprocess, 'run', _success(files))
    args = dict(module_pickle=pickle.dumps(mod), fault_set=FaultSet(), seed=42,
                run_id=7, run_dir=str(run_dir), archive_waveforms=archive,
                reconcile_specs=True)
    first = run_one(**args)
    assert first.status == 'simulated'  # the parent scores and disposes
    assert first.seed == 42 and first.run_id == 7
    assert Path(files.rawfile_path).exists()
    prior_manifest = (run_dir / 'manifest.json').read_bytes()
    Path(files.rawfile_path).write_text(_raw(1.8))
    monkeypatch.setattr(subprocess, 'run', _failure(files, 'missing'))
    failed = run_one(**args)
    assert failed.status == 'errored'
    assert 'SimulationAborted' in failed.error
    assert failed.seed == first.seed and failed.manifest.to_json() == first.manifest.to_json()
    assert not Path(files.rawfile_path).exists()
    assert (run_dir / 'manifest.json').read_bytes() == prior_manifest


@pytest.mark.parametrize('missing', ['rawfile_path', 'noise_path'])
def test_xyce_runner_requires_each_explicit_output(tmp_path, monkeypatch, missing, mock_ngspice_version):
    # Exercise the runner's explicit output contract directly. A silk-generated
    # mixed-analysis deck is refused before launch (TASK-220).
    files = RunFiles.resolve(str(tmp_path / 'external.cir'), str(tmp_path), noise=True)
    Path(files.netlist_path).write_text(_external_deck(files))
    runner = XyceRunner(files.netlist_path, files.directory, run_files=files)
    monkeypatch.setattr(subprocess, 'run', _success(files))
    runner.run()
    assert runner.load().npoints == 2
    monkeypatch.setattr(subprocess, 'run', _success(replace(files, **{missing: None})))
    with pytest.raises(SimulationAborted):
        runner.run()
    with pytest.raises(SimulationError, match='No current Xyce results'):
        runner.load()


@pytest.mark.parametrize('noise_only', [False, True])
def test_xyce_manual_external_deck_keeps_legacy_candidate_outputs(tmp_path, monkeypatch, noise_only, mock_ngspice_version):
    mod = _module(tmp_path, 'xyce')
    mod.set_output_options(mode='overwrite')
    Path(os.path.join(mod.dir_folder, mod.netlistname)).write_text('* external deck retained verbatim\n.end\n')
    files = mod.backend._resolve_run_files(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder)
    produced = replace(files, rawfile_path=None) if noise_only else replace(files, noise_path=None)
    Path(files.netlist_path).write_text(_external_deck(files))
    monkeypatch.setattr(subprocess, 'run', _success(produced))
    for _ in range(2):
        mod.run_simulator()
        assert (mod._raw.native is None) == noise_only
    assert Path(os.path.join(mod.dir_folder, mod.netlistname)).read_text() == _external_deck(files)


def test_native_backend_neither_cleans_files_nor_needs_rawfile(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation.backend import RawResult, SpiceBackend

    native = SimpleNamespace(npoints=2)

    class NativeBackend(SpiceBackend):
        def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
            return RawResult(native, netlist_path)

        def make_netlister(self, pdk_details):
            return PySpiceNetlister(pdk_details)

    monkeypatch.setitem(SimulatorBackend._registry, 'native_freshness', NativeBackend)
    mod = _module(tmp_path, 'native_freshness')
    mod.set_output_options(mode='overwrite')
    mod.terminal_current_method = 'savecurrents'
    unrelated = tmp_path / 'rawfile.raw'
    unrelated.write_text('unrelated native-backend file')
    cleanup = Mock(side_effect=AssertionError('native backend must not clear files'))
    monkeypatch.setattr(RunFiles, 'clear_previous_outputs', cleanup)
    mod.simulate()
    assert mod._raw.native is native
    assert mod.results_file == os.path.join(mod.dir_folder, mod.netlistname)
    native.npoints = 0
    with pytest.raises(SimulationAborted):
        mod.simulate()
    _assert_invalid(mod)
    assert unrelated.read_text() == 'unrelated native-backend file'
    cleanup.assert_not_called()


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_direct_backend_keeps_resolved_nondefault_paths(tmp_path, monkeypatch, simulator, mock_ngspice_version):
    from tests.test_run_files import _rename_outputs

    backend = SimulatorBackend.get(simulator)
    _rename_outputs(monkeypatch, type(backend))
    files = backend._resolve_run_files(str(tmp_path / 'external.cir'), str(tmp_path))
    Path(files.netlist_path).write_text(_external_deck(files))
    Path(files.rawfile_path).write_text('old output')
    unrelated = tmp_path / 'rawfile.raw'
    unrelated.write_text('keep default basename')
    monkeypatch.setattr(subprocess, 'run', _success(files))
    raw = backend.run(files.netlist_path, files.directory)
    assert raw.source_path == files.rawfile_path
    assert unrelated.read_text() == 'keep default basename'


# ----------------------------------------------------------------------
# TASK-160 — a deck is stale the moment the configuration it was generated
# from moves. TASK-145 refuses a consumed preparation (new mode only) and
# TASK-139 refuses a changed temperature request or a changed deck ON DISK.
# Neither sees a configuration change: it leaves the file untouched, so its
# digest still matches and the previous deck launched silently.
# ----------------------------------------------------------------------

# Every entry changes the emitted deck. The message fragment is the guard that
# owns the refusal: 'Circuit configuration changed' is TASK-160's general
# comparison, 'Temperature configuration' is TASK-139's request tuple (custom
# lines are part of it, which is why that one case was already caught).
_STALE_CHANGES = {
    'source value': (
        lambda m: setattr(m.sig_sources_dict['vin'], 'dc', 2),
        'Circuit configuration changed'),
    'added element': (
        lambda m: m.add(elements.resistor('rextra', p='out', n='0', value=100)),
        'Circuit configuration changed'),
    'removed element': (
        lambda m: m._element_mgr.elem_dict.pop('rload'),
        'Circuit configuration changed'),
    'added analysis': (
        lambda m: m.add_sim_type(AC('lin', 3, 1, 1e6)),
        'Circuit configuration changed'),
    'changed analysis': (
        lambda m: m.add_sim_type(Transient(1e-9, 1e-8)),
        'Circuit configuration changed'),
    'save list': (
        lambda m: m.set_save_list(['out']),
        'Circuit configuration changed'),
    'simulator option': (
        lambda m: m.set_simulator_options(gmin=1e-13),
        'Circuit configuration changed'),
    'device currents': (
        lambda m: setattr(m, 'save_device_currents', True),
        'Circuit configuration changed'),
    # A comment carries no temperature, so the general gate names the change.
    # A custom line that DOES mention a temperature still trips the narrower
    # temperature refusal, because recognition changes the request itself.
    'custom line': (
        lambda m: m.add_custom_line('* added after generation'),
        'Circuit configuration changed'),
    # 27 C matches what the shared mocked launch reports, so the staleness
    # refusal is what fires rather than the observation mismatch.
    'custom temperature line': (
        lambda m: m.add_custom_line('.temp 27'),
        'Temperature configuration'),
    # 27 C, not 85: the mocked launch below reports the 27 C that the shared
    # _success() log fixture observes, so the recovery run is confirmable.
    'temperature': (
        lambda m: m.set_temperature(27),
        'Temperature configuration'),
}


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
@pytest.mark.parametrize('change', sorted(_STALE_CHANGES))
def test_configuration_change_after_generation_refuses_to_launch(
        tmp_path, monkeypatch, mode, change, mock_ngspice_version):
    mutate, expected = _STALE_CHANGES[change]
    mod = _module(tmp_path)
    mod.set_output_options(mode=mode)
    mod.generate_netlist()
    files = mod._run_files
    prepared = Path(files.netlist_path).read_bytes()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    launch = Mock(side_effect=AssertionError('a stale deck must not launch'))
    monkeypatch.setattr(subprocess, 'run', launch)

    mutate(mod)
    with pytest.raises(SimulationError, match=expected) as error:
        mod.run_simulator()
    assert 'generate_netlist()' in str(error.value)
    launch.assert_not_called()
    # A guard rewrites nothing: the prepared deck and every other file survive.
    assert Path(files.netlist_path).read_bytes() == prepared
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert not mod.sim_results_loaded

    # Regenerating is the instructed recovery. The mocked simulator needs no
    # real models; custom lines conservatively require a library, so use the
    # documented no-library placeholder for this recovery phase only.
    monkeypatch.setattr(mod.pdk_details, 'library_location', os.devnull)
    mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run', _success(mod._run_files))
    mod.run_simulator()
    assert mod.sim_results_loaded


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_stale_deck_precedes_missing_pdk_library(tmp_path, monkeypatch, mode, mock_ngspice_version):
    monkeypatch.setenv('PDK_ROOT', str(tmp_path / 'absent-pdk'))
    mod = _module(tmp_path)
    mod.set_output_options(mode=mode)
    mod.generate_netlist()
    mod.add_custom_line('* added after generation')
    launch = Mock(side_effect=AssertionError('a stale deck must not launch'))
    monkeypatch.setattr(subprocess, 'run', launch)

    with pytest.raises(SimulationError, match='Circuit configuration changed'):
        mod.run_simulator()
    launch.assert_not_called()

    # A fresh deck still reports the missing library before launching.
    mod.generate_netlist()
    with pytest.raises(SimulationError, match="model library is not on disk"):
        mod.run_simulator()
    launch.assert_not_called()


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_unchanged_configuration_still_launches_the_prepared_deck(
        tmp_path, monkeypatch, mode, mock_ngspice_version):
    mod = _module(tmp_path)
    mod.set_output_options(mode=mode)
    mod.generate_netlist()
    files = mod._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    # Configuration that never reaches the deck is not deck-affecting.
    mod.debug = True
    mod.run_simulator(timeout=60)
    np.testing.assert_allclose(mod.load_signal('out').y, 1.8)
    assert mod._run_files is files


def test_overwrite_relaunch_of_an_unchanged_deck_is_still_allowed(tmp_path, monkeypatch, mock_ngspice_version):
    """Overwrite mode has always permitted a second launch of the same deck.

    The general guard compares configuration, not consumption, so that stays
    true — only new mode refuses a consumed preparation (TASK-145).
    """
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run', _success(mod._run_files))
    for _ in range(3):
        mod.run_simulator()
        np.testing.assert_allclose(mod.load_signal('out').y, 1.8)


def test_refusal_names_the_changed_deck_line(tmp_path, monkeypatch, mock_ngspice_version):
    mod = _module(tmp_path)
    mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run',
                        Mock(side_effect=AssertionError('must not launch')))
    mod.sig_sources_dict['vin'].dc = 2
    with pytest.raises(SimulationError) as error:
        mod.run_simulator()
    assert "prepared 'Vvin out 0 DC 1 AC 1'" in str(error.value)
    assert "now 'Vvin out 0 DC 2 AC 1'" in str(error.value)

    mod.generate_netlist()
    mod.add(elements.resistor('rextra', p='out', n='0', value=100))
    with pytest.raises(SimulationError) as error:
        mod.run_simulator()
    # An insertion names the inserted line, not the blank everything shifted on.
    assert 'adds "*** rextra.  Connections = [\'p\', \'n\']."' in str(error.value)


def test_configuration_that_no_longer_emits_a_deck_refuses_at_launch(tmp_path, monkeypatch, mock_ngspice_version):
    mod = _module(tmp_path)
    mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run',
                        Mock(side_effect=AssertionError('must not launch')))
    monkeypatch.setattr(type(mod.netlister), 'generate_complete_netlist',
                        Mock(side_effect=ValueError('injected emission failure')))
    with pytest.raises(SimulationError, match='no longer emits a deck') as error:
        mod.run_simulator()
    assert 'injected emission failure' in str(error.value)
    assert 'generate_netlist()' in str(error.value)


def test_freshness_check_is_silent_about_the_deck_it_rebuilds(tmp_path, monkeypatch, capsys, mock_ngspice_version):
    """Emission talks to the user; a guard that runs on every launch must not.

    A save list prints an inventory line and warns nothing, and a save list
    beside a Noise analysis prints a note — once, for the generation the caller
    asked for.
    """
    mod = _module(tmp_path, analyses=[Transient(1e-7, 1e-6),
                                      Noise('out', '0', 'vin', 'dec', 1, 1, 10)])
    mod.set_save_list(['out'])
    mod.generate_netlist()
    generated = capsys.readouterr()
    assert 'save list' in (generated.out + generated.err)
    monkeypatch.setattr(subprocess, 'run', _success(mod._run_files))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mod.run_simulator()
    launched = capsys.readouterr()
    assert 'save list' not in (launched.out + launched.err)


def test_explicit_deck_without_a_preparation_is_never_second_guessed(tmp_path, monkeypatch, mock_ngspice_version):
    """A caller who supplies a deck by path has no recorded configuration."""
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    Path(os.path.join(mod.dir_folder, mod.netlistname)).write_text('* external deck retained verbatim\n.end\n')
    assert mod._deck_preparation is None
    files = mod.backend._resolve_run_files(os.path.join(mod.dir_folder, mod.netlistname), mod.dir_folder)
    monkeypatch.setattr(subprocess, 'run', _success(files))
    mod.add(elements.resistor('rextra', p='out', n='0', value=100))
    mod.run_simulator()
    assert mod.sim_results_loaded
    assert Path(os.path.join(mod.dir_folder, mod.netlistname)).read_text() == '* external deck retained verbatim\n.end\n'


def test_subcircuit_generation_records_nothing_and_writes_nothing(tmp_path):
    from silk import Module, load_pdk

    sub = Module('sky130', str(tmp_path), 'sub.cir', load_pdk('sky130'),
                 module_name='sub')
    sub.add(elements.resistor('rload', p='out', n='0', value=1000))
    sub.define_module_ports({'out': 'output'})
    text = sub.generate_netlist()
    assert text.startswith('.subckt sub')
    assert sub._deck_preparation is None
    assert sub._run_files is None
    assert [p.name for p in tmp_path.iterdir()] == []


def test_a_failed_generation_leaves_no_launchable_preparation(tmp_path, monkeypatch, mock_ngspice_version):
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    monkeypatch.setattr(type(mod.netlister), 'generate_complete_netlist',
                        Mock(side_effect=ValueError('injected generation failure')))
    with pytest.raises(ValueError):
        mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run',
                        Mock(side_effect=AssertionError('must not launch')))
    with pytest.raises(SimulationError, match='did not complete'):
        mod.run_simulator()


@needs_ngspice
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_real_ngspice_stale_deck_is_refused_and_the_regenerated_one_runs(tmp_path, mode):
    """The filed case, end to end: generate, change the configuration, run.

    Before TASK-160 this launched the previous deck and reported 1 V while the
    live configuration said 2 V.
    """
    mod = _module(tmp_path)
    mod.set_output_options(mode=mode)
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 1)

    # A fresh preparation, so the refusal below is this guard's and not new
    # mode's consumed-preparation refusal (TASK-145).
    mod.generate_netlist()
    prepared = Path(mod._run_files.netlist_path).read_bytes()
    mod.sig_sources_dict['vin'].dc = 2
    with pytest.raises(SimulationError, match='Circuit configuration changed') as error:
        mod.run_simulator()
    assert "now 'Vvin out 0 DC 2 AC 1'" in str(error.value)
    assert Path(mod._run_files.netlist_path).read_bytes() == prepared

    mod.generate_netlist()
    mod.run_simulator()
    mod.read_raw_file()
    np.testing.assert_allclose(mod.load_signal('out').y, 2)


@needs_ngspice
def test_real_ngspice_temperature_change_keeps_its_own_refusal(tmp_path):
    """The originally filed reproduction, still owned by TASK-139's guard."""
    mod = _module(tmp_path)
    mod.set_temperature(85)
    mod.generate_netlist()
    assert '.options temp=85' in Path(mod._run_files.netlist_path).read_text()
    mod.set_temperature(-40)
    with pytest.raises(SimulationError, match='Temperature configuration'):
        mod.run_simulator()
    mod.generate_netlist()
    assert '.options temp=-40' in Path(mod._run_files.netlist_path).read_text()
    mod.run_simulator()
    assert mod.operating_conditions.temperature_c == -40


@pytest.mark.parametrize('installed', ['42', None])
def test_target_mismatch_preserves_results_before_freshness(tmp_path, monkeypatch, installed):
    from silk.simulation import _ngspice_version
    from silk.circuit import modules_netlist
    from silk.exceptions import SimulationConfigurationError
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: installed)
    mod = _module(tmp_path)
    mod.generate_netlist(ngspice_version='45.2')
    previous = object()
    mod._raw = previous
    files = mod._run_files
    Path(files.rawfile_path).write_text('previous result')
    monkeypatch.setattr(modules_netlist, '_require_fresh_deck',
                        lambda module: pytest.fail('freshness reached before target guard'))
    with pytest.raises(SimulationConfigurationError, match=r'45.2.*(42|unknown)'):
        mod.run_simulator()
    assert mod._raw is previous
    assert Path(files.rawfile_path).read_text() == 'previous result'
    assert not mod._run_consumed


def test_same_target_and_default_regeneration_can_run(tmp_path, monkeypatch):
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    mod = _module(tmp_path)
    mod.generate_netlist(ngspice_version='42')
    monkeypatch.setattr(subprocess, 'run', _success(mod._run_files))
    mod.run_simulator()
    mod.generate_netlist(ngspice_version='45.2')
    mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run', _success(mod._run_files))
    mod.run_simulator()
