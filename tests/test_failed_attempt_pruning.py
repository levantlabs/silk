"""TASK-586: a later accepted attempt prunes that folder's earlier failed leftovers.

Failed launched attempts leave a hidden ``.silk-<id>.cir`` deck and a
``.silk-generation-*`` directory (known_issues O6). When a later attempt in the
same output folder is accepted they are removed; with no later success they
stay as the diagnosis, and a generation held by a reader is never touched.
"""
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from silk.exceptions import SimulationError
from silk.simulation import generation
from silk.simulation.backend import RawResult, SimulatorBackend
from silk.simulation.run_files import RunFiles
from silk.signals.rawfile import read_rawfile

from tests._env import needs_ngspice
from tests.test_run_generations import _patch_simulator_launch, raw_bytes


def _files(folder, name='deck.cir'):
    folder.mkdir(exist_ok=True)
    files = RunFiles.resolve(str(folder / name), str(folder))
    Path(files.netlist_path).write_text('* deck\n.end\n')
    return files


def _hidden(folder):
    return {p.name for p in folder.iterdir() if p.name.startswith('.silk-')}


def _own(gen):
    """Names of the hidden artifacts that belong to *gen* itself."""
    return {Path(gen.directory).name, Path(gen.staged.netlist_path).name}


def _fail(files, launched=True):
    """One failed attempt, as the runners leave it."""
    gen = generation.begin(files)
    gen.launched = launched
    generation.abandon(gen)
    return gen


def _succeed(files, value=1):
    gen = generation.begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(value))
    raw = generation.attach(
        RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path),
        generation.offer(gen))
    generation.accept(raw)
    return raw


def test_fail_fail_success_prunes_both_failures(tmp_path):
    files = _files(tmp_path)
    first, second = _fail(files), _fail(files)
    assert _own(first) <= _hidden(tmp_path) and _own(second) <= _hidden(tmp_path)
    raw = _succeed(files)
    assert _hidden(tmp_path) == _own(raw._generation)


def test_success_then_fail_keeps_leftover(tmp_path):
    files = _files(tmp_path)
    raw = _succeed(files)
    failed = _fail(files)
    assert _hidden(tmp_path) == _own(raw._generation) | _own(failed)


def test_fail_without_later_success_stays(tmp_path):
    files = _files(tmp_path)
    failed = [_fail(files), _fail(files)]
    assert _hidden(tmp_path) == _own(failed[0]) | _own(failed[1])


def test_other_folder_is_untouched(tmp_path):
    a, b = _files(tmp_path / 'a'), _files(tmp_path / 'b')
    failed = _fail(a)
    raw = _succeed(b)
    assert _hidden(tmp_path / 'a') == _own(failed)
    assert _hidden(tmp_path / 'b') == _own(raw._generation)


def test_later_started_failure_stays_after_earlier_acceptance(tmp_path):
    files = _files(tmp_path)
    gen = generation.begin(files)
    Path(gen.staged.rawfile_path).write_bytes(raw_bytes(1))
    later = _fail(files)  # begins while ``gen`` is still pending
    raw = generation.attach(
        RawResult(read_rawfile(gen.staged.rawfile_path, lazy=True), files.rawfile_path),
        generation.offer(gen))
    generation.accept(raw)
    assert _own(later) <= _hidden(tmp_path)


def test_held_reader_generation_survives_pruning(tmp_path):
    files = _files(tmp_path)
    held = _succeed(files, 3)
    _fail(files)
    newest = _succeed(files, 5)  # relocates the held generation's artifacts
    assert _hidden(tmp_path) == _own(held._generation) | _own(newest._generation)
    assert_array_equal(held.native.vectors[2].data, [13, 13])
    assert_array_equal(newest.native.vectors[2].data, [15, 15])


def test_prune_error_never_fails_acceptance(tmp_path, monkeypatch):
    files = _files(tmp_path)
    failed = _fail(files)
    calls = []

    def refuse(path, *args, **kwargs):
        calls.append(path)
        raise OSError('read-only')

    monkeypatch.setattr(generation.shutil, 'rmtree', refuse)
    raw = _succeed(files)
    assert raw._generation.accepted
    assert str(failed.directory) in [str(c) for c in calls]
    assert_array_equal(read_rawfile(files.rawfile_path).vectors[2].data, [11, 11])
    monkeypatch.undo()
    generation.shutil.rmtree(failed.directory, ignore_errors=True)


def test_unlaunched_abandoned_attempt_is_pruned(tmp_path):
    # A worker failing before offer never tells the parent it launched.
    files = _files(tmp_path)
    _fail(files, launched=False)
    _fail(files, launched=False)
    raw = _succeed(files)
    assert _hidden(tmp_path) == _own(raw._generation)


def test_native_reload_error_keeps_earlier_diagnostics(tmp_path, monkeypatch):
    files = _files(tmp_path)
    failed = _fail(files)
    monkeypatch.setattr(generation, '_load_native',
                        lambda raw: (_ for _ in ()).throw(OSError('reload failed')))
    with pytest.raises(SimulationError):
        _succeed(files)
    assert _own(failed) <= _hidden(tmp_path)


@pytest.mark.parametrize('broken', ['dir', 'deck', 'partial'])
def test_failed_removal_is_kept_and_retried(tmp_path, monkeypatch, broken):
    files = _files(tmp_path)
    a, b = _fail(files), _fail(files)
    real_rmtree, real_unlink = generation.shutil.rmtree, Path.unlink
    fail_dir = {'dir': {a.directory}, 'partial': {a.directory}, 'deck': set()}[broken]
    fail_deck = {'deck': {a.staged.netlist_path}, 'dir': set(), 'partial': set()}[broken]

    def rmtree(path, *args, **kwargs):
        if str(path) in fail_dir:
            raise OSError('busy')
        return real_rmtree(path, *args, **kwargs)

    def unlink(self, *args, **kwargs):
        if str(self) in fail_deck:
            raise OSError('busy')
        return real_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(generation.shutil, 'rmtree', rmtree)
        patch.setattr(Path, 'unlink', unlink)
        first = _succeed(files)  # cleanup error must not fail the result
    assert first._generation.accepted
    left = _hidden(tmp_path) - _own(first._generation)
    assert left and not left & _own(b) - {Path(a.directory).name, Path(a.staged.netlist_path).name}
    second = _succeed(files)
    assert _hidden(tmp_path) == _own(second._generation) | _own(first._generation)


@needs_ngspice
def test_backend_fail_fail_success_prunes(tmp_path, monkeypatch):
    files = _files(tmp_path)
    outcomes = iter([None, None, 7])

    def launch(argv, **kwargs):
        value = next(outcomes)
        if value is None:
            raise OSError('launch failed')
        Path(argv[argv.index('-r') + 1]).write_bytes(raw_bytes(value))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    _patch_simulator_launch(monkeypatch, launch)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    backend = SimulatorBackend.get('ngspice')
    for _ in range(2):
        with pytest.raises(SimulationError):
            backend.run(files.netlist_path, files.directory, run_files=files)
    assert len(list(tmp_path.glob('.silk-generation-*'))) == 2
    ok = backend.run(files.netlist_path, files.directory, run_files=files)
    assert _hidden(tmp_path) == _own(ok._generation)


def _flaky_worker(run_dir, files, flag):
    from tests.test_run_generations import _backend_worker
    return _backend_worker(run_dir, files, 'ngspice')


@needs_ngspice
def test_worker_failure_is_pruned_by_later_parent_accept(tmp_path, monkeypatch):
    from silk.simulation.executor import RunTask, run_tasks
    files = _files(tmp_path)
    flag = tmp_path / 'fail.flag'
    flag.write_text('x')

    def launch(argv, **kwargs):
        if flag.exists():
            raise OSError('launch failed')
        Path(argv[argv.index('-r') + 1]).write_bytes(raw_bytes(9))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    _patch_simulator_launch(monkeypatch, launch)
    monkeypatch.setattr('silk.simulation.runner.warn_version_advice', lambda *a: None)
    monkeypatch.setattr('silk.simulation._ngspice_compat.warn_version_advice', lambda *a: None)

    def one():
        task = RunTask(_flaky_worker, str(tmp_path), args=(files, str(flag)), simulator='ngspice')
        return run_tasks([task], n_workers=2)[0]

    assert one().error is not None
    assert len(list(tmp_path.glob('.silk-generation-*'))) == 1
    flag.unlink()
    result = one()
    assert result.error is None, result.error
    raw = result.value.netlist._raw
    generation.accept(raw)
    assert _hidden(tmp_path) == _own(raw._generation)


@pytest.mark.parametrize('stage', ['after_dir', 'after_deck', 'clear_outputs'])
def test_begin_unwind_is_pruned_with_old_reader_held(tmp_path, monkeypatch, stage):
    files = _files(tmp_path)
    held = _succeed(files, 3)
    assert held.native.vectors[2]._data is None
    if stage == 'after_dir':
        monkeypatch.setattr(generation.Path, 'write_text',
                            lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    elif stage == 'after_deck':
        real_replace = generation.replace

        def replace(obj, **changes):  # the staging call is the one with 'completion'
            if 'completion' in changes:
                raise OSError('boom')
            return real_replace(obj, **changes)

        monkeypatch.setattr(generation, 'replace', replace)
    else:
        monkeypatch.setattr(type(files), 'clear_previous_outputs',
                            lambda self: (_ for _ in ()).throw(OSError('busy')))
    with pytest.raises((SimulationError, OSError)):
        generation.begin(files)
    monkeypatch.undo()
    leftover = _hidden(tmp_path) - _own(held._generation)
    assert leftover  # kept as diagnosis until a later success
    newest = _succeed(files, 5)
    assert _hidden(tmp_path) == _own(held._generation) | _own(newest._generation)
    assert_array_equal(held.native.vectors[2].data, [13, 13])


def test_failure_of_an_exited_process_is_not_pruned(tmp_path):
    # Only failures known to the current process are pruned (known_issues O6).
    import sys
    files = _files(tmp_path)
    code = ('import sys\nfrom silk.simulation import generation\n'
            'from silk.simulation.run_files import RunFiles\n'
            'f = RunFiles.resolve(sys.argv[1], sys.argv[2])\n'
            'g = generation.begin(f); g.launched = True; generation.abandon(g)\n')
    subprocess.run([sys.executable, '-c', code, files.netlist_path, str(tmp_path)], check=True)
    before = _hidden(tmp_path)
    assert before
    raw = _succeed(files)
    assert _hidden(tmp_path) == before | _own(raw._generation)


def test_retargeted_alias_cleans_original_and_leaves_other_tree(tmp_path):
    real_a, real_b = tmp_path / 'a', tmp_path / 'b'
    real_a.mkdir(), real_b.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real_a)
    files = _files(alias)
    failed = _fail(files)
    names = _own(failed)
    for name in names:  # an unrelated tree with the same private names
        (real_b / name).mkdir() if name.startswith('.silk-generation') else (real_b / name).write_text('keep')
    alias.unlink()
    alias.symlink_to(real_b)
    raw = _succeed(_files(real_a))  # the original folder, by its real path
    assert _hidden(real_a) == _own(raw._generation)
    assert _hidden(real_b) == names


def test_relative_and_absolute_spellings_match_one_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'out').mkdir()
    relative = RunFiles.resolve('out/deck.cir', 'out')
    Path(relative.netlist_path).write_text('* deck\n.end\n')
    failed = _fail(relative)
    absolute = _files(tmp_path / 'out')
    raw = _succeed(absolute)
    assert _hidden(tmp_path / 'out') == _own(raw._generation)
    assert not _own(failed) & _hidden(tmp_path / 'out') - _own(raw._generation)


def test_deck_outside_output_folder_is_pruned_with_its_attempt(tmp_path):
    (tmp_path / 'decks').mkdir()
    out = tmp_path / 'out'
    out.mkdir()
    deck = tmp_path / 'decks' / 'deck.cir'
    deck.write_text('* deck\n.end\n')
    files = RunFiles.resolve(str(deck), str(out))
    failed = _fail(files)
    assert _hidden(tmp_path / 'decks') == {Path(failed.staged.netlist_path).name}
    _succeed(files)
    assert not (_hidden(tmp_path / 'decks') & {Path(failed.staged.netlist_path).name})


def test_unrelated_private_artifacts_stay(tmp_path):
    files = _files(tmp_path)
    other = tmp_path / '.silk-generation-foreign'
    other.mkdir()
    (other / 'rawfile.raw').write_text('someone else')
    stray = tmp_path / '.silk-deadbeef.cir'
    stray.write_text('* not ours')
    _fail(files)
    raw = _succeed(files)
    assert _hidden(tmp_path) == _own(raw._generation) | {other.name, stray.name}


def test_private_log_named_by_old_error_goes_only_after_success(tmp_path):
    files = _files(tmp_path)
    gen = generation.begin(files)
    gen.launched = True
    Path(gen.staged.logfile_path).write_text('diagnostic')
    Path(files.logfile_path or tmp_path / 'output.log').write_text('intervening')
    error = SimulationError('failed')
    generation.publish_failure_log(gen, error)
    generation.abandon(gen)
    assert Path(error.log_path).exists() and Path(gen.directory) in Path(error.log_path).parents
    _succeed(files)
    assert not Path(error.log_path).exists()


def test_forked_child_success_does_not_prune_parent_failure(tmp_path):
    import os
    files = _files(tmp_path)
    failed = _fail(files)
    pid = os.fork()
    if pid == 0:
        status = 1
        try:
            _succeed(files)
            status = 0
        finally:
            os._exit(status)
    _, wait = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(wait) == 0
    assert _own(failed) <= _hidden(tmp_path)
