"""silk.simulation.executor — shared run executor (TASK-301.1)."""
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

import silk.simulation.executor as executor_mod
from silk.simulation.executor import RunTask, resolve_n_workers, run_tasks, spawn_tracked


def _ok_task(run_dir, value):
    with open(os.path.join(run_dir, 'value.txt'), 'w') as f:
        f.write(str(value))
    return value


def _raising_task(run_dir, message):
    raise RuntimeError(message)


def _make_tasks(tmp_path):
    return [
        RunTask(fn=_ok_task, run_dir=str(tmp_path / 'a'), args=(1,)),
        RunTask(fn=_raising_task, run_dir=str(tmp_path / 'b'), args=('boom',)),
        RunTask(fn=_ok_task, run_dir=str(tmp_path / 'c'), args=(3,)),
    ]


def _assert_order_and_error_capture(results, tasks):
    assert [r.index for r in results] == [0, 1, 2]
    assert [r.run_dir for r in results] == [t.run_dir for t in tasks]

    assert results[0].ok
    assert results[0].value == 1
    assert results[2].ok
    assert results[2].value == 3

    assert not results[1].ok
    assert results[1].value is None
    assert 'RuntimeError: boom' in results[1].error

    for t in tasks:
        assert os.path.isdir(t.run_dir)
    with open(os.path.join(tasks[0].run_dir, 'value.txt')) as f:
        assert f.read() == '1'
    with open(os.path.join(tasks[2].run_dir, 'value.txt')) as f:
        assert f.read() == '3'
    assert not os.path.exists(os.path.join(tasks[1].run_dir, 'value.txt'))


class TestRunTasksSerial:
    def test_order_and_error_capture(self, tmp_path):
        tasks = _make_tasks(tmp_path)
        results = run_tasks(tasks, n_workers=1)
        _assert_order_and_error_capture(results, tasks)

    def test_none_n_workers_is_serial(self, tmp_path):
        tasks = _make_tasks(tmp_path)
        results = run_tasks(tasks, n_workers=None)
        _assert_order_and_error_capture(results, tasks)

    def test_empty_task_list(self):
        assert run_tasks([], n_workers=1) == []


class TestRunTasksPool:
    def test_order_and_error_capture(self, tmp_path):
        tasks = _make_tasks(tmp_path)
        results = run_tasks(tasks, n_workers=2)
        _assert_order_and_error_capture(results, tasks)

    def test_more_workers_than_tasks(self, tmp_path):
        tasks = _make_tasks(tmp_path)
        results = run_tasks(tasks, n_workers=8)
        _assert_order_and_error_capture(results, tasks)


class TestResolveNWorkers:
    def test_none_and_one_are_serial(self):
        assert resolve_n_workers(None) == 1
        assert resolve_n_workers(1) == 1

    def test_int_capped_at_cpu_count(self):
        cpu = os.cpu_count() or 1
        assert resolve_n_workers(cpu + 10) == cpu
        assert resolve_n_workers(1) == 1
        if cpu > 1:
            assert resolve_n_workers(2) == min(2, cpu)

    def test_all_means_every_cpu(self):
        assert resolve_n_workers('all') == (os.cpu_count() or 1)

    @pytest.mark.parametrize('bad', [0, -1, -5])
    def test_int_below_one_raises(self, bad):
        with pytest.raises(ValueError):
            resolve_n_workers(bad)

    @pytest.mark.parametrize('bad', [1.5, 'nope', object(), True])
    def test_non_int_non_sentinel_raises(self, bad):
        with pytest.raises(ValueError):
            resolve_n_workers(bad)


    def test_behavioral_forces_serial_regardless_of_request(self):
        assert resolve_n_workers('all', backend='behavioral') == 1
        assert resolve_n_workers(8, backend='behavioral') == 1
        assert resolve_n_workers(None, backend='behavioral') == 1

    def test_behavioral_forced_via_backend_object_name(self):
        class FakeBackend:
            name = 'behavioral'

        assert resolve_n_workers(8, backend=FakeBackend()) == 1

    def test_spice_backend_object_uses_normal_resolution(self):
        class FakeBackend:
            name = 'ngspice'

        cpu = os.cpu_count() or 1
        assert resolve_n_workers(4, backend=FakeBackend()) == min(4, cpu)


class TestPicklingFailureIsolation:
    """TASK-301.7 AC#2: an unpicklable task must not abort the whole call.

    Before the fix, a task whose payload could not be pickled raised out of
    ``pool.py``'s ``_handle_tasks`` -> ``ar.get()`` and propagated straight
    out of ``run_tasks``, aborting every other task in the same batch. Now
    it is pickled in the parent before dispatch and folded into that task's
    own ``RunOutcome.error``, naming the unpicklable object.
    """

    def test_unpicklable_task_folds_into_its_own_outcome(self, tmp_path):
        tasks = [
            RunTask(fn=_ok_task, run_dir=str(tmp_path / 'a'), args=(1,)),
            # A lambda `fn` can never be pickled (local/anonymous object) --
            # module-level functions like _ok_task above always can.
            RunTask(fn=(lambda run_dir: None), run_dir=str(tmp_path / 'b')),
            RunTask(fn=_ok_task, run_dir=str(tmp_path / 'c'), args=(3,)),
        ]

        results = run_tasks(tasks, n_workers=2)

        assert [r.index for r in results] == [0, 1, 2]
        assert [r.run_dir for r in results] == [t.run_dir for t in tasks]

        assert results[0].ok and results[0].value == 1
        assert results[2].ok and results[2].value == 3

        assert not results[1].ok
        assert results[1].value is None
        assert '<lambda>' in results[1].error
        assert 'pickl' in results[1].error.lower()

        with open(os.path.join(tasks[0].run_dir, 'value.txt')) as f:
            assert f.read() == '1'
        with open(os.path.join(tasks[2].run_dir, 'value.txt')) as f:
            assert f.read() == '3'

    def test_all_tasks_unpicklable_returns_all_as_errors(self, tmp_path):
        tasks = [
            RunTask(fn=(lambda run_dir: None), run_dir=str(tmp_path / 'x')),
            RunTask(fn=(lambda run_dir: None), run_dir=str(tmp_path / 'y')),
        ]
        results = run_tasks(tasks, n_workers=2)
        assert [r.index for r in results] == [0, 1]
        assert all(not r.ok for r in results)
        assert all('<lambda>' in r.error for r in results)


class TestSpawnTrackedMainThreadOnly:
    """TASK-301.7 AC#4.

    ``PR_SET_PDEATHSIG`` fires when the thread that called ``prctl`` exits,
    not when the process does. Installing it unconditionally meant a child
    launched via ``spawn_tracked`` from a short-lived ``threading.Thread``
    was killed the instant that thread returned, even though the launching
    *process* was still alive. It is now only installed when the call runs
    on the process's main thread.
    """

    def test_child_from_short_lived_thread_survives_thread_exit(self):
        holder = {}

        def launch():
            holder['proc'] = spawn_tracked(
                [sys.executable, '-c', 'import time; time.sleep(2)'])

        t = threading.Thread(target=launch)
        t.start()
        t.join()
        assert not t.is_alive()

        proc = holder['proc']
        try:
            time.sleep(1.0)
            assert proc.poll() is None, (
                "child was killed shortly after its launching thread exited "
                "-- PDEATHSIG must not be installed off the main thread")
            proc.wait(timeout=5)
            assert proc.returncode == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class TestPoolForkContext:
    """TASK-301.9: pool workers must start with an explicit 'fork' context
    (TASK-301.8's name-based device rebuild needs the parent's already-loaded
    PDK classes, which only a forked, copy-on-write worker inherits), and
    fall back to serial execution -- with a warning, never a failure inside
    a worker -- on a platform where 'fork' is unavailable."""

    def test_fork_context_used_on_linux(self, tmp_path, monkeypatch):
        assert 'fork' in mp.get_all_start_methods()

        requested = []
        real_get_context = mp.get_context

        def spy_get_context(method=None):
            requested.append(method)
            return real_get_context(method)

        monkeypatch.setattr(executor_mod.mp, 'get_context', spy_get_context)

        tasks = _make_tasks(tmp_path)
        results = run_tasks(tasks, n_workers=2)
        _assert_order_and_error_capture(results, tasks)

        assert 'fork' in requested

    def test_serial_fallback_when_fork_unavailable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            executor_mod.mp, 'get_all_start_methods',
            lambda: ['spawn', 'forkserver'])

        def _fail_get_context(method=None):
            raise AssertionError(
                'get_context must not be called once get_all_start_methods '
                'excludes fork')

        monkeypatch.setattr(executor_mod.mp, 'get_context', _fail_get_context)

        tasks = _make_tasks(tmp_path)
        with pytest.warns(RuntimeWarning, match='fork'):
            results = run_tasks(tasks, n_workers=2)

        _assert_order_and_error_capture(results, tasks)

    def test_serial_fallback_never_raises_out_of_run_tasks(self, tmp_path, monkeypatch):
        """The fallback message says workers never fail; make sure the
        fallback path itself can't blow up run_tasks for an ordinary task
        list, matching the pool path's own error-capture contract."""
        monkeypatch.setattr(
            executor_mod.mp, 'get_all_start_methods', lambda: ['spawn'])

        tasks = [RunTask(fn=_raising_task, run_dir=str(tmp_path / 'z'), args=('boom',))]
        with pytest.warns(RuntimeWarning):
            results = run_tasks(tasks, n_workers=4)

        assert len(results) == 1
        assert not results[0].ok
        assert 'RuntimeError: boom' in results[0].error


class TestKeyboardInterruptCleanup:
    def test_sigint_kills_pool_and_tracked_subprocesses(self, tmp_path):
        """Simulates Ctrl-C on a running pool: sends SIGINT to a driver
        process (silk/tests/_executor_ctrlc_driver.py) that has two pool
        workers each babysitting a heartbeat-writing child process started
        via spawn_tracked. Asserts the driver exits and the heartbeat files
        stop growing — i.e. the tracked children were killed along with
        the pool rather than orphaned."""
        driver = os.path.join(os.path.dirname(__file__), '_executor_ctrlc_driver.py')
        worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env['PYTHONPATH'] = worktree_root + os.pathsep + env.get('PYTHONPATH', '')

        base = str(tmp_path)
        proc = subprocess.Popen([sys.executable, driver, base], env=env)
        try:
            heartbeats = [os.path.join(base, f'task_{i}', 'heartbeat.txt')
                          for i in range(2)]
            # Generous bounds: these only cap a hang. Under a pinned,
            # niced heavy run the driver's imports + two spawned workers
            # take far longer than on an idle box. Measure interrupt latency
            # only after both heartbeat children have started.
            deadline = time.time() + 120
            while time.time() < deadline:
                if all(os.path.exists(h) and os.path.getsize(h) > 0
                       for h in heartbeats):
                    break
                time.sleep(0.1)
            else:
                pytest.fail('heartbeat files never appeared — driver '
                            'did not start its tasks')

            interrupted_at = time.monotonic()
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=120)
            assert time.monotonic() - interrupted_at < 15, (
                'driver did not exit promptly after SIGINT — heartbeat children '
                'may have finished on their own instead of being cancelled'
            )

            sizes_at_exit = [os.path.getsize(h) for h in heartbeats]
            time.sleep(1.0)
            sizes_after_wait = [os.path.getsize(h) for h in heartbeats]
            assert sizes_after_wait == sizes_at_exit, (
                "heartbeat file(s) kept growing after the driver exited — "
                "a tracked simulator subprocess was orphaned"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.parametrize('simulator,workers,expected', [
    ('ngspice', 2, ['/selected/ngspice']),
    ('ngspice', 1, []),
    ('xyce', 2, []),
    (None, 2, []),
])
def test_parent_advice_only_for_parallel_ngspice(monkeypatch, tmp_path,
                                               simulator, workers, expected):
    from silk.simulation import _ngspice_compat, runner
    advised = []
    monkeypatch.setattr(runner, 'ngspice_executable', lambda: '/selected/ngspice')
    monkeypatch.setattr(_ngspice_compat, 'warn_version_advice', advised.append)
    tasks = [RunTask(_ok_task, str(tmp_path / str(i)), args=(i,), simulator=simulator)
             for i in range(2)]
    outcomes = run_tasks(tasks, n_workers=workers)
    assert all(outcome.ok for outcome in outcomes)
    if workers > 1 and executor_mod._pool_context() is None:
        expected = []
    assert advised == expected
