"""TASK-301.3 reviewer correction: ngspice launches stay IN the caller's
process group (no start_new_session) and rely on PR_SET_PDEATHSIG plus
subprocess.run's own kill-on-exception for cleanup, rather than
spawn_tracked's group isolation (which would have taken ngspice OUT of a
process group a supervisor might kill by group).

Uses a real, killable OS process for the "ngspice" launch (a heartbeat-
writing stand-in, tests/_slow_ngspice_stub.py, via $SILK_NGSPICE_BIN) driven
through the REAL NgspiceRunner.run()/subprocess.run() path, in its own
driver process (tests/_run_corners_kill_driver.py) so the test can send it
a real SIGKILL (of its whole process group) or a real SIGINT (to just it).
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

_DRIVER = os.path.join(os.path.dirname(__file__), '_run_corners_kill_driver.py')
_STUB = os.path.join(os.path.dirname(__file__), '_slow_ngspice_stub.py')


def _spawn_driver(mode, base, *, own_group=False):
    worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env['PYTHONPATH'] = worktree_root + os.pathsep + env.get('PYTHONPATH', '')
    kwargs = {'start_new_session': True} if own_group else {}
    return subprocess.Popen(
        [sys.executable, _DRIVER, mode, base, _STUB], env=env, **kwargs)


def _wait_for_heartbeats(base, count, deadline_s=20):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        # Simulator outputs may be inside a private generation directory.
        found = list(Path(base).rglob('*.heartbeat'))
        if len(found) >= count and all(os.path.getsize(h) > 0 for h in found):
            return found
        time.sleep(0.1)
    pytest.fail(f'expected {count} heartbeat file(s) under {base}, found none in time '
                f'(driver never reached the simulator launch)')


def _no_growth_after(heartbeats, wait_s=1.0):
    sizes_before = [os.path.getsize(h) for h in heartbeats]
    time.sleep(wait_s)
    sizes_after = [os.path.getsize(h) for h in heartbeats]
    return sizes_after == sizes_before


@pytest.mark.slow
@pytest.mark.parametrize('mode,n_heartbeats', [('serial', 1), ('corners', 2)])
def test_sigkill_of_the_whole_process_group_leaves_no_ngspice(tmp_path, mode, n_heartbeats):
    """(a) A supervisor that kills the whole process GROUP (os.killpg) --
    a job scheduler, a cancelled CI job, a terminal Ctrl-C that signals the
    foreground group -- must still reach ngspice: the reviewer's correction
    keeps it IN that group (spawn_tracked/subprocess.run no longer set
    start_new_session)."""
    base = str(tmp_path)
    driver = _spawn_driver(mode, base, own_group=True)
    try:
        heartbeats = _wait_for_heartbeats(base, n_heartbeats)
        pgid = os.getpgid(driver.pid)
        os.killpg(pgid, signal.SIGKILL)
        driver.wait(timeout=15)
        assert _no_growth_after(heartbeats), (
            'a heartbeat kept growing after the whole process group was '
            'SIGKILLed -- an ngspice-stub child survived its driver')
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()


@pytest.mark.slow
def test_sigint_during_parallel_run_corners_leaves_no_ngspice(tmp_path):
    """(b) Ctrl-C (a real SIGINT to just the driver pid, not its group --
    the case spawn_tracked's group-kill mechanism can't cover on its own)
    during run_corners(n_workers=2) must still leave no ngspice behind.
    KeyboardInterrupt in the parent -> Pool.terminate() -> SIGTERM to each
    worker -> SystemExit unwinds subprocess.run()'s kill-on-exception and
    publishes the cancelled attempt's log through the parent coordinator.
    PR_SET_PDEATHSIG remains a fallback for uncatchable worker termination."""
    base = str(tmp_path)
    driver = _spawn_driver('corners', base)
    try:
        heartbeats = _wait_for_heartbeats(base, 2)
        driver.send_signal(signal.SIGINT)
        driver.wait(timeout=15)
        assert _no_growth_after(heartbeats), (
            'a heartbeat kept growing after Ctrl-C to run_corners(n_workers=2) '
            '-- an ngspice-stub child was orphaned')
        for heartbeat in heartbeats:
            public = heartbeat.parent.parent
            assert (public / 'output.log').is_file()
            assert not (public / 'rawfile.raw').exists()
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()


@pytest.mark.slow
@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux child PID evidence')
@pytest.mark.parametrize('mode', ['authentication', 'reply', 'watchdog'])
def test_sigterm_with_silent_coordinator_exits_and_reaps_simulator(tmp_path, mode):
    import socket
    import threading
    from multiprocessing.connection import Listener
    from silk.simulation.executor import _WORKER_CANCEL_TIMEOUT

    release = threading.Event()
    requested = threading.Event()
    if mode == 'authentication':
        server = socket.socket()
        server.bind(('127.0.0.1', 0))
        server.listen()
        port = server.getsockname()[1]
    else:
        server = Listener(('127.0.0.1', 0), authkey=b'cancel-test-key')
        port = server.address[1]

    def silent_coordinator():
        connection = None
        try:
            if mode == 'authentication':
                connection, _ = server.accept()
            else:
                connection = server.accept()
                method, _ = connection.recv()
                assert method == 'publish_failure_log'
            requested.set()
            release.wait(15)
        finally:
            if connection is not None:
                connection.close()

    thread = threading.Thread(target=silent_coordinator, daemon=True)
    thread.start()
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    driver = subprocess.Popen([
        sys.executable, str(Path(__file__).with_name('_cancelled_generation_driver.py')),
        str(tmp_path), str(port), _STUB, mode,
    ], env=env)
    child_pids = []
    try:
        heartbeats = _wait_for_heartbeats(tmp_path, 1)
        child_pids = [int(pid) for pid in Path(
            f'/proc/{driver.pid}/task/{driver.pid}/children').read_text().split()]
        assert child_pids, 'stand-in simulator did not start'
        started = time.monotonic()
        driver.send_signal(signal.SIGTERM)
        driver.wait(timeout=_WORKER_CANCEL_TIMEOUT + 2)
        elapsed = time.monotonic() - started
        assert driver.returncode != 0
        assert elapsed < _WORKER_CANCEL_TIMEOUT + 2
        if mode != 'watchdog':
            assert elapsed < 3, 'failure RPC deadline did not release the worker'
        assert requested.wait(1), 'worker never attempted diagnostic publication'
        assert _no_growth_after(heartbeats)
        for pid in child_pids:
            assert not Path(f'/proc/{pid}').exists(), f'simulator child {pid} survived'
        assert not (tmp_path / 'output.log').exists()
        assert not (tmp_path / 'rawfile.raw').exists()
        assert list(tmp_path.glob('.silk-generation-*/output.log'))
    finally:
        if driver.poll() is None:
            driver.kill()
            driver.wait()
        release.set()
        server.close()
        thread.join(timeout=2)
