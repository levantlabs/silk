"""silk.simulation.executor — shared run executor.

One scheduler for every group run (corners, tests, sweeps, fault and
statistical campaigns): given a list of picklable :class:`RunTask` specs,
run each in its own folder and return a :class:`RunOutcome` per task, in
submission order. ``n_workers<=1`` runs serially in-process (the default,
and the easiest mode to debug); ``n_workers>1`` runs across a
:class:`multiprocessing.Pool`. A task whose ``fn`` raises never poisons the
run — its exception is captured as that task's own error and the rest
complete.

Modelled on :class:`silk.faults.campaign.FaultCampaign._execute` /
:func:`silk.faults.runner.run_one`, generalized: this module owns only the
scheduling, folder-per-task and Ctrl-C mechanics. Callers own what ``fn``
actually does (write a deck, invoke a simulator, evaluate a spec) — no
existing caller is switched to this executor by this task.

Ctrl-C (``KeyboardInterrupt``) must stop every worker AND every simulator
subprocess a worker started — a subprocess a worker forked is *not* killed
by ``Pool.terminate()`` (that only signals the worker processes themselves)
and would otherwise be orphaned, reparented, and keep running. A task that
launches a subprocess directly (via ``Popen``) should start it with
:func:`spawn_tracked` rather than bare :func:`subprocess.Popen`; a caller
that needs to keep using ``subprocess.run`` itself (see
:mod:`silk.simulation.runner`) can still get the PDEATHSIG half of the same
cleanup via :func:`linux_preexec_fn`. The subprocess is deliberately kept
IN the caller's process group (no ``start_new_session``) — an earlier
version of this module set that flag, which is wrong: it takes the child
OUT of the group, so anything that kills the job by process group
(``os.killpg``, a job scheduler, a terminal Ctrl-C that signals the
foreground group directly) would no longer reach it and would orphan it.
Cleanup instead layers three independent mechanisms, from most to least
targeted: (1) the launching process kills its own tracked children with
``proc.kill()`` (a single-process ``SIGKILL``, not a group kill) whenever a
``KeyboardInterrupt`` or other exception interrupts the wait — see each
worker's ``SIGTERM`` handler (installed by ``_worker_init``, run in turn
when ``run_tasks`` reacts to ``KeyboardInterrupt`` by calling
``pool.terminate()``, which ``SIGTERM``s every worker); (2) on Linux,
``preexec_fn`` sets ``PR_SET_PDEATHSIG=SIGKILL`` on the child (see
:func:`_die_with_parent`/:func:`linux_preexec_fn`), so it dies on its own
the instant the process that launched it dies for any reason a handler
can't catch (a worker ``SIGKILL``, an OOM kill) — this is also what
``subprocess.run``-based callers rely on instead of (1), since they never
get a Popen handle to track; (3) because the subprocess was never moved to
a new session, a signal or ``os.killpg`` aimed at the *whole group* (the
case (1)/(2) don't cover) reaches it directly, the same as it always would
have.

Per-worker memory cost: each pool worker is a separate OS process, and a
simulator invocation inside it (ngspice, Xyce) re-parses the PDK library
independently — so ``n_workers`` parsed PDKs are resident at once, not one
shared copy. Size ``n_workers`` with that in mind on a memory-constrained
box, not just by CPU count.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import sys
import threading
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

#: Linux ``prctl(2)`` opcode for PR_SET_PDEATHSIG. Not defined by the
#: stdlib; ``ctypes`` calls into libc's ``prctl`` directly.
_PR_SET_PDEATHSIG = 1

#: Resolved once, here, at module import time — in the parent, never in a
#: forked child. ``preexec_fn`` runs in the child between ``fork()`` and
#: ``exec()``, and the stdlib docs warn that it is unsafe in the presence
#: of threads: a fresh ``import`` there (``ctypes``, then ``ctypes.CDLL``,
#: which itself imports) can try to take an import lock that a *different*
#: thread of the pre-fork process held at the moment of the fork and will
#: now never release in the child — a classic fork-deadlock, and exactly
#: the failure mode a threaded caller (e.g. a web backend) would hit.
#: Resolving the library handle and the ``prctl`` function pointer here
#: means the child (:func:`_die_with_parent`) does no importing at all —
#: it only calls an already-bound C function. Best-effort: any failure to
#: resolve it (non-glibc libc, no ``ctypes`` support) leaves this ``None``,
#: and :func:`linux_preexec_fn` then returns ``None`` rather than install a
#: ``preexec_fn`` that would do nothing.
try:
    import ctypes as _ctypes
    _libc_prctl = _ctypes.CDLL('libc.so.6', use_errno=True).prctl
except Exception:
    _libc_prctl = None


def _die_with_parent():
    """``preexec_fn`` (Linux only): SIGKILL this child when ITS PARENT dies.

    Runs in the child, after ``fork()`` and before ``exec()``. Covers the
    cases the explicit ``proc.kill()`` calls below cannot: a worker that is
    itself ``SIGKILL``ed (uncatchable — its own ``SIGTERM`` handler never
    runs) or otherwise dies without a chance to clean up. Makes exactly one
    call, into the ``prctl`` function pointer :func:`_libc_prctl` already
    resolved at import time — no ``import`` and no ``ctypes.CDLL`` here,
    since either is unsafe this soon after a fork (see the module-level
    comment above ``_libc_prctl``). Still best-effort: the call itself is
    guarded in case the resolved pointer misbehaves for some other reason.
    """
    if _libc_prctl is None:
        return
    try:
        _libc_prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:
        pass


def linux_preexec_fn():
    """:func:`_die_with_parent` on Linux when ``prctl`` resolved, else ``None``.

    A ``subprocess.run``/``Popen`` caller launching a simulator should pass
    ``preexec_fn=linux_preexec_fn()`` directly — this is the whole PDEATHSIG
    half of :func:`spawn_tracked`'s cleanup, factored out so a caller that
    needs to keep calling bare ``subprocess.run`` (for test-mock
    compatibility, e.g. :class:`~silk.simulation.runner.NgspiceRunner`) gets
    it without going through :func:`spawn_tracked`.
    ``None`` is a valid no-op ``preexec_fn`` value on every platform.
    """
    if not sys.platform.startswith('linux') or _libc_prctl is None:
        return None
    return _die_with_parent


@dataclass
class RunTask:
    """One unit of work for :func:`run_tasks`.

    ``fn`` is invoked as ``fn(run_dir, *args, **kwargs)``. ``run_dir`` is
    created (``exist_ok=True``) before ``fn`` runs, so every task always
    gets its own folder whether it runs serially or in a pool worker. In
    pool mode, ``fn`` and everything in ``args``/``kwargs`` must be
    picklable (a module-level function, not a closure or lambda).
    """
    fn: Callable[..., Any]
    run_dir: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    # Optional simulator identity for parent-side advice before workers fork.
    simulator: Optional[str] = None


@dataclass
class RunOutcome:
    """Result of one :class:`RunTask`. Exactly one of ``value``/``error`` is set.

    ``index`` is the task's position in the list passed to :func:`run_tasks`
    — results are returned sorted by this field, so a pool (which completes
    tasks out of order) still yields results in submission order.
    """
    index: int
    run_dir: str
    value: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --- tracked-subprocess registry (per OS process) --------------------------
#
# Populated by spawn_tracked() in whichever process calls it — the serial
# path's single process, or a pool worker. _kill_tracked_children() is what
# both the worker SIGTERM handler and the serial KeyboardInterrupt handler
# call to reap what THIS process started.

_TRACKED_CHILDREN: List[subprocess.Popen] = []


def spawn_tracked(cmd, **kwargs) -> subprocess.Popen:
    """Start *cmd* IN THE CALLER'S process group and track it for cleanup.

    A task's ``fn`` should use this (instead of bare ``subprocess.Popen``)
    to launch a simulator. Deliberately does **not** set
    ``start_new_session`` — the child stays in the launching process's
    process group/session, exactly like any other subprocess, so a signal
    or ``os.killpg`` aimed at that group still reaches it directly. On
    Linux, ``preexec_fn`` also sets ``PR_SET_PDEATHSIG=SIGKILL`` (see
    :func:`_die_with_parent`) so the child cannot outlive whichever process
    called this. Tracked in a process-local registry so
    :func:`_kill_tracked_children` can ``proc.kill()`` it directly (a single
    process, not a group) on ``KeyboardInterrupt`` or a worker's
    ``SIGTERM``. A caller-supplied ``preexec_fn`` is chained after this one.

    PDEATHSIG is only installed when THIS call runs on its process's main
    thread. ``PR_SET_PDEATHSIG`` fires when the thread that
    called ``prctl`` exits, not when the process does — a child forked from
    a non-main thread would be ``SIGKILL``ed the instant that thread
    returns, even though the launching *process* (and every other thread in
    it, including whatever is waiting on the returned ``Popen``) is still
    alive. A pool worker's task always runs on that worker process's own
    main thread, so this never changes pool behaviour; a caller that spawns
    from a short-lived ``threading.Thread`` gets no PDEATHSIG protection for
    that child (the other two cleanup layers in the module docstring still
    apply), rather than a child silently killed out from under it.
    """
    if sys.platform.startswith('linux') and threading.current_thread() is threading.main_thread():
        caller_preexec = kwargs.get('preexec_fn')
        if caller_preexec is None:
            kwargs['preexec_fn'] = _die_with_parent
        else:
            def _chained(_caller=caller_preexec):
                _die_with_parent()
                _caller()
            kwargs['preexec_fn'] = _chained
    proc = subprocess.Popen(cmd, **kwargs)
    _TRACKED_CHILDREN.append(proc)
    return proc


def untrack(proc: subprocess.Popen) -> None:
    """Stop tracking *proc* (call once it has been waited on normally)."""
    try:
        _TRACKED_CHILDREN.remove(proc)
    except ValueError:
        pass


def _kill_tracked_children() -> None:
    """SIGKILL every subprocess THIS process tracked via :func:`spawn_tracked`.

    ``proc.kill()`` — a single-process signal, never ``os.killpg`` — since
    the child was deliberately left in the caller's own process group (see
    :func:`spawn_tracked`) rather than moved to a new one.
    """
    for proc in list(_TRACKED_CHILDREN):
        try:
            proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        untrack(proc)


_WORKER_CANCEL_TIMEOUT = 5


def _arm_worker_exit_watchdog():
    # A dead parent coordinator must never hold Pool.terminate() indefinitely.
    # Do not let repeated SIGTERMs extend the original exit deadline.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGALRM, lambda *_: os._exit(1))
    signal.alarm(_WORKER_CANCEL_TIMEOUT)


def _worker_sigterm_handler(signum, frame):
    _arm_worker_exit_watchdog()
    _kill_tracked_children()
    # Unwind runner cleanup so the cancelled attempt's log is published.
    raise SystemExit(1)


def _worker_init() -> None:
    """Pool worker initializer: Ctrl-C is handled by the parent process
    only (SIGINT ignored here), and a SIGTERM (as sent by
    ``Pool.terminate()``) kills this worker's tracked children first."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _worker_sigterm_handler)


def _run_task(task: RunTask, index: int) -> RunOutcome:
    os.makedirs(task.run_dir, exist_ok=True)
    try:
        value = task.fn(task.run_dir, *task.args, **task.kwargs)
        return RunOutcome(index=index, run_dir=task.run_dir, value=value)
    except Exception as e:
        err = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        return RunOutcome(index=index, run_dir=task.run_dir, error=err)


def _run_serially(dispatchable, results: List[Optional[RunOutcome]]) -> None:
    """Run every ``(index, task)`` pair in *dispatchable* in this process,
    writing each result into ``results[index]``.

    Shared by the true ``n_workers<=1`` path and the fork-unavailable pool
    fallback (see :func:`_pool_context`) so both get identical error-capture
    and ``KeyboardInterrupt`` behaviour: a task that raises becomes its own
    error, and Ctrl-C kills this process's tracked simulator subprocesses
    before re-raising.
    """
    try:
        for i, task in dispatchable:
            results[i] = _run_task(task, i)
    except KeyboardInterrupt:
        _kill_tracked_children()
        raise


def _pool_context():
    """Return an explicit ``'fork'`` multiprocessing context, or ``None``
    when the platform cannot fork.

    A device instance unpickles by looking up its class by
    NAME in the worker's own ``silk.elements.factory`` namespace, because
    ``BasePDK.__getstate__`` drops the per-device class table before
    pickling. That lookup only succeeds because a pool worker started with
    the ``'fork'`` start method is a copy-on-write clone of the parent
    process: the parent already ran every ``load_pdk()`` and registered
    those classes, and the fork inherits that registry for free. A worker
    started fresh instead (any other start method — ``'spawn'``,
    ``'forkserver'``) has an empty factory namespace, so the same name
    lookup would raise inside the worker for every composed-circuit task.

    :func:`run_tasks` therefore never relies on :func:`multiprocessing.Pool`
    picking a start method on its own: it asks for ``'fork'`` explicitly via
    :func:`multiprocessing.get_context`, and only when
    :func:`multiprocessing.get_all_start_methods` actually offers it on this
    platform. ``get_all_start_methods()`` reports every start method the
    platform *can* use, not just its *default* — macOS lists ``'fork'``
    even though ``'spawn'`` is its default since Python 3.8, and POSIX
    still lists ``'fork'`` even where CPython 3.14 makes ``'forkserver'``
    the default — so an explicit request for ``'fork'`` is what protects
    ``run_tasks`` from a non-fork *default* on those platforms; the plain
    ``mp.Pool()`` call this replaced would have picked up that default and
    broken there. Only a platform with no ``'fork'`` start method at all —
    Windows is the only one today — takes the fallback: the caller runs
    every task serially (see :func:`run_tasks`) rather than start a pool
    that would fail inside every worker.
    """
    if 'fork' in mp.get_all_start_methods():
        return mp.get_context('fork')
    return None


def run_tasks(tasks: Sequence[RunTask], n_workers: Optional[int] = 1) -> List[RunOutcome]:
    from silk.simulation.generation import worker_coordinator
    if n_workers is not None and n_workers > 1 and tasks:
        with worker_coordinator() as start_coordinator:
            outcomes = _run_tasks(tasks, n_workers, _start_coordinator=start_coordinator)
            from silk.simulation.generation import replacement_for
            for task, outcome in zip(tasks, outcomes):
                for arg in task.args:
                    attempt_id = getattr(arg, 'attempt_id', None)
                    if attempt_id is not None:
                        started = replacement_for(attempt_id)
                        outcome._replacement_started = started
            return outcomes
    return _run_tasks(tasks, n_workers)


def _run_tasks(tasks: Sequence[RunTask], n_workers: Optional[int] = 1, *, _start_coordinator=None) -> List[RunOutcome]:
    """Run every task in *tasks*, each in its own folder, and return the
    :class:`RunOutcome` list in submission order.

    ``n_workers`` here is the already-resolved worker count (see
    :func:`resolve_n_workers`): ``None`` or ``1`` runs serially in-process;
    any int ``> 1`` runs across a :class:`multiprocessing.Pool` started with
    an explicit ``'fork'`` context (see :func:`_pool_context` for why) —
    falling back to serial execution, with a ``RuntimeWarning`` naming the
    reason, on a platform where ``'fork'`` is unavailable, rather than
    starting a pool whose workers would fail every composed-circuit task. A
    task that raises is captured as that task's own error; the rest of the
    list still completes. ``KeyboardInterrupt`` terminates the pool (and,
    via each worker's SIGTERM handler, every simulator subprocess a worker
    started with :func:`spawn_tracked`) and re-raises.

    A task whose ``fn``/``args``/``kwargs`` cannot be pickled is never
    handed to the pool at all: it is pickled here, in this process, before
    dispatch, exactly as :class:`multiprocessing.pool.Pool` would pickle it
    to send to a worker. A failure there becomes that task's own
    :class:`RunOutcome` error (naming the unpicklable object, same as the
    underlying :class:`TypeError`/``AttributeError`` does) instead of
    escaping :func:`run_tasks` and aborting every other task — the pool's
    own send path (``pool.py``'s ``_handle_tasks`` thread) has no such
    per-task recovery, and a pickling failure there is fatal to the whole
    ``pool.apply_async`` batch.
    """
    indexed = list(enumerate(tasks))
    workers = 1 if n_workers is None else n_workers

    if workers <= 1 or not indexed:
        results: List[Optional[RunOutcome]] = [None] * len(indexed)
        _run_serially(indexed, results)
        return results

    results: List[Optional[RunOutcome]] = [None] * len(indexed)
    dispatchable = []
    for i, task in indexed:
        try:
            pickle.dumps((task, i))
        except Exception as e:
            results[i] = RunOutcome(
                index=i, run_dir=task.run_dir,
                error=(f'{type(e).__name__}: {e}\n'
                       f'Task at index {i} (run_dir={task.run_dir!r}) could '
                       f'not be pickled for the worker pool.\n'
                       f'{traceback.format_exc()}'))
        else:
            dispatchable.append((i, task))

    if not dispatchable:
        return results

    ctx = _pool_context()
    if ctx is None:
        warnings.warn(
            "silk.simulation.executor.run_tasks: no 'fork' multiprocessing "
            f"start method on this platform (available: "
            f"{mp.get_all_start_methods()!r}) -- falling back to serial "
            "execution instead of a worker pool. Pool workers rebuild PDK "
            "device instances by looking up their class by NAME in the "
            "worker's own silk.elements.factory namespace; that only works "
            "when the worker is a forked, copy-on-write clone of a parent "
            "that already ran load_pdk(). A worker started fresh (e.g. "
            "spawn, the only start method Windows offers) would fail every "
            "composed-circuit task instead.",
            RuntimeWarning, stacklevel=2)
        _run_serially(dispatchable, results)
        return results

    if any(task.simulator == 'ngspice' for _, task in dispatchable):
        from silk.simulation._ngspice_compat import warn_version_advice
        from silk.simulation.runner import ngspice_executable
        # Resolve and advise in the user's process. Forked workers inherit the
        # advised-binary record, avoiding repeats and pool.py attribution.
        warn_version_advice(ngspice_executable())

    pool = ctx.Pool(workers, initializer=_worker_init)
    if _start_coordinator is not None:
        _start_coordinator()
    try:
        async_results = [(i, pool.apply_async(_run_task, (task, i)))
                          for i, task in dispatchable]
        pool.close()
        for i, ar in async_results:
            results[i] = ar.get()
    except BaseException:
        # KeyboardInterrupt in particular: terminate() sends SIGTERM to
        # every worker, running _worker_sigterm_handler in each so tracked
        # simulator subprocesses are killed too, not just the workers.
        pool.terminate()
        pool.join()
        raise
    else:
        pool.join()
    return results


def resolve_n_workers(n_workers: Union[None, int, str] = None, backend=None) -> int:
    """Resolve a requested worker count to an actual pool size.

    - ``None`` or ``1`` -> ``1`` (serial, in-process).
    - an ``int N >= 1`` -> ``min(N, os.cpu_count())``; ``N < 1`` raises
      ``ValueError``.
    - ``'all'`` -> every CPU (``os.cpu_count()``).
    - *backend* naming (or a backend object whose ``.name`` is)
      ``'behavioral'`` forces ``1`` regardless of the requested
      ``n_workers``. Behavioral models may be non-picklable closures.

    Each additional worker is a full extra OS process, and a simulator
    invocation inside it (ngspice, Xyce) re-parses the PDK library on its
    own — so memory cost scales as roughly one parsed PDK per worker, not
    one shared copy. Pick ``n_workers`` with that in mind on a
    memory-constrained machine, not purely by CPU count.
    """
    backend_name = getattr(backend, 'name', backend)
    if backend_name == 'behavioral':
        return 1
    if n_workers is None:
        return 1
    if isinstance(n_workers, bool):
        raise ValueError(
            f"n_workers must be None, 1, an int >= 1, or 'all'; got {n_workers!r}"
        )
    if n_workers == 1:
        return 1
    cpu_count = os.cpu_count() or 1
    if n_workers == 'all':
        return cpu_count
    if not isinstance(n_workers, int):
        raise ValueError(
            f"n_workers must be None, 1, an int >= 1, or 'all'; got {n_workers!r}"
        )
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1; got {n_workers}")
    return min(n_workers, cpu_count)
