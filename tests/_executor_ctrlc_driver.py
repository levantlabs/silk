"""Driver script for test_run_executor's Ctrl-C test.

Runs in its own OS process (spawned via subprocess.Popen by the test) so
the test can send it a real SIGINT, the way a terminal Ctrl-C would reach a
foreground process. Each of its pool tasks launches a heartbeat-writing
child through ``spawn_tracked`` to stand in for a long-running simulator;
the test polls the heartbeat file for activity, signals this process, and
then confirms the heartbeat stops (the tracked child died with the pool,
rather than being orphaned and left running).
"""
import os
import signal
import sys

# Background launchers can pass SIG_IGN through exec; this driver must turn
# the test's SIGINT into KeyboardInterrupt regardless of its launch context.
signal.signal(signal.SIGINT, signal.default_int_handler)


def _heartbeat_task(run_dir):
    from silk.simulation.executor import spawn_tracked

    heartbeat = os.path.join(run_dir, 'heartbeat.txt')
    child_script = os.path.join(run_dir, 'child.py')
    with open(child_script, 'w') as f:
        f.write(
            "import time\n"
            "with open(%r, 'a') as hb:\n"
            "    for _ in range(600):\n"
            "        hb.write('x')\n"
            "        hb.flush()\n"
            "        time.sleep(0.05)\n" % heartbeat
        )
    proc = spawn_tracked([sys.executable, child_script])
    proc.wait()
    return proc.returncode


def main():
    from silk.simulation.executor import RunTask, run_tasks

    base = sys.argv[1]
    tasks = [
        RunTask(fn=_heartbeat_task, run_dir=os.path.join(base, f'task_{i}'))
        for i in range(2)
    ]
    run_tasks(tasks, n_workers=2)


if __name__ == '__main__':
    main()
