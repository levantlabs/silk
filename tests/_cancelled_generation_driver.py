"""Isolated worker for cancellation with a deliberately silent coordinator."""
import os
from pathlib import Path
import sys

import silk
from silk.simulation import generation
from silk.simulation.executor import _worker_init
from silk.simulation.run_files import RunFiles
from silk.simulation.runner import NgspiceRunner


def main():
    base, port, stub, mode = sys.argv[1:]
    print(silk.__file__, flush=True)
    os.environ['SILK_NGSPICE_BIN'] = stub
    files = RunFiles.resolve(str(Path(base) / 'deck.cir'), base, log=True)
    Path(files.netlist_path).write_text('* cancellation fixture\n.end\n')
    gen = generation.begin(files)
    # The launch uses a prepared generation; only failure cleanup needs RPC.
    generation._owner = os.getppid()
    generation._endpoint = (('127.0.0.1', int(port)), b'cancel-test-key')
    if mode == 'watchdog':
        # Deliberately defeat the RPC deadline to exercise the hard backstop.
        generation._FAILURE_RPC_TIMEOUT = 60
    _worker_init()
    runner = NgspiceRunner(gen.staged.netlist_path, base, run_files=gen.staged)
    runner._generation = gen
    runner._generation_managed = True
    runner.run()


if __name__ == '__main__':
    main()
