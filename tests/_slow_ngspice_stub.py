#!/usr/bin/env python3
"""Stand-in for the ngspice binary (TASK-301.3), pointed to via
$SILK_NGSPICE_BIN. Invoked exactly like real ngspice
(``stub -b -r <rawfile> -o <logfile> <netlist>``); instead of simulating,
writes a growing heartbeat file next to the requested rawfile so a test can
tell it is alive, and keeps writing until it is killed (never exits on its
own within the test's patience) -- standing in for a long-running real
simulation for process-group / Ctrl-C / PDEATHSIG acceptance tests that
don't need real SPICE evidence, only a real, killable OS process launched
exactly the way NgspiceRunner launches one.
"""
import os
import sys
import time


def main():
    args = sys.argv[1:]
    rawfile = args[args.index('-r') + 1]
    heartbeat = rawfile + '.heartbeat'
    for _ in range(600):
        with open(heartbeat, 'a') as f:
            f.write('x')
            f.flush()
        time.sleep(0.05)


if __name__ == '__main__':
    main()
