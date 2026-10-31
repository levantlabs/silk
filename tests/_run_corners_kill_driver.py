"""Driver script for TASK-301.3's process-group/Ctrl-C acceptance tests.

Runs in its own OS process (spawned by the test) so a real SIGKILL of the
whole process group, or a real SIGINT to just this process, can be sent the
way a job scheduler or a terminal Ctrl-C would. Builds a PDK-free RC
testbench, points $SILK_NGSPICE_BIN at tests/_slow_ngspice_stub.py (a
heartbeat-writing stand-in that never exits on its own), and drives it
through ``run()`` (mode 'serial') or ``run_corners(n_workers=2)`` (mode
'corners') -- proving the real launch path (NgspiceRunner.run's
subprocess.run(preexec_fn=linux_preexec_fn())), not the generic executor's
own spawn_tracked (already covered by _executor_ctrlc_driver.py).
"""
import os
import signal
import sys

# Background launchers can pass SIG_IGN through exec; this driver must turn
# the test's SIGINT into KeyboardInterrupt regardless of its launch context.
signal.signal(signal.SIGINT, signal.default_int_handler)


def _make_bench(tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.elements import passives
    from silk import sources
    from silk.simulation import analysis
    from silk.pdk.basepdk import BasePDK

    stub_lib = os.path.join(tmp_dir, 'stub.lib.spice')
    with open(stub_lib, 'w') as f:
        f.write('* stub\n')
    pdk_yaml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'silk', 'pdk', 'data', 'skywater_130_device_details.yaml')
    pdk = BasePDK(pdk_yaml, library_location=stub_lib)
    dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=pdk, module_name='rc')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        passives.Resistor('r1', pos='a', neg='b', resistance=1e3)
        passives.Capacitor('c1', pos='b', neg='0', capacitance=1e-9)
    test = dut.create_test('rc')
    test.add_module_under_test({'a': 'vin', 'b': 'vout'})
    test.add_source(sources.VoltageSource('vs', positive='vin', negative='0', dc=1.0))
    test.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))
    return test


def main():
    mode = sys.argv[1]
    base = sys.argv[2]
    stub = sys.argv[3]
    os.environ['SILK_NGSPICE_BIN'] = stub
    test = _make_bench(base)
    if mode == 'serial':
        test.run()
    elif mode == 'corners':
        test.run_corners(['tt', 'ff'], n_workers=2)
    else:
        raise ValueError(f'unknown mode {mode!r}')


if __name__ == '__main__':
    main()
