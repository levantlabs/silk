"""A documented-form probe metric on a block placed with ``add_module``
measures the COPY's run (TASK-354).

``add_module`` stores a deep copy of the block. A metric registered as
``tb.add_metric(name, fn, sig=probe)`` is rebound to the copied bench, so it
reads the run the parent drove. A hand-written wrapper that calls
``probe.resolve()`` closes over the ORIGINAL bench, which never runs, and
reports "bench ... has not run yet"; that form is not the documented one.
"""
import numpy as np
import pytest

from silk import Module, Port, load_pdk, sources
from silk.elements import elements
from silk.metrics.statistics import average
from silk.simulation import Transient

from tests._env import needs_ngspice


def _rc(tmp_path):
    rc = Module('PDK_skywater130', str(tmp_path), 'rc.cir',
                pdk_details=load_pdk('sky130'), module_name='rc')
    rc.define_module_ports({'a': Port.analog_input(), 'b': Port.analog_output()})
    with rc:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    tb = rc.create_test('step')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0,
        pulse=sources.Pulse(v1=0, v2=1, td=0, tr=10e-9, tf=10e-9,
                            pw=5e-6, per=10e-6)))
    tb.add_sim_type(Transient(step=10e-9, stop=10e-6))
    vout = tb.probe('vout')
    tb.add_metric('out_avg', average, sig=vout)
    tb.add_metric('out_max', _peak, sig=vout)
    return rc


def _peak(sig):
    return float(np.max(np.asarray(sig.y, dtype=float)))


@needs_ngspice
def test_probe_metrics_on_placed_block_measure_the_copy(tmp_path):
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir',
                 pdk_details=load_pdk('sky130'), module_name='top')
    top.define_module_ports({'a': Port.analog_input(), 'b': Port.analog_output()})
    top.add_module(_rc(tmp_path), 'u_rc', nets={'a': 'a', 'b': 'b'})

    report_metrics = top.run_all_tests(corner='tt')

    metrics = report_metrics['rc.step'].metrics
    assert metrics['out_avg'] == pytest.approx(0.5, abs=0.01)
    assert metrics['out_max'] == pytest.approx(0.99, abs=0.02)
