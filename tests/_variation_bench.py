"""Behavioural chip of offset adders for the silk.simulation.variation tests."""
from silk import PortDir, sources
from silk.circuit.modules import Module
from silk.metrics import final_value
from silk.simulation import Transient


def add_offset(ports, t, params):
    shift = params['shift'] if 'shift' in params else 0.0
    trim = params['trim'] if 'trim' in params else 0.0
    ports.y.v = ports.a.v + params.offset + shift + trim


def adder(pdk, tmp_path, params, *, mismatch=True, name='adder'):
    mod = Module('PDK_skywater130', str(tmp_path / name), f'{name}.cir',
                 pdk_details=pdk, module_name=name)
    mod.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT})
    mod.set_behavioral_model(add_offset, params=params, execution='vector')
    if not mismatch:
        mod.set_mismatch(False)
    return mod


def chip(pdk, tmp_path, params, *, n=3, names=None, mismatch=True,
         module_name='chip', analyses=None):
    """A chip placing *n* adders c0..c{n-1} (or *names*), one test 'offsets'
    whose metrics y0.. are the instances' final outputs."""
    names = names or [f'c{k}' for k in range(n)]
    block = adder(pdk, tmp_path, params, mismatch=mismatch)
    top = Module('PDK_skywater130', str(tmp_path / module_name),
                 f'{module_name}.cir', pdk_details=pdk,
                 module_name=module_name, db_path=str(tmp_path / 'runs.db'))
    ports = {'a': PortDir.IN, **{f'y{k}': PortDir.OUT for k in range(len(names))}}
    top.define_module_ports(ports)
    for k, inst in enumerate(names):
        top.add_module(block, inst, nets={'a': 'a', 'y': f'y{k}'})
    top.simulator = 'behavioral'
    tb = top.create_test('offsets')
    tb.add_module_under_test({p: p for p in ports})
    tb.add_source(sources.VoltageSource('input', positive='a', negative='gnd', dc=0.0))
    for analysis in analyses or [Transient(step=1e-9, stop=10e-9)]:
        tb.add_sim_type(analysis)
    for k in range(len(names)):
        tb.add_metric(f'y{k}', final_value, sig=tb.probe(f'y{k}'))
    return top, tb
