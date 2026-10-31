"""TASK-448.5: hierarchical behavioural evaluation, "a model is a leaf".

The engine walks the bench hierarchy. An
instance whose module has a behavioural model is one block, even when it also
holds transistors (a twin). An instance without one is entered, and its
internal nets take SPICE's spelling (``xdut.vcomp``). A transistor-level device
with no model above it refuses, naming its path.
"""
import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.examples.behavioral_example import build_top
from silk.exceptions import NotSupportedError, SimulationError
from silk.simulation import Transient
from silk.simulation.behavioral import BehavioralEngine
from silk.sources.classes import VoltageSource

_STOP = 120e-9
_STEP = 1e-9


def _module(pdk, tmp_path, name, *, module_name=None):
    return Module('PDK_skywater130', str(tmp_path / name), f'{name}.cir',
                  pdk_details=pdk, module_name=module_name)


def _bench(pdk, tmp_path):
    bench = _module(pdk, tmp_path, 'bench')
    bench.add_sim_type(Transient(step=_STEP, stop=_STOP))
    bench.simulator = 'behavioral'
    return bench


def _gain(pdk, tmp_path, gain, *, name='gain'):
    blk = _module(pdk, tmp_path, name, module_name=name)
    blk.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_1(ports, t, params):
        ports.vout.v = gain * ports.vin.v
    blk.set_behavioral_model(model_1, execution='vector')
    return blk


# AC #1 ----------------------------------------------------------------------

def test_composed_behavioral_example_runs_as_a_dut(pdk, tmp_path):
    bench = _bench(pdk, tmp_path)
    bench.add_module(build_top(), 'dut')
    bench.simulate()

    ref = build_top().simulate_behavioral(t_step=_STEP, t_stop=_STOP)
    for net in ('vin', 'vref', 'vcomp', 'vfilt'):
        np.testing.assert_array_equal(
            bench.load_signal(f'xdut.{net}').y, ref.load_signal(net).y)
    assert np.max(bench.load_signal('xdut.vcomp').y) == pytest.approx(1.8)
    # simulate() reads its rawfile, so names are spelled as in SPICE's
    # (TASK-448.3): the instance's nets in its own net order.
    assert bench.get_signal_names() == [
        'time', 'v(xdut.vin)', 'v(xdut.vref)', 'v(xdut.vcomp)',
        'v(xdut.vfilt)']


def test_composed_dut_through_its_registered_test(tmp_path):
    top = build_top()
    top.define_module_ports({})
    top.simulator = 'behavioral'
    tb = top.create_test('tran')
    tb.add_module_under_test({})
    tb.add_sim_type(Transient(step=_STEP, stop=_STOP))
    vcomp = tb.probe('xdut.vcomp')
    tb.run()
    assert np.max(vcomp.y) == pytest.approx(1.8)
    assert np.min(vcomp.y) == pytest.approx(0.0)


def test_ports_alias_to_the_parent_net_and_nesting_composes(pdk, tmp_path):
    """Two entered levels: dut -> stage -> gain. A port net is the parent's
    net (no vector of its own); an internal net is x-prefixed per level."""
    stage = _module(pdk, tmp_path, 'stage', module_name='stage')
    stage.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT})
    stage.add_module(_gain(pdk, tmp_path, 2.0, name='g2'), 'g1',
                     nets={'vin': 'a', 'vout': 'mid'})
    stage.add_module(_gain(pdk, tmp_path, 3.0, name='g3'), 'g2',
                     nets={'vin': 'mid', 'vout': 'y'})

    dut = _module(pdk, tmp_path, 'dut', module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut.add_module(stage, 'st', nets={'a': 'vin', 'y': 'vout'})

    bench = _bench(pdk, tmp_path)
    bench.add_source(VoltageSource('vs', positive='in', negative='gnd', dc=0.1))
    bench.add_module(dut, 'dut', nets={'vin': 'in', 'vout': 'out'})
    bench.simulate()

    np.testing.assert_allclose(bench.load_signal('out').y, 0.6)
    np.testing.assert_allclose(bench.load_signal('xdut.xst.mid').y, 0.2)
    names = bench.get_signal_names()
    assert names == ['time', 'v(in)', 'v(gnd)', 'v(out)', 'v(xdut.xst.mid)']
    # A port is not a vector of its own, as in a SPICE rawfile.
    with pytest.raises(SimulationError, match='not found'):
        bench.load_signal('xdut.vout')


def test_source_inside_an_entered_block_drives_its_internal_net(pdk, tmp_path):
    dut = _module(pdk, tmp_path, 'dut', module_name='dut')
    dut.define_module_ports({'vout': PortDir.OUT})
    dut.add_source(VoltageSource('vr', positive='ref', negative='gnd', dc=0.25))
    dut.add_module(_gain(pdk, tmp_path, 4.0), 'g1',
                   nets={'vin': 'ref', 'vout': 'vout'})

    bench = _bench(pdk, tmp_path)
    bench.add_module(dut, 'dut', nets={'vout': 'out'})
    bench.simulate()

    np.testing.assert_allclose(bench.load_signal('xdut.ref').y, 0.25)
    np.testing.assert_allclose(bench.load_signal('out').y, 1.0)


# AC #2 ----------------------------------------------------------------------

def test_twin_block_runs_its_model_not_its_transistors(pdk, tmp_path):
    twin = _module(pdk, tmp_path, 'twin', module_name='twin')
    twin.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    with twin:
        pdk.nmos1p8('m1', d='vout', g='vin', s='gnd', b='gnd', w=1.0, l=0.15)
    def model_2(ports, t, params):
        ports.vout.v = 1.8 - ports.vin.v
    twin.set_behavioral_model(model_2, execution='vector')

    bench = _bench(pdk, tmp_path)
    bench.add_source(VoltageSource('vs', positive='in', negative='gnd', dc=0.5))
    bench.add_module(twin, 'dut', nets={'vin': 'in', 'vout': 'out'})
    bench.simulate()

    np.testing.assert_allclose(bench.load_signal('out').y, 1.3)
    # The model is the leaf: nothing inside it is evaluated or listed.
    assert bench.get_signal_names() == ['time', 'v(in)', 'v(gnd)', 'v(out)']


def test_modelled_leaf_never_reads_back_a_copied_waveform(pdk, tmp_path):
    """add_module deep-copies a block with its nets' waveforms. A modelled
    leaf's internal net must not read back a value this run did not compute."""
    blk = _gain(pdk, tmp_path, 2.0)
    blk.add_net('internal')
    blk.net_dict['internal'].voltage = np.ones(5)
    dut = _module(pdk, tmp_path, 'dut', module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut.add_module(blk, 'g1', nets={'vin': 'vin', 'vout': 'vout'})

    bench = _bench(pdk, tmp_path)
    bench.add_source(VoltageSource('vs', positive='in', negative='gnd', dc=0.5))
    bench.add_module(dut, 'dut', nets={'vin': 'in', 'vout': 'out'})
    bench.simulate()

    with pytest.raises(NotSupportedError, match='internal nets'):
        bench.load_signal('xdut.xg1.internal')


# AC #3 ----------------------------------------------------------------------

def test_unmodelled_transistor_child_refuses_naming_its_path(pdk, tmp_path):
    dut = _module(pdk, tmp_path, 'dut', module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    with dut:
        pdk.nmos1p8('m1', d='vout', g='vin', s='gnd', b='gnd', w=1.0, l=0.15)

    bench = _bench(pdk, tmp_path)
    bench.add_source(VoltageSource('vs', positive='in', negative='gnd', dc=0.5))
    bench.add_module(dut, 'dut', nets={'vin': 'in', 'vout': 'out'})

    with pytest.raises(NotSupportedError) as excinfo:
        bench.simulate()
    msg = str(excinfo.value)
    assert "Instance 'xdut'" in msg
    assert "'xdut.m1'" in msg
    assert 'set_behavioral_model' in msg
    assert not bench.sim_results_loaded


def test_transistor_refusal_comes_before_any_model_runs(pdk, tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    blk = _module(pdk, tmp_path, 'spy', module_name='spy')
    blk.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_3(ports, t, params):
        calls.append(1)
        ports.vout.v = ports.vin.v
    blk.set_behavioral_model(model_3, execution='vector')

    amp = _module(pdk, tmp_path, 'amp', module_name='amp')
    amp.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    with amp:
        pdk.nmos1p8('mn', d='vout', g='vin', s='gnd', b='gnd', w=1.0, l=0.15)

    dut = _module(pdk, tmp_path, 'dut', module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut.add_module(blk, 'spy', nets={'vin': 'vin', 'vout': 'mid'})
    dut.add_module(amp, 'amp', nets={'vin': 'mid', 'vout': 'vout'})

    top = _module(pdk, tmp_path, 'top')
    top.add_source(VoltageSource('vs', positive='in', negative='gnd', dc=0.5))
    top.add_module(dut, 'dut', nets={'vin': 'in', 'vout': 'out'})

    with pytest.raises(NotSupportedError,
                       match=r"Instance 'xdut\.xamp'.*'xdut\.xamp\.mn'"):
        BehavioralEngine().run(top, t_step=_STEP, t_stop=_STOP)
    assert calls == []


def test_top_level_transistor_refuses_with_not_supported(pdk, tmp_path):
    top = _module(pdk, tmp_path, 'top')
    with top:
        pdk.nmos1p8('m1', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
    with pytest.raises(NotSupportedError, match=r"Element 'm1'.*top level"):
        BehavioralEngine().run(top, t_step=_STEP, t_stop=_STOP)


def test_empty_unmodelled_block_still_names_the_missing_model(pdk, tmp_path):
    empty = _module(pdk, tmp_path, 'empty', module_name='empty')
    empty.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut = _module(pdk, tmp_path, 'dut', module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut.add_module(empty, 'e1', nets={'vin': 'vin', 'vout': 'vout'})
    top = _module(pdk, tmp_path, 'top')
    top.add_module(dut, 'dut', nets={'vin': 'in', 'vout': 'out'})
    with pytest.raises(SimulationError, match=r"Block 'xdut\.e1'.*no behavioral model"):
        BehavioralEngine().run(top, t_step=_STEP, t_stop=_STOP)
