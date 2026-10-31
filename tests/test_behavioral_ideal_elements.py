"""Owner D4: ideal elements around behavioral models have explicit meanings."""
import warnings

import numpy as np
import pytest

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk.elements import elements
from silk.exceptions import NotSupportedError
from silk.simulation import Transient
from silk.sources.classes import (
    VoltageSource, VCVS, VoltageControlledCurrentSource,
    CurrentControlledCurrentSource, CurrentControlledVoltageSource,
    BehavioralVoltage, BehavioralCurrent,
)


def bench(pdk, tmp_path, *, constant=False):
    mod = Module('sky130', str(tmp_path), 'ideal.cir', pdk_details=pdk)
    mod.simulator = 'behavioral'
    mod.add_sim_type(Transient(step=1e-9, stop=4e-9))
    mod.add_source(VoltageSource('input', positive='vin', negative='gnd',
                                 dc=.7 if constant else 0,
                                 pwl=None if constant else [(0, 0.1), (4e-9, 0.9)]))
    return mod


@pytest.mark.parametrize('kind', ['resistor', 'capacitor'])
@pytest.mark.parametrize('rail', ['gnd', 'vdd', 'configured'])
def test_load_preserves_driver(pdk, tmp_path, kind, rail):
    mod = bench(pdk, tmp_path)
    if rail != 'gnd':
        mod.add_source(VoltageSource('supply', positive=rail, negative='0', dc=1.8))
        mod.set_supply_voltage(1.8, source='supply' if rail == 'vdd' else None)
    mod.add(getattr(elements, kind)('load', p='vin', n=rail, value=1))
    with pytest.warns(UserWarning, match='load.*ignored') as caught:
        mod.simulate()
    assert len(caught) == 1
    np.testing.assert_allclose(mod.load_signal('vin').y, np.linspace(.1, .9, 5))


@pytest.mark.parametrize('kind', ['resistor', 'inductor'])
@pytest.mark.parametrize('reverse', [False, True])
def test_series_path_drives_far_net(pdk, tmp_path, kind, reverse):
    mod = bench(pdk, tmp_path)
    a, b = ('far', 'vin') if reverse else ('vin', 'far')
    mod.add(getattr(elements, kind)('series', p=a, n=b, value=1))
    mod.add(elements.capacitor('load', p='far', n='gnd', value=1))
    with pytest.warns(UserWarning, match='series.*wire') as caught:
        mod.simulate()
    assert len(caught) == 1
    np.testing.assert_array_equal(mod.load_signal('far').y, mod.load_signal('vin').y)


def test_divider_refuses_before_models(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    mod.add(elements.resistor('upper', p='vin', n='mid', value=1))
    mod.add(elements.resistor('lower', p='mid', n='gnd', value=1))
    with pytest.raises(NotSupportedError, match='upper.*divider'):
        mod.simulate()


def test_series_capacitor_refuses(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    mod.add(elements.capacitor('series_cap', p='vin', n='far', value=1))
    with pytest.raises(NotSupportedError, match='series_cap.*series C'):
        mod.simulate()


@pytest.mark.parametrize('source', [
    lambda: VoltageControlledCurrentSource('bad', 'out', 'gnd', 'vin', 'gnd', 2),
    lambda: CurrentControlledCurrentSource('bad', 'out', 'gnd', 2, 'input'),
    lambda: CurrentControlledVoltageSource('bad', 'out', 'gnd', 2, 'input'),
    lambda: BehavioralVoltage('bad', 'out', 'gnd', 'V(vin)'),
    lambda: BehavioralCurrent('bad', 'out', 'gnd', 'V(vin)'),
])
def test_unsupported_sources_refuse(pdk, tmp_path, source):
    mod = bench(pdk, tmp_path)
    mod.add_source(source())
    with pytest.raises(NotSupportedError, match='bad.*behavioral'):
        mod.simulate()


def test_vcvs_orders_controls_and_output_reference(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    # Deliberately reverse insertion order. Nonzero output reference matters.
    mod.add_source(VCVS('second', 'out', 'vin', 'mid', 'vin', 3))
    mod.add_source(VCVS('first', 'mid', 'gnd', 'vin', 'gnd', 2))
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 4 * mod.load_signal('vin').y)


def test_entered_passives_and_model_leaf(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    child = Module('sky130', str(tmp_path / 'child'), 'child.cir',
                   pdk_details=pdk, module_name='child')
    child.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    child.add_passive(elements.resistor('link', p='a', n='z', value=1))
    mod.add_module(child, 'dut', nets={'a': 'vin', 'z': 'out'})
    with pytest.warns(UserWarning, match='xdut.link.*wire'):
        mod.simulate()
    np.testing.assert_array_equal(mod.load_signal('out').y, mod.load_signal('vin').y)
    def model_1(ports, t, params):
        ports.z.v = 2 * ports.a.v
    child.set_behavioral_model(model_1, execution='vector')
    leaf = bench(pdk, tmp_path / 'leaf')
    leaf.add_module(child, 'dut', nets={'a': 'vin', 'z': 'out'})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UserWarning)
        leaf.simulate()
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    np.testing.assert_allclose(leaf.load_signal('out').y, 2 * leaf.load_signal('vin').y)


def test_dc_input_is_not_automatically_a_supply(pdk, tmp_path):
    mod = bench(pdk, tmp_path, constant=True)
    mod.add(elements.resistor('series', p='vin', n='far', value=1))
    with pytest.warns(UserWarning, match='wire'):
        mod.simulate()
    np.testing.assert_allclose(mod.load_signal('far').y, .7)


def test_report_snapshots_notices_and_rawfile_reload(pdk, tmp_path):
    dut = Module('sky130', str(tmp_path / 'dut'), 'dut.cir',
                 pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'out': PortDir.OUT})
    def model_2(ports, t, params):
        ports.out.v = 2 * ports.vin.v
    dut.set_behavioral_model(model_2, execution='vector')
    tb = dut.create_test('ideal')
    tb.add_module_under_test({'vin': 'vin', 'out': 'out'})
    tb.netlist.add_source(VoltageSource('input', 'vin', 'gnd', dc=.4))
    tb.add_sim_type(Transient(step=1e-9, stop=4e-9))
    tb.netlist.simulator = 'behavioral'
    tb.netlist.add(elements.capacitor('load', p='out', n='gnd', value=1))
    for _ in range(2):
        with pytest.warns(UserWarning, match='load.*ignored') as caught:
            tb.netlist.simulate()
        assert len(caught) == 1
        report = tb.evaluate()
        assert report.summary().count('load: ignored') == 1
    path = tb.netlist.results_file
    tb.netlist._behavioral_element_notes = ()
    reloaded = tb.evaluate(rawfile=path)
    assert reloaded.summary().count('load: ignored') == 1
    assert report.summary().count('load: ignored') == 1


def test_series_chain_orders_models(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    for name, a, b in [('second', 'mid', 'far'), ('first', 'near', 'mid')]:
        mod.add(elements.inductor(name, p=a, n=b, value=1))
    child = Module('sky130', str(tmp_path / 'gain'), 'gain.cir',
                   pdk_details=pdk, module_name='gain')
    child.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    def model_3(ports, t, params):
        ports.z.v = 2 * ports.a.v
    child.set_behavioral_model(model_3, execution='vector')
    mod.add_module(child, 'consumer', nets={'a': 'far', 'z': 'out'})
    mod.add_module(child, 'driver', nets={'a': 'vin', 'z': 'near'})
    with pytest.warns(UserWarning) as caught:
        mod.simulate()
    assert len(caught) == 1
    for name in ('first', 'second'):
        assert str(caught[0].message).count(name + ': wire') == 1
    np.testing.assert_allclose(mod.load_signal('out').y, 4 * mod.load_signal('vin').y)


def test_vcvs_variable_gain(pdk, tmp_path):
    from silk.circuit.variable import Variable
    mod = bench(pdk, tmp_path)
    gain = Variable(name='gain', value=3)
    mod.add_source(VCVS('amp', 'out', 'gnd', 'vin', 'gnd', gain))
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 3 * mod.load_signal('vin').y)


def test_vcvs_reversed_output_preserves_ground(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    mod.add_source(VCVS('amp', 'gnd', 'out', 'vin', 'gnd', 2))
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('gnd').y, 0)
    np.testing.assert_allclose(mod.load_signal('out').y, -2 * mod.load_signal('vin').y)


def test_source_and_passive_can_share_instance_name(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    mod.add_source(VCVS('link', 'mid', 'gnd', 'vin', 'gnd', 2))
    mod.add(elements.resistor('link', p='mid', n='out', value=1))
    with pytest.warns(UserWarning, match='link.*wire'):
        mod.simulate()
    np.testing.assert_allclose(mod.load_signal('out').y, 2 * mod.load_signal('vin').y)
    np.testing.assert_array_equal(mod.load_signal('mid').y, mod.load_signal('out').y)


def test_vcvs_refuses_conflicting_output_drivers(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    mod.add_source(VCVS('conflict', 'vin', 'gnd', 'vin', 'gnd', 2))
    with pytest.raises(NotSupportedError, match='conflict.*driven'):
        mod.simulate()


def test_vcvs_self_feedback_refuses(pdk, tmp_path):
    from silk.exceptions import SimulationError
    mod = bench(pdk, tmp_path)
    mod.add_source(VCVS('feedback', 'out', 'gnd', 'out', 'gnd', 2))
    with pytest.raises(SimulationError, match='Circular dependency.*feedback'):
        mod.simulate()


def test_two_passive_registries_keep_distinct_elements(pdk, tmp_path):
    mod = bench(pdk, tmp_path)
    mod.add_source(VoltageSource('other', 'other', 'gnd', dc=.7))
    mod.add_passive(elements.resistor('link', p='vin', n='first', value=1))
    mod.add(elements.inductor('link', p='other', n='second', value=1))
    with pytest.warns(UserWarning) as caught:
        mod.simulate()
    assert len(caught) == 1
    message = str(caught[0].message)
    assert message.count('link: wire (R,') == 1
    assert message.count('link: wire (L,') == 1
    np.testing.assert_array_equal(mod.load_signal('first').y, mod.load_signal('vin').y)
    np.testing.assert_allclose(mod.load_signal('second').y, .7)


def test_notices_warn_once_per_run_across_analyses(pdk, tmp_path):
    """Merge fix (448.11 x 448.12): a run with a transient and a DC sweep
    evaluates the topology twice but lists its approximations once (D4)."""
    from silk.simulation import DC
    bench = Module('PDK_skywater130', str(tmp_path / 'b'), 'b.cir',
                   pdk_details=pdk)
    bench.add_source(VoltageSource('vs', positive='in', negative='gnd', dc=0.5))
    bench.add(elements.resistor('rs', p='in', n='out', value=1e3))
    bench.add_sim_type(Transient(step=1e-9, stop=5e-9))
    bench.add_sim_type(DC(source='vs', start=0, stop=1, step=0.5))
    bench.simulator = 'behavioral'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        bench.simulate()
    notices = [w for w in caught if 'ideal-element approximations' in str(w.message)]
    assert len(notices) == 1
    np.testing.assert_allclose(bench.load_signal('out', analysis='dc').y,
                               [0.0, 0.5, 1.0])
