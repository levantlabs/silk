"""TASK-448.6: design section 5.2, one bench and two engines."""
import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import NotSupportedError, SimulationError
from silk.signals.rawfile import read_rawfile
from silk.simulation import Transient
from silk.sources import CurrentSource, VoltageSource
from silk.sources.waveforms import PWL
from tests._env import needs_ngspice, needs_pdk


def _bench(tmp_path, pdk, backend, *, leaf_sources=(), dut_sources=(), current_wave=None):
    leaf = Module('PDK_skywater130', netlistname='probe.cir', pdk_details=pdk, dir_folder=str(tmp_path / 'leaf'),
                  module_name='leaf')
    leaf.define_module_ports({'out': PortDir.OUT})
    leaf.add_source(VoltageSource('vo', positive='out', negative='gnd', dc=0.4))
    leaf.add_source(VoltageSource('vn', positive='n', negative='gnd', dc=0.8))
    with leaf:
        pdk.nmos1p8('m1', d='n', g='n', s='gnd', b='gnd', w=1, l=0.15)
    def model_1(ports, t, params):
        ports.out.v = 0.4 + 0 * t
    leaf.set_behavioral_model(model_1, execution='vector')
    for source in leaf_sources:
        leaf.add_source(source)
    dut = Module('PDK_skywater130', netlistname='probe.cir', pdk_details=pdk, dir_folder=str(tmp_path / 'dut'),
                 module_name='dut')
    dut.define_module_ports({'out': PortDir.OUT})
    dut.add_module(leaf, 'leaf', nets={'out': 'out'})
    dut.add_source(VoltageSource('vn', positive='n', negative='gnd', dc=0.3))
    for source in dut_sources:
        dut.add_source(source)
    tb = dut.create_test('probes')
    tb.add_module_under_test({'out': 'vout'})
    tb.add_source(VoltageSource('s', positive='vin', negative='gnd', dc=0.2))
    tb.add_source(CurrentSource('b', positive='bias', negative='gnd', dc=1e-3, pwl=current_wave))
    tb.netlist.add(elements.resistor('rb', p='bias', n='gnd', value=1000))
    tb.netlist.add(elements.resistor('rs', p='vin', n='gnd', value=1000))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.netlist.simulator = backend
    return tb


# Every row of section 5.2; device rows include both probe families and op_table.
_ROWS = [
    ('node', 'vout', 'V', 0.4, None),
    ('entered', 'xdut.n', 'V', 0.3, None),
    ('leaf', 'xdut.xleaf.n', 'V', None, 'internal nets'),
    ('node-current', 'vout', 'I', None, 'not found'),
    ('source-current', 'ib', 'I', 1e-3, None),
    ('voltage-current', 'vs', 'I', None, 'electrical solve'),
    ('wrapped-current', 'i(vs)', 'V', None, 'electrical solve'),
    ('terminal', 'xdut.xleaf.m1:d', 'V', None, 'device'),
    ('parameter', 'leaf.m1:gm', 'V', None, 'device'),
    ('op-table', None, 'V', None, 'operating-point'),
    ('arithmetic', None, 'V', 0.3, None),
]


@pytest.mark.parametrize('row,name,selector,expected,reason', _ROWS,
                         ids=[r[0] for r in _ROWS])
@pytest.mark.parametrize('shortcut', [False, True])
def test_behavioral_probe_table(tmp_path, pdk, row, name, selector, expected, reason, shortcut):
    tb = _bench(tmp_path, pdk, 'behavioral')
    # Device probes may refuse at declaration, before SPICE-only validation.
    if row in ('terminal', 'parameter'):
        with pytest.raises(NotSupportedError, match=reason):
            tb.probe(name, selector)
        return
    probe = ((tb.probe('vout') + tb.probe('vin')) / 2
             if row == 'arithmetic' else
             tb.probe(name, selector) if name else None)
    if shortcut:
        tb.netlist.simulate_behavioral(1e-9, 5e-9)
    else:
        tb.netlist.simulate()
    if row == 'op-table':
        with pytest.raises(NotSupportedError, match=reason):
            tb.evaluate().op_table()
    elif reason:
        error = SimulationError if row == 'node-current' else NotSupportedError
        with pytest.raises(error, match=reason):
            probe.y
    else:
        np.testing.assert_allclose(probe.y, expected)
        assert probe.type == ('current' if selector == 'I' else 'voltage')


def test_source_current_is_its_own_waveform_and_roundtrips(tmp_path, pdk):
    tb = _bench(tmp_path, pdk, 'behavioral', current_wave=PWL([(0, 0), (5e-9, 2e-3)]))
    # A voltage-producing model and a current branch share connectivity only.
    writer = Module('PDK_skywater130', netlistname='probe.cir', pdk_details=pdk,
                    dir_folder=str(tmp_path / 'writer'), module_name='writer')
    writer.define_module_ports({'out': PortDir.OUT})
    def model_2(ports, t, params):
        ports.out.v = 9 + 0 * t
    writer.set_behavioral_model(model_2, execution='vector')
    tb.netlist.add_module(writer, 'writer', nets={'out': 'bias'})
    tb.netlist.sim_analysis_dict['transient'] = Transient(
        step=1e-9, stop=5e-9, start=2e-9)
    sig = tb.probe('i(ib)')
    tb.netlist.simulate()
    np.testing.assert_allclose(sig.y, 1e-3 + 4e5*sig.x)
    np.testing.assert_allclose(tb.probe('bias').y, 9)
    assert sig.x[0] == pytest.approx(2e-9)
    vectors = {v.name: v for v in read_rawfile(tb.netlist.results_file).vectors}
    assert vectors['i(ib)'].type == 'current'
    np.testing.assert_array_equal(vectors['i(ib)'].data, sig.y)


@pytest.mark.parametrize('name,selector,reason', [
    ('XDUT.XLEAF.N', 'V', 'internal nets'),
    ('vs', 'I', 'electrical solve'),
    ('leaf.m1:gm', 'V', 'device'),
    ('xdut.xleaf.m1:d', 'V', 'device'),
])
def test_direct_load_refuses_too(tmp_path, pdk, name, selector, reason):
    tb = _bench(tmp_path, pdk, 'behavioral')
    tb.netlist.simulate()
    with pytest.raises(NotSupportedError, match=reason):
        tb.netlist.load_signal(name, selector)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('row,name,selector,expected', [
    ('node', 'vout', 'V', 0.4),
    ('entered', 'xdut.n', 'V', 0.3),
    ('leaf', 'xdut.xleaf.n', 'V', 0.8),
    ('node-current', 'vout', 'I', None),
    ('voltage-current', 'vs', 'I', -2e-4),
    ('wrapped-current', 'i(vs)', 'V', -2e-4),
    ('arithmetic', None, 'V', 0.3),
])
def test_ngspice_probe_table(tmp_path, row, name, selector, expected):
    from silk.pdk import load_pdk
    tb = _bench(tmp_path, load_pdk('sky130'), 'ngspice')
    sig = ((tb.probe('vout') + tb.probe('vin')) / 2 if row == 'arithmetic'
           else tb.probe(name, selector))
    tb.run()
    if row == 'node-current':
        with pytest.raises(SimulationError, match='not found'):
            sig.y
    else:
        np.testing.assert_allclose(sig.y, expected, atol=1e-10)


@pytest.mark.parametrize('name', ['xdut.ib', 'I(XDUT.IB)'])
def test_entered_current_source_and_ignored_leaf_sources(tmp_path, pdk, name):
    tb = _bench(tmp_path, pdk, 'behavioral',
                dut_sources=(CurrentSource('b', positive='bias', negative='gnd', dc=2e-3),),
                leaf_sources=(CurrentSource('hidden', positive='n', negative='gnd', dc=7),))
    sig = tb.probe(name, 'I')
    tb.netlist.simulate()
    np.testing.assert_allclose(sig.y, 2e-3)
    assert not any('hidden' in n for n in tb.netlist.get_signal_names())
    with pytest.raises(NotSupportedError, match='internal nets'):
        tb.probe('xdut.xleaf.ihidden', 'I').y


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('row', ['parameter', 'terminal', 'op-table', 'source-current'])
def test_ngspice_device_and_source_rows(tmp_path, row):
    from silk.pdk import load_pdk
    tb = _bench(tmp_path, load_pdk('sky130'), 'ngspice')
    if row == 'parameter':
        sig = tb.probe('leaf.m1:gm')
    elif row == 'terminal':
        tb.module.save_device_currents = True
        tb.module.terminal_current_method = 'ammeter_wrapper'
        sig = tb.probe('xdut.xleaf.m1:d')
    elif row == 'op-table':
        tb.module.save_device_params = True
    else:
        sig = tb.probe('ib', 'I')
    report = tb.run()
    if row == 'op-table':
        assert report.op_table()['dut.leaf.m1'].gm > 0
    elif row == 'source-current':
        # Explicit source-current saves expose the measured constant current.
        np.testing.assert_allclose(sig.y, 1e-3)
    else:
        assert np.all(np.isfinite(sig.y))
        assert np.all(sig.y > 0)


def test_inherited_backend_refuses_devices_before_netlist_exists(tmp_path, pdk):
    dut = Module('PDK_skywater130', str(tmp_path), 'dut.cir', pdk_details=pdk)
    dut.simulator = 'behavioral'
    tb = dut.create_test('lazy')
    assert tb.netlist is None
    assert tb.probe('out').name == 'out'
    with pytest.raises(NotSupportedError, match='device'):
        tb.probe('m1:gm')
    with pytest.raises(ValueError, match='suffix'):
        tb.probe('m1:bad')


@pytest.mark.parametrize('nested', [False, True])
def test_current_source_name_can_contain_an_index(tmp_path, pdk, nested):
    source = CurrentSource('bias[0]', positive='extra', negative='gnd', dc=2e-3)
    tb = _bench(tmp_path, pdk, 'behavioral', dut_sources=(source,) if nested else ())
    if not nested:
        tb.netlist.add_source(source)
    name = 'xdut.ibias[0]' if nested else 'ibias[0]'
    sig = tb.probe(name, 'I')
    tb.netlist.simulate()
    np.testing.assert_allclose(sig.y, 2e-3)


def test_colon_net_is_refused(tmp_path, pdk):
    with pytest.raises(ValueError, match='hierarchy levels'):
        VoltageSource('colon', positive='bus:foo', negative='gnd', dc=0.7)


def test_behavioral_refusals_follow_loaded_evidence(tmp_path, pdk):
    tb = _bench(tmp_path, pdk, 'behavioral')
    sig = tb.probe('vs', 'I')
    tb.netlist.simulate()
    tb.netlist.simulator = 'ngspice'
    with pytest.raises(NotSupportedError, match='electrical solve'):
        sig.y


def test_spice_result_stays_readable_when_next_backend_changes(tmp_path, pdk):
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
    tb = _bench(tmp_path, pdk, 'ngspice')
    path = tmp_path / 'spice.raw'
    raw = RawFile([RawPlot({'plotname': 'Transient Analysis', 'command': 'ngspice'},
        [RawVector('time', 'time', data=np.array([0., 1e-9])),
         RawVector('i(vs)', 'current', data=np.array([-2e-4, -2e-4]))],
        npoints=2, real=True, padded=True)])
    write_rawfile(path, raw)
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    tb.netlist.simulator = 'behavioral'
    np.testing.assert_allclose(tb.probe('vs', 'I').y, -2e-4)


def test_shortcut_restores_the_selection_and_refuses_from_evidence(tmp_path, pdk):
    """TASK-448.10 review: simulate_behavioral() must not leave the module on
    the behavioural backend, or a later SPICE run silently runs behaviourally
    (abo_th_example and flash_adc_example printed behavioural numbers under
    their SPICE headings). Refusals follow the loaded behavioural result."""
    tb = _bench(tmp_path, pdk, 'ngspice')
    before = (tb.netlist.simulator, tb.netlist._simulator_explicit)
    tb.netlist.simulate_behavioral(1e-9, 5e-9)
    assert (tb.netlist.simulator, tb.netlist._simulator_explicit) == before
    assert tb.netlist.backend.name == 'ngspice'
    np.testing.assert_allclose(tb.probe('vout').y, 0.4)
    with pytest.raises(NotSupportedError, match='device'):
        tb.probe('leaf.m1:gm')
    with pytest.raises(NotSupportedError, match='electrical solve'):
        tb.probe('vs', 'I').y


def test_abo_registered_sfdr_records_behavioral_evidence(tmp_path, monkeypatch):
    from silk.examples.abo_th_example import make_th_testbench, T_STEP, T_STOP

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    test, _ = make_th_testbench(return_test=True)
    test.netlist.simulate_behavioral(T_STEP, T_STOP)
    report = test.evaluate()
    assert report.operating_conditions.backend == 'behavioral'
    assert report.passed
    assert set(report.metrics) == {'sfdr_track_end', 'sfdr_hold_end'}
    outcomes = test.module.reconcile_specs().outcomes
    assert len(outcomes) == 2
    for outcome in outcomes:
        assert outcome.measurement.source == 'behavioral'
        assert outcome.spec.latest(source='spice') is None
        assert outcome.measurement.value == report.metrics[outcome.spec.metric]
    # A subsequent direct SPICE run must still use the original backend.
    assert test.netlist.simulator == 'ngspice'


def test_current_branch_can_share_a_declared_voltage_output(tmp_path, pdk):
    tb = _bench(tmp_path, pdk, 'behavioral')
    writer = Module('sky130', str(tmp_path / 'writer'), 'writer.cir',
                    pdk_details=pdk, module_name='writer')
    writer.define_module_ports({'out': PortDir.OUT})
    def model_3(ports, t, params):
        ports.out.v = 9 + 0 * t
    writer.set_behavioral_model(model_3, execution='vector')
    tb.netlist.add_module(writer, 'writer', nets={'out': 'bias'})
    tb.netlist.simulate()
    np.testing.assert_allclose(tb.netlist.load_signal('bias').y, 9)
    np.testing.assert_allclose(tb.netlist.load_signal('ib', 'I').y, 1e-3)
