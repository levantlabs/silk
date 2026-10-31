"""TASK-448.3: a behavioural run is a nutmeg-shaped RawResult on disk.

``Module.simulate()`` on the ``'behavioral'`` backend packages the engine's
arrays as the same :class:`~silk.signals.rawfile.RawFile` a SPICE run
produces, writes it to the run's rawfile path and sets ``results_file``, so
``load_signal``, ``view()``, ``emit_view`` and ``evaluate(rawfile=...)`` read
it the way they read SPICE results. Provenance per design §7.3 (D5).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.exceptions import NotSupportedError, SimulationError
from silk.signals.rawfile import read_rawfile
from silk.simulation import Transient
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import Pulse


def _pdk():
    from silk.pdk.basepdk import BasePDK
    pdk_yaml = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk',
                            'data', 'skywater_130_device_details.yaml')
    return BasePDK(pdk_yaml, library_location='/fake/lib')


def _block(tmp_path, model, ports=None, name='blk', *, execution='vector'):
    blk = Module('PDK_skywater130', str(tmp_path / name), f'{name}.cir',
                 pdk_details=_pdk())
    blk.module_name = name
    blk.define_module_ports(ports or {'vin': PortDir.IN, 'vout': PortDir.OUT})
    blk.set_behavioral_model(model, execution=execution)
    return blk


def _top(tmp_path, model=None, *, source=None):
    """A source on 'vin' driving one modelled block whose output is 'vout'."""
    top = Module('PDK_skywater130', str(tmp_path / 'top'), 'top.cir',
                 pdk_details=_pdk())
    top.add_source(source or VoltageSource('vs', positive='vin',
                                           negative='gnd', dc=0.4))
    def model_1(ports, t, params):
        ports.vout.v = 2.0 * ports.vin.v + 100000000.0 * t
    model = model or (model_1)
    top.add_module(_block(tmp_path, model), 'g1',
                   nets={'vin': 'vin', 'vout': 'vout'})
    top.add_sim_type(Transient(step=1e-9, stop=10e-9))
    top.simulator = 'behavioral'
    return top


# AC #1 ----------------------------------------------------------------------

def test_load_signal_reads_the_model_equation_from_the_written_rawfile(tmp_path):
    top = _top(tmp_path).simulate()
    assert Path(top.results_file).parent.parent == Path(top.dir_folder)
    assert Path(top.results_file).name == 'rawfile.raw'
    assert os.path.isfile(top.results_file)
    sig = top.load_signal('vout')
    np.testing.assert_allclose(sig.y, 2.0 * 0.4 + 1e8 * sig.x)
    assert sig.type == 'voltage' and sig.xlabel == 'time'
    # Read through the SPICE reader, not the simulate_behavioral() branch.
    assert top._raw.native is top.simResultsAll
    # Net.voltage is still written, from the same arrays (lint STR-001).
    np.testing.assert_array_equal(top.net_dict['vout'].voltage, sig.y)
    np.testing.assert_array_equal(top._behavioral_t, sig.x)


def test_rawfile_is_spelled_and_typed_as_ngspice_writes_it(tmp_path):
    top = _top(tmp_path).simulate()
    on_disk = read_rawfile(top.results_file)
    plot = on_disk.current_plot
    assert plot.plotname == 'Transient Analysis'
    assert plot.command == 'silk behavioral engine'
    assert plot.real
    names = [v.name for v in plot.vectors]
    assert names[0] == 'time' and plot.vectors[0].type == 'time'
    assert {'v(vin)', 'v(vout)'} <= set(names)
    assert all(v.type == 'voltage' for v in plot.vectors[1:])
    # The file holds exactly what the in-memory result holds.
    for vec, mem in zip(plot.vectors, top._raw.native.vectors):
        assert vec.name == mem.name
        np.testing.assert_array_equal(vec.data, mem.data)
    assert top.get_signal_names() == names




def test_view_and_emit_view_read_the_results_file(tmp_path):
    from silk.signals.view import emit_view
    top = _top(tmp_path).simulate()
    page = top.view(open=False)
    assert os.path.isfile(page)
    out = emit_view(top.results_file, tmp_path / 'explore.html')
    assert os.path.isfile(out.html_path)
    assert out.n_plots == 1 and out.n_points == len(top.load_signal('vout').x)


def _bench(tmp_path):
    def model_2(ports, t, params):
        ports.vout.v = 3.0 * ports.vin.v
    dut = _block(tmp_path, model_2,
                 name='amp')
    tb = dut.create_test('gain')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(VoltageSource('vs', positive='vin', negative='gnd', dc=0.2))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.add_metric('vfinal', lambda c: float(c.load_signal('vout').y[-1]))
    tb.netlist.simulator = 'behavioral'
    return tb


def test_evaluate_reads_the_results_file_back_as_external_evidence(tmp_path):
    tb = _bench(tmp_path)
    tb.netlist.simulate()
    path = tb.netlist.results_file
    assert tb.netlist.operating_conditions.origin == 'silk'

    report = tb.evaluate(rawfile=path)
    assert report.metrics['vfinal'] == pytest.approx(0.6)
    conditions = report.operating_conditions
    assert conditions.origin == 'external'
    assert conditions.corner_basis in ('unknown', 'caller-declared')
    assert tb.netlist.results_file == path


def test_evaluate_from_memory_keeps_silk_origin(tmp_path):
    tb = _bench(tmp_path)
    tb.netlist.simulate()
    report = tb.evaluate()
    assert report.metrics['vfinal'] == pytest.approx(0.6)
    assert report.operating_conditions.origin == 'silk'
    assert report.operating_conditions.backend == 'behavioral'


# AC #2 ----------------------------------------------------------------------

def test_operating_conditions_are_silk_on_the_behavioral_backend(tmp_path):
    top = _top(tmp_path).simulate()
    c = top.operating_conditions
    assert (c.origin, c.backend) == ('silk', 'behavioral')
    assert (c.corner_kind, c.corner_basis) == ('none', 'proven-behavioral')
    assert (c.temperature_kind, c.temperature_basis) == ('none', 'proven-behavioral')
    # A 0.4 V DC source with no configured supply: unknown, never invented.
    assert (c.supply_kind, c.supply_basis) == ('unknown', 'unknown')
    assert top._launch_record['simulator'] == 'behavioral'
    assert top._launch_record['simulator_binary_source'] == 'in-process'


def test_no_supply_is_proven_none(tmp_path):
    pulse = VoltageSource('vs', positive='vin', negative='gnd',
                          pulse=Pulse(v1=0, v2=1, td=0, tr=1e-10, tf=1e-10,
                                      pw=4e-9, per=8e-9))
    top = _top(tmp_path, source=pulse).simulate()
    c = top.operating_conditions
    assert (c.supply_kind, c.supply_basis) == ('none', 'proven-no-supply')


def _supplied_top(tmp_path, model):
    top = Module('PDK_skywater130', str(tmp_path / 'top'), 'top.cir',
                 pdk_details=_pdk())
    top.add_source(VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.0))
    top.add_module(_block(tmp_path, model, ports={'vdd': PortDir.IN,
                                                  'vout': PortDir.OUT},
                          name='half'),
                   'h1', nets={'vdd': 'vdd', 'vout': 'vout'})
    top.add_sim_type(Transient(step=1e-9, stop=5e-9))
    top.simulator = 'behavioral'
    return top


def test_configured_supply_is_configured_and_driven(tmp_path):
    def model_3(ports, t, params):
        ports.vout.v = ports.vdd.v / 2
    top = _supplied_top(tmp_path, model_3)
    top.set_supply_voltage(1.2, source='vdd')
    top.simulate()
    c = top.operating_conditions
    assert (c.supply_kind, c.supply_voltage_v, c.supply_basis, c.supply_source) \
        == ('fixed', 1.2, 'configured-and-driven', 'vdd')
    np.testing.assert_allclose(top.load_signal('vout').y, 0.6)


def test_supply_overridden_on_its_net_refuses_and_publishes_nothing(tmp_path):
    def model_4(ports, t, params):
        ports.vout.v = ports.vdd.v
    top = _supplied_top(tmp_path, model_4)
    # A second block drives the supply net itself.
    top.add_source(VoltageSource('vx', positive='vx', negative='gnd',
                                 pulse=Pulse(v1=0, v2=1, td=0, tr=1e-10,
                                             tf=1e-10, pw=2e-9, per=4e-9)))
    def model_5(ports, t, params):
        ports.vout.v = 0.5 + 0 * t
    top.add_module(_block(tmp_path, model_5,
                          name='rogue'),
                   'r1', nets={'vin': 'vx', 'vout': 'vdd'})
    top.set_supply_voltage(1.2, source='vdd')
    # This conflicting source/model ownership now refuses before any model
    # runs, instead of detecting a changed supply value after evaluation.
    with pytest.raises(NotSupportedError, match='vdd.*both terminals already owned.*SPICE'):
        top.simulate()
    assert top.sim_results_loaded is False
    assert top.results_file is None and top.operating_conditions is None


def test_configured_and_driven_is_in_the_exported_vocabulary():
    from silk.simulation.conditions import export_conditions_schema
    supply = export_conditions_schema()['facts']['supply']
    assert 'configured-and-driven' in supply['bases']['fixed']
    assert supply['categories']['configured-and-driven'] == 'configured'


# AC #3 ----------------------------------------------------------------------

def _failing_top(tmp_path):
    state = {'fail': False}

    def model(ports, t, params):
        if state['fail']:
            raise RuntimeError('model blew up')
        ports.vout.v = ports.vin.v
    return _top(tmp_path, model), state


def test_overwrite_mode_replaces_and_a_failed_run_leaves_no_results(tmp_path):
    top, state = _failing_top(tmp_path)
    top.set_output_options(mode='overwrite')
    first = top.simulate().results_file
    top.simulate()
    assert top.results_file == first and os.path.isfile(first)

    state['fail'] = True
    with pytest.raises(Exception, match='model blew up'):
        top.simulate()
    assert top.sim_results_loaded is False
    assert top._raw is None and top.results_file is None
    assert top.operating_conditions is None
    # The previous run's rawfile is removed before the attempt, so nothing
    # stale can be reloaded from the deterministic path.
    assert not os.path.exists(first)
    with pytest.raises(SimulationError):
        top.load_signal('vout')


def test_new_mode_keeps_each_run_and_a_failed_run_leaves_no_results(tmp_path):
    top, state = _failing_top(tmp_path)
    top.set_output_options(mode='new')
    first = top.simulate().results_file
    second = top.simulate().results_file
    assert first != second
    for path in (first, second):
        assert os.path.isfile(path)
        assert os.path.dirname(os.path.dirname(path)) == top.dir_folder
        assert os.path.basename(os.path.dirname(path)).startswith('run-')

    state['fail'] = True
    with pytest.raises(Exception, match='model blew up'):
        top.simulate()
    assert top.sim_results_loaded is False
    assert top._raw is None and top.results_file is None
    # New mode keeps every earlier run's directory.
    assert os.path.isfile(first) and os.path.isfile(second)


def test_simulate_behavioral_uses_backend(tmp_path):
    """The shortcut publishes the same rawfile and provenance as simulate."""
    top = _top(tmp_path)
    top.simulator = 'ngspice'
    top.simulate_behavioral(t_step=1e-9, t_stop=10e-9)
    assert os.path.isfile(top.results_file) and top._raw is not None
    assert top.operating_conditions.origin == 'silk'
    assert 'v(vout)' in top.get_signal_names()
    np.testing.assert_allclose(top.load_signal('vout').y[0], 0.8)


def test_shortcut_grid_selector_and_view(tmp_path):
    from silk.signals.view import emit_view
    top = _top(tmp_path)
    declared = top.sim_analysis_dict
    assert top.simulate_behavioral(1e-9, 12e-9) is top
    assert top.sim_analysis_dict is declared
    sig = top.load_signal('vout', analysis='tran')
    assert len(sig.x) == 13
    assert sig.x[-1] == pytest.approx(12e-9)
    with pytest.raises(SimulationError, match='ac'):
        top.load_signal('vout', analysis='ac')
    with pytest.raises(NotSupportedError, match='device parameters'):
        top.load_signal('xg1:gm')
    with pytest.raises(NotSupportedError, match='internal nets'):
        top.load_signal('xg1.hidden')
    assert os.path.isfile(top.view(open=False))
    assert os.path.isfile(emit_view(top.results_file, tmp_path / 'shortcut.html').html_path)


def test_shortcut_failure_restores_declarations_and_clears_results(tmp_path):
    top, state = _failing_top(tmp_path)
    declared = top.sim_analysis_dict
    top.simulate_behavioral(1e-9, 5e-9)
    state['fail'] = True
    with pytest.raises(Exception, match='model blew up'):
        top.simulate_behavioral(1e-9, 5e-9)
    assert top.sim_analysis_dict is declared
    assert top._raw is None and top.results_file is None
    assert not top.sim_results_loaded


def test_shortcut_net_voltage_remains_usable_by_str001(tmp_path):
    from silk.lint.rules.stress import JunctionOverrangeRule
    def model_6(ports, t, params):
        ports.vout.v = 2.5 + 0 * t
    top = _top(tmp_path, model=model_6)
    top.simulate_behavioral(1e-9, 5e-9)
    # Attach a catalog device for a post-run stress check of the produced
    # waveform. It is not part of the behavioral simulation topology.
    with top:
        top.pdk_details.nmos1p8('m1', d='vin', g='vout', s='gnd', b='gnd',
                               w=1, l=0.15)
    rule = JunctionOverrangeRule()
    assert rule.simulation_data_usable(top)
    violations = list(rule.check(top))
    vgs = [v for v in violations if v.location.terminal == 'Vgs']
    assert len(vgs) == 1
    assert vgs[0].measured_v == pytest.approx(2.5)
    assert vgs[0].rule_id == 'STR-001'
    np.testing.assert_array_equal(top.net_dict['vout'].voltage,
                                  top.load_signal('vout').y)


def test_shortcut_new_output_mode_preserves_both_runs(tmp_path):
    top = _top(tmp_path)
    top.set_output_options(mode='new')
    first = top.simulate_behavioral(1e-9, 5e-9).results_file
    second = top.simulate_behavioral(1e-9, 10e-9).results_file
    assert first != second
    assert os.path.isfile(first) and os.path.isfile(second)
    assert read_rawfile(first).current_plot.npoints == 6
    assert read_rawfile(second).current_plot.npoints == 11
