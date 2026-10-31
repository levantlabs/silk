"""Ordered DC model evaluation and its ordinary testbench read path."""
from pathlib import Path

import numpy as np
import pytest

from silk.exceptions import NotSupportedError, SimulationError
from silk.simulation import AC, DC, Transient
from silk.signals.rawfile import read_rawfile
from silk.sources.classes import CurrentSource, VoltageSource
from silk.sources.waveforms import Pulse
from tests.test_behavioral_rawresult import _block


def _bench(tmp_path, model=None, analysis=None, source=None):
    def model_3(ports, t, params):
        ports.vout.v = (ports.vin.v >= 0.5).astype(float)
    dut = _block(tmp_path, model or (model_3))
    tb = dut.create_test('transfer')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(source or VoltageSource('drive', positive='vin', negative='gnd', dc=9))
    tb.add_sim_type(analysis or DC('drive', 0, 1, 0.25))
    tb.netlist.simulator = 'behavioral'
    return tb


def test_comparator_probe_metric_report_and_disk_view(tmp_path):
    tb = _bench(tmp_path)
    output = tb.probe('vout', analysis='dc')
    tb.add_metric('threshold', lambda sig: float(sig.x[np.flatnonzero(sig.y)[0]]),
                  sig=output)
    report = tb.run()
    np.testing.assert_allclose(output.x, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_array_equal(output.y, [0, 0, 1, 1, 1])
    assert report.metrics['threshold'] == 0.5
    sig = tb.netlist.load_signal('vout', analysis='dc')
    assert sig.xlabel == 'v(v-sweep)'
    np.testing.assert_array_equal(sig.y, output.y)
    raw = read_rawfile(tb.netlist.results_file)
    assert raw.current_plot.plotname == 'DC transfer characteristic'
    assert raw.vectors[0].type == 'voltage'
    np.testing.assert_array_equal(raw.vectors[0].data, output.x)
    assert Path(tb.netlist.view(open=False)).is_file()
    reread = tb.evaluate(rawfile=tb.netlist.results_file)
    assert reread.metrics['threshold'] == 0.5
    assert tb.netlist.sig_sources_dict['drive'].dc == 9


def test_hysteresis_calls_each_point_in_order_and_keeps_state(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    state = {'high': False}
    seen = []

    def schmitt(ports, t, params):
        assert ports.vin.v.shape == t.shape == (1,)
        value = ports.vin.v[0]
        seen.append(value)
        if value >= 0.75:
            state['high'] = True
        elif value <= 0.25:
            state['high'] = False
        ports.vout.v = float(state['high'])

    tb = _bench(tmp_path, schmitt)
    tb.netlist.simulate()
    np.testing.assert_array_equal(tb.netlist.load_signal('vout').y, [0, 0, 0, 1, 1])
    tb.netlist.add_sim_type(DC('drive', 1, 0, -0.25))
    tb.netlist.simulate()
    np.testing.assert_array_equal(tb.netlist.load_signal('vout').x, [1, 0.75, 0.5, 0.25, 0])
    np.testing.assert_array_equal(tb.netlist.load_signal('vout').y, [1, 1, 1, 0, 0])
    assert seen == [0, 0.25, 0.5, 0.75, 1, 1, 0.75, 0.5, 0.25, 0]


@pytest.mark.parametrize('source_name', ['drive', 'DRIVE', 'Vdrive', 'vDRIVE'])
def test_dc_source_accepts_designer_and_emitted_names(tmp_path, source_name):
    tb = _bench(tmp_path, analysis=DC(source_name, '0', '1', '300m'))
    tb.netlist.simulate()
    np.testing.assert_allclose(tb.netlist.load_signal('vin').x, [0, 0.3, 0.6, 0.9])
    np.testing.assert_allclose(tb.netlist.load_signal('vin').y, [0, 0.3, 0.6, 0.9])


def test_current_sweep_preserves_axis_without_driving_model_voltage(tmp_path):
    source = CurrentSource('drive', positive='vin', negative='gnd', dc=9,
                           pulse=Pulse(v1=99, v2=100, td=0, tr=1, tf=1, pw=1, per=4))
    tb = _bench(tmp_path, analysis=DC('Idrive', '750m', '1', '250m'), source=source)
    tb.netlist.simulate()
    sig = tb.netlist.load_signal('vout')
    np.testing.assert_allclose(sig.x, [0.75, 1])
    np.testing.assert_allclose(sig.y, 0)
    assert 'v(vin)' not in tb.netlist.get_signal_names()
    assert sig.xlabel == 'i(i-sweep)'
    assert read_rawfile(tb.netlist.results_file).vectors[0].type == 'current'


@pytest.mark.parametrize('name, error, message', [
    ('missing', SimulationError, 'missing'),
    ('temp', NotSupportedError, 'temperature'),
])
def test_invalid_dc_target_refuses_before_model_runs(tmp_path, name, error, message):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    def model_1(ports, t, params):
        return calls.append(t)
    tb = _bench(tmp_path, model_1, DC(name, 0, 1, 0.5))
    with pytest.raises(error, match=message):
        tb.netlist.simulate()
    assert calls == []
    assert tb.netlist.results_file is None


def test_dc_plus_ac_refuses_the_whole_run(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    def model_2(ports, t, params):
        return calls.append(t)
    tb = _bench(tmp_path, model_2)
    tb.add_sim_type(AC('dec', 10, 1, 100))
    with pytest.raises(NotSupportedError, match='AC'):
        tb.netlist.simulate()
    assert calls == []


def test_dc_and_transient_both_publish_selectable_plots(tmp_path):
    tb = _bench(tmp_path)
    tb.add_sim_type(Transient(step=1, stop=2))
    tb.netlist.simulate()
    np.testing.assert_allclose(tb.netlist.load_signal('vin', analysis='dc').y,
                               [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(tb.netlist.load_signal('vin', analysis='tran').y, [9, 9, 9])


def test_single_point_dc_keeps_shared_spice_reader_convention(tmp_path):
    tb = _bench(tmp_path, analysis=DC('drive', 0.75, 0.75, 1))
    tb.netlist.simulate()
    raw = read_rawfile(tb.netlist.results_file)
    np.testing.assert_array_equal(raw.vectors[0].data, [0.75])
    sig = tb.netlist.load_signal('vin', analysis='dc')
    # The common reader presents one-point real plots as operating points.
    assert sig.xlabel == 'op'
    np.testing.assert_array_equal(sig.x, [0])
    np.testing.assert_array_equal(sig.y, [0.75])


def test_unswept_sources_hold_dc_and_models_receive_point_indices(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    seen = []

    def model(ports, t, params):
        seen.extend(t)
        ports.vout.v = ports.vin.v

    tb = _bench(tmp_path, model)
    tb.add_source(VoltageSource('bias', positive='bias', negative='gnd', dc='200m',
                               pulse=Pulse(v1=99, v2=100, td=0, tr=1, tf=1, pw=1, per=4)))
    tb.netlist.simulate()
    np.testing.assert_allclose(tb.netlist.load_signal('bias').y, [99.2] * 5)
    assert seen == [0, 1, 2, 3, 4]


def test_emitted_name_collision_warns_and_matches_spice_precedence(tmp_path):
    tb = _bench(tmp_path, analysis=DC('Vdrive', 0, 1, 0.5))
    tb.add_source(VoltageSource('Vdrive', positive='other', negative='gnd', dc=2))
    tb.set_supply_voltage(9, source='drive')
    with pytest.warns(UserWarning, match='ambiguous') as notices:
        tb.netlist.simulate()
    assert len(notices) == 1
    assert str(notices[0].message) == (
        "DC source 'Vdrive' is ambiguous; resolved to source "
        "'drive' (emitted names win for DC).")
    assert tb.netlist.operating_conditions.supply_kind == 'sweep'
    np.testing.assert_allclose(tb.netlist.load_signal('vin').y, [0, 0.5, 1])
    np.testing.assert_allclose(tb.netlist.load_signal('other').y, [2, 2, 2])


def test_dc_enters_hierarchy_and_evaluates_cascaded_models(tmp_path):
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from tests.test_behavioral_rawresult import _pdk

    composed = Module('PDK_skywater130', str(tmp_path / 'composed'),
                      'composed.cir', pdk_details=_pdk())
    composed.module_name = 'composed'
    composed.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    for name, source, dest in [('first', 'vin', 'middle'), ('second', 'middle', 'vout')]:
        def model_6(ports, t, params):
            ports.vout.v = ports.vin.v * 2
        composed.add_module(_block(tmp_path, model_6,
                                   name=name), name, nets={'vin': source, 'vout': dest})
    tb = composed.create_test('dc')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(VoltageSource('drive', positive='vin', negative='gnd', dc=0))
    tb.add_sim_type(DC('drive', 0, 1, 0.5))
    tb.netlist.simulator = 'behavioral'
    tb.netlist.simulate()
    np.testing.assert_allclose(tb.netlist.load_signal('xdut.middle').y, [0, 1, 2])
    np.testing.assert_allclose(tb.netlist.load_signal('vout').y, [0, 2, 4])


def malformed_shape(ports, t, params):
    ports.vout.v = [1, 2]


def missing_output(ports, t, params):
    if t[0] == 0:
        ports.vout.v = [1]


@pytest.mark.parametrize('model, error, message', [
    (malformed_shape, ValueError, 'broadcast'),
    (missing_output, SimulationError, 'output nets'),
])
def test_malformed_point_output_never_publishes_results(tmp_path, model, error, message):
    tb = _bench(tmp_path, model)
    with pytest.raises(error, match=message):
        tb.netlist.simulate()
    assert tb.netlist.results_file is None
    assert not tb.netlist.sim_results_loaded


def test_module_report_uses_dc_sweep_axis_not_time(tmp_path):
    from silk.circuit.report import generate_report

    tb = _bench(tmp_path)
    tb.netlist.simulate()
    summary = generate_report(tb.netlist, auto_discover_config=False).simulation
    assert summary.analysis == 'dc'
    assert summary.sweep_name == 'v(v-sweep)'
    assert summary.sweep_start == 0 and summary.sweep_stop == 1
    assert summary.t_start is None and summary.t_stop is None


def test_emitted_source_name_selects_dc_sweep(tmp_path):
    source = VoltageSource('drive_input', positive='vin', negative='gnd', dc=9)
    tb = _bench(tmp_path, source=source, analysis=DC('Vdrive_input', 0, 1, 0.5))
    tb.netlist.simulate()
    np.testing.assert_allclose(tb.netlist.load_signal('vin').y, [0, 0.5, 1])
