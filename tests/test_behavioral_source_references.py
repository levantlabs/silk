"""Voltage stimuli are differences; only proven producers supply references."""
from pathlib import Path

import numpy as np
import pytest

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk.elements import elements
from silk.exceptions import NotSupportedError, UnboundSignalError
from silk.simulation import DC, Transient
from silk.simulation.behavioral import BehavioralEngine
from silk.sources import CurrentSource, VoltageSource, VCVS

from tests.test_behavioral_rawresult import _pdk, _block


def bench(tmp_path):
    top = Module('sky130', str(tmp_path), 'reference.cir', pdk_details=_pdk())
    top.simulator = 'behavioral'
    top.add_sim_type(Transient(step=0.25, stop=1))
    return top


def voltage(top, name, positive, negative, value):
    top.add_source(VoltageSource(name, positive, negative, dc=value))


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('polarity', [1, -1])
@pytest.mark.parametrize('entry', ['engine', 'simulate', 'shortcut', 'dc'])
def test_reference_chain_is_differential_in_any_order(tmp_path, reverse, polarity, entry):
    top = bench(tmp_path)
    entries = [('reference', 'ref', '0', 1), ('offset', 'out', 'ref', 0.5), ('tail', 'tail', 'out', 0.25)]
    if polarity == -1:
        entries = [(n, b, a, -v) for n, a, b, v in entries]
    for row in reversed(entries) if reverse else entries:
        voltage(top, *row)
    if entry == 'engine':
        signals, _ = BehavioralEngine().run(top, 0.25, 1)
    else:
        if entry == 'dc':
            top.add_sim_type(DC('offset', polarity * 0.5, polarity * 0.5, 0.1))
        if entry == 'shortcut':
            top.simulate_behavioral(0.25, 1)
        else:
            top.simulate()
        signals = {
            n: top.load_signal(n, analysis='dc' if entry == 'dc' else 'tran').y
            for n in ('ref', 'out', 'tail')}
    for net, want in [('ref', 1), ('out', 1.5), ('tail', 1.75)]:
        np.testing.assert_allclose(signals[net], want)


@pytest.mark.parametrize('kind', ['direct', 'resistor', 'inductor'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('dependent', [False, True])
def test_modeled_reference_orders_through_wire(tmp_path, kind, reverse, dependent):
    top = bench(tmp_path)
    def model_1(ports, t, params):
        ports.vout.v = 1 + 0 * t
    model = _block(tmp_path, model_1)

    def add_model():
        top.add_module(model, 'producer', nets={'vin': '0', 'vout': 'ref'})

    def add_source():
        if dependent:
            voltage(top, 'control', 'control', '0', 0.25)
            top.add_source(VCVS('offset', 'out', 'wired' if kind != 'direct' else 'ref', 'control', '0', 2))
        else:
            voltage(top, 'offset', 'out', 'wired' if kind != 'direct' else 'ref', 0.5)
    for fn in [add_source, add_model] if reverse else [add_model, add_source]:
        fn()
    if kind != 'direct':
        top.add(getattr(elements, kind)('link', p='ref', n='wired', value=1))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out').y, 1.5)


def test_entered_port_aliases_preserve_negative_reference(tmp_path):
    top = bench(tmp_path)
    child = Module('sky130', str(tmp_path / 'child'), 'child.cir', pdk_details=_pdk(), module_name='child')
    child.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    voltage(child, 'offset', 'z', 'a', 0.5)
    top.add_module(child, 'dut', nets={'a': 'reference', 'z': 'out'})
    voltage(top, 'reference', 'reference', '0', 1)
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out').y, 1.5)


@pytest.mark.parametrize('swept', ['reference', 'offset'])
def test_dc_uses_same_reference_equation_and_additive_initial_value(tmp_path, swept):
    top = bench(tmp_path)
    top.add_source(VoltageSource('offset', 'out', 'ref', dc=0.2, pwl=[(0, 0.3), (1, 0.8)]))
    voltage(top, 'reference', 'ref', '0', 1)
    top.add_sim_type(DC(swept, 0, 1, 0.5))
    top.simulate()
    want = [0.5, 1, 1.5] if swept == 'reference' else [1.3, 1.8, 2.3]
    np.testing.assert_allclose(top.load_signal('out', analysis='dc').y, want)
    np.testing.assert_allclose(top.load_signal('out', analysis='tran').y, [1.5, 1.625, 1.75, 1.875, 2])


@pytest.mark.parametrize('reason', ['unknown', 'resistor_ground', 'current', 'current_wire', 'rail_wire'])
@pytest.mark.parametrize('analysis', ['tran', 'dc'])
def test_unproven_reference_refuses_with_source_and_terminals(tmp_path, reason, analysis):
    top = bench(tmp_path)
    voltage(top, 'sweep', 'control', '0', 0.1)
    voltage(top, 'offset', 'out', 'ref', 0.5)
    if reason == 'resistor_ground':
        top.add(elements.resistor('ground_path', p='ref', n='0', value=1))
    if reason in ('current', 'current_wire'):
        top.add_source(CurrentSource('current', 'a' if reason == 'current_wire' else 'ref', '0', dc=0.1))
    if reason in ('current_wire', 'rail_wire'):
        top.add(elements.resistor('link', p='a', n='ref', value=1))
    if reason == 'rail_wire':
        def model_8(ports, t, params):
            ports.vout.v = 0 * t
        model = _block(tmp_path, model_8, ports={'power': PortDir.POWER, 'vout': PortDir.OUT})
        top.add_module(model, 'undriven', nets={'power': 'a', 'vout': 'unused'})
    if reason in ('unknown', 'rail_wire'):
        # Pass wiring admission without introducing a proven voltage producer:
        # this test targets the engine's reference eligibility, not disconnection.
        top.add(elements.capacitor('connectivity', p='ref', n='0', value=1e-12))
    if analysis == 'dc':
        top.add_sim_type(DC('sweep', 0, 1, 0.5))
    with pytest.raises(NotSupportedError) as error:
        top.simulate()
    message = str(error.value)
    assert all((word in message for word in ('offset', 'out', 'ref', 'SPICE')))
    assert 'floating' not in message.lower()
    assert top.results_file is None


@pytest.mark.parametrize('value', [0.5, 0.6])
def test_both_owned_constraints_refuse_even_when_consistent(tmp_path, value):
    top = bench(tmp_path)
    voltage(top, 'reference', 'ref', '0', 1)
    voltage(top, 'output', 'out', '0', 1.5)
    voltage(top, 'offset', 'out', 'ref', value)
    with pytest.raises(NotSupportedError, match='SPICE'):
        top.simulate()


@pytest.mark.parametrize('kind', ['direct', 'resistor'])
def test_missing_reference_output_is_checked_at_every_dc_point(tmp_path, kind):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    top = bench(tmp_path)
    voltage(top, 'sweep', 'input', '0', 0)
    calls = []

    def model(ports, t, params):
        calls.append(float(t[0]))
        if t[0] == 0:
            ports.vout.v = np.ones_like(t)
    top.add_module(_block(tmp_path, model), 'producer', nets={'vin': 'input', 'vout': 'ref'})
    if kind == 'resistor':
        top.add(elements.resistor('link', p='ref', n='wired', value=1))
    voltage(top, 'offset', 'out', 'wired' if kind == 'resistor' else 'ref', 0.5)
    top.sim_analysis_dict.clear()
    top.add_sim_type(DC('sweep', 0, 1, 0.5))
    with pytest.raises(NotSupportedError, match='offset.*SPICE'):
        top.simulate()
    assert calls == [0, 1]
    assert top.results_file is None


def test_static_refusal_keeps_earlier_results_and_probes(tmp_path):
    # TASK-590: the static admission checks run before result replacement, so
    # a refused behavioral run leaves the previous run's evidence readable.
    def model_2(ports, t, params):
        ports.vout.v = ports.vin.v
    dut = _block(tmp_path, model_2)
    tb = dut.create_test('reference')
    tb.add_module_under_test({'vin': 'out', 'vout': 'result'})
    tb.add_source(VoltageSource('reference', 'ref', '0', dc=1))
    tb.add_source(VoltageSource('offset', 'out', 'ref', dc=0.5))
    tb.add_sim_type(Transient(step=0.25, stop=1))
    tb.netlist.simulator = 'behavioral'
    sig = tb.probe('result')
    tb.add_metric('level', lambda sig: float(np.mean(sig.y)), sig=sig)
    assert tb.run().metrics['level'] == 1.5
    results = tb.netlist.results_file
    raw_bytes = Path(results).read_bytes()
    report_before = tb.netlist.operating_conditions
    tb.netlist.sig_sources_dict['offset'].negative = 'missing'
    # A disconnected group would refuse as ground connectivity; retain ground
    # wiring so this is the engine's static reference refusal.
    tb.netlist.add(elements.capacitor('connectivity', p='missing', n='0', value=1e-12))
    with pytest.raises(NotSupportedError):
        tb.run()
    assert tb.netlist.results_file == results
    assert float(np.mean(sig.y)) == 1.5
    assert Path(results).read_bytes() == raw_bytes
    assert tb.netlist.operating_conditions is report_before


def test_failed_replacement_clears_probes_and_report(tmp_path):
    fail = []

    def model(ports, t, params):
        if fail:
            raise RuntimeError('model failed during the run')
        ports.vout.v = ports.vin.v

    dut = _block(tmp_path, model)
    tb = dut.create_test('reference')
    tb.add_module_under_test({'vin': 'out', 'vout': 'result'})
    tb.add_source(VoltageSource('reference', 'ref', '0', dc=1))
    tb.add_source(VoltageSource('offset', 'out', 'ref', dc=0.5))
    tb.add_sim_type(Transient(step=0.25, stop=1))
    tb.netlist.simulator = 'behavioral'
    sig = tb.probe('result')
    tb.add_metric('level', lambda sig: float(np.mean(sig.y)), sig=sig)
    assert tb.run().metrics['level'] == 1.5
    fail.append(True)
    with pytest.raises(RuntimeError):
        tb.run()
    assert tb.netlist.results_file is None
    with pytest.raises(UnboundSignalError):
        _ = sig.y


@pytest.mark.parametrize('net', ['ref', '0', 'gnd'])
def test_reference_with_multiple_model_writers_or_written_ground_refuses(tmp_path, net):
    top = bench(tmp_path)
    for index in range(2 if net == 'ref' else 1):
        def model_9(ports, t, params):
            ports.vout.v = 1 + 0 * t
        model = _block(tmp_path, model_9, name=f'p{index}')
        top.add_module(model, f'producer{index}', nets={'vin': '0', 'vout': net})
    voltage(top, 'offset', 'out', '0' if net == 'gnd' else net, 0.5)
    with pytest.raises(NotSupportedError, match='offset.*SPICE'):
        top.simulate()


def test_current_branch_does_not_interfere_with_voltage_reference_wire(tmp_path):
    top = bench(tmp_path)
    def model_10(ports, t, params):
        ports.vout.v = 1 + 0 * t
    top.add_module(_block(tmp_path, model_10), 'producer', nets={'vin': '0', 'vout': 'a'})
    top.add(elements.resistor('link', p='a', n='ref', value=1))
    top.add_source(CurrentSource('bias', 'ref', '0', dc=0.1))
    voltage(top, 'offset', 'out', 'ref', 0.5)
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out').y, 1.5)
    np.testing.assert_allclose(top.load_signal('ibias', 'I').y, .1)


def test_logical_waveform_is_evaluated_once_without_second_dc_offset(tmp_path):
    top = bench(tmp_path)
    voltage(top, 'reference', 'ref', '0', 1)
    source = VoltageSource('offset', 'out', 'ref', dc=0.2, pwl=[(0, 0.3), (1, 0.8)])
    calls = []
    original = source.to_numpy

    def observed(t):
        calls.append(t.copy())
        return original(t)
    source.to_numpy = observed
    top.add_source(source)
    top.simulate()
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(top.load_signal('out').y, [1.5, 1.625, 1.75, 1.875, 2])


@pytest.mark.parametrize('family,want', [
    ('dc', [1.2] * 5),
    ('pulse', [1.5, 1.5, 2, 2, 1.5]),
    ('sine', [1.5, 2, 1.5, 1, 1.5]),
    ('pwl', [1.5, 1.625, 1.75, 1.875, 2]),
    ('bits', [1.5, 1.5, 1.5, 2, 2]),
    ('prbs', [1.5] * 5),
])
def test_supported_stimulus_families_share_voltage_reference(tmp_path, family, want):
    from silk import sources
    waves = {
        'dc': None,
        'pulse': sources.Pulse(0.3, 0.8, td=0.25, tr=0.25, tf=0.25, pw=0.25, per=2),
        'sine': sources.Sine(0.3, 0.5, 1),
        'pwl': sources.PWL([(0, 0.3), (1, 0.8)]),
        'bits': sources.BitPattern([0, 1], v1=0.3, v2=0.8, ui=0.5, tr=0.25, tf=0.25),
        'prbs': sources.PRBS(7, 2, v1=0.3, v2=0.8, ui=0.5, tr=0.25, tf=0.25),
    }
    top = bench(tmp_path)
    voltage(top, 'reference', 'ref', '0', 1)
    top.add_source(VoltageSource('offset', 'out', 'ref', dc=0.2, waveform=waves[family]))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out').y, want)


@pytest.mark.parametrize('family', ['exp', 'sffm', 'am', 'noise', 'random'])
@pytest.mark.parametrize('current', [False, True])
def test_unsupported_transient_families_keep_dc_initial_values(tmp_path, family, current):
    from silk import sources
    waves = {
        'exp': sources.Exponential(0.3, 0.8, td1=0.1, tau1=0.1, td2=0.5, tau2=0.1),
        'sffm': sources.SFFM(0.3, 0.5, 1, 1, 0.1),
        'am': sources.AM(0.5, 0.3, 0.1, 1),
        'noise': sources.TransientNoise(0.1, 0.25),
        'random': sources.RandomSource(1, 0.25, param2=0.3),
    }
    top = bench(tmp_path)
    voltage(top, 'reference', 'ref', '0', 1)
    cls = CurrentSource if current else VoltageSource
    top.add_source(cls('offset', 'out', 'ref', dc=0.2, waveform=waves[family]))
    with pytest.raises(NotImplementedError):
        top.simulate()
    assert top.results_file is None
    top.sim_analysis_dict.clear()
    top.add_sim_type(DC('offset', 0, 0.5, 0.5))
    top.simulate()
    initial = {'exp': 0.3, 'sffm': 0.3, 'am': 0, 'noise': 0, 'random': 0.3}[family]
    if current:
        assert 'v(out)' not in top.get_signal_names()
    else:
        np.testing.assert_allclose(top.load_signal('out').y, [1 + initial, 1.5 + initial])


def test_current_only_input_is_undriven_with_branch_waveforms(tmp_path):
    top = bench(tmp_path)
    top.add_source(CurrentSource('bias', 'input', 'return', dc=0.2, pwl=[(0, 0.3), (1, 0.8)]))
    top.add(elements.capacitor('connectivity', p='return', n='0', value=1e-12))
    def model_3(ports, t, params):
        ports.vout.v = 2 * ports.vin.v
    model = _block(tmp_path, model_3)
    top.add_module(model, 'consumer', nets={'vin': 'input', 'vout': 'out'})
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out').y, 0)
    np.testing.assert_allclose(top.load_signal('ibias', 'I').y, [0.5, 0.625, 0.75, 0.875, 1])
    np.testing.assert_allclose(top.load_signal('ibias__silk_pwl', 'I').y, [0.3, 0.425, 0.55, 0.675, 0.8])
    np.testing.assert_allclose(top.load_signal('ibias__silk_dc', 'I').y, 0.2)
    assert 'v(return)' not in top.get_signal_names()


def test_inout_model_cannot_overwrite_a_source_owned_voltage(tmp_path):
    top = bench(tmp_path)
    voltage(top, 'offset', 'out', '0', 0.5)
    voltage(top, 'other', 'input', '0', 1)
    def model_4(ports, t, params):
        ports.out.v = 99 + 0 * t
    model = _block(tmp_path, model_4, ports={'vin': PortDir.IN, 'out': PortDir.INOUT})
    top.add_module(model, 'writer', nets={'vin': 'input', 'out': 'out'})
    with pytest.raises(NotSupportedError, match='offset.*out.*SPICE'):
        top.simulate()


@pytest.mark.parametrize('rail', ['0', 'supply'])
def test_proven_rail_can_feed_an_inductive_wire_reference(tmp_path, rail):
    top = bench(tmp_path)
    if rail == 'supply':
        voltage(top, 'supply', 'supply', '0', 1)
        top.set_supply_voltage(1, source='supply')
    top.add(elements.inductor('link', p=rail, n='ref', value=1))
    voltage(top, 'offset', 'out', 'ref', 0.5)
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out').y, 0.5 if rail == '0' else 1.5)


@pytest.mark.parametrize('direction', [PortDir.POWER, PortDir.GROUND])
def test_dc_retains_unrelated_default_rail_vectors(tmp_path, direction):
    top = bench(tmp_path)
    voltage(top, 'sweep', 'input', '0', 0)
    def model_5(ports, t, params):
        ports.vout.v = ports.rail.v + ports.vin.v
    model = _block(
        tmp_path, model_5,
        ports={'vin': PortDir.IN, 'rail': direction, 'vout': PortDir.OUT})
    top.add_module(model, 'consumer', nets={'vin': 'input', 'rail': 'unused_rail', 'vout': 'out'})
    top.add_sim_type(DC('sweep', 0, 1, 0.5))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('unused_rail', analysis='dc').y, 0)
    np.testing.assert_allclose(top.load_signal('unused_rail', analysis='tran').y, 0)
    np.testing.assert_allclose(top.load_signal('out', analysis='dc').y, [0, 0.5, 1])


@pytest.mark.parametrize('selection', ['copied', 'selected'])
def test_copied_and_selected_benches_keep_reference_probes(tmp_path, selection):
    import copy
    def model_6(ports, t, params):
        ports.vout.v = ports.vin.v
    dut = _block(tmp_path, model_6)
    tb = dut.create_test('reference')
    tb.add_module_under_test({'vin': 'out', 'vout': 'result'})
    tb.add_source(VoltageSource('reference', 'ref', '0', dc=1))
    tb.add_source(VoltageSource('offset', 'out', 'ref', dc=.5))
    tb.add_sim_type(Transient(step=.25, stop=1))
    tb.netlist.simulator = 'behavioral'
    tb.add_metric('level', lambda sig: float(np.mean(sig.y)), sig=tb.probe('result'))
    if selection == 'copied':
        tb = copy.deepcopy(tb)
        report = tb.run()
    else:
        report = dut.run_test('reference')
    assert report.metrics['level'] == 1.5
    np.testing.assert_allclose(tb.probe('result').y, 1.5)


@pytest.mark.parametrize('kind', ['direct', 'resistor', 'inductor'])
def test_dc_recomputes_modeled_reference_and_wire_at_each_point(tmp_path, kind):
    top = bench(tmp_path)
    voltage(top, 'offset', 'out', 'ref' if kind == 'direct' else 'wired', .5)
    def model_7(ports, t, params):
        ports.vout.v = 1 + 2 * ports.vin.v
    model = _block(tmp_path, model_7)
    top.add_module(model, 'producer', nets={'vin': 'input', 'vout': 'ref'})
    voltage(top, 'sweep', 'input', '0', 0)
    if kind != 'direct':
        top.add(getattr(elements, kind)('link', p='ref', n='wired', value=1))
    top.sim_analysis_dict.clear()
    top.add_sim_type(DC('sweep', 0, 1, .5))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('ref').y, [1, 2, 3])
    np.testing.assert_allclose(top.load_signal('out').y, [1.5, 2.5, 3.5])

@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('analysis', ['tran', 'dc'])
def test_current_branch_cannot_overwrite_voltage_or_model_input(tmp_path, reverse, analysis):
    top = bench(tmp_path)
    entries = [VoltageSource('supply', 'vdd', '0', dc=1.8),
               CurrentSource('load', 'vdd', '0', dc=0.01)]
    for source in reversed(entries) if reverse else entries:
        top.add_source(source)
    def model_11(ports, t, params):
        ports.vout.v = ports.vin.v
    top.add_module(_block(tmp_path, model_11),
                   'consumer', nets={'vin': 'vdd', 'vout': 'out'})
    if analysis == 'dc':
        top.add_sim_type(DC('load', 0.01, 0.03, 0.01))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('vdd', analysis=analysis).y, 1.8)
    np.testing.assert_allclose(top.load_signal('out', analysis=analysis).y, 1.8)
    if analysis == 'tran':
        np.testing.assert_allclose(top.load_signal('iload', 'I').y, 0.01)


def test_current_only_input_matches_ordinary_undriven_input(tmp_path):
    top = bench(tmp_path)
    def model_12(ports, t, params):
        ports.vout.v = ports.vin.v
    top.add_module(_block(tmp_path, model_12),
                   'consumer', nets={'vin': 'undriven', 'vout': 'out'})
    engine = BehavioralEngine()
    before, _ = engine.run(top, .25, 1)
    top.add_source(CurrentSource('load', 'undriven', 'return', dc=.7))
    after, _ = engine.run(top, .25, 1)
    assert set(after) == set(before)
    for net in before:
        np.testing.assert_array_equal(after[net], before[net])
    np.testing.assert_allclose(next(iter(engine.source_currents.values())), .7)
