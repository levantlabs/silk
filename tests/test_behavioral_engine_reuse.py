"""Independent engine calls refresh run state without changing held evidence."""
import warnings

import numpy as np
import pytest

from silk import Port, PortDir
from silk.elements import elements
from silk.exceptions import RewriteError, SimulationError
from silk.simulation import DC
from silk.simulation.behavioral import BehavioralEngine
from silk.sources import CurrentSource, VoltageSource
from tests.test_behavioral_ports import block, top


ORDERS = [('tran', 'tran'), ('dc', 'dc'), ('tran', 'dc'), ('dc', 'tran')]


def constant(p, t, q):
    p.vout.v = q.value


def replacement(p, t, q):
    p.vout.v = 4


def contract(p, t, q):
    p.vout.v = 2 if np.ndim(t) == 0 else 3


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, p, t, q):
        self.count += 1
        p.vout.v = self.count


def bench(tmp_path, model=constant, *, params=None, execution='vector'):
    dut = block(tmp_path, 'leaf', model, params={'value': 1} if params is None else params,
                execution=execution)
    mod = top(tmp_path)
    mod.add_module(dut, 'leaf', nets={'vin': 'in', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    mod.add_source(CurrentSource('bias', positive='in', negative='gnd', dc=.002))
    return mod, dut


def run(engine, mod, kind):
    if kind == 'dc':
        return engine.run_dc(mod, DC('drive', 0, 2, 1))
    signals, axis = engine.run(mod, 1, 2)
    return signals, axis, 'time'


def same_result(actual, expected, kind):
    signals, axis, axis_kind = actual
    fresh_signals, fresh_axis, fresh_kind = expected
    assert axis_kind == fresh_kind == ('voltage' if kind == 'dc' else 'time')
    np.testing.assert_array_equal(axis, [0, 1, 2])
    np.testing.assert_array_equal(axis, fresh_axis)
    assert signals.keys() == fresh_signals.keys()
    for name in signals:
        np.testing.assert_array_equal(signals[name], fresh_signals[name])


@pytest.mark.parametrize('first,second', ORDERS)
@pytest.mark.parametrize('change', ['model', 'params', 'contract', 'layout', 'counter'])
def test_independent_call_matches_fresh_after_change(tmp_path, first, second, change):
    model = Counter() if change == 'counter' else contract if change == 'contract' else constant
    mod, dut = bench(tmp_path, model)
    engine = BehavioralEngine()
    run(engine, mod, first)
    if change == 'model':
        dut.set_behavioral_model(replacement)
        want = [4, 4, 4]
    elif change == 'params':
        dut.set_behavioral_model(constant, params={'value': 5})
        want = [5, 5, 5]
    elif change == 'contract':
        dut.set_behavioral_model(contract, execution='sample')
        want = [2, 2, 2]
    elif change == 'layout':
        def other(p, t, q):
            p.other.v = 4
        mod = top(tmp_path / 'other')
        leaf = block(tmp_path / 'other', 'different', other,
                     ports={'other': PortDir.OUT})
        mod.add_module(leaf, 'leaf', nets={'other': 'out'})
        mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
        mod.add_source(CurrentSource('bias', positive='in', negative='gnd', dc=.003))
        want = [4, 4, 4]
    else:
        want = [1, 2, 3] if second == 'dc' else [1, 1, 1]
    fresh = BehavioralEngine(scope=engine.scope, sample=engine.sample)
    expected = run(fresh, mod, second)
    actual = run(engine, mod, second)
    same_result(actual, expected, second)
    np.testing.assert_array_equal(actual[0]['out'], want)
    assert engine.source_currents.keys() == fresh.source_currents.keys() == {'Ibias'}
    for name, values in engine.source_currents.items():
        np.testing.assert_array_equal(values, fresh.source_currents[name])
        np.testing.assert_array_equal(values, [.003 if change == 'layout' else .002] * len(values))


@pytest.mark.parametrize('first,second', ORDERS)
def test_retry_refreshes_failed_attachment_and_notices(tmp_path, first, second):
    def broken(p, t, q):
        p.vout.v = 1
        p.vout.v = 2
    mod, dut = bench(tmp_path, broken)
    mod.add(elements.capacitor('load', p='out', n='gnd', value=1))
    engine = BehavioralEngine()
    for candidate in (BehavioralEngine(), engine):
        with pytest.warns(UserWarning, match='load.*ignored') as notices:
            with pytest.raises(RewriteError, match='leaf.*once') as error:
                run(candidate, mod, first)
        assert len(notices) == 1
        if candidate is engine:
            assert str(error.value) == failure_message
        else:
            failure_message = str(error.value)
    dut.set_behavioral_model(replacement)
    with pytest.warns(UserWarning, match='load.*ignored') as notices:
        expected = run(BehavioralEngine(), mod, second)
    assert len(notices) == 1
    with pytest.warns(UserWarning, match='load.*ignored') as notices:
        actual = run(engine, mod, second)
    assert len(notices) == 1
    same_result(actual, expected, second)
    np.testing.assert_array_equal(actual[0]['out'], [4, 4, 4])


@pytest.mark.parametrize('first,second', ORDERS)
def test_held_results_currents_and_sample_handles_keep_original_state(tmp_path, first, second):
    # Observe held handles across independent callable copies.
    global saved
    saved = []
    def model(p, t, q):
        p.vout.v = t + q.value
        p.code[8].v = t + q.value + 100
        if not saved:
            saved.extend([p.vout.v, p.code[8].v])
        else:
            # The retained sample handle stays bound to point zero of run one.
            assert float(saved[0]) == 10
            assert float(saved[1]) == 110
    mod = top(tmp_path)
    dut = block(tmp_path, 'retained', model, execution='sample', params={'value': 10},
                ports={'vin': PortDir.IN, 'vout': PortDir.OUT,
                       'code': Port.digital_output(bits=range(8, 9))})
    mod.add_module(dut, 'leaf', nets={'vin': 'in', 'vout': 'out', 'code[8]': 'b8'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    mod.add_source(CurrentSource('bias', positive='in', negative='gnd', dc=.002))
    engine = BehavioralEngine()
    held, axis, _ = run(engine, mod, first)
    old_state = engine.state
    old_writers = tuple(dict(d) for d in engine._writers)
    held_current = engine.source_currents
    snapshots = {key: value.copy() for key, value in held.items()}
    currents = {key: value.copy() for key, value in held_current.items()}
    dut.set_behavioral_model(model, params={'value': 20}, execution='sample')
    # New writer identities must not be inserted into the preceding state's map.
    mod.add_module(block(tmp_path, 'additional', replacement), 'additional',
                   nets={'vin': 'in', 'vout': 'extra'})
    actual = run(engine, mod, second)
    np.testing.assert_array_equal(actual[0]['out'], [20, 21, 22])
    assert held.keys() == snapshots.keys()
    assert held_current.keys() == currents.keys()
    for key in held:
        np.testing.assert_array_equal(held[key], snapshots[key])
    for key in held_current:
        np.testing.assert_array_equal(held_current[key], currents[key])
    np.testing.assert_array_equal(axis, [0, 1, 2])
    assert engine.source_currents is not held_current
    assert engine.state is not old_state
    assert tuple(dict(d) for d in (old_state._writer_ids, old_state._writer_names)) == old_writers
    for value in saved:
        with pytest.raises(SimulationError, match='only defined during a behavioral run'):
            value[()]


@pytest.mark.parametrize('first,second', ORDERS)
def test_ideal_notices_repeat_for_independent_calls(tmp_path, first, second):
    mod, _ = bench(tmp_path)
    mod.add(elements.capacitor('load', p='out', n='gnd', value=1))
    engine = BehavioralEngine()
    for kind in (first, second):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            run(engine, mod, kind)
        assert len(caught) == 1
        assert caught[0].category is UserWarning
        assert 'load' in str(caught[0].message) and 'ignored' in str(caught[0].message)
        assert len(mod._behavioral_element_notes) == 1


@pytest.mark.parametrize('first,second', ORDERS)
def test_seeded_resolution_refreshes_and_replays_same_sample(tmp_path, first, second):
    from silk.circuit.monte_carlo import _SampleSpec
    from silk.simulation._behavioral_variation import VariationScope
    from silk.simulation.variation import Normal
    sample = _SampleSpec(index=5, seed=19, variation='both', section=None,
                         wrapper_path=None, base_seed=2**80 + 19)
    scope = VariationScope()
    mod, dut = bench(tmp_path / 'run', params={'value': Normal(1, .1)})
    engine = BehavioralEngine(scope=scope, sample=sample)
    run(engine, mod, first)
    old_resolution = engine.resolution
    old_draws = dict(old_resolution.draws)
    # A different nominal shifts this same seeded draw by exactly four.
    dut.set_behavioral_model(constant, params={'value': Normal(5, .1)})
    fresh = BehavioralEngine(scope=scope, sample=sample)
    expected = run(fresh, mod, second)
    actual = run(engine, mod, second)
    same_result(actual, expected, second)
    assert engine.scope is scope and engine.sample is sample
    assert engine.resolution.draws == fresh.resolution.draws
    assert engine.resolution is not old_resolution
    assert old_resolution.draws == old_draws
    assert engine.resolution.draws[('leaf', 'value')] == pytest.approx(old_draws[('leaf', 'value')] + 4)


@pytest.mark.parametrize('first', ['tran', 'dc'])
def test_reused_engine_current_sweep_keeps_axis_kind(tmp_path, first):
    mod, dut = bench(tmp_path)
    engine = BehavioralEngine()
    run(engine, mod, first)
    dut.set_behavioral_model(replacement)
    sweep = DC('bias', 0, .002, .001)
    expected = BehavioralEngine().run_dc(mod, sweep)
    actual = engine.run_dc(mod, sweep)
    assert actual[2] == expected[2] == 'current'
    np.testing.assert_array_equal(actual[1], [0, .001, .002])
    for name in actual[0]:
        np.testing.assert_array_equal(actual[0][name], expected[0][name])
    np.testing.assert_array_equal(actual[0]['out'], [4, 4, 4])
    np.testing.assert_array_equal(engine.source_currents['Ibias'], [.002])


@pytest.mark.parametrize('route', ['engine', 'shortcut', 'testbench'])
def test_single_integrator_starts_fresh_on_three_runs(tmp_path, route):
    total = 0.0
    def integrator(ports, t, params):
        nonlocal total
        total += ports.vin.v
        ports.vout.v = total

    dut = block(tmp_path, 'leaf', integrator, execution='sample')
    mod = top(tmp_path)
    mod.add_module(dut, 'leaf', nets={'vin': 'in', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=.1))
    if route == 'testbench':
        from silk.simulation import Transient
        tb = dut.create_test('integrate')
        tb.add_module_under_test({'vin': 'in', 'vout': 'out'})
        tb.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=.1))
        tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
        tb.netlist.simulator = 'behavioral'
    engine = BehavioralEngine()
    for index in range(3):
        if route == 'engine':
            if index == 2:
                engine = BehavioralEngine()
            signals, _ = engine.run(mod, 1e-9, 10e-9)
            actual = signals['out']
        elif route == 'shortcut':
            mod.simulate_behavioral(1e-9, 10e-9)
            actual = mod.load_signal('out').y
        else:
            tb.run()
            actual = tb.netlist.load_signal('out').y
        np.testing.assert_allclose(actual, np.arange(1, 12) * .1)
    assert total == 0.0


@pytest.mark.parametrize('campaign', ['monte_carlo', 'corners', 'all_tests', 'supply_sweep', 'analyses'])
def test_single_function_state_across_launch_boundaries(tmp_path, campaign):
    from silk.metrics import final_value
    from silk.simulation import Transient
    from silk.simulation.variation import Normal

    total = 0.0
    captured = []
    def model(p, t, q):
        nonlocal total
        total += .1
        captured.append(total)
        p.vout.v = total

    _, dut = bench(tmp_path, model, execution='sample', params={'draw': Normal(1, .1)})
    dut.simulator = 'behavioral'
    for name in (['first', 'second', 'third'] if campaign == 'all_tests' else ['counter']):
        tb = dut.create_test(name)
        tb.add_module_under_test({'vin': 'in', 'vout': 'out'})
        tb.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=.1))
        tb.add_sim_type(DC('drive', 0, .2, .1) if campaign == 'supply_sweep'
                        else Transient(step=1, stop=2))
        tb.add_metric('last', final_value, sig=tb.probe('out', analysis='tran' if campaign == 'analyses' else None))
    if campaign == 'monte_carlo':
        reports = tb.set_monte_carlo(n=3, seed=17).run().values()
    elif campaign == 'corners':
        with pytest.warns(UserWarning, match='identical'):
            reports = tb.run_corners(['tt', 'ff', 'ss']).values()
    elif campaign == 'all_tests':
        reports = dut.run_all_tests().values()
    elif campaign == 'supply_sweep':
        tb.set_supply_voltage(.1, source='drive')
        reports = [tb.run(), tb.run()]
        assert all(r.operating_conditions.supply_kind == 'sweep' for r in reports)
    else:
        # One launch shares the callable across its three transient and DC points.
        tb.add_sim_type(DC('drive', 0, .2, .1))
        reports = []
        for _ in range(2):
            reports.append(tb.run())
            np.testing.assert_allclose(tb.netlist.load_signal('out', analysis='dc').y,
                                       [.4, .5, .6])
    reports = list(reports)
    assert len(reports) == (2 if campaign in ['supply_sweep', 'analyses'] else 3)
    assert all(r.metrics['last'] == pytest.approx(.3) for r in reports)
    assert total == 0 and captured == []
