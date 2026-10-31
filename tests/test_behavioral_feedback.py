"""Feedback scheduling and reader-local transient delays."""
import warnings

import numpy as np
import pytest

from silk import PortDir, Port, analysis
from silk.exceptions import SimulationError
from silk.simulation import DC
from silk.simulation.behavioral import BehavioralEngine
from silk.sources import VoltageSource
from tests.test_behavioral_ports import block, top


def loop(tmp_path, execution='sample', reverse=False, delay=False, mixed=False, wire=False):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []

    def a(p, t, q):
        calls.append(('a', np.shape(t)))
        p.vin.v.initial_value = 0
        p.vout.v = (p.vin.v.previous_value if delay else p.vin.v) + 1

    def b(p, t, q):
        calls.append(('b', np.shape(t)))
        p.vout.v = 0.5 * p.vin.v
    mod = top(tmp_path)
    aa = block(tmp_path, 'a', a, execution=execution)
    bb = block(tmp_path, 'b', b, execution='vector' if mixed else execution)
    placements = [('a', aa, {'vin': 'b', 'vout': 'a'}), ('b', bb, {'vin': 'wire' if wire else 'a', 'vout': 'b'})]
    if reverse:
        placements.reverse()
    for name, dut, nets in placements:
        mod.add_module(dut, name, nets=nets)
    return (mod, calls)


@pytest.mark.parametrize('execution', ['sample', 'vector'])
@pytest.mark.parametrize('reverse', [False, True])
def test_loop_values_and_complete_call_order(tmp_path, execution, reverse):
    mod, calls = loop(tmp_path, execution, reverse)
    with pytest.warns(UserWarning, match='[Ff]eedback') as caught:
        values, _ = BehavioralEngine().run(mod, 1, 2)
    assert len(caught) == 1
    if execution == 'sample':
        np.testing.assert_allclose(values['a'], [1, 1.5, 1.75])
        np.testing.assert_allclose(values['b'], [0, .5, .75] if reverse else [.5, .75, .875])
    else:
        np.testing.assert_array_equal(values['a'], [1, 1, 1])
        np.testing.assert_array_equal(values['b'], [0, 0, 0] if reverse else [.5, .5, .5])
    pair = [('b', () if execution == 'sample' else (3,)),
            ('a', () if execution == 'sample' else (3,))] if reverse else [
                ('a', () if execution == 'sample' else (3,)),
                ('b', () if execution == 'sample' else (3,))]
    assert calls == pair * (3 if execution == 'sample' else 1)


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_explicit_delay_removes_feedback_warning(tmp_path, execution):
    mod, _ = loop(tmp_path, execution, delay=True)
    with warnings.catch_warnings(record=True) as caught:
        values, _ = BehavioralEngine().run(mod, 1, 2)
    assert not caught
    np.testing.assert_allclose(values['a'], [1, 1.5, 1.75] if execution == 'sample' else [1, 1, 1])


def test_mixed_contract_loop_refused_before_calls(tmp_path):
    mod, calls = loop(tmp_path, mixed=True)
    with pytest.raises(SimulationError, match='a.*b|b.*a') as caught:
        BehavioralEngine().run(mod, 1, 2)
    assert type(caught.value).__name__ == 'LoopError'
    assert 'execution=' in str(caught.value) and 'one function' in str(caught.value)
    assert calls == []


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_dc_loop_runs_point_then_refuses(tmp_path, execution):
    mod, calls = loop(tmp_path, execution)
    mod.add_source(VoltageSource('drive', positive='sweep', negative='gnd', dc=0))
    with pytest.raises(SimulationError) as caught:
        BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
    assert type(caught.value).__name__ == 'LoopError'
    message = str(caught.value)
    assert all(word in message for word in ['point', 'a', 'b', 'read', 'transient', 'one model'])
    assert [x[0] for x in calls] == ['a', 'b']


@pytest.mark.parametrize('execution', ['vector', 'sample'])
@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('bus', [False, True])
@pytest.mark.parametrize('dc', [False, True])
def test_initial_previous_values_and_dc_refusal(tmp_path, execution, quantity, bus, dc):
    def model(p, t, q):
        value = getattr(p.code if bus else p.vin, quantity)
        value.initial_value = 7
        delayed = value.previous_value
        if execution == 'vector':
            for k in range(len(t)):
                p.vout.v[k] = delayed[0, k] if bus else delayed[k]
        else:
            p.vout.v = delayed[0] if bus else delayed
    dut = block(tmp_path, 'follower', model, execution=execution)
    if bus:
        dut.define_module_ports({'code': Port.digital_input(bits=range(1)), 'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=3))
    mod.add_module(dut, 'follower', nets={'code[0]' if bus else 'vin': 'in', 'vout': 'out'})
    if dc:
        with pytest.raises(SimulationError, match='follower.*(code|vin).*silk.analysis'):
            BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
    else:
        result, _ = BehavioralEngine().run(mod, 1, 2)
        np.testing.assert_allclose(result['out'], [7, 3, 3] if quantity == 'v' else [7, 0, 0])
        np.testing.assert_array_equal(result['in'], [3, 3, 3])


def test_dc_analysis_branch_and_pure_watcher(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    seen = []
    def watcher(p, t, q):
        p.vin.v.initial_value = 99
        seen.append(float(p.vin.v if analysis() == 'dc' else p.vin.v.previous_value))
    dut = block(tmp_path, 'watcher', watcher, execution='sample')
    mod = top(tmp_path)
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    mod.add_module(dut, 'watcher', nets={'vin': 'in', 'vout': 'unused'})
    BehavioralEngine().run_dc(mod, DC('drive', 0, 1, .5))
    assert seen == [0, .5, 1]


def test_equal_values_still_warn_and_runs_warn_independently(tmp_path):
    def a(p, t, q):
        p.vout.v = p.vin.v * 0
    def b(p, t, q):
        p.vout.v = p.vin.v
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'a', a, execution='sample'), 'a', nets={'vin': 'b', 'vout': 'a'})
    mod.add_module(block(tmp_path, 'b', b, execution='sample'), 'b', nets={'vin': 'a', 'vout': 'b'})
    for _ in range(2):
        with pytest.warns(UserWarning, match='Feedback') as caught:
            values, _ = BehavioralEngine().run(mod, 1, 3)
        assert len(caught) == 1
        np.testing.assert_array_equal(values['a'], 0)


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_chained_delay_and_initial_changes_are_reader_local(tmp_path, execution):
    def producer(p, t, q):
        p.vout.v = t + 1
    def reader(p, t, q):
        p.vin.v.initial_value = 8
        p.vout.v = p.vin.v.previous_value.previous_value
        p.vin.v.initial_value = float('nan')
        p.vin.v.initial_value = 4
        if execution == 'vector':
            assert p.vin.v.previous_value[0] == 4
        elif t == 0:
            assert float(p.vin.v.previous_value) == 4
    mod = top(tmp_path)
    # Reverse placement verifies drivers still precede their readers.
    mod.add_module(block(tmp_path, 'reader', reader, execution=execution), 'reader', nets={'vin': 'in', 'vout': 'out'})
    mod.add_module(block(tmp_path, 'producer', producer, execution=execution), 'producer', nets={'vin': 'gnd', 'vout': 'in'})
    result, _ = BehavioralEngine().run(mod, 1, 3)
    np.testing.assert_array_equal(result['out'], [8, 8, 1, 2])
    np.testing.assert_array_equal(result['in'], [1, 2, 3, 4])




def test_dc_loop_runs_downstream_watcher_before_error(tmp_path):
    # The watcher and loop models share a module-global observer.
    global calls
    mod, calls = loop(tmp_path)
    def watcher(p, t, q):
        calls.append(('watcher', float(p.vin.v)))
    mod.add_module(block(tmp_path, 'watcher', watcher, execution='sample'), 'watcher', nets={'vin': 'b', 'vout': 'unused'})
    mod.add_source(VoltageSource('drive', positive='sweep', negative='gnd', dc=0))
    with pytest.raises(SimulationError) as caught:
        BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
    assert type(caught.value).__name__ == 'LoopError'
    assert calls[-1] == ('watcher', .5)


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_dc_initial_value_does_not_change_unpublished_voltage(tmp_path, execution):
    def model(p, t, q):
        p.vin.v.initial_value = 9
        p.vout.v = p.vin.v
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'reader', model, execution=execution), 'reader', nets={'vin': 'unset', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='sweep', negative='gnd', dc=0))
    result, _, _ = BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
    np.testing.assert_array_equal(result['out'], [0, 0])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_ground_delay_uses_initial_only_before_first_point(tmp_path, execution):
    def model(p, t, q):
        p.vin.v.initial_value = 7
        p.vout.v = p.vin.v.previous_value
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'reader', model, execution=execution), 'reader', nets={'vin': 'gnd', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [7, 0, 0])


@pytest.mark.parametrize('dc', [False, True])
def test_feedback_through_ideal_wire(tmp_path, dc):
    from silk.elements import elements
    mod, calls = loop(tmp_path, execution='vector', wire=True)
    mod.add(elements.resistor('link', p='a', n='wire', value=1))
    engine = BehavioralEngine()
    if dc:
        mod.add_source(VoltageSource('drive', positive='sweep', negative='gnd', dc=0))
        with pytest.warns(UserWarning, match='wire'):
            with pytest.raises(SimulationError) as caught:
                engine.run_dc(mod, DC('drive', 0, 1, 1))
        assert type(caught.value).__name__ == 'LoopError'
    else:
        with pytest.warns(UserWarning) as caught:
            engine.run(mod, 1, 2)
        assert any('Feedback' in str(w.message) for w in caught)


def test_reused_engine_warns_for_each_direct_run(tmp_path):
    mod, _ = loop(tmp_path)
    engine = BehavioralEngine()
    for _ in range(2):
        with pytest.warns(UserWarning, match='Feedback') as caught:
            engine.run(mod, 1, 2)
        assert len(caught) == 1


@pytest.mark.parametrize('shared', [True, False])
def test_backend_warning_lifetime_spans_two_actual_dispatches(tmp_path, monkeypatch, shared):
    from silk.simulation import Transient
    from silk.simulation import behavioral_backend
    mod, calls = loop(tmp_path)
    mod.add_sim_type(Transient(step=1, stop=2))
    monkeypatch.setattr(behavioral_backend, '_declared_analyses',
                        lambda module: [Transient(step=1, stop=2), Transient(step=1, stop=3)])
    original = BehavioralEngine.run
    grids = []
    def dispatch(engine, module, t_step, t_stop):
        if not shared:
            engine._shared_run = False
        result, grid = original(engine, module, t_step, t_stop)
        grids.append(grid.copy())
        return result, grid
    monkeypatch.setattr(BehavioralEngine, 'run', dispatch)
    with pytest.warns(UserWarning, match='Feedback') as caught:
        mod.simulate()
    assert len(grids) == 2
    np.testing.assert_array_equal(grids[0], [0, 1, 2])
    np.testing.assert_array_equal(grids[1], [0, 1, 2, 3])
    assert calls == [('a', ()), ('b', ())] * 7
    assert len(caught) == (1 if shared else 2)


@pytest.mark.parametrize('dc', [False, True])
def test_own_authored_currents_do_not_create_feedback(tmp_path, dc):
    def a(p, t, q):
        p.vin.i = 1
        p.vout.v = p.vin.i
    def b(p, t, q):
        p.vin.i = -1
        p.vout.v = p.vin.i
    mod = top(tmp_path)
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    mod.add_module(block(tmp_path, 'a', a), 'a', nets={'vin': 'in', 'vout': 'a'})
    mod.add_module(block(tmp_path, 'b', b), 'b', nets={'vin': 'in', 'vout': 'b'})
    with warnings.catch_warnings(record=True) as caught:
        if dc:
            result, _, _ = BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
        else:
            result, _ = BehavioralEngine().run(mod, 1, 2)
    assert not caught
    np.testing.assert_array_equal(result['a'], 1)
    np.testing.assert_array_equal(result['b'], -1)




@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_authored_current_delay_keeps_waveform_and_bus_bit_initials(tmp_path, execution):
    def model(p, t, q):
        p.code.i.initial_value = [7, 8]
        if execution == 'vector':
            p.code.i = np.stack([t + 1, t + 2])
            for k in range(len(t)):
                p.vout.v[k] = p.code.i.previous_value[0, k] + p.code[9].i.previous_value[k]
            np.testing.assert_array_equal(p.code.i, [[1, 2, 3], [2, 3, 4]])
        else:
            p.code.i = [t + 1, t + 2]
            p.vout.v = p.code.i.previous_value[0] + p.code[9].i.previous_value
            np.testing.assert_array_equal(p.code.i, [t + 1, t + 2])
    dut = block(tmp_path, 'currents', model, execution=execution)
    dut.define_module_ports({'code': Port.digital_input(bits=range(8, 10)), 'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_module(dut, 'currents', nets={'code[8]': 'one', 'code[9]': 'two', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [15, 3, 5])


def test_saved_delay_view_is_refused_when_read_in_dc(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global saved
    saved = []
    def model(p, t, q):
        if analysis() == 'transient':
            saved.append(p.vin.v.previous_value)
        p.vout.v = saved[0]
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'reader', model), 'reader', nets={'vin': 'in', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=1))
    engine = BehavioralEngine()
    engine.run(mod, 1, 2)
    with pytest.raises(SimulationError, match='reader.*vin.*silk.analysis'):
        engine.run_dc(mod, DC('drive', 0, 1, 1))


def test_delayed_kcl_read_does_not_relabel_authored_current_samples(tmp_path):
    def a(p, t, q):
        p.vin.i[0] = 1
        delayed = p.vin.i.previous_value[2]
        p.vout.v = delayed + p.vin.i[0]
    def b(p, t, q):
        p.vout.i = p.vin.v
        p.vout.v = 0
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'a', a), 'a', nets={'vin': 'back', 'vout': 'out'})
    mod.add_module(block(tmp_path, 'b', b), 'b', nets={'vin': 'out', 'vout': 'back'})
    with warnings.catch_warnings(record=True) as caught:
        result, _ = BehavioralEngine().run(mod, 1, 2)
    assert not caught
    np.testing.assert_array_equal(result['out'], [1, 1, 1])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_explicit_delay_reads_exact_prior_stored_sample(tmp_path, execution):
    def producer(p, t, q):
        if execution == 'vector':
            p.vout.v[[0, 2]] = [1, 3]
        elif t != 1:
            p.vout.v = t + 1
    def reader(p, t, q):
        p.vin.v.initial_value = 7
        p.vout.v = p.vin.v.previous_value
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'producer', producer, execution=execution), 'producer', nets={'vin': 'gnd', 'vout': 'in'})
    mod.add_module(block(tmp_path, 'reader', reader, execution=execution), 'reader', nets={'vin': 'in', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['in'], [1, 0, 3])
    np.testing.assert_array_equal(result['out'], [7, 1, 0])


def test_plain_sample_read_retains_negative_index_publication(tmp_path):
    def producer(p, t, q):
        p.vout.v[-2] = 7
    def reader(p, t, q):
        p.vout.v = p.vin.v
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'producer', producer), 'producer', nets={'vin': 'gnd', 'vout': 'in'})
    mod.add_module(block(tmp_path, 'reader', reader, execution='sample'), 'reader', nets={'vin': 'in', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 3)
    np.testing.assert_array_equal(result['out'], [0, 0, 7, 7])


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('publisher_inout', [False, True])
@pytest.mark.parametrize('backend', [False, True])
@pytest.mark.parametrize('delay', [False, True])
def test_inout_mixed_loop_refusal_boundary(tmp_path, reverse, publisher_inout, backend, delay):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    from silk.exceptions import LoopError
    calls = []

    def first(p, t, q):
        calls.append('sample_reader')
        p.vout.v = (p.vin.v.previous_value if delay else p.vin.v) + 1

    def second(p, t, q):
        calls.append('array_reader')
        p.vout.v = p.vin.v + 2
    ports = {'vin': PortDir.INOUT, 'vout': PortDir.INOUT if publisher_inout else PortDir.OUT}
    a = block(tmp_path, 'sample_reader', first, ports, execution='sample')
    b = block(tmp_path, 'array_reader', second, ports)
    placements = [('sample_reader', a, {'vin': 'b', 'vout': 'a'}), ('array_reader', b, {'vin': 'a', 'vout': 'b'})]
    mod = top(tmp_path)
    for name, dut, nets in reversed(placements) if reverse else placements:
        mod.add_module(dut, name, nets=nets)
    sentinel = object()
    result = sentinel
    with pytest.warns(UserWarning, match='INOUT'):
        with pytest.raises(LoopError) as caught:
            if backend:
                from silk.simulation import Transient
                mod.add_sim_type(Transient(step=1, stop=2))
                result = mod.simulate()
            else:
                result = BehavioralEngine().run(mod, 1, 2)
    assert type(caught.value) is LoopError
    message = str(caught.value)
    assert all((text in message for text in ['sample_reader', 'array_reader', 'same execution=', 'one function']))
    assert result is sentinel
    assert mod.results_file is None
    assert not mod.sim_results_loaded
    expected = ['array_reader'] + ['sample_reader'] * 3 if reverse else ['sample_reader'] * 3 + ['array_reader']
    assert calls == (expected if publisher_inout else [])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
@pytest.mark.parametrize('dc', [False, True])
@pytest.mark.parametrize('feedback', [False, True])
def test_voltage_bus_selection_records_only_requested_rows(tmp_path, execution, dc, feedback):
    from silk.exceptions import LoopError
    def a(p, t, q):
        p.vout.v = p.code.v[1 if feedback else 0]
    def b(p, t, q):
        p.vout.v = 2 * p.vin.v
    aa = block(tmp_path, 'bus_reader', a, execution=execution)
    aa.define_module_ports({'code': Port.digital_input(bits=range(2)), 'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_source(VoltageSource('drive', positive='source', negative='gnd', dc=2))
    mod.add_module(aa, 'bus_reader', nets={'code[0]': 'source', 'code[1]': 'feedback', 'vout': 'out'})
    mod.add_module(block(tmp_path, 'follower', b, execution=execution), 'follower', nets={'vin': 'out', 'vout': 'feedback'})
    with warnings.catch_warnings(record=True) as caught:
        if dc and feedback:
            with pytest.raises(LoopError, match='DC feedback at point 0'):
                BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
        elif dc:
            result, _, _ = BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
            np.testing.assert_array_equal(result['out'], [0, 1])
            np.testing.assert_array_equal(result['feedback'], [0, 2])
        else:
            result, _ = BehavioralEngine().run(mod, 1, 2)
            np.testing.assert_array_equal(result['out'], [0, 0, 0] if feedback else [2, 2, 2])
            np.testing.assert_array_equal(result['feedback'], [0, 0, 0] if feedback else [4, 4, 4])
    diagnostics = [str(item.message) for item in caught if 'Feedback without' in str(item.message)]
    assert len(diagnostics) == int(feedback and not dc)


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_unrelated_voltage_bus_row_does_not_undo_explicit_delay(tmp_path, execution):
    def a(p, t, q):
        p.vout.v = p.code.v[0] + p.code.v.previous_value[1]
    def b(p, t, q):
        p.vout.v = 2 * p.vin.v
    aa = block(tmp_path, 'bus_reader', a, execution=execution)
    aa.define_module_ports({'code': Port.digital_input(bits=range(2)), 'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_source(VoltageSource('drive', positive='source', negative='gnd', dc=2))
    mod.add_module(aa, 'bus_reader', nets={'code[0]': 'source', 'code[1]': 'feedback', 'vout': 'out'})
    mod.add_module(block(tmp_path, 'follower', b, execution=execution), 'follower', nets={'vin': 'out', 'vout': 'feedback'})
    with warnings.catch_warnings(record=True) as caught:
        result, _ = BehavioralEngine().run(mod, 1, 2)
    assert not any('Feedback without' in str(item.message) for item in caught)
    expected = [2, 2, 2] if execution == 'vector' else [2, 6, 14]
    np.testing.assert_array_equal(result['out'], expected)
    np.testing.assert_array_equal(result['feedback'], np.asarray(expected) * 2)



def test_plain_feedback_read_leaves_next_producer_sample_unpublished(tmp_path):
    from silk.simulation._analysis_context import analysis_scope
    from silk.simulation._behavioral_state import SharedState
    from silk.simulation._behavioral_views import PortsView
    state = SharedState(3, {('reader', 'vin'): 'b', ('producer', 'vout'): 'b'})
    dut = block(tmp_path, 'reader', lambda p, t, q: None, execution='sample')
    state.write_voltage('producer', 'vout', .5, index=0)
    with analysis_scope('transient'):
        view = PortsView(state, 'reader', dut, index=1)
        assert float(view.vin.v) == .5
    quantity = state._voltage('b')
    assert quantity.value[1] == 0
    assert not quantity.assigned[1]
    state.write_voltage('producer', 'vout', .75, index=1)
    np.testing.assert_array_equal(quantity.value, [.5, .75, 0])




@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('publisher_inout', [False, True])
def test_inout_admission_does_not_reorder_mixed_feedforward_models(tmp_path, reverse, publisher_inout):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []

    def reader(p, t, q):
        calls.append('reader')
        p.vout.v = p.vin.v

    def producer(p, t, q):
        calls.append('producer')
        p.vout.v = 3
    ports = {'vin': PortDir.INOUT, 'vout': PortDir.INOUT if publisher_inout else PortDir.OUT}
    aa = block(tmp_path, 'reader', reader, ports, execution='sample')
    bb = block(tmp_path, 'producer', producer, ports if publisher_inout else None)
    mod = top(tmp_path)
    placements = [('reader', aa, {'vin': 'in', 'vout': 'out'}), ('producer', bb, {'vin': 'gnd', 'vout': 'in'})]
    for name, dut, nets in reversed(placements) if reverse else placements:
        mod.add_module(dut, name, nets=nets)
    with pytest.warns(UserWarning, match='INOUT') as caught:
        result, _ = BehavioralEngine().run(mod, 1, 2)
    assert len(caught) == (4 if publisher_inout else 1)
    assert calls == (['producer', 'reader', 'reader', 'reader'] if reverse else ['reader', 'reader', 'reader', 'producer'])
    np.testing.assert_array_equal(result['out'], [3, 3, 3] if reverse else [0, 0, 0])
