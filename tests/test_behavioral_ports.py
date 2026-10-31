"""Electrical port views through normal behavioral dispatch."""
from functools import partial

import numpy as np
import pytest

from silk import PortDir, Port, analysis
from silk.circuit.modules import Module
from silk.exceptions import SimulationError, RewriteError, ConflictError, KCLError, CircuitError, NotSupportedError
from silk.simulation import DC, Transient
from silk.simulation.behavioral import BehavioralEngine
from silk.sources import VoltageSource
from tests.test_behavioral_rawresult import _pdk, _block, _top


def block(tmp_path, name, fn, ports=None, execution='vector', params=None):
    mod = _block(tmp_path, fn, ports=ports, name=name)
    mod.set_behavioral_model(fn, params=params, execution=execution)
    return mod


def top(tmp_path):
    mod = Module('PDK_skywater130', str(tmp_path / 'top'), 'top.cir', pdk_details=_pdk())
    mod.simulator = 'behavioral'
    return mod


@pytest.mark.parametrize('execution', ['vector', 'sample'])
@pytest.mark.parametrize('bus', [False, True])
@pytest.mark.parametrize('name', ['_state', '_path', '_index', '_module',
                                  '_reject_dictionary_access', '__x'])
def test_declared_port_names_take_precedence(tmp_path, execution, bus, name):
    """Private-looking port names must publish values into the result nets."""
    def model(p, t, q):
        assert p.__class__.__name__ == 'PortsView'
        with pytest.raises(AttributeError, match='no declared port.*missing'):
            p.missing
        with pytest.raises(SimulationError, match='ports are electrical pins'):
            p[name]
        with pytest.raises(SimulationError, match='ports are electrical pins'):
            p[name] = 0
        value = getattr(p, name)
        if bus:
            value[2].v = t + 1
            value[3].v = t + 5
        else:
            value.v = t + 1
        # A cached view still refers to the electrical pin, including _module.
        assert getattr(p, name) is value

    ports = {name: Port.digital_output(bits=range(2, 4)) if bus else PortDir.OUT}
    dut = block(tmp_path, 'names', model, ports=ports, execution=execution)
    mod = top(tmp_path)
    wiring = ({name + '[2]': 'left', name + '[3]': 'right'} if bus else {name: 'out'})
    mod.add_module(dut, 'names', nets=wiring)
    mod.add_sim_type(Transient(step=1, stop=2))
    mod.simulate()
    np.testing.assert_array_equal(mod.load_signal('left' if bus else 'out').y, [1, 2, 3])
    if bus:
        np.testing.assert_array_equal(mod.load_signal('right').y, [5, 6, 7])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
@pytest.mark.parametrize('dc', [False, True])
def test_ports_voltage_current_params_and_call_contract(tmp_path, execution, dc):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    def model(p, t, params):
        calls.append((analysis(), np.shape(t), float(t) if np.ndim(t) == 0 else t.copy()))
        assert params.gain == params['gain'] == 3
        p.vout.v = params.gain * p.vin.v
        p.vout.i = p.vin.v * 0.01
        np.testing.assert_allclose(p.vout.i, np.asarray(p.vin.v) * 0.01)
    dut = block(tmp_path, 'gain', model, execution=execution, params={'gain': 3})
    mod = top(tmp_path)
    mod.add_module(dut, 'gain', nets={'vin': 'in', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0.5))
    if dc:
        values, axis, _ = BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 0.5))
        np.testing.assert_allclose(values['out'], [0, 1.5, 3])
        assert len(calls) == 3
        assert [float(x[2]) if execution == 'sample' else x[2][0] for x in calls] == [0, 1, 2]
    else:
        values, axis = BehavioralEngine().run(mod, 1, 2)
        np.testing.assert_allclose(values['out'], [1.5] * 3)
        assert len(calls) == (1 if execution == 'vector' else 3)
    assert all(c[0] == ('dc' if dc else 'transient') for c in calls)
    assert all(c[1] == (() if execution == 'sample' else ((1,) if dc else (3,))) for c in calls)


def test_ports_returns_must_be_none_and_name_model(tmp_path):
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'bad', lambda p, t, q: {}), 'bad',
                   nets={'vin': 'in', 'vout': 'out'})
    with pytest.raises(SimulationError, match='bad.*None'):
        BehavioralEngine().run(mod, 1, 2)


def test_indexed_assignments_copy_and_enforce_one_write(tmp_path):
    def model(p, t, q):
        local = np.arange(len(t), dtype=float)
        p.vout.v[:] = local
        local[:] = 99
        alias = np.asarray(p.vout.v)
        try:
            alias[:] = 77
        except ValueError:
            pass
        np.testing.assert_array_equal(p.vout.v, [0, 1, 2])
        with pytest.raises(RewriteError, match='local.*once'):
            p.vout.v[1] = 3
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'copy', model), 'copy', nets={'vin': 'in', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [0, 1, 2])


def test_unset_voltage_is_zero_without_completeness_error(tmp_path):
    def model(p, t, q):
        np.testing.assert_array_equal(p.vout.v, [0, 0, 0])
        p.vout.v[1] = 4
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'partial', model), 'partial', nets={'vin': 'in', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [0, 4, 0])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
@pytest.mark.parametrize('bits', [range(8, 11), range(8, 9)])
def test_bus_views_shapes_declared_indices_and_current(tmp_path, execution, bits):
    from silk.circuit.circuit_elements import PortRole
    names = list(bits)
    def model(p, t, q):
        expected = (len(names), len(t)) if execution == 'vector' else (len(names),)
        assert np.shape(p.code.v) == expected
        p.code.i = np.zeros(expected)
        assert np.shape(p.code.i) == expected
        with pytest.raises(ValueError, match='bus.*code'):
            p.code.v = np.zeros((len(names) + 1,))
        for i, k in enumerate(names):
            p.code[k].v = i + 2
        np.testing.assert_allclose(np.asarray(p.code.v)[0], 2)
        with pytest.raises(CircuitError):
            p.code[0]
    dut = block(tmp_path, 'bus', model, execution=execution)
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'code': Port.digital_output(bits=bits)})
    mod = top(tmp_path)
    mod.add_module(dut, 'bus', nets={'vin': 'in', 'vout': 'unused',
                                    **{f'code[{k}]': f'bit{k}' for k in names}})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    for i, k in enumerate(names):
        np.testing.assert_allclose(result[f'bit{k}'], i + 2)


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_whole_bus_voltage_assignment(tmp_path, execution):
    def model(p, t, q):
        p.code.v = np.array([[2, 3, 4]]) if execution == 'vector' else np.array([t + 2])
    dut = block(tmp_path, 'bus', model, execution=execution)
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'code': Port.digital_output(bits=1)})
    mod = top(tmp_path)
    mod.add_module(dut, 'bus', nets={'vin': 'in', 'vout': 'unused', 'code[0]': 'bit'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['bit'], [2, 3, 4])


def test_declared_dependency_order_and_mixed_contracts(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    def driver(p, t, q):
        calls.append(('driver', t))
        p.vout.v = t + 1
    def reader(p, t, q):
        calls.append(('reader', len(t)))
        p.vout.v = 2 * p.vin.v
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'reader', reader), 'reader', nets={'vin': 'mid', 'vout': 'out'})
    mod.add_module(block(tmp_path, 'driver', driver, execution='sample'), 'driver',
                   nets={'vin': 'in', 'vout': 'mid'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [2, 4, 6])
    assert calls == [('driver', 0), ('driver', 1), ('driver', 2), ('reader', 3)]


def test_sample_models_interleave_in_model_order(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    def first(p, t, q):
        calls.append(('first', t))
        p.vout.v = t + 1
    def second(p, t, q):
        calls.append(('second', t))
        p.vout.v = p.vin.v * 2
    mod = top(tmp_path)
    for name, fn, a, y in [('first', first, 'in', 'mid'), ('second', second, 'mid', 'out')]:
        mod.add_module(block(tmp_path, name, fn, execution='sample'), name, nets={'vin': a, 'vout': y})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    assert calls == [('first', 0), ('second', 0), ('first', 1), ('second', 1), ('first', 2), ('second', 2)]
    np.testing.assert_array_equal(result['out'], [2, 4, 6])


def test_cml_sources_taps_load_then_voltage_reader(tmp_path, monkeypatch):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    from silk.simulation._behavioral_elements import AlgebraicModel
    calls = []
    original = AlgebraicModel._invoke_behavioral_model
    def record_source(device, inputs, t):
        result = original(device, inputs, t)
        if device.independent:
            calls.append('source:' + device.path)
        return result
    monkeypatch.setattr(AlgebraicModel, '_invoke_behavioral_model', record_source)
    mod = top(tmp_path)
    for name in ['pre', 'main', 'post1', 'post2']:
        mod.add_source(VoltageSource(name, positive=name, negative='gnd', dc=1))
    mod.add_source(VoltageSource('supply', positive='vdd', negative='gnd', dc=1.8))
    ports = {'D': PortDir.IN, 'VDD': PortDir.POWER, 'OUTP': PortDir.OUT, 'OUTN': PortDir.OUT}
    for name, mult, negative in [('pre', 15, True), ('main', 40, False), ('post1', 24, True), ('post2', 4, False)]:
        def tap(p, t, q):
            calls.append(q.name)
            # All source equations, including the last shared VDD, have completed.
            np.testing.assert_allclose(p.D.v, 1)
            np.testing.assert_allclose(p.VDD.v, 1.8)
            value = q.mult * 200e-6 * p.D.v
            p.OUTP.i = value if q.negative else 0
            p.OUTN.i = 0 if q.negative else value
        dut = block(tmp_path, name, tap, ports=ports,
                    params={'name': name, 'mult': mult, 'negative': negative})
        mod.add_module(dut, name, nets={'D': name, 'VDD': 'vdd', 'OUTP': 'outp', 'OUTN': 'outn'})
    def load(p, t, q):
        calls.append('load')
        np.testing.assert_allclose(p.OUTP.i, -0.0078)
        np.testing.assert_allclose(p.OUTN.i, -0.0088)
        p.OUTP.v = p.VDD.v + q.rl * p.OUTP.i
        p.OUTN.v = p.VDD.v + q.rl * p.OUTN.i
    mod.add_module(block(tmp_path, 'load', load,
                         ports={k: v for k, v in ports.items() if k != 'D'}, params={'rl': 50}),
                   'load', nets={'VDD': 'vdd', 'OUTP': 'outp', 'OUTN': 'outn'})
    def reader(p, t, q):
        calls.append('reader')
        p.vout.v = p.vin.v
    mod.add_module(block(tmp_path, 'reader', reader), 'reader', nets={'vin': 'outp', 'vout': 'observed'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    assert calls == ['source:pre', 'source:main', 'source:post1', 'source:post2',
                     'source:supply', 'pre', 'main', 'post1', 'post2', 'load', 'reader']
    np.testing.assert_allclose(result['outp'], 1.41)
    np.testing.assert_allclose(result['outn'], 1.36)
    np.testing.assert_allclose(result['observed'], 1.41)


def test_ports_mixed_execution_cycle_refused_before_any_call(tmp_path):
    def fn(p, t, q):
        raise AssertionError('model invoked before cycle refusal')
    mod = top(tmp_path)
    for name, a, y in [('a', 'y', 'x'), ('b', 'x', 'y')]:
        mod.add_module(block(tmp_path, name, fn, execution='sample' if name == 'a' else 'vector'),
                       name, nets={'vin': a, 'vout': y})
    with pytest.raises(SimulationError, match='same execution='):
        BehavioralEngine().run(mod, 1, 2)


def test_ports_failure_after_assignment_publishes_no_result(tmp_path):
    mod = _top(tmp_path)
    def fail(p, t, q):
        p.vout.v = 4
        raise RuntimeError('failed after assignment')
    mod.elem_dict['g1']['device'].set_behavioral_model(fail, execution='vector')
    with pytest.raises(RuntimeError, match='failed after assignment'):
        mod.simulate()
    assert mod.results_file is None
    assert not mod.sim_results_loaded


class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, p, t, q):
        self.count += 1
        p.vout.v = self.count


def closure_counter():
    count = 0
    def model(p, t, q):
        nonlocal count
        count += 1
        p.vout.v = count
    return model


def shared_cell_counter():
    count = 0
    def advance():
        nonlocal count
        count += 1
    def model(p, t, q):
        advance()
        p.vout.v = count
    return model


def array_closure_counter():
    import numpy as local_np
    state = local_np.zeros(1)
    def model(p, t, q):
        state[0] += 1
        p.vout.v = local_np.full_like(t, state[0])
    return model


def indirect_closure_counter(shape):
    count = 0
    def advance(step=1):
        nonlocal count
        count += step
        return count
    def read():
        return count
    if shape == 'partial':
        tick = partial(advance, 1)
    elif shape == 'partial_attributes':
        tick = partial(advance, 1)
        tick.scale = 2.0
        def model(p, t, q):
            p.vout.v = tick.scale * tick()
        return model
    elif shape == 'default':
        def model(p, t, q, tick=advance):
            p.vout.v = tick()
        return model
    elif shape == 'kwdefault':
        def model(p, t, q, *, tick=advance):
            p.vout.v = tick()
        return model
    elif shape == 'shared_helpers':
        helpers = [advance, read]
        def model(p, t, q):
            helpers[0]()
            p.vout.v = helpers[1]()
        return model
    else:
        kind, wrapped = shape.split('_')
        tick = partial(advance, 1) if wrapped == 'partial' else advance
        if kind == 'dict':
            holder = {'tick': tick}
            def model(p, t, q):
                p.vout.v = holder['tick']()
        else:
            holder = [tick] if kind == 'list' else (tick,)
            def model(p, t, q):
                p.vout.v = holder[0]()
        return model
    def model(p, t, q):
        p.vout.v = tick()
    return model


@pytest.mark.parametrize('factory,expected', [(factory, [1, 2, 3]) for factory in
                                    [Counter, closure_counter, array_closure_counter,
                                     shared_cell_counter, lambda: partial(closure_counter()),
                                     *[partial(indirect_closure_counter, shape) for shape in
                                       ['partial', 'default', 'kwdefault', 'list_function',
                                        'list_partial', 'dict_function', 'dict_partial',
                                        'tuple_function', 'tuple_partial', 'shared_helpers']]]]
                         + [pytest.param(partial(indirect_closure_counter, 'partial_attributes'),
                                         [2, 4, 6], id='partial_attributes')])
@pytest.mark.parametrize('placements', [1, 2])
@pytest.mark.parametrize('dc', [False, True])
def test_each_placement_has_independent_callable_state(tmp_path, factory, expected, dc, placements):
    mod = top(tmp_path)
    dut = block(tmp_path, 'counter', factory(), execution='sample')
    for name in ['a', 'b'][:placements]:
        mod.add_module(dut, name, nets={'vin': 'in', 'vout': name})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    engine = BehavioralEngine()
    for candidate in (engine, engine, BehavioralEngine()):
        if dc:
            values, _, _ = candidate.run_dc(mod, DC('drive', 0, 1, 0.5))
        else:
            values, _ = candidate.run(mod, 1, 2)
        for name in ['a', 'b'][:placements]:
            np.testing.assert_array_equal(values[name], expected)


def test_attachment_contract_is_fixed_during_run(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global calls, dut
    calls = []
    def replacement(p, t, q):
        raise AssertionError('attachment changed mid-run')
    def model(p, t, q):
        calls.append(t)
        dut.set_behavioral_model(replacement, execution='vector')
        p.vout.v = t
    dut = block(tmp_path, 'fixed', model, execution='sample')
    mod = top(tmp_path)
    mod.add_module(dut, 'fixed', nets={'vin': 'in', 'vout': 'out'})
    values, _ = BehavioralEngine().run(mod, 1, 2)
    assert calls == [0, 1, 2]
    np.testing.assert_array_equal(values['out'], [0, 1, 2])


def test_ports_inout_warns_and_retains_placement_order(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global calls
    calls = []
    def one(p, t, q):
        calls.append('one')
        p.vout.v = p.vin.v + 1
    def two(p, t, q):
        calls.append('two')
        p.vout.v = p.vin.v + 2
    mod = top(tmp_path)
    ports = {'vin': PortDir.INOUT, 'vout': PortDir.OUT}
    mod.add_module(block(tmp_path, 'one', one, ports), 'one', nets={'vin': 'b', 'vout': 'a'})
    mod.add_module(block(tmp_path, 'two', two, ports), 'two', nets={'vin': 'a', 'vout': 'b'})
    with pytest.warns(UserWarning) as caught:
        values, _ = BehavioralEngine().run(mod, 1, 2)
    assert sum('INOUT' in str(w.message) for w in caught) == 2
    assert sum('Feedback' in str(w.message) for w in caught) == 1
    assert calls == ['one', 'two']
    np.testing.assert_allclose(values['a'], 1)
    np.testing.assert_allclose(values['b'], 3)


def test_ports_conflicting_voltage_writers_name_both(tmp_path):
    def fn(p, t, q):
        p.vout.v = 1
    mod = top(tmp_path)
    for name in ['first', 'second']:
        mod.add_module(block(tmp_path, name, fn), name, nets={'vin': 'in', 'vout': 'shared'})
    with pytest.raises(ConflictError, match='shared.*first.*second'):
        BehavioralEngine().run(mod, 1, 2)


def test_two_unknown_currents_refuse_through_ports(tmp_path):
    def fn(p, t, q):
        np.asarray(p.vout.i)
    mod = top(tmp_path)
    for name in ['first', 'second']:
        mod.add_module(block(tmp_path, name, fn), name, nets={'vin': 'in', 'vout': 'shared'})
    with pytest.raises(KCLError, match='second unknown'):
        BehavioralEngine().run(mod, 1, 2)


def test_static_refusal_preserves_prior_results_and_failed_run_keeps_history(tmp_path):
    from pathlib import Path
    mod = _top(tmp_path)
    state = {'fail': False}
    def fn(p, t, q):
        p.vout.v = p.vin.v * 2
        if state['fail']:
            raise RuntimeError('failed after assignment')
    mod.elem_dict['g1']['device'].set_behavioral_model(fn, execution='vector')
    mod.set_output_options(mode='new')
    mod.simulate()
    previous = mod.results_file
    evidence = Path(previous).read_bytes()
    mod.add_sim_type(DC('missing', 0, 1, 1))
    with pytest.raises(SimulationError, match='missing'):
        mod.simulate()
    assert mod.results_file == previous
    assert Path(previous).read_bytes() == evidence
    mod.sim_analysis_dict.pop('dc', None)
    # Analysis dictionaries use analysis keys; restore only the transient request.
    mod.sim_analysis_dict = {k: v for k, v in mod.sim_analysis_dict.items() if isinstance(v, Transient)}
    state['fail'] = True
    with pytest.raises(RuntimeError, match='failed after assignment'):
        mod.simulate()
    assert mod.results_file is None
    assert Path(previous).read_bytes() == evidence


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_ports_monte_carlo_uses_each_paths_resolution_and_replays(pdk, tmp_path, execution):
    from silk.simulation.variation import Normal
    from tests._variation_bench import chip
    mod, tb = chip(pdk, tmp_path, {'offset': Normal(0, 0.1)})
    def model(p, t, q):
        assert q.offset == q['offset']
        p.y.v = p.a.v + q.offset
    mod.elem_dict['c0']['device'].set_behavioral_model(
        model, params={'offset': Normal(0, 0.1)}, execution=execution)
    campaign = tb.set_monte_carlo(n=2, seed=588).run()
    key = next(iter(campaign.keys()))
    metrics = campaign[key].metrics
    assert len({metrics['y0'], metrics['y1'], metrics['y2']}) == 3
    assert campaign.rerun(key).metrics == pytest.approx(metrics)


def test_modeled_leaf_hides_contents_and_internal_probes(tmp_path):
    mod = _top(tmp_path)
    dut = mod.elem_dict['g1']['device']
    with dut:
        dut.pdk_details.nmos1p8('hidden', d='hidden', g='vin', s='gnd', b='gnd', w=1, l=0.5)
    def model(p, t, q):
        p.vout.v = p.vin.v * 2
    dut.set_behavioral_model(model, execution='vector')
    mod.simulate()
    np.testing.assert_allclose(mod.load_signal('vout').y, 0.8)
    with pytest.raises(NotSupportedError, match='internal|not found'):
        mod.load_signal('xg1.hidden')


@pytest.mark.parametrize('publish', [True, False])
def test_ports_production_obeys_source_reference_precondition(tmp_path, publish):
    def model(p, t, q):
        if publish:
            p.vout.v = 1
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'ref', model), 'ref', nets={'vin': 'gnd', 'vout': 'ref'})
    mod.add_source(VoltageSource('offset', positive='out', negative='ref', dc=0.5))
    if publish:
        values, _ = BehavioralEngine().run(mod, 1, 2)
        np.testing.assert_allclose(values['out'], 1.5)
    else:
        with pytest.raises(NotSupportedError, match='offset.*required voltage producer'):
            BehavioralEngine().run(mod, 1, 2)


@pytest.mark.parametrize('first', ['dc', 'transient'])
def test_ports_analysis_follows_actual_multi_analysis_dispatch(tmp_path, first):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    seen = []
    def model(p, t, q):
        seen.append(analysis())
        p.vout.v = p.vin.v
    mod = _top(tmp_path)
    mod.elem_dict['g1']['device'].set_behavioral_model(model, execution='vector')
    dc = DC('vs', 0, 1, 0.5)
    transient = Transient(step=1, stop=2)
    mod.sim_analysis_dict = {}
    for request in ([dc, transient] if first == 'dc' else [transient, dc]):
        mod.add_sim_type(request)
    mod.simulate()
    assert seen == (['dc'] * 3 + ['transient'] if first == 'dc' else ['transient'] + ['dc'] * 3)
    with pytest.raises(SimulationError):
        analysis()


def test_indexed_bus_voltage_publication_uses_shared_write_checks(tmp_path):
    def model(p, t, q):
        p.code.v[:, 0] = [1, 2]
        p.code.v[:, 1:] = [[3, 4], [5, 6]]
        with pytest.raises(RewriteError):
            p.code.v[0, 0] = 9
        alias = np.asarray(p.code.v)
        alias[:] = 99
        np.testing.assert_array_equal(p.code.v, [[1, 3, 4], [2, 5, 6]])
    dut = block(tmp_path, 'indexed_bus', model)
    dut.define_module_ports({'code': Port.digital_output(bits=range(8, 10))})
    mod = top(tmp_path)
    mod.add_module(dut, 'bus', nets={'vin': 'unused1', 'vout': 'unused2',
                                    'code[8]': 'bit8', 'code[9]': 'bit9'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['bit8'], [1, 3, 4])
    np.testing.assert_array_equal(result['bit9'], [2, 5, 6])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_input_bus_reads_connected_voltages_in_declared_order(tmp_path, execution):
    def model(p, t, q):
        values = np.asarray(p.code.v)
        np.testing.assert_allclose(values[0], 3)
        np.testing.assert_allclose(values[1], 2)
        p.vout.v = p.code[8].v + 10 * p.code[9].v
    dut = block(tmp_path, 'input_bus', model, execution=execution)
    dut.define_module_ports({'code': Port.digital_input(bits=range(8, 10))})
    mod = top(tmp_path)
    mod.add_source(VoltageSource('b8', positive='bit8', negative='gnd', dc=3))
    mod.add_source(VoltageSource('b9', positive='bit9', negative='gnd', dc=2))
    mod.add_module(dut, 'bus', nets={'vin': 'unused', 'vout': 'out', 'code[8]': 'bit8', 'code[9]': 'bit9'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_allclose(result['out'], 23)




def test_bus_current_read_fills_only_selected_sample(tmp_path):
    def model(p, t, q):
        for k in range(len(t)):
            p.drive.i[0, k] = k + 1
            p.vout.v[k] = p.sense.i[0, k]
        np.testing.assert_array_equal(p.drive.i, [[1, 2, 3]])
        np.testing.assert_array_equal(p.sense.i, [[-1, -2, -3]])
    dut = block(tmp_path, 'sliced', model)
    dut.define_module_ports({'drive': Port.digital_output(bits=range(1)),
                             'sense': Port.digital_input(bits=range(1)),
                             'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_module(dut, 'sliced', nets={'drive[0]': 'wire', 'sense[0]': 'wire',
                                       'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [-1, -2, -3])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_bus_current_read_does_not_fill_unselected_pin(tmp_path, execution):
    def model(p, t, q):
        np.testing.assert_array_equal(p.sense.i[0], np.zeros_like(t))
        # This write must remain available: reading row zero cannot fill row one.
        p.sense[1].i = 7
        p.vout.v = p.sense.i[1]
    dut = block(tmp_path, 'rows', model, execution=execution)
    dut.define_module_ports({'sense': Port.digital_input(bits=range(2)),
                             'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_module(dut, 'rows', nets={'sense[0]': 'a', 'sense[1]': 'b', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [7, 7, 7])


@pytest.mark.parametrize('first', ['dc', 'transient'])
def test_attachment_is_fixed_across_all_analyses_in_backend_run(tmp_path, first):
    # Module globals deliberately observe writes from run-local model copies.
    global calls, dut
    calls = []
    def replacement(p, t, q):
        assert np.ndim(t) == 1
        calls.append(('replacement', analysis(), len(t)))
        p.vout.v = 9
    def original(p, t, q):
        assert np.ndim(t) == 0
        calls.append(('original', analysis(), float(t)))
        dut.set_behavioral_model(replacement, execution='vector')
        p.vout.v = t + 1
    mod = top(tmp_path)
    dut = block(tmp_path, 'fixed', original, execution='sample')
    mod.add_module(dut, 'fixed', nets={'vin': 'in', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    analyses = {'dc': DC('drive', 0, 1, 0.5),
                'transient': Transient(step=1, stop=2)}
    order = [first, 'transient' if first == 'dc' else 'dc']
    for kind in order:
        mod.add_sim_type(analyses[kind])
    mod.simulate()
    assert calls == [('original', kind, float(k)) for kind in order for k in range(3)]
    for kind in order:
        np.testing.assert_array_equal(mod.load_signal('out', analysis='tran' if kind == 'transient' else kind).y, [1, 2, 3])
    calls.clear()
    mod.simulate()
    assert calls == [entry for kind in order for entry in
                     ([('replacement', kind, 1)] * 3 if kind == 'dc'
                      else [('replacement', kind, 3)])]
    for kind in order:
        np.testing.assert_array_equal(mod.load_signal('out', analysis='tran' if kind == 'transient' else kind).y, [9, 9, 9])


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_integer_bus_voltage_is_read_as_physical_voltage(tmp_path, execution):
    def producer(ports, t, params):
        ports.code.v = np.arange(len(t), dtype=np.int16)[None, :]
    def reader(p, t, q):
        values = np.asarray(p.code.v)
        assert values.dtype == np.dtype(float)
        p.vout.v = values[0].astype(np.int16) & 1
    source = _block(tmp_path, producer, name='producer')
    source.define_module_ports({'code': Port.digital_output(bits=range(1))})
    sink = block(tmp_path, 'reader', reader, execution=execution)
    sink.define_module_ports({'code': Port.digital_input(bits=range(1)),
                              'vout': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_module(source, 'producer', nets={'code[0]': 'code'})
    mod.add_module(sink, 'reader', nets={'code[0]': 'code', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 3)
    assert result['code'].dtype == np.dtype(float)
    np.testing.assert_array_equal(result['out'], [0, 1, 0, 1])


def test_source_follower_ports_model_uses_calibrated_gain_offset_and_pole():
    from types import SimpleNamespace
    from silk.examples.source_follower_example import build_source_follower, _sf_behavioral_model

    dut = build_source_follower(with_tests=False)
    for name, value in {'gain': 2.0, 'offset': 0.3,
                        'bw_hz': np.log(2) / (2 * np.pi)}.items():
        dut.calibrate_behavioral(name, value)
    pins = SimpleNamespace(VIN=SimpleNamespace(v=np.array([0.5, 1.5, 1.5])),
                           VOUT=SimpleNamespace())
    result = _sf_behavioral_model(pins, np.array([0., 1., 2.]),
                                  SimpleNamespace(**dut._behavioral_params))
    assert result is None
    np.testing.assert_allclose(pins.VOUT.v, [0.7, 1.7, 2.2])


def test_ldo_ports_model_uses_calibrated_supply_clamp_and_slew():
    from types import SimpleNamespace
    from silk.examples.ldo_regulator_example import build_ldo, _LDOModel

    dut = build_ldo(with_tests=False)
    for name, value in {'gain': 2., 'bw_hz': 1 / (2 * np.pi), 'zeta': 0.5,
                        'psrr_db': 20., 'vreg_max': 1.5, 'vdd_nom': 3.,
                        'slew_v_per_us': 0.25e-6}.items():
        dut.calibrate_behavioral(name, value)
    from silk.simulation._analysis_context import analysis_scope
    model = _LDOModel()
    values = []
    with analysis_scope('transient'):
        for t, v in enumerate([0.4, 1., 1., 1.]):
            pins = SimpleNamespace(VCTRL=SimpleNamespace(v=v),
                                   VDD=SimpleNamespace(v=5.),
                                   VREG=SimpleNamespace())
            assert model(pins, float(t), SimpleNamespace(**dut._behavioral_params)) is None
            values.append(pins.VREG.v)
    np.testing.assert_allclose(values, [1., 1.25, 1.5, 1.5])


@pytest.mark.parametrize('kinds', [('tran',), ('dc',), ('tran', 'dc'), ('dc', 'tran')])
@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
def test_sample_structural_setup_once_per_backend_run(tmp_path, monkeypatch, kinds, naming):
    """Repeating name resolution/bus expansion per point regresses this count."""
    import inspect
    from collections import Counter
    from silk.circuit.bus_spec import BusSpec
    from silk.simulation import _behavioral_views as views

    counts = Counter()
    def model(p, t, q):
        # Repeated access, both quantities, aliases, and both bus shapes.
        assert float(p.vin.v) == float(p.alias.v)
        p.vout.v = q.gain * p.vin.v
        p.vout.i = 0.25
        assert float(p.vout.i) == 0.25
        p.code.v = np.array([1., 2.])
        p.single.i = np.array([0.])
        assert float(p.code[8].v) == 1
        assert float(p.code[9].v) == 2
        assert float(p.single[4].i) == 0
    dut = block(tmp_path, 'cached', model, execution='sample', params={'gain': 2})
    dut.define_module_ports({'vin': PortDir.IN, 'alias': PortDir.IN,
                             'vout': PortDir.OUT,
                             'code': Port.digital_output(bits=range(8, 10), naming=naming),
                             'single': Port.digital_output(bits=range(4, 5))})
    mod = top(tmp_path)
    mod.add_module(dut, 'cached', nets={'vin': 'in', 'alias': 'in', 'vout': 'out',
                   ('code[8]' if naming == 'bracketed' else 'code_8'): 'b8',
                   ('code[9]' if naming == 'bracketed' else 'code_9'): 'b9',
                   'single[4]': 'b4'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0.5))
    for kind in kinds:
        mod.add_sim_type(Transient(step=1, stop=4) if kind == 'tran'
                         else DC('drive', 0, 1, 0.25))
    original_get = Module.__getattribute__
    original_expand = BusSpec.expanded_names
    original_params = views.ParamsView.__init__
    def getattribute(self, name):
        if name in ('pin_list', '_buses') and inspect.currentframe().f_back.f_code.co_filename.endswith('_behavioral_views.py'):
            counts[name] += 1
        return original_get(self, name)
    def expand(self):
        if inspect.currentframe().f_back.f_code.co_filename.endswith('_behavioral_views.py'):
            counts['expand'] += 1
        return original_expand(self)
    def params(self, value):
        counts['params'] += 1
        original_params(self, value)
    monkeypatch.setattr(Module, '__getattribute__', getattribute)
    monkeypatch.setattr(BusSpec, 'expanded_names', expand)
    monkeypatch.setattr(views.ParamsView, '__init__', params)
    for run in (1, 2):
        mod.simulate()
        assert counts == {'pin_list': run, '_buses': run, 'expand': 2 * run,
                          'params': run}
        for kind in kinds:
            signal = mod.load_signal('out', analysis=kind)
            np.testing.assert_array_equal(signal.y, [1.] * 5 if kind == 'tran'
                                          else [0., .5, 1., 1.5, 2.])


@pytest.mark.parametrize('dc', [False, True])
def test_sample_handles_keep_original_state_and_point(tmp_path, dc):
    saved = []
    def model(p, t, q):
        p.code[8].v = t + 10
        p.vout.v = t + 1
        if t == 0:
            saved.append((p.vout.v, p.code[8].v))
            if not dc:
                saved.append(p.vout.v.previous_value)
                saved[-1].initial_value = -7
        else:
            assert float(saved[0][0]) == 1
            assert float(saved[0][1]) == 10
            if not dc:
                assert float(saved[1]) == -7
    dut = block(tmp_path, 'retained', model, execution='sample')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'code': Port.digital_output(bits=range(8, 9))})
    mod = top(tmp_path)
    mod.add_module(dut, 'retained', nets={'vin': 'in', 'vout': 'out', 'code[8]': 'b8'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    if dc:
        BehavioralEngine().run_dc(mod, DC('drive', 0, 2, 1))
    else:
        BehavioralEngine().run(mod, 1, 2)


def test_caught_sample_current_subscript_keeps_kcl_fill(tmp_path):
    def model(p, t, q):
        with pytest.raises(IndexError) as exc:
            p.vout.i[0]
        assert str(exc.value) == 'too many indices for array: array is 0-dimensional, but 1 were indexed'
        assert float(p.vout.i) == 0
        with pytest.raises(RewriteError) as exc:
            p.vout.i = 1
        assert str(exc.value) == ("Model 'caught' assigned 'caught.vout current' more than once at the same sample. "
                                  'Compute in a local value and assign it to the port once.')
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'caught', model, execution='sample'), 'caught',
                   nets={'vin': 'in', 'vout': 'out'})
    BehavioralEngine().run(mod, 1, 2)


def test_sample_scalar_access_avoids_general_array_setup(tmp_path, monkeypatch):
    from silk.simulation._behavioral_state import SharedState
    original = SharedState._check
    def check(self, quantity, writer, index, label, **kwargs):
        assert isinstance(index, slice), 'sample scalar publication used array checks'
        return original(self, quantity, writer, index, label, **kwargs)
    def model(p, t, q):
        p.vout.v = float(p.vin.v) + 1
        p.vout.i = t
        assert float(p.vout.i) == t
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'scalar', model, execution='sample'), 'scalar',
                   nets={'vin': 'in', 'vout': 'out'})
    monkeypatch.setattr(SharedState, '_check', check)
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['out'], [1, 1, 1])


_SAMPLE_ERROR_OPERATIONS = {
 'unknown': lambda p: p.missing.v,
 'dictionary': lambda p: p['vin'],
 'dictionary-set': lambda p: p.__setitem__('vin', 1),
 'bit-type': lambda p: p.code[True],
 'bit-range': lambda p: p.code[0],
 'bus-shape': lambda p: setattr(p.code, 'v', np.zeros(2)),
 'bus-index': lambda p: p.code.v[7],
 'scalar-read': lambda p: p.vin.v[0],
 'scalar-write': lambda p: p.vout.v.__setitem__(0, 1),
 'scalar-length-one': lambda p: setattr(p.vout, 'v', np.array([1.])),
 'scalar-length-two': lambda p: setattr(p.vout, 'i', np.array([1., 2.])),
 'non-numeric': lambda p: setattr(p.vout, 'v', 'invalid'),
 'conflict': lambda p: setattr(p.vin, 'v', 1),
 'rewrite-v': lambda p: (setattr(p.vout, 'v', 1), setattr(p.vout, 'v', 2)),
 'rewrite-i': lambda p: (setattr(p.vout, 'i', 1), setattr(p.vout, 'i', 2)),
 'alias-rewrite': lambda p: (setattr(p.vout, 'v', 1), setattr(p.alias, 'v', 2)),
 'kcl': lambda p: (float(p.vout.i), float(p.alias.i)),
 'delay-dc': lambda p: p.vout.v.previous_value,
}


@pytest.mark.parametrize('label', list(_SAMPLE_ERROR_OPERATIONS))
def test_sample_pin_errors_keep_baseline_type_and_message(tmp_path, label):
    import json
    from pathlib import Path
    expected = json.loads((Path(__file__).parent / 'fixtures' /
                           'behavioral_sample_errors.json').read_text())[label]
    def model(p, t, q):
        _SAMPLE_ERROR_OPERATIONS[label](p)
    dut = block(tmp_path, 'errors', model, execution='sample')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'alias': PortDir.OUT,
                             'code': Port.digital_output(bits=range(8, 9))})
    mod = top(tmp_path)
    mod.add_module(dut, 'errors', nets={'vin': 'in', 'vout': 'out',
                                       'alias': 'out', 'code[8]': 'b8'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=1))
    with pytest.raises(Exception) as exc:
        if label == 'delay-dc':
            BehavioralEngine().run_dc(mod, DC('drive', 0, 1, 1))
        else:
            BehavioralEngine().run(mod, 1, 2)
    assert [type(exc.value).__name__, str(exc.value)] == expected


def test_sample_scalar_reads_do_not_broadcast_the_grid(tmp_path, monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError('a scalar read used whole-array broadcasting')
    def model(p, t, q):
        # Lazy electrical storage may initialize once, before the guard.
        assert float(p.vin.v) == 0
        with monkeypatch.context() as scoped:
            scoped.setattr(np, 'broadcast_to', forbid)
            assert float(p.vin.v) == 0
            p.vout.v = float(p.vin.v) + 1
            p.vout.i = 0.25
            assert float(p.vout.i) == 0.25
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'readscalar', model, execution='sample'), 'readscalar',
                   nets={'vin': 'in', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 4)
    np.testing.assert_array_equal(result['out'], [1] * 5)


def test_sample_live_parameter_edits_are_not_snapshots(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global params
    params = {'gain': 1}
    def model(p, t, q):
        p.vout.v = q.gain
        params['gain'] += 1
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'live', model, execution='sample', params=params),
                   'live', nets={'vin': 'in', 'vout': 'out'})
    result, _ = BehavioralEngine().run(mod, 1, 3)
    np.testing.assert_array_equal(result['out'], [1, 2, 3, 4])


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason='requires a wider longdouble exponent range')
def test_sample_conversion_warning_precedes_publication(tmp_path):
    def model(p, t, q):
        with np.errstate(over='warn'):
            with pytest.warns(RuntimeWarning, match='overflow encountered in cast'):
                p.vout.v = np.longdouble('1e4000')
        assert float(p.vout.v) == float('inf')
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'overflow', model, execution='sample'), 'overflow',
                   nets={'vin': 'in', 'vout': 'out'})
    BehavioralEngine().run(mod, 1, 1)


def test_writer_identity_setup_is_shared_across_dc_points_and_analyses(tmp_path, monkeypatch):
    from silk.simulation import _behavioral_state as state
    labels = []
    original = state._label
    def label(path):
        labels.append(path)
        return original(path)
    def model(p, t, q):
        p.vout.v = t
        p.vout.i = 0.25
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'writer', model, execution='sample'), 'writer',
                   nets={'vin': 'in', 'vout': 'out'})
    mod.add_source(VoltageSource('drive', positive='in', negative='gnd', dc=0))
    mod.add_sim_type(DC('drive', 0, 1, 0.25))
    mod.add_sim_type(Transient(step=1, stop=4))
    monkeypatch.setattr(state, '_label', label)
    for run in (1, 2):
        mod.simulate()
        assert labels.count('writer') == run
        assert labels.count(('source', 'drive')) == run


@pytest.mark.parametrize('quantity,delayed', [('v', False), ('v', True), ('i', True)])
def test_sample_initial_array_indexed_read_is_detached(tmp_path, quantity, delayed):
    def model(p, t, q):
        value = getattr(p.vout, quantity)
        value.initial_value = np.array(3.)
        reader = value.previous_value if delayed else value
        if t == 0:
            read = reader[...]
            assert np.isscalar(read)
            copy = np.asarray(read)
            copy[...] = 9
            assert float(value.initial_value) == 3
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'copyinitial', model, execution='sample'), 'copyinitial',
                   nets={'vin': 'in', 'vout': 'out'})
    BehavioralEngine().run(mod, 1, 1)


@pytest.mark.parametrize('error_mode', ['warn', 'raise'])
def test_sample_kcl_arithmetic_keeps_array_error_message(tmp_path, error_mode):
    import warnings
    def model(p, t, q):
        p.vout.i = 1e308
        p.alias.i = 1e308
        with np.errstate(over=error_mode):
            if error_mode == 'raise':
                with pytest.raises(FloatingPointError) as exc:
                    float(p.third.i)
                assert str(exc.value) == 'overflow encountered in add'
            else:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    assert float(p.third.i) == -float('inf')
                assert [str(w.message) for w in caught] == ['overflow encountered in add']
    dut = block(tmp_path, 'sum', model, execution='sample')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                             'alias': PortDir.OUT, 'third': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_module(dut, 'sum', nets={'vin': 'in', 'vout': 'out', 'alias': 'out', 'third': 'out'})
    BehavioralEngine().run(mod, 1, 1)


def test_sample_kcl_retains_error_from_another_published_point(tmp_path):
    def producer(p, t, q):
        p.vout.i = np.array([0., 1e308])
        p.alias.i = np.array([0., 1e308])
    def reader(p, t, q):
        # Baseline KCL sums known currents over the grid before publishing
        # the requested point. Even point zero observes the later overflow.
        with np.errstate(over='raise'):
            with pytest.raises(FloatingPointError, match='overflow encountered in add'):
                float(p.vin.i)
        p.vin.i = 0  # failed arithmetic must not fill this current
    mod = top(tmp_path)
    dut = block(tmp_path, 'known', producer)
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT, 'alias': PortDir.OUT})
    mod.add_module(dut, 'known', nets={'vin': 'in', 'vout': 'mid', 'alias': 'mid'})
    mod.add_module(block(tmp_path, 'unknown', reader, execution='sample'), 'unknown',
                   nets={'vin': 'mid', 'vout': 'out'})
    BehavioralEngine().run(mod, 1, 1)


def test_run_cache_does_not_shadow_declared_underscore_ports(tmp_path):
    def model(p, t, q):
        p._layout.v = t + 2
        p._views.v = t + 3
        p._views.i = 0.25
        assert float(p._views.i) == 0.25
    dut = block(tmp_path, 'names', model, execution='sample')
    dut.define_module_ports({'_layout': PortDir.OUT, '_views': PortDir.OUT})
    mod = top(tmp_path)
    mod.add_module(dut, 'names', nets={'_layout': 'left', '_views': 'right'})
    result, _ = BehavioralEngine().run(mod, 1, 2)
    np.testing.assert_array_equal(result['left'], [2, 3, 4])
    np.testing.assert_array_equal(result['right'], [3, 4, 5])
