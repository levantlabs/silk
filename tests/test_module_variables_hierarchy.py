"""TASK-401 — Module.variables() searches placed subcircuits at every level."""
import copy
import inspect
import warnings
from pathlib import Path

import pytest

from silk import Module, load_pdk, sources
from silk.sources import waveforms
from silk.sources.schema import _load_metadata
from silk.circuit.variable import Variable
from silk.elements import elements
from silk.optimize.parameter_space import VariableSpace


def _block(tmp_path, name, *values):
    mod = Module('sky130', str(tmp_path), f'{name}.cir', load_pdk('sky130'),
                 module_name=name)
    mod.define_module_ports({'a': 'inout', 'b': 'inout'})
    for i, v in enumerate(values):
        mod.add(elements.resistor(f'r{i}', p='a', n='b', value=v))
    return mod


def _top(tmp_path, name='top'):
    return Module('sky130', str(tmp_path), f'{name}.cir', load_pdk('sky130'))


def _text(mod):
    mod.generate_netlist()
    return Path(mod._run_files.netlist_path).read_text()


def test_two_level_shared_expression_and_twice_placed(tmp_path):
    a, b, c, d = (Variable(n, 1e3) for n in 'abcd')
    leaf = _block(tmp_path, 'leaf', c, 2 * d)        # d only inside expression
    mid = _block(tmp_path, 'mid', b, a)              # a shared with parent
    mid.add_module(leaf, 'xl', a='a', b='b')
    top = _top(tmp_path)
    top.add(elements.resistor('rt', p='n1', n='0', value=a))
    top.add_module(mid, 'xm1', a='n1', b='0')
    top.add_module(mid, 'xm2', a='n1', b='0')        # same child twice
    got = top.variables()
    assert [v.name for v in got] == ['a', 'b', 'c', 'd']
    assert len({id(v) for v in got}) == 4


def test_testbench_lists_dut_and_own_variables_once(tmp_path):
    w, v = Variable('w', 1e3), Variable('v', 2e3)
    dut = _block(tmp_path, 'dut', w)
    tb = dut.create_test('t')
    tb.add_module_under_test({'a': 'inp', 'b': 'out'})
    tb.netlist.add(elements.resistor('rl', p='out', n='0', value=v))
    tb.netlist.add(elements.resistor('rw', p='inp', n='0', value=w))
    assert sorted(x.name for x in tb.netlist.variables()) == ['v', 'w']
    assert len(tb.netlist.variables()) == 2


def test_variable_space_reaches_child_subckt_text(tmp_path):
    w = Variable('w', 1e3, bounds=(1e3, 9e3))
    child = _block(tmp_path, 'blk', w)
    top = _top(tmp_path)
    top.add_module(child, 'x1', a='n1', b='0')
    space = VariableSpace(top.variables())
    assert 'Rr0 a b 1k' in _text(top)
    space.apply({'w': 5e3})
    text = _text(top)
    assert 'Rr0 a b 5k' in text and 'Rr0 a b 1k' not in text


def test_add_submodule_only_is_not_searched(tmp_path):
    w = Variable('w', 1e3)
    child = _block(tmp_path, 'blk', w)
    top = _top(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        top.add_submodule('c', child)
    assert top.variables() == []


@pytest.mark.parametrize('source_class', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('kind', [
    'pulse', 'pwl', 'sine', 'exponential', 'sffm', 'am', 'trnoise', 'trrandom',
])
def test_every_waveform_discovers_unassigned_values(tmp_path, source_class, kind):
    cls = getattr(waveforms, _load_metadata()['waveforms'][kind])
    if kind == 'pwl':
        a, b, repeat, delay = [Variable('same', 1) for _ in range(4)]
        wave = cls([(a, b * 2), (a, b)], r=repeat, td=delay)
        expected = [a, b, repeat, delay]
    else:
        args = {n: 1 for n in inspect.signature(cls).parameters}
        wave = cls(**args)
        expected = [Variable('same', 1) for _ in args]
        # Each field needs its own identity: omitting any field must fail.
        for i, (name, var) in enumerate(zip(args, expected)):
            setattr(wave, name, var if i % 2 == 0 else var * 2)
    # Avoid changing existing Pulse/PWL constructor validation; discovery
    # must remain safe after callers clear values used in expressions.
    for var in expected:
        var.value = None
    mod = _top(tmp_path)
    mod.add_source(source_class('stim', positive='a', negative='0', **{kind: wave}))
    assert [id(v) for v in mod.variables()] == [id(v) for v in expected]


def test_generic_authored_containers_hierarchy_and_copy(tmp_path):
    class Nested(waveforms._Waveform):
        def __init__(self, values):
            self.values = values

        @property
        def resolved(self):
            raise AssertionError('discovery must never read derived properties')

    scalar, amplitude, time = (Variable(n) for n in ('scalar', 'amplitude', 'time'))
    values = {'levels': [amplitude * 2, (time, amplitude)], 'again': time}
    values['cycle'] = values
    child = _block(tmp_path, 'child', scalar)
    wave = Nested(values)
    # An ownership reference must not pull unrelated Variables into discovery.
    unrelated = _block(tmp_path, 'unrelated', Variable('unrelated'))
    wave.owner = unrelated
    child.add_source(sources.VoltageSource('stim', positive='a', negative='b',
                                           dc=scalar, waveform=wave))
    child.add_source(sources.CurrentSource('again', positive='a', negative='b', waveform=wave))
    top = _top(tmp_path)
    top.add_module(child, 'x1', a='n1', b='0')
    top.add_module(child, 'x2', a='n1', b='0')
    expected = [id(scalar), id(amplitude), id(time)]
    for mod in (child, top, copy.deepcopy(top)):
        assert [id(v) for v in mod.variables()] == expected


@pytest.mark.parametrize('kind,value,first,second', [
    ('resistor', 1e3, '1k', '2k'),
    ('capacitor', 1e-12, '1p', '2p'),
    ('inductor', 1e-6, '1u', '2u'),
])
@pytest.mark.parametrize('registration', ['add', 'add_passive'])
def test_passive_discovery_and_space_in_hierarchy(tmp_path, kind, value, first, second,
                                                 registration):
    var = Variable('value', value, bounds=(value / 2, 4 * value))
    child = _top(tmp_path, 'passive')
    child.module_name = 'passive'
    child.define_module_ports({'a': 'inout', 'b': 'inout'})
    getattr(child, registration)(getattr(elements, kind)('load', p='a', n='b', value=2 * var))
    assert child.variables() == [var]
    top = _top(tmp_path)
    top.add_module(child, 'x1', a='in', b='0')
    top.add_module(child, 'x2', a='in', b='0')
    tb = child.create_test('passive_test')
    tb.add_module_under_test({'a': 'in', 'b': '0'})
    assert top.variables() == tb.netlist.variables() == [var]
    space = VariableSpace(top.variables())
    space.apply({'value': value / 2})
    assert f' a b {first}' in _text(top)
    space.apply({'value': value})
    assert f' a b {second}' in _text(top)


def test_all_parameter_locations_have_first_use_identity_order(tmp_path):
    from silk.simulation.variation import Normal, Uniform
    device, source, passive, shared, same1, same2, child_only = [
        Variable(n) for n in ('device', 'source', 'passive', 'shared', 'same', 'same', 'child')]
    child = _block(tmp_path, 'child', child_only, shared)
    child.set_behavioral_model(lambda *args: None, params={'same': same2})
    top = _top(tmp_path)
    top.add_module(child, 'x1', a='n', b='0')
    top.add_module(child, 'x2', a='n', b='0')
    top.add(elements.resistor('device', p='n', n='0', value=device))
    top.add_source(sources.VCVS('src', out_pos='n', out_neg='0',
                                ctrl_pos='n', ctrl_neg='0', gain=source * 2))
    top.add_passive(elements.resistor('passive', p='n', n='0', value=passive))
    top.set_behavioral_model(lambda *args: None, params={
        'normal': Normal(shared, same1), 'uniform': Uniform(same2, shared)})
    expected = [device, source, passive, shared, same1, same2, child_only]
    assert [id(v) for v in top.variables()] == [id(v) for v in expected]
    assert [id(v) for v in copy.deepcopy(top).variables()] == [id(v) for v in expected]


def test_behavioral_stored_containers_and_callable_boundaries(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from silk.simulation.variation import Normal
    from silk.exceptions import NotSupportedError
    variables = [Variable(str(i)) for i in range(8)]
    hidden = Variable('hidden')

    def model(*args):
        raise AssertionError('model must not execute')

    model.hidden = hidden
    array = np.empty(1, dtype=object)
    array[0] = variables[6] * 2
    record = np.empty(1, dtype=[('data', object)])
    record['data'][0] = variables[7]
    params = {variables[0]: variables[1], 'list': [variables[2]],
              'tuple': (variables[3],), 'set': {variables[4]},
              'frozen': frozenset([variables[5]]), 'array': array,
              'record': record[0], 'owner': SimpleNamespace(value=hidden),
              'callback': model}
    params['cycle'] = params
    top = _top(tmp_path)
    top.set_behavioral_model(model, params=params)
    assert top.variables() == variables
    # Discovery of mutated storage does not relax attachment/run validation.
    params['nested'] = [Normal(hidden, 0)]
    assert top.variables() == variables + [hidden]
    with pytest.raises(NotSupportedError, match='top-level'):
        top.set_behavioral_model(model, params=params)


def test_discovered_behavioral_knobs_change_output_and_declaration_fields(tmp_path):
    from silk.simulation.variation import Normal, Uniform
    from silk.simulation import Transient
    k, nominal, sigma, low, high = [Variable(n, v, bounds=(0, 10)) for n, v in
                                  [('k', 2), ('nominal', 1), ('sigma', 0), ('low', 0), ('high', 2)]]
    child = _top(tmp_path, 'model')
    child.module_name = 'model'
    child.define_module_ports({'a': 'input', 'b': 'output'})
    params = {'k': k, 'offset': Normal(2 * nominal, sigma), 'range': Uniform(low, high)}

    def model(ports, t, params):
        ports.b.v = ports.a.v * params['k'].value + params.offset + params.range

    child.set_behavioral_model(model, params=params)
    top = _top(tmp_path)
    top.add_source(sources.VoltageSource('input', positive='a', negative='0', dc=1))
    top.add_module(child, 'x1', a='a', b='b')
    top.simulator = 'behavioral'
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    assert top.variables() == [k, nominal, sigma, low, high]
    top.simulate()
    assert top.load_signal('b').y[-1] == pytest.approx(5)
    VariableSpace(top.variables()).apply({'k': 3, 'nominal': 2, 'sigma': 1, 'low': 2, 'high': 4})
    top.simulate()
    assert top.load_signal('b').y[-1] == pytest.approx(10)
    assert params['offset'].resolve().fields == (4, 1)
    assert params['range'].resolve().fields == (2, 4)


def test_waveform_uses_the_full_stored_container_inventory(tmp_path):
    import numpy as np
    from silk.simulation.variation import Uniform
    a, b, c, d = [Variable(n) for n in 'abcd']
    wave = waveforms._Waveform()
    record = np.empty(2, dtype=[('field', object)])
    record['field'][0], record['field'][1] = b * 2, c
    wave.values = {a: [record, frozenset([d])]}
    wave.values['cycle'] = wave.values
    wave.declaration = Uniform(a, d)
    top = _top(tmp_path)
    top.add_source(sources.VoltageSource('stim', positive='a', negative='0', waveform=wave))
    assert top.variables() == [a, b, c, d]
