import pytest
from silk import sources
from silk.circuit.circuit_elements import register_module_definition
from silk.circuit.variable import Variable
from silk.exceptions import CircuitError, NotSupportedError, VariationError
from silk.simulation import DC, Transient
from silk.specs.spec import CalibrationBinding
from silk.simulation.variation import Normal, Uniform
from tests._variation_bench import adder, chip


def metrics(report, n=3):
    return [report.metrics[f'y{k}'] for k in range(n)]


def test_nominal_outside_monte_carlo(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.25, sigma=1e-3)})
    assert metrics(tb.run()) == pytest.approx([0.25] * 3)
    _, tb = chip(pdk, tmp_path / 'u', {'offset': Uniform(low=0.1, high=0.3)})
    assert metrics(tb.run()) == pytest.approx([0.2] * 3)


def test_declared_params_resolve_without_overwriting_declarations(pdk, tmp_path, monkeypatch):
    from silk.simulation.behavioral import BehavioralEngine
    captured = []
    original_evaluate = BehavioralEngine._evaluate_blocks

    def observe_storage(engine, *args, **kwargs):
        result = original_evaluate(engine, *args, **kwargs)
        captured.append((engine.resolution, engine._attachments))
        return result

    monkeypatch.setattr(BehavioralEngine, '_evaluate_blocks', observe_storage)
    params = {'offset': Normal(nominal=0.25, sigma=1e-3), 'gain': 2.0}
    top, tb = chip(pdk, tmp_path, params)
    def model(ports, t, params):
        assert type(params.offset) is float
        assert params.gain == 2.0
        ports.y.v = ports.a.v + params.offset * params.gain
    top.elem_dict['c0']['device'].set_behavioral_model(model, params=params, execution='vector')
    assert metrics(tb.run()) == pytest.approx([0.5]*3)
    assert isinstance(params['offset'], Normal)
    assert params['gain'] == 2.0
    assert len(captured) == 1
    resolution, attachments = captured[0]
    paths = [f'xdut.c{k}' for k in range(3)]
    storage = [attachments[path][3] for path in paths]
    assert len({id(values) for values in storage}) == 3
    for path, values in zip(paths, storage):
        assert values is resolution.params[path]
        assert values is not params
        assert values == {'offset': 0.25, 'gain': 2.0}


def test_plain_params_view_reads_the_live_dict(pdk, tmp_path):
    params = {'offset': 0.1}
    top, tb = chip(pdk, tmp_path, params)
    assert metrics(tb.run()) == pytest.approx([0.1]*3)
    params['offset'] = 0.2
    assert metrics(tb.run()) == pytest.approx([0.2]*3)
    assert top.elem_dict['c0']['device']._behavioral_params is params


def test_dc_sweep_resolves_nominal(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.25, sigma=1e-3)},
                 analyses=[DC('input', 0.0, 0.2, 0.1)])
    report = tb.run()
    assert report.metrics['y0'] == pytest.approx(0.45)   # final point a=0.2


def test_process_conflict_raises_through_run_all_tests(pdk, tmp_path):
    params = {'offset': 0.0,
              'shift': Normal(nominal=0.0, sigma=1e-3, variation='process', name='shift')}
    top, tb = chip(pdk, tmp_path, params)
    other = adder(pdk, tmp_path, {'offset': 0.0, 'shift': Normal(
        nominal=0.0, sigma=2e-3, variation='process', name='shift')}, name='adder2')
    top.add_net('y_odd')
    top.add_module(other, 'odd', nets={'a': 'a', 'y': 'y_odd'})
    with pytest.raises(VariationError, match='shift'):
        top.run_all_tests()          # propagates, never folded into a result entry


def test_nested_declaration_refused(pdk, tmp_path):
    with pytest.raises(NotSupportedError, match='offset'):
        chip(pdk, tmp_path, {'cfg': {'offset': Normal(nominal=0.0, sigma=1e-3)}})[1].run()


def test_spice_values_refuse_declarations(pdk):
    from silk.utils._utils import _render_param, _resolve_value
    with pytest.raises(NotSupportedError, match='behavioural-only'):
        _resolve_value(Normal(nominal=1.0, sigma=0.1))
    with pytest.raises(NotSupportedError):
        _render_param(Uniform(low=1.0, high=2.0))
    src = sources.VoltageSource('v', positive='a', negative='gnd',
                                dc=Normal(nominal=1.0, sigma=0.1))
    with pytest.raises(NotSupportedError):
        src.to_spice()


def test_variable_value_cannot_be_a_declaration():
    v = Variable('v', value=1.0, bounds=(0.0, 2.0))
    with pytest.raises(TypeError):
        v.value = Normal(nominal=1.0, sigma=0.1)


def test_one_definition_for_many_placements(pdk, tmp_path):
    registry = {}
    a = adder(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=1e-3)})
    b = adder(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=1e-3)})
    assert register_module_definition(registry, a, validate=True)[0] == register_module_definition(registry, b, validate=True)[0]
    c = adder(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=2e-3)})
    with pytest.raises(CircuitError, match='behavioral model differs'):
        register_module_definition(registry, c, validate=True)


def test_calibration_refuses_a_declaration(pdk, tmp_path):
    block = adder(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=1e-3), 'gain': 1.0})
    with pytest.raises(NotSupportedError, match='field'):
        block.calibrate_behavioral('offset', 1e-3)
    assert isinstance(block._behavioral_params['offset'], Normal)
    block.calibrate_behavioral('gain', 2.0)          # plain entries unchanged
    assert block._behavioral_params['gain'] == 2.0


@pytest.mark.parametrize('entry', ['simulate', 'simulate_behavioral', 'run_all_tests'])
def test_every_run_entry_resolves_declarations(pdk, tmp_path, entry):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0.25, 0.001)})
    if entry == 'run_all_tests':
        reports = top.run_all_tests()
        report = next(iter(reports.values()))
        assert metrics(report) == pytest.approx([0.25] * 3)
    else:
        if entry == 'simulate':
            tb.netlist.simulator = 'behavioral'
            tb.netlist.simulate()
        else:
            tb.netlist.simulate_behavioral(1e-9, 10e-9)
        assert tb.netlist.load_signal('y0').y[-1] == pytest.approx(0.25)


def test_resolution_holds_across_dc_points_and_analyses_but_refreshes_next_run(pdk, tmp_path, monkeypatch):
    # Module globals deliberately observe writes from run-local model copies.
    global seen, sigma
    from silk.simulation.behavioral_backend import BehavioralBackend
    from silk.simulation.behavioral import BehavioralEngine
    storage = []
    original_evaluate = BehavioralEngine._evaluate_blocks

    def observe_storage(engine, *args, **kwargs):
        result = original_evaluate(engine, *args, **kwargs)
        values = engine._attachments['xdut.c0'][3]
        assert values is engine.resolution.params['xdut.c0']
        storage.append(values)
        return result

    monkeypatch.setattr(BehavioralEngine, '_evaluate_blocks', observe_storage)
    sigma = Variable('nominal', value=0.25, bounds=(0.0, 1.0))
    top, tb = chip(pdk, tmp_path, {'offset': Normal(sigma, 0.001)}, n=1,
                   analyses=[DC('input', 0, 0.2, 0.1), Transient(1e-9, 2e-9)])
    block = top.elem_dict['c0']['device']
    seen = []
    original = block._behavioral_model

    def change_nominal(ports, t, params):
        seen.append(params)
        sigma.value = 0.75
        return original(ports, t, params)

    block.set_behavioral_model(change_nominal, params=block._behavioral_params, execution='vector')
    backend = BehavioralBackend()
    backend.run_module(tb.netlist)
    assert len(seen) == 4
    assert all(p['offset'] == 0.25 for p in seen)
    assert len(storage) == 4
    assert all(values is storage[0] for values in storage)
    first = storage[0]
    seen.clear()
    storage.clear()
    backend.run_module(tb.netlist)
    assert len(seen) == len(storage) == 4
    assert all(p['offset'] == 0.75 for p in seen)
    assert all(values is storage[0] for values in storage)
    assert storage[0] is not first
    assert block._behavioral_params['offset'].nominal is sigma


def test_raw_segments_scope_and_inherited_mismatch(pdk, tmp_path):
    from silk.simulation.behavioral import BehavioralEngine
    from silk.simulation._behavioral_variation import resolve_run, VariationScope
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0.0, 0.001)},
                   names=['xc0', 'c1'])
    top.set_mismatch(False)
    helper = adder(pdk, tmp_path, {'offset': Normal(0.0, 0.001)}, name='helper')
    tb.netlist.add_module(helper, 'xstim', nets={'a': 'a', 'y': 'helper_y'})
    blocks, _ = BehavioralEngine()._flatten(tb.netlist)
    resolution = resolve_run(blocks, scope=VariationScope('$/a~1b/c~0d', 'offsets', 'dut'))
    records = resolution.records
    assert records[0].block == 'xdut.xc0'
    assert records[0].display == 'dut.xc0'
    assert records[0].key == ('mismatch', ('a/b', 'c~d', 'xc0'))
    assert records[0].mismatch_enabled is False
    assert records[1].mismatch_enabled is False
    assert records[2].key == ('bench', '$/a~1b/c~0d', 'offsets', ('xstim',))
    assert records[2].mismatch_enabled is True
    direct, _ = BehavioralEngine()._flatten(top)
    assert resolve_run(direct).records[0].key == ('mismatch', ('xc0',))
    tb.run()
    assert tb.netlist._variation_scope == VariationScope('$', 'offsets', 'dut')


@pytest.mark.parametrize('other', [Uniform(-0.001, 0.001, variation='process', name='shift'),
                                  Normal(0.0, 0.002, variation='process', name='shift')])
def test_process_conflict_names_both_blocks_before_model_calls(pdk, tmp_path, other):
    # Observe even an accidental call outside copied closure state.
    global called
    params = {'offset': Normal(0, 0.001, variation='process', name='shift')}
    top, tb = chip(pdk, tmp_path, params, n=1)
    second = adder(pdk, tmp_path, {'offset': other}, name='other')
    top.add_module(second, 'second', nets={'a': 'a', 'y': 'second_y'})
    called = []
    first = top.elem_dict['c0']['device']
    def model_3(ports, t, p):
        return called.append(p)
    first._behavioral_model = model_3
    with pytest.raises(VariationError, match='shift.*xdut.c0.*xdut.second'):
        tb.run()
    assert called == []


def test_identical_process_declarations_and_hidden_leaf_contents(pdk, tmp_path):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0.1, 0.001, variation='process', name='shift')})
    block = top.elem_dict['c0']['device']
    hidden = adder(pdk, tmp_path, {'offset': Normal(0, 0.002, variation='process', name='shift')}, name='hidden')
    block.add_module(hidden, 'hidden', nets={'a': 'a', 'y': 'hidden_y'})
    assert metrics(tb.run()) == pytest.approx([0.1] * 3)


@pytest.mark.parametrize('container', [lambda d: [d], lambda d: (d,), lambda d: {d},
                                      lambda d: {'nested': d}])
def test_nested_declaration_containers_refuse_at_attachment(pdk, tmp_path, container):
    with pytest.raises(NotSupportedError, match='cfg'):
        adder(pdk, tmp_path, {'cfg': container(Normal(0, 1))})


def test_params_mutated_after_attachment_are_rechecked(pdk, tmp_path):
    import numpy as np
    top, tb = chip(pdk, tmp_path, {'offset': 0.1})
    top.elem_dict['c0']['device']._behavioral_params['cfg'] = np.array([Normal(0, 1)], dtype=object)
    with pytest.raises(NotSupportedError, match='xdut.c0.*cfg'):
        tb.run()


def test_spice_waveform_device_and_fmt_refuse(pdk, tmp_path):
    from silk.utils._utils import _fmt
    from silk.sources.waveforms import Sine
    from silk.elements import elements
    decl = Normal(1, 0.1)
    with pytest.raises(NotSupportedError, match='behavioural-only'):
        _fmt(decl)
    with pytest.raises(NotSupportedError, match='behavioural-only'):
        Sine(voffset=0, vampl=decl, freq=1e3).to_spice()
    resistor = elements.resistor('r', p='a', n='b', value=decl)
    with pytest.raises(NotSupportedError, match='behavioural-only'):
        resistor.to_spice()


def test_identity_encodes_variables_expressions_and_explicit_copies(pdk, tmp_path):
    import copy
    from silk.circuit._behavioral_identity import behavioral_key
    s = Variable('sigma', value=0.001, bounds=(0.0, 1.0))
    a = adder(pdk, tmp_path, {'offset': Normal(0, s * 2 + 0.001)})
    b = copy.deepcopy(a)
    assert behavioral_key(a) == behavioral_key(b)
    before = behavioral_key(a)
    s.value = 0.002
    assert behavioral_key(a) != before
    assert behavioral_key(a) == behavioral_key(b)  # Variables are shared knobs
    registry = {}
    register_module_definition(registry, a, validate=True)
    register_module_definition(registry, copy.deepcopy(a), validate=True)
    assert len(registry) == 1
    top, tb = chip(pdk, tmp_path / 'chip', {'offset': Normal(0, s)}, n=5)
    from silk.elements import elements
    top.elem_dict['c0']['device'].add(elements.resistor('r', p='a', n='y', value=1000))
    tb.netlist.simulator = 'ngspice'
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    from pathlib import Path
    text = Path(tb_netlist_deck_path).read_text()
    assert text.count('.subckt adder ') == 1


def test_nested_declaration_in_structured_object_array_is_refused(pdk, tmp_path):
    import numpy as np
    cfg = np.array([(Normal(0, 1),)], dtype=[('offset', object)])
    with pytest.raises(NotSupportedError, match='cfg.*offset'):
        adder(pdk, tmp_path, {'cfg': cfg})


def test_categorical_variable_also_refuses_a_declaration():
    variable = Variable('choice', value='small', var_type='categorical', categories=['small'])
    with pytest.raises(TypeError, match='statistical declaration'):
        variable.value = Uniform(0, 1)
    assert variable.value == 'small'


def test_variant_definitions_deduplicate_equal_statistics(pdk, tmp_path):
    registry = {}
    names = []
    for sigma in (0.001, 0.002, 0.001):
        variable = Variable('sigma', value=sigma, bounds=(0.0, 1.0))
        block = adder(pdk, tmp_path, {'offset': Normal(0, variable * 2)})
        block.variants = True
        names.append(register_module_definition(registry, block, validate=True)[0])
    assert names[0] == names[2]
    assert names[0] != names[1]
    assert len(registry) == 2
