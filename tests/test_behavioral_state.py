"""Shared electrical values and behavioral ports."""
import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.exceptions import SimulationError
from silk.simulation import DC, Transient
from silk.simulation.behavioral import BehavioralEngine
from silk.sources import VoltageSource, CurrentSource
from tests.test_behavioral_rawresult import _block, _pdk


def state(size=4, pins=None):
    from silk.simulation._behavioral_state import SharedState
    return SharedState(size, pins or {('a', 'out'): 'out', ('b', 'in'): 'out'})


def chain(tmp_path, producer, consumer):
    top = Module('sky130', str(tmp_path / 'top'), 'top.cir', pdk_details=_pdk())
    top.simulator = 'behavioral'
    top.add_source(VoltageSource('drive', 'vin', 'gnd', dc=1))
    top.add_module(_block(tmp_path, producer, name='producer'), 'producer',
                   nets={'vin': 'vin', 'vout': 'mid'})
    top.add_module(_block(tmp_path, consumer, name='consumer'), 'consumer',
                   nets={'vin': 'mid', 'vout': 'out'})
    return top


def test_ports_integer_voltage_reaches_bitwise_reader(tmp_path):
    def model_1(ports, t, params):
        ports.vout.v = np.arange(len(t), dtype=np.int16)
    def model_2(ports, t, params):
        ports.vout.v = np.asarray(ports.vin.v, dtype=np.int16) & 1
    top = chain(tmp_path, model_1,
                model_2)
    values, _ = BehavioralEngine().run(top, 1, 3)
    assert values['mid'].dtype == float
    np.testing.assert_array_equal(values['out'], [0, 1, 0, 1])


def test_ports_scalar_assignment_broadcasts_for_reader(tmp_path):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    seen = []
    def read(ports, t, params):
        seen.append(np.isscalar(ports.vin.v))
        ports.vout.v = ports.vin.v + 1
    def model_3(ports, t, params):
        ports.vout.v = 3
    top = chain(tmp_path, model_3, read)
    values, _ = BehavioralEngine().run(top, 1, 3)
    assert seen == [False]
    np.testing.assert_array_equal(values['mid'], [3]*4)
    np.testing.assert_array_equal(values['out'], [4]*4)


@pytest.mark.parametrize('analysis', ['transient', 'dc'])
def test_ports_bad_shape_fails_before_downstream_call(tmp_path, analysis):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    seen = []
    def read(ports, t, params):
        seen.append(np.shape(ports.vin.v))
        ports.vout.v = 7
    def model_4(ports, t, params):
        ports.vout.v = np.array([1, 2, 3])
    top = chain(tmp_path, model_4, read)
    top.add_sim_type(Transient(step=1, stop=1) if analysis == 'transient'
                     else DC(source='drive', start=0, stop=1, step=1))
    with pytest.raises(ValueError, match='broadcast'):
        top.simulate()
    assert seen == []


def test_unset_voltage_and_pin_currents_are_separate():
    s = state()
    np.testing.assert_array_equal(s.read_voltage('a', 'out'), [0, 0, 0, 0])
    s.write_current('a', 'out', [1, 2], index=slice(0, 2))
    np.testing.assert_array_equal(s.read_voltage('b', 'in'), [0, 0, 0, 0])
    np.testing.assert_array_equal(s.read_current('a', 'out', index=slice(0, 2)), [1, 2])
    s.write_voltage('a', 'out', 3, index=1)
    np.testing.assert_array_equal(s.read_voltage('b', 'in'), [0, 3, 0, 0])


@pytest.mark.parametrize('quantity', ['voltage', 'current'])
def test_each_sample_written_once_and_assignment_copies(quantity):
    from silk.exceptions import RewriteError
    s = state()
    write = getattr(s, 'write_' + quantity)
    read = getattr(s, 'read_' + quantity)
    local = np.array([1., 2.])
    write('a', 'out', local, index=slice(0, 2))
    local[:] = 99
    np.testing.assert_array_equal(read('a', 'out', index=slice(0, 2)), [1, 2])
    alias = read('a', 'out', index=slice(0, 2))
    alias[:] = 44
    np.testing.assert_array_equal(read('a', 'out', index=slice(0, 2)), [1, 2])
    with pytest.raises(RewriteError, match='local.*assign.*once'):
        write('a', 'out', [2, 3], index=slice(1, 3))
    # Failed overlapping writes cannot publish their previously unset portion.
    write('a', 'out', [3, 4], index=slice(2, 4))
    np.testing.assert_array_equal(read('a', 'out'), [1, 2, 3, 4])


def test_voltage_conflict_names_both_models_and_net():
    from silk.exceptions import ConflictError
    s = state()
    s.write_voltage('a', 'out', 1)
    s.write_current('b', 'in', 2)
    with pytest.raises(ConflictError, match="out.*a.*b"):
        s.write_voltage('b', 'in', 1)
    assert issubclass(ConflictError, SimulationError)


def test_kcl_fills_only_one_unknown_per_net_and_sample():
    from silk.exceptions import KCLError
    s = state(pins={('tap', 'out'): 'out', ('load', 'out'): 'out',
                    ('watch', 'in'): 'out'})
    s.write_current('tap', 'out', [1, 2, 3, 4])
    np.testing.assert_array_equal(s.read_current('load', 'out', index=slice(0, 2)), [-1, -2])
    np.testing.assert_array_equal(s.read_current('load', 'out', index=slice(0, 2)), [-1, -2])
    # A different sample can have a different missing current.
    s.write_current('load', 'out', 0, index=slice(2, 4))
    np.testing.assert_array_equal(s.read_current('watch', 'in', index=slice(2, 4)), [-3, -4])
    with pytest.raises(KCLError, match='out.*load.*watch'):
        s.read_current('watch', 'in')
    assert issubclass(KCLError, SimulationError)


def test_kcl_zero_unknown_and_fill_is_a_write():
    from silk.exceptions import RewriteError
    s = state()
    np.testing.assert_array_equal(s.read_current('b', 'in'), [0, 0, 0, 0])
    with pytest.raises(RewriteError):
        s.write_current('b', 'in', 0)


def test_cml_currents_load_voltage_and_voltage_only_watcher():
    pins = {(path, port): port for path in ('pre', 'main', 'post1', 'post2', 'load', 'watch')
            for port in ('OUTP', 'OUTN')}
    s = state(1, pins)
    for path, positive, negative in [('pre', .003, 0), ('main', 0, .008),
                                     ('post1', .0048, 0), ('post2', 0, .0008)]:
        s.write_current(path, 'OUTP', positive)
        s.write_current(path, 'OUTN', negative)
    ip = s.read_current('load', 'OUTP')
    inn = s.read_current('load', 'OUTN')
    np.testing.assert_allclose(ip, [-.0078])
    np.testing.assert_allclose(inn, [-.0088])
    s.write_voltage('load', 'OUTP', 1.8 + 50 * ip)
    s.write_voltage('load', 'OUTN', 1.8 + 50 * inn)
    np.testing.assert_allclose(s.read_voltage('watch', 'OUTP'), [1.41])
    np.testing.assert_allclose(s.read_voltage('watch', 'OUTN'), [1.36])


@pytest.mark.parametrize('operation', ['slice', 'inplace', 'ufunc'])
def test_ports_input_write_conflict_names_both_models(tmp_path, operation):
    def mutate(ports, t, params):
        if operation == 'slice':
            ports.vin.v[:] = 0
        elif operation == 'inplace':
            ports.vin.v -= 1
        else:
            np.add(ports.vin.v, 1, out=ports.vin.v)
    def produce(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, produce, mutate)
    from silk.exceptions import ConflictError
    with pytest.raises(ConflictError, match='mid.*producer.*consumer'):
        BehavioralEngine().run(top, 1, 3)


def test_local_input_copy_and_calculation_are_writable(tmp_path):
    def calculate(ports, t, params):
        local = np.array(ports.vin.v, copy=True)
        local -= 1
        local[:] += 3
        other = ports.vin.v - 1
        other[:] += 3
        ports.vout.v = local + other
    def model_6(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_6, calculate)
    values, _ = BehavioralEngine().run(top, 1, 3)
    np.testing.assert_array_equal(values['out'], [6, 6, 6, 6])


def test_ports_multiple_writers_raise_conflict(tmp_path):
    from silk.exceptions import ConflictError
    def model_7(ports, t, params):
        ports.vout.v = np.ones(len(t))
    def model_8(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_7,
                model_8)
    def model_9(ports, t, params):
        ports.vout.v = np.ones(len(t))
    extra = _block(tmp_path, model_9, name='extra')
    top.add_module(extra, 'extra', nets={'vin': 'vin', 'vout': 'mid'})
    with pytest.raises(ConflictError, match='mid.*producer.*extra'):
        BehavioralEngine().run(top, 1, 3)


def test_independent_current_source_terminals_and_dc_reset(tmp_path):
    def model_10(ports, t, params):
        ports.vout.v = ports.vin.v
    def model_11(ports, t, params):
        ports.vout.v = ports.vin.v
    top = chain(tmp_path, model_10,
                model_11)
    top.add_source(CurrentSource('bias', 'mid', 'out', dc=.002))
    engine = BehavioralEngine()
    engine.run(top, 1, 3)
    np.testing.assert_allclose(engine.state.read_current('producer', 'vout'), [-.002]*4)
    np.testing.assert_allclose(engine.state.read_current('consumer', 'vout'), [.002]*4)
    np.testing.assert_allclose(engine.source_currents['Ibias'], [.002]*4)
    values, _, _ = engine.run_dc(top, DC(source='drive', start=0, stop=1, step=.5))
    np.testing.assert_allclose(values['out'], [0, .5, 1])
    np.testing.assert_allclose(engine.state.read_current('producer', 'vout'), [-.002])


def test_ports_bus_input_write_preserves_source_ownership(tmp_path):
    from silk.circuit.circuit_elements import Port
    from silk.exceptions import ConflictError
    def mutate(ports, t, params):
        ports.code.v[0, :] = 9
    mod = _block(tmp_path, mutate,
                 ports={'code': Port.digital_input(bits=range(3, 5)), 'vout': PortDir.OUT})
    top = Module('sky130', str(tmp_path), 'bus.cir', pdk_details=_pdk())
    top.add_source(VoltageSource('drive', 'bit3', 'gnd', dc=1))
    top.add_module(mod, 'consumer', nets={'code[3]': 'bit3', 'code[4]': 'bit4', 'vout': 'out'})
    with pytest.raises(ConflictError, match='bit3.*consumer'):
        BehavioralEngine().run(top, 1, 3)


def test_ports_assigned_array_is_copied_before_next_model(tmp_path):
    local = np.arange(4.)
    def read(ports, t, params):
        local[:] = 99
        ports.vout.v = ports.vin.v + 1
    def model_13(ports, t, params):
        ports.vout.v = local
    top = chain(tmp_path, model_13, read)
    values, _ = BehavioralEngine().run(top, 1, 3)
    np.testing.assert_array_equal(values['mid'], [0, 1, 2, 3])
    np.testing.assert_array_equal(values['out'], [1, 2, 3, 4])


def test_multiple_ports_on_same_model_cannot_rewrite_net():
    from silk.exceptions import RewriteError
    s = state(pins={('model', 'a'): 'shared', ('model', 'b'): 'shared'})
    s.write_voltage('model', 'a', 1)
    with pytest.raises(RewriteError):
        s.write_voltage('model', 'b', 1)


def test_voltage_writers_checked_per_sample():
    from silk.exceptions import ConflictError
    s = state()
    s.write_voltage('a', 'out', 1, index=slice(0, 2))
    s.write_voltage('b', 'in', 2, index=slice(2, 4))
    np.testing.assert_array_equal(s.read_voltage('a', 'out'), [1, 1, 2, 2])
    with pytest.raises(ConflictError):
        s.write_voltage('a', 'out', 3, index=3)


def test_array_aliases_never_expose_unset_storage():
    s = state()
    alias = s.read_voltage('a', 'out')
    alias[:] = 9
    s.write_voltage('a', 'out', 1)
    np.testing.assert_array_equal(s.read_voltage('a', 'out'), [1, 1, 1, 1])


def test_failed_numeric_assignment_does_not_claim_samples():
    s = state()
    with pytest.raises(ValueError):
        s.write_voltage('a', 'out', [1, 2])
    s.write_voltage('b', 'in', 4)
    np.testing.assert_array_equal(s.read_voltage('a', 'out'), [4, 4, 4, 4])


def test_failed_kcl_read_does_not_fill_other_samples():
    from silk.exceptions import KCLError
    s = state(pins={('load', 'out'): 'out', ('watch', 'in'): 'out'})
    s.read_current('load', 'out', index=1)
    with pytest.raises(KCLError):
        s.read_current('watch', 'in')
    # Its failed whole-array read must not consume the fill at sample zero.
    s.read_current('load', 'out', index=0)


def test_current_sinks_on_distinct_nets_do_not_conflict():
    s = state(pins={('a', 'out'): 'out', ('b', 'out'): 'other',
                    ('load', 'a'): 'out', ('load', 'b'): 'other'})
    s.write_current('a', 'out', 1)
    s.write_current('b', 'out', 2)
    np.testing.assert_array_equal(s.read_current('load', 'a'), [-1]*4)
    np.testing.assert_array_equal(s.read_current('load', 'b'), [-2]*4)


def test_series_wire_does_not_transfer_current_bookkeeping(tmp_path):
    from silk.elements import elements
    top = Module('sky130', str(tmp_path), 'wire.cir', pdk_details=_pdk())
    top.add_source(VoltageSource('drive', 'near', 'gnd', dc=.4))
    top.add_source(CurrentSource('bias', 'near', 'gnd', dc=.002))
    top.add(elements.resistor('wire', p='near', n='far', value=1))
    def model_14(ports, t, params):
        ports.vout.v = ports.vin.v
    reader = _block(tmp_path, model_14)
    top.add_module(reader, 'reader', nets={'vin': 'far', 'vout': 'out'})
    engine = BehavioralEngine()
    with pytest.warns(UserWarning, match='wire'):
        engine.run(top, 1, 3)
    np.testing.assert_array_equal(engine.state.read_current('reader', 'vin'), [0]*4)


def test_current_source_signs_use_hierarchical_resolved_terminals(tmp_path):
    container = Module('sky130', str(tmp_path / 'container'), 'container.cir',
                       pdk_details=_pdk(), module_name='container')
    container.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_15(ports, t, params):
        ports.vout.v = ports.vin.v
    leaf = _block(tmp_path, model_15)
    container.add_module(leaf, 'leaf', nets={'vin': 'vin', 'vout': 'vout'})
    container.add_source(CurrentSource('bias', 'vin', 'vout', dc=.003))
    top = Module('sky130', str(tmp_path / 'top'), 'top.cir', pdk_details=_pdk())
    top.add_source(VoltageSource('drive', 'in', 'gnd', dc=.5))
    top.add_module(container, 'container', nets={'vin': 'in', 'vout': 'out'})
    engine = BehavioralEngine()
    engine.run(top, 1, 3)
    np.testing.assert_allclose(engine.state.read_current('xcontainer.leaf', 'vin'), [-.003]*4)
    np.testing.assert_allclose(engine.state.read_current('xcontainer.leaf', 'vout'), [.003]*4)


@pytest.mark.parametrize('operation', ['slice', 'fill'])
def test_ports_numpy_alias_is_detached_from_storage(tmp_path, operation):
    def mutate(ports, t, params):
        alias = np.asarray(ports.vin.v)
        alias.setflags(write=True)
        if operation == 'fill':
            alias.fill(99)
        else:
            alias[:] = 99
        ports.vout.v = ports.vin.v
    def produce(ports, t, params):
        ports.vout.v = np.ones(len(t))
    values, _ = BehavioralEngine().run(chain(tmp_path, produce, mutate), 1, 3)
    np.testing.assert_array_equal(values['out'], [1]*4)


@pytest.mark.parametrize('name', ['ConflictError', 'RewriteError', 'KCLError', 'LoopError'])
def test_electrical_errors_are_exported_simulation_errors(name):
    import silk.exceptions as errors
    assert name in errors.__all__
    error = getattr(errors, name)('diagnostic', log_path='run.log')
    assert isinstance(error, SimulationError)
    assert error.log_path == 'run.log'


def test_kcl_read_keeps_the_value_present_at_that_moment():
    s = state(pins={('first', 'out'): 'out', ('later', 'out'): 'out',
                    ('load', 'out'): 'out'})
    s.write_current('first', 'out', 1)
    np.testing.assert_array_equal(s.read_current('load', 'out'), [-1]*4)
    s.write_current('later', 'out', 2)
    np.testing.assert_array_equal(s.read_current('load', 'out'), [-1]*4)


def test_only_actual_ports_assignments_claim_voltage(tmp_path):
    def model_17(ports, t, params):
        ports.vout.v = np.ones(len(t))
    def model_18(ports, t, params):
        ports.vout.v = ports.vin.v
    top = chain(tmp_path, model_17,
                model_18)
    def model_19(ports, t, params):
        return
    missing = _block(tmp_path, model_19, name='missing')
    top.add_module(missing, 'missing', nets={'vin': 'vin', 'vout': 'mid'})
    values, _ = BehavioralEngine().run(top, 1, 3)
    np.testing.assert_array_equal(values['out'], [1]*4)


@pytest.mark.parametrize('analysis', ['transient', 'dc'])
def test_ports_non_numeric_assignment_fails_before_reader(tmp_path, analysis):
    # Module globals deliberately observe writes from run-local model copies.
    global seen
    malformed = object()
    seen = []
    def read(ports, t, params):
        seen.append(ports.vin.v is malformed)
        ports.vout.v = 7
    def model_20(ports, t, params):
        ports.vout.v = malformed
    top = chain(tmp_path, model_20, read)
    top.add_sim_type(Transient(step=1, stop=1) if analysis == 'transient'
                     else DC(source='drive', start=0, stop=1, step=1))
    with pytest.raises(TypeError):
        top.simulate()
    assert seen == []


def test_ports_lists_are_copied_for_publication_and_each_consumer(tmp_path):
    # Keep the producer mutation externally observable while testing publication copies.
    global original
    original = [1, 2, 3, 4]
    def produce(ports, t, params):
        ports.vout.v = original
    def mutate(ports, t, params):
        original[1] = 88
        local = np.asarray(ports.vin.v)
        local[0] = 99
        ports.vout.v = local
    def read(ports, t, params):
        ports.vout.v = ports.vin.v
    top = chain(tmp_path, produce, mutate)
    top.add_module(_block(tmp_path, read, name='observer'), 'observer',
                   nets={'vin': 'mid', 'vout': 'observed'})
    values, _ = BehavioralEngine().run(top, 1, 3)
    np.testing.assert_array_equal(values['mid'], [1, 2, 3, 4])
    np.testing.assert_array_equal(values['out'], [99, 2, 3, 4])
    np.testing.assert_array_equal(values['observed'], [1, 2, 3, 4])
    assert original == [1, 88, 3, 4]


def test_object_input_alias_is_detached_after_voltage_conversion(tmp_path):
    def produce(ports, t, params):
        ports.vout.v = np.array([1, 2, 3, 4], dtype=object)
    def mutate(ports, t, params):
        alias = np.asarray(ports.vin.v)
        alias.setflags(write=True)
        alias[:] = 99
        ports.vout.v = ports.vin.v
    values, _ = BehavioralEngine().run(chain(tmp_path, produce, mutate), 1, 3)
    np.testing.assert_array_equal(values['out'], [1, 2, 3, 4])


def test_unrelated_model_value_error_is_preserved(tmp_path):
    def fail(ports, t, params):
        raise ValueError('calibration file is read-only')
    def model_24(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_24, fail)
    with pytest.raises(ValueError, match='^calibration file is read-only$'):
        BehavioralEngine().run(top, 1, 3)


def test_local_read_only_array_error_is_preserved(tmp_path):
    def fail(ports, t, params):
        local = np.arange(len(t))
        local.setflags(write=False)
        local[0] = 99
    def model_25(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_25, fail)
    with pytest.raises(ValueError, match='^assignment destination is read-only$'):
        BehavioralEngine().run(top, 1, 3)


def test_right_hand_side_local_error_precedes_input_assignment(tmp_path):
    def local_values(t):
        local = np.arange(len(t))
        local.setflags(write=False)
        local[0] = 99
        return local
    def fail(ports, t, params):
        ports.vin.v[:] = local_values(t)
    def model_26(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_26, fail)
    with pytest.raises(ValueError, match='^assignment destination is read-only$'):
        BehavioralEngine().run(top, 1, 3)


def test_input_write_from_callable_without_source_has_silk_error(tmp_path):
    namespace = {}
    exec(compile("def mutate(ports, t, params):\n    ports.vin.v[:] = 0\n",
                 '<behavioral-model-without-source>', 'exec'), namespace)
    def model_27(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_27, namespace['mutate'])
    with pytest.raises(SimulationError, match='mid.*producer.*consumer'):
        BehavioralEngine().run(top, 1, 3)


def test_input_numpy_view_mutation_leaves_source_storage_unchanged(tmp_path):
    def produce(ports, t, params):
        ports.vout.v = np.ones(len(t))
    def mutate(ports, t, params):
        view = np.asarray(ports.vin.v).view()[1:]
        view[:] = 0
        ports.vout.v = ports.vin.v
    values, _ = BehavioralEngine().run(chain(tmp_path, produce, mutate), 1, 3)
    np.testing.assert_array_equal(values['out'], [1]*4)


def test_ufunc_local_read_only_output_keeps_native_error(tmp_path):
    def fail(ports, t, params):
        local = np.array(ports.vin.v, copy=True)
        local.setflags(write=False)
        np.add(ports.vin.v, 1, out=local)
    def model_29(ports, t, params):
        ports.vout.v = np.ones(len(t))
    top = chain(tmp_path, model_29, fail)
    with pytest.raises(ValueError, match='^output array is read-only$'):
        BehavioralEngine().run(top, 1, 3)
