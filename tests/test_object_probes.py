"""Placement references, declared indices and aggregate probe acquisition."""
import copy

import numpy as np
import pytest

from tests._legacy_names import add_legacy_net, legacy_net

from silk import Module
from silk.circuit.net import Net
from silk.circuit.module_test import _unbound_leaves
from silk.elements import elements
from silk.exceptions import CircuitError, UnboundSignalError
from silk.signals.base import Signal
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile


def block(tmp_path, name='leaf'):
    m = Module(None, str(tmp_path), name + '.cir', None, module_name=name)
    m.define_module_ports({'out': 'output'})
    m.add_net('mid')
    return m


def bench(tmp_path):
    dut = block(tmp_path, 'dut')
    tb = dut.create_test('read')
    h = tb.add_module(block(tmp_path), 'stage', {'out': 'parent'})
    return tb, h


def raw(tb, tmp_path, names):
    vectors = [RawVector('time', 'time', data=np.array([0., 1.]))]
    for number, name in enumerate(names, 1):
        vectors.append(RawVector('v(' + name + ')', 'voltage', data=np.array([number, number + .5])))
    path = tmp_path / 'literal.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis', 'title': 'refs'},
                                        vectors, npoints=2, real=True, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()


D = (8, 9, 10, 11)
S = (-3, -1, 2, 5)
R = (11, 10, 9, 8)
ROWS = [
    (D, 8, (8,)), (S, -1, (-1,)),
    (D, slice(8, 11), (8, 9, 10)), (D, slice(None, 10), (8, 9)),
    (D, slice(10, None), (10, 11)), (D, slice(None), D),
    (D, slice(None, None, -1), R), (D, slice(11, 8, -1), (11, 10, 9)),
    (D, slice(None, None, 2), (8, 10)), (D, slice(11, 7, -2), (11, 9)),
    (S, slice(-3, 5), (-3, -1, 2)), (S, slice(-3, 6, 2), (-3, -1, 5)),
    (S, slice(None, None, -1), (5, 2, -1, -3)),
    (R, slice(None), R), (R, slice(8, 11), (10, 9, 8)),
    (R, slice(None, None, -1), D), ((8,), 8, (8,)), ((8,), slice(None), (8,)),
]


def aggregate(kind, indices):
    if kind == 'net':
        return Net._make_bus_handle('d', tuple(legacy_net(f'd[{k}]') for k in indices),
                                    tuple(('d', k) for k in indices))
    return Signal._make_bus_handle('d', tuple(Signal(str(k), 'voltage', 'time', 'V', [0], [k])
                                             for k in indices), indices=indices)


@pytest.mark.parametrize('kind', ['net', 'signal'])
@pytest.mark.parametrize('indices,key,expected', ROWS)
def test_declared_number_truth_table(kind, indices, key, expected):
    bus = aggregate(kind, indices)
    selected = bus[key]
    got = (selected,) if isinstance(key, int) else tuple(selected)
    assert tuple(int(x.name.split('[')[-1].rstrip(']')) for x in got) == expected


@pytest.mark.parametrize('kind', ['net', 'signal'])
@pytest.mark.parametrize('indices,key', [(D, 0), (D, -1), (S, -2), ((8,), 0),
    (D, slice(8, 11, -1)), (D, slice(20, 30)), (D, slice(8, 8))])
def test_missing_declared_elements_refuse(kind, indices, key):
    with pytest.raises(CircuitError, match='declared'):
        aggregate(kind, indices)[key]


@pytest.mark.parametrize('kind', ['net', 'signal'])
def test_zero_step(kind):
    with pytest.raises(ValueError, match='zero'):
        aggregate(kind, D)[::0]


def test_array_keys_are_actual_declared_numbers(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_module(block(tmp_path, 'a'), 'X[8]', {'out': 'mid'})
    stage.subcircuit.add_module(block(tmp_path, 'b'), 'X[2]', {'out': 'mid'})
    for k in (3, 9):
        stage.subcircuit.add(elements.resistor(f'M[{k}]', p='mid', n='out', value=1))
    assert stage.X[8].path('mid') == 'xstage.xX[8].mid'
    assert stage.X[2].path('out') == 'xstage.mid'
    assert stage.M[3].name == 'M[3]'
    with pytest.raises(CircuitError, match='declared'):
        stage.X[0]
    stage.subcircuit.add(elements.resistor('M[03]', p='mid', n='out', value=1))
    with pytest.raises(CircuitError, match=r'M\[03\].*M\[3\]|M\[3\].*M\[03\]'):
        stage.device('M')


@pytest.mark.parametrize('name', ['M1:custom', 'M1:gm', 'i(stim)', 'v(out)', 'data[8]'])
def test_literal_voltage_reference_readback(tmp_path, name):
    tb, stage = bench(tmp_path)
    add_legacy_net(stage.subcircuit, name)
    p = tb.probe(stage.net(name))
    raw(tb, tmp_path, ['xstage.' + name])
    np.testing.assert_array_equal(p.y, [1., 1.5])
    with pytest.raises(CircuitError, match='current'):
        tb.probe(stage.net(name), type='I')


def test_deferred_aggregate_walk_and_decode(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_net('d', bits=range(8, 12))
    p = tb.probe(stage.d[8:10])
    assert [leaf.name for leaf in _unbound_leaves(p)] == ['xstage.d[8]', 'xstage.d[9]']
    code = p.as_int()
    assert len(list(_unbound_leaves(code))) == 2
    assert not p.bound
    raw(tb, tmp_path, ['xstage.d[8]', 'xstage.d[9]'])
    np.testing.assert_array_equal(code.y, [3., 3.])
    assert copy.deepcopy(p)[8].name == p[8].name
    one = tb.probe(stage.d[8:9])
    np.testing.assert_array_equal(one.as_int().y, [1., 1.])


def test_connected_port_and_placement_identity(tmp_path):
    tb, stage = bench(tmp_path)
    port, internal = stage.out, stage.mid
    a = tb.probe(port)
    assert a.name == tb.probe('parent').name
    with pytest.raises(UnboundSignalError):
        _ = internal.y
    entry = tb.netlist.elem_dict['stage']
    tb.netlist.add_net('rewired')
    entry['pin_list']['out'].net = 'rewired'
    assert tb.probe(port).name == 'rewired'
    tb.netlist._element_mgr.elem_dict['stage'] = dict(entry)
    with pytest.raises(CircuitError, match='replaced'):
        tb.probe(internal)
    # Already bound probes still follow the bench rather than placement identity.
    raw(tb, tmp_path, ['parent'])
    np.testing.assert_array_equal(a.y, [1., 1.5])


def test_reference_arithmetic_and_ownership(tmp_path):
    tb, stage = bench(tmp_path)
    p = tb.probe(stage.out)
    for result in (stage.out - stage.mid, p + stage.mid, stage.mid + p, 2 - stage.out):
        assert isinstance(result, Signal)
        assert not result.bound
    other, _ = bench(tmp_path / 'other')
    with pytest.raises(CircuitError, match='placement'):
        other.probe(stage.mid)
    with pytest.raises(CircuitError, match='placement'):
        tb.probe(stage.subcircuit.net_dict['mid'])


def test_shared_contexts_refuse_bare_reads_before_any_run(tmp_path):
    tb, stage = bench(tmp_path)
    second = tb.module.create_test('other')
    second.netlist = tb.netlist
    for owner in (tb, second):
        assert owner.probe(stage.mid).name == 'xstage.mid'
    with pytest.raises(UnboundSignalError, match='tb.probe'):
        _ = stage.mid.y
    raw(tb, tmp_path, ['xstage.mid'])
    with pytest.raises(UnboundSignalError, match='tb.probe'):
        _ = stage.mid.y
    del tb.module._tests['other']
    np.testing.assert_array_equal(stage.mid.y, [1., 1.5])


def test_nested_reference_copy_and_replacement(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_module(block(tmp_path, 'inside'), 'inner', {'out': 'mid'})
    ref = stage.inner.out
    assert tb.probe(ref).name == 'xstage.mid'
    cloned, cloned_ref = copy.deepcopy((tb, ref))
    assert cloned.probe(cloned_ref).name == 'xstage.mid'
    with pytest.raises(CircuitError):
        cloned.probe(ref)
    stage.subcircuit._element_mgr.elem_dict['inner'] = dict(stage.subcircuit.elem_dict['inner'])
    with pytest.raises(CircuitError, match='replaced'):
        tb.probe(ref)


def test_held_array_never_attaches_to_replacement(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_module(block(tmp_path, 'arrayleaf'), 'X[8]', {'out': 'mid'})
    array = stage.X
    stage.subcircuit._element_mgr.elem_dict['X[8]'] = dict(stage.subcircuit.elem_dict['X[8]'])
    with pytest.raises(CircuitError, match='replaced'):
        array[8]


def test_cross_kind_and_reserved_names(tmp_path):
    tb, stage = bench(tmp_path)
    for name in ('child', 'path', 'vbias[0]'):
        stage.subcircuit.add_net(name)
        assert tb.probe(stage.net(name)).name == 'xstage.' + name
    stage.subcircuit.add(elements.resistor('mid', p='out', n='mid', value=1))
    with pytest.raises(CircuitError, match='Ambiguous'):
        _ = stage.mid
    assert tb.probe(stage.net('mid')).name == 'xstage.mid'
    assert stage.device('mid').name == 'mid'


def selected_bus(tmp_path):
    dut = block(tmp_path, 'selected_dut')
    leaf = block(tmp_path, 'bus_leaf')
    leaf.add_net('d', bits=range(8, 12))
    stage = dut.add_module(leaf, 'stage', {'out': 'out'})
    tb = dut.create_test('bus')
    tb.add_module_under_test({'out': 'out'})
    top = Module(None, str(tmp_path), 'root.cir', None)
    top.add_module(dut, 'one', {'out': 'a'})
    top.add_module(dut, 'two', {'out': 'b'})
    return top, tb, stage


def test_selected_aggregate_contexts_and_analysis_override(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    two = top.get_test('bus', instance='two')
    assert one.netlist is two.netlist
    p = one.probe(stage.d[8:10])
    q = two.probe(stage.d[8:10], analysis='tran')
    assert all(leaf._bench() is one for leaf in _unbound_leaves(p.as_int()))
    assert all(leaf._bench() is two for leaf in _unbound_leaves(q))
    assert all(leaf._analysis is None for leaf in _unbound_leaves(p))
    assert all(leaf._analysis == 'tran' for leaf in _unbound_leaves(q))
    with pytest.raises(UnboundSignalError, match='tb.probe'):
        _ = stage.d[8].y
    from silk.metrics.statistics import average
    one.add_metric('own', average, sig=p.as_int())
    with pytest.raises(TypeError, match='bench|test'):
        one.add_metric('foreign', average, sig=q.as_int())


def test_selected_owner_path_reference_can_bind(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    # Root placement references preserve the selected embedding, never a sibling.
    from silk.circuit._module_base import SubcircuitInstance
    entry = top.elem_dict['one']
    owner = SubcircuitInstance(top, 'xone', entry['device'], {'out': 'a'}, top)
    selected = top.get_test('bus', instance='one')
    assert selected.probe(owner.stage.mid).name == 'xdut.xstage.mid'
    with pytest.raises(CircuitError):
        top.get_test('bus', instance='two').probe(owner.stage.mid)


def test_aggregate_metric_deepcopy_and_pickle(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_net('d', bits=range(8, 10))
    from silk.metrics.statistics import average
    tb.add_metric('code', average, sig=tb.probe(stage.d).as_int())
    cloned = copy.deepcopy(tb)
    from silk.circuit.module_test import _metric_slot_leaves
    assert len(list(_metric_slot_leaves(cloned._metrics))) == 2
    assert all(leaf._bench() is cloned for leaf in _metric_slot_leaves(cloned._metrics))
    import pickle
    restored = pickle.loads(pickle.dumps(tb))
    assert all(leaf._bench() is restored for leaf in _metric_slot_leaves(restored._metrics))


def test_aggregate_retained_corner_sample_and_snapshot(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_net('d', bits=range(8, 10))
    p = tb.probe(stage.d)
    raw(tb, tmp_path, ['xstage.d[8]', 'xstage.d[9]'])
    path = str(tmp_path / 'literal.raw')
    tb._sweep_table = {'ff': path}
    np.testing.assert_array_equal(p.as_int().corner('ff').y, [3., 3.])
    from silk.circuit.module_test import _sample_label
    tb._sweep_table = {_sample_label(('tt', 2)): path}
    tb._mc_samples = {('tt', 2): path}
    np.testing.assert_array_equal(p.as_int().sample(('tt', 2)).y, [3., 3.])
    frozen = p.snapshot()
    assert frozen[8].bound
    tb.netlist.sim_results_loaded = False
    np.testing.assert_array_equal(frozen[8].y, [1., 1.5])
    with pytest.raises(UnboundSignalError):
        _ = p[8].y


def test_unsaved_internal_node_and_behavioral_leaf_refuse(tmp_path):
    tb, stage = bench(tmp_path)
    p = tb.probe(stage.mid)
    raw(tb, tmp_path, ['parent'])
    from silk.exceptions import SimulationError, NotSupportedError
    with pytest.raises(SimulationError, match='not found'):
        _ = p.y
    def model_1(ports, t, params):
        ports.out.v = np.ones_like(t)
    stage.subcircuit.set_behavioral_model(model_1, execution='vector')
    tb.netlist.simulator = 'behavioral'
    with pytest.raises(NotSupportedError, match='internal'):
        tb.probe(stage.mid)


def test_rebuild_bound_signal_follows_bench_structural_reference_is_stale(tmp_path):
    tb, stage = bench(tmp_path)
    ref = stage.out
    p = tb.probe(ref)
    tb._create_netlist()
    tb.add_module(block(tmp_path), 'stage', {'out': 'parent'})
    with pytest.raises(CircuitError, match='placement'):
        tb.probe(ref)
    raw(tb, tmp_path, ['parent'])
    np.testing.assert_array_equal(p.y, [1., 1.5])


def test_literal_voltage_registry_never_requests_device_parameters(tmp_path):
    tb, stage = bench(tmp_path)
    add_legacy_net(stage.subcircuit, 'M1:gm')
    tb.probe(stage.net('M1:gm'))
    from silk.circuit.module_test import _device_probe_requests
    assert _device_probe_requests(tb._probe_registry) == ()


def test_selected_metric_copy_preserves_owner_selection(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    from silk.metrics.statistics import average
    one.add_metric('code', average, sig=one.probe(stage.d[8:10]).as_int())
    cloned = copy.deepcopy(top)
    selected = cloned.get_test('bus', instance='one')
    from silk.circuit.module_test import _metric_slot_leaves
    leaves = list(_metric_slot_leaves(selected._metrics))
    assert len(leaves) == 2
    assert all(leaf._bench() is selected for leaf in leaves)


def test_selected_bus_readback_and_removal(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    p = one.probe(stage.d[8:10])
    raw(tb, tmp_path, ['xdut.xstage.d[8]', 'xdut.xstage.d[9]'])
    one.evaluate(rawfile=str(tmp_path / 'literal.raw'))
    np.testing.assert_array_equal(p.as_int().y, [3., 3.])
    with pytest.raises(UnboundSignalError, match='tb.probe'):
        _ = stage.d[8].y
    del top._element_mgr.elem_dict['two']
    np.testing.assert_array_equal(stage.d[8].y, [1., 1.5])


def test_width_one_decoder_and_report_consumers(tmp_path):
    from silk.signals.assemble import codes_from_bits
    from silk.circuit.report import render_bus_html_row, render_bus_trace
    import matplotlib.pyplot as plt
    bus = Signal._make_bus_handle('d', (Signal('d[8]', 'voltage', 'time', 'V', [0, 1], [0, 1]),), indices=(8,))
    np.testing.assert_array_equal(codes_from_bits(bus, msb_first=False).y, [0, 1])
    m = block(tmp_path)
    m.add_net('d', bits=range(8, 9))
    row = render_bus_html_row(m, 'd', behavioral_results={'d[8]': np.array([0, 1])})
    assert '1-bit bus' in row and 'min=0, max=1' in row
    fig, ax = plt.subplots()
    try:
        render_bus_trace(ax, m, 'd', behavioral_results={'d[8]': np.array([0, 1])})
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 1])
    finally:
        plt.close(fig)


def test_signal_copy_alone_keeps_selected_owner_without_copying_graph(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    p = selected.probe(stage.d)
    copied = copy.deepcopy(p)
    assert all(leaf._bench() is selected for leaf in _unbound_leaves(copied))


def test_direct_module_context_is_separate_from_authored_bench(tmp_path):
    tb, stage = bench(tmp_path)
    # This root is an execution deck and uses its bench context only.
    assert len(stage.mid._contexts()) == 1
    leaf = block(tmp_path, 'direct_leaf')
    root = Module(None, str(tmp_path), 'direct.cir', None)
    h = root.add_module(leaf, 'cell', {'out': 'wire'})
    ref = h.mid
    with pytest.raises(UnboundSignalError):
        _ = ref.y
    # Synthetic loading is the same Module result route as a direct simulation.
    from types import SimpleNamespace
    raw(SimpleNamespace(netlist=root), tmp_path, ['xcell.mid'])
    np.testing.assert_array_equal(ref.y, [1., 1.5])
    from silk.metrics.statistics import average
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        (ref + 2).plot(ax=ax)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [3., 3.5])
    finally:
        plt.close(fig)
    root.define_module_name('direct_root')
    root.define_module_ports({})
    measurement = root.create_test('measure')
    measurement.add_module_under_test({})
    with pytest.raises(UnboundSignalError, match='tb.probe'):
        _ = ref.y


def test_width_one_bus_wires_the_actual_bit(tmp_path):
    from silk.circuit.ports import Port
    child = Module(None, str(tmp_path), 'one.cir', None, module_name='onebit')
    child.define_module_ports({'d': Port.digital_input(bits=range(8, 9))})
    parent = Module(None, str(tmp_path), 'top.cir', None)
    bits = parent.add_net('data', bits=range(3, 4))
    h = parent.add_module(child, 'one', d=bits)
    assert h.path('d[8]') == 'data[3]'


def test_stale_deck_reference_cannot_turn_into_direct_module_reader(tmp_path):
    tb, stage = bench(tmp_path)
    ref = stage.mid
    raw(tb, tmp_path, ['xstage.mid'])
    tb._create_netlist()
    with pytest.raises(CircuitError, match='placement|context'):
        _ = ref.y


def test_two_placements_one_definition_read_separate_internal_nodes(tmp_path):
    tb, first = bench(tmp_path)
    second = tb.add_module(first.subcircuit, 'other', {'out': 'elsewhere'})
    p, q = tb.probe(first.mid), tb.probe(second.mid)
    raw(tb, tmp_path, ['xstage.mid', 'xother.mid', 'parent', 'elsewhere'])
    np.testing.assert_array_equal(p.y, [1., 1.5])
    np.testing.assert_array_equal(q.y, [2., 2.5])
    np.testing.assert_array_equal(tb.probe(second.out).y, [4., 4.5])


def test_object_reference_keeps_actual_raw_path_in_bracket_collision(tmp_path):
    from silk.exceptions import AmbiguousSignalError
    dut = block(tmp_path, 'arr_dut')
    tb = dut.create_test('collision')
    stage = tb.add_module(block(tmp_path), 'X[8]', {'out': 'out'})
    p = tb.probe(stage.mid)
    raw(tb, tmp_path, ['xX<8>.mid', 'xX[8].mid'])
    np.testing.assert_array_equal(p.y, [1., 1.5])
    with pytest.raises(AmbiguousSignalError):
        _ = tb.probe('xX[8].mid').y


def test_selected_aggregate_retained_corner_and_sample(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    p = selected.probe(stage.d[8:10]).as_int()
    raw(tb, tmp_path, ['xdut.xstage.d[8]', 'xdut.xstage.d[9]'])
    selected.evaluate(rawfile=str(tmp_path / 'literal.raw'))
    np.testing.assert_array_equal(p.corner('tt').y, [3., 3.])
    from silk.circuit.test_placement import execution_scope
    from silk.circuit.module_test import _sample_label
    with execution_scope(selected._entry()):
        tb._mc_samples = {('tt', 4): str(tmp_path / 'literal.raw')}
        tb._sweep_table = {_sample_label(('tt', 4)): str(tmp_path / 'literal.raw')}
    np.testing.assert_array_equal(p.sample(('tt', 4)).y, [3., 3.])


def test_selected_context_does_not_fall_back_after_failed_evaluation(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    p = selected.probe(stage.d[8])
    raw(tb, tmp_path, ['xdut.xstage.d[8]'])
    selected.evaluate(rawfile=str(tmp_path / 'literal.raw'))
    np.testing.assert_array_equal(p.y, [1., 1.5])
    from silk.circuit.test_placement import begin_execution, execution_scope
    with execution_scope(selected._entry(), executing=True):
        begin_execution(tb)
    with pytest.raises(UnboundSignalError):
        _ = p.y
    with pytest.raises(UnboundSignalError, match='tb.probe'):
        _ = stage.d[8].y


def test_reflected_arithmetic_reads_both_references(tmp_path):
    tb, stage = bench(tmp_path)
    p = tb.probe(stage.out)
    expressions = [stage.out - stage.mid, p + stage.mid, stage.mid + p,
                   stage.out / stage.mid, 2 * stage.mid, 2 - stage.out]
    raw(tb, tmp_path, ['parent', 'xstage.mid'])
    expected = [[-1, -1], [3, 4], [3, 4], [.5, .6], [4, 5], [1, .5]]
    for expression, values in zip(expressions, expected):
        np.testing.assert_allclose(expression.y, values)


def test_held_handles_refuse_in_place_definition_replacement(tmp_path):
    tb, stage = bench(tmp_path)
    ref = stage.mid
    stage.subcircuit.add_module(block(tmp_path, 'arrayleaf'), 'X[8]', {'out': 'mid'})
    array = stage.X
    stage.subcircuit.elem_dict['X[8]']['device'] = block(tmp_path, 'other')
    with pytest.raises(CircuitError, match='replaced'):
        array[8]
    tb.netlist.elem_dict['stage']['device'] = block(tmp_path, 'replacement')
    with pytest.raises(CircuitError, match='replaced'):
        tb.probe(ref)
    with pytest.raises(CircuitError, match='replaced'):
        stage.mid


def test_literal_hierarchy_readback_preserves_local_colons(tmp_path):
    tb, stage = bench(tmp_path)
    add_legacy_net(stage.subcircuit, 'mid:literal')
    signal = tb.probe(stage.net('mid:literal'))
    raw(tb, tmp_path, ['xstage:mid:literal'])
    np.testing.assert_array_equal(signal.y, [1., 1.5])


def test_implicit_plot_and_bound_signal_survive_structural_replacement(tmp_path):
    import matplotlib.pyplot as plt
    tb, stage = bench(tmp_path)
    reference = stage.mid
    bound = tb.probe(reference)
    raw(tb, tmp_path, ['xstage.mid'])
    fig, ax = plt.subplots()
    try:
        reference.plot(ax=ax)
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1., 1.5])
    finally:
        plt.close(fig)
    tb.netlist.elem_dict['stage']['device'] = block(tmp_path, 'replacement')
    with pytest.raises(CircuitError, match='replaced'):
        _ = reference.y
    np.testing.assert_array_equal(bound.y, [1., 1.5])


def test_held_bus_slice_never_refreshes_placement_identity(tmp_path):
    tb, stage = bench(tmp_path)
    stage.subcircuit.add_net('d', bits=range(8, 10))
    held = stage.d
    replacement = block(tmp_path, 'replacement')
    replacement.add_net('d', bits=range(8, 10))
    tb.netlist.elem_dict['stage']['device'] = replacement
    with pytest.raises(CircuitError, match='replaced'):
        held[:]
