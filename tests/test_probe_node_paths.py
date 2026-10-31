"""Designer node strings preserve raw identity and deferred acquisition."""
import copy
import warnings

import numpy as np
import pytest

from tests._legacy_names import add_legacy_net

from silk import Module
from silk.circuit.modules_results import DUT_ELEMENT_NAME
from silk.exceptions import AmbiguousSignalError, CircuitError, NotSupportedError, SimulationError, UnboundSignalError
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile


def block(tmp_path, name='leaf'):
    mod = Module(None, str(tmp_path), name + '.cir', None, module_name=name)
    mod.define_module_ports({'out': 'output'})
    mod.add_net('mid')
    return mod


def bench(tmp_path):
    dut = block(tmp_path, 'design')
    dut.add_module(block(tmp_path, 'child'), 'x1[0]', {'out': 'mid'})
    tb = dut.create_test('read')
    tb.add_module_under_test({'out': 'vout'})
    tb.add_module(block(tmp_path, 'helper'), 'stage', {'out': 'vout'})
    return tb


def load(tb, tmp_path, names, *, filename='data.raw', extra=()):
    def plot(title, vectors):
        axis = 'frequency' if title == 'AC Analysis' else 'time'
        rows = [RawVector(axis, axis, data=np.array([1., 2.]))]
        rows += [RawVector(name, 'voltage', data=np.array([i, i + .5]))
                 for i, name in enumerate(vectors, 1)]
        return RawPlot({'plotname': title, 'title': 'node paths'}, rows,
                       npoints=2, real=True, padded=True)
    path = tmp_path / filename
    write_rawfile(path, RawFile([plot('Transient Analysis', names),
                                *(plot(title, rows) for title, rows in extra)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    return str(path)


@pytest.mark.parametrize('designer,emitted', [
    ('dut.mid', 'xdut.mid'), ('dut.x1[0].mid', 'xdut.xx1<0>.mid'),
    ('stage.mid', 'xstage.mid'), ('dut.out', 'vout'),
    ('dut.x1[0].out', 'xdut.mid'), ('v(dut.mid)', 'xdut.mid'),
])
def test_graph_strings_match_emitted_voltage(tmp_path, designer, emitted):
    tb = bench(tmp_path)
    probe = tb.probe(designer)
    load(tb, tmp_path, [f'v({emitted})'])
    np.testing.assert_array_equal(probe.y, tb.probe(emitted).y)
    assert tb.netlist.get_signal_names(raw=True) == ['time', f'v({emitted})']


def test_declaration_before_and_after_external_file_has_same_raw_precedence(tmp_path):
    tb = bench(tmp_path)
    before = tb.probe('dut.mid')
    load(tb, tmp_path, ['v(dut.mid)', 'v(xdut.mid)'])
    after = tb.probe('dut.mid')
    for p in (before, after):
        with pytest.warns(UserWarning, match=r'dut.mid.*xdut.mid'):
            np.testing.assert_array_equal(p.y, [1., 1.5])


def test_raw_ambiguity_cannot_be_hidden_by_graph(tmp_path):
    tb = bench(tmp_path)
    p = tb.probe('dut.x1[0].mid')
    load(tb, tmp_path, ['v(dut.x1[0].mid)', 'v(dut.x1<0>.mid)', 'v(xdut.xx1<0>.mid)'])
    with pytest.raises(AmbiguousSignalError):
        _ = p.y


def test_dotted_instance_is_refused_but_nested_selector_works(tmp_path):
    tb = bench(tmp_path)
    with pytest.raises(CircuitError, match='hierarchy levels'):
        tb.add_module(block(tmp_path, 'flat'), 'a.b', {'out': 'vout'})
    parent = block(tmp_path, 'parent')
    parent.add_module(block(tmp_path, 'nested'), 'b', {'out': 'out'})
    tb.add_module(parent, 'a', {'out': 'vout'})
    load(tb, tmp_path, ['v(xa.xb.mid)'])
    np.testing.assert_array_equal(tb.probe('a.b.mid').y, [1., 1.5])


def test_absent_results_missing_vector_and_removed_placement_are_distinct(tmp_path):
    tb = bench(tmp_path)
    p = tb.probe('stage.mid')
    with pytest.raises(UnboundSignalError):
        _ = p.y
    load(tb, tmp_path, ['v(vout)'])
    with pytest.raises(SimulationError, match='not found'):
        _ = p.y
    tb.netlist.elem_dict['stage']['device'].remove_net('mid')
    with pytest.raises(CircuitError, match='(absent|removed|no longer)'):
        _ = p.y


def test_analysis_corner_and_reread_use_selected_inventory(tmp_path):
    tb = bench(tmp_path)
    p = tb.probe('dut.mid', analysis='tran')
    first = load(tb, tmp_path, ['v(xdut.mid)'], filename='first.raw',
                 extra=[('AC Analysis', ['v(dut.mid)', 'v(xdut.mid)'])])
    np.testing.assert_array_equal(p.y, [1., 1.5])
    with pytest.warns(UserWarning):
        np.testing.assert_array_equal(tb.probe('dut.mid', analysis='ac').y, [1., 1.5])
    tb._sweep_table = {'tt': first}
    np.testing.assert_array_equal(p.corner('tt').y, [1., 1.5])
    load(tb, tmp_path, ['v(vout)', 'v(xdut.mid)'], filename='second.raw')
    np.testing.assert_array_equal(p.y, [2., 2.5])
    np.testing.assert_array_equal(p.corner('tt').y, [1., 1.5])


def test_copy_uses_its_own_graph(tmp_path):
    tb = bench(tmp_path)
    original = tb.probe('dut.out')
    cloned = copy.deepcopy(tb)
    cloned.netlist.elem_dict[DUT_ELEMENT_NAME]['pin_list']['out'].net = 'other'
    load(tb, tmp_path, ['v(vout)', 'v(other)'])
    load(cloned, tmp_path, ['v(vout)', 'v(other)'], filename='copy.raw')
    np.testing.assert_array_equal(original.y, [1., 1.5])
    np.testing.assert_array_equal(copy.deepcopy(original).resolve(cloned.netlist).y, [2., 2.5])


@pytest.mark.parametrize('relative', ['mid', 'x1[0].mid'])
def test_missing_dut_prefix_is_explained(tmp_path, relative):
    tb = bench(tmp_path)
    load(tb, tmp_path, ['v(xdut.mid)', 'v(xdut.xx1<0>.mid)'])
    with pytest.raises((ValueError, CircuitError, SimulationError), match=r'DUT is placed as.*dut') as exc:
        _ = tb.probe(relative).y
    assert DUT_ELEMENT_NAME + '.' + relative in str(exc.value)


def test_literal_leaf_and_xyce_hierarchy(tmp_path):
    tb = bench(tmp_path)
    child = tb.module.elem_dict['x1[0]']['device']
    child.add_net('g[8]')
    add_legacy_net(child, 'literal:mid')
    load(tb, tmp_path, ['V(XDUT:XX1<0>:g[8])', 'V(XDUT:XX1<0>:literal:mid)'])
    np.testing.assert_array_equal(tb.probe('dut.x1[0].g[8]').y, [1., 1.5])
    np.testing.assert_array_equal(tb.probe('v(dut.x1[0].literal:mid)').y, [2., 2.5])


@pytest.mark.parametrize('selector,expected', [
    ('dut.x1[0].mid', ['xdut.xx1<0>.mid']),
    ('dut.out', ['vout']), ('dut.x1[0].out', ['xdut.mid']),
    ('dut.x1[*].mid', ['xdut.xx1<0>.mid']),
    ('STAGE.MID', ['xstage.mid']),
])
def test_save_designer_selectors(tmp_path, selector, expected):
    from silk.circuit.modules_netlist import _resolve_save_list
    tb = bench(tmp_path)
    tb.netlist.set_save_list(selector)
    assert _resolve_save_list(tb.netlist) == expected


def test_save_exclude_checks_full_inventory_and_physical_depth(tmp_path):
    from silk.circuit.modules_netlist import _resolve_save_list
    tb = bench(tmp_path)
    tb.netlist.set_save_list('dut.out', exclude='stage.mid', max_depth=0)
    assert _resolve_save_list(tb.netlist) == ['vout']
    tb.netlist.set_save_list(['dut', 'stage'], exclude='dut.x1[0].mid')
    assert set(_resolve_save_list(tb.netlist)) == {'vout', 'xdut.mid', 'xstage.mid'}


def test_save_emitted_set_wins_with_warning_including_exclude(tmp_path):
    from silk.circuit.modules_netlist import _resolve_save_list
    tb = bench(tmp_path)
    add_legacy_net(tb.netlist, 'stage.mid')
    tb.netlist.set_save_list('stage.mid')
    with pytest.warns(UserWarning, match='stage.mid'):
        assert _resolve_save_list(tb.netlist) == ['stage.mid']
    tb.netlist.set_save_list('*', exclude='stage.mid')
    with pytest.warns(UserWarning, match='stage.mid'):
        assert 'xstage.mid' in _resolve_save_list(tb.netlist)


def test_save_ground_and_wrapped_names_do_not_invent_vectors(tmp_path):
    from silk.circuit.modules_netlist import _resolve_save_list
    from silk.exceptions import CircuitError
    tb = bench(tmp_path)
    tb.netlist.elem_dict[DUT_ELEMENT_NAME]['pin_list']['out'].net = '0'
    for selector in ['dut.out', 'v(dut.mid)']:
        tb.netlist.set_save_list(selector)
        with pytest.raises(CircuitError, match='matched no signal'):
            _resolve_save_list(tb.netlist)


def test_save_nested_targets_and_declared_indices_are_sets(tmp_path):
    from silk.circuit.modules_netlist import _resolve_save_list
    tb = bench(tmp_path)
    parent = block(tmp_path, 'parent')
    parent.add_module(block(tmp_path, 'nested'), 'b', {'out': 'out'})
    tb.add_module(parent, 'a', {'out': 'vout'})
    for key in ['other[3]', 'other[8]', 'other_8']:
        tb.add_module(block(tmp_path, 'indexed'), key, {'out': 'vout'})
    tb.netlist.set_save_list(['a.b.mid', 'other[*].mid', 'other_8.mid'])
    assert set(_resolve_save_list(tb.netlist)) == {
        'xa.xb.mid', 'xother<3>.mid', 'xother<8>.mid', 'xother_8.mid'}


@pytest.mark.parametrize('roundtrip', ['none', 'copy', 'pickle'])
def test_selected_string_metrics_and_probes_do_not_follow_siblings(tmp_path, roundtrip):
    import pickle
    from silk.metrics.statistics import average
    tb = bench(tmp_path)
    top = Module(None, str(tmp_path), 'root.cir', None)
    top.add_module(tb.module, 'one', {'out': 'a'})
    top.add_module(tb.module, 'two', {'out': 'b'})
    one = top.get_test('read', instance='one')
    first = load(tb, tmp_path, ['v(xdut.mid)'], filename='one.raw')
    second = load(tb, tmp_path, ['v(vout)', 'v(xdut.mid)'], filename='two.raw')
    two = top.get_test('read', instance='two')
    two.evaluate(rawfile=second)
    np.testing.assert_array_equal(two.probe('dut.mid').y, [2., 2.5])
    one.add_metric('level', average, sig=one.probe('dut.mid'))
    if roundtrip == 'copy':
        top = copy.deepcopy(top)
    elif roundtrip == 'pickle':
        top = pickle.loads(pickle.dumps(top))
    one = top.get_test('read', instance='one')
    two = top.get_test('read', instance='two')
    one.evaluate(rawfile=first)
    from silk.exceptions import MetricBindingError
    with pytest.raises(MetricBindingError):
        two.evaluate(rawfile=second)
    np.testing.assert_array_equal(one.probe('dut.mid').y, [1., 1.5])
    assert one.metrics['level'] == pytest.approx(1.25)


def test_wrapped_literal_colon_does_not_become_device_probe(tmp_path):
    tb = bench(tmp_path)
    load(tb, tmp_path, ['v(literal:gm)'])
    np.testing.assert_array_equal(tb.probe('v(literal:gm)').y, [1., 1.5])


def test_unconnected_other_placement_does_not_break_valid_node(tmp_path):
    tb = bench(tmp_path)
    tb.add_module(block(tmp_path, 'unwired'), 'unwired')
    load(tb, tmp_path, ['v(xdut.mid)'])
    np.testing.assert_array_equal(tb.probe('dut.mid').y, [1., 1.5])


def test_typed_port_address_does_not_alias_literal_bracketed_parent(tmp_path):
    tb = bench(tmp_path)
    add_legacy_net(tb.netlist, 'xx1[0].mid')
    tb.netlist.elem_dict['stage']['pin_list']['out'].net = 'xx1[0].mid'
    load(tb, tmp_path, ['v(xx1[0].mid)', 'v(xx1<0>.mid)'])
    np.testing.assert_array_equal(tb.probe('stage.out').y, [1., 1.5])


def test_same_emitted_and_designer_target_does_not_warn(tmp_path):
    tb = bench(tmp_path)
    load(tb, tmp_path, ['v(vout)'])
    with warnings.catch_warnings(record=True) as caught:
        np.testing.assert_array_equal(tb.probe('vout').y, [1., 1.5])
    assert not caught


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_equivalent_save_spellings_emit_identical_decks(tmp_path, simulator):
    from silk.simulation import Transient
    tb = bench(tmp_path)
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    tb.netlist.simulator = simulator
    tb.netlist.set_output_options(mode='overwrite')
    tb.netlist.set_save_list(['dut.x1[0].mid', 'dut.out'])
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    from pathlib import Path
    designer_deck = Path(tb_netlist_deck_path).read_bytes()
    tb.netlist.set_save_list(['xdut.xx1<0>.mid', 'vout'])
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    assert Path(tb_netlist_deck_path).read_bytes() == designer_deck


def test_ground_probe_does_not_return_fabricated_zero(tmp_path):
    tb = bench(tmp_path)
    tb.netlist.elem_dict[DUT_ELEMENT_NAME]['pin_list']['out'].net = '0'
    load(tb, tmp_path, ['v(xdut.mid)'])
    with pytest.raises(SimulationError, match='not found'):
        _ = tb.probe('dut.out').y


def test_raw_reading_wins_over_unconnected_designer_port(tmp_path):
    tb = bench(tmp_path)
    tb.add_module(block(tmp_path, 'unwired'), 'unwired')
    load(tb, tmp_path, ['v(unwired.out)'])
    np.testing.assert_array_equal(tb.probe('unwired.out').y, [1., 1.5])


@pytest.mark.parametrize('name', ['b.mid', 'a:b'])
def test_literal_hierarchy_components_cannot_be_authored(tmp_path, name):
    tb = bench(tmp_path)
    with pytest.raises(ValueError, match='hierarchy levels'):
        tb.module.add_net(name)


def test_designer_fallback_reads_bare_external_voltage_rows(tmp_path):
    tb = bench(tmp_path)
    load(tb, tmp_path, ['xdut.mid'])
    np.testing.assert_array_equal(tb.probe('dut.mid').y, tb.probe('xdut.mid').y)


@pytest.mark.parametrize('copied', [False, True])
def test_wrapped_designer_voltage_prepares_without_device_requests(tmp_path, copied):
    from silk.circuit.module_test import _device_probe_requests
    from silk.metrics.statistics import average
    from silk.simulation import Transient

    tb = bench(tmp_path)
    tb.module.add_net('literal[3]')
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    tb.add_metric('level', average,
                  sig=tb.probe(f'v({DUT_ELEMENT_NAME}.literal[3])'))
    if copied:
        tb = copy.deepcopy(tb)
    assert len(tb._probe_registry) == 1
    tb.prepare()  # Deck generation only; no simulator or preflight is allowed.
    assert _device_probe_requests(tb._probe_registry) == ()
    assert tb.netlist._pending_device_probes == ()


def test_genuine_device_parameter_remains_pending(tmp_path):
    from silk.circuit.module_test import _device_probe_requests
    from silk.pdk import load_pdk
    from silk.simulation import Transient
    pdk = load_pdk('sky130')
    module = Module('sky130', str(tmp_path), 'device.cir',
                    pdk_details=pdk, module_name='device')
    module.define_module_ports({'out': 'output'})
    module.add(pdk.nmos1p8('M1', d='out', g='out', s='0', b='0', w=1, l=.15))
    tb = module.create_test('prepare_device')
    tb.add_module_under_test({'out': 'vout'})
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    tb.probe('M1:gm')
    tb.prepare()  # gm is declared by the catalog: no auxiliary preflight.
    assert _device_probe_requests(tb._probe_registry) == (('M1', 'gm'),)
    assert tb.netlist._pending_device_probes == (('M1', 'gm'),)


@pytest.mark.parametrize('suffix', ['gm', 'd'])
def test_dut_hint_preserves_voltage_and_reads_suggested_vector(tmp_path, suffix):
    import ast
    import re
    from silk.circuit.emitted_names import emitted_submodule_name

    tb = bench(tmp_path)
    add_legacy_net(tb.module, f'literal:{suffix}')
    emitted = emitted_submodule_name(DUT_ELEMENT_NAME)
    load(tb, tmp_path, [f'v({emitted}.literal:{suffix})'])
    with pytest.raises(CircuitError) as error:
        _ = tb.probe(f'v(literal:{suffix})').y
    argument = re.search(r'use probe\((.+)\) for that node', str(error.value)).group(1)
    suggested = ast.literal_eval(argument)
    assert suggested == f'v({DUT_ELEMENT_NAME}.literal:{suffix})'
    np.testing.assert_array_equal(tb.probe(suggested).y, [1., 1.5])


def test_behavioral_raw_voltage_is_not_reclassified_as_device(tmp_path):
    from silk.simulation.behavioral_backend import BEHAVIORAL_COMMAND
    tb = bench(tmp_path)
    load(tb, tmp_path, ['v(literal:gm)'])
    tb.netlist._raw.native.current_plot.header['command'] = BEHAVIORAL_COMMAND
    np.testing.assert_array_equal(tb.probe('v(literal:gm)').y, [1., 1.5])


def test_retained_corner_literal_object_voltage_preserves_kind(tmp_path):
    tb = bench(tmp_path)
    module = block(tmp_path, 'colon_leaf')
    add_legacy_net(module, 'literal:gm')
    stage = tb.add_module(module, 'colon', {'out': 'vout'})
    path = load(tb, tmp_path, ['v(xcolon.literal:gm)'])
    tb._sweep_table = {'tt': path}
    probe = tb.probe(stage.net('literal:gm'))
    np.testing.assert_array_equal(probe.corner('tt').y, [1., 1.5])


def test_voltage_request_does_not_enter_source_current_validation(tmp_path):
    from silk.circuit.modules_results import _source_current_probe_name
    from silk.circuit.node_paths import NodeVoltageRequest
    from silk.sources import CurrentSource

    tb = bench(tmp_path)
    tb.add_source(CurrentSource('stim', positive='vout', negative='0', dc=1))
    load(tb, tmp_path, [], extra=[('AC Analysis', ['v(i(stim))'])])
    request = NodeVoltageRequest('i(stim)')
    assert _source_current_probe_name(tb.netlist, tb.netlist._raw, request,
                                      'V', 'ac') is request



def connected_block(tmp_path, name):
    from silk.elements import elements
    mod = block(tmp_path, name)
    mod.add(elements.resistor('r', p='out', n='mid', value=1000))
    mod.add(elements.resistor('load', p='mid', n='0', value=1000))
    return mod


def xyce_bench(tmp_path):
    dut = connected_block(tmp_path, 'design')
    dut.add_module(connected_block(tmp_path, 'child'), 'x1[0]', {'out': 'mid'})
    tb = dut.create_test('read')
    tb.add_module_under_test({'out': 'vout'})
    tb.add_module(connected_block(tmp_path, 'helper'), 'stage', {'out': 'vout'})
    return tb

def xyce_deck(tb):
    from silk.simulation import Transient
    tb.netlist.simulator = 'xyce'
    if not tb.netlist.sim_analysis_dict:
        tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    spec = tb._prepare_run(corner='tt')
    spec_netlist_deck_path = spec.netlist.generate_netlist(corner=spec.corner_str)
    from pathlib import Path
    return Path(spec_netlist_deck_path).read_text()


def test_xyce_default_captures_declared_nodes_and_port_aliases(tmp_path):
    tb = xyce_bench(tmp_path)
    tb.probe('dut.x1[0].mid')
    tb.probe('xdut.xx1[0].mid')
    tb.probe('dut.x1[0].out')
    tb.probe('dut.out')
    line = next(line for line in xyce_deck(tb).splitlines() if line.startswith('.print'))
    assert line.count('v(xdut:xx1<0>:mid)') == 1
    assert line.count('v(xdut:mid)') == 1
    assert line.count('v(vout)') == 1


@pytest.mark.parametrize('name', ['dut.missing', 'xdut.opaque#body', 'v(0)'])
def test_xyce_invalid_node_probe_refuses_before_launch(tmp_path, name):
    tb = xyce_bench(tmp_path)
    tb.netlist.add_net('0')
    tb.probe(name)
    with pytest.raises(CircuitError) as exc:
        xyce_deck(tb)
    assert name.removeprefix('v(').removesuffix(')') in str(exc.value)
    assert 'probe' in str(exc.value)
    assert 'Xyce' in str(exc.value)


def test_xyce_explicit_save_list_still_omits_declared_node(tmp_path):
    tb = xyce_bench(tmp_path)
    tb.probe('dut.x1[0].mid')
    tb.netlist.set_save_list(['vout'])
    line = next(line for line in xyce_deck(tb).splitlines() if line.startswith('.print'))
    assert line.endswith('v(vout)')
    tb.netlist.set_save_list(None)
    assert 'v(xdut:xx1<0>:mid)' in xyce_deck(tb)


def test_xyce_next_run_includes_new_probe(tmp_path):
    tb = xyce_bench(tmp_path)
    assert 'v(xstage:mid)' not in xyce_deck(tb)
    tb.probe('stage.mid')
    assert 'v(xstage:mid)' in xyce_deck(tb)


@pytest.mark.parametrize('name', ['xdut.xx1[0].mid', 'a.b', 'a,b', 'a(b)'])
def test_xyce_invalid_authored_net_refuses_at_construction(tmp_path, name):
    tb = xyce_bench(tmp_path)
    with pytest.raises(ValueError, match='add_net: invalid net name') as exc:
        tb.netlist.add_net(name)
    assert name in str(exc.value)
    assert name not in tb.netlist.net_dict


def test_xyce_legacy_ambiguous_inventory_refuses_at_capture_boundary(tmp_path):
    from silk.circuit.node_paths import NodeVoltageRequest, default_output_nodes
    tb = xyce_bench(tmp_path)
    # Synthetic pre-553 inventory: a literal leaf aliases an emitted path.
    # Call capture directly; emission correctly refuses this graph earlier.
    add_legacy_net(tb.netlist, 'xdut.xx1[0].mid')
    tb.netlist.simulator = 'xyce'
    tb.netlist._pending_node_probes = (NodeVoltageRequest('xdut.xx1[0].mid'),)
    with pytest.raises(NotSupportedError, match='Xyce node probe.*xx1') as exc:
        default_output_nodes(tb.netlist)
    assert 'expected one connected design node' in str(exc.value)


def test_xyce_placement_reference_requests_internal_voltage(tmp_path):
    tb = xyce_bench(tmp_path)
    stage = tb.add_module(connected_block(tmp_path, 'object'), 'object', {'out': 'vout'})
    tb.probe(stage.net('mid'))
    assert 'v(xobject:mid)' in xyce_deck(tb)


def test_xyce_declared_but_unused_node_refuses(tmp_path):
    name = 'unused'
    tb = xyce_bench(tmp_path)
    mod = connected_block(tmp_path, 'bad')
    mod.add_net(name)
    stage = tb.add_module(mod, 'bad', {'out': 'vout'})
    tb.probe(stage.net(name))
    with pytest.raises(CircuitError, match='Xyce node probe') as exc:
        xyce_deck(tb)
    assert name in str(exc.value)


@pytest.mark.parametrize('name', ['a.b', 'a,b', 'a(b)'])
def test_xyce_legacy_unprintable_inventory_refuses_at_capture_boundary(tmp_path, name):
    from silk.circuit.node_paths import NodeVoltageRequest, default_output_nodes
    mod = connected_block(tmp_path, 'legacy')
    # Model historical inventory with connected pins, so punctuation rather
    # than missing connectivity is the reason capture must refuse the node.
    net = add_legacy_net(mod, name)
    net.pins.extend(mod.net_dict['mid'].pins)
    assert net.pins
    mod.simulator = 'xyce'
    mod._pending_node_probes = (NodeVoltageRequest(name),)
    with pytest.raises(NotSupportedError, match='Xyce node probe') as exc:
        default_output_nodes(mod)
    assert name in str(exc.value)
    assert 'expected one connected design node' in str(exc.value)


def test_xyce_noise_cannot_capture_declared_node(tmp_path):
    from silk.simulation import Noise
    tb = xyce_bench(tmp_path)
    from silk import sources
    tb.add_source(sources.VoltageSource('vin', positive='vout', negative='0', dc=1))
    tb.probe('dut.mid')
    tb.add_sim_type(Noise('vout', '0', 'vin', 'dec', 10, 1, 1e6))
    with pytest.raises(CircuitError, match='Xyce node probe.*dut.mid.*noise'):
        xyce_deck(tb)


def test_xyce_generated_source_private_node_is_printable(tmp_path):
    from silk import sources
    from silk.sources.waveforms import Pulse
    tb = xyce_bench(tmp_path)
    tb.add_source(sources.VoltageSource('bias[0]', positive='vout', negative='0',
                                       pulse=Pulse(0, 1, tr=1e-9, tf=1e-9, pw=2e-9, per=4e-9)))
    tb.probe('__silk_src_Vbias[0]')
    assert 'v(__silk_src_Vbias[0])' in xyce_deck(tb)

@pytest.mark.parametrize('name', ['literal:gm', 'xdut.xx1[0].mid', 'a.b', 'a,b', 'a(b)'])
def test_legacy_invalid_net_cannot_be_emitted(tmp_path, name):
    from silk.simulation import Transient
    tb = bench(tmp_path)
    add_legacy_net(tb.module, name)
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    with pytest.raises(ValueError, match='invalid net name') as exc:
        tb.prepare()
    assert name in str(exc.value)


@pytest.mark.parametrize('binding', [
    {'signal': 'dut.x1[0].mid'},
    {'signals': {'sig': 'dut.x1[0].mid'}},
])
def test_xyce_declarative_metric_captures_node(tmp_path, binding):
    tb = xyce_bench(tmp_path)
    tb.add_metric('level', type='average', **binding)
    assert 'v(xdut:xx1<0>:mid)' in xyce_deck(tb)
    assert not tb._probe_registry


@pytest.mark.parametrize('path', ['0', 'dut.missing', 'dut.unused', 'xdut.opaque#body'])
def test_xyce_declarative_metric_refusal_names_metric_and_path(tmp_path, path):
    tb = xyce_bench(tmp_path)
    tb.module.add_net('unused')
    tb.netlist.add_net('0')
    tb.add_metric('bad_level', type='average', signal=path)
    with pytest.raises(NotSupportedError) as exc:
        xyce_deck(tb)
    assert 'bad_level' in str(exc.value)
    assert path in str(exc.value)
    assert 'Use a connected design node' in str(exc.value)


def test_declarative_designer_metric_reads_emitted_node(tmp_path):
    tb = bench(tmp_path)
    tb.add_metric('level', type='average', signal='dut.x1[0].mid')
    path = load(tb, tmp_path, ['v(xdut.xx1<0>.mid)'])
    report = tb.evaluate(rawfile=path)
    assert report.metrics['level'] == pytest.approx(1.25)


def test_xyce_declarative_every_slot_deduplicates_and_refreshes(tmp_path):
    tb = xyce_bench(tmp_path)
    tb.add_metric('product', type='average_power',
                  signals={'sig_v': 'dut.mid', 'sig_i': 'stage.mid'})
    tb.add_metric('alias', type='average', signal='xdut.mid', analysis='tran')
    line = next(line for line in xyce_deck(tb).splitlines() if line.startswith('.print'))
    assert line.count('v(xdut:mid)') == line.count('v(xstage:mid)') == 1
    tb.add_metric('later', type='average', signal='dut.x1[0].mid')
    assert 'v(xdut:xx1<0>:mid)' in xyce_deck(tb)


def test_declarative_save_authority_and_ngspice_deck_identity(tmp_path):
    from pathlib import Path
    from silk.simulation import Transient
    tb = xyce_bench(tmp_path)
    tb.module.set_output_options(mode='overwrite')
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    deck_path = tb.prepare()
    before = Path(deck_path).read_bytes()
    tb.add_metric('level', type='average', signal='dut.x1[0].mid')
    deck_path = tb.prepare()
    assert Path(deck_path).read_bytes() == before
    tb.netlist.set_save_list(['vout'])
    assert 'v(xdut:xx1<0>:mid)' not in xyce_deck(tb)
    tb.netlist.set_save_list(None)
    assert 'v(xdut:xx1<0>:mid)' in xyce_deck(tb)


@pytest.mark.parametrize('path', ['i(vrail)', 'M1:gm', 'xdut.M1:d',
                                  '@m1[gm]', 'v(@m1[vth])', 'vrail#branch', 'tran:vout', 'noise_out', 'inoise_spectrum'])
def test_declarative_special_paths_keep_existing_routes(tmp_path, path, monkeypatch):
    from silk.circuit.module_test import _load_metric_path
    tb = xyce_bench(tmp_path)
    tb.add_metric('special', type='average', signal=path)
    xyce_deck(tb)
    assert tb.netlist._pending_metric_nodes == ()
    calls = []
    monkeypatch.setattr(tb.netlist, 'load_signal', lambda *a, **kw: calls.append((a, kw)))
    _load_metric_path(tb.netlist, path, analysis='tran')
    expected = (('vrail', 'I'), {'analysis': 'tran'}) if path == 'i(vrail)' else (
        (path,), {'analysis': 'tran'})
    assert calls == [expected]


def generated_node_metric_bench(tmp_path, paths):
    from silk.codegen.testbench_generator import generate_testbenches
    from silk.elements import elements
    def build(pdk):
        dut = connected_block(tmp_path, 'design')
        dut.add(elements.resistor('bus_upper', p='mid', n='y[2]', value=1000))
        dut.add(elements.resistor('bus_lower', p='y[2]', n='0', value=1000))
        return dut
    spec = {
        'circuit_name': 'design', 'circuit_builder': '_build_design',
        'tests': [{'name': 'read', 'dut_portmap': {'out': 'rail'},
                   'sources': [{'type': 'VoltageSource', 'instance': 'rail',
                                'positive': 'rail', 'negative': '0', 'dc': 1.0}],
                   'analyses': [{'kind': 'transient', 'step': 1e-9, 'stop': 5e-9}],
                   'metrics': [{'name': f'level_{i}', 'type': 'average', 'signal': path}
                               for i, path in enumerate(paths)]}]}
    code = generate_testbenches(spec)
    for i, path in enumerate(paths):
        assert f"tb.add_metric('level_{i}', type='average', signal={path!r})" in code
    ns = {'_build_design': build}
    exec(code, ns)
    return ns['build_design_with_tests'](None)._tests['read']


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('path,term', [('xdut.mid', 'v(xdut:mid)'),
                                      ('xdut.y[2]', 'v(xdut:y[2])'),
                                      ('0', None), ('xdut.unknown', None)])
def test_generated_declarative_paths_prepare(tmp_path, simulator, path, term):
    from pathlib import Path
    tb = generated_node_metric_bench(tmp_path, [path])
    tb.netlist.simulator = simulator
    if simulator == 'xyce' and term is None:
        with pytest.raises(NotSupportedError) as exc:
            tb.prepare()
        assert 'level_0' in str(exc.value) and path in str(exc.value)
    else:
        deck_path = tb.prepare()
        deck = Path(deck_path).read_text()
        assert (term in deck) if simulator == 'xyce' else ('save all' in deck)


@pytest.mark.parametrize('path', ['v(dut.mid)', 'dut.mid'])
def test_declarative_metric_analysis_selection_and_designer_readback(tmp_path, path):
    tb = bench(tmp_path)
    from silk.simulation import AC, Transient
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.add_sim_type(AC('dec', 10, 1, 1e6))
    tb.add_metric('selected', type='average', signal=path, analysis='tran')
    raw = load(tb, tmp_path, ['v(xdut.mid)'], extra=[('AC Analysis', ['v(xdut.mid)'])])
    report = tb.evaluate(rawfile=raw)
    assert report.metrics['selected'] == pytest.approx(1.25)


@pytest.mark.parametrize('name', ['a.b', 'a,b', 'a(b)'])
def test_declarative_legacy_punctuation_reuses_probe_refusal(tmp_path, name):
    from silk.circuit.module_test import _metric_node_requests
    from silk.circuit.node_paths import NodeVoltageRequest, default_output_nodes
    tb = xyce_bench(tmp_path)
    net = add_legacy_net(tb.netlist, name)
    net.pins.extend(tb.netlist.net_dict['vout'].pins)
    tb.netlist.simulator = 'xyce'
    tb.netlist._pending_node_probes = (NodeVoltageRequest(name),)
    with pytest.raises(NotSupportedError) as probe_error:
        default_output_nodes(tb.netlist)
    tb.netlist._pending_node_probes = ()
    tb.add_metric('legacy_level', type='average', signal=name)
    tb.netlist._pending_metric_nodes = tuple(_metric_node_requests(tb._metrics, ['tran']))
    with pytest.raises(NotSupportedError) as metric_error:
        default_output_nodes(tb.netlist)
    assert 'legacy_level' in str(metric_error.value)
    assert name in str(metric_error.value)
    assert str(metric_error.value).split(': ', 1)[1] == str(probe_error.value).split(': ', 1)[1]


def test_declarative_noise_spectrum_is_not_a_node_capture(tmp_path):
    from silk import sources
    from silk.simulation import Noise
    tb = xyce_bench(tmp_path)
    tb.add_source(sources.VoltageSource('vin', positive='vout', negative='0', dc=1))
    tb.add_sim_type(Noise('vout', '0', 'vin', 'dec', 10, 1, 1e6))
    tb.add_metric('noise', type='integrated_noise', signal='onoise_spectrum')
    xyce_deck(tb)
    assert tb.netlist._pending_metric_nodes == ()


@pytest.mark.parametrize('path', ['noise_out', 'onoise_spectrum', 'onoise',
                                  'noise_in', 'inoise_spectrum', 'inoise'])
def test_declarative_xyce_noise_reads_csv_with_electrical_rawfile(tmp_path, path):
    from pathlib import Path
    from silk.circuit.module_test import _load_metric_path
    from silk.signals.rawfile import read_rawfile
    from silk.simulation.backend import RawResult
    fixtures = Path(__file__).parent / 'fixtures' / 'rawfiles'
    mod = Module(None, str(tmp_path), 'noise.cir', None)
    mod.simulator = 'xyce'
    mod._raw = RawResult(native=read_rawfile(str(fixtures / 'e_xy_tran.raw')),
                         source_path=str(fixtures / 'e_xy_noise' / 'rawfile.raw'))
    mod.sim_results_loaded = True
    expected = mod.load_signal(path, analysis='noise')
    actual = _load_metric_path(mod, path, analysis='noise')
    np.testing.assert_array_equal(actual.y, expected.y)
    assert len(actual.y) == 121
