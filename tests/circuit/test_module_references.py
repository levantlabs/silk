"""Live module definitions with independent placement wiring (TASK-458)."""
from pathlib import Path

import pytest

from silk import Module, sources
from silk.elements import elements
from silk.exceptions import CircuitError
from silk.simulation import Transient
from silk.circuit.ports import PortDir
from tests._env import needs_ngspice


def block(tmp_path, name='load', *, variants=False):
    child = Module(None, str(tmp_path), name + '.cir', None, module_name=name,
                   variants=variants)
    child.define_module_ports({'p': 'in', 'n': 'ground'})
    child.add(elements.resistor('r1', p='p', n='n', value=1000))
    return child


def parent(tmp_path):
    return Module(None, str(tmp_path), 'top.cir', None)


def deck(top):
    top_deck_path = top.generate_netlist()
    return Path(top_deck_path).read_text()


def test_placed_module_tracks_new_elements_and_changed_values(tmp_path):
    child = block(tmp_path)
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    child.elem_dict['r1']['device'].value = 2000
    child.add(elements.resistor('r2', p='p', n='n', value=3000))
    text = deck(top)
    assert 'r1 p n 2k' in text.lower()
    assert 'r2 p n 3k' in text.lower()


def test_place_change_place_uses_one_final_body_and_independent_wiring(tmp_path):
    child = block(tmp_path)
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    child.add(elements.resistor('r2', p='p', n='n', value=3000))
    top.add_module(child, 'b', p='two', n='0')
    text = deck(top)
    assert text.count('.subckt load ') == 1
    assert 'r2 p n 3k' in text.lower()
    assert 'xa one 0 load\n' in text
    assert 'xb two 0 load\n' in text
    assert child.pin_list['p'].net is None
    assert top.elem_dict['a']['device'] is child
    assert top.elem_dict['b']['device'] is child


def test_collision_uses_current_bodies_and_names_both_instances(tmp_path):
    a, b = block(tmp_path), block(tmp_path)
    top = parent(tmp_path)
    top.add_module(a, 'left', p='one', n='0')
    top.add_module(b, 'right', p='two', n='0')
    b.add(elements.resistor('r2', p='p', n='n', value=3000))
    with pytest.raises(CircuitError, match='left.*right|right.*left'):
        deck(top)


def test_collision_can_be_repaired_before_emission(tmp_path):
    a, b = block(tmp_path), block(tmp_path)
    b.elem_dict['r1']['device'].value = 2000
    top = parent(tmp_path)
    top.add_module(a, 'left', p='one', n='0')
    top.add_module(b, 'right', p='two', n='0')
    a.elem_dict['r1']['device'].value = 2000
    assert deck(top).count('.subckt load ') == 1


def test_bench_uses_calibration_after_dut_placement(tmp_path):
    child = block(tmp_path)
    def model_1(ports, t, params):
        ports.p.v = t * 0 + params['level']
    child.set_behavioral_model(model_1, params={'level': 1.0}, execution='vector')
    child.pin_list['p'].direction = PortDir.OUT
    child.simulator = 'behavioral'
    tb = child.create_test('live')
    tb.add_module_under_test({'p': 'out', 'n': '0'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    out = tb.probe('out')
    child.calibrate_behavioral('level', 2.5)
    tb.run()
    assert list(out.y) == pytest.approx([2.5, 2.5, 2.5])


def test_public_instance_walk_and_test_entries_share_runner_identity(tmp_path):
    leaf = block(tmp_path)
    leaf.pin_list['p'].direction = PortDir.OUT
    def model_2(ports, t, params):
        ports.p.v = t * 0 + 1.0
    leaf.set_behavioral_model(model_2, execution='vector')
    tb = leaf.create_test('level')
    tb.add_module_under_test({'p': 'out', 'n': '0'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    middle = block(tmp_path, 'middle')
    middle.add_module(leaf, 'inside', p='p', n='n')
    top = parent(tmp_path)
    top.add_module(middle, 'left', p='one', n='0')
    top.add_module(middle, 'right', p='two', n='0')
    placements = tuple(top.iter_instances(include_self=True))
    assert [(p.path, p.spice_path, p.owner_path) for p in placements] == [
        ('', '', '$'), ('left', 'xleft', '$/left'),
        ('left.inside', 'xleft.xinside', '$/left/inside'),
        ('right', 'xright', '$/right'),
        ('right.inside', 'xright.xinside', '$/right/inside')]
    assert placements[2].module is placements[4].module is leaf
    for placement in placements:
        placement.module.simulator = 'behavioral'
    entries = top.list_test_entries()
    assert [(entry.path, entry.spice_path) for entry in entries] == [
        ('left.inside', 'xleft.xinside'), ('right.inside', 'xright.xinside')]
    assert [entry.report_key for entry in entries] == [
        '$/left/inside:load.level', '$/right/inside:load.level']
    reports = top.run_all_tests()
    assert list(reports) == [entry.report_key for entry in entries]
    assert all(not isinstance(report, dict) for report in reports.values())
    assert all(report.sim_results.simulator == 'behavioral' for report in reports.values())
    assert tb.owner_path == '$'


def test_shared_specs_keep_each_instance_owner_and_local_parent(tmp_path):
    from silk.db.identity import _fingerprint_spec_set
    child = block(tmp_path)
    ceiling = child.declare_spec('ceiling', 'level', '<=', 10.0)
    child.declare_spec('floor', 'level', '>=', 1.0, parent=ceiling)
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    top.add_module(child, 'b', p='two', n='0')
    _, definitions = _fingerprint_spec_set(top)
    assert [(d.owner_path, d.key, d.parent_owner_path) for d in definitions] == [
        ('$/a', 'ceiling', None), ('$/a', 'floor', '$/a'),
        ('$/b', 'ceiling', None), ('$/b', 'floor', '$/b')]


def test_variant_names_follow_post_placement_changes(tmp_path):
    import re
    child = block(tmp_path, variants=True)
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    before = deck(top)
    child.elem_dict['r1']['device'].value = 2000
    after = deck(top)
    before_name = re.search(r'^\.subckt (\S+)', before, re.M).group(1)
    after_name = re.search(r'^\.subckt (\S+)', after, re.M).group(1)
    assert before_name != after_name
    assert f'xa one 0 {after_name}\n' in after
    assert child.module_name == 'load'


def test_template_devices_still_have_independent_dimensions_and_pins(pdk, tmp_path):
    top = Module('sky130', str(tmp_path), 'devices.cir', pdk)
    with pytest.warns(DeprecationWarning):
        top.add_element(pdk.nmos1p8, 'a', nets={'d': 'one', 'g': 'g', 's': '0', 'b': '0'},
                        parameters={'w': 1.0, 'l': 0.15, 'm': 1})
    with pytest.warns(DeprecationWarning):
        top.add_element(pdk.nmos1p8, 'b', nets={'d': 'two', 'g': 'g', 's': '0', 'b': '0'},
                        parameters={'w': 2.0, 'l': 0.3, 'm': 1})
    a, b = (top.elem_dict[n]['device'] for n in ('a', 'b'))
    a.param_list['w'].value.value = 4.0
    assert b.param_list['w'].value.value == 2.0
    assert b.param_list['l'].value.value == 0.3
    assert a.pin_list['d'].net == 'one'
    assert b.pin_list['d'].net == 'two'


def test_late_port_role_inference_reaches_parent_bindings(tmp_path):
    from silk.circuit.port_roles import infer_port_roles, RoleRuleset
    from silk.circuit.ports import PortRole
    child = Module(None, str(tmp_path), 'roles.cir', None, module_name='roles')
    child.define_module_ports({'vdd': 'inout'})
    top = parent(tmp_path)
    top.add_module(child, 'a', vdd='rail')
    infer_port_roles(child, RoleRuleset.default())
    assert top.net_dict['rail'].classification == PortRole.POWER


def test_recursive_placement_is_refused_before_emission(tmp_path):
    a, b = block(tmp_path, 'a'), block(tmp_path, 'b')
    a.add_module(b, 'inside', p='p', n='n')
    with pytest.raises(CircuitError, match='[Rr]ecursive|[Cc]ycle|[Cc]ircular'):
        b.add_module(a, 'back', p='p', n='n')


def test_nested_casefolded_definitions_emit_once(tmp_path):
    left, right = block(tmp_path, 'left'), block(tmp_path, 'right')
    left.add_module(block(tmp_path, 'Bias'), 'bias', p='p', n='n')
    right.add_module(block(tmp_path, 'bias'), 'bias', p='p', n='n')
    top = parent(tmp_path)
    top.add_module(left, 'left', p='one', n='0')
    top.add_module(right, 'right', p='two', n='0')
    text = deck(top)
    assert text.lower().count('.subckt bias ') == 1
    assert text.count('xbias p n Bias\n') == 2
    names = [i.module_name for i in top.iter_instances() if i.path.endswith('.bias')]
    assert names == ['Bias', 'Bias']


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('variants', [False, True])
def test_repeated_bench_reports_retain_distinct_owner_paths_and_files(tmp_path, workers, variants):
    child = block(tmp_path, variants=variants)
    child.pin_list['p'].direction = PortDir.OUT
    def model_3(ports, t, params):
        ports.p.v = t * 0 + 1.0
    child.set_behavioral_model(model_3, execution='vector')
    child.simulator = 'behavioral'
    tb = child.create_test('level')
    tb.add_module_under_test({'p': 'out', 'n': '0'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    top = parent(tmp_path)
    top.add_module(child, 'left', p='one', n='0')
    top.add_module(child, 'right', p='two', n='0')
    reports = list(top.run_all_tests(n_workers=workers).values())
    assert [r.owner_path for r in reports] == ['$/left', '$/right']
    assert [r.module_name for r in reports] == [i.module_name for i in top.iter_instances()]
    paths = [r.sim_results.results_file for r in reports]
    assert paths[0] != paths[1]
    assert all(Path(path).is_file() for path in paths)


def test_public_walk_reaches_legacy_registered_test_owners(tmp_path):
    child = block(tmp_path)
    top = parent(tmp_path)
    with pytest.warns(DeprecationWarning):
        top.add_submodule('legacy', child)
    child.create_test('level')
    instances = tuple(top.iter_instances())
    assert len(instances) == 1
    assert instances[0].module is child
    assert instances[0].placed is False
    assert instances[0].owner_path == '$/legacy'
    assert top.list_test_entries()[0].module is instances[0].module


def test_named_root_and_child_share_collision_namespace(tmp_path):
    root = block(tmp_path)
    root.add_module(block(tmp_path), 'inside', p='p', n='n')
    with pytest.raises(CircuitError, match='structurally different'):
        root.generate_netlist()


def test_port_added_after_placement_can_be_connected(tmp_path):
    child = block(tmp_path)
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    child.define_module_ports({'extra': 'in'})
    top.connect_module('a', extra='extra_net')
    assert 'extra_net' in deck(top)


@needs_ngspice
@pytest.mark.parametrize('workers', [1, 2])
def test_shared_spice_bench_preserves_reports_and_callable_probes(tmp_path, workers):
    from silk.metrics import average
    child = block(tmp_path)
    tb = child.create_test('level')
    tb.add_module_under_test({'p': 'out', 'n': '0'})
    tb.add_source(sources.VoltageSource('bias', positive='out', negative='0', dc=2))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    out = tb.probe('out')
    tb.add_metric('level', lambda netlist: average(out))
    top = parent(tmp_path)
    top.add_module(child, 'left', p='one', n='0')
    top.add_module(child, 'right', p='two', n='0')
    reports = list(top.run_all_tests(n_workers=workers).values())
    assert [r.owner_path for r in reports] == ['$/left', '$/right']
    assert [r.metrics['level'] for r in reports] == pytest.approx([2., 2.])
    assert reports[0].sim_results is not reports[1].sim_results
    assert reports[0].rawfile_path == reports[1].rawfile_path
    assert all(Path(r.rawfile_path).is_file() for r in reports)


def test_standalone_variant_root_validates_its_emitted_name(tmp_path):
    root = block(tmp_path, variants=True)
    root.add_module(block(tmp_path), 'inside', p='p', n='n')
    with pytest.raises(CircuitError, match='structurally different'):
        root.generate_netlist(subckt=True)


def test_module_pickle_keeps_tests_but_worker_payload_omits_registries(tmp_path):
    import pickle
    from silk.simulation._worker_pickle import dump_worker_netlist
    child = block(tmp_path)
    child.create_test('level')
    restored = pickle.loads(pickle.dumps(child))
    assert restored.list_tests() == ['level']
    assert restored.list_test_entries()[0].test.module is restored
    child._tests['level'].add_metric('local', lambda netlist: 1)
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    worker = pickle.loads(dump_worker_netlist(top))
    assert next(worker.iter_instances()).module.list_tests() == []
    assert child.list_tests() == ['level']
    assert 'r1 p n 1k' in deck(worker).lower()


def test_tracker_keeps_indexed_instance_names_and_local_wiring(tmp_path):
    from silk.db.structure_tracker import CircuitStructureTracker
    middle = block(tmp_path, 'middle')
    middle.add_module(block(tmp_path), 'indexed[3]', p='p', n='n')
    top = parent(tmp_path)
    top.add_module(middle, 'a', p='one', n='0')
    tracker = CircuitStructureTracker(None, None)
    hierarchy = tracker.extract_structure(top)['hierarchy']
    child = next(entry for entry in hierarchy if entry['instance_name'] == 'a.indexed[3]')
    assert child['ports'] == {'p': 'p', 'n': 'n'}


def test_public_paths_preserve_instance_names_starting_with_x(tmp_path):
    leaf = block(tmp_path)
    leaf.create_test('leaf')
    top = parent(tmp_path)
    top.create_test('root')
    top.add_module(leaf, 'xunit', p='one', n='0')
    assert [(i.path, i.spice_path) for i in top.iter_instances()] == [
        ('xunit', 'xxunit')]
    assert [(e.path, e.spice_path) for e in top.list_test_entries()] == [
        ('', ''), ('xunit', 'xxunit')]
    assert [(e.path, e.spice_path) for e in top.list_test_entries(recursive=False)] == [
        ('', '')]


def placement_fixture(tmp_path):
    child = block(tmp_path)
    child.pin_list['p'].direction = PortDir.OUT
    def model_4(ports, t, params):
        ports.p.v = t * 0 + 1.0
    child.set_behavioral_model(model_4, execution='vector')
    child.simulator = 'behavioral'
    tb = child.create_test('level')
    tb.add_module_under_test({'p': 'out', 'n': '0'})
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    signal = tb.probe('out')
    tb.add_metric('level', lambda netlist: float(signal.y[-1]))
    top = parent(tmp_path)
    top.add_module(child, 'a', p='one', n='0')
    top.add_module(child, 'b', p='two', n='0')
    return top, tb, signal


@pytest.mark.parametrize('workers', [1, 2])
def test_selected_handles_match_runner_results(tmp_path, workers):
    top, tb, signal = placement_fixture(tmp_path)
    entries = top.list_test_entries()
    manual = [entry.test.run() for entry in entries]
    results = top.run_all_tests(n_workers=workers)
    assert list(results) == [entry.report_key for entry in entries]
    for entry, previous in zip(entries, manual):
        report = results[entry.report_key]
        assert report.owner_path == entry.owner_path
        assert report.metrics == previous.metrics
        assert entry.test.probe('out').y[-1] == 1.0
    assert entries[0].test.netlist is entries[1].test.netlist is tb.netlist


def test_selected_external_evaluation_preserves_each_owner(tmp_path):
    from tests.test_simulation_freshness import _raw
    top, tb, signal = placement_fixture(tmp_path)
    tb.netlist.simulator = 'ngspice'
    tb.add_metric('double', lambda netlist: 2 * signal.y[-1])
    entries = top.list_test_entries()
    for entry, value in zip(entries, (1.25, 2.5)):
        raw = tmp_path / f'{entry.path}.raw'
        raw.write_text(_raw(value))
        report = entry.test.evaluate(rawfile=str(raw))
        assert report.metrics == {'level': value, 'double': 2 * value}
        assert report.owner_path == entry.owner_path
        assert entry.test.results_file == str(raw)
    for entry, value in zip(entries, (1.25, 2.5)):
        assert entry.test.probe('out').y[-1] == value


def test_selected_entries_carry_current_inherited_conditions(tmp_path):
    top, tb, _ = placement_fixture(tmp_path)
    top.set_temperature(45)
    top.set_supply_voltage(1.8)
    top.set_output_options(mode='new')
    for entry in top.list_test_entries():
        assert entry.test._inherited_temperature() == entry.inherited_temperature
        assert entry.test._inherited_supply() == entry.inherited_supply
        assert entry.output_mode == 'new'
    assert tb._inherited_temperature() is None
    assert tb._inherited_supply() is None


def test_dotted_instance_is_refused_and_nested_selection_remains(tmp_path):
    top, tb, _ = placement_fixture(tmp_path)
    middle = block(tmp_path, 'middle')
    middle.add_module(tb.module, 'b', p='p', n='n')
    top.add_module(middle, 'nested', p='three', n='0')
    with pytest.raises(CircuitError, match='hierarchy levels'):
        top.add_module(tb.module, 'nested.b', p='four', n='0')
    entries = [e for e in top.list_test_entries() if e.path == 'nested.b']
    assert len(entries) == 1
    assert top.get_test('level', instance='nested.b') is entries[0].test


def test_removing_a_wired_definition_port_refuses_without_discarding_binding(tmp_path):
    child = block(tmp_path)
    top = parent(tmp_path)
    top.add_module(child, 'left', p='vin', n='0')
    pins = top._element_mgr.elem_dict['left']['pin_list']
    del child.pin_list['p']
    with pytest.raises(CircuitError, match="left.*p"):
        top.generate_netlist()
    assert pins['p'].net == 'vin'
    assert top._element_mgr._pin_to_net[('left', 'p')] == 'vin'
