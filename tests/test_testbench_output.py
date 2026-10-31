"""TASK-136: saved artifacts belong to the live test owner and selected label."""
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, load_pdk, sources
from silk.circuit.module_test import ModuleTestbench
from silk.elements import elements
from silk.simulation import Transient
from tests._env import needs_ngspice


def _load(root, *, name='load', tests=('level',), cache=False, db_path=None):
    mod = Module('sky130', str(root), 'dut.cir', load_pdk('sky130'),
                 module_name=name, db_path=db_path)
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    for name in tests:
        tb = ModuleTestbench(mod, name, cache=cache, parameters={'load_ohm': 1000})
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
        tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1))
        tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
        tb.add_metric('voltage', lambda ckt: float(ckt.load_signal('out').y[-1]))
        tb.add_spec('Voltage limit', 'voltage', '<', 1.5, key=f'{name}_voltage_limit')
    return mod


def _copies(root, *, tests=('level',), names=('a', 'b')):
    child = _load(root, tests=tests)
    parent = Module('sky130', str(root), 'parent.cir', child.pdk_details,
                    module_name='parent')
    for index, name in enumerate(names):
        child = _load(root, tests=tests)
        parent.add_module(child, name, nets={'out': f'out{index}', 'vss': '0'})
    return parent


def _test(parent, name, test='level'):
    return parent.elem_dict[name]['device']._tests[test]


@pytest.fixture
def fixture_runner(monkeypatch, mock_ngspice_version):
    """Keep emission, runners, loading and evaluation real; replace only the CLI."""
    fixtures = Path(__file__).parent / 'fixtures' / 'rawfiles'
    calls = []

    def run(argv, **kwargs):
        if '-b' in argv:
            raw = Path(argv[argv.index('-r') + 1])
            log = Path(argv[argv.index('-o') + 1])
            deck = Path(argv[-1])
            assert f'write {raw}' in deck.read_text()
            assert raw.parent == log.parent
            assert raw.parent.parent == deck.parent
            log.write_text('Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n')
            simulator = 'ng'
        else:
            deck = Path(argv[-1])
            import re
            match = re.search(r'format=raw file=([^\s]+)', deck.read_text())
            assert match is not None
            raw = Path(match.group(1).strip(chr(34)))
            assert Path(kwargs['cwd']) == deck.parent
            assert raw.parent.parent == deck.parent
            simulator = 'xy'
        write_complete_rawfile(raw, deck.read_text(), simulator='ngspice' if simulator == 'ng' else 'xyce', template=fixtures / f'e_{simulator}_tran.raw')
        calls.append(deck.parent / raw.name)
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return calls


@pytest.mark.parametrize('recursive', [False, True])
@needs_ngspice
def test_copied_instances_retain_their_own_real_waveforms(tmp_path, recursive):
    parent = _copies(tmp_path)
    parent.set_output_options(mode='overwrite')
    a, b = _test(parent, 'a'), _test(parent, 'b')
    a.module.set_output_options(mode='overwrite')
    b.module.set_output_options(mode='overwrite')
    assert a.module is parent.elem_dict['a']['device']
    b.netlist.sig_sources_dict['supply'].dc = 2
    if recursive:
        reports = parent.run_all_tests(recursive=True)
        assert set(reports) == {'$/a:load.level', '$/b:load.level'}
        ra, rb = reports['$/a:load.level'], reports['$/b:load.level']
    else:
        ra, rb = a.run(run_label='a'), b.run(run_label='b')
    assert ra.metrics['voltage'] == pytest.approx(1)
    assert rb.metrics['voltage'] == pytest.approx(2)
    assert ra.passed and not rb.passed
    # Fresh loaders expose the original bug even though both reports were right.
    for tb, report, expected in ((a, ra, 1), (b, rb, 2)):
        reader = Module('sky130', str(tmp_path / 'reader'), 'reader.cir', parent.pdk_details)
        reader.load_file(report.sim_results.results_file)
        reader.read_raw_file()
        np.testing.assert_allclose(reader.load_signal('out').y, expected)
        assert report.metrics['voltage'] == pytest.approx(expected)
        directory = Path(report.sim_results.results_file).parent
        assert Path(os.path.join(report.sim_results.dir_folder, report.sim_results.netlistname)).parent == directory
        assert f'write {report.sim_results.results_file}' in Path(os.path.join(report.sim_results.dir_folder, report.sim_results.netlistname)).read_text()
        assert (directory / 'output.log').stat().st_size > 0
    assert ra.sim_results.results_file != rb.sim_results.results_file
    paths = ra.sim_results.results_file, rb.sim_results.results_file
    if recursive:
        parent.run_all_tests()
    else:
        a.run(run_label='a')
        b.run(run_label='b')
    assert (ra.sim_results.results_file, rb.sim_results.results_file) == paths


def test_nested_owners_share_only_same_test_and_root_keeps_own_run(tmp_path, fixture_runner):
    branch = _copies(tmp_path, tests=('level', 'other'), names=('a',))
    branch.define_module_ports({})
    tree = _load(tmp_path, name='tree')
    tree.set_output_options(mode='overwrite')
    for name in ('left', 'right'):
        tree.add_module(branch, name)
    root_folder = Path(tree.get_test('level').prepare(run_label='baseline')).parent
    reports = tree.run_all_tests(run_label='baseline')
    assert set(reports) == {'tree.level'} | {
        f'$/{branch}/a:load.{test}'
        for branch in ('left', 'right') for test in ('level', 'other')
    }
    paths = {report.sim_results.results_file for report in reports.values()}
    assert len(paths) == 3
    assert len({id(report) for report in reports.values()}) == 5
    assert len(fixture_runner) == 3
    root_path = reports['tree.level'].sim_results.results_file
    assert Path(root_path).parent == root_folder
    for test in ('level', 'other'):
        left = reports[f'$/left/a:load.{test}'].sim_results.results_file
        right = reports[f'$/right/a:load.{test}'].sim_results.results_file
        assert left == right
        assert left != root_path
    assert all(Path(path).is_file() for path in paths)


def test_owner_escaping_does_not_flatten_paths(tmp_path, fixture_runner):
    parent = _copies(tmp_path, names=('a_b', 'a__b'))
    branch = _copies(tmp_path, names=('b',))
    parent.add_module(branch, 'a')
    for index, entry in enumerate(parent.list_test_entries()):
        entry.test.netlist.sig_sources_dict['supply'].dc = index + 1
    reports = parent.run_all_tests()
    paths = [report.sim_results.results_file for report in reports.values()]
    assert len(paths) == 3
    assert len(set(paths)) == 3
    assert all(Path(path).resolve().is_relative_to(tmp_path) for path in paths)


@pytest.mark.parametrize('name', ['a/b', 'a~1b', 'a~b', '../escape', '/absolute'])
def test_path_like_owner_names_are_not_constructible(tmp_path, name):
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='invalid instance name'):
        _copies(tmp_path, names=(name,))


@pytest.mark.parametrize('module_name,test_name', [
    ('a_b', 'c'), ('a', 'b_c'), ('a', 'b%5Fc'),
    ('a', 'b~run~c'), ('a', '../escape'), ('a', '/absolute'),
    ('a', r'..\escape'), ('a', 'a/b'), ('a', 'a%2Fb'), ('a', 'a b'),
    ('a', 'é'),
    # '../module' and '/module' used to live here as module_name cases —
    # TASK-292.1's module_name validator now refuses both at construction
    # (see test_path_like_module_names_are_not_constructible below), a
    # stronger guarantee than "the escape attempt is sanitized". test_name
    # is a different namespace (not module_name-validated), so its
    # path-traversal-looking cases above are unaffected and still covered.
])
def test_test_names_stay_safe_and_distinct(tmp_path, module_name, test_name):
    tb = _load(tmp_path, name=module_name, tests=(test_name,))._tests[test_name]
    deck = Path(os.path.join(tb.netlist.dir_folder, tb.netlist.netlistname))
    assert deck.parent.parent == tmp_path
    assert deck.resolve().is_relative_to(tmp_path.resolve())
    # This corpus also runs together below to check injectivity.


@pytest.mark.parametrize('module_name', ['../module', '/module'])
def test_path_like_module_names_are_not_constructible(tmp_path, module_name):
    """TASK-292.1: a module_name that looks like a path-traversal attempt
    can no longer be constructed at all, so it can never reach the output
    path-safety logic these used to exercise via test_test_names_stay_safe_and_distinct."""
    with pytest.raises(ValueError, match=re.escape(repr(module_name))):
        _load(tmp_path, name=module_name)


def test_module_test_and_label_delimiters_cannot_alias(tmp_path, fixture_runner):
    pairs = [('a_b', 'c'), ('a', 'b_c'), ('a', 'b%5Fc'),
             ('a', 'b~run~c'), ('a', 'b'), ('a', 'a/b'), ('a', 'a%2Fb')]
    benches = [_load(tmp_path, name=m, tests=(t,))._tests[t] for m, t in pairs]
    paths = []
    for tb in benches:
        tb.run()
        paths.append(tb.results_file)
    benches[4].run(run_label='c')
    paths.append(benches[4].results_file)
    assert len(set(paths)) == len(paths)


@pytest.mark.parametrize('label', ['', '.', '..', '../escape', '/tmp/escape',
                                  r'C:\escape', r'a\b', 'a/b', 'a b', 'a\n',
                                  'a~run~b', 'é', 'a' * 65, 42, Path('label')])
def test_invalid_labels_preserve_existing_artifacts(tmp_path, fixture_runner, label):
    tb = _load(tmp_path)._tests['level']
    tb.run()
    before = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    saved = tb.results_file
    with pytest.raises(ValueError, match='run_label'):
        tb.run(run_label=label)
    assert tb.results_file == saved
    assert {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()} == before
    assert len(fixture_runner) == 1


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_label_selection_reuse_reset_and_runner_paths(tmp_path, fixture_runner, simulator):
    mod = _load(tmp_path)
    mod.set_output_options(mode='overwrite')
    tb = mod._tests['level']
    tb.netlist.simulator = simulator
    mod.run_test('level')
    unlabeled = tb.results_file
    assert Path(unlabeled).parent == tmp_path / 'test_load~level'
    mod.run_test('level', run_label='baseline')
    baseline = tb.results_file
    reports = mod.run_all_tests(run_label='higher_bias')
    higher_bias = reports['load.level'].sim_results.results_file
    mod.run_test('level', run_label='baseline')
    assert tb.results_file == baseline
    tb.run()
    assert tb.results_file == unlabeled
    assert len({baseline, higher_bias, unlabeled}) == 3
    assert all(Path(p).is_file() for p in (baseline, higher_bias, unlabeled))
    assert fixture_runner == list(map(Path, [unlabeled, baseline, higher_bias, baseline, unlabeled]))


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_selected_directory_symlink_cannot_escape_root(tmp_path, fixture_runner, mode):
    tb = _load(tmp_path / 'output')._tests['level']
    tb.module.set_output_options(mode=mode)
    tb.run(run_label='baseline')
    selected = Path(tb.netlist.dir_folder)
    shutil.rmtree(selected)
    outside = tmp_path / 'outside'
    outside.mkdir()
    selected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='output root'):
        tb.run(run_label='baseline')
    assert list(outside.iterdir()) == []
    assert len(fixture_runner) == 1


def test_cache_true_warns_and_every_run_claims_a_fresh_label(tmp_path, fixture_runner):
    """Formerly test_cache_hit_does_not_claim_a_new_label_or_rewrite_context:
    verified a cache hit skipped simulation, reused the earlier label's
    directory/context, and raised on mismatched conditions. The report cache
    is disabled in 0.2.0 (TASK-284), so a cache hit can no longer occur: this
    now verifies cache=True warns once at construction and then behaves
    exactly like cache=False, simulating and claiming a fresh label every
    run()."""
    with pytest.warns(UserWarning, match='TASK-130'):
        tb = _load(tmp_path, cache=True)._tests['level']
        tb.module.set_output_options(mode='overwrite')
    tb.set_temperature(27)
    first = tb.run(run_label='baseline', temperature=27, voltage=1)
    saved = tb.results_file
    directory = tb.netlist.dir_folder
    tb.netlist.sig_sources_dict['supply'].dc = 2
    second = tb.run(run_label='higher_bias', voltage=2)
    assert second is not first
    assert tb.results_file != saved
    assert tb.netlist.dir_folder != directory
    with pytest.raises(ValueError, match='run_label'):
        tb.run(run_label='../escape')
    tb.clear_cache()  # remains callable and harmless
    tb.run(run_label='baseline')
    assert tb.results_file == saved
    assert len(fixture_runner) == 3


@needs_ngspice
def test_two_label_example_reads_saved_path_after_second_run(tmp_path):
    tb = _load(tmp_path)._tests['level']
    tb.module.set_output_options(mode='overwrite')
    baseline = tb.run(run_label='baseline')
    baseline_path = tb.results_file
    tb.netlist.sig_sources_dict['supply'].dc = 2
    higher_bias = tb.run(run_label='higher_bias')
    higher_bias_path = tb.results_file
    assert baseline.metrics['voltage'] == pytest.approx(1)
    assert higher_bias.metrics['voltage'] == pytest.approx(2)
    # An old report deliberately references the live Module, not a snapshot.
    assert baseline.sim_results is higher_bias.sim_results is tb.netlist
    assert baseline.sim_results.load_signal('out').y[-1] == pytest.approx(2)
    reread = tb.evaluate(rawfile=baseline_path)
    assert reread.metrics['voltage'] == pytest.approx(1)
    assert reread.passed
    assert tb.evaluate(rawfile=higher_bias_path).metrics['voltage'] == pytest.approx(2)
    tb.netlist.sig_sources_dict['supply'].dc = 3
    tb.run(run_label='baseline')
    assert tb.results_file == baseline_path
    assert tb.evaluate(rawfile=baseline_path).metrics['voltage'] == pytest.approx(3)
    assert tb.evaluate(rawfile=higher_bias_path).metrics['voltage'] == pytest.approx(2)


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_labels_preserve_structure_profile_specs_and_storage_policy(tmp_path, fixture_runner, mode):
    mod = _load(tmp_path, db_path=str(tmp_path / 'evidence.db'))
    tb = mod._tests['level']
    mod.set_output_options(mode=mode)
    tb.set_temperature(27)
    mod.create_circuit_version()
    tracker = mod._tracker
    structure = tracker.compute_structure_hash_for(mod)
    fingerprint = tb._fingerprint()
    spec = tb._specs[0]['spec']
    declaration = (spec.key, spec.target, tb.test_name, tb.owner_path, dict(tb.parameters))
    reports = []
    try:
        for label in ('baseline', 'higher_bias'):
            reports.append(mod.run_test('level', run_label=label,
                                        temperature=27, voltage=1, store=True))
        assert tracker.compute_structure_hash_for(mod) == structure
        assert tb._fingerprint() == fingerprint
        assert (spec.key, spec.target, tb.test_name, tb.owner_path, tb.parameters) == declaration
        assert mod.list_tests() == ['level']
        assert reports[0].characterization_context == reports[1].characterization_context
        assert all(report.storage['cached'] is False for report in reports)
        conn = tracker.db.conn
        profiles = conn.execute('SELECT profile_id, owner_path, test_name FROM characterization_profiles').fetchall()
        assert len(profiles) == 1
        assert (profiles[0]['owner_path'], profiles[0]['test_name']) == ('$', 'level')
        rows = mod.query_results(parameter='voltage', profile_id=profiles[0]['profile_id'])
        assert len(rows) == 2  # Same profile, appended observations.
        assert rows[0].value == rows[1].value
        measurements = conn.execute('SELECT profile_id FROM spec_measurements').fetchall()
        assert len(measurements) == 2
        assert {row['profile_id'] for row in measurements} == {profiles[0]['profile_id']}
        mod.run_test('level', run_label='unstored', temperature=27, voltage=1)
        assert len(mod.query_results(parameter='voltage', profile_id=profiles[0]['profile_id'])) == 2
    finally:
        mod.close()


@pytest.mark.parametrize('label', ['A-b.c_1', 'x' * 64])
def test_supported_labels_preserve_their_text(tmp_path, fixture_runner, label):
    tb = _load(tmp_path)._tests['level']
    tb.module.set_output_options(mode='overwrite')
    tb.run(run_label=label)
    assert label in Path(tb.results_file).parent.name
    assert Path(tb.results_file).is_file()


def test_unnamed_module_cannot_alias_literal_none(tmp_path):
    """The ``%N`` sentinel in ``_output_paths``, still live after TASK-167.

    A Module with no ``module_name`` can no longer be a DEVICE under test —
    it is a top-level circuit, so there is no ``.subckt`` for the testbench to
    instantiate and ``add_module_under_test`` now refuses. It can still OWN a
    test, which is how this case is still reached: the testbench directory is
    selected the same lazy way by ``add_source``, and must not alias a
    subcircuit literally named ``None``.

    ``'%N'`` was the third case here (TASK-98/167 era) — a module_name
    containing ``%`` used to reach ``encode(module_name)`` and rely on
    percent-encoding to avoid aliasing the ``%N`` sentinel. TASK-292.1's
    module_name validator (``^[A-Za-z_][A-Za-z0-9_]*$``) now refuses ``%``
    at construction, before that encoding step is ever reached — a stronger
    guarantee (the collision is unconstructible, not merely escaped), so
    that case moved to ``test_percent_n_is_not_a_constructible_module_name``
    below instead of being exercised through ``dir_folder`` comparison.
    """
    def bench(name):
        mod = Module('sky130', str(tmp_path), 'dut.cir', load_pdk('sky130'),
                     module_name=name)
        mod.define_module_ports({'out': 'out', 'vss': 'ground'})
        mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
        tb = ModuleTestbench(mod, 'level')
        tb.add_source(sources.VoltageSource('supply', positive='out',
                                            negative='0', dc=1))
        return tb

    names = (None, 'None')
    assert len({bench(name).netlist.dir_folder for name in names}) == len(names)


def test_percent_n_is_not_a_constructible_module_name(tmp_path):
    """TASK-292.1: a module_name of ``'%N'`` — the sentinel
    ``_output_paths`` reserves for "no module_name" — can no longer be
    constructed at all, so it can never alias that sentinel."""
    with pytest.raises(ValueError, match=r"'%N'"):
        Module('sky130', str(tmp_path), 'dut.cir', load_pdk('sky130'),
              module_name='%N')


@needs_ngspice
def test_relative_output_root_and_standalone_simulate_conventions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tb = _load(Path('output'))._tests['level']
    tb.run(run_label='baseline')
    assert Path(tb.results_file).resolve().is_relative_to(tmp_path / 'output')
    # Standalone Module.simulate still uses the caller's directory/deck name.
    standalone = Module('sky130', 'standalone', 'caller.cir', tb.pdk_details)
    standalone.set_output_options(mode='overwrite')
    standalone.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1))
    standalone.add(elements.resistor('load', p='out', n='0', value=1000))
    standalone.add_sim_type(Transient(step=1e-6, stop=2e-6))
    standalone.simulate()
    assert standalone.netlistname == 'caller.cir'
    assert standalone.results_file == 'standalone/rawfile.raw'
    assert Path('standalone/output.log').is_file()


@pytest.mark.parametrize('names', [('😀' * 50, '😀' * 49 + '😁'),
                                  ('a' * 220, 'a' * 219 + 'b')])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_long_escaped_test_names_and_labels_remain_runnable(tmp_path, fixture_runner, mode, names):
    mod = _load(tmp_path, tests=names)
    mod.set_output_options(mode=mode)
    paths = []
    for tb in mod._tests.values():
        tb.run(run_label='x' * 64)
        paths.append(tb.results_file)
        assert len(Path(tb.results_file).parent.name.encode()) <= 255
        assert len(tb.netlist.netlistname.encode()) <= 255
        assert Path(tb.netlist._run_files.netlist_path).is_file()
    assert len(set(paths)) == 2
    saved = paths[0]
    mod._tests[names[0]].run(run_label='x' * 64)
    assert (mod._tests[names[0]].results_file == saved) == (mode == 'overwrite')
    mod._tests[names[0]].run(run_label='y' * 64)
    assert mod._tests[names[0]].results_file != saved
    assert Path(saved).is_file()


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_long_nested_owner_paths_remain_isolated(tmp_path, fixture_runner, mode):
    # Each level is a valid instance name; the combined owner exceeds NAME_MAX.
    tree = _load(tmp_path)
    for depth in range(4):
        parent = Module('sky130', str(tmp_path), 'parent.cir', tree.pdk_details,
                        module_name=f'parent{depth}')
        parent.add_module(tree, f'level{depth}_' + 'x' * 64)
        tree = parent
    top = Module('sky130', str(tmp_path), 'top.cir', tree.pdk_details)
    for name, voltage in [('left', 1), ('right', 2)]:
        carrier = Module('sky130', str(tmp_path), name + '.cir', tree.pdk_details)
        carrier.set_supply_voltage(voltage)
        carrier.define_module_name('carrier')
        carrier.define_module_ports({})
        carrier.add_module(tree, 'tree')
        top.add_module(carrier, name)
    top.set_output_options(mode=mode)
    reports = top.run_all_tests()
    paths = [report.sim_results.results_file for report in reports.values()]
    assert len(set(paths)) == 2
    assert all(Path(path).is_file() for path in paths)
    assert all(len(Path(path).parent.name.encode()) <= 255 for path in paths)
