"""Circuit output policy across launch routes and non-launch operations.

Route matrix: standalone/manual below; embedded/direct/all/corners/optimizer
below; store=True in test_testbench_output; fault workers in faults/test_runner.
Both real CLIs retain their prepared-path tests in test_run_files.
"""

import copy
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from silk import Variable
from silk.exceptions import SimulationError
from silk.optimize import Optimizer, VariableSpace
from silk.simulation.run_files import RunFiles
from tests._env import needs_ngspice
from tests.test_run_files import _module, fixture_cli  # noqa: F401
from tests.test_testbench_output import _load, _copies, _test, fixture_runner  # noqa: F401


def _artifacts(root):
    return {str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
            for p in root.rglob('*')}


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_generated_path_keeps_filename_configuration(tmp_path, fixture_cli, simulator, mode):
    mod = _module(tmp_path, simulator)
    mod.set_output_options(mode=mode)
    assert mod.netlistname == 'deck.cir'
    first = mod.generate_netlist()
    assert isinstance(first, str)
    assert Path(first).is_file()
    assert mod.netlistname == 'deck.cir'
    assert mod.dir_folder == str(tmp_path)
    mod.run_simulator()
    assert Path(mod.results_file).parent == Path(first).parent
    second = mod.generate_netlist()
    if mode == 'new':
        assert first != second
        assert Path(first).parent.parent == Path(second).parent.parent == tmp_path
    else:
        assert first == second == str(tmp_path / 'deck.cir')
    mod.dir_folder = str(tmp_path / 'other')
    mod.netlistname = 'renamed.cir'
    third = mod.generate_netlist()
    assert Path(third).is_file()
    assert Path(third).name == 'renamed.cir'
    assert Path(third).is_relative_to(tmp_path / 'other')
    assert mod.netlistname == 'renamed.cir'


@pytest.mark.parametrize('assign_later', [False, True])
@pytest.mark.parametrize('name', ['nested/deck.cir', '/absolute/deck.cir'])
def test_filename_path_refused_before_writing(tmp_path, assign_later, name):
    from silk import Module, load_pdk
    from silk.exceptions import ConfigurationError

    mod = Module('sky130', str(tmp_path), 'deck.cir' if assign_later else name,
                 load_pdk('sky130'))
    if assign_later:
        mod.netlistname = name
    with pytest.raises(ConfigurationError, match='dir_folder'):
        mod.generate_netlist()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mode', [None, '', 'NEW', 'fresh', 1, [], {}])
def test_invalid_options_leave_policy_and_evidence_untouched(tmp_path, fixture_cli, mode):
    mod = _module(tmp_path)
    mod.set_output_options(mode='new')
    mod.simulate()
    files, source, raw = mod._run_files, mod.results_file, mod._raw
    before = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="mode must be 'overwrite' or 'new'"):
        mod.set_output_options(mode=mode)
    assert mod._output_mode == 'new'
    assert (mod._run_files, mod.results_file, mod._raw) == (files, source, raw)
    assert _artifacts(tmp_path) == before
    mod.set_output_options()
    assert mod._output_mode == 'new'
    assert _artifacts(tmp_path) == before


def test_exclusive_allocation_retries_collision_and_preserves_existing(tmp_path, monkeypatch):
    collision = tmp_path / 'run-collision'
    collision.mkdir()
    (collision / 'keep').write_text('saved')
    candidates = iter(['collision', 'unique', 'unique', 'another'])
    monkeypatch.setattr(tempfile, '_get_candidate_names', lambda: candidates)
    layout = RunFiles.resolve(str(tmp_path / 'deck.cir'), str(tmp_path), log=True)
    first, second = layout.prepare_new(), layout.prepare_new()
    assert Path(first.directory).name == 'run-unique'
    assert Path(second.directory).name == 'run-another'
    assert (collision / 'keep').read_text() == 'saved'
    assert first.directory != second.directory
    assert all(Path(p.directory).parent == tmp_path for p in (first, second))


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_standalone_mode_transitions_keep_stable_root(tmp_path, fixture_cli, simulator):
    mod = _module(tmp_path, simulator)
    configured = mod.dir_folder, mod.netlistname
    paths = []
    for mode in ('overwrite', 'new', 'new', 'overwrite'):
        mod.set_output_options(mode=mode)
        mod.simulate()
        paths.append(Path(mod.results_file))
        assert (mod.dir_folder, mod.netlistname) == configured
        assert mod._raw.run_files is mod._run_files
        assert Path(mod._run_files.netlist_path).parent == paths[-1].parent
    assert paths[0] == paths[3] == tmp_path / 'rawfile.raw'
    assert paths[1].parent.parent == paths[2].parent.parent == tmp_path
    assert paths[1] != paths[2]
    assert all(path.is_file() for path in paths)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_manual_new_run_requires_an_unconsumed_preparation(tmp_path, fixture_cli, simulator):
    mod = _module(tmp_path, simulator)
    mod.set_output_options(mode='new')
    with pytest.raises(SimulationError, match='generate_netlist'):
        mod.run_simulator()
    assert list(tmp_path.iterdir()) == []
    mod.generate_netlist()
    files = mod._run_files
    assert len(list(tmp_path.iterdir())) == 1
    mod.run_simulator()
    before = _artifacts(tmp_path)
    with pytest.raises(SimulationError, match='generate_netlist'):
        mod.run_simulator()
    assert _artifacts(tmp_path) == before
    assert mod.results_file == files.rawfile_path
    mod.generate_netlist()
    mod.run_simulator()
    assert mod.results_file != files.rawfile_path
    assert Path(files.rawfile_path).is_file()
    assert len(fixture_cli) == 2


@pytest.mark.parametrize('change', ['directory', 'deck', 'backend', 'mode', 'files', 'missing_deck'])
def test_mismatched_new_preparation_is_rejected_before_mutation(tmp_path, fixture_cli, change):
    mod = _module(tmp_path)
    mod.set_output_options(mode='new')
    mod.generate_netlist()
    if change == 'directory':
        mod.dir_folder = str(tmp_path / 'other')
    elif change == 'deck':
        mod.netlistname = 'other.cir'
    elif change == 'backend':
        mod.simulator = 'xyce'
    elif change == 'mode':
        mod.set_output_options(mode='overwrite')
    elif change == 'files':
        mod._run_files = None
    else:
        Path(mod._run_files.netlist_path).unlink()
    before = _artifacts(tmp_path)
    with pytest.raises(SimulationError, match='generate_netlist'):
        mod.run_simulator()
    assert _artifacts(tmp_path) == before
    assert fixture_cli == []


@pytest.mark.parametrize('copy_module', [copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))])
def test_new_preparation_roundtrip_and_subsequent_sibling(tmp_path, fixture_cli, copy_module):
    mod = _module(tmp_path)
    mod.set_output_options(mode='new')
    mod.generate_netlist()
    copied = copy_module(mod)
    copied.run_simulator()
    first = copied.results_file
    before = _artifacts(tmp_path)
    # The original still holds an unconsumed in-memory snapshot of these paths.
    with pytest.raises(SimulationError, match='generate_netlist'):
        mod.run_simulator()
    assert _artifacts(tmp_path) == before
    copied = copy_module(copied)
    with pytest.raises(SimulationError, match='generate_netlist'):
        copied.run_simulator()
    copied.simulate()
    assert copied.results_file != first
    assert Path(copied.results_file).parent.parent == tmp_path
    assert Path(first).is_file()


@pytest.mark.parametrize('route', ['direct', 'run_test', 'all_nonrecursive', 'all_recursive', 'corners', 'optimizer'])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_embedded_launch_route_matrix(tmp_path, fixture_runner, route, mode):
    mod = _load(tmp_path)
    tb = mod._tests['level']
    mod.set_output_options(mode=mode)  # After construction.
    if route == 'corners':
        reports = tb.run_corners(['tt', 'ff', 'ss'])
        assert set(reports) == {'tt', 'ff', 'ss'}
    elif route == 'optimizer':
        opt = Optimizer(tb, VariableSpace([Variable('knob', value=1, bounds=(1, 2))]))
        assert opt.evaluate({'knob': 1})['status'] == 'success'
        opt.sweep(n=2)
        assert len(opt.successes) == 3
    else:
        for _ in range(3):
            if route == 'direct':
                tb.run()
            elif route == 'run_test':
                mod.run_test('level')
            else:
                reports = mod.run_all_tests(recursive=route == 'all_recursive')
                assert set(reports) == {'load.level'}
                assert not isinstance(reports['load.level'], dict)
    assert len(fixture_runner) == 3
    if route == 'optimizer':
        # TASK-301.5 (BREAKING): every Optimizer launch now gets its own
        # base output dir -- evaluate() still uses the unlabeled path, but
        # each sweep() point gets run_label='point_<id>' (id = history
        # length at that point -- 1 here, since the evaluate() call above
        # already added one history record before sweep() started), so the
        # three launches land under three DISTINCT bases regardless of
        # `mode`, not one shared base with new-mode children.
        assert len(set(fixture_runner)) == 3
        expected_bases = {
            tmp_path / 'test_load~level',
            tmp_path / 'test_load~level~run~point_1',
            tmp_path / 'test_load~level~run~point_2',
        }
        actual_bases = {(p.parent.parent if mode == 'new' else p.parent)
                        for p in fixture_runner}
        assert actual_bases == expected_bases
        assert all(p.is_file() for p in fixture_runner)
        return
    assert len(set(fixture_runner)) == (3 if mode == 'new' or route == 'corners' else 1)
    base = tmp_path / 'test_load~level'
    # run_corners() (TASK-301.3) always nests one extra <corner>/ level under
    # the testbench directory (base), on top of whatever level 'new' mode adds.
    if route == 'corners':
        depth = 3 if mode == 'new' else 2
    else:
        depth = 2 if mode == 'new' else 1
    def ancestor(path, levels):
        for _ in range(levels):
            path = path.parent
        return path
    assert all(ancestor(p, depth) == base for p in fixture_runner)
    assert all(p.is_file() for p in fixture_runner)


@pytest.mark.parametrize('root_mode, child_mode', [('new', 'overwrite'), ('overwrite', 'new')])
def test_recursive_policy_is_scoped_to_root_for_instances_and_legacy(
        tmp_path, fixture_runner, root_mode, child_mode):
    parent = _copies(tmp_path)
    legacy = _load(tmp_path, name='legacy')
    parent.add_submodule('legacy', legacy)
    parent.set_output_options(mode=root_mode)
    benches = [_test(parent, 'a'), _test(parent, 'b'), legacy._tests['level']]
    for tb in benches:
        tb.module.set_output_options(mode=child_mode)
    unrelated = _load(tmp_path / 'unrelated')
    for _ in range(2):
        reports = parent.run_all_tests()
        assert set(reports) == {'$/a:load.level', '$/b:load.level', 'legacy.level'}
        assert all(not isinstance(r, dict) for r in reports.values())
    # The two identical load placements share one launch; legacy is distinct.
    assert len(set(fixture_runner)) == (4 if root_mode == 'new' else 2)
    for tb in benches:
        assert tb.module._output_mode == child_mode
        assert not hasattr(tb, '_output_mode_override')
        saved = []
        for _ in range(2):
            tb.run()
            saved.append(tb.results_file)
        assert (saved[0] != saved[1]) == (child_mode == 'new')
    assert unrelated._output_mode == 'new'
    assert unrelated._tests['level'].results_file is None


def test_recursive_override_is_restored_on_failure(tmp_path, fixture_runner, monkeypatch):
    parent = _copies(tmp_path)
    parent.set_output_options(mode='new')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    reports = parent.run_all_tests()
    assert all('error' in r for r in reports.values())
    for name in ('a', 'b'):
        tb = _test(parent, name)
        assert not hasattr(tb, '_output_mode_override')
        assert tb.module._output_mode == 'new'
        assert tb.results_file is None


def test_new_labels_are_siblings_and_cache_true_does_not_change_that(tmp_path, fixture_runner):
    """Formerly test_new_labels_are_siblings_and_cache_hits_do_not_allocate_or_relabel:
    also verified a cache hit reused the earlier label's report/path and
    raised on mismatched conditions. The report cache is disabled in 0.2.0
    (TASK-284), so a cache hit can no longer occur -- this keeps the
    new-labels-are-siblings coverage and drops the now-impossible cache-hit
    assertions."""
    with pytest.warns(UserWarning, match='TASK-130'):
        mod = _load(tmp_path, cache=True)
    tb = mod._tests['level']
    mod.set_output_options(mode='overwrite')
    first = tb.run(run_label='baseline', voltage=1)
    mod.set_output_options(mode='new')
    source = tb.results_file
    tb.set_supply_voltage(1)
    paths = []
    for label in ('baseline', 'baseline', 'next', None):
        tb.clear_cache()  # remains callable and harmless
        report = tb.run(run_label=label)
        assert report is not first
        paths.append(Path(tb.results_file))
    assert len(set(paths)) == 4
    assert paths[0].parent.parent == paths[1].parent.parent == Path(source).parent
    assert paths[2].parent.parent.name.endswith('~run~next')
    assert paths[3].parent.parent.name == 'test_load~level'
    mod.set_output_options(mode='overwrite')
    tb.clear_cache()
    tb.run(run_label='baseline')
    assert tb.results_file == source
    assert all(p.is_file() for p in paths)


@needs_ngspice
def test_real_new_readback_overwrite_replacement_and_failure_after_success(tmp_path):
    mod = _load(tmp_path)
    tb = mod._tests['level']
    mod.set_output_options(mode='new')
    paths, reports = [], []
    for voltage in (1, 2):
        tb.netlist.sig_sources_dict['supply'].dc = voltage
        reports.append(tb.run(voltage=voltage))
        paths.append(tb.results_file)
    assert paths[0] != paths[1]
    for path, voltage in zip(paths, (1, 2)):
        reader = _module(tmp_path / 'reader')
        reader.load_file(path)
        reader.read_raw_file()
        np.testing.assert_allclose(reader.load_signal('out').y, voltage)
    assert reports[0].metrics['voltage'] == pytest.approx(1)
    assert reports[1].metrics['voltage'] == pytest.approx(2)
    assert reports[0].sim_results is reports[1].sim_results is tb.netlist
    for mode in ('new', 'overwrite'):
        mod.set_output_options(mode=mode)
        for voltage in (3, 4):
            tb.netlist.sig_sources_dict['supply'].dc = voltage
            tb.run(voltage=voltage)
            if voltage == 3:
                prior = tb.results_file
            else:
                assert (prior == tb.results_file) == (mode == 'overwrite')
                if mode == 'overwrite':
                    assert tb.evaluate(rawfile=prior).metrics['voltage'] == pytest.approx(4)
        successful = tb.results_file
        before = Path(successful).read_bytes()
        tb.netlist.add_custom_line('invalid_device out 0 invalid_model')
        with pytest.raises(SimulationError):
            tb.run()
        assert tb.results_file is None
        assert tb.netlist.characterization_context is None
        with pytest.raises(SimulationError):
            tb.netlist.read_raw_file()
        if mode == 'new':
            assert Path(successful).read_bytes() == before
        else:
            assert not Path(successful).exists()
        tb.netlist.custom_lines.pop()
    assert all(Path(p).is_file() for p in paths)


def test_no_launch_operations_do_not_prepare_or_delete_files(tmp_path, fixture_runner, monkeypatch):
    mod = _load(tmp_path)
    tb = mod._tests['level']
    tb.run()
    source = tb.results_file
    held = tb.netlist._raw  # Keep A leased while checking read-only operations.
    mod.set_output_options(mode='new')
    tb.netlist.set_output_options(mode='new')
    before = _artifacts(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError('non-launch operation touched the run lifecycle')

    monkeypatch.setattr(RunFiles, 'prepare', forbidden)
    monkeypatch.setattr(RunFiles, 'prepare_new', forbidden)
    monkeypatch.setattr(RunFiles, 'clear_previous_outputs', forbidden)
    assert tb.evaluate(rawfile=source).metrics['voltage'] > 0
    assert tb.results_file == source
    assert '.subckt load' in mod.generate_netlist()
    tb._output_paths('read-only')
    RunFiles.resolve('deck.cir', str(tmp_path / 'absent'))
    assert _artifacts(tmp_path) == before
    assert held.native.npoints > 0


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('failure', ['missing', 'timeout', 'nonzero', 'launch', 'malformed', 'empty', 'constants'])
def test_new_mode_failure_keeps_prior_files_and_invalidates_live_results(
        tmp_path, monkeypatch, simulator, failure):
    from tests.test_simulation_freshness import _success, _failure, _assert_invalid

    mod = _module(tmp_path, simulator)
    mod.set_output_options(mode='new')
    mod.generate_netlist()
    files = mod._run_files
    monkeypatch.setattr(subprocess, 'run', _success(files))
    mod.run_simulator()
    held = mod._raw  # Retained generations keep their private artifacts.
    before = _artifacts(Path(files.directory))
    mod.generate_netlist()
    failed_files = mod._run_files
    monkeypatch.setattr(subprocess, 'run', _failure(failed_files, failure))
    with pytest.raises(SimulationError):
        mod.run_simulator()
    _assert_invalid(mod)
    assert _artifacts(Path(files.directory)) == before
    assert held.native.npoints > 0
    assert Path(failed_files.directory).is_dir()
    with pytest.raises(SimulationError, match='generate_netlist'):
        mod.run_simulator()
    mod.generate_netlist()
    monkeypatch.setattr(subprocess, 'run', _success(mod._run_files, value=0.9))
    mod.run_simulator()
    np.testing.assert_allclose(mod.load_signal('out').y, 0.9)
    assert len(list(tmp_path.glob('run-*'))) == 3


def test_unique_directory_survives_allocating_python_process_exit(tmp_path):
    subprocess.run([
        sys.executable, '-c',
        'from pathlib import Path; import sys; '
        'from silk.simulation.run_files import RunFiles; '
        'f = RunFiles.resolve("deck.cir", sys.argv[1]).prepare_new(); '
        'Path(f.netlist_path).write_text("persistent deck")', str(tmp_path),
    ], check=True)
    [deck] = list(tmp_path.glob('run-*/deck.cir'))
    assert deck.read_text() == 'persistent deck'


def test_failed_regeneration_cannot_bypass_new_guard_by_switching_mode(
        tmp_path, fixture_cli, monkeypatch):
    mod = _module(tmp_path)
    mod.set_output_options(mode='new')
    mod.simulate()
    files = mod._run_files
    saved = _artifacts(Path(files.directory))

    def broken(*args, **kwargs):
        raise ValueError('failed deck generation')

    monkeypatch.setattr(mod.netlister, 'generate_complete_netlist', broken)
    with pytest.raises(ValueError, match='failed deck generation'):
        mod.generate_netlist()
    mod.set_output_options(mode='overwrite')
    with pytest.raises(SimulationError, match='generate_netlist'):
        mod.run_simulator()
    assert _artifacts(Path(files.directory)) == saved
    assert mod.results_file == files.rawfile_path
    assert len(fixture_cli) == 1


def test_default_runs_preserve_previous_artifacts(tmp_path, fixture_cli):
    mod = _module(tmp_path)
    assert mod._output_mode == 'new'
    mod.simulate()
    first = Path(mod.results_file).parent
    before = {p: p.read_bytes() for p in first.iterdir()
              if p.is_file() and not p.name.startswith('.')}
    mod.simulate()
    second = Path(mod.results_file).parent
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert all(p.read_bytes() == data for p, data in before.items())
    mod.set_output_options(mode='overwrite')
    mod.set_output_options()
    assert mod._output_mode == 'new'
