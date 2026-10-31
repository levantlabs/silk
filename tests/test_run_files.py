"""Prepared decks share concrete paths through emission, launch and readback."""

import copy
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import pickle
import re
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from tests._mock_rawfile import write_complete_rawfile, complete_noise_csv

from silk import Module, load_pdk, sources
from silk.circuit.spice_netlister import PySpiceNetlister
from silk.elements import elements
from silk.exceptions import SimulationAborted
from silk.simulation import Noise, Transient
from silk.simulation.backend import (
    NgspiceBackend, RawResult, SimulatorBackend, SpiceBackend, XyceBackend,
)
from silk.simulation.run_files import RunFiles
from silk.simulation.runner import NgspiceRunner
from silk.simulation.xyce_runner import XyceRunner, _xyce_executable
from tests._env import needs_ngspice


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_generated_path_preserves_relative_root(tmp_path, monkeypatch, fixture_cli, simulator, mode):
    monkeypatch.chdir(tmp_path)
    mod = _module('relative', simulator)
    mod.set_output_options(mode=mode)
    path = mod.generate_netlist()
    assert isinstance(path, str)
    assert not os.path.isabs(path)
    assert Path(path).is_file()
    assert path.startswith('relative/')
    assert (mod.dir_folder, mod.netlistname) == ('relative', 'deck.cir')
    mod.run_simulator()
    assert Path(mod.results_file).is_file()
    assert Path(mod.results_file).resolve().parent == Path(path).resolve().parent


@pytest.mark.parametrize('absolute_allocation', [False, True])
def test_new_allocation_keeps_root_spelling_across_python_versions(
        tmp_path, monkeypatch, absolute_allocation):
    import tempfile

    monkeypatch.chdir(tmp_path)
    allocate = tempfile.mkdtemp

    def allocated_path(**kwargs):
        path = allocate(**kwargs)
        return os.path.abspath(path) if absolute_allocation else os.path.relpath(path)

    monkeypatch.setattr(tempfile, 'mkdtemp', allocated_path)
    files = RunFiles.resolve('relative/deck.cir', 'relative', log=True).prepare_new()
    assert not os.path.isabs(files.directory)
    assert Path(files.directory).parent == Path('relative')
    assert Path(files.directory).is_dir()
    assert files.netlist_path == files.directory + '/deck.cir'
    assert files.logfile_path == files.directory + '/output.log'


@pytest.mark.parametrize('mode', [None, 'overwrite', 'new'])
def test_snapshot_keeps_configured_deck_path(tmp_path, mode):
    mod = Module(None, str(tmp_path), 'snapshot.cir', None, db_path=':memory:')
    mod.add(elements.resistor('load', p='out', n='0', value=1000))
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='0', dc=1))
    mod.add_sim_type(Transient(1e-7, 1e-6))
    try:
        if mode is not None:
            mod.set_output_options(mode=mode)
            generated = mod.generate_netlist()
            assert Path(generated).is_file()
        version = mod.create_circuit_version()
        info = mod._tracker.db.get_circuit_by_id(version)
        assert info['netlist_path'] == str(tmp_path / 'snapshot.cir')
        assert mod.netlistname == 'snapshot.cir'
    finally:
        mod.close()


_RAW_TEXT = (
    'Title: path test\nDate: d\nPlotname: Transient Analysis\nFlags: real\n'
    'No. Variables: 2\nNo. Points: 2\n'
    'Variables:\n\t0\ttime\ttime\n\t1\tv(out)\tvoltage\n'
    'Values:\n0\t0.0\n\t1.0\n1\t1e-6\n\t1.8\n'
)
_NOISE_TEXT = '"FREQ","ONOISE","INOISE"\n1,4e-10,1e-12\n10,1e-10,2.5e-13\n'


def _module(root, simulator='ngspice', analyses=None):
    mod = Module('sky130', str(root), 'deck.cir', load_pdk('sky130'))
    mod.add_source(sources.VoltageSource('vin', positive='out', negative='0',
                                         dc=1, ac=1))
    mod.add(elements.resistor('rload', p='out', n='0', value=1000))
    for analysis in analyses if analyses is not None else [Transient(1e-7, 1e-6)]:
        mod.add_sim_type(analysis)
    mod.simulator = simulator
    return mod


def _rename_outputs(monkeypatch, backend_cls):
    resolve = backend_cls._resolve_run_files

    def renamed(self, netlist_path, results_dir, **kwargs):
        files = resolve(self, netlist_path, results_dir, **kwargs)
        return replace(
            files,
            rawfile_path=(str(Path(results_dir) / 'testrawfile123.rawfile')
                          if files.rawfile_path else None),
            logfile_path=(str(Path(results_dir) / 'testlog123.log')
                          if files.logfile_path else None),
            noise_path=(str(Path(results_dir) / 'testnoise123.csv')
                        if files.noise_path else None),
        )

    monkeypatch.setattr(backend_cls, '_resolve_run_files', renamed)


@pytest.fixture
def fixture_cli(monkeypatch, mock_ngspice_version):
    calls = []

    def run(argv, **kwargs):
        deck = Path(argv[-1])
        text = deck.read_text()
        if '-b' in argv:
            raw = Path(argv[argv.index('-r') + 1])
            log = Path(argv[argv.index('-o') + 1])
            assert f'write {raw}\n' in text or '.save all\n.end' in text
            write_complete_rawfile(raw, text, values={'v(out)': [1., 1.8]})
            log.write_text('fixture log\n')
        else:
            assert kwargs['cwd'] == str(deck.parent)
            for kind, path in re.findall(r'format=(raw|csv) file=(\S+)', text):
                if kind == 'raw':
                    write_complete_rawfile(path, text, simulator='xyce', values={'v(out)': [1., 1.8]})
                else:
                    Path(path).write_text(complete_noise_csv(text))
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', run)
    return calls


def test_resolution_is_pure_and_preparation_keeps_existing_files(tmp_path):
    root = tmp_path / 'absent'
    files = RunFiles.resolve(str(root / 'deck.cir'), str(root), log=True)
    assert not root.exists()
    assert files.rawfile_path == str(root / 'rawfile.raw')
    assert files.logfile_path == str(root / 'output.log')
    assert files.noise_path is None
    files.prepare()
    raw = Path(files.rawfile_path)
    raw.write_text('previous output')
    files.prepare()
    assert raw.read_text() == 'previous output'
    with pytest.raises(FrozenInstanceError):
        files.directory = 'elsewhere'
    assert copy.deepcopy(files) == pickle.loads(pickle.dumps(files)) == files


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_nondefault_files_reach_emission_launch_and_load(
        tmp_path, monkeypatch, fixture_cli, simulator, mode):
    cls = NgspiceBackend if simulator == 'ngspice' else XyceBackend
    _rename_outputs(monkeypatch, cls)
    mod = _module(tmp_path, simulator)
    mod.set_output_options(mode=mode)
    assert mod.simulate() is mod
    files = mod._run_files
    assert mod._raw.run_files is files
    assert mod.results_file == mod.dataFile == files.rawfile_path
    assert Path(files.rawfile_path).name == 'testrawfile123.rawfile'
    np.testing.assert_allclose(mod.load_signal('out').y, [1, 1.8])
    assert not (tmp_path / 'rawfile.raw').exists()
    assert len(fixture_cli) == 1


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_manual_launch_consumes_the_prepared_location(
        tmp_path, monkeypatch, fixture_cli, simulator):
    mod = _module(tmp_path, simulator)
    mod.set_output_options(mode='overwrite')
    mod.generate_netlist()
    files = mod._run_files
    deck = Path(files.netlist_path).read_bytes()
    mod.dir_folder = str(tmp_path / 'next')
    mod.netlistname = 'next.cir'

    def unexpected_resolution(*args, **kwargs):
        raise AssertionError('An already-prepared deck must keep its files')

    monkeypatch.setattr(type(mod.backend), '_resolve_run_files', unexpected_resolution)
    mod.run_simulator()
    assert mod._run_files is files
    assert mod.results_file == files.rawfile_path
    assert Path(files.netlist_path).read_bytes() == deck
    assert not (tmp_path / 'next').exists()


def test_active_location_does_not_replace_configured_root(tmp_path, monkeypatch, fixture_cli):
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    configured = mod.dir_folder, mod.netlistname
    resolve = NgspiceBackend._resolve_run_files
    selected = tmp_path / 'selected'

    def select(self, netlist_path, results_dir, **kwargs):
        if results_dir == '/__silk_execution__':
            # The producing digest emits against fixed synthetic RunFiles.
            return resolve(self, netlist_path, results_dir, **kwargs)
        assert (results_dir, netlist_path) == (str(tmp_path), str(tmp_path / 'deck.cir'))
        return resolve(self, str(selected / 'deck.cir'), str(selected), **kwargs)

    monkeypatch.setattr(NgspiceBackend, '_resolve_run_files', select)
    for _ in range(2):
        mod.simulate()
        assert (mod.dir_folder, mod.netlistname) == configured
        assert mod._run_files.directory == str(selected)
        assert mod.results_file == str(selected / 'rawfile.raw')


@pytest.mark.parametrize('renamed', [False, True])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_xyce_noise_companion_is_shared(
        tmp_path, monkeypatch, fixture_cli, renamed, mode):
    if renamed:
        _rename_outputs(monkeypatch, XyceBackend)
    analyses = [Noise('out', '0', 'vin', 'dec', 1, 1, 10)]
    mod = _module(tmp_path, 'xyce', analyses)
    mod.set_output_options(mode=mode)
    mod.simulate()
    files = mod._run_files
    assert Path(files.logfile_path).name == ('testlog123.log' if renamed else 'output.log')
    assert Path(files.logfile_path).is_file()
    assert files.rawfile_path is None
    assert mod.results_file == files.noise_path
    np.testing.assert_allclose(mod.load_signal('noise_out').y, [2e-5, 1e-5])
    np.testing.assert_allclose(mod.load_signal('noise_in').y, [1e-6, 5e-7])
    assert mod._raw.native is None
    assert not (tmp_path / 'rawfile.raw').exists()


@pytest.mark.parametrize('runner_cls', [NgspiceRunner, XyceRunner])
def test_direct_runner_paths_do_not_prepare_or_move_a_deck(tmp_path, runner_cls):
    deck = tmp_path / 'external.cir'
    deck.write_text('external deck')
    out = tmp_path / 'absent'
    runner = runner_cls(str(deck), str(out))
    assert runner.netlist_path == str(deck)
    assert runner.rawfile_path == str(out / 'rawfile.raw')
    assert not out.exists()
    assert deck.read_text() == 'external deck'
    with pytest.raises(AttributeError):
        runner.rawfile_path = str(out / 'drift.raw')


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_direct_backend_explicit_paths_remain_supported(tmp_path, fixture_cli, simulator):
    mod = _module(tmp_path, simulator)
    mod.set_output_options(mode='overwrite')
    deck_path = mod.generate_netlist()
    raw = mod.backend.run(deck_path, mod.dir_folder)
    assert raw.source_path == mod._run_files.rawfile_path
    np.testing.assert_allclose(mod.backend.load_signal(raw, 'out').y, [1, 1.8])


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_subcircuit_emission_has_no_preparation(mode, tmp_path):
    root = tmp_path / 'sub'
    mod = Module('sky130', str(root), 'sub.cir', load_pdk('sky130'), module_name='sub')
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('rload', p='out', n='vss', value=1000))
    mod.set_output_options(mode=mode)
    root.rmdir()  # Module construction still creates its configured directory.
    assert '.subckt sub' in mod.generate_netlist()
    assert mod._run_files is None
    assert not root.exists()


@pytest.mark.parametrize('copy_module', [copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))])
def test_prepared_module_survives_copy_and_pickle(tmp_path, copy_module):
    mod = _module(tmp_path)
    mod.generate_netlist()
    copied = copy_module(mod)
    assert copied._run_files == mod._run_files
    assert (copied.dir_folder, copied.netlistname) == (mod.dir_folder, mod.netlistname)


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_native_result_survives_simulate_and_another_file_still_loads(mode, tmp_path, monkeypatch):
    native = SimpleNamespace(npoints=2, payload='in memory')

    class InMemoryBackend(SpiceBackend):
        def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
            return RawResult(native, netlist_path)

        def make_netlister(self, pdk_details):
            return PySpiceNetlister(pdk_details)

    monkeypatch.setitem(SimulatorBackend._registry, 'inmemory_paths', InMemoryBackend)
    mod = _module(tmp_path, 'inmemory_paths')
    mod.terminal_current_method = 'savecurrents'
    mod.set_output_options(mode=mode)
    mod.simulate()
    assert mod.simResultsAll is mod._raw.native is native
    assert mod.results_file == mod.dataFile == mod._run_files.netlist_path
    assert mod._raw.run_files is mod._run_files
    assert not (tmp_path / 'rawfile.raw').exists()
    other = tmp_path / 'other.raw'
    other.write_text(_RAW_TEXT)
    mod.load_file(str(other))
    mod.read_raw_file()
    assert mod._raw.native is not native
    assert mod._raw.source_path == str(other)
    np.testing.assert_allclose(mod.load_signal('out').y, [1, 1.8])


def test_nondefault_rawfile_is_the_file_validated(tmp_path, monkeypatch, mock_ngspice_version):
    _rename_outputs(monkeypatch, NgspiceBackend)
    mod = _module(tmp_path)
    mod.set_output_options(mode='overwrite')
    (tmp_path / 'rawfile.raw').write_text(_RAW_TEXT)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(SimulationAborted, match='testrawfile123.rawfile'):
        mod.simulate()


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_prepared_backend_keeps_the_data_gate(tmp_path, monkeypatch, simulator):
    mod = _module(tmp_path, simulator)
    mod.generate_netlist()

    def empty(self, files, **kwargs):
        return RawResult(SimpleNamespace(npoints=0), files.rawfile_path)

    monkeypatch.setattr(type(mod.backend), '_launch_prepared', empty)
    with pytest.raises(SimulationAborted):
        mod.run_simulator()


@needs_ngspice
@pytest.mark.parametrize('batch_write', [False, True])
def test_real_ngspice_nondefault_rawfile(tmp_path, monkeypatch, batch_write):
    _rename_outputs(monkeypatch, NgspiceBackend)
    mod = _module(tmp_path)
    mod.save_device_currents = batch_write
    mod.simulate()
    assert Path(mod.results_file).name == 'testrawfile123.rawfile'
    np.testing.assert_allclose(mod.load_signal('out').y, 1)
    assert Path(mod._run_files.logfile_path).is_file()
    assert not (tmp_path / 'rawfile.raw').exists()


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.parametrize('noise_only', [False, True])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_real_xyce_nondefault_outputs(mode, tmp_path, monkeypatch, noise_only):
    if shutil.which(_xyce_executable()) is None:
        pytest.skip('Xyce unavailable; set SILK_XYCE_BIN or put Xyce on PATH')
    _rename_outputs(monkeypatch, XyceBackend)
    analyses = [Noise('out', '0', 'vin', 'dec', 1, 1, 10)] if noise_only else None
    mod = _module(tmp_path, 'xyce', analyses)
    mod.set_output_options(mode=mode)
    mod.simulate()
    files = mod._run_files
    assert mod._raw.run_files is files
    assert mod.results_file == (files.noise_path if noise_only else files.rawfile_path)
    signal = mod.load_signal('noise_out' if noise_only else 'out')
    assert len(signal.y) > 0 and np.all(np.isfinite(signal.y))
    assert not (tmp_path / 'rawfile.raw').exists()
    assert not (tmp_path / 'noise.csv').exists()
