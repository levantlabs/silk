"""Embedded tests select one backend for emission, execution and readback."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk import Module, Variable, sources
from silk.exceptions import SimulationAborted, SimulationError
from silk.optimize import Optimizer, VariableSpace
from silk.simulation import Transient
from silk.simulation.analysis import AnalysisBase
from silk.simulation.backend import SpiceBackend
from tests._env import needs_ngspice
from tests.test_simulation_freshness import _raw
from tests.test_testbench_output import _copies, _load, _test


@pytest.fixture
def pipeline(monkeypatch, mock_ngspice_version):
    """Replace only subprocess output; exercise both real runners and loaders."""
    launches, loads = [], []
    load_signal = SpiceBackend.load_signal

    def read(self, *args, **kwargs):
        loads.append(self.name)
        return load_signal(self, *args, **kwargs)

    def launch(argv, **kwargs):
        deck = Path(argv[-1])
        text = deck.read_text()
        if '-b' in argv:
            backend = 'ngspice'
            raw = Path(argv[argv.index('-r') + 1])
            log = Path(argv[argv.index('-o') + 1])
            assert '.control' in text and '.print tran' not in text
            assert f'write {raw}' in text
            # The runner opens a fresh staged diagnostic before launch so even
            # a silent cancellation can retain a log; no old content survives.
            assert log.read_bytes() == b''
            assert log.parent.name.startswith('.silk-generation-')
            assert not (deck.parent / log.name).exists()
            log.write_text('fresh ngspice log\n')
        else:
            backend = 'xyce'
            assert '.control' not in text and '.print tran' in text
            raw = Path(re.search(r'\bfile=(\S+)', text).group(1).strip('"'))
            assert f'file={raw}' in text
            assert deck.parent == Path(kwargs['cwd'])
        assert raw.parent.parent == deck.parent
        assert not raw.exists(), 'previous output was not removed before launch'
        write_complete_rawfile(raw, text, simulator=backend, values={'v(out)': 1 if backend == 'ngspice' else 1.25})
        launches.append(backend)
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', launch)
    monkeypatch.setattr(SpiceBackend, 'load_signal', read)
    return launches, loads


def _assert_run(report, pipeline, backend):
    launches, loads = pipeline
    assert launches[-1] == loads[-1] == backend
    assert report.sim_results.backend.name == backend
    assert report.metrics['voltage'] == pytest.approx(1 if backend == 'ngspice' else 1.25)
    assert report.passed


@pytest.mark.parametrize('before_construction', [False, True])
def test_selection_before_and_after_test_construction(tmp_path, pipeline, before_construction):
    mod = _load(tmp_path, tests=() if before_construction else ('level',))
    mod.simulator = 'xyce'
    if before_construction:
        # Reuse the ordinary public factory setup with the selection already
        # present when ModuleTestbench constructs its executing Module.
        tb = mod.create_test('level')
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
        tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1))
        tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
        tb.add_metric('voltage', lambda ckt: float(ckt.load_signal('out').y[-1]))
        tb.add_spec('Voltage limit', 'voltage', '<', 1.5)
    _assert_run(mod.run_test('level'), pipeline, 'xyce')


@pytest.mark.parametrize('entry', ['direct', 'run_test', 'store', 'all', 'recursive', 'corners', 'optimizer'])
@pytest.mark.parametrize('override', [None, 'ngspice', 'xyce'])
def test_entry_points_switch_and_preserve_explicit_overrides(tmp_path, pipeline, entry, override):
    mod = _load(tmp_path, db_path=str(tmp_path / 'evidence.db') if entry == 'store' else None)
    tb = mod._tests['level']
    if override is not None:
        tb.netlist.simulator = override
    if entry == 'store':
        mod.create_circuit_version()
    variable = Variable('trial', value=1.0, bounds=(1.0, 2.0))
    optimizer = Optimizer(tb, VariableSpace([variable]))
    try:
        for backend in ('ngspice', 'xyce', 'ngspice'):
            previous_netlister = tb.netlist.netlister
            previous_backend = tb.netlist.simulator
            mod.simulator = backend
            expected = override or backend
            if entry == 'direct':
                report = tb.run()
            elif entry in ('run_test', 'store'):
                report = mod.run_test('level', store=entry == 'store')
            elif entry in ('all', 'recursive'):
                report = mod.run_all_tests(recursive=entry == 'recursive')['load.level']
            elif entry == 'corners':
                report = tb.run_corners(['tt', 'ff'])['ff']
            else:
                record = optimizer.evaluate({'trial': 1.0})
                assert record['status'] == 'success', record
                assert record['metrics']['voltage'] == pytest.approx(1 if expected == 'ngspice' else 1.25)
                report = None
            if report is not None:
                _assert_run(report, pipeline, expected)
            assert pipeline[0][-1] == pipeline[1][-1] == expected
            if entry in ('all', 'recursive'):
                assert tb.netlist.netlister is previous_netlister
                assert tb.netlist.simulator == previous_backend
            else:
                assert (tb.netlist.netlister is previous_netlister) == (previous_backend == expected)
            assert mod.simulator == backend
        assert len(pipeline[0]) == (6 if entry == 'corners' else 3)
        if entry == 'store':
            assert len(mod.query_results(parameter='voltage')) == 3
    finally:
        mod.close()


def test_copied_nested_and_legacy_tests_follow_distinct_live_owners(tmp_path, pipeline):
    branch = _copies(tmp_path, tests=('level', 'override'))
    _test(branch, 'a', 'override').netlist.simulator = 'ngspice'
    _test(branch, 'b', 'override').netlist.simulator = 'xyce'
    tree = Module('sky130', str(tmp_path), 'tree.cir', branch.pdk_details)
    tree.add_module(branch, 'nested')  # Keep the live hierarchy.
    live = tree.elem_dict['nested']['device']
    a, b = _test(live, 'a').module, _test(live, 'b').module
    a.simulator, b.simulator = 'xyce', 'ngspice'
    legacy = _load(tmp_path, name='legacy')
    legacy.simulator = 'xyce'
    tree.add_submodule('legacy', legacy)
    tree.simulator = 'ngspice'
    # vntol is an ngspice tolerance with no Xyce TIMEINT equivalent (see
    # xyce.yaml), so the warning proves the live owner's backend chose the
    # config. TEMP is deliberately not the probe: Xyce supports it (TASK-140).
    with pytest.warns(UserWarning, match="Option 'vntol'"):
        a.set_simulator_options(reltol=1e-7, vntol=80)
    a._tests['level'].netlist.set_simulator_options(reltol=1e-4, tnom=20)
    reports = tree.run_all_tests()
    expected = {
        '$/nested/a:load.level': 'xyce', '$/nested/a:load.override': 'ngspice',
        '$/nested/b:load.level': 'ngspice', '$/nested/b:load.override': 'xyce',
        'legacy.level': 'xyce',
    }
    assert set(reports) == set(expected)
    assert pipeline[0] == pipeline[1] == list(expected.values())
    for key, backend in expected.items():
        assert reports[key].sim_results.simulator == backend
    assert a._tests['level'].module is a
    assert b._tests['level'].module is b
    assert (tree.simulator, a.simulator, b.simulator, legacy.simulator) == ('ngspice', 'xyce', 'ngspice', 'xyce')
    assert a.simulator_options == {'reltol': 1e-7, 'vntol': 80}
    # TASK-224: 'vntol' is a's own explicit option and the test never set it
    # itself, so it reaches the test's netlist as an inherited default; the
    # test's own 'reltol'/'tnom' still win over the owner's 'reltol'.
    assert reports['$/nested/a:load.level'].sim_results.simulator_options == {
        'reltol': 1e-4, 'tnom': 20, 'vntol': 80}
    assert b.simulator_options == b._tests['level'].netlist.simulator_options == {}
    assert _test(branch, 'a').module.simulator == 'xyce'
    a.simulator, b.simulator = 'ngspice', 'xyce'
    tree.run_all_tests()
    assert pipeline[0][-5:] == pipeline[1][-5:] == ['ngspice', 'ngspice', 'xyce', 'xyce', 'xyce']
    assert tree.run_all_tests(recursive=False) == {}


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_cache_true_still_simulates_every_effective_backend(tmp_path, pipeline, mode):
    """Formerly test_cache_keys_effective_backend_without_artifact_work_on_hits:
    also asserted that re-selecting a backend used earlier reused that
    backend's cached report without any artifact work. The report cache is
    disabled in 0.2.0 (TASK-284), so a cache hit can no longer occur -- every
    backend selection re-simulates, even one already used before."""
    with pytest.warns(UserWarning, match='TASK-130'):
        mod = _load(tmp_path, cache=True)
    mod.set_output_options(mode=mode)
    tb = mod._tests['level']
    first = tb.run(run_label='baseline')
    mod.simulator = 'xyce'
    second = tb.run(run_label='xyce')
    _assert_run(second, pipeline, 'xyce')
    assert second is not first
    # Explicitly selecting ngspice again still re-simulates -- no cache to
    # reuse ngspice evidence from, even with the owner selecting Xyce.
    tb.netlist.simulator = 'ngspice'
    third = tb.run(run_label='unused')
    _assert_run(third, pipeline, 'ngspice')
    assert third is not first
    tb.netlist.simulator = 'xyce'
    mod.simulator = 'ngspice'
    fourth = tb.run(run_label='xyce')
    _assert_run(fourth, pipeline, 'xyce')
    assert fourth is not second
    assert pipeline[0] == ['ngspice', 'xyce', 'ngspice', 'xyce']
    tb.clear_cache()  # remains callable and harmless
    _assert_run(tb.run(run_label='xyce'), pipeline, 'xyce')


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_switching_preserves_output_lifecycle_and_explicit_readback(tmp_path, pipeline, mode):
    mod = _load(tmp_path)
    mod.set_output_options(mode=mode)
    tb = mod._tests['level']
    paths = []
    for backend in ('ngspice', 'xyce', 'ngspice'):
        mod.simulator = backend
        _assert_run(tb.run(run_label='baseline'), pipeline, backend)
        paths.append(tb.results_file)
    assert len(set(paths)) == (3 if mode == 'new' else 1)
    old_path = Path(tb.results_file)
    old_bytes = old_path.read_bytes()
    mod.simulator = 'xyce'
    held_report = tb.run(run_label='other')
    assert old_path.read_bytes() == old_bytes
    before = set(tmp_path.rglob('*'))
    # Explicit evaluation reads disk afresh even while the owner changes.
    mod.simulator = 'ngspice'
    assert tb.evaluate(rawfile=old_path).metrics['voltage'] == pytest.approx(1)
    old_path.write_text(_raw(0.75))
    assert tb.evaluate(rawfile=old_path).metrics['voltage'] == pytest.approx(0.75)
    # Retain its generation while comparing files; releasing the last reader
    # intentionally removes private artifacts.
    assert held_report.metrics['voltage'] == pytest.approx(1.25)
    assert set(tmp_path.rglob('*')) == before
    assert len(pipeline[0]) == 4


@pytest.mark.parametrize('failure', ['missing', 'no_data'])
def test_switch_failure_invalidates_results_without_fallback(tmp_path, pipeline, monkeypatch, failure):
    tb = _load(tmp_path)._tests['level']
    tb.module.set_output_options(mode='overwrite')
    tb.run()
    old_file = Path(tb.results_file)
    tb.module.simulator = 'xyce'
    calls = []
    monkeypatch.setenv('SILK_XYCE_BIN', '/missing/selected-Xyce')

    def fail(argv, **kwargs):
        calls.append(argv[0])
        assert not old_file.exists()
        if failure == 'missing':
            raise FileNotFoundError(argv[0])
        return subprocess.CompletedProcess(argv, 0, stdout='End of Xyce\n', stderr='')

    monkeypatch.setattr(subprocess, 'run', fail)
    error = SimulationError if failure == 'missing' else SimulationAborted
    with pytest.raises(error, match='Xyce|xyce'):
        tb.run()
    assert calls == ['/missing/selected-Xyce']
    assert tb.results_file is None
    assert not tb.netlist.sim_results_loaded
    with pytest.raises(SimulationError):
        tb.netlist.load_signal('out')


def test_selection_is_inert_and_unknown_name_does_not_pin_override(tmp_path, pipeline):
    tb = _load(tmp_path)._tests['level']
    tb.run()
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='No simulator backend registered'):
        tb.netlist.simulator = 'unknown-backend'
    tb.module.simulator = 'xyce'
    assert {p: p.read_bytes() for p in before} == before
    assert len(pipeline[0]) == 1
    _assert_run(tb.run(), pipeline, 'xyce')
    tb.netlist.simulator = 'ngspice'
    assert len(pipeline[0]) == 2
    _assert_run(tb.run(), pipeline, 'ngspice')


def test_unsupported_analysis_names_selected_backend_before_launch(tmp_path, pipeline):
    class UnsupportedAnalysis(AnalysisBase):
        pass

    tb = _load(tmp_path)._tests['level']
    tb.add_sim_type(UnsupportedAnalysis())
    tb.module.simulator = 'xyce'
    with pytest.raises(TypeError, match='XyceBackend cannot render analysis UnsupportedAnalysis'):
        tb.run()
    assert pipeline[0] == []


def test_standalone_simulate_keeps_its_own_selection(tmp_path, pipeline):
    tb = _load(tmp_path)._tests['level']
    tb.module.simulator = 'xyce'
    # Calling the executing Module directly retains ordinary Module semantics.
    tb.netlist.simulate()
    assert pipeline[0] == ['ngspice']
    tb.netlist.simulator = 'xyce'
    tb.netlist.simulate()
    assert pipeline[0] == ['ngspice', 'xyce']


_XYCE_BIN = os.environ.get('SILK_XYCE_BIN') or shutil.which('Xyce') or os.path.expanduser('~/xyce/Xyce-7.10.0/bin/Xyce')


@pytest.mark.slow
@pytest.mark.requires_xyce
@needs_ngspice
@pytest.mark.skipif(not os.path.isfile(_XYCE_BIN), reason='needs the Xyce executable')
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_real_inheritance_switching_and_opposite_override(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('SILK_XYCE_BIN', _XYCE_BIN)
    actual_run = subprocess.run
    executables = []

    def observe(argv, **kwargs):
        # The cached identity probe is not a testbench simulation.
        if '--version' not in argv:
            executables.append(argv[0])
        return actual_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, 'run', observe)
    mod = _load(tmp_path)
    mod.set_output_options(mode=mode)
    tb = mod._tests['level']
    expected = []
    for backend in ('ngspice', 'xyce', 'ngspice', 'xyce'):
        mod.simulator = backend
        report = mod.run_test('level')
        assert report.passed
        assert report.metrics['voltage'] == pytest.approx(1)
        expected.append(_XYCE_BIN if backend == 'xyce' else os.environ.get('SILK_NGSPICE_BIN', 'ngspice'))
    tb.netlist.simulator = 'ngspice'  # Opposite of the live Xyce owner.
    report = tb.run()
    assert report.passed and report.metrics['voltage'] == pytest.approx(1)
    expected.append(os.environ.get('SILK_NGSPICE_BIN', 'ngspice'))
    assert executables == expected
