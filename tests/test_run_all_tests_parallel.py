"""run_all_tests(n_workers=) -- parallel across tests (TASK-301.4).

``n_workers=None``/``1`` must keep calling ``test.run(**config)`` once per
test, in-process, exactly as before (reviewer constraint, TASK-301.4 notes) --
that path is untouched code, so it is not re-proved here. What these tests
pin is the *parallel* walk (``n_workers>1``): identical per-test reports on a
clean multi-test tree (AC#1), a shared output folder per test surviving the
sweep with no path collisions (AC#2 second half), and ``ConfigurationError``
still stopping the whole call while an ordinary run failure still becomes
just that test's ``{'error': ...}`` entry (AC#2 first half).

Every DUT that a parallel run actually dispatches to the pool here is a real
sky130 NMOS device (the exact shape ``tests/test_run_split.py``'s AC#2 test
already proves pickles cleanly), sharing ONE ``load_pdk('sky130')`` call
across the whole tree -- deliberately, not the ideal
``silk.elements.elements.resistor``/``capacitor``: those share their class
*name* with whatever PDK-specific device a YAML also happens to call
'resistor' ("last registration wins" in ``silk.elements.factory``'s pickle
registry, by that module's own design), so a pickled ideal-element netlist
can go stale mid-suite once some *other* test file elsewhere in the same
process has caused that name to be re-registered -- reproduced here while
authoring against the full fast-suite chunk containing it, not in isolation. A PDK
device has no such collision, and loading the PDK exactly once (not once per
DUT) avoids the *other* half of the same class of bug (``load_pdk`` rebuilds
its whole device catalog on each call, so a second call can equally leave an
earlier DUT's own class reference stale). Every real launch here is a
sub-nanosecond-scale transient, so this stays fast despite using the real
simulator throughout; two tests inject a synthetic failure by monkeypatching
``ModuleTestbench._simulate_run`` directly rather than faking ngspice, since
the point there is exception classification, not the simulator.
"""
import os
import time
from pathlib import Path

import pytest

from silk import Port, load_pdk, sources
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.modules import Module
from silk.exceptions import (
    CharacterizationContextError, RunArgumentError,
    SimulationAborted, SimulationConfigurationError,
)
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk

pytestmark = [needs_pdk, needs_ngspice]


def _nmos_dut(tmp_path, name, pdk, *, w=2.0, l=0.5, gate_dc=1.0, child=None,
              defer_test=False):
    """A tiny real sky130 NMOS DUT + test, sharing an already-loaded *pdk*.

    Same shape as ``tests/test_run_split.py``'s ``_build_nmos_tb`` (that
    file's AC#2 test already proves this pickles across a real process
    boundary) -- *pdk* must be the SAME object across every DUT built for one
    tree, never a fresh ``load_pdk('sky130')`` per DUT (see module docstring).
    """
    dut = Module('sky130', str(tmp_path), f'{name}.cir', pdk_details=pdk,
                module_name=name)
    dut.define_module_ports({'D': Port.analog_output(), 'G': Port.analog_input(),
                             'S': Port.ground()})
    with dut:
        pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=w, l=l)
    if child is not None:
        dut.add_submodule(child.module_name, child)
    return dut, None if defer_test else _attach_level_test(dut, gate_dc)


def _attach_level_test(dut, gate_dc=1.0):
    tb = dut.create_test('level')
    tb.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
    tb.add_source(sources.VoltageSource('drain', positive='drain', negative='0', dc=0.8))
    tb.add_source(sources.VoltageSource('gate', positive='gate', negative='0', dc=gate_dc))
    tb.add_sim_type(Transient(step=1e-9, stop=5e-9))
    tb.add_metric('drain_current_A',
                  lambda ckt: -float(ckt.load_signal('vdrain', 'I').y[-1]))
    tb.add_spec('conducts', 'drain_current_A', '>', 0.0)
    return tb


@pytest.fixture(scope='module')
def pdk():
    return load_pdk('sky130')


@pytest.fixture
def tree3(tmp_path, pdk):
    """A three-test tree: root.level, mid.level (registered submodule),
    leaf.level (a submodule of mid) -- deep enough that the recursive walk
    exercises both elem_dict and module_dict recursion at more than one
    level, matching what _iter_test_nodes/_run_test_entries both traverse.
    """
    # TASK-458 restores order-independence; finalize composition before placement.
    leaf, _ = _nmos_dut(tmp_path / 'leaf', 'leaf', pdk, defer_test=True)
    mid, _ = _nmos_dut(tmp_path / 'mid', 'mid', pdk, child=leaf, defer_test=True)
    root, _ = _nmos_dut(tmp_path / 'root', 'root', pdk, child=mid, defer_test=True)
    # Attach top-down so nested circuit copies do not carry child-bench lambdas.
    root_tb = _attach_level_test(root)
    mid_tb = _attach_level_test(mid)
    leaf_tb = _attach_level_test(leaf)
    yield root, root_tb, mid, mid_tb, leaf, leaf_tb
    root.close()
    mid.close()
    leaf.close()


class TestParallelIdenticalToSerial:
    def test_clean_tree_gives_identical_reports_serial_vs_parallel(self, tree3):
        root, _root_tb, mid, _mid_tb, leaf, _leaf_tb = tree3
        serial = root.run_all_tests(recursive=True, n_workers=1)
        assert set(serial) == {'root.level', 'mid.level', 'leaf.level'}
        for key in serial:
            assert serial[key].metrics['drain_current_A'] > 0
            assert serial[key].results[0].passed is True

        parallel = root.run_all_tests(recursive=True, n_workers=4)
        assert set(parallel) == set(serial)
        for key in serial:
            assert parallel[key].metrics == pytest.approx(serial[key].metrics)
            assert parallel[key].results[0].passed == serial[key].results[0].passed
            assert parallel[key].operating_conditions.temperature_c == \
                serial[key].operating_conditions.temperature_c
            # Overwrite mode (the default) reuses one path per bench across
            # repeated runs -- that's why this is the SAME path both times,
            # and "no two tests share a path" is proved separately below.
            assert os.path.isfile(parallel[key].rawfile_path)

    def test_first_class_spec_measurement_matches_serial(self, tree3):
        """AC#1's "database records" half: add_spec() attaches a first-class
        Spec (silk.specs.Spec), and evaluate()'s shared tail calls
        spec.record_measurement() on it identically whether this bench's
        report came from run() directly or from the split
        prepare/simulate(worker)/evaluate sequence.
        """
        root, root_tb, _mid, _mid_tb, _leaf, _leaf_tb = tree3
        spec_obj = root_tb._specs[0]['spec']
        assert spec_obj is not None

        root.run_all_tests(recursive=True, n_workers=1)
        serial_measurement = next(iter(spec_obj.measurements.values()))

        root.run_all_tests(recursive=True, n_workers=4)
        parallel_measurement = next(iter(spec_obj.measurements.values()))

        assert parallel_measurement.value == pytest.approx(serial_measurement.value)
        assert parallel_measurement.passes == serial_measurement.passes
        assert parallel_measurement.corner == serial_measurement.corner

    def test_n_workers_all_and_capped_int_also_match_serial(self, tree3):
        root, *_ = tree3
        serial = root.run_all_tests(recursive=True, n_workers=1)
        for n_workers in (2, 'all'):
            got = root.run_all_tests(recursive=True, n_workers=n_workers)
            assert got.keys() == serial.keys()
            for key in serial:
                assert got[key].metrics == pytest.approx(serial[key].metrics)

    def test_no_two_tests_share_an_output_path(self, tree3):
        root, *_ = tree3
        results = root.run_all_tests(recursive=True, n_workers=3)
        paths = [os.path.normpath(r.rawfile_path) for r in results.values()]
        assert len(paths) == len(set(paths)) == 3
        for path in paths:
            assert os.path.isfile(path)   # still readable after the sweep

    def test_iter_test_nodes_ambient_matches_nearest_setting_wins(self, tree3):
        """``_iter_test_nodes`` feeds the parallel walker's per-test override
        installation (TASK-301.4); this pins that its ambient/supply values,
        not just its traversal order, match the rule
        ``_run_test_entries`` has always used (TASK-162/TASK-168: a module
        that sets its own temperature replaces it for itself and its
        descendants, so the nearest setting up the tree wins).
        """
        from silk.circuit.modules_testbench import _iter_test_nodes

        root, _root_tb, mid, _mid_tb, _leaf, _leaf_tb = tree3
        root.set_temperature(45)
        mid.set_temperature(85)   # nearest setting wins for mid + leaf

        nodes = list(_iter_test_nodes(root, recursive=True))
        # own_temperature_setting() returns (value, basis); only the value
        # matters for "which setting wins".
        ambient_by_key = {legacy_key: ambient[0]
                          for legacy_key, _owner, _test, ambient, _supply
                          in nodes}
        assert ambient_by_key == {'root.level': 45, 'mid.level': 85,
                                  'leaf.level': 85}


class TestParallelErrorClassification:
    def test_configuration_error_still_propagates_and_stops_the_tree(self, tree3):
        root, _root_tb, _mid, _mid_tb, _leaf, _leaf_tb = tree3
        with pytest.deprecated_call():
            with pytest.raises(CharacterizationContextError,
                               match='requires matching explicit fixed TEMP'):
                root.run_all_tests(temperature=85, recursive=True, n_workers=4)

    def test_configuration_error_from_the_simulate_step_still_propagates(
            self, tree3, monkeypatch):
        """A ConfigurationError raised *inside* the worker's simulate step
        (not at prepare time -- e.g. modules_sim.py's deck-staleness guards,
        which read the emitted deck and can raise SimulationConfigurationError
        from inside netlist.simulate()) must still propagate with its real
        type once collected back from the pool, not fold into a per-test
        {'error': ...} entry the way an ordinary exception would.
        ``_simulate_run_task`` (the executor adapter) is what makes that
        happen: it classifies a ConfigurationError itself, before the
        executor's own generic catch-all ever sees it.
        """
        # Every _RunSpec crosses into its own forked pool worker as a fresh
        # unpickled netlist copy, so an identity check against a live
        # bench's netlist would not survive the round trip -- raise
        # unconditionally instead; what this test proves is that ANY
        # ConfigurationError out of the simulate step still propagates, not
        # which specific test hit it.
        def _simulate_run(spec):
            raise SimulationConfigurationError(
                'synthetic staleness guard for this test')

        monkeypatch.setattr(ModuleTestbench, '_simulate_run',
                           staticmethod(_simulate_run))
        root, *_ = tree3
        with pytest.raises(SimulationConfigurationError,
                           match='synthetic staleness guard'):
            root.run_all_tests(recursive=True, n_workers=4)

    def test_ordinary_failure_becomes_that_tests_error_the_rest_complete(
            self, tree3, monkeypatch):
        """One test's simulate step raises an ordinary (non-Configuration)
        exception; the other two, dispatched to the same pool, still
        complete with real reports. ``spec.tb_dir`` (a plain string,
        survives the pickle round trip) is how the injected failure targets
        only ``root`` -- an object-identity check would not, since the
        worker's spec is an unpickled copy (see the test above).
        """
        real_simulate_run = ModuleTestbench._simulate_run

        def _simulate_run(spec):
            if 'root' in os.path.normpath(spec.tb_dir).split(os.sep):
                raise SimulationAborted('synthetic abort for this test',
                                       simulator='ngspice', data_rows=0)
            return real_simulate_run(spec)

        monkeypatch.setattr(ModuleTestbench, '_simulate_run',
                           staticmethod(_simulate_run))
        root, _root_tb, mid, _mid_tb, leaf, _leaf_tb = tree3

        results = root.run_all_tests(recursive=True, n_workers=4)

        assert set(results) == {'root.level', 'mid.level', 'leaf.level'}
        assert set(results['root.level']) == {'error'}
        assert not isinstance(results['mid.level'], dict)
        assert not isinstance(results['leaf.level'], dict)
        assert results['mid.level'].metrics['drain_current_A'] > 0
        assert results['leaf.level'].metrics['drain_current_A'] > 0

    def test_run_argument_error_raises_before_any_worker_runs(self, tree3):
        root, *_ = tree3
        with pytest.raises(RunArgumentError):
            root.run_all_tests(recursive=True, n_workers=4, store=True)

    def test_invalid_n_workers_raises_value_error(self, tree3):
        root, *_ = tree3
        with pytest.raises(ValueError):
            root.run_all_tests(recursive=True, n_workers=0)
        with pytest.raises(ValueError):
            root.run_all_tests(recursive=True, n_workers=-1)


class TestOverriddenRunIsNotSplit:
    """A subclass overriding ``run`` (the examples' shape --
    ``PVTTestbench``/``_TxTest``) must still run in-process under
    ``n_workers>1``, not through the split/executor path, since its override
    may not even use ``_prepare_run``/``_simulate_run``/``_evaluate_run`` --
    :class:`ModuleTestbench`'s own tail those methods drive.

    Incrementing a counter *on the live test object itself* from inside the
    override is the discriminator: a call dispatched to a pool worker (via
    ``fork``) mutates that worker's own copy, invisible back in this
    (parent) process, so seeing the increment here proves the call ran here
    too -- exactly like the split-eligible path's own behavioral in-process
    branch, and exactly what a real override's side effects (e.g. state a
    behavioral model keeps between calls) need to keep working.
    """

    def test_override_runs_in_process_under_a_pool(self, tree3):
        root, root_tb, _mid, _mid_tb, leaf, _leaf_tb = tree3

        from silk.circuit.spec_report import SpecReport

        class _CountingOverride(type(root_tb)):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.calls = 0

            def run(self, **config):
                self.calls += 1
                return SpecReport(test_name=self.test_name,
                                  module_name=self.module.module_name,
                                  metrics={'calls': self.calls})

        # Splice the override in after construction (the real examples build
        # their subclass from the start; this only needs the same run()
        # override shape on an already-registered test).
        root_tb.__class__ = _CountingOverride
        root_tb.calls = 0

        results = root.run_all_tests(recursive=True, n_workers=4)
        assert results['root.level'].metrics['calls'] == 1
        assert root_tb.calls == 1   # visible HERE -> ran in this process
        # leaf (mid's submodule) is split-eligible and really dispatched.
        assert not isinstance(results['leaf.level'], dict)
        assert results['leaf.level'].metrics['drain_current_A'] > 0


def _rawfile_bytes_ignoring_date(path):
    lines = Path(path).read_bytes().split(b'\n', 2)
    assert lines[1].startswith(b'Date:')
    return lines[0] + b'\n' + lines[2]


class TestParallelRealSimulator:
    def test_real_ngspice_identical_and_measures_speedup(self, tmp_path, pdk):
        """Eight independent real sky130 NMOS benches, run serially and in
        parallel, sharing the ONE ``pdk`` object the module-scoped fixture
        already loaded.

        Proves byte-identical rawfiles/metrics against a real simulator (not
        a mock) and records a measured speed-up (TASK-301.4 AC#3) -- printed
        so it lands in this task's implementation notes. Each bench's own
        transient is long enough that the speed-up reflects real per-launch
        simulator time rather than being swamped by pool start-up overhead.
        """
        n_tests = 8

        def _tree(root_dir):
            # TASK-458: finish the full tree before any bench places its DUT.
            root, _ = _nmos_dut(root_dir / 'root', 'root', pdk, defer_test=True)
            mods = [root]
            for i in range(1, n_tests):
                child, _ = _nmos_dut(root_dir / f'child{i}', f'child{i}', pdk,
                                     w=2.0 + 0.1 * i, defer_test=True)
                root.add_submodule(f'child{i}', child)
                mods.append(child)
            for mod in mods:
                _attach_level_test(mod)
            return root, mods

        serial_root, serial_mods = _tree(tmp_path / 'serial')
        parallel_root, parallel_mods = _tree(tmp_path / 'parallel')
        try:
            t0 = time.monotonic()
            serial = serial_root.run_all_tests(recursive=True, n_workers=1)
            serial_elapsed = time.monotonic() - t0

            t0 = time.monotonic()
            parallel = parallel_root.run_all_tests(recursive=True,
                                                   n_workers=n_tests)
            parallel_elapsed = time.monotonic() - t0

            assert len(serial) == len(parallel) == n_tests
            for key in serial:
                s, p = serial[key], parallel[key]
                assert p.metrics == pytest.approx(s.metrics)
                assert p.results[0].passed == s.results[0].passed
                # Byte-identical except ngspice's own "Date: <wall clock>"
                # header line, which legitimately differs between any two
                # separate launches -- serial vs serial rerun, not just
                # serial vs parallel.
                assert _rawfile_bytes_ignoring_date(p.rawfile_path) == \
                    _rawfile_bytes_ignoring_date(s.rawfile_path)

            speedup = serial_elapsed / parallel_elapsed
            print(f"\nTASK-301.4 measured speed-up ({n_tests} tests, "
                 f"n_workers={n_tests}): serial={serial_elapsed:.2f}s "
                 f"parallel={parallel_elapsed:.2f}s speedup={speedup:.2f}x")
            assert speedup > 1.0
        finally:
            for mod in serial_mods + parallel_mods:
                mod.close()


def test_shared_test_selections_match_across_real_workers(tmp_path, pdk):
    child, tb = _nmos_dut(tmp_path, 'shared', pdk)
    top = Module('sky130', str(tmp_path), 'top.cir', pdk)
    for name in ('a', 'b'):
        top.add_module(child, name, D=f'd_{name}', G=f'g_{name}', S='0')
    entries = top.list_test_entries()
    original = tb.netlist
    manual = [entry.test.run() for entry in entries]
    for workers in (1, 2):
        reports = top.run_all_tests(n_workers=workers)
        assert list(reports) == [entry.report_key for entry in entries]
        for entry, expected in zip(entries, manual):
            actual = reports[entry.report_key]
            assert actual.metrics == pytest.approx(expected.metrics)
            assert actual.owner_path == entry.owner_path
            assert entry.test._metric_values == actual.metrics
        assert tb.netlist is original
    top.close()
    child.close()
