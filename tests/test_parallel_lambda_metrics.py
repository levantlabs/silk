"""Lambda-metric benches must survive the parallel path (TASK-301.7 AC#1).

The bug (measured on ``silk/examples/current_mirror_example.py``'s
``match``/``zout`` pair): a worker-bound payload -- a testbench's own
executing netlist, or a whole module for ``FaultCampaign`` -- can embed a
device-under-test copy whose ``Module._specs`` holds a ``Spec``, and
``Spec._owning_testbench`` used to travel with it, reaching a SIBLING
testbench's metric callables. Two tests sharing one DUT via
``add_module_under_test`` is exactly what ``current_mirror_example.py`` does
(``match`` then ``zout``); the second test's embedded DUT copy carries a
replica of the first test's Spec, because that Spec was declared before the
second test's own ``add_module_under_test`` call froze the copy. Lambda
metrics are ordinary, supported silk usage (the ``match`` test itself, and
``source_follower_optimize.py``'s objective), so this reference broke every
parallel entry point once two tests shared one DUT: ``run_all_tests``,
``run_corners``, ``Optimizer.sweep`` and ``FaultCampaign`` (which pickles a
whole module, ``silk/faults/campaign.py``).

The fix (``Spec.__getstate__``, ``silk/specs/spec.py``) drops
``_owning_testbench`` only when PICKLING -- ``copy.deepcopy`` (composition)
is unchanged; see ``Spec.__deepcopy__`` and
``tests/specs/test_testbench_integration.py::
test_composition_copies_protected_specs_and_runtime_relationships``.

These tests build a minimal DUT reproducing that exact two-tests-one-DUT
shape directly (rather than importing ``current_mirror_example`` itself),
sharing ONE ``load_pdk('sky130')`` call for the whole module -- importing
more than one ``silk.examples`` file that each call ``load_pdk`` at their
own import time re-registers ``silk.elements.factory``'s device classes and
leaves an earlier example's already-built circuit unpicklable for an
unrelated reason (documented in ``tests/test_run_all_tests_parallel.py``'s
module docstring); the actual reported reproducer is exercised directly,
once, in ``TestReportedReproducer`` below.
"""
import os
import sys

import pytest

from silk import Port, load_pdk, sources
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.faults import FaultCampaign, FaultSet, stuck_open
from silk.optimize import Optimizer, VariableSpace
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk

pytestmark = [needs_pdk, needs_ngspice]


def _named_drain_current(ckt):
    """Module-level (picklable) counterpart to ``first``'s lambda below."""
    return -float(ckt.load_signal('vdrain', 'I').y[-1])


@pytest.fixture(scope='module')
def pdk():
    return load_pdk('sky130')


def _dual_test_dut(tmp_dir, pdk, name, *, w=2.0):
    """Two tests on one DUT: ``first`` (lambda metric) then ``second``.

    ``second.add_module_under_test()`` runs AFTER ``first.add_spec()`` has
    already appended to the shared DUT's ``_specs`` -- so ``second``'s
    embedded DUT copy carries a replica of ``first``'s Spec, whose
    ``_owning_testbench`` is ``first`` (holding the lambda). Any payload
    built from ``second`` (its own netlist, or a module that embeds it)
    reaches that lambda unless the fix cuts the reference.
    """
    dut = Module('sky130', tmp_dir, f'{name}.cir', pdk_details=pdk,
                 module_name='dual_' + name)
    dut.define_module_ports({'D': Port.analog_output(), 'G': Port.analog_input(),
                             'S': Port.ground()})
    with dut:
        pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=w, l=0.5)

    first = dut.create_test('first')
    first.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
    first.add_source(sources.VoltageSource('drain', positive='drain', negative='0', dc=0.8))
    first.add_source(sources.VoltageSource('gate', positive='gate', negative='0', dc=1.0))
    first.add_sim_type(Transient(step=1e-9, stop=5e-9))
    first.add_metric('drain_current_A',
                     lambda ckt: -float(ckt.load_signal('vdrain', 'I').y[-1]))
    first.add_spec('first_conducts', 'drain_current_A', '>', 0.0)

    second = dut.create_test('second')
    second.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
    second.add_source(sources.VoltageSource('drain', positive='drain', negative='0', dc=0.8))
    second.add_source(sources.VoltageSource('gate', positive='gate', negative='0', dc=1.0))
    second.add_sim_type(Transient(step=1e-9, stop=5e-9))
    second.add_metric('drain_current_A', _named_drain_current)
    second.add_spec('second_conducts', 'drain_current_A', '>', 0.0)
    return dut


class TestRunAllTests:
    def test_n_workers_2_matches_serial(self, tmp_path, pdk):
        serial = _dual_test_dut(str(tmp_path / 'serial'), pdk, 'a')
        parallel = _dual_test_dut(str(tmp_path / 'parallel'), pdk, 'b')

        serial_results = serial.run_all_tests(corner='tt', n_workers=1)
        parallel_results = parallel.run_all_tests(corner='tt', n_workers=2)

        assert {k.split('.', 1)[1] for k in serial_results} == \
            {k.split('.', 1)[1] for k in parallel_results} == {'first', 'second'}
        for s_key, p_key in zip(sorted(serial_results), sorted(parallel_results)):
            s, p = serial_results[s_key], parallel_results[p_key]
            assert not isinstance(s, dict), s
            assert not isinstance(p, dict), p
            assert p.metrics == pytest.approx(s.metrics)
            assert p.results[0].passed == s.results[0].passed


class TestRunCorners:
    def test_second_test_n_workers_2_matches_serial(self, tmp_path, pdk):
        """``second`` is the one whose netlist reaches ``first``'s lambda."""
        serial_dut = _dual_test_dut(str(tmp_path / 'serial'), pdk, 'a')
        parallel_dut = _dual_test_dut(str(tmp_path / 'parallel'), pdk, 'b')

        serial = serial_dut._tests['second'].run_corners(
            corners=['tt', 'ff'], n_workers=1)
        parallel = parallel_dut._tests['second'].run_corners(
            corners=['tt', 'ff'], n_workers=2)

        assert set(serial) == set(parallel) == {'tt', 'ff'}
        for corner in serial:
            assert parallel[corner].metrics == pytest.approx(serial[corner].metrics)


class TestOptimizerSweep:
    def test_sweep_on_second_test_n_workers_2_matches_serial(self, tmp_path, pdk):
        w_serial = Variable('w1', value=2.0, bounds=(1.0, 4.0))
        serial_dut = _dual_test_dut(str(tmp_path / 'serial'), pdk, 'a', w=w_serial)
        serial_opt = Optimizer(serial_dut._tests['second'], VariableSpace([w_serial]))
        serial_opt.add_objective('drain_current_A', minimize=True)
        serial_opt.sweep(n=4, method='grid', corner='tt', n_workers=1)

        w_parallel = Variable('w1', value=2.0, bounds=(1.0, 4.0))
        parallel_dut = _dual_test_dut(str(tmp_path / 'parallel'), pdk, 'b', w=w_parallel)
        parallel_opt = Optimizer(parallel_dut._tests['second'], VariableSpace([w_parallel]))
        parallel_opt.add_objective('drain_current_A', minimize=True)
        parallel_opt.sweep(n=4, method='grid', corner='tt', n_workers=2)

        assert len(serial_opt._history) == len(parallel_opt._history) == 4
        for s, p in zip(serial_opt._history, parallel_opt._history):
            assert s['status'] == p['status'] == 'success'
            assert p['params'] == s['params']
            assert p['metrics'] == pytest.approx(s['metrics'])
        # w1 sits inside the embedded DUT. Until TASK-400 the DUT held a
        # deep copy of it, every point simulated the same width, and this
        # comparison passed whatever points the two sweeps sampled. A
        # deterministic grid gives both sweeps the same points; distinct
        # currents prove the width reached the deck.
        currents = [r['metrics']['drain_current_A'] for r in serial_opt._history]
        assert len(set(currents)) == 4, currents


class TestFaultCampaign:
    """``FaultCampaign`` pickles the WHOLE module (``silk/faults/campaign.py``:
    ``pickle.dumps(self.module)``). Using ``second``'s own netlist directly
    as the fault target puts the sibling-lambda reference right where
    ``FaultCampaign`` would pickle it."""

    def test_campaign_n_workers_2_matches_serial(self, tmp_path, pdk):
        fault_set = FaultSet([stuck_open('xdut.m1')])

        serial_dir = tmp_path / 'serial'
        serial_dut = _dual_test_dut(str(serial_dir), pdk, 'a')
        serial_target = serial_dut._tests['second'].netlist
        serial_campaign = FaultCampaign(
            serial_target, fault_set, n_runs=2, seed=0,
            out_dir=str(serial_dir / 'camp'), n_workers=1,
            archive_waveforms=False, reconcile_specs=False, progress=False)
        serial_result = serial_campaign.run()

        parallel_dir = tmp_path / 'parallel'
        parallel_dut = _dual_test_dut(str(parallel_dir), pdk, 'b')
        parallel_target = parallel_dut._tests['second'].netlist
        parallel_campaign = FaultCampaign(
            parallel_target, fault_set, n_runs=2, seed=0,
            out_dir=str(parallel_dir / 'camp'), n_workers=2,
            archive_waveforms=False, reconcile_specs=False, progress=False)
        parallel_result = parallel_campaign.run()

        assert [r.status for r in serial_result.runs] == ['simulated', 'simulated']
        assert [r.status for r in parallel_result.runs] == ['simulated', 'simulated']
        for s, p in zip(serial_result.runs, parallel_result.runs):
            assert s.seed == p.seed
            assert [a.target for a in s.manifest.applied] == \
                [a.target for a in p.manifest.applied]


class TestReportedReproducer:
    """The exact reproducer from the task description, run in its own
    process so it cannot collide with any other example module's
    ``load_pdk('sky130')`` call elsewhere in the same pytest session (see
    module docstring)."""

    def test_current_mirror_example_run_all_tests_n_workers_2(self):
        script = os.path.join(os.path.dirname(__file__),
                              '_current_mirror_parallel_repro.py')
        import subprocess
        env = dict(os.environ)
        worktree_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env['PYTHONPATH'] = worktree_root + os.pathsep + env.get('PYTHONPATH', '')
        # This includes two serial runs before the parallel comparison.
        # With ngspice 45.2, sky130 include parsing brought the measured full
        # comparison to 334 s, with matching results. Allow hosted-runner margin.
        proc = subprocess.run([sys.executable, script], env=env,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, (
            f"reproducer failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
