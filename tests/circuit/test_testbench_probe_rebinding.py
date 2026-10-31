"""A copied or restored bench's probes point at THAT bench (TASK-215.8).

A probe holds a *weak* reference to its testbench. That is what keeps a
metric slot from dragging a whole bench graph into every deep copy — the
TASK-205 blow-up — but it is also why copying one is not enough: deep-copying
a weak reference is atomic, so the copy still points at the ORIGINAL bench.
Explicitly copied benches must therefore rebind their probes. Placement
keeps a live bench instead (TASK-458), and its selected handle routes readback to retained run evidence.

Pickling has the mirror-image problem (§3.0). A weak reference cannot be
pickled at all, so ``Signal.__getstate__`` (TASK-215.2) drops it and the
restored probe is *detached*. Ordinary Module pickles retain registered testbenches, so a bench that comes back with dead probes is a bench
whose metrics no longer run.

Both are closed by the same walk over the metric slot values:
``ModuleTestbench.__deepcopy__`` re-points the leaves that still name the
bench being copied, and ``ModuleTestbench.__setstate__`` re-points the ones
pickling detached.

The fast legs stand the simulator in with the committed TASK-74 rawfile
fixture (real ngspice-42 output) exactly as ``tests/test_metric_binding_errors``
and ``tests/test_testbench_evaluate`` do, so ``run()`` exercises its real
tail on real simulator output without launching anything.
"""
from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from silk import Module, load_pdk, sources
from silk.circuit.module_test import ModuleTestbench, _metric_slot_leaves
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.metrics.statistics import average
from silk.simulation import Transient
from tests._env import needs_pdk
from tests._temperature import mark_simulated


FIXTURE = (Path(__file__).resolve().parents[1]
           / 'fixtures' / 'rawfiles' / 'e_ng_tran.raw')


@pytest.fixture
def pdk():
    return load_pdk('sky130')


def _block(tmp_path, pdk, name='blk'):
    """A one-resistor subcircuit that can be composed into a parent."""
    block = Module('sky130', str(tmp_path), f'{name}.cir', pdk, module_name=name)
    block.define_module_ports({'out': PortDir.OUT, 'vss': PortDir.GROUND})
    block.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return block


def _probe_bench(module, name='measure', signal='out'):
    """A bench whose only metric is bound to a probe of its own netlist."""
    tb = module.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vsup', positive='out', negative='0',
                                        dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('vavg', average, sig=tb.probe(signal))
    return tb


def _load_fixture(netlist, corner='tt'):
    """Put the committed rawfile on a netlist as a finished run would."""
    netlist.load_file(str(FIXTURE))
    netlist.read_raw_file()
    netlist.results_file = str(FIXTURE)
    mark_simulated(netlist, corner, voltage=1.8, supply_source='vsup')
    return netlist


def _stub(bench):
    """Stand ``bench.netlist.simulate`` in for the simulator."""
    def fake_simulate(corner=None, **_):
        return _load_fixture(bench.netlist, corner)
    return patch.object(bench.netlist, 'simulate', side_effect=fake_simulate)


def _leaves(bench):
    return list(_metric_slot_leaves(bench._metrics))


# ---------------------------------------------------------------------------
# AC #2 — a composed block measures its own results
# ---------------------------------------------------------------------------

class TestCompositionRebinding:

    @needs_pdk
    def test_placement_preserves_the_live_bench_and_probe(self, tmp_path, pdk):
        block = _block(tmp_path, pdk)
        template = _probe_bench(block)
        parent = Module('sky130', str(tmp_path), 'parent.cir', pdk,
                        module_name='parent')
        parent.add_module(block, 'inst', nets={'out': 'out', 'vss': '0'})
        entry = parent.list_test_entries()[0]
        assert entry.test is parent.get_test('measure', instance='inst')
        assert entry.test.netlist is template.netlist
        leaves = _leaves(template)
        assert leaves
        assert all(leaf._bench_or_none() is template for leaf in leaves)

    @needs_pdk
    def test_recursive_run_reads_results_with_original_probe(self, tmp_path, pdk):
        block = _block(tmp_path, pdk)
        template = _probe_bench(block)
        original = template.netlist
        parent = Module('sky130', str(tmp_path), 'parent.cir', pdk,
                        module_name='parent')
        parent.add_module(block, 'inst', nets={'out': 'out', 'vss': '0'})
        entry = parent.list_test_entries()[0]
        with _stub(template):
            results = parent.run_all_tests(recursive=True)
        report = results[entry.report_key]
        assert not isinstance(report, dict), report
        expected = average(original.load_signal('out'))
        assert report.metrics['vavg'] == pytest.approx(expected, rel=1e-12)
        assert template.netlist is original

    @needs_pdk
    def test_explicit_copy_extractor_reads_its_own_slots(self, tmp_path, pdk):
        """User-requested copies still rebuild the extractor closure."""
        template = _probe_bench(_block(tmp_path, pdk))
        copied = copy.deepcopy(template)
        copied_metric = copied._metrics[0]
        extractor = copied_metric['extractor']
        assert extractor is not template._metrics[0]['extractor']
        assert extractor.__defaults__[0] is copied_metric
        assert all(leaf._bench_or_none() is copied for leaf in _leaves(copied))

    @needs_pdk
    def test_dut_spec_keeps_its_original_bench_without_orphan_replicas(self, tmp_path, pdk):
        block = _block(tmp_path, pdk)
        first = _probe_bench(block, 'first')
        first.add_spec('first_spec', 'vavg', '>', 0.0)
        second = _probe_bench(block, 'second')
        dut = next(second.netlist.iter_instances()).module
        assert dut is block
        owner = dut._specs[0]._owning_testbench
        assert owner is first
        assert owner.netlist is not None
        leaves = _leaves(owner)
        assert leaves
        for leaf in leaves:
            assert leaf._bench_or_none() is first
            assert leaf.bound is False


# ---------------------------------------------------------------------------
# AC #3 — pickle
# ---------------------------------------------------------------------------

class TestPickleRebinding:

    @needs_pdk
    def test_probe_bearing_bench_round_trips_and_rebinds(self, tmp_path, pdk):
        block = _block(tmp_path, pdk)
        tb = _probe_bench(block)

        restored = pickle.loads(pickle.dumps(tb))

        assert restored is not tb
        leaves = _leaves(restored)
        assert leaves, 'the restored bench lost its probe'
        for leaf in leaves:
            assert leaf._bench_or_none() is restored
        # The dropped closure is back and the dict shape is unchanged.
        metric = restored._metrics[0]
        assert sorted(metric) == ['extractor', 'fn', 'name', 'params', 'slots']
        assert callable(metric['extractor'])
        assert metric['fn'] is average

    @needs_pdk
    def test_pickling_does_not_disturb_the_original(self, tmp_path, pdk):
        block = _block(tmp_path, pdk)
        tb = _probe_bench(block)
        before = _leaves(tb)

        pickle.dumps(tb)

        for leaf in before:
            assert leaf._bench_or_none() is tb
        assert callable(tb._metrics[0]['extractor'])

    @needs_pdk
    def test_module_round_trips_the_fault_campaign_way(self, tmp_path, pdk):
        """The exact shape ``FaultCampaign`` uses.

        ``silk/faults/campaign.py`` does ``pickle.dumps(self.module)`` and
        ``silk/faults/runner.py`` does ``pickle.loads`` in the worker, so a
        Module carrying a probe-bearing bench has to survive that pair.
        """
        block = _block(tmp_path, pdk)
        _probe_bench(block)

        module_pickle = pickle.dumps(block)          # campaign.py
        restored_module = pickle.loads(module_pickle)  # runner.py

        restored = restored_module._tests['measure']
        leaves = _leaves(restored)
        assert leaves
        for leaf in leaves:
            assert leaf._bench_or_none() is restored

    @needs_pdk
    def test_restored_bench_runs_and_its_metric_resolves(self, tmp_path, pdk):
        """A restored bench is not merely intact, it works."""
        block = _block(tmp_path, pdk)
        _probe_bench(block)
        restored_module = pickle.loads(pickle.dumps(block))
        restored_module.dir_folder = str(tmp_path)
        restored = restored_module._tests['measure']

        with _stub(restored):
            report = restored.run('tt')

        expected = average(restored.netlist.load_signal('out'))
        assert report.metrics['vavg'] == pytest.approx(expected, rel=1e-12)

    @needs_pdk
    def test_named_function_metrics_still_pickle(self, tmp_path, pdk):
        """The control: what pickled before this task still does."""
        block = _block(tmp_path, pdk)
        tb = block.create_test('named')
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
        tb.add_metric('n', _named_extractor)

        restored = pickle.loads(pickle.dumps(tb))
        assert restored._metrics[0]['extractor'] is _named_extractor
        assert sorted(restored._metrics[0]) == ['extractor', 'name']

    @needs_pdk
    def test_declarative_metrics_still_refuse_to_pickle(self, tmp_path, pdk):
        """Pinned, not fixed: the declarative form never pickled.

        Its extractor is a closure over a binding the metric dict does not
        record, so there is nothing to rebuild it from. This task did not
        change that, and this test is here so a later change cannot happen
        silently.
        """
        block = _block(tmp_path, pdk)
        tb = block.create_test('declared')
        tb.add_module_under_test({'out': 'out', 'vss': '0'})
        tb.add_metric('g', type='average', signal='out')

        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(tb)

    @needs_pdk
    def test_a_probe_pickled_alone_stays_detached(self, tmp_path, pdk):
        """Only probes a bench can reach are rebound (design §6).

        A probe pickled on its own has no bench to be restored beside it, so
        TASK-215.2's detached state stands.
        """
        from silk.exceptions import UnboundSignalError
        block = _block(tmp_path, pdk)
        tb = _probe_bench(block)

        loose = pickle.loads(pickle.dumps(tb.probe('out')))
        assert loose.bound is False
        with pytest.raises(UnboundSignalError, match='detached'):
            loose.y
        assert _leaves(tb)[0]._bench_or_none() is tb


def _named_extractor(ckt):
    """A module-level extractor — picklable by reference."""
    return 1.0


# ---------------------------------------------------------------------------
# Shallow copy: the one path that must NOT rebind
# ---------------------------------------------------------------------------

@needs_pdk
def test_shallow_copy_leaves_the_originals_probes_alone(tmp_path, pdk):
    """``copy.copy`` routes through the same getstate/setstate pair.

    A shallow copy shares the probe objects with the original, so rebinding
    them would re-point the ORIGINAL bench's probes at the copy. Selecting on
    the detached marker rather than on 'has no live reference' is what keeps
    that from happening.
    """
    import copy as _copy

    block = _block(tmp_path, pdk)
    tb = _probe_bench(block)
    shallow = _copy.copy(tb)

    assert isinstance(shallow, ModuleTestbench)
    for leaf in _leaves(tb):
        assert leaf._bench_or_none() is tb
