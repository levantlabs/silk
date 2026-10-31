"""Live DUT references keep registration linear; explicit copies rebind probes.

TASK-205's growth bound remains a regression gate after TASK-458 removes
placement copies entirely.
"""
import copy
import types

import pytest

from silk import Module, load_pdk, sources
from silk.circuit.module_test import (
    ModuleTestbench, _metric_slot_leaves,
)
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.metrics.statistics import average
from silk.simulation import Transient
from tests._env import needs_pdk


_ATOMS = (str, bytes, int, float, bool, type(None))
_OPAQUE = (type, types.ModuleType, types.FunctionType,
           types.BuiltinFunctionType, types.MethodType)


def _reachable_benches(root, exclude):
    """Every ModuleTestbench reachable from *root* by full descent.

    Ids in *exclude* are neither counted nor descended into: the live DUT is
    excluded so its back-references cannot masquerade as routes out of the
    netlist. Classes, modules and functions are opaque — the route this
    guards runs through plain containers and instance dicts.
    """
    seen, stack, found = {id(root)}, [root], []
    while stack:
        obj = stack.pop()
        if isinstance(obj, ModuleTestbench):
            found.append(obj)
        if isinstance(obj, _ATOMS) or isinstance(obj, _OPAQUE):
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            children = list(obj)
        elif isinstance(obj, dict):
            children = list(obj.keys()) + list(obj.values())
        else:
            attrs = getattr(obj, '__dict__', None)
            children = list(attrs.values()) if isinstance(attrs, dict) else []
        for child in children:
            if id(child) in seen or id(child) in exclude:
                continue
            seen.add(id(child))
            stack.append(child)
    return found


@pytest.fixture
def pdk():
    return load_pdk('sky130')


def _block(tmp_path, pdk, name='dut'):
    """A one-resistor subcircuit that can be a DUT or a composed block."""
    block = Module('sky130', str(tmp_path), f'{name}.cir', pdk, module_name=name)
    block.define_module_ports({'out': PortDir.OUT, 'vss': PortDir.GROUND})
    block.add(elements.resistor('rload', p='out', n='vss', value=1000))
    return block


def _bench_with_spec(module, name):
    """A registered bench that declares one spec on *module*."""
    tb = module.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('supply', lambda ckt: 0.0)
    tb.add_spec(f'{name}_spec', 'supply', '>', 0.0)
    return tb


@needs_pdk
def test_registering_benches_is_linear_not_exponential(tmp_path, pdk):
    """Eight benches, each declaring a spec: the copy graph stays linear.

    Unpatched this sequence was ``3**(N-1) - 1`` exactly — 0, 2, 8, 26, 80,
    242, 728, 2186 — and the eighth bench alone allocated ~740 MB on this
    one-resistor circuit. The assertion is on the *increment* per bench,
    because that is what distinguishes linear from geometric no matter what
    constant the copy machinery carries.
    """
    dut = _block(tmp_path, pdk)
    counts = []
    for i in range(1, 9):
        tb = _bench_with_spec(dut, f'tb{i}')
        counts.append(len(_reachable_benches(tb.netlist, set())))

    increments = [b - a for a, b in zip(counts, counts[1:])]
    assert max(increments) <= 4, f'growth is not linear: {counts}'
    assert counts[-1] <= 4 * len(counts), f'growth is not linear: {counts}'
    # Every registered bench keeps its own working netlist and DUT copy.
    for name in dut.list_tests():
        bench = dut._tests[name]
        assert bench.netlist is not None
        assert 'dut' in bench.netlist.elem_dict
        assert 'rload' in bench.netlist.elem_dict['dut']['device'].elem_dict


def _bench_with_probe_metric(module, name):
    """A registered bench whose metric is a function-form probe expression.

    The metric slot holds an *expression* (``probe('out') * 2``), not a bare
    probe, so the growth walk and the rebinding walk both have to descend a
    recipe rather than find a leaf at the top.
    """
    tb = module.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    tb.add_metric('vavg', average, sig=tb.probe('out') * 2)
    # The spec is what opens the replica route: a DUT copy carries replicas of
    # the earlier benches through ``Spec._owning_testbench``, and that is where
    # a copied probe can be inspected.
    tb.add_spec(f'{name}_spec', 'vavg', '>', 0.0)
    return tb


@needs_pdk
def test_probe_bearing_benches_stay_linear_and_their_copies_rebind(tmp_path, pdk):
    """TASK-215.8: probes cost nothing to copy, and copies point at the copy.

    Two claims in one sequence, because they pull against each other. A probe
    must stay *weak* — a strong bench reference in a metric slot would drag a
    whole bench graph into every deep copy and re-open TASK-205 — and a copied
    probe must nevertheless end up pointing at the copy, or a composed block
    measures its parent's results (design §6). Deep-copying a weak reference
    is atomic, so the copy points at the ORIGINAL until ``__deepcopy__``
    rebinds it; that rebinding is what the second half asserts.

    The growth bound is the same increment test as the spec-bearing
    sequence above, for the same reason: the increment is what separates
    linear from geometric whatever constant the copy machinery carries.
    """
    dut = _block(tmp_path, pdk)
    counts = []
    for i in range(1, 9):
        tb = _bench_with_probe_metric(dut, f'ptb{i}')
        counts.append(len(_reachable_benches(tb.netlist, set())))

    increments = [b - a for a, b in zip(counts, counts[1:])]
    assert max(increments) <= 4, f'growth is not linear: {counts}'
    assert counts[-1] <= 4 * len(counts), f'growth is not linear: {counts}'

    # No bench is reachable from a probe at all: the reference is weak, so the
    # metric slots hold names and a weakref, never a bench object.
    for name in dut.list_tests():
        bench = dut._tests[name]
        leaves = list(_metric_slot_leaves(bench._metrics))
        assert leaves, f'{name} lost its probe'
        for leaf in leaves:
            for value in leaf.__dict__.values():
                assert not isinstance(value, (ModuleTestbench, Module))

    # Every copied bench's probes point at THAT copy. The DUT copy embedded in
    # a later bench's netlist carries replicas of the earlier benches through
    # its specs; those are orphans (no netlist, never run), and their probes
    # follow them rather than staying on the template.
    last = dut._tests['ptb8']
    copied_dut = copy.deepcopy(dut)
    replicas = _reachable_benches(copied_dut, set())
    assert replicas, 'no bench replica reached through the DUT copy'
    for replica in replicas:
        assert replica is not dut._tests.get(replica.test_name)
        leaves = list(_metric_slot_leaves(replica._metrics))
        assert leaves, f'replica of {replica.test_name} lost its probe'
        for leaf in leaves:
            assert leaf._bench_or_none() is replica

    # The direct route the TASK-205 test uses: the replica reached through the
    # DUT copy's specs is an orphan with no netlist, and asking its probe
    # whether it is bound is still safe.
    dut_copy = copied_dut
    orphan = dut_copy._specs[0]._owning_testbench
    assert orphan.netlist is not None
    for leaf in _metric_slot_leaves(orphan._metrics):
        assert leaf._bench_or_none() is orphan
        assert leaf.bound is False


@needs_pdk
def test_placements_share_registered_benches_and_explicit_copy_is_independent(tmp_path, pdk):
    dut = _block(tmp_path, pdk)
    first = _bench_with_spec(dut, 'first')
    second = _bench_with_spec(dut, 'second')
    assert second.netlist.elem_dict['dut']['device'] is dut
    assert dut._specs[0]._owning_testbench is first
    parent = Module('sky130', str(tmp_path), 'parent.cir', pdk, module_name='parent')
    parent.add_module(dut, 'blk', nets={'out': 'out', 'vss': '0'})
    assert parent.elem_dict['blk']['device'] is dut
    independent = copy.deepcopy(parent)
    block_copy = independent.elem_dict['blk']['device']
    composed = block_copy._tests['first']
    assert block_copy is not dut
    assert composed is not first and composed.module is block_copy
    assert composed.netlist.elem_dict['dut']['device'] is block_copy
    assert block_copy._specs[0]._owning_testbench is composed


# ---------------------------------------------------------------------------
# TASK-215.9: tb.plot()'s no-argument probe registry
# ---------------------------------------------------------------------------

@needs_pdk
def test_probe_registry_stays_linear_and_rebinds_on_copy(tmp_path, pdk):
    """``_probe_registry`` (``tb.plot()`` with no arguments) costs nothing
    extra and follows copies the same way the metric slots do.

    ``probe()`` appends a ``weakref.ref`` to the bench's own registry so
    ``tb.plot()`` has something to draw with no arguments — but a
    ``weakref.ref`` deep-copies atomically (same object, pointing at the
    ORIGINAL probe), and cannot be pickled at all, so the registry cannot
    simply travel with the rest of ``__dict__`` the way ``_metrics`` does.
    It is instead REBUILT after a copy or an unpickle, from the same
    (already rebound) metric slots ``_metric_slot_leaves`` walks — see
    ``_registry_from_metric_leaves`` in ``silk/circuit/module_test.py``.

    Two claims, mirroring the metric-bearing sequence above: the growth
    stays linear (the registry itself is never part of the reachable-bench
    walk — a bare ``weakref.ref`` exposes no ``__dict__`` for it to descend
    into), and every bench's registry — including a composed replica's —
    lists exactly the probes reachable from ITS OWN metrics, each bound to
    that same bench.
    """
    dut = _block(tmp_path, pdk)
    counts = []
    for i in range(1, 9):
        tb = _bench_with_probe_metric(dut, f'rtb{i}')
        counts.append(len(_reachable_benches(tb.netlist, set())))

    increments = [b - a for a, b in zip(counts, counts[1:])]
    assert max(increments) <= 4, f'growth is not linear: {counts}'
    assert counts[-1] <= 4 * len(counts), f'growth is not linear: {counts}'

    for name in dut.list_tests():
        bench = dut._tests[name]
        registry_probes = bench._live_probes()
        metric_leaves = list(_metric_slot_leaves(bench._metrics))
        assert registry_probes, f'{name} registry lost its probe'
        assert {id(p) for p in registry_probes} == {id(p) for p in metric_leaves}
        for probe in registry_probes:
            assert probe._bench_or_none() is bench

    # A composed replica's registry is rebuilt from ITS OWN rebound metric
    # slots, not shared with (or still pointing at) the template.
    last = dut._tests['rtb8']
    copied_dut = copy.deepcopy(dut)
    replicas = _reachable_benches(copied_dut, set())
    assert replicas, 'no bench replica reached through the DUT copy'
    for replica in replicas:
        registry_probes = replica._live_probes()
        metric_leaves = list(_metric_slot_leaves(replica._metrics))
        assert registry_probes, f'replica of {replica.test_name} lost its probe'
        assert {id(p) for p in registry_probes} == {id(p) for p in metric_leaves}
        for probe in registry_probes:
            assert probe._bench_or_none() is replica


@needs_pdk
def test_probe_registry_survives_pickle_round_trip(tmp_path, pdk):
    """A pickled-and-restored bench's registry matches its rebound metrics.

    ``__getstate__`` empties ``_probe_registry`` (a raw ``weakref.ref``
    cannot pickle); ``__setstate__`` rebuilds it the same way
    ``__deepcopy__`` does, from the metric slots pickling detached and just
    re-attached. A bench whose metrics are plain named functions pickles
    today (TASK-215.8) — this only adds the registry to that contract.
    """
    import pickle

    dut = _block(tmp_path, pdk)
    tb = _bench_with_probe_metric(dut, 'pkl')
    before = list(_metric_slot_leaves(tb._metrics))
    assert before

    restored = pickle.loads(pickle.dumps(tb))
    registry_probes = restored._live_probes()
    metric_leaves = list(_metric_slot_leaves(restored._metrics))
    assert registry_probes
    assert {id(p) for p in registry_probes} == {id(p) for p in metric_leaves}
    for probe in registry_probes:
        assert probe._bench_or_none() is restored


def test_simulation_snapshot_omits_bench_graph_but_keeps_live_model(tmp_path):
    from silk import Module
    from silk.simulation._worker_pickle import copy_simulation_netlist

    class MustNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError('simulation snapshot reached a registered bench')

    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({})
    def model_1(ports, t, params):
        return
    model = model_1
    child.set_behavioral_model(model, execution='vector')
    bench = child.create_test('measure')
    bench.add_module_under_test({})
    bench._snapshot_tripwire = MustNotCopy()
    copied = copy_simulation_netlist(bench.netlist)
    copied_child = next(copied.iter_instances()).module
    assert copied_child is not child
    assert copied_child.list_tests() == []
    assert copied_child._behavioral_model is model
    assert child.list_tests() == ['measure']
