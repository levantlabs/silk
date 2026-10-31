"""PLL spec-tree example: u_pll owns the system specs, the normal testbench
path measures them, and one reconcile on the top still covers the tree.

Block SPICE testbenches are not run here (no simulator needed): the block
specs stay unmeasured, which is enough to check ownership, links and evidence.
"""
import pytest

from tests._env import needs_pdk
from silk.examples import pll_spec_tree_example as ex
from silk.examples.pll_example import (
    LOCKTIME_METRIC, FERR_METRIC, VCTRL_METRIC, PHERR_METRIC)

pytestmark = needs_pdk

_MEASURED = (LOCKTIME_METRIC, FERR_METRIC, VCTRL_METRIC, PHERR_METRIC)


@pytest.fixture(scope="module")
def tree():
    top = ex.build_pll_spec_tree()
    pll = ex._live(top, "u_pll")
    tb = ex.attach_lock_test(pll)
    system = ex.declare_system_specs(top, tb)
    placed = top.get_test(pll.list_tests()[-1], instance="u_pll")
    res = ex.run_lock(pll, testbench=placed)
    return top, pll, tb, system, res


def test_specs_owned_by_live_pll_not_top(tree):
    top, pll, tb, system, _ = tree
    assert top._specs == []
    assert sorted(s.metric for s in pll._specs) == sorted(
        _MEASURED + (ex.JITTER_METRIC,))
    assert all(s.module is pll for s in system.values())
    assert len(pll.list_tests()) == 1


def test_lock_test_records_through_normal_path(tree):
    _, pll, _, system, _ = tree
    for metric in _MEASURED:
        ms = list(system[metric].measurements.values())
        assert len(ms) == 1, metric
        assert ms[0].source == "behavioral" and ms[0].value is not None
    assert len(system[ex.JITTER_METRIC].measurements) == 0
    assert not hasattr(ex, "record_behavioral")


def test_links_preserved(tree):
    top, _, _, system, _ = tree
    assert len(system[ex.FERR_METRIC].children) == 2
    assert len(system[ex.VCTRL_METRIC].children) == 1
    assert len(system[ex.PHERR_METRIC].children) == 2
    assert system[ex.LOCKTIME_METRIC].children == []
    for sp in system.values():
        for child in sp.children:
            assert child.parent is sp
            assert child.module is not ex._live(top, "u_pll")


def test_reconcile_once_covers_tree_and_jitter_unmeasured(tree):
    top, pll, _, system, _ = tree
    recon = top.reconcile_specs()
    specs = [o.spec for o in recon.outcomes]
    for sp in system.values():
        assert sum(s is sp for s in specs) == 1
    by = {id(o.spec): o for o in recon.outcomes}
    assert by[id(system[ex.JITTER_METRIC])].status == "UNMEASURED"
    for metric in _MEASURED:
        assert by[id(system[metric])].status in ("PASS", "FAIL")
        assert by[id(system[metric])].measurement.source == "behavioral"
    assert not recon.passed
    block = [s for s in specs if s.module is not pll]
    assert block, "block specs must be in the same reconciliation"
    # a system verdict is the PLL's own measurement, never derived from children
    assert all(by[id(s)].status == "UNMEASURED" for s in block)


def test_printer_finds_system_specs_on_pll(tree, capsys):
    top, pll, _, system, _ = tree
    ex.print_two_level_summary(top, top.reconcile_specs(), pll)
    out = capsys.readouterr().out
    assert "SYSTEM specs on 'pll'" in out
    assert "[UNMEASURED] output period jitter" in out
    assert "ConditionState" not in out
    assert "(behavioral)" in out


def test_failed_lock_attempt_is_attributed_to_the_pll_placement():
    from silk.exceptions import SimulationError
    top = ex.build_pll_spec_tree()
    pll = ex._live(top, "u_pll")
    tb = ex.attach_lock_test(pll)
    system = ex.declare_system_specs(top, tb)

    def broken(*args, **kwargs):
        raise SimulationError("lock model failed")
    pll.set_behavioral_model(broken, params={})
    placed = top.get_test(pll.list_tests()[-1], instance="u_pll")
    with pytest.raises(SimulationError):
        placed.run()
    rows = {o.spec.metric: o for o in top.reconcile_specs().outcomes}
    for metric in _MEASURED:
        o = rows[metric]
        assert o.status == "FAIL" and o.measurement is None
        assert o.owner_path == "$/u_pll"
        assert "lock model failed" in o.latest_attempt.reason
    assert rows[ex.JITTER_METRIC].status == "UNMEASURED"


def test_orchestration_registers_after_block_tree_and_runs_lock_once(monkeypatch):
    calls = []
    top_holder = {}
    real_build = ex.build_pll_spec_tree

    def build(**kw):
        top = real_build(**kw)
        pll = ex._live(top, "u_pll")
        top_holder["pll"] = pll

        def walk(*a, **k):
            calls.append(("walk", len(pll.list_tests())))
            return {}
        top.run_all_tests = walk
        return top

    real_attach, real_run = ex.attach_lock_test, ex.run_lock
    monkeypatch.setattr(ex, "build_pll_spec_tree", build)
    monkeypatch.setattr(ex, "attach_lock_test",
                        lambda m, **k: (calls.append(("attach",)), real_attach(m, **k))[1])
    monkeypatch.setattr(ex, "run_lock",
                        lambda m, **k: (calls.append(("lock",)), real_run(m, **k))[1])
    out = ex.run_pll_spec_tree(verbose=False)
    assert calls == [("walk", 0), ("attach",), ("lock",)]
    assert len(top_holder["pll"].list_tests()) == 1
    assert out["top"]._specs == []


@pytest.mark.parametrize("instance", ["missing", "u_pll.controller"])
def test_live_lookup_keeps_missing_instance_keyerror(instance):
    top = ex.build_pll_spec_tree()
    with pytest.raises(KeyError) as error:
        ex._live(top, instance)
    assert error.value.args == (instance,)
