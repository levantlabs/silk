"""Reconcile PLL system specs and block specs in one module tree.

PFD, charge pump, LDO, ring and /24 divider testbenches measure their blocks
in SPICE. A behavioral PLL in the same tree owns four system specs,
measured by its lock test. Parent/child links explain intended support;
each spec is measured independently, and block verdicts do not determine
system verdicts.

A fifth system spec, output period jitter, has no measurement. It appears
as UNMEASURED and keeps the overall result from passing. Block failures,
including the LDO terminal-voltage check, also remain visible.

The top module holds the tree; it is not simulated as a complete circuit.
Block tests run first. The PLL lock test is then registered and run through
its placement handle so the top-level report finds its evidence.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.pll_spec_tree_example

The script prints system measurements, linked block outcomes, unlinked
block outcomes and execution times.
"""
import time

from silk import load_pdk
from silk.elements import elements
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.specs import Spec
from silk.examples import example_output_dir

from silk.examples.pfd_example import (build_pfd, build_divider24,
                                       PULSE_METRIC, DIV_METRIC)
from silk.examples.charge_pump_example import build_charge_pump, MATCH_METRIC
from silk.examples.ldo_regulator_example import build_ldo, BEH_VREGMAX_METRIC
from silk.examples.ring_oscillator_example import build_ring_oscillator
from silk.examples.pll_example import (
    build_pll, design_loop_filter, attach_lock_test, add_lock_specs, run_lock, F_REF, N_DIV, F_TARGET,
    LOCKTIME_METRIC, FERR_METRIC, VCTRL_METRIC, PHERR_METRIC)

pdk = load_pdk('sky130')


JITTER_METRIC = "out_period_jitter_ps"
JITTER_MAX_PS = 10.0

_INSTANCES = ("u_pfd", "u_cp", "u_ldo", "u_ring", "u_div24", "u_pll")


def build_pll_spec_tree(*, filt=None) -> Module:
    """Build a container holding characterized blocks and the behavioral PLL."""
    filt = filt or design_loop_filter(verbose=False)
    top = Module("PDK_skywater130", example_output_dir(), "pll_spec_tree.cir",
                 pdk_details=pdk, module_name="pll_spec_tree")
    top.define_module_ports({
        "VDD33": Port.supply(), "VDD18": Port.supply(), "VSS": Port.ground(),
        "REF": Port.analog_input(), "EN": Port.analog_input(),
        "RSTB": Port.analog_input(),
        "IBIAS_AMP": Port.analog_input(), "IBIAS_CP": Port.analog_input(),
        "OUT": Port.analog_output(), "FB": Port.analog_output(),
        "VCTRL": Port.analog_output(), "VREG": Port.analog_output(),
    })

    top.add_module(build_pfd(with_tests=True), "u_pfd", nets={
        "VDD": "VDD18", "VSS": "VSS", "REF": "REF", "FB": "FB",
        "UP": "up", "DN": "dn"})
    top.add_module(build_charge_pump(with_tests=True), "u_cp", nets={
        "VDD": "VDD18", "VSS": "VSS", "IBIAS": "IBIAS_CP",
        "UP": "up", "DN": "dn", "OUT": "VCTRL"})
    with top:
        elements.resistor("RF", p="VCTRL", n="nc1",
                          value=filt["r_ohm"])
        elements.capacitor("C1", p="nc1", n="VSS", value=filt["c1_f"])
        elements.capacitor("C2", p="VCTRL", n="VSS", value=filt["c2_f"])
    top.add_module(build_ldo(with_tests=True), "u_ldo", nets={
        "VDD": "VDD33", "VSS": "VSS", "VCTRL": "VCTRL", "VREG": "VREG",
        "IBIAS_AMP": "IBIAS_AMP"})
    top.add_module(build_ring_oscillator(with_tests=True), "u_ring", nets={
        "VDD": "VREG", "VSS": "VSS", "EN": "EN", "OUT": "OUT"})
    top.add_module(build_divider24(with_tests=True), "u_div24", nets={
        "VDD": "VDD18", "VSS": "VSS", "CLK": "OUT", "RSTB": "RSTB",
        "OUT": "FB"})

    top.add_module(build_pll(filt=filt), "u_pll", nets={
        "REF": "REF", "OUT": "pll_out", "VCTRL": "pll_vctrl",
        "VREG": "pll_vreg", "PHERR": "pll_pherr"})
    return top


def _live(top: Module, instance: str) -> Module:
    """The module placed under ``instance`` in the tree."""
    for placed in top.iter_instances():
        if (placed.placed and placed.owner_path.count("/") == 1
                and placed.path == instance):
            return placed.module
    raise KeyError(instance)


def _block_spec(module: Module, metric: str, op: str = None) -> Spec:
    """Find the one spec a block testbench declared on ``metric`` (and op)."""
    hits = [s for s in module._specs
            if s.metric == metric and (op is None or s.op == op)]
    if len(hits) != 1:
        raise LookupError(
            f"{module.module_name}: expected exactly one spec on metric "
            f"{metric!r}{'' if op is None else ' op ' + op}, found {len(hits)}")
    return hits[0]


def _link(child: Spec, parent: Spec) -> None:
    """Link an existing block spec to a system spec through parent and children."""
    child.parent = parent
    if child not in parent.children:
        parent.children.append(child)


def declare_system_specs(top: Module, tb) -> dict:
    """Declare the four lock specs and an unmeasured jitter spec on u_pll.

    Link supporting block specs and return a dict indexed by metric name.
    """
    pfd, cp, ldo, div = (_live(top, n) for n in ("u_pfd", "u_cp", "u_ldo", "u_div24"))
    pll = _live(top, "u_pll")
    system = add_lock_specs(tb)

    system[LOCKTIME_METRIC].description = (
        "cold start from a dead ring; no block spec bounds the pull-in, so "
        "this spec has no children")

    s_ferr = system[FERR_METRIC]
    s_ferr.description = (
        f"f_out = {F_REF/1e6:.0f} MHz x {N_DIV} = "
        f"{F_TARGET/1e6:.0f} MHz; supported by the divider ratio")
    _link(_block_spec(div, DIV_METRIC, ">"), s_ferr)
    _link(_block_spec(div, DIV_METRIC, "<"), s_ferr)

    s_vctrl = system[VCTRL_METRIC]
    s_vctrl.description = (
        "the ring is dead below 1.0 V (a calibrated constant, vdd_min, not a "
        "spec on the ring); the LDO must be able to deliver the ring's full "
        "supply")
    _link(_block_spec(ldo, BEH_VREGMAX_METRIC), s_vctrl)

    s_pherr = system[PHERR_METRIC]
    s_pherr.description = (
        "charge balance over the CP mismatch during the PFD reset pulse sets "
        "the offset the loop must hold")
    _link(_block_spec(cp, MATCH_METRIC), s_pherr)
    _link(_block_spec(pfd, PULSE_METRIC), s_pherr)

    system[JITTER_METRIC] = pll.declare_spec(
        f"output period jitter < {JITTER_MAX_PS:.0f} ps", JITTER_METRIC, "<",
        JITTER_MAX_PS, units="ps",
        description="No measurement: the twin models no device noise "
                    "and no testbench records this metric")
    return system


def _fmt(o) -> str:
    """One outcome, the way SpecReconciliationReport.summary() prints it."""
    if o.measurement is None:
        return f"[{o.status}] {o.spec.name}: (no measurement)"
    val = f"{o.measurement.value:g}{o.spec.units}"
    tgt = f"{o.spec.op}{o.spec.target:g}{o.spec.units}"
    corner = o.measurement.corner
    src = o.measurement.source + (f"@{corner}" if isinstance(corner, str) and corner
                                  else "")
    return f"[{o.status}] {o.spec.name}: {val} {tgt}   ({src})"


def print_two_level_summary(top: Module, recon, system_owner: Module) -> None:
    """Print system outcomes with linked children, then unlinked block outcomes."""
    by_spec = {id(o.spec): o for o in recon.outcomes}
    print(f"Spec reconciliation — one reconcile_specs() on '{top.module_name}': "
          f"{recon.pass_count} PASS, {recon.fail_count} FAIL, "
          f"{len(recon.unmeasured)} UNMEASURED  ->  passed() = {recon.passed}")
    print()
    print(f"SYSTEM specs on '{system_owner.module_name}'  (children = the block specs "
          f"linked as support; a child's verdict is not rolled up)")
    claimed = set()
    for o in recon.outcomes:
        if o.spec.module is not system_owner:
            continue
        print(f"  {_fmt(o)}")
        if o.spec.description:
            print(f"      {o.spec.description}")
        if not o.spec.children:
            print("      (no children)")
        for child in o.spec.children:
            co = by_spec[id(child)]
            claimed.add(id(child))
            print(f"      +- {child.module.module_name:<12} {_fmt(co)}")
    print()
    print("BLOCK specs with no system parent  (measured by their own testbenches)")
    for o in recon.outcomes:
        if o.spec.module is system_owner or id(o.spec) in claimed:
            continue
        print(f"  {o.spec.module.module_name:<12} {_fmt(o)}")


def run_pll_spec_tree(*, filt=None, verbose=True) -> dict:
    """Build the tree, run block tests then the PLL lock test, reconcile once.

    Returns ``{'top', 'pll', 'tb', 'system', 'res', 'recon', 'spice_results'}``.
    """
    say = print if verbose else (lambda *a, **k: None)
    filt = filt or design_loop_filter(verbose=False)
    top = build_pll_spec_tree(filt=filt)
    pll = _live(top, "u_pll")

    say("\n  running the block testbenches through the tree (SPICE, tt)...")
    t0 = time.perf_counter()
    spice_results = top.run_all_tests(corner="tt")
    t_spice = time.perf_counter() - t0
    errors = [k for k, v in spice_results.items()
              if isinstance(v, dict) and "error" in v]
    if errors:
        raise RuntimeError(f"block testbenches failed: {errors}")

    # Register after the block tests so the tree walk does not run this twice.
    tb = attach_lock_test(pll)
    system = declare_system_specs(top, tb)
    say(f"  system specs owned by '{pll.module_name}': "
        f"{len(pll._specs)} ({sum(len(s.children) for s in pll._specs)} "
        f"parent links declared); top owns {len(top._specs)}")

    say("\n  cold-start lock transient (behavioral, 0.2 ns, 80 us)...")
    t0 = time.perf_counter()

    # Run through the placement handle so reconciliation attributes its evidence.
    placed = top.get_test(pll.list_tests()[-1], instance="u_pll")
    res = run_lock(pll, testbench=placed)
    t_beh = time.perf_counter() - t0
    say(f"    locked f   = {res['f_locked']/1e6:9.2f} MHz "
        f"({res['f_err_pct']:+.3f}%)")
    say(f"    lock time  = {res['lock_time_us']:9.1f} us")
    say(f"    VCTRL      = {res['v_ctrl_locked']:9.3f} V")
    say(f"    phase err  = {res['pherr_ps']:+9.1f} ps static")

    recon = top.reconcile_specs()
    return {"top": top, "pll": pll, "tb": tb, "system": system, "res": res,
            "recon": recon, "spice_results": spice_results,
            "t_spice": t_spice, "t_beh": t_beh}


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=" * 72)
    print(f"PLL spec tree — block specs and system specs, one reconcile "
          f"({F_REF/1e6:.0f} MHz x {N_DIV} = {F_TARGET/1e6:.0f} MHz)")
    print("=" * 72)
    print(f"  tree: {', '.join(_INSTANCES)}")

    out = run_pll_spec_tree()
    top, pll, recon, system = out["top"], out["pll"], out["recon"], out["system"]
    spice_results = out["spice_results"]
    n_block = sum(len(_live(top, n)._specs) for n in _INSTANCES if n != "u_pll")
    print(f"  specs: {n_block} on the blocks, {len(pll._specs)} on '{pll.module_name}'")
    print()
    print_two_level_summary(top, recon, pll)

    jitter = next(o for o in recon.outcomes if o.spec is system[JITTER_METRIC])
    assert jitter.status == "UNMEASURED", jitter.status
    assert not recon.passed, "an unmeasured spec must not pass"
    measured = [o for o in recon.outcomes if o.status != "UNMEASURED"]
    assert len(measured) == len(recon.outcomes) - 1,\
        "exactly one spec — the jitter gap — should be unmeasured"
    assert all(o.measurement.source == "spice"
               for o in measured if o.spec.module is not pll)
    assert all(o.measurement.source == "behavioral"
               for o in measured if o.spec.module is pll)

    print()
    print(f"  {len(spice_results)} SPICE testbenches: {out['t_spice']:.0f} s; "
          f"behavioral lock: {out['t_beh']:.0f} s; total wall-clock: "
          f"{time.perf_counter() - t_start:.0f} s")
    failed = [o for o in recon.outcomes if o.status == "FAIL"]
    print(f"  passed() = {recon.passed}:")
    for o in failed:
        print(f"    FAIL        {o.spec.module.module_name}: '{o.spec.name}' "
              f"— the block's own verdict, unchanged by the tree")
    print(f"    UNMEASURED  {pll.module_name}: '{system[JITTER_METRIC].name}' "
          f"— declared, nothing records it; on its own this holds passed() "
          f"False even with every FAIL fixed.")
