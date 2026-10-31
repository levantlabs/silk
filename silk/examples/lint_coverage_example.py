"""Build structural modules and print lint findings and annotation coverage.

The scenarios show clean and faulty connections, no-connect markers, a YAML
waiver, and filtering of findings on an excluded subtree. The example also
prints rule help, saves HTML, JSON and JUnit XML reports, and compares a saved
lint report with its reloaded copy.

Run: python -m silk.examples.lint_coverage_example
No PDK or simulator is needed. Reports and the waiver YAML are written under
example_output_dir().
"""
from __future__ import annotations

from pathlib import Path

from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.modules import Module
from silk.circuit.report import generate_report
from silk.examples import example_output_dir
from silk.lint.registry import RuleRegistry
from silk.lint.rules import DEFAULT_CATALOG
from silk.lint.exclusions import ScopeExclusion
from silk.lint.help import help_rule, list_rules


def _sub(out_dir: str, name: str, ports: dict) -> Module:
    """Build a tiny submodule with the given port definitions."""
    m = Module("TEST", out_dir, f"{name}.cir", pdk_details=None)
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


# ---------------------------------------------------------------------------
# Module builders
# ---------------------------------------------------------------------------

def build_clean_module(out_dir: str) -> Module:
    """A clean two-instance design that produces no violations."""
    src = _sub(out_dir, "src", {
        "out": (PortDir.OUT, PortRole.ANALOG),
    })
    dst = _sub(out_dir, "dst", {
        "inp": (PortDir.IN, PortRole.ANALOG),
    })
    top = Module("TEST", out_dir, "clean_top.cir", pdk_details=None)
    top.add_module(src, "src1", nets={"out": "wire1"})
    top.add_module(dst, "dst1", nets={"inp": "wire1"})
    return top


def build_problem_module(out_dir: str) -> Module:
    """Build a design with driver, connectivity and supply violations.

    Violations expected:
      - INT-001 DRIVER_CONFLICT  (two outputs on 'shared')
      - STRUCT-003 FLOATING_NET  (orphan net 'orphan')
      - STRUCT-001 DANGLING_PORT (sub instance 'dangler' with 'out' unbound)
      - STRUCT-002 POWER_PORT_UNCONNECTED (a 'vdd' POWER pin unbound)
    """
    a = _sub(out_dir, "a", {"out": PortDir.OUT})
    b = _sub(out_dir, "b", {"out": PortDir.OUT})
    dangler = _sub(out_dir, "dangler", {
        "in": PortDir.IN,
        "out": PortDir.OUT,
    })
    powered = _sub(out_dir, "powered", {
        "in": PortDir.IN,
        "vdd": (PortDir.IN, PortRole.POWER),
    })

    top = Module("TEST", out_dir, "problem_top.cir", pdk_details=None)
    # INT-001 DRIVER_CONFLICT: two outputs on 'shared'
    top.add_module(a, "a1", nets={"out": "shared"})
    top.add_module(b, "b1", nets={"out": "shared"})
    # STRUCT-001 DANGLING_PORT: 'out' left unbound on dangler instance
    top.add_module(dangler, "dangler1", nets={"in": "wire_in"})
    # STRUCT-002 POWER_PORT_UNCONNECTED: 'vdd' POWER pin not bound
    top.add_module(powered, "powered1", nets={"in": "wire_in"})
    # STRUCT-003 FLOATING_NET: a net with zero pins
    top.add_net("orphan")
    return top


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_clean(out_dir: str) -> None:
    print("=" * 60)
    print("Scenario 1: Clean module (zero violations)")
    print("=" * 60)
    m = build_clean_module(out_dir)
    rep = generate_report(m, auto_discover_config=False)
    print(f"  Module: {rep.module_name}")
    print(f"  Lint summary: {rep.lint.summary()}")
    print(f"  Coverage rules tracked: {len(rep.coverage.rule_coverage.root.local)}")
    print()


def scenario_problem(out_dir: str) -> None:
    print("=" * 60)
    print("Scenario 2: Module with several violations")
    print("=" * 60)
    m = build_problem_module(out_dir)
    rep = generate_report(m, auto_discover_config=False)
    summary = rep.lint.summary()
    print(f"  errors={summary['errors']} warnings={summary['warnings']} "
          f"infos={summary['infos']} waived={summary['waived']}")
    print()
    print("  Violations:")
    for v in rep.lint.violations:
        print(f"    [{v.severity.to_string():7s}] {v.rule_id}: {v.message}")
    print()
    print("  Compliance matrix:")
    for rule_id, entry in sorted(rep.lint.compliance_matrix().items()):
        print(f"    {rule_id}: {entry}")
    print()


def scenario_no_connect(out_dir: str) -> None:
    """Show net and per-pin no-connect markers suppressing lint violations."""
    print("=" * 60)
    print("Scenario 3: Net and per-pin no-connect markers")
    print("=" * 60)
    m = Module("TEST", out_dir, "noconnect_top.cir", pdk_details=None)
    m.add_net("probe_out")
    print("  Without NoConnect:")
    rep_before = generate_report(m, auto_discover_config=False)
    floating = [v for v in rep_before.lint.violations if v.rule_id == "STRUCT-003"]
    print(f"    {len(floating)} FLOATING_NET violation(s)")

    # Mark it intentional
    m.add_no_connect("probe_out", reason="differential test tap")
    print("  After m.add_no_connect('probe_out', reason='...'):")
    rep_after = generate_report(m, auto_discover_config=False)
    floating = [v for v in rep_after.lint.violations if v.rule_id == "STRUCT-003"]
    print(f"    {len(floating)} FLOATING_NET violation(s)")
    print(f"    LintReport.no_connects: {len(rep_after.lint.no_connects)} marker(s)")
    for inst, net, nc in rep_after.lint.no_connects:
        print(f"      - {net}: {nc.reason}")
    print()

    m = build_problem_module(out_dir)
    print("  Without NoConnectPort on dangler1.out:")
    rep_before = generate_report(m, auto_discover_config=False)
    dangling = [
        v for v in rep_before.lint.violations
        if v.rule_id == "STRUCT-001"
        and v.location.module_name == "dangler1"
        and v.location.port_name == "out"
    ]
    print(f"    {len(dangling)} STRUCT-001 DANGLING_PORT violation(s) on dangler1.out")

    m.add_no_connect_port("dangler1", "out", reason="unused diagnostic output")
    print("  After m.add_no_connect_port('dangler1', 'out', reason='...'):")
    rep_after = generate_report(m, auto_discover_config=False)
    dangling = [
        v for v in rep_after.lint.violations
        if v.rule_id == "STRUCT-001"
        and v.location.module_name == "dangler1"
        and v.location.port_name == "out"
    ]
    print(f"    {len(dangling)} STRUCT-001 DANGLING_PORT violation(s) on dangler1.out")
    print(f"    LintReport.no_connect_ports: {len(rep_after.lint.no_connect_ports)} marker(s)")
    for marker in rep_after.lint.no_connect_ports:
        print(f"      - {marker.instance}.{marker.port}: {marker.reason}")
    print()


def scenario_waiver(out_dir: str) -> None:
    """A YAML waiver for a specific rule+net suppresses one violation."""
    print("=" * 60)
    print("Scenario 4: Waiver suppresses INT-001 with required reason")
    print("=" * 60)
    waiver_yaml = Path(out_dir) / "silk_waivers.yaml"
    waiver_yaml.write_text(
        "waivers:\n"
        "  - rule_id: INT-001\n"
        "    matcher:\n"
        "      location_type: net\n"
        "      fields:\n"
        "        net_name: shared\n"
        "    reason: intentional dotted-OR for production test mux\n"
        "    author: example_user\n"
    )
    reg = RuleRegistry.default()
    reg.load_waivers(str(waiver_yaml))

    m = build_problem_module(out_dir)
    rep = generate_report(m, registry=reg, auto_discover_config=False)
    int001_active = [v for v in rep.lint.violations if v.rule_id == "INT-001"]
    int001_waived = [w for w in rep.lint.waivers_applied if w.rule_id == "INT-001"]
    print(f"  INT-001 active:  {len(int001_active)}  (was 1 without waiver)")
    print(f"  INT-001 waived:  {len(int001_waived)}")
    if int001_waived:
        print(f"    waived violation: {int001_waived[0].message}")
    print()


def scenario_exclusion(out_dir: str) -> None:
    """A ScopeExclusion filters violations whose location is on the excluded subtree."""
    print("=" * 60)
    print("Scenario 5: ScopeExclusion filters findings on an excluded subtree")
    print("=" * 60)
    reg = RuleRegistry(
        rules=DEFAULT_CATALOG(),
        exclusions=[ScopeExclusion(
            target_type="instance",
            target_path="dangler1",
            reason="findings on this block are checked separately",
        )],
    )

    m = build_problem_module(out_dir)
    rep = generate_report(m, registry=reg, auto_discover_config=False)
    on_dangler = [
        v for v in rep.lint.violations
        if getattr(v.location, "module_name", "") == "dangler1"
    ]
    print(f"  Violations on dangler1 instance: {len(on_dangler)} "
          f"(would be ≥1 without exclusion)")
    print(f"  Total active violations: {len(rep.lint.violations)}")
    print()


def scenario_artifacts(out_dir: str) -> None:
    """Save HTML, JSON, JUnit XML, and demonstrate diff."""
    print("=" * 60)
    print("Scenario 6: Save report artifacts and diff against itself")
    print("=" * 60)
    m = build_problem_module(out_dir)
    rep = generate_report(m, auto_discover_config=False)

    full_html = Path(out_dir) / "circuit_dashboard.html"
    lint_html = Path(out_dir) / "lint_only.html"
    cov_html = Path(out_dir) / "coverage_scorecard.html"
    junit_xml = Path(out_dir) / "lint_junit.xml"
    lint_json = Path(out_dir) / "lint.json"
    cov_json = Path(out_dir) / "coverage.json"

    rep.save_html(str(full_html))
    rep.lint.save_html(lint_html)
    rep.coverage.save_html(cov_html)
    rep.lint.save_junit_xml(junit_xml)
    rep.lint.save_json(lint_json)
    rep.coverage.save_json(cov_json)

    print("  Wrote:")
    for p in [full_html, lint_html, cov_html, junit_xml, lint_json, cov_json]:
        size = p.stat().st_size
        print(f"    {p.name}  ({size} bytes)")
    print()

    # Diff against a re-loaded copy: should be empty
    from silk.lint.reports.lint_report import LintReport
    reloaded = LintReport.load_json(lint_json)
    diff = rep.lint.diff(reloaded)
    print(f"  Self-diff: {len(diff.added)} added, {len(diff.removed)} removed")
    print()


def scenario_help_text() -> None:
    """Print the rationale for one rule, and show how list_rules works."""
    print("=" * 60)
    print("Scenario 7: help_rule and list_rules")
    print("=" * 60)
    default_rules = list_rules()
    print(f"  list_rules() returns {len(default_rules)} rules:")
    print(f"    {', '.join(default_rules)}")
    optional = [r for r in list_rules(include_optional=True) if r not in default_rules]
    print(f"  opt-in rules (RuleRegistry.default(include_optional=True)): "
          f"{', '.join(optional)}")
    print()
    print("  help_rule('INT-001'):")
    print()
    text = help_rule("INT-001")
    for line in text.splitlines():
        print(f"    {line}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = example_output_dir()
    print(f"\nSilk lint and coverage walk-through — output dir: {out_dir}\n")

    scenario_clean(out_dir)
    scenario_problem(out_dir)
    scenario_no_connect(out_dir)
    scenario_waiver(out_dir)
    scenario_exclusion(out_dir)
    scenario_artifacts(out_dir)
    scenario_help_text()

    print("=" * 60)
    print("Done. HTML / JSON / JUnit artifacts written to:")
    print(f"  {out_dir}")
    print("Open circuit_dashboard.html in a browser to see the full report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
