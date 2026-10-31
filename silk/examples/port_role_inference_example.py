"""Infer port roles from names and print annotation coverage.

The four scenarios use default rules, added analog and bias rules, a YAML
ruleset, and an explicit role that inference preserves. Name rules fill unset
roles. Infer roles on the block definition; placed instances use that definition.

Run: python -m silk.examples.port_role_inference_example
The example writes a YAML ruleset under example_output_dir(). It needs no
simulator or installed PDK.
"""
import textwrap

from silk.examples import example_output_dir
from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.interface_check import annotation_coverage
from silk.circuit.modules import Module
from silk.circuit.port_roles import RoleRuleset, infer_port_roles


# Inference fills roles left unset in these port definitions.
_PORTS = {
    'avdd':  PortDir.IN,    # -> POWER  (default *vdd*)
    'avss':  PortDir.IN,    # -> GROUND (default *vss*)
    'clk':   PortDir.IN,    # -> CLOCK  (default *clk*)
    'vinp':  PortDir.IN,    # analog — no default rule
    'vinn':  PortDir.IN,    # analog — no default rule
    'vout':  PortDir.OUT,   # analog — no default rule
    'ibias': PortDir.IN,    # bias   — no default rule
}


def _ota(out_dir, name='ota'):
    """A 7-port block with every role UNSET."""
    m = Module('TEST', out_dir, f'{name}.cir', pdk_details=None)
    m.define_module_name(name)
    m.define_module_ports(dict(_PORTS))
    return m


def _coverage_in_parent(out_dir, leaf):
    """Place the block in a parent and return instance annotation coverage."""
    top = Module('TEST', out_dir, 'top.cir', pdk_details=None)
    top.add_module(leaf, 'u_ota', nets={p: p for p in _PORTS})
    return annotation_coverage(top)


def _report(title, leaf, report, out_dir):
    filled = [(n, r.value) for n, r in report.filled]
    cov = _coverage_in_parent(out_dir, leaf)
    print(title)
    print(f"  filled     : {filled}")
    print(f"  unmatched  : {report.unmatched}")
    print(f"  coverage   : {cov.annotated}/{cov.total} ports "
          f"({cov.coverage_pct:.0f}%) once instantiated")
    print()


def scenario_default(out_dir):
    print("=" * 60)
    print("Scenario 1: silk's shipped default ruleset")
    print("  fills rails and clocks; leaves unmatched signals unset")
    print("=" * 60)
    leaf = _ota(out_dir)
    report = infer_port_roles(leaf, RoleRuleset.default())
    _report("RoleRuleset.default():", leaf, report, out_dir)


def scenario_house_add(out_dir):
    print("=" * 60)
    print("Scenario 2: extend the default with house conventions (.add)")
    print("  add analog/bias name rules -> full coverage")
    print("=" * 60)
    ruleset = (RoleRuleset.default()
               .add('*vin*',  PortRole.ANALOG)
               .add('*vout*', PortRole.ANALOG)
               .add('*bias*', PortRole.ANALOG))
    leaf = _ota(out_dir)
    report = infer_port_roles(leaf, ruleset)
    _report("default + house .add():", leaf, report, out_dir)


def scenario_house_yaml(out_dir):
    print("=" * 60)
    print("Scenario 3: the same house rules authored as a YAML file")
    print("  (schema: docs/authoring-port-role-rules.md)")
    print("=" * 60)
    yaml_path = f"{out_dir}/house-roles.yaml"
    with open(yaml_path, "w") as f:
        f.write(textwrap.dedent("""
            version: 1
            rules:
              - {match: "*vdd*",  role: power}
              - {match: "*vss*",  role: ground}
              - {match: "*clk*",  role: clock}
              - {match: "*vin*",  role: analog}
              - {match: "*vout*", role: analog}
              - {match: "*bias*", role: analog}
        """))
    leaf = _ota(out_dir)
    report = infer_port_roles(leaf, RoleRuleset.from_yaml(yaml_path))
    _report("RoleRuleset.from_yaml('house-roles.yaml'):", leaf, report, out_dir)


def scenario_never_clobber(out_dir):
    print("=" * 60)
    print("Scenario 4: inference never overwrites a hand-set role")
    print("  'avdd' is declared ANALOG by hand; the default would say POWER")
    print("=" * 60)
    m = Module('TEST', out_dir, 'held.cir', pdk_details=None)
    m.define_module_name('held')
    ports = dict(_PORTS)
    ports['avdd'] = (PortDir.IN, PortRole.ANALOG)   # explicit — must survive
    m.define_module_ports(ports)
    report = infer_port_roles(m, RoleRuleset.default())
    print(f"  avdd role  : {m.pin_list['avdd'].role.value}  (kept, not clobbered)")
    print(f"  filled     : {[(n, r.value) for n, r in report.filled]}")
    print()


if __name__ == '__main__':
    out_dir = example_output_dir()
    print("\nPort-Role Inference Demo")
    print("silk.circuit.PortRole : ANALOG | DIGITAL | CLOCK | POWER | GROUND")
    print("infer_port_roles(module, ruleset) -> fills unset roles on module.pin_list\n")
    scenario_default(out_dir)
    scenario_house_add(out_dir)
    scenario_house_yaml(out_dir)
    scenario_never_clobber(out_dir)
    print("Author your own ruleset: docs/authoring-port-role-rules.md")
