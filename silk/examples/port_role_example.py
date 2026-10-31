"""port_role_example.py — PortDir + PortRole and interface checking.

Demonstrates how to annotate subcircuit ports with semantic roles and
use check_interfaces() to catch wiring mistakes.

Three scenarios:
  1. Correct wiring  — passes interface check
  2. Driver conflict — two outputs on same net (INT-001, ERROR)
  3. Role mismatch   — digital output driving analog input (INT-003 and
                       INT-004, both ERROR)

The three interface rules demonstrated here default to ERROR severity; see
docs/lint_and_coverage_user_guide.md for the catalog and a walk-through of
the lint framework.

Run: python -m silk.examples.port_role_example
The example prints interface findings and needs no simulator or installed PDK.
"""

from silk.examples import example_output_dir
from silk.circuit.circuit_elements import PortDir, PortRole
from silk.circuit.interface_check import check_interfaces
from silk.circuit.modules import Module


def make_comparator(out_dir):
    """A comparator — analog input, digital output."""
    cmp = Module('TEST', out_dir, 'cmp.cir', pdk_details=None)
    cmp.define_module_name('comp')
    cmp.define_module_ports({
        'vip':  (PortDir.IN,  PortRole.ANALOG),
        'vim':  (PortDir.IN,  PortRole.ANALOG),
        'vdd':  (PortDir.IN,  PortRole.POWER),
        'vss':  (PortDir.IN,  PortRole.GROUND),
        'vout': (PortDir.OUT, PortRole.DIGITAL),
    })
    return cmp


def make_digital_block(out_dir):
    """A digital block consuming a comparator output."""
    blk = Module('TEST', out_dir, 'digital.cir', pdk_details=None)
    blk.define_module_name('digital')
    blk.define_module_ports({
        'din':  (PortDir.IN,  PortRole.DIGITAL),
        'dout': (PortDir.OUT, PortRole.DIGITAL),
        'vdd':  (PortDir.IN,  PortRole.POWER),
        'vss':  (PortDir.IN,  PortRole.GROUND),
    })
    return blk


def make_opamp(out_dir):
    """A simple op-amp with annotated ports (analog inputs, analog output)."""
    amp = Module('TEST', out_dir, 'opamp.cir', pdk_details=None)
    amp.define_module_name('opamp')
    amp.define_module_ports({
        'vip':  (PortDir.IN,  PortRole.ANALOG),
        'vim':  (PortDir.IN,  PortRole.ANALOG),
        'vdd':  (PortDir.IN,  PortRole.POWER),
        'vss':  (PortDir.IN,  PortRole.GROUND),
        'vout': (PortDir.OUT, PortRole.ANALOG),
    })
    return amp


def _print_violations(violations):
    if not violations:
        print("  Interface check: PASS")
        return
    for v in violations:
        # Each violation is a silk.lint.violations.InterfaceViolation.
        # severity is a Severity enum; to_string() gives 'error'/'warning'/'info'.
        # rule_id is the stable catalog id ('INT-001', 'INT-003', etc.).
        print(f"  [{v.severity.to_string():7s}] {v.rule_id}: {v.message}")


def scenario_correct(out_dir):
    print("=" * 55)
    print("Scenario 1: Correct circuit")
    print("  comparator DIGITAL out → digital block DIGITAL in")
    print("=" * 55)

    cmp = make_comparator(out_dir)
    blk = make_digital_block(out_dir)

    top = Module('TEST', out_dir, 'top_ok.cir', pdk_details=None)
    # add_module binds each submodule pin to a named net via the nets dict.
    top.add_module(cmp, 'cmp1', nets={
        'vip': 'vin_p', 'vim': 'vin_m',
        'vdd': 'vdd',   'vss': 'gnd',
        'vout': 'cmp_out',
    })
    top.add_module(blk, 'blk1', nets={
        'din':  'cmp_out',
        'dout': 'dig_out',
        'vdd':  'vdd',
        'vss':  'gnd',
    })

    _print_violations(check_interfaces(top))
    print()


def scenario_driver_conflict(out_dir):
    print("=" * 55)
    print("Scenario 2: Driver conflict (INT-001, ERROR)")
    print("  two op-amp outputs driving the same net")
    print("=" * 55)

    amp1 = make_opamp(out_dir)
    amp2 = make_opamp(out_dir)

    top = Module('TEST', out_dir, 'top_bad.cir', pdk_details=None)
    top.add_module(amp1, 'amp1', nets={
        'vip': 'in1_p', 'vim': 'in1_m',
        'vdd': 'vdd',   'vss': 'gnd',
        'vout': 'shared_out',
    })
    top.add_module(amp2, 'amp2', nets={
        'vip': 'in2_p', 'vim': 'in2_m',
        'vdd': 'vdd',   'vss': 'gnd',
        'vout': 'shared_out',   # <-- second driver on the same net
    })

    _print_violations(check_interfaces(top))
    print()


def scenario_role_mismatch(out_dir):
    print("=" * 55)
    print("Scenario 3: Role mismatch (INT-003 and INT-004, ERROR)")
    print("  comparator DIGITAL output → op-amp ANALOG input")
    print("=" * 55)

    cmp = make_comparator(out_dir)
    amp = make_opamp(out_dir)

    top = Module('TEST', out_dir, 'top_warn.cir', pdk_details=None)
    top.add_module(cmp, 'cmp1', nets={
        'vip': 'vin_p', 'vim': 'vin_m',
        'vdd': 'vdd',   'vss': 'gnd',
        'vout': 'mixed',
    })
    top.add_module(amp, 'amp1', nets={
        'vip': 'mixed',      # DIGITAL signal landing on an ANALOG input
        'vim': 'vin_m',
        'vdd': 'vdd',
        'vss': 'gnd',
        'vout': 'amp_out',
    })

    _print_violations(check_interfaces(top))
    print()


if __name__ == '__main__':
    out_dir = example_output_dir()

    print("\nPort Direction + Role — Interface Checking Demo")
    print("silk.circuit.PortDir  : IN | OUT | INOUT")
    print("silk.circuit.PortRole : ANALOG | DIGITAL | CLOCK | POWER | GROUND")
    print()

    scenario_correct(out_dir)
    scenario_driver_conflict(out_dir)
    scenario_role_mismatch(out_dir)
