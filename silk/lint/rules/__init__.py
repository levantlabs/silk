"""Default rule catalog assembly.

Three views of silk's shipped rules:

- ``DEFAULT_CATALOG()`` — what ``RuleRegistry.default()`` runs.
- ``OPTIONAL_CATALOG()`` — implemented, tested rules that are NOT run by
  default because they need a port-declaration contract the examples do
  not yet follow (see the rule's rationale and); opt in
  with ``RuleRegistry.default(include_optional=True)`` or ``add_rule``.
- ``FULL_CATALOG()`` — default + optional, the complete set ``help_rule``
  can describe.
"""
from silk.lint.rule import Rule
from silk.lint.rules.interface import (
    DriverConflictRule, PowerGroundShortRule, RoleMismatchRule,
    NoDriverRule, ClockOnAnalogRule,
)
from silk.lint.rules.domain import (
    RailMismatchRule, DomainCrossingRule, SupplyDisabledStressRule,
)
from silk.lint.rules.stress import JunctionOverrangeRule
from silk.lint.rules.structural import (
    FloatingNetRule, DanglingPortRule, PowerPortUnconnectedRule, FloatingGateRule,
    BodyNotTiedRule, MissingParamRule, SignalOnSupplyNetRule, UnusedInstanceRule,
    BusPartialWiringRule, PowerGroundPortRequired, PortDirectionSanity,
    SingleConnectionNetRule, UnportableGroundPortRule, HiddenSupplyNetRule,
)


def DEFAULT_CATALOG() -> list[Rule]:
    """Return silk's built-in rule catalog (5 interface + 3 domain + 1 stress + 12 structural + 1 bus = 22)."""
    return [
        DriverConflictRule(),
        PowerGroundShortRule(),
        RoleMismatchRule(),
        NoDriverRule(),
        ClockOnAnalogRule(),
        RailMismatchRule(),
        DomainCrossingRule(),
        SupplyDisabledStressRule(),
        JunctionOverrangeRule(),
        FloatingNetRule(),
        SingleConnectionNetRule(),
        DanglingPortRule(),
        PowerPortUnconnectedRule(),
        FloatingGateRule(),
        BodyNotTiedRule(),
        MissingParamRule(),
        SignalOnSupplyNetRule(),
        UnusedInstanceRule(),
        BusPartialWiringRule(),
        UnportableGroundPortRule(),
        HiddenSupplyNetRule(),
        PortDirectionSanity(),
    ]


def OPTIONAL_CATALOG() -> list[Rule]:
    """Return the shipped rules that are opt-in (1 structural).

    STRUCT-009 POWER_GROUND_PORT_REQUIRED is stricter than the default catalog
    wants, on shipped evidence (982 module builds over
    every shipped example): it demands a declared POWER and GROUND port from
    every device-bearing subcircuit, which nine legitimate blocks do not have
    (a PMOS-only current bank, a BJT/resistor bandgap branch, a diode-connected
    bias reference, a reference ladder, a cap-DAC).

    The *defect* STRUCT-009 was written to catch — a block wiring devices to a
    supply net it never declares — is carried by STRUCT-013
    HIDDEN_SUPPLY_NET, which IS in the default catalog: it keys on the
    wiring rather than on the port declaration, so it is silent on all nine
    supply-less-by-design blocks .

    STRUCT-010 PORT_DIRECTION_SANITY used to sit here too, because the same
    gate found it warning on an LDO's regulated output — legitimately a
    POWER-role OUT port. it exempts a supply-role OUT port in a
    block that also receives a rail of the same role, is silent on every
    shipped example, and is in the default catalog.
    """
    return [
        PowerGroundPortRequired(),
    ]


def FULL_CATALOG() -> list[Rule]:
    """Return every shipped rule: ``DEFAULT_CATALOG() + OPTIONAL_CATALOG()``."""
    return DEFAULT_CATALOG() + OPTIONAL_CATALOG()
