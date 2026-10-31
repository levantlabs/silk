"""Voltage-domain rules: DOM-001..DOM-003.

Each rule re-runs the domain resolution helper to stay self-contained.
The compatibility function ``check_voltage_domains`` runs these rules through
the registry.
"""
from collections import defaultdict
from itertools import combinations
from typing import Iterator
from silk.lint.rule import Rule
from silk.lint.severity import Severity
from silk.lint.violations import InterfaceViolation
from silk.lint.locations import NetLocation
from silk.circuit.modules import Module
from silk.circuit.interface_check import _resolve_net_domains

# DOM-003 threshold: a domain counts as "live" (able to stress a disabled
# neighbour) when its upper bound exceeds this many volts. 0.0 V means any
# domain with a positive v_max() is live; the disabled state is always
# VoltageDomain.v_low() == 0.0 V. Heuristic — not a PDK parameter.
# Which domain can BE disabled is not a heuristic: only domains declared
# ``VoltageDomain(..., switchable=True)`` are candidates.
_LIVE_RAIL_MIN_VMAX_V = 0.0


def _resolve_module_domains(module):
    """Return (net_bindings, all_domains) with downward propagation applied.

    Mirrors the setup section of check_voltage_domains so each Rule can
    operate against fully-resolved per-net binding data.
    """
    net_bindings: dict = defaultdict(list)
    net_dict = getattr(module, 'net_dict', {})
    for net_name, net in net_dict.items():
        for b in net.pins:
            if b.kind != 'submodule':
                continue
            net_bindings[net_name].append(b)

    resolved = _resolve_net_domains(module, net_bindings)

    # Downward propagation
    for net_name, net_domain in resolved.items():
        if net_domain is None:
            continue
        for b in net_bindings.get(net_name, []):
            if b.domain is None:
                b.domain = net_domain

    # Collect all known domains (identity dedup since VoltageDomain isn't hashable)
    all_domains: list = []
    seen_ids: set = set()

    def _add(d):
        if d is not None and id(d) not in seen_ids:
            seen_ids.add(id(d))
            all_domains.append(d)

    for d in getattr(module, '_domains', []):
        _add(d)
    for d in resolved.values():
        _add(d)
    for bindings in net_bindings.values():
        for b in bindings:
            if b.domain is not None:
                _add(b.domain)

    return net_bindings, all_domains


def _net_loc(net_name: str, ports: list) -> NetLocation:
    port_names = [f"{b.instance}.{b.port}" for b in ports]
    return NetLocation(net_name=net_name, port_names=port_names)


class RailMismatchRule(Rule):
    rule_id = "DOM-001"
    category = "domain"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A single net carries POWER-role ports incompatible with its explicitly "
        "assigned voltage domain or with each other (e.g., a 1.8V analog "
        "supply and a 3.3V IO supply). This is "
        "a wiring mistake that either shorts the two supplies (if both are "
        "sourced) or floats one of them. Either way, silicon behavior is "
        "undefined. Fix by routing each supply only through its matched "
        "domain, or explicitly connecting via a level-shifter block."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        net_bindings, _ = _resolve_module_domains(module)
        for net, bindings in net_bindings.items():
            power_bindings = [
                b for b in bindings
                if b.role == 'power' and b.domain is not None
            ]
            net_domain = module.net_dict[net].domain
            if net_domain is not None:
                for binding in power_bindings:
                    if net_domain.is_compatible(binding.domain):
                        continue
                    yield InterfaceViolation(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"Net '{net}' explicitly assigned domain "
                            f"({net_domain.name}, {net_domain.nominal}V) "
                            f"connects incompatible POWER port "
                            f"{binding.instance}.{binding.port}"
                            f"({binding.domain.name}, {binding.domain.nominal}V)"
                        ),
                        location=_net_loc(net, [binding]),
                    )
            for b_a, b_b in combinations(power_bindings, 2):
                if b_a.domain.is_compatible(b_b.domain):
                    continue
                names = (
                    f"{b_a.instance}.{b_a.port}"
                    f"({b_a.domain.name}, {b_a.domain.nominal}V) vs "
                    f"{b_b.instance}.{b_b.port}"
                    f"({b_b.domain.name}, {b_b.domain.nominal}V)"
                )
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=f"Net '{net}' connects incompatible domains: {names}",
                    location=_net_loc(net, [b_a, b_b]),
                )


class DomainCrossingRule(Rule):
    rule_id = "DOM-002"
    category = "domain"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A signal net carries ports whose modules are powered by different "
        "voltage domains, without an annotated domain crossing. Signals "
        "from a higher-voltage domain applied to a lower-voltage gate can "
        "exceed Vgs-max and damage the device. Signals from a lower-voltage "
        "domain applied to a higher-voltage gate may not switch the device "
        "at all. Add an explicit domain crossing (level shifter, isolation "
        "cell) or waive with a justification."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        net_bindings, _ = _resolve_module_domains(module)
        for net, bindings in net_bindings.items():
            signal_bindings = [
                b for b in bindings
                if b.role not in ('power', 'ground') and b.domain is not None
            ]
            if len(signal_bindings) < 2:
                continue
            seen_names: dict = {}
            for b in signal_bindings:
                seen_names.setdefault(b.domain.name, b)
            if len(seen_names) <= 1:
                continue
            names = ', '.join(
                f"{b.instance}.{b.port}({b.domain.name})"
                for b in signal_bindings
            )
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"Net '{net}' connects domains: {names}",
                location=_net_loc(net, signal_bindings),
            )


class SupplyDisabledStressRule(Rule):
    rule_id = "DOM-003"
    category = "domain"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A live (non-zero) voltage domain is driving a gate in a device "
        "whose own supply can be disabled (a powered-down subblock on a "
        "rail declared VoltageDomain(..., switchable=True)). While that "
        "rail is at 0 V the live signal forward-biases the disabled "
        "device's parasitic diodes, which can sink unbounded current from "
        "the live supply into the off supply. Only switchable rails are "
        "considered; an unannotated crossing is DOM-002's finding alone. "
        "Use a level-shifter or isolation cell that is immune to domain "
        "shutdown, explicitly sequence the power-up, or waive with a "
        "justification."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        net_bindings, all_domains = _resolve_module_domains(module)
        for disabled in all_domains:
            if not getattr(disabled, 'switchable', False):
                continue
            for net, bindings in net_bindings.items():
                disabled_ports = [
                    b for b in bindings
                    if b.domain is not None
                    and b.domain.is_compatible(disabled)
                    and b.role not in ('power', 'ground')
                ]
                if not disabled_ports:
                    continue
                live_ports = [
                    b for b in bindings
                    if b.domain is not None
                    and not b.domain.is_compatible(disabled)
                    and b.domain.v_max() > _LIVE_RAIL_MIN_VMAX_V
                ]
                if not live_ports:
                    continue
                live_domain = live_ports[0].domain
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=(
                        f"Net '{net}' may stress domain '{disabled.name}' "
                        f"(switchable, 0V when powered down): driven from "
                        f"domain '{live_domain.name}' "
                        f"({live_domain.nominal}V nominal)"
                    ),
                    location=_net_loc(net, disabled_ports + live_ports),
                )
