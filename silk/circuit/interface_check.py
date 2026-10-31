"""Interface and voltage-domain checks and annotation audits used by circuit reports.

Checks run the corresponding rules from silk.lint.rules. Annotation audits
measure port-role coverage and can enforce a minimum coverage threshold.

  check_interfaces(module)      — runs all interface-category rules
  check_voltage_domains(module) — runs all domain-category rules

Also provides annotation audit tools (not lint rules):

  list_unannotated_ports(module)     — ports with no PortRole declared
  annotation_coverage(module)        — coverage counts and percentage
  require_annotation_coverage(...)   — CI gate; raises if below threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# PortBinding was moved to silk.circuit.net as part of the Phase 1 refactor.
# Re-export here so that existing callers importing it from this module
# continue to work.
from silk.circuit.net import PortBinding  # noqa: F401  (re-export)

from silk.lint.registry import RuleRegistry
from silk.lint.violations import InterfaceViolation  # re-export for legacy imports


def check_interfaces(module) -> List[InterfaceViolation]:
    """Run all interface-category rules against *module*.

    Returns a list of :class:`silk.lint.violations.InterfaceViolation`.
    Empty list means no issues found.
    """
    reg = RuleRegistry.default()
    result = reg.run_category("interface", module)
    return result.violations


def check_voltage_domains(module) -> List[InterfaceViolation]:
    """Run all domain-category rules against *module*.

    Returns a list of :class:`silk.lint.violations.InterfaceViolation`.
    Empty list means no domain violations found.
    """
    reg = RuleRegistry.default()
    result = reg.run_category("domain", module)
    return result.violations


# ---------------------------------------------------------------------------
# Annotation coverage metrics — NOT lint rules, kept here unchanged.
# ---------------------------------------------------------------------------


@dataclass
class UnannotatedPort:
    """A port with no PortRole declared.

    Parameters
    ----------
    instance : str
        Instance name in the parent's ``module_dict`` (e.g. ``'comp1'``).
    module_name : str
        Subcircuit name (e.g. ``'comp'``).
    port : str
        Port name (e.g. ``'vp'``).
    direction : str, optional
        The :class:`~silk.circuit.ports.PortDir` value (``'input'``,
        ``'output'``, ...), or ``None`` when the pin has no direction.
    """
    #: Instance name in the parent's ``module_dict`` (e.g. ``'comp1'``).
    instance: str
    #: Subcircuit name (e.g. ``'comp'``).
    module_name: str
    #: Port name (e.g. ``'vp'``).
    port: str
    #: The port direction as its string value, or ``None`` when unset.
    direction: Optional[str]


@dataclass
class AnnotationCoverage:
    """Annotation coverage summary for a module hierarchy.

    Parameters
    ----------
    total : int
        Number of ports across all direct child instances.
    annotated : int
        Number of those ports with a :class:`~silk.circuit.ports.PortRole`.
    unannotated_ports : list of UnannotatedPort
        The ports missing a role.
    """
    #: Number of ports across all direct child instances.
    total: int
    #: Number of those ports with a ``PortRole`` declared.
    annotated: int
    #: The ports missing a role, one record each.
    unannotated_ports: List[UnannotatedPort]

    @property
    def unannotated(self) -> int:
        """Number of ports with no role (``len(unannotated_ports)``)."""
        return len(self.unannotated_ports)

    @property
    def coverage_pct(self) -> float:
        """Percentage (0-100) of ports that have a role.

        ``100.0`` when the module has no child ports at all.
        """
        return 100.0 * self.annotated / self.total if self.total > 0 else 100.0

    def __str__(self) -> str:
        return (
            f"Annotation coverage: {self.annotated}/{self.total} ports "
            f"({self.coverage_pct:.1f}%)"
        )


def list_unannotated_ports(module) -> List[UnannotatedPort]:
    """Return all subcircuit ports that have no PortRole declared.

    Parameters
    ----------
    module : Module
        Top-level module whose ``module_dict`` is traversed.

    Returns
    -------
    list of UnannotatedPort
        Each entry identifies the instance, subcircuit name, port name,
        and direction of an unannotated port. Empty list means every port
        has a role declared.
    """
    result: List[UnannotatedPort] = []
    for instance_name, entry in module.module_dict.items():
        submodule = entry.get('device')
        if submodule is None:
            continue
        pin_list = getattr(submodule, 'pin_list', None)
        if not pin_list:
            continue
        sub_name = getattr(submodule, 'module_name', instance_name) or instance_name
        for port_name, pin in pin_list.items():
            role = getattr(pin, 'role', None)
            if role is None:
                direction = getattr(pin, 'direction', None)
                result.append(UnannotatedPort(
                    instance=instance_name,
                    module_name=sub_name,
                    port=port_name,
                    direction=direction.value if hasattr(direction, 'value') else direction,
                ))
    return result


def annotation_coverage(module) -> AnnotationCoverage:
    """Return annotation coverage counts for a module hierarchy.

    Parameters
    ----------
    module : Module
        Top-level module whose ``module_dict`` is traversed.

    Returns
    -------
    AnnotationCoverage
        Contains total port count, annotated count, coverage percentage,
        and the list of unannotated ports for further inspection.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk import annotation_coverage, require_annotation_coverage
    >>> block = Module('sky130', folder.name, 'block', pdk, module_name='block')
    >>> block.define_module_ports({'out': 'output'})
    >>> _ = dut.add_module(block, 'x1', out='out')
    >>> cov = annotation_coverage(dut)
    >>> cov.total, cov.annotated
    (1, 0)
    >>> cov.unannotated_ports[0].port
    'out'
    >>> require_annotation_coverage(dut, min_pct=0).coverage_pct
    0.0
    """
    total = 0
    unannotated = list_unannotated_ports(module)
    for entry in module.module_dict.values():
        submodule = entry.get('device')
        if submodule is None:
            continue
        pin_list = getattr(submodule, 'pin_list', None)
        if pin_list:
            total += len(pin_list)
    annotated = total - len(unannotated)
    return AnnotationCoverage(total=total, annotated=annotated, unannotated_ports=unannotated)


class AnnotationCoverageError(Exception):
    """Raised by :func:`require_annotation_coverage` when coverage is insufficient.

    Attributes
    ----------
    coverage : AnnotationCoverage
        Full coverage object — inspect ``.unannotated_ports`` for the list.
    min_pct : float
        The required minimum percentage that was not met.
    """

    def __init__(self, coverage: 'AnnotationCoverage', min_pct: float) -> None:
        self.coverage = coverage
        self.min_pct = min_pct
        port_lines = '\n'.join(
            f"  {u.instance}.{u.port} ({u.direction}) [{u.module_name}]"
            for u in coverage.unannotated_ports
        )
        super().__init__(
            f"Annotation coverage {coverage.coverage_pct:.1f}% is below required "
            f"{min_pct:.1f}%.\nUnannotated ports:\n{port_lines}"
        )


def require_annotation_coverage(module, min_pct: float = 80.0) -> AnnotationCoverage:
    """Raise AnnotationCoverageError if port annotation coverage is below threshold.

    Designed to drop into a pytest test for CI enforcement — annotation coverage
    becomes a ratchet, not a suggestion.

    Parameters
    ----------
    module : Module
        Top-level module whose full hierarchy is checked.
    min_pct : float
        Minimum acceptable coverage percentage (default 80.0).

    Returns
    -------
    AnnotationCoverage
        The coverage object, when the threshold is met.

    Raises
    ------
    AnnotationCoverageError
        If coverage is below *min_pct*.  The exception carries ``.coverage``
        (an :class:`AnnotationCoverage` object) and ``.min_pct``.

    Examples
    --------
    >>> from tempfile import TemporaryDirectory
    >>> from silk import Module, load_pdk
    >>> folder = TemporaryDirectory()
    >>> pdk = load_pdk('sky130')
    >>> dut = Module('sky130', folder.name, 'demo', pdk)
    >>> from silk import require_annotation_coverage
    >>> block = Module('sky130', folder.name, 'block', pdk, module_name='block')
    >>> block.define_module_ports({'out': 'output'})
    >>> _ = dut.add_module(block, 'x1', out='out')
    >>> require_annotation_coverage(dut, min_pct=0).coverage_pct
    0.0
    """
    cov = annotation_coverage(module)
    if cov.coverage_pct < min_pct:
        raise AnnotationCoverageError(cov, min_pct)
    return cov


# ---------------------------------------------------------------------------
# Domain resolution helper — used by silk/lint/rules/domain.py.
# Kept here to avoid a circular import during Task 16 migration.
# ---------------------------------------------------------------------------


def _resolve_net_domains(module, net_bindings: dict) -> dict:
    """Build {net_name: VoltageDomain | None} via explicit assignment + upward propagation.

    Reads explicit assignments from ``Net.domain`` directly (Phase 2a)
    instead of the legacy ``module._net_domains`` dict (removed).
    """
    net_dict = getattr(module, 'net_dict', {})
    resolved = {
        n.name: n.domain
        for n in net_dict.values()
        if n.domain is not None
    }

    for net, bindings in net_bindings.items():
        if net in resolved:
            continue
        domains = [b.domain for b in bindings if b.domain is not None]
        if not domains:
            continue
        first = domains[0]
        if all(first.is_compatible(d) for d in domains[1:]):
            resolved[net] = first
        # Incompatible: leave unresolved — RAIL_MISMATCH checker will flag it

    return resolved


__all__ = [
    "check_interfaces",
    "check_voltage_domains",
    "InterfaceViolation",
    "PortBinding",
    "UnannotatedPort",
    "AnnotationCoverage",
    "AnnotationCoverageError",
    "list_unannotated_ports",
    "annotation_coverage",
    "require_annotation_coverage",
]
