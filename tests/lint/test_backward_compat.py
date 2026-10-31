"""Backward-compat smoke tests for the wrapper migration (Task 16).

Validates that check_interfaces / check_voltage_domains / check_junction_overrange
return the new lint violation types after the wrapper migration, and that the
annotation_coverage metric helpers still work.
"""
from unittest.mock import MagicMock

from silk.circuit.interface_check import (
    check_interfaces,
    check_voltage_domains,
    annotation_coverage,
    require_annotation_coverage,
)
from silk.circuit.junction_check import check_junction_overrange
from silk.lint.violations import InterfaceViolation, JunctionViolation
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def _sub(tmp_path, name, ports):
    m = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


def test_check_interfaces_returns_lint_interface_violations(tmp_path):
    a = _sub(tmp_path, 'a', {'out': PortDir.OUT})
    b = _sub(tmp_path, 'b', {'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'shared'})
    top.add_module(b, 'b1', nets={'out': 'shared'})
    violations = check_interfaces(top)
    assert all(isinstance(v, InterfaceViolation) for v in violations)
    assert any(v.rule_id == "INT-001" for v in violations)


def test_check_voltage_domains_returns_list_on_empty_module(tmp_path):
    """Smoke test: empty module returns empty list, no crash."""
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    violations = check_voltage_domains(m)
    assert violations == []


def test_check_junction_overrange_skips_without_simulation(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    violations = check_junction_overrange(m)
    assert violations == []
    # Sanity: the return type annotation is JunctionViolation list; empty is fine.
    assert all(isinstance(v, JunctionViolation) for v in violations)


def test_annotation_coverage_still_works(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    cov = annotation_coverage(m)
    assert cov.coverage_pct == 100.0  # 0 ports → 100%


def test_require_annotation_coverage_unchanged_api(tmp_path):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    # Should not raise on an empty module
    cov = require_annotation_coverage(m, min_pct=99.0)
    assert cov.coverage_pct == 100.0
