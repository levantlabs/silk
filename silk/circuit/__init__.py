"""silk.circuit — the central design object."""
from silk.circuit.emitted_names import validate_instance_name, validate_net_name
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.circuit.module_test import ModuleTestbench
from silk.circuit.metric_handle import MetricHandle
from silk.circuit.spec_report import SpecReport, SpecResult, CornerResults
from silk.circuit.monte_carlo import MonteCarloResults
from silk.circuit.test_failure import TestFailure, RunRecord
from silk.circuit.circuit_elements import ElementType, PortDir, PortRole, Port
from silk.circuit.net import Net, PortBinding
from silk.circuit.voltage_domain import VoltageDomain
from silk.circuit.interface_check import (
    check_interfaces, check_voltage_domains,
    require_annotation_coverage, AnnotationCoverageError,
    InterfaceViolation,
    list_unannotated_ports, annotation_coverage, UnannotatedPort, AnnotationCoverage,
)
from silk.circuit.junction_check import check_junction_overrange, JunctionViolation
from silk.circuit.port_roles import RoleRuleset, infer_port_roles
from silk.elements.schema import export_schema, module_schema, export_schema_json
from silk.circuit.report import CircuitReport, generate_report
from silk.circuit.report_diff import ReportDiff, diff_reports
from silk.circuit.emitted_names import ElementNames, element_names, source_element_names

__all__ = ["validate_instance_name", "validate_net_name", "Module", "ModuleTestbench", "MetricHandle", "Variable",
           "SpecReport", "SpecResult", "CornerResults", "MonteCarloResults",
           "ElementType", "PortDir", "PortRole", "Port",
           "Net", "PortBinding",
           "VoltageDomain",
           "check_interfaces", "check_voltage_domains",
           "require_annotation_coverage", "AnnotationCoverageError",
           "InterfaceViolation",
           "list_unannotated_ports", "annotation_coverage", "UnannotatedPort", "AnnotationCoverage",
           "check_junction_overrange", "JunctionViolation",
           "RoleRuleset", "infer_port_roles",
           "export_schema", "module_schema", "export_schema_json",
           "CircuitReport", "generate_report",
           "ReportDiff", "diff_reports",
           "ElementNames", "element_names", "source_element_names"]

from silk.circuit.view_nodes import view_node_context
__all__.append("view_node_context")
