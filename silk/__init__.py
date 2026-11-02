"""silk — analog circuit design automation framework.

Install:  pip install circuitsilk
Import:   import silk  /  from silk import Module

Quick start::

    from silk import Module, ModuleTestbench
    from silk.pdk import load_pdk
    from silk.simulation import Transient

"""
# Core circuit objects
from silk.circuit import (
    Module, ModuleTestbench, Variable, CornerResults, MonteCarloResults,
    Port, PortDir, PortRole,
    Net, PortBinding,
    VoltageDomain,
    check_interfaces, check_voltage_domains,
    require_annotation_coverage, AnnotationCoverageError,
    InterfaceViolation,
    check_junction_overrange, JunctionViolation,
    RoleRuleset, infer_port_roles,
    list_unannotated_ports, annotation_coverage, UnannotatedPort, AnnotationCoverage,
    CircuitReport, generate_report,
    ReportDiff, diff_reports,
)
from silk.circuit import export_schema, module_schema, export_schema_json
# Source + metric schema exports (parallel to the device export_schema above).
from silk.sources import export_source_schema
from silk.signals import export_metric_schema

# PDK
from silk.pdk import BasePDK, UNIT_PDK, UNIT_SI, load_pdk

# Simulation analysis types and corner objects
from silk.simulation import Transient, AC, DC, Noise, Corner

# The analysis a behavioural model is running in (no engine import)
from silk.simulation._analysis_context import analysis

from silk.simulation.variation import Normal, Uniform

# Signals
from silk.signals import Signal, NgspiceSignal

# Optimization
from silk.optimize import VariableSpace, Optimizer

# Database
from silk.db import Database, DesignStage

# Sources
from silk import sources

# Code generation
from silk.codegen import Generator

# Exceptions
from silk.exceptions import (
    CircuitError,
    DuplicateNameError,
    UndeclaredNetError,
    SimulationError,
    PDKError,
)

__version__ = "0.3.0rc1"


def view(source, out_path=None, *, open=True, **kwargs):
    """Open the interactive waveform explorer on a rawfile or simulated module.

    The top-level shortcut::

        import silk
        silk.view("sim.raw")          # emit view.html beside it and open it
        silk.view(tb)                 # a simulated Module/testbench

    *source* is a nutmeg ``.raw`` path, a
    :class:`silk.signals.rawfile.RawFile`, or any simulated module carrying
    ``.results_file`` (for which this simply delegates to
    :meth:`Module.view <silk.circuit.Module.view>`, so its defaults
    and provenance apply). Extra keyword arguments — ``signals``, ``exclude``,
    ``max_depth``, ``title``, ``inline``, ``context``, ``serve``, ``port`` —
    pass straight through. Returns the emitted HTML path.

    ``silk.signals.view`` is imported lazily, so importing ``silk`` never
    pulls the viewer in.
    """
    module_view = getattr(source, 'view', None)
    if callable(module_view) and hasattr(source, 'results_file'):
        return module_view(out_path=out_path, open=open, **kwargs)

    import os
    import webbrowser
    from silk.signals.view import emit_view

    if out_path is None:
        if isinstance(source, (str, os.PathLike)):
            rawdir = os.path.dirname(os.path.abspath(os.fspath(source))) or '.'
            out_path = os.path.join(rawdir, 'view.html')
        else:
            out_path = os.path.join(os.getcwd(), 'view.html')
    kwargs.setdefault('labels', 'designer')       # TASK-530; an explicit labels= wins
    result = emit_view(source, out_path, **kwargs)
    if open:
        webbrowser.open('file://' + os.path.abspath(result.html_path))
    return result.html_path

# Lint and coverage framework (added by analog-lint-and-coverage initiative)
from silk.lint import (
    Severity,
    Rule, RuleRegistry, VisitEvent, VisitOutcome, RegistryRunResult,
    Waiver, WaiverFile, WaiverMatcher, apply_waivers,
    ScopeExclusion, ExclusionFile,
)
from silk.lint.reports import (
    ReportMetadata, SCHEMA_VERSION,
    LintReport, LintDiff,
    CoverageReport, RuleCoverageReport, RuleCoverageNode,
)
from silk.lint.help import help_rule, list_rules
from silk.lint_markers import NoConnect

# Specs (design-stage bridge)
from silk.specs import (
    CalibrationBinding,
    Measurement,
    Spec,
    SpecOutcome,
    SpecReconciliationReport,
)

# Layout intelligence
from silk.layout import (
    RoutingModel,
    HalfCircuit,
    LayoutInferenceError,
    LayoutWarning,
    LayoutEstimate,
    NetLayout,
    DeviceLayout,
    estimate_layout,
    capture_layout,
    inject_parasitics,
    default_escalation_rule,
)

__all__ = [
    # circuit
    "Module", "ModuleTestbench", "Variable", "CornerResults", "MonteCarloResults",
    "Port", "PortDir", "PortRole",
    "Net", "PortBinding",
    "VoltageDomain",
    "check_interfaces", "check_voltage_domains",
    "require_annotation_coverage", "AnnotationCoverageError",
    "InterfaceViolation",
    "check_junction_overrange", "JunctionViolation",
    "RoleRuleset", "infer_port_roles",
    "list_unannotated_ports", "annotation_coverage", "UnannotatedPort", "AnnotationCoverage",
    "CircuitReport", "generate_report",
    "ReportDiff", "diff_reports",
    "export_schema", "module_schema", "export_schema_json",
    # pdk
    "BasePDK", "UNIT_PDK", "UNIT_SI", "load_pdk",
    "Normal", "Uniform",
    # simulation
    "Transient", "AC", "DC", "Noise", "Corner", "analysis",
    # signals
    "Signal", "NgspiceSignal",
    "view",
    # optimize
    "VariableSpace", "Optimizer",
    # db
    "Database", "DesignStage",
    # sources
    "sources",
    # codegen
    "Generator",
    # exceptions
    "CircuitError", "DuplicateNameError", "UndeclaredNetError",
    "SimulationError", "PDKError",
    # lint framework
    "Severity",
    "Rule", "RuleRegistry", "VisitEvent", "VisitOutcome", "RegistryRunResult",
    "Waiver", "WaiverFile", "WaiverMatcher", "apply_waivers",
    "ScopeExclusion", "ExclusionFile",
    # lint reports
    "ReportMetadata", "SCHEMA_VERSION",
    "LintReport", "LintDiff",
    "CoverageReport", "RuleCoverageReport", "RuleCoverageNode",
    # lint help
    "help_rule", "list_rules",
    # lint markers
    "NoConnect",
    # specs (design-stage bridge)
    "CalibrationBinding", "Measurement", "Spec",
    "SpecOutcome", "SpecReconciliationReport",
    # layout intelligence
    "RoutingModel",
    "HalfCircuit", "LayoutInferenceError",
    "LayoutWarning", "LayoutEstimate", "NetLayout", "DeviceLayout",
    "estimate_layout", "capture_layout", "inject_parasitics",
    "default_escalation_rule",
]
