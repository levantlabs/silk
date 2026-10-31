"""Analog design-rule lint and rule-coverage framework.

A :class:`Rule` inspects a design (usually a :class:`~silk.circuit.Module`) and
yields :class:`Violation` findings, each pointing at a
:class:`ViolationLocation` and carrying a :class:`Severity`. A
:class:`RuleRegistry` holds the rules, dispatches them over a scope and
records a :class:`VisitEvent` per dispatch in a :class:`RegistryRunResult`.
Findings can be suppressed with a :class:`Waiver` (loaded from YAML through
:class:`WaiverFile`) or skipped by subtree with a :class:`ScopeExclusion`.
:class:`LintReport` collects violations, waivers and intentional-exception
markers for JSON, HTML and JUnit output; :class:`CoverageReport` turns a visit
log into per-rule, per-module checked/unchecked counts.

Everything above is re-exported at ``silk.lint`` and much of it at ``silk``.
Rule classes live in ``silk.lint.rules``; ``silk.lint.help.help_rule`` and
``silk.lint.help.list_rules`` describe the shipped catalog.

Examples
--------
>>> from silk.lint import RuleRegistry, Severity
>>> registry = RuleRegistry.default()
>>> registry.rules[0].rule_id
'INT-001'
>>> Severity.from_string('error') > Severity.WARNING
True
"""
from silk.lint.severity import Severity
from silk.lint.locations import (
    ViolationLocation,
    NetLocation,
    PortLocation,
    InstanceLocation,
    TerminalLocation,
    ModuleLocation,
)

__all__ = [
    "Severity",
    "ViolationLocation",
    "NetLocation",
    "PortLocation",
    "InstanceLocation",
    "TerminalLocation",
    "ModuleLocation",
]

from silk.lint.violations import Violation, InterfaceViolation, JunctionViolation

__all__ += ["Violation", "InterfaceViolation", "JunctionViolation"]

from silk.lint.rule import Rule, RuleCategory, VALID_CATEGORIES

__all__ += ["Rule", "RuleCategory", "VALID_CATEGORIES"]

from silk.lint.registry import (
    RuleRegistry, VisitEvent, VisitOutcome, RegistryRunResult,
    module_has_simulation_data,
)

__all__ += [
    "RuleRegistry",
    "VisitEvent",
    "VisitOutcome",
    "RegistryRunResult",
    "module_has_simulation_data",
]

from silk.lint.waivers import Waiver, WaiverFile, WaiverMatcher, apply_waivers

__all__ += ["Waiver", "WaiverFile", "WaiverMatcher", "apply_waivers"]

from silk.lint.exclusions import ScopeExclusion, ExclusionFile

__all__ += ["ScopeExclusion", "ExclusionFile"]

from silk.lint.reports import (
    ReportMetadata, SCHEMA_VERSION, LintReport, LintDiff,
    CoverageReport, RuleCoverageReport, RuleCoverageNode,
)

__all__ += [
    "ReportMetadata", "SCHEMA_VERSION", "LintReport", "LintDiff",
    "CoverageReport", "RuleCoverageReport", "RuleCoverageNode",
]
