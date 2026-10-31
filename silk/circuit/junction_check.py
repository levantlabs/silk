"""Backward-compat wrapper for check_junction_overrange.

The rule implementation (including the hierarchical + SPICE walk) lives in
silk/lint/rules/stress.py — this module has no detection logic of its own,
it just dispatches through the registry. This module re-exports the
:class:`silk.lint.violations.JunctionViolation` type so legacy imports keep
working.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

from silk.lint.stress_coverage import StressCoverage, DeviceStressCoverage

from silk.lint.registry import RuleRegistry, module_has_simulation_data
from silk.lint.violations import JunctionViolation  # re-export for legacy imports


if TYPE_CHECKING:
    from silk.circuit.modules import Module


def check_junction_overrange(module) -> List[JunctionViolation]:
    """Run STR-001 junction overrange stress check.

    Skips silently if *module* has neither behavioral nor SPICE simulation
    data (returns ``[]``).

    Parameters
    ----------
    module : Module
        Top-level module that has been simulated, either via
        ``simulate_behavioral()`` or a real SPICE run (``simulate()``).

    Returns
    -------
    list of JunctionViolation
        Empty list if no violations or simulation has not run.
    """
    reg = RuleRegistry.default()
    result = reg.run_category(
        "stress", module, has_simulation_data=module_has_simulation_data(module))
    return result.violations


def junction_stress_coverage(module: Module) -> StressCoverage:
    """Return STR-001 device coverage from its shared evaluation pass.

    No violations does not imply full coverage. Failure overlaps coverage:
    a device with a failing rated pair and an unrated pair is failed AND partial.
    Missing simulation data is reported as no_waveform, not a passing check.

    Parameters
    ----------
    module : Module
        Circuit whose current simulation data should be checked.

    Returns
    -------
    StressCoverage
        Per-device outcomes and independent coverage/failure counts. Evaluates
        current results afresh; does not run a simulation.
    """
    # Local import: the rule catalog imports the circuit package.
    from silk.lint.rules.stress import _evaluate_stress
    return _evaluate_stress(module)[0]


__all__ = ["check_junction_overrange", "JunctionViolation",
           "junction_stress_coverage", "StressCoverage", "DeviceStressCoverage"]
