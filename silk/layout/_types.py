"""silk.layout._types — shared frozen dataclasses for the layout package.

Separated from estimator.py to avoid circular imports with matching.py.
Not part of the public API — import from silk.layout instead.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LayoutWarning:
    """A non-fatal issue discovered during layout estimation.

    Warnings are run-scoped: they live on the returned ``LayoutEstimate``,
    not on ``Net`` objects.  They expire when the next estimation run
    produces a fresh estimate.

    Parameters
    ----------
    net_name : str or None
        The net the warning pertains to. ``None`` for module-level warnings.
    reason : str
        Machine-readable tag: ``'no_current_data'``,
        ``'matching_conflict_large_class'``, ``'multi_pin_rc_approximation'``,
        ``'lumped_approximation_coarse'``, ``'required_width_exceeds_max'``
        (the current-based required width for the selected layer exceeds
        that layer's ``max_width``; the single-wire estimate is capped —
        this does NOT mean the circuit is unroutable, and it does not
        endorse any specific multi-wire or alternate-layer remedy).
    severity : str
        ``'info'``, ``'warn'``, or ``'error'``.
    detail : str
        Human-readable explanation shown in the report.
    """
    net_name: Optional[str]
    reason: str
    severity: str
    detail: str
