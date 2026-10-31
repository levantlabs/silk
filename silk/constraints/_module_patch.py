"""Idempotent monkey-patch attaching constraint methods to Module.

Per ``silk/constraints/DESIGN.md``: zero edits to ``silk/circuit``;
``Module.add_constraint`` lives here so the constraint package is the
single source of truth for the API surface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from silk.constraints.envelope import ConstraintEnvelope


def _add_constraint(self, constraint: "ConstraintEnvelope") -> "ConstraintEnvelope":
    """Attach a layout-intent constraint to this module.

    Available as ``Module.add_constraint`` once :mod:`silk.constraints` has
    been imported.

    Parameters
    ----------
    constraint : ConstraintEnvelope
        A constraint instance, e.g. ``SymmetricBlocks`` or ``GroupBlocks``.

    Returns
    -------
    ConstraintEnvelope
        The same object, so a call can be written as an assignment.

    Raises
    ------
    TypeError
        If ``constraint`` is not a :class:`silk.constraints.ConstraintEnvelope`.
    """
    from silk.constraints.envelope import ConstraintEnvelope

    if not isinstance(constraint, ConstraintEnvelope):
        raise TypeError(
            f"add_constraint() expects a ConstraintEnvelope subclass; got "
            f"{type(constraint).__name__}"
        )
    if not hasattr(self, "_constraints"):
        self._constraints = []
    self._constraints.append(constraint)
    return constraint


def _list_constraints(self) -> List["ConstraintEnvelope"]:
    """Return the constraints added to this module, in insertion order.

    Available as ``Module.list_constraints`` once :mod:`silk.constraints`
    has been imported.

    Returns
    -------
    list of ConstraintEnvelope
        A new list; changing it does not change the module.
    """
    if not hasattr(self, "_constraints"):
        self._constraints = []
    return list(self._constraints)


def install() -> None:
    """Attach add_constraint / list_constraints to Module if absent."""
    from silk.circuit.modules import Module

    if not hasattr(Module, "add_constraint"):
        Module.add_constraint = _add_constraint  # type: ignore[attr-defined]
    if not hasattr(Module, "list_constraints"):
        Module.list_constraints = _list_constraints  # type: ignore[attr-defined]
