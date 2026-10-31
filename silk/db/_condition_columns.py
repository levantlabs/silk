"""Row columns for operating-condition states (schema v7).

Each of ``simulation_results``, ``spec_measurements`` and
``net_current_sets`` stores corner, temperature and voltage as a state column
plus a value column (design 2026-10-01 operating-conditions-none, §5.1). In
Python the pair is one widened scalar (§2.4):

- a value is a ``str`` (corner) or a ``float`` -> state ``'fixed'``;
- ``ConditionState.NONE`` -> state ``'none'``, value NULL;
- ``None`` (unknown) -> state ``'unknown'``, value NULL;
- ``ConditionState.SWEEP`` (temperature or supply) -> state ``'sweep'``, value NULL.

Every writer derives the state from the scalar here, so a state can never
disagree with its value, and every reader and selector maps it back here.
"""

from __future__ import annotations

from enum import Enum
from numbers import Real
from typing import Any, Optional, Tuple

#: Allowed states per row column, as stored (design §5.1).
COLUMN_STATES = {
    "corner": ("fixed", "none", "unknown"),
    "temperature": ("fixed", "none", "unknown", "sweep"),
    "voltage": ("fixed", "none", "unknown", "sweep"),
}

#: States that may carry a characterization profile (design §2.2).
PROFILE_STATES = ("fixed", "none")


def _condition_state():
    # Imported on use, not at module load (see silk.db.context._condition_state:
    # importing silk.simulation at load reorders metric registration).
    from silk.simulation.conditions import ConditionState
    return ConditionState


def state_columns_sql(*, schema_version=7) -> str:
    """State columns and CHECKs for a current or historical CREATE TABLE body."""
    lines = []
    for column, states in COLUMN_STATES.items():
        if column == "voltage" and schema_version < 7:
            states = tuple(state for state in states if state != "sweep")
        allowed = ",".join(f"'{state}'" for state in states)
        lines.append(
            f"{column}_state TEXT NOT NULL CHECK({column}_state IN ({allowed}))"
        )
    for column in COLUMN_STATES:
        lines.append(
            f"CHECK(({column}_state = 'fixed') = ({column} IS NOT NULL))"
        )
    return ",\n                ".join(lines)


def profile_check_sql() -> str:
    """The v5 profile CHECK: a scoped row has every fact fixed or none."""
    allowed = ",".join(f"'{state}'" for state in PROFILE_STATES)
    terms = " AND\n                       ".join(
        f"{column}_state IN ({allowed})" for column in COLUMN_STATES
    )
    return f"CHECK(profile_id IS NULL OR\n                      ({terms}))"


def condition_columns(
    column: str,
    scalar: Any,
    *,
    error_type: type[Exception] = TypeError,
    subject: str = "Condition",
) -> Tuple[str, Any]:
    """Return ``(state, column_value)`` for one widened condition scalar.

    A str-mixin corner enum (``Corner.TT``) is stored as its string value.
    ``ConditionState.FIXED`` / ``UNKNOWN`` are states, not scalars, and
    ``SWEEP`` applies to temperature or supply; each raises ``error_type``.
    """
    ConditionState = _condition_state()
    if scalar is None:
        return "unknown", None
    if isinstance(scalar, ConditionState):
        if scalar is ConditionState.NONE:
            return "none", None
        if scalar is ConditionState.SWEEP and column in ("temperature", "voltage"):
            return "sweep", None
        raise error_type(
            f"{subject} {column} cannot be {scalar}: pass the value itself, "
            "ConditionState.NONE, None for unknown"
            + (", or ConditionState.SWEEP" if column in ("temperature", "voltage") else "")
        )
    if column == "corner":
        if isinstance(scalar, Enum) and isinstance(scalar, str):
            scalar = scalar.value
        if not isinstance(scalar, str):
            raise error_type(
                f"{subject} corner must be a string, ConditionState.NONE or None"
            )
        return "fixed", scalar
    if isinstance(scalar, bool) or not isinstance(scalar, Real):
        raise error_type(
            f"{subject} {column} must be a real number, ConditionState.NONE "
            "or None"
        )
    return "fixed", float(scalar)


def condition_scalar(column: str, state: Optional[str], value: Any):
    """Map a stored ``(state, value)`` pair back to the widened scalar."""
    ConditionState = _condition_state()
    if state == "fixed":
        return value
    if state == "unknown":
        return None
    if state in COLUMN_STATES[column]:
        return ConditionState(state)
    raise ValueError(f"Stored {column}_state {state!r} is not a {column} state")


def condition_predicate(
    column: str,
    scalar: Any,
    *,
    alias: Optional[str] = None,
    subject: str = "Condition selector",
) -> Tuple[list, list]:
    """SQL clauses/params matching one condition selector, NULL-safely.

    ``None`` is "no filter" (an omitted selector, unchanged meaning). Any
    other scalar matches ``x_state = ? AND x IS ?``: the state tells none
    from unknown and ``IS`` compares the NULL value of a stateful row.
    """
    if scalar is None:
        return [], []
    state, value = condition_columns(column, scalar, subject=subject)
    prefix = f"{alias}." if alias else ""
    return (
        [f"{prefix}{column}_state = ?", f"{prefix}{column} IS ?"],
        [state, value],
    )


__all__ = [
    "COLUMN_STATES",
    "PROFILE_STATES",
    "condition_columns",
    "condition_predicate",
    "condition_scalar",
    "profile_check_sql",
    "state_columns_sql",
]
