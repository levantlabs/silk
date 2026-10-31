"""silk.db.context — runtime characterization-context descriptor.

A typed, immutable descriptor of *which registered test, on
which live owner, with which declared knobs, at which process/temperature/
voltage point* produced a piece of evidence.

This module is Python metadata only. It never touches netlist generation —
:class:`CharacterizationContext` instances ride on :class:`ModuleTestbench`,
:class:`~silk.circuit.spec_report.SpecReport`, and the generated simulation
``Module`` as plain attributes; nothing here writes to the database  or changes emitted SPICE bytes.

Two entry points:

- :class:`CharacterizationContext` — the frozen descriptor itself. Direct
  construction validates every field.
- :func:`build_characterization_context` — the "all six fields, or none of
  them" factory used by callers that assemble a context from optional
  pieces (e.g. a testbench run where PVT is only sometimes supplied). It
  returns ``None`` for the explicitly unscoped case and raises
  :class:`~silk.exceptions.CharacterizationContextError` for a partial one.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Any, Mapping, Optional

from silk.exceptions import CharacterizationContextError
from silk.utils._utils import _actual_values

if TYPE_CHECKING:
    from silk.simulation.conditions import ConditionState

# The canonical PVT field names carried by CharacterizationContext itself.
# A declared test parameter must never reuse one of these — see design §2.1.
RESERVED_PVT_PARAMETER_NAMES = frozenset({
    'process_corner', 'temperature_c', 'supply_voltage_v',
})

# The six facts a scoped characterization context carries (design §2.1).
_CONTEXT_FIELD_NAMES = (
    'owner_path', 'test_name', 'parameters', 'corner', 'temperature', 'voltage',
)


def freeze_parameters(parameters: Mapping[str, Any]) -> dict:
    """Validate and snapshot a declared test-parameter map.

    ``parameters`` must be a mapping with non-empty string keys; none of the
    keys may be one of :data:`RESERVED_PVT_PARAMETER_NAMES`. An empty mapping
    (``{}``) is valid — it means the test declares no parameters, not that
    the caller forgot to declare any.

    Returns a fresh ``dict`` that is deep-copied from ``parameters`` so later
    mutation of the caller's original object is never observable through the
    stored snapshot. (A plain ``dict`` rather than
    ``types.MappingProxyType`` deliberately — a proxy over a live mapping
    cannot be pickled or ``copy.deepcopy``-d, which would break
    ``Module.__reduce_ex__``/``__deepcopy__`` the moment a context is
    attached to a module. Treat the returned mapping as read-only by
    convention, the same way ``ModuleTestbench._specs`` is.)
    """
    if not isinstance(parameters, Mapping):
        raise CharacterizationContextError(
            "characterization context parameters must be a mapping "
            f"(dict-like); got {type(parameters).__name__}"
        )
    for key in parameters:
        if not isinstance(key, str) or not key:
            raise CharacterizationContextError(
                "characterization context parameter keys must be non-empty "
                f"strings; got {key!r}"
            )
        if key in RESERVED_PVT_PARAMETER_NAMES:
            raise CharacterizationContextError(
                f"{key!r} is a reserved PVT field name (one of "
                f"{sorted(RESERVED_PVT_PARAMETER_NAMES)}) and cannot also be "
                "used as a declared test parameter key"
            )
    try:
        return copy.deepcopy(dict(parameters))
    except Exception as exc:  # pragma: no cover - defensive, exercised via test
        raise CharacterizationContextError(
            "characterization context parameters must be deep-copyable "
            f"(snapshotting failed: {exc})"
        ) from exc


def _actual_parameter_values(parameters: dict) -> dict:
    """*parameters* with every Variable or Variable expression replaced by
    its current value; a Variable with no value raises
    :class:`~silk.exceptions.CharacterizationContextError`."""
    resolved = {}
    for key, value in parameters.items():
        try:
            resolved[key] = _actual_values(value)
        except ValueError as exc:
            raise CharacterizationContextError(
                f"characterization context parameter {key!r} holds a Variable "
                f"with no value ({exc}); assign it a value before the run"
            ) from exc
    return resolved


def _validate_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CharacterizationContextError(
            f"characterization context {field_name} must be a non-empty "
            f"string; got {value!r}"
        )
    return value


def _condition_state() -> type[ConditionState]:
    # Imported on use, not at module load: importing silk.simulation here
    # changes the order in which silk.metrics registers its metrics, and so
    # the byte order of export_metric_schema() (pinned by
    # tests/test_multi_output_metrics.py).
    from silk.simulation.conditions import ConditionState
    return ConditionState


def _validate_condition_state(value: ConditionState, field_name: str) -> ConditionState:
    """Accept ``ConditionState.NONE``; refuse the members that cannot scope.

    A profile is a promise that evidence can be selected again at the same
    point, so each fact must be a value or none.
    """
    ConditionState = _condition_state()
    if value is ConditionState.NONE:
        return value
    reason = {
        ConditionState.UNKNOWN: 'unknown cannot form a profile (pass None for an '
                                'unscoped context instead)',
        ConditionState.SWEEP: 'a sweep cannot form a profile',
        ConditionState.FIXED: 'FIXED is a state, not a value (pass the value itself)',
    }[value]
    raise CharacterizationContextError(
        f"characterization context {field_name} must be a value or "
        f"ConditionState.NONE; got {value}: {reason}"
    )


def _validate_corner(value: Any) -> str | ConditionState:
    # Checked before the generic Enum reduction below: a plain-Enum NONE
    # reduced to its value would become the corner named 'none'.
    if isinstance(value, _condition_state()):
        return _validate_condition_state(value, 'corner')
    if isinstance(value, Enum):
        value = value.value
    return _validate_nonempty_str(value, 'corner')


def _validate_finite_real(value: Any, field_name: str) -> float | ConditionState:
    if isinstance(value, _condition_state()):
        return _validate_condition_state(value, field_name)
    if isinstance(value, bool):
        raise CharacterizationContextError(
            f"characterization context {field_name} must be a real number, "
            "not bool"
        )
    if not isinstance(value, Real):
        raise CharacterizationContextError(
            f"characterization context {field_name} must be a real number; "
            f"got {type(value).__name__}"
        )
    as_float = float(value)
    if not math.isfinite(as_float):
        raise CharacterizationContextError(
            f"characterization context {field_name} must be finite; got "
            f"{value!r}"
        )
    if as_float == 0.0:
        # Design §2.3: PVT selection compares exact stored numeric values
        # with no tolerance, so -0.0 and 0.0 (equal under ==, distinct under
        # math.copysign) must not silently create two selection keys for the
        # same nominal point. Normalize to positive zero.
        as_float = 0.0
    return as_float


@dataclass(frozen=True)
class CharacterizationContext:
    """Immutable runtime descriptor of one characterization context.

    A context names which registered test, on which live owner, with which
    declared parameters, at which process, temperature and supply point,
    produced a piece of evidence. All six fields are required and are validated
    and normalised when the object is built. The factory
    ``build_characterization_context`` adds the "all six fields or none"
    convenience for callers that assemble a context from optional pieces. A ``Variable`` in ``parameters`` is recorded
    as its current value.

    Parameters
    ----------
    owner_path : str
        Root or instance path of the module that owns the registered test.
    test_name : str
        Exact name of the test.
    parameters : dict
        Complete effective declared parameter map.
    corner : str or ConditionState
        Process corner.
    temperature : float or ConditionState
        Temperature in degrees Celsius.
    voltage : float or ConditionState
        Primary supply volts.

    Attributes
    ----------
    owner_path : str
        Root/instance-path form of the live module that owns the registered
        test (``$``, ``$/amp1``, ...); instance segments already carry the
        ``~0``/``~1`` escaping used elsewhere in ``silk.db``.
    test_name : str
        Exact, unmodified test name (unique per module).
    parameters : dict
        Complete effective declared parameter map with defaults
        materialized. ``{}`` is valid. Treat it as read-only.
    corner : str or ConditionState
        Process-corner string. A ``Corner`` enum value is reduced to its
        exact string value. ``ConditionState.NONE`` records that no process
        corner applies (for example a deck with no PDK device).
    temperature : float or ConditionState
        Temperature in degrees Celsius. Finite; bool is rejected; ``-0.0`` is normalised to
        ``0.0``. ``ConditionState.NONE`` records that no ambient temperature
        applies (behavioural evidence).
    voltage : float or ConditionState
        Primary supply volts, with the same rules as ``temperature``.
        ``ConditionState.NONE`` records that the circuit has no supply.

    Raises
    ------
    silk.exceptions.CharacterizationContextError
        A field is invalid: ``owner_path`` or ``test_name`` is not a non-empty
        string, ``parameters`` is not a mapping with non-empty string keys
        (none of them ``process_corner``, ``temperature_c`` or
        ``supply_voltage_v``) that can be deep-copied, a parameter holds a
        ``Variable`` with no value, or a corner, temperature or voltage is not a
        value or ``ConditionState.NONE`` (unknown, sweep and fixed states are
        refused; a non-finite number or bool is refused).

    Notes
    -----
    None of the three PVT facts enters the profile hash (``silk.profile/v1``),
    so a none-bearing context shares its profile with value-bearing contexts of
    the same test.

    Examples
    --------
    >>> from silk.db.context import CharacterizationContext
    >>> context = CharacterizationContext(
    ...     owner_path='$', test_name='dc_gain', parameters={'ibias': 1e-05},
    ...     corner='tt', temperature=27, voltage=1.8)
    >>> context.temperature
    27.0
    """

    owner_path: str
    test_name: str
    parameters: dict
    corner: str | ConditionState
    temperature: float | ConditionState
    voltage: float | ConditionState

    def __post_init__(self) -> None:
        owner_path = _validate_nonempty_str(self.owner_path, 'owner_path')
        test_name = _validate_nonempty_str(self.test_name, 'test_name')
        # A Variable declared as a test parameter is recorded as its value
        # now, when the run's context is built (TASK-402): the profile must
        # name the design point the run used, and a later assignment to the
        # Variable must not change this record.
        parameters = _actual_parameter_values(freeze_parameters(self.parameters))
        corner = _validate_corner(self.corner)
        temperature = _validate_finite_real(self.temperature, 'temperature')
        voltage = _validate_finite_real(self.voltage, 'voltage')
        # frozen dataclasses forbid plain attribute assignment even from
        # inside their own __post_init__; object.__setattr__ is the
        # documented escape hatch for normalizing fields at construction.
        object.__setattr__(self, 'owner_path', owner_path)
        object.__setattr__(self, 'test_name', test_name)
        object.__setattr__(self, 'parameters', parameters)
        object.__setattr__(self, 'corner', corner)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'voltage', voltage)


def build_characterization_context(
    *,
    owner_path: Optional[str] = None,
    test_name: Optional[str] = None,
    parameters: Optional[Mapping[str, Any]] = None,
    corner: Optional[Any] = None,
    temperature: Optional[Any] = None,
    voltage: Optional[Any] = None,
) -> Optional[CharacterizationContext]:
    """Build a :class:`CharacterizationContext` from its six declared facts.

    Profile-scoped writes are all-or-nothing. A partially
    supplied descriptor raises a typed, actionable error before a profile or
    evidence row is written. This factory is the seam that enforces exactly
    that, for callers assembling a context from optional pieces:

    - Supplying *none* of the six fields returns ``None`` — the explicitly
      unscoped path. Nothing is inferred or fabricated.
    - Supplying *some but not all* raises
      :class:`~silk.exceptions.CharacterizationContextError` naming every
      missing field.
    - Supplying *all six* returns a validated, immutable
      :class:`CharacterizationContext`.

    ``parameters={}`` counts as supplied (it is a real declared empty
    parameter map, not an omission), and so does ``ConditionState.NONE`` for
    corner, temperature or voltage (the fact provably does not apply); only
    ``None`` counts as "not given."
    """
    provided = {
        'owner_path': owner_path,
        'test_name': test_name,
        'parameters': parameters,
        'corner': corner,
        'temperature': temperature,
        'voltage': voltage,
    }
    missing = [name for name in _CONTEXT_FIELD_NAMES if provided[name] is None]
    if len(missing) == len(_CONTEXT_FIELD_NAMES):
        return None
    if missing:
        raise CharacterizationContextError(
            "Incomplete characterization context: missing "
            f"{', '.join(missing)}. A scoped characterization context "
            "requires owner_path, test_name, parameters, corner, "
            "temperature, and voltage together, or none of them "
            "(explicitly unscoped)."
        )
    return CharacterizationContext(
        owner_path=owner_path,
        test_name=test_name,
        parameters=parameters,
        corner=corner,
        temperature=temperature,
        voltage=voltage,
    )


__all__ = [
    'CharacterizationContext',
    'RESERVED_PVT_PARAMETER_NAMES',
    'build_characterization_context',
    'freeze_parameters',
]
