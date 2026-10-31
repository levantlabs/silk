"""Operating-condition configuration and immutable evidence from one run.

Only top-level literal caller settings are recognized. Registered model
libraries remain trusted; a startup log cannot certify arbitrary control code.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from silk.circuit.modules import Module
    from silk.db.context import CharacterizationContext
    from silk.simulation.backend import RawResult

from silk.exceptions import (
    CharacterizationContextError, EvidenceMismatchError, NotSupportedError,
    SimulationConfigurationError, SimulationError,
)
from silk.simulation.analysis import DC
from silk.simulators import load_simulator_config
from silk.utils._utils import _parse_si, _render_param


# ---------------------------------------------------------------------------
# Condition states (TASK-394 design, revision 3, sections 2 and 4.1)
#
# Each of corner, temperature and supply is in exactly one state: a value
# ('fixed' on the wire), none (provably does not apply), unknown (applies but
# silk cannot establish it) or sweep (temperature or supply). In a scalar slot a
# value is a str (corner) or a float, none is ConditionState.NONE, unknown is
# None and sweep is ConditionState.SWEEP.
#
# ConditionState is deliberately a PLAIN Enum, not a str mixin: a str-mixin
# NONE equals the string 'none' and so collides with a corner of that name,
# and it is written out silently by json.dumps and SQLite. A plain member
# equals only itself, and json.dumps, SQLite binds and float() all refuse it,
# so a writer that forgot encode_condition fails instead of writing "none".
# ---------------------------------------------------------------------------

class ConditionState(Enum):
    """The state of one operating condition (corner, temperature or supply).

    A plain :class:`~enum.Enum`, deliberately not a ``str`` subclass: a member
    equals only itself, so ``NONE`` can never be mistaken for a corner named
    ``'none'``, and JSON or SQLite writers refuse it unless
    :func:`encode_condition` was used. The member values are the wire
    spellings.

    Attributes
    ----------
    FIXED : ConditionState
        The condition has a value (``'fixed'``).
    NONE : ConditionState
        The condition provably does not apply (``'none'``).
    UNKNOWN : ConditionState
        The condition applies but silk cannot establish it (``'unknown'``).
    SWEEP : ConditionState
        The condition was swept across a range; temperature or supply
        (``'sweep'``).

    Examples
    --------
    >>> from silk.simulation import ConditionState
    >>> ConditionState('sweep') is ConditionState.SWEEP
    True
    """

    FIXED = 'fixed'
    NONE = 'none'
    UNKNOWN = 'unknown'
    SWEEP = 'sweep'


#: Version of :func:`export_conditions_schema`'s payload.
CONDITIONS_SCHEMA_FORMAT_VERSION = '1.3'

#: The closed set of categories a basis word falls into (design revision 3,
#: A4). A consumer labels a basis by its exact text and groups one it does not
#: know by its category. ``default`` is reserved: no basis word uses it yet.
BASIS_CATEGORIES = ('observed', 'configured', 'declared', 'proven',
                    'default', 'legacy', 'unknown')

#: Per fact: the states it may take, the states that may scope a
#: characterization profile, and the basis words for each state. Data, not
#: code; read it through :func:`export_conditions_schema` rather than
#: hard-coding the strings.
CONDITION_VOCABULARY = {
    'corner': {
        'states': ('fixed', 'none', 'unknown'),
        'profile_states': ('fixed', 'none'),
        'bases': {
            # 'legacy' marks a corner migrated from a pre-v5 database record,
            # whose basis was never recorded (design section 5.2).
            'fixed': ('configured-and-emitted', 'caller-declared', 'legacy'),
            'none': ('proven-no-pdk-device', 'proven-behavioral', 'caller-declared'),
            'unknown': ('unknown',),
        },
        'categories': {
            'configured-and-emitted': 'configured',
            'caller-declared': 'declared',
            'legacy': 'legacy',
            'proven-no-pdk-device': 'proven',
            'proven-behavioral': 'proven',
            'unknown': 'unknown',
        },
    },
    'temperature': {
        'states': ('fixed', 'none', 'unknown', 'sweep'),
        'profile_states': ('fixed', 'none'),
        'bases': {
            'fixed': ('configured-and-observed', 'native-and-observed',
                      'raw-and-observed', 'backend-observed', 'caller-declared'),
            'none': ('proven-behavioral', 'caller-declared'),
            'unknown': ('unknown',),
            'sweep': ('configured', 'unknown'),
        },
        'categories': {
            'configured-and-observed': 'observed',
            'native-and-observed': 'observed',
            'raw-and-observed': 'observed',
            'backend-observed': 'observed',
            'caller-declared': 'declared',
            'proven-behavioral': 'proven',
            'configured': 'configured',
            'unknown': 'unknown',
        },
    },
    'supply': {
        'states': ('fixed', 'none', 'unknown', 'sweep'),
        'profile_states': ('fixed', 'none'),
        'bases': {
            # 'configured-and-driven': a behavioural run drove the configured
            # source at the configured value; there is no deck to emit into
            # (TASK-448.3, design 2026-10-02 section 7.3).
            'fixed': ('configured-and-emitted', 'configured-and-driven',
                      'declared-and-emitted', 'caller-declared'),
            'none': ('proven-no-supply', 'caller-declared'),
            'unknown': ('unknown',),
            'sweep': ('configured', 'unknown'),
        },
        'categories': {
            'configured-and-emitted': 'configured',
            'configured-and-driven': 'configured',
            'declared-and-emitted': 'declared',
            'caller-declared': 'declared',
            'proven-no-supply': 'proven',
            'unknown': 'unknown',
            'configured': 'configured',
        },
    },
}

#: The tagged wire form is reserved for the two states that are not a value
#: and not unknown; a value is itself and unknown is null.
_TAGGED_STATES = (ConditionState.NONE, ConditionState.SWEEP)


def _check_fact(fact: str) -> dict:
    try:
        return CONDITION_VOCABULARY[fact]
    except (KeyError, TypeError):
        raise ValueError(
            f'Unknown operating-condition fact {fact!r}; expected one of '
            f'{sorted(CONDITION_VOCABULARY)}.') from None


def _is_real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def encode_condition(value):
    """Encode one condition scalar for JSON. This is the only wire encoder.

    A value (a corner ``str`` or a real number) encodes as itself, ``None``
    (unknown) as ``None``, and ``ConditionState.NONE`` / ``SWEEP`` as the
    tagged object ``{'state': 'none'}`` / ``{'state': 'sweep'}``, which no
    corner name, number or null can be mistaken for. A ``str``-subclass enum
    member such as :class:`~silk.simulation.Corner` ``TT`` is written as its
    plain string value.

    Parameters
    ----------
    value : str or float or None or ConditionState
        A condition scalar: a corner name, a real number (``bool`` excluded),
        ``None``, ``ConditionState.NONE`` or ``ConditionState.SWEEP``.

    Returns
    -------
    str or float or None or dict
        The JSON-ready form.

    Raises
    ------
    ValueError
        If *value* is ``ConditionState.FIXED`` or ``UNKNOWN``, which are
        states, not scalars.
    TypeError
        If *value* is of any other type.

    See Also
    --------
    decode_condition : The inverse.

    Examples
    --------
    >>> from silk.simulation import ConditionState, encode_condition
    >>> encode_condition(ConditionState.NONE)
    {'state': 'none'}
    >>> encode_condition('tt'), encode_condition(1.8), encode_condition(None)
    ('tt', 1.8, None)
    """
    if value is None:
        return None
    if isinstance(value, ConditionState):
        if value not in _TAGGED_STATES:
            raise ValueError(
                f'{value} is not a condition scalar: a value is written as '
                'itself and unknown as None.')
        return {'state': value.value}
    if isinstance(value, str):
        # A str-mixin enum (Corner.TT) is written as its plain string.
        return value.value if isinstance(value, Enum) else value
    if _is_real(value):
        return value
    raise TypeError(
        f'Cannot encode condition scalar of type {type(value).__name__}; '
        'expected a str, a real number, None, ConditionState.NONE or SWEEP.')


def decode_condition(fact: str, wire):
    """Decode one wire condition scalar for *fact*. This is the only wire decoder.

    The inverse of :func:`encode_condition`. ``None`` is unknown. A corner
    value must be a non-empty string and a temperature or supply value a
    real number (returned as a ``float``). A tagged object must be exactly
    ``{'state': 'none'}`` or ``{'state': 'sweep'}`` and name a state that
    *fact* allows (``'sweep'`` is allowed for temperature or supply).

    Parameters
    ----------
    fact : {'corner', 'temperature', 'supply'}
        Which condition the scalar belongs to.
    wire : str or float or dict or None
        The wire form.

    Returns
    -------
    str or float or None or ConditionState
        The corner name, the real value, ``None`` for unknown, or
        ``ConditionState.NONE`` / ``SWEEP`` for a tagged state.

    Raises
    ------
    ValueError
        If *fact* is not a known fact, or *wire* is not a valid wire form
        for it.

    See Also
    --------
    encode_condition : The inverse.

    Examples
    --------
    >>> from silk.simulation import ConditionState, decode_condition
    >>> decode_condition('temperature', {'state': 'sweep'}) is ConditionState.SWEEP
    True
    >>> decode_condition('supply', 1)
    1.0
    """
    vocabulary = _check_fact(fact)
    if wire is None:
        return None
    if isinstance(wire, dict):
        allowed = [m.value for m in _TAGGED_STATES if m.value in vocabulary['states']]
        state = wire.get('state') if set(wire) == {'state'} else None
        if state in allowed:
            return ConditionState(state)
        raise ValueError(
            f'Invalid {fact} condition {wire!r}: a tagged {fact} state is one of '
            f'{[{"state": s} for s in allowed]}.')
    if fact == 'corner':
        if isinstance(wire, str) and wire:
            return wire
        raise ValueError(f'Invalid corner condition {wire!r}: a corner value is a non-empty string.')
    if _is_real(wire):
        return float(wire)
    raise ValueError(f'Invalid {fact} condition {wire!r}: a {fact} value is a real number.')


def export_conditions_schema() -> dict:
    """Return a JSON-ready, versioned copy of :data:`silk.simulation.CONDITION_VOCABULARY`.

    Consumers read the states, the profile-forming states and the basis
    words from here instead of hard-coding the strings. Each fact's
    ``categories`` maps every basis word to one of the schema's top-level
    ``categories``.

    Returns
    -------
    dict
        Keys: ``format_version`` (str), ``categories`` (list of str, the
        closed set of basis categories), ``wire`` (the wire encodings of the
        unknown and tagged states) and ``facts``. ``facts`` maps
        ``'corner'``, ``'temperature'`` and ``'supply'`` each to a dict with
        ``states``, ``profile_states``, ``bases`` (state to basis words) and
        ``categories`` (basis word to category). Lists and dicts are
        copies, so mutating the result does not change the vocabulary.

    Examples
    --------
    >>> from silk.simulation import export_conditions_schema
    >>> schema = export_conditions_schema()
    >>> sorted(schema['facts'])
    ['corner', 'supply', 'temperature']
    >>> schema['facts']['temperature']['states']
    ['fixed', 'none', 'unknown', 'sweep']
    """
    facts = {
        fact: {
            'states': list(entry['states']),
            'profile_states': list(entry['profile_states']),
            'bases': {state: list(words) for state, words in entry['bases'].items()},
            'categories': dict(entry['categories']),
        }
        for fact, entry in CONDITION_VOCABULARY.items()
    }
    return {
        'format_version': CONDITIONS_SCHEMA_FORMAT_VERSION,
        'categories': list(BASIS_CATEGORIES),
        'wire': {
            'unknown': None,
            'tagged': {member.value: {'state': member.value} for member in _TAGGED_STATES},
        },
        'facts': facts,
    }


@dataclass(frozen=True)
class ConditionSweep:
    """Immutable DC condition bounds; units follow the containing field.

    ``temperature_sweep`` uses degrees Celsius; ``supply_sweep`` uses volts.
    Start, stop and step are the configured numeric analysis bounds.

    Parameters
    ----------
    start : float
        First configured value, in the containing field's units.
    stop : float
        Last configured value, in the containing field's units.
    step : float
        Configured increment, in the containing field's units.

    Examples
    --------
    >>> from silk.simulation import ConditionSweep
    >>> ConditionSweep(start=-40.0, stop=125.0, step=5.0).stop
    125.0
    """

    #: First configured value, in the containing field's units.
    start: float
    #: Last configured value, in the containing field's units.
    stop: float
    #: Configured increment, in the containing field's units.
    step: float


@dataclass(frozen=True)
class OperatingConditions:
    """Conditions of loaded evidence, independent of complete-PVT identity.

    Immutable provenance for one set of results: which process corner,
    ambient temperature and primary-supply voltage they correspond to, and
    how silk came to know it. Backend observations establish the ambient
    ``TEMP``, not individual junction temperatures.

    Each fact has a kind (``corner_kind``, ``temperature_kind``,
    ``supply_kind``), one of the wire spellings of :class:`ConditionState`
    (``'fixed'``, ``'none'``, ``'unknown'``, and ``'sweep'`` for
    temperature or supply). The value fields ``corner``, ``temperature_c`` and
    ``supply_voltage_v`` are set exactly when their kind is ``'fixed'``;
    they hold ``None`` for both none and unknown, and never a
    :class:`ConditionState` member. Read the kind with :meth:`state` and the
    typed scalar with :meth:`scalar`. ``temperature_sweep`` is set exactly
    when ``temperature_kind`` is ``'sweep'``; ``supply_sweep`` follows
    ``supply_kind`` in the same way.

    A caller that leaves ``corner_kind`` or ``supply_kind`` at ``'unknown'``
    while giving a value gets ``'fixed'`` inferred, so positional
    ``OperatingConditions('silk', backend, corner)`` keeps its meaning. Only
    the kind is inferred, never the basis. (``temperature_kind`` is not
    inferred.)

    Parameters
    ----------
    origin : str
        Where the results came from: ``'silk'`` for a run silk performed,
        ``'external'`` for results loaded from outside.
    backend : str or None
        Name of the simulator backend (for example ``'ngspice'``), or
        ``None`` if not known.
    corner : str or None
        Process corner name; set exactly when ``corner_kind`` is ``'fixed'``.
    temperature_kind : str, optional
        State of the temperature fact; one of the :class:`ConditionState`
        values. Defaults to ``'unknown'``.
    temperature_c : float or None, optional
        Ambient temperature in degrees Celsius; set exactly when
        ``temperature_kind`` is ``'fixed'``. It is absent for unknown and
        sweep-wide evidence.
    temperature_basis : str, optional
        Why silk holds that temperature state, a word from
        ``CONDITION_VOCABULARY['temperature']['bases']``. Defaults to
        ``'unknown'``.
    temperature_sweep : ConditionSweep or None, optional
        The swept range; set exactly when ``temperature_kind`` is
        ``'sweep'``.
    supply_voltage_v : float or None, optional
        Primary-supply setpoint of the deck that was launched, in volts,
        confirmed against that deck's own source line. Absent unless someone
        established it (``set_supply_voltage(...)`` or a ``run(voltage=...)``
        assertion). It is a claim about the DC value the simulator was
        given, never a measurement of the rail the circuit settled at.
    supply_basis : str, optional
        Why silk holds that supply state, a word from
        ``CONDITION_VOCABULARY['supply']['bases']``. Defaults to
        ``'unknown'``.
    supply_source : str or None, optional
        Name of the source the supply voltage was read from.
    corner_kind : str, optional
        State of the corner fact. Defaults to ``'unknown'``.
    corner_basis : str, optional
        Why silk holds that corner state, a word from
        ``CONDITION_VOCABULARY['corner']['bases']``. Defaults to
        ``'unknown'``.
    supply_sweep : ConditionSweep or None, optional
        Configured DC bounds in volts when the designated primary supply is
        swept; supply_voltage_v is then None. This is launch-wide evidence,
        including mixed-analysis launches, and cannot scope a fixed profile.
    supply_kind : str, optional
        State of the supply fact. Defaults to ``'unknown'``.

    Raises
    ------
    ValueError
        On construction, if a value field holds a :class:`ConditionState`
        member, if a kind is not a state its fact allows, if a value is set
        when its kind is not ``'fixed'`` (or missing when it is), or if
        ``temperature_sweep`` and ``temperature_kind='sweep'`` do not go
        together.

    Examples
    --------
    >>> from silk.simulation import OperatingConditions
    >>> oc = OperatingConditions('silk', 'ngspice', 'tt')
    >>> oc.state('corner').value, oc.scalar('corner')
    ('fixed', 'tt')
    >>> oc.state('temperature').value, oc.scalar('temperature')
    ('unknown', None)
    """

    #: Source of the results, ``'silk'`` for a run silk performed or
    #: ``'external'`` for results loaded from outside.
    origin: str
    #: Name of the simulator backend, or ``None`` if not known.
    backend: Optional[str]
    #: Process corner name; set exactly when ``corner_kind`` is ``'fixed'``.
    corner: Optional[str]
    #: State of the temperature fact (degrees Celsius when fixed), a
    #: :class:`ConditionState` value spelling (``'fixed'``, ``'none'``,
    #: ``'unknown'`` or ``'sweep'``).
    temperature_kind: str = 'unknown'
    #: Ambient temperature in degrees Celsius; set exactly when
    #: ``temperature_kind`` is ``'fixed'``.
    temperature_c: Optional[float] = None
    #: Why silk holds the temperature state (a word from
    #: ``CONDITION_VOCABULARY['temperature']['bases']``). The temperature
    #: itself is in degrees Celsius.
    temperature_basis: str = 'unknown'
    #: The swept range in degrees Celsius; set exactly when
    #: ``temperature_kind`` is ``'sweep'``.
    temperature_sweep: Optional[ConditionSweep] = None
    #: Primary-supply setpoint in volts that the launched deck was given;
    #: ``None`` unless established.
    supply_voltage_v: Optional[float] = None
    #: Why silk holds the supply state (a word from
    #: ``CONDITION_VOCABULARY['supply']['bases']``).
    supply_basis: str = 'unknown'
    #: Name of the source the supply voltage was read from.
    supply_source: Optional[str] = None
    #: State of the corner fact, a :class:`ConditionState` value spelling.
    corner_kind: str = 'unknown'
    #: Why silk holds the corner state (a word from
    #: ``CONDITION_VOCABULARY['corner']['bases']``).
    corner_basis: str = 'unknown'
    #: State of the supply fact, a :class:`ConditionState` value spelling.
    supply_kind: str = 'unknown'
    #: Designated-supply DC bounds in volts; set exactly for supply_kind='sweep'.
    supply_sweep: Optional[ConditionSweep] = None

    #: fact -> (kind field, value field)
    _FIELDS = {
        'corner': ('corner_kind', 'corner'),
        'temperature': ('temperature_kind', 'temperature_c'),
        'supply': ('supply_kind', 'supply_voltage_v'),
    }

    def __post_init__(self) -> None:
        for fact, (kind_field, value_field) in self._FIELDS.items():
            kind, value = getattr(self, kind_field), getattr(self, value_field)
            if isinstance(value, ConditionState):
                raise ValueError(
                    f'{value_field} holds a value or None, never {value}; '
                    f'set {kind_field}={value.value!r} and leave {value_field}=None.')
            if isinstance(kind, ConditionState):
                kind = kind.value
            if kind == 'unknown' and value is not None and fact != 'temperature':
                kind = 'fixed'
            if kind not in CONDITION_VOCABULARY[fact]['states']:
                raise ValueError(
                    f'{kind_field}={kind!r} is not a {fact} state; expected one of '
                    f'{list(CONDITION_VOCABULARY[fact]["states"])}.')
            if (kind == 'fixed') != (value is not None):
                raise ValueError(
                    f'{kind_field}={kind!r} with {value_field}={value!r}: '
                    f'{value_field} is set if and only if {kind_field} is \'fixed\' '
                    '(none and unknown hold None).')
            object.__setattr__(self, kind_field, kind)
        if (self.temperature_kind == 'sweep') != (self.temperature_sweep is not None):
            raise ValueError(
                f'temperature_kind={self.temperature_kind!r} with '
                f'temperature_sweep={self.temperature_sweep!r}: temperature_sweep '
                "is set if and only if temperature_kind is 'sweep'.")

        if (self.supply_kind == 'sweep') != (self.supply_sweep is not None):
            raise ValueError("supply_sweep is set if and only if supply_kind is 'sweep'.")

    def state(self, fact: str) -> ConditionState:
        """Return the :class:`ConditionState` of *fact*.

        Parameters
        ----------
        fact : {'corner', 'temperature', 'supply'}
            Which condition to read.

        Returns
        -------
        ConditionState
            The state, read from the matching ``*_kind`` field.

        Raises
        ------
        ValueError
            If *fact* is not a known fact.
        """
        _check_fact(fact)
        return ConditionState(getattr(self, self._FIELDS[fact][0]))

    def scalar(self, fact: str):
        """Return *fact* as a condition scalar.

        Parameters
        ----------
        fact : {'corner', 'temperature', 'supply'}
            Which condition to read.

        Returns
        -------
        str or float or None or ConditionState
            The value for a fixed fact, ``ConditionState.NONE`` or
            ``ConditionState.SWEEP`` for those states, and ``None`` for
            unknown. This is the form :func:`encode_condition` accepts.

        Raises
        ------
        ValueError
            If *fact* is not a known fact.
        """
        state = self.state(fact)
        if state is ConditionState.FIXED:
            return getattr(self, self._FIELDS[fact][1])
        return None if state is ConditionState.UNKNOWN else state

    def to_dict(self) -> dict:
        """Return JSON-ready descriptive provenance (never profile identity).

        Returns
        -------
        dict
            The fields as a dict (via :func:`dataclasses.asdict`), with
            ``temperature_sweep`` nested as a dict or ``None``. ``supply_sweep``
            is a nested range dict only when present, otherwise omitted to
            preserve historical condition/provenance bytes. The kinds
            are plain strings.
        """
        data = asdict(self)
        if self.supply_sweep is None:
            data.pop('supply_sweep')
        return data


def validate_temperature(value: float | str | None, *, native: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (Real, str) if native else Real):
        raise TypeError('Temperature must be a finite real Celsius value or None.')
    value = _parse_si(value) if native else float(value)
    if not math.isfinite(value) or value <= -273.15:
        raise ValueError('Temperature must be finite and above absolute zero (-273.15 C).')
    return value


def validate_supply(value: Optional[float]) -> Optional[float]:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise CharacterizationContextError('Declared supply voltage must be a finite real number.')
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Primary supply (TASK-158)
#
# silk cannot know which source is "the supply" from a name, and must not
# learn one: port-name-to-role rules are user configuration in this project
# (see silk/circuit/port_roles.py), never a list baked in here. So a source
# becomes the primary supply only on evidence the user supplied, in this
# order:
#
#   1. The author names it -- set_supply_voltage(volts, source='vdd').
#   2. Declared roles pick it out -- among the purely-DC voltage sources
#      referenced to ground, exactly one whose positive net is classified
#      POWER. Classification comes from declared ports (define_module_ports,
#      Port(role=), or opt-in infer_port_roles), i.e. from user configuration.
#   3. There is only one of them -- no POWER classification anywhere and
#      exactly one purely-DC voltage source to ground in the whole bench.
#      With a single DC rail there is nothing else it could be; this is a
#      structural fact about the circuit, not a guess about a name.
#
# Anything else is UNRESOLVED. Unresolved is silent while nobody asks (the
# supply is simply unknown) and refuses the moment somebody does.
# ---------------------------------------------------------------------------

#: SPICE's global ground node. This is the SPICE language, not a net-name
#: convention silk is inventing: node 0 is ground in every deck silk emits.
_SPICE_GROUND = '0'

#: Waveform slots on a source. Any one of them set makes the source
#: time-varying, so no scalar setpoint can be true of it.
_WAVEFORM_SLOTS = ('pwl', 'pulse', 'sine', 'exponential', 'sffm', 'am',
                   'trnoise', 'trrandom', 'waveform')


@dataclass(frozen=True)
class _SupplyRequest:
    """What this run establishes about the primary supply, before launch.

    ``voltage`` is the DC value the designated source carries into the deck,
    whether a setter chose it (``configured``) or it was simply wired that
    way. ``reason`` explains an unresolved supply and is the body of every
    refusal message, so the caller is told what silk found rather than that
    it failed. ``opaque`` marks a DC-subclass launch: the nominal card is
    still prepared and checked, but no fixed or swept supply is published.
    """

    source: Optional[str] = None
    instance: Optional[str] = None
    voltage: Optional[float] = None
    configured: bool = False
    reason: Optional[str] = None
    sweep: Optional[ConditionSweep] = None
    opaque: bool = False

    @property
    def supply_identity(self) -> tuple:
        """The supply-relevant state, for staleness comparison."""
        return (self.source, self.voltage, self.configured, self.sweep, self.opaque)


def _is_voltage_source(source) -> bool:
    from silk.sources import VoltageSource
    return isinstance(source, VoltageSource)


def _time_varying(source) -> bool:
    return any(getattr(source, slot, None) is not None for slot in _WAVEFORM_SLOTS)


def _net_classification(module, net_name):
    net = getattr(module, 'net_dict', {}).get(net_name)
    return None if net is None else net.classification


def _is_ground(module, net_name) -> bool:
    from silk.circuit.ports import PortRole
    return (net_name == _SPICE_GROUND
            or _net_classification(module, net_name) is PortRole.GROUND)


def _is_power(module, net_name) -> bool:
    from silk.circuit.ports import PortRole
    return _net_classification(module, net_name) is PortRole.POWER


def _literal_volts(value) -> Optional[float]:
    """The volts a source's ``dc`` denotes, or ``None`` when it is not a
    literal. A parameter reference or an unassigned variable establishes no
    setpoint, and silk does not evaluate SPICE expressions."""
    if isinstance(value, bool):
        return None
    try:
        return validate_supply(_parse_si(value))
    except (TypeError, ValueError, CharacterizationContextError):
        return None


def _identify_supply(module) -> tuple:
    """Apply the identification rule; return ``(name, source, reason)``.

    Exactly one of *source* / *reason* is ever set.
    """
    sources = getattr(module, 'sig_sources_dict', {}) or {}
    voltage_sources = [(name, src) for name, src in sources.items()
                       if _is_voltage_source(src)]
    if not voltage_sources:
        return None, None, 'this circuit has no voltage source to read a supply from'
    varying = [name for name, src in voltage_sources if _time_varying(src)]
    grounded = [(name, src) for name, src in voltage_sources
                if not _time_varying(src) and _is_ground(module, src.negative)]
    if not grounded:
        detail = (f'the only voltage sources ({", ".join(sorted(varying))}) carry a '
                  'waveform, so no fixed supply setpoint is true of them'
                  if varying and len(varying) == len(voltage_sources) else
                  'no voltage source is referenced to ground (SPICE node 0 or a '
                  'net whose declared role is GROUND)')
        return None, None, detail
    powered = [(name, src) for name, src in grounded
               if _is_power(module, src.positive)]
    tier, candidates = ('declared POWER', powered) if powered else ('DC', grounded)
    if len(candidates) == 1:
        name, src = candidates[0]
        return name, src, None
    return None, None, (
        f'{len(candidates)} sources are equally supply-shaped '
        f'({", ".join(sorted(name for name, _ in candidates))}, each a grounded '
        f'{tier} rail), so which one is the primary supply is the author\'s '
        "choice, not silk's")


def _resolve_supply_source(module, named: Optional[str]) -> tuple:
    """Resolve the designated source; return ``(name, source, reason)``."""
    if named is None:
        return _identify_supply(module)
    source = (getattr(module, 'sig_sources_dict', {}) or {}).get(named)
    if source is None:
        return None, None, (f'source {named!r} is not on this circuit (it has '
                            f'{sorted(getattr(module, "sig_sources_dict", {}) or {}) or "no sources"})')
    if not _is_voltage_source(source):
        return None, None, (f'source {named!r} is a {type(source).__name__}, '
                            'and only a voltage source carries a supply setpoint')
    if _time_varying(source):
        return None, None, (f'source {named!r} carries a waveform, so no fixed '
                            'supply setpoint is true of it; record the waveform '
                            'knobs as declared test parameters instead')
    return named, source, None


def own_supply_setting(module: Module) -> Optional[tuple]:
    """Return a Module's OWN deliberate supply setting, or ``None``.

    ``None`` means this Module never configured a supply, so an owner -- or
    nothing at all -- decides. A returned ``(volts, source)`` pair is a
    deliberate choice, and ``(None, None)`` is the deliberate choice to
    configure no supply at all. Mirrors :func:`own_temperature_setting`.
    """
    if not getattr(module, '_supply_explicit', False):
        return None
    return (getattr(module, '_supply_voltage', None),
            getattr(module, '_supply_source', None))


def apply_inherited_supply(module: Module, setting: Optional[tuple]) -> None:
    """Write an inherited setting onto an executing Module, without owning it.

    Deliberately bypasses the public setter, for the same reason
    :func:`apply_inherited_temperature` does: an inherited value must never
    become a sticky local override, or the next run could not follow a change
    made on the owner. The source is re-resolved against the executing
    Module's own sources at every run, which is what lets a setting made on a
    circuit reach the separate testbench Module that actually holds the rail.
    """
    volts, source = (None, None) if setting is None else setting
    module._supply_voltage = volts
    module._supply_source = source


def own_simulator_options_setting(module: Module) -> Optional[dict]:
    """Return a Module's OWN explicitly-set simulator options, or ``None``.

    ``None`` means this Module never explicitly set any option (other than
    ``temp``, which inherits through :func:`own_temperature_setting` instead)
    so an owner -- or nothing at all -- decides. A returned dict is a
    snapshot of this Module's own settings, key by key; mutating it does not
    affect the Module. Mirrors :func:`own_supply_setting`.
    """
    own = getattr(module, '_explicit_simulator_options', None)
    return dict(own) if own else None


def apply_inherited_simulator_options(module: Module, options: Optional[dict]) -> None:
    """Write an owner's options onto an executing Module's own options, as
    defaults, without making them this Module's own explicit setting.

    Deliberately bypasses ``set_simulator_options`` -- an inherited value
    must never become a sticky local override, or the next run could not
    follow a change made on the owner. Called before every run, exactly as
    :func:`apply_inherited_temperature`/:func:`apply_inherited_supply` are.
    A key this Module set explicitly always wins and is left untouched here;
    a key this function previously wrote that the owner no longer supplies
    is removed, so a cleared owner option stops being inherited too.
    """
    own_keys = set(getattr(module, '_explicit_simulator_options', None) or {})
    previous = getattr(module, '_inherited_simulator_option_keys', None) or frozenset()
    for key in previous - own_keys:
        module.simulator_options.pop(key, None)
    incoming = {k: v for k, v in (options or {}).items() if k not in own_keys}
    module.simulator_options.update(incoming)
    module._inherited_simulator_option_keys = frozenset(incoming)


def prepare_supply(module: Module, *, apply: bool = True) -> _SupplyRequest:
    """Resolve -- and, when configured, actually set -- the primary supply.

    Called before emission, so a configured value is what the netlister
    renders and the declaration is true by construction. With ``apply=False``
    nothing is written, which is how the launch side re-derives the same
    request to compare against the one the deck was prepared from. For
    behavioral launches, sweep classification is a separate step after this
    nominal identification, since flattening itself reads the nominal rail.

    Raises
    ------
    SimulationConfigurationError
        A configured supply whose source cannot be resolved. Deliberately a
        configuration error: the request itself is wrong, so every
        test in a recursive run is about to fail for the same reason.
    """
    volts = getattr(module, '_supply_voltage', None)
    named = getattr(module, '_supply_source', None)
    name, source, reason = _resolve_supply_source(module, named)
    if volts is not None:
        if source is None:
            raise SimulationConfigurationError(
                f'Cannot set a {volts:g} V supply: {reason}. Name the source '
                "explicitly -- set_supply_voltage(volts, source='<name>') -- "
                'or declare the rail with a POWER port role.')
        if apply:
            source.dc = volts
        request = _SupplyRequest(name, source.instance_name, float(volts), True)
    elif source is None:
        request = _SupplyRequest(reason=reason)
    else:
        wired = _literal_volts(source.dc)
        if wired is None:
            request = _SupplyRequest(name, source.instance_name, reason=(
                f'source {name!r} carries {source.dc!r}, which is not a literal '
                'voltage silk can read; set_supply_voltage(volts) writes one'))
        else:
            request = _SupplyRequest(name, source.instance_name, wired)
    return (_classify_supply_sweep(module, request)
            if module.simulator != 'behavioral' else request)


def _classify_supply_sweep(module, request):
    """Classify the resolved rail after nominal identification, without notices.

    Behavioral callers enter here separately from prepare_supply: flattening
    itself uses prepare_supply for nominal rail identification.
    """
    from dataclasses import replace
    analyses = tuple(module.sim_analysis_dict.values())
    # A custom DC emitter can override any rail, irrespective of its target.
    # Retain the nominal card value for preparation/verification, not evidence.
    if any(isinstance(a, DC) and type(a) is not DC for a in analyses):
        return replace(request, sweep=None, opaque=True)
    if request.source is None:
        return request
    # Only ordinary built-in DC is covered; custom analyses stay unprovable.
    dc = next((a for a in analyses if type(a) is DC), None)
    if dc is None or str(dc.source).lower() == 'temp':
        return request
    sources = module.sig_sources_dict
    if module.simulator == 'behavioral':
        from silk.simulation.behavioral import BehavioralEngine
        engine = BehavioralEngine()
        _, entries = engine._flatten(module, report=False)
        target, _ = engine._dc_target(entries, dc)
        selected = target[1] is sources[request.source]
    elif module.simulator in ('ngspice', 'xyce'):
        from silk.circuit.spice_netlister import _resolve_dc
        target = _resolve_dc(sources, dc.source)
        source = sources[request.source]
        name = source.physical_layout().dc_name
        selected = str(target).lower() == str(name).lower()
    else:
        return request
    if not selected:
        return request
    return replace(request, sweep=ConditionSweep(
        _parse_si(dc.start), _parse_si(dc.stop), _parse_si(dc.step)))


def assert_supply(request: _SupplyRequest, voltage: Optional[float]) -> None:
    """Check a deprecated ``run(voltage=...)`` declaration; never set anything."""
    if voltage is None:
        return
    warnings.warn(
        'run(voltage=...) is deprecated and only asserts the configured '
        'supply; use set_supply_voltage(...) before run().',
        DeprecationWarning, stacklevel=3)
    value = validate_supply(voltage)
    if request.opaque:
        raise SimulationConfigurationError(
            f'run(voltage={value!r}) cannot be confirmed: the custom DC analysis '
            'prevents confirming one supply voltage.')
    if request.sweep is not None:
        raise SimulationConfigurationError(
            f'run(voltage={value!r}) cannot confirm source {request.source!r}: '
            f'its DC sweep {request.sweep!r} cannot be confirmed as one voltage.')
    if request.voltage is None:
        raise SimulationConfigurationError(
            f'run(voltage={value!r}) cannot be confirmed: {request.reason}. '
            'Use set_supply_voltage(volts) -- or '
            "set_supply_voltage(volts, source='<name>') -- before run(), so "
            'the supply is configured rather than declared.')
    if value != request.voltage:
        raise SimulationConfigurationError(
            f'run(voltage={value!r}) contradicts source {request.source!r}, '
            f'which this deck runs at {request.voltage!r} V. Use '
            'set_supply_voltage(volts) before run() to change the supply; a '
            'declaration never sets one.')


def observed_supply(request: _SupplyRequest, deck_text: str) -> tuple:
    """Confirm the LAUNCHED deck carries the supply this run claims.

    The evidence is the deck the simulator was handed: the one line that
    declares the designated source, and the DC value on it. That is a weaker
    claim than the backend's own temperature report -- it proves what the
    simulator was ASKED for, not what the rail settled at -- and the recorded
    basis says so. It is still the thing that was missing: custom SPICE text
    that redeclares the source, or a value that never reached the deck, is
    caught here instead of being reported as a simulated condition.

    Returns ``(volts, basis, source)``.
    """
    expected = _parse_si(_render_param(request.voltage))
    lines = [line for line in deck_text.splitlines()
             if line.split() and line.split()[0].lower() == request.instance.lower()]
    if len(lines) != 1:
        raise SimulationError(
            f'Configured supply {request.voltage:g} V on source '
            f'{request.source!r} but the launched deck declares '
            f'{request.instance} {len(lines)} times. Remove the custom SPICE '
            'text that redeclares it; no report was published.')
    tokens = lines[0].split()
    index = next((i for i, token in enumerate(tokens[3:], 3)
                  if token.lower() == 'dc'), None)
    actual = None if index is None or index + 1 >= len(tokens) else \
        _literal_volts(tokens[index + 1])
    if actual is None or actual != expected:
        raise SimulationError(
            f'Configured supply {request.voltage:g} V on source '
            f'{request.source!r} but the launched deck reads '
            f'{lines[0].strip()!r}. Check the deck and any custom SPICE; no '
            'report was published.')
    basis = 'configured-and-emitted' if request.configured else 'declared-and-emitted'
    return request.voltage, basis, request.source


def driven_supply(request: _SupplyRequest, module: Module, driven, *, axis=None) -> tuple:
    """Confirm a behavioural run DROVE the configured supply.

    The behavioural counterpart of :func:`observed_supply`. There is no deck,
    so the evidence is the run's own waveform on the designated source's
    positive net: every sample must be the configured value (or, for a
    supply sweep, the matching DC plot's actual axis value). A model output
    or a second source on that net that overrode it is caught here instead
    of being reported as the supply the circuit ran at.

    *driven* maps net name -> the array the run produced for it. Returns
    ``(volts, 'configured-and-driven', source)``.
    """
    import numpy as np
    source = (getattr(module, 'sig_sources_dict', {}) or {}).get(request.source)
    net = getattr(source, 'positive', None)
    values = None if net is None else driven.get(net)
    if values is None:
        raise SimulationError(
            f'Configured supply {request.voltage:g} V on source '
            f'{request.source!r}, but the behavioural run produced no waveform '
            f'for its net {net!r}; no report was published.')
    values = np.asarray(values, dtype=float)
    expected = np.asarray(axis, dtype=float) if axis is not None else request.voltage
    if ((axis is not None and (not values.size or values.shape != expected.shape))
            or (values.size and not np.all(values == expected))):
        actual_drive = (f'between {values.min():g} V and {values.max():g} V'
                        if axis is None else 'inconsistently with the expected supply drive')
        raise SimulationError(
            f'Configured supply {request.voltage:g} V on source '
            f'{request.source!r}, but the behavioural run drove its net '
            f'{net!r} {actual_drive}. '
            'Check for a model output or second source on that net; no '
            'report was published.')
    return request.voltage, 'configured-and-driven', request.source


#: Backends that can independently confirm the ambient TEMP they solved at,
#: and may therefore be asked to apply one. ngspice prints it into its run log;
#: Xyce prints a ``{TEMP}`` probe alongside its rawfile. Any other
#: backend must refuse a temperature request rather than label it applied.
TEMPERATURE_BACKENDS = frozenset({'ngspice', 'xyce'})


@dataclass(frozen=True)
class _TemperatureRequest:
    backend: str
    temperature: Optional[float]
    basis: str
    sweep: Optional[ConditionSweep]
    custom: tuple
    opaque: bool

    @property
    def temperature_identity(self) -> tuple:
        """The temperature-relevant state, for staleness comparison.

        ``custom`` carries the raw deck text this request was RECOGNISED from,
        including the whole emitted native-options line. Comparing it would make
        any option change -- ``gmin`` as much as ``temp`` -- look like a
        temperature change and claim a temperature reason for it. Everything the
        recognition concluded is already in the other fields, so dropping the
        source text loses nothing here. The general staleness gate is what
        catches an unrelated configuration change.
        """
        return (self.backend, self.temperature, self.basis, self.sweep, self.opaque)


# These are abstention checks, not an attempt to interpret caller SPICE.
_OPAQUE_TOKEN = re.compile(
    r'(?:^|[^a-z])(?:temp|tnom|dtemp)(?![a-z])|^\s*\.(?:include|inc|lib)\b',
    re.IGNORECASE | re.MULTILINE)
_OPTION_ASSIGNMENT = re.compile(r'(?<!\S)([a-z][\w]*)\s*=\s*', re.IGNORECASE)
_QUOTED_OR_BRACED = r"'[^']*(?:'|$)|\"[^\"]*(?:\"|$)|\{[^}]*(?:\}|$)"
_COMMENT_OR_VALUE = re.compile(_QUOTED_OR_BRACED + r'|(?P<comment>\$|;|//).*')


def _logical_custom_lines(custom):
    """Join SPICE + continuations for inspection only, retaining source labels."""
    pending, locations = '', []
    for source, text in custom:
        for number, physical in enumerate(text.splitlines(), 1):
            line = physical.strip()
            if not line or line.startswith('*'):
                continue
            line = _COMMENT_OR_VALUE.sub(
                lambda m: '' if m['comment'] else m[0], line).rstrip()
            if not line:
                continue
            location = f'{source} line {number}'
            if line.startswith('+') and pending:
                pending += ' ' + line[1:].strip()
                locations.append(location)
            else:
                if pending:
                    yield ', '.join(locations), pending
                pending, locations = line, [location]
    if pending:
        yield ', '.join(locations), pending


def _literal_temperature(text, declaration):
    """Return a scalar only for a whole SPICE number; reject invalid domains."""
    try:
        value = _parse_si(text.strip())
    except (TypeError, ValueError):
        # Nonfinite numeric spellings are invalid literals, not expressions.
        if not re.fullmatch(r'[+-]?(?:inf(?:inity)?|nan)', text.strip(), re.IGNORECASE):
            return None
        value = float(text)
    try:
        return validate_temperature(value)
    except (TypeError, ValueError) as exc:
        raise SimulationConfigurationError(
            f'Invalid temperature in {declaration}: {exc}') from exc


def _inspect_temperature_text(custom):
    """Inspect emitted caller text, never includes, model files or subcircuits.

    The only recognized declarations are .temp and TEMP assignments on
    .option(s). Everything requiring expression/control evaluation abstains.
    Declaration collection precedes numeric validation so conflicts name all
    inputs, even when one of them is not a valid scalar.
    """
    declarations = []
    opaque = False
    in_control = False
    subckt_depth = 0
    for location, line in _logical_custom_lines(custom):
        parts = line.split(maxsplit=1)
        directive = parts[0].lower()
        body = parts[1] if len(parts) == 2 else ''
        if directive == '.control':
            in_control = opaque = True
        if in_control:
            if directive == '.endc':
                in_control = False
            continue
        if directive == '.subckt':
            subckt_depth += 1
            opaque = True
        if subckt_depth:
            if directive == '.ends':
                subckt_depth -= 1
            continue
        if directive in ('.if', '.elseif', '.else', '.endif', '.endc'):
            opaque = True
        if directive == '.temp':
            declarations.append((f'{location}: {line}', body))
        elif directive in ('.option', '.options'):
            # Mask opaque values only to locate declarations outside them.
            # Never interpret expressions or TEMP tokens inside quoted text.
            masked = re.sub(_QUOTED_OR_BRACED, lambda m: 'x' * len(m[0]), body)
            opaque |= masked != body
            assignments = list(_OPTION_ASSIGNMENT.finditer(masked))
            for index, match in enumerate(assignments):
                if match[1].lower() == 'temp':
                    end = (assignments[index + 1].start()
                           if index + 1 < len(assignments) else len(body))
                    setting = body[match.start():end].strip()
                    declarations.append((f'{location}: {setting}', body[match.end():end]))
            if not assignments and _OPAQUE_TOKEN.search(body):
                opaque = True
        elif _OPAQUE_TOKEN.search(line):
            opaque = True
    if len(declarations) > 1:
        raise SimulationConfigurationError(
            'Multiple explicit temperature declarations: '
            + '; '.join(label for label, _ in declarations)
            + '. Remove the extra declarations or use one set_temperature() setting.')
    temperature = None
    if declarations:
        label, value = declarations[0]
        temperature = _literal_temperature(value, label)
        opaque |= temperature is None
    return temperature, opaque


def own_temperature_setting(module: Module) -> Optional[tuple]:
    """Return a Module's OWN deliberate temperature setting, or ``None``.

    ``None`` means this Module never configured a temperature, so something
    else — an owner, or nothing at all — decides. A returned
    ``(value, basis)`` pair is a deliberate choice, and ``(None, basis)`` is
    the deliberate choice to run with no TEMP directive at all. The value is
    passed on verbatim, so a native SPICE-number string keeps its spelling
    and the emitted deck text is unchanged by inheritance.
    """
    if not getattr(module, '_temperature_explicit', False):
        return None
    return (module.simulator_options.get('temp'),
            getattr(module, '_temperature_basis', 'native-and-observed'))


def apply_inherited_temperature(module: Module, setting: Optional[tuple]) -> None:
    """Write an inherited setting onto an executing Module, without owning it.

    Deliberately bypasses the public setter — an inherited value must never
    become a sticky local override, or the next run could not follow a
    change made on the owner. Called before every run, so clearing the
    owner's setting clears the inherited one too.
    """
    value = None if setting is None else setting[0]
    if value is None:
        module.simulator_options.pop('temp', None)
        # No temperature, no basis to describe one. prepare_temperature()
        # reads the basis only when a temperature is present; resetting it
        # keeps that implicit invariant plainly true.
        module._temperature_basis = 'native-and-observed'
    else:
        module.simulator_options['temp'] = value
        module._temperature_basis = setting[1]


def prepare_temperature(module: Module) -> _TemperatureRequest:
    """Resolve after backend selection and before allocation, cache or launch."""
    temperature = validate_temperature(module.simulator_options.get('temp'), native=True)
    sweep = next((ConditionSweep(_parse_si(a.start), _parse_si(a.stop), _parse_si(a.step))
                  for a in module.sim_analysis_dict.values()
                  if isinstance(a, DC) and a.source.lower() == 'temp'), None)
    custom = tuple((f'custom_lines[{i}]', line or '')
                   for i, line in enumerate(module.custom_lines))
    if module.simulator_options:
        options = load_simulator_config(module.simulator).emit_options_line(module.simulator_options)
        custom += (('native options', options),)
    raw_temperature, opaque = _inspect_temperature_text(custom)
    if temperature is not None or sweep is not None:
        if module.simulator not in TEMPERATURE_BACKENDS:
            raise NotSupportedError(
                f'Applied temperature is not supported for {module.simulator!r}. '
                f'Use one of {sorted(TEMPERATURE_BACKENDS)} or clear TEMP with '
                'set_temperature(None).')
        if opaque:
            raise NotSupportedError(
                'Cannot confirm managed TEMP alongside custom SPICE text that '
                'may set a temperature. Remove that line and use '
                'set_temperature(); expressions, controls and caller includes '
                'cannot establish a fixed temperature.')
    basis = (getattr(module, '_temperature_basis', 'native-and-observed')
             if temperature is not None else 'unknown')
    if temperature is None and raw_temperature is not None and not opaque:
        if module.simulator == 'ngspice':
            temperature, basis = raw_temperature, 'raw-and-observed'
        else:
            # Xyce ignores .temp and needs DEVICE options; no raw temperature
            # syntax on an unverified backend establishes a scalar here.
            opaque = True
    return _TemperatureRequest(
        module.simulator, temperature, basis, sweep, custom, opaque)


def assert_temperature(request: _TemperatureRequest, temperature: Optional[float]) -> None:
    if temperature is None:
        return
    warnings.warn(
        'run(temperature=...) is deprecated and only asserts configured TEMP; '
        'use set_temperature(...) before run().', DeprecationWarning, stacklevel=3)
    if request.backend not in TEMPERATURE_BACKENDS:
        raise NotSupportedError(f'Temperature assertions are not supported for {request.backend!r}.')
    value = validate_temperature(temperature)
    if request.sweep is not None or request.temperature is None or value != request.temperature:
        raise CharacterizationContextError(
            f'run temperature={value!r} requires matching explicit fixed TEMP '
            f'(configured {request.temperature!r}); use set_temperature(...) '
            'before run(), and omit scalar assertions on temperature sweeps.')


_TEMP_OBSERVATION = re.compile(
    r'\bDoing analysis at TEMP\s*=\s*([^\s]+)', re.IGNORECASE)


def read_ngspice_temperatures(log_path: str | Path | None) -> tuple[float, ...]:
    """Read only the log returned by this launch, never a guessed prior path."""
    if log_path is None:
        return ()
    try:
        tokens = _TEMP_OBSERVATION.findall(Path(log_path).read_text(errors='replace'))
        return tuple(validate_temperature(float(token)) for token in tokens)
    except (OSError, TypeError, ValueError):
        return ()


def read_xyce_temperatures(probe_path: str | Path | None) -> tuple[float, ...]:
    """Read the ``{TEMP}`` column Xyce printed for THIS launch, never a guess.

    The deck's temperature probe (see
    :class:`~silk.circuit.xyce_netlister.XyceNetlister`) is a Xyce
    ``format=csv`` print: a quoted-or-bare header row, then one row per solve
    point. Only the ``{TEMP}`` column is read, so the leading sweep/time column
    Xyce adds for some analyses is ignored. Anything unreadable is no evidence
    at all (an empty tuple), which records the temperature as unknown rather
    than confirming the request.
    """
    if probe_path is None:
        return ()
    try:
        lines = Path(probe_path).read_text(errors='replace').splitlines()
        header = [c.strip().strip('"').lower() for c in lines[0].split(',')]
        column = header.index('{temp}')
        return tuple(validate_temperature(float(row.split(',')[column]))
                     for row in lines[1:] if row.strip())
    except (OSError, IndexError, TypeError, ValueError):
        return ()


# ngspice's own closing banner, e.g. 'ngspice-42 done' / 'ngspice-47 done'
# (measured on both binaries — see tests/fixtures/device_param_probe).
_NGSPICE_VERSION = re.compile(r'^ngspice-(\S+)\s+done\s*$', re.MULTILINE)


def read_ngspice_version(log_path: str | Path | None) -> Optional[str]:
    """The ngspice BINARY version this launch reports, read from THIS run's
    own log (never invoked separately). A device-parameter
    probe's read-time error needs the version that actually
    PRODUCED the rawfile, not whatever ngspice happens to be on PATH right
    now: ngspice-42 and ngspice-47 answer the same '@' address differently
    for some columns (cgso/cgdo/cgbo), and only the
    log says which one ran. ``'ngspice-<version>'``, or ``None`` if
    unreadable.
    """
    if log_path is None:
        return None
    try:
        match = _NGSPICE_VERSION.search(Path(log_path).read_text(errors='replace'))
        return f'ngspice-{match.group(1)}' if match else None
    except (OSError, TypeError):
        return None


# Xyce's own opening banner, e.g.
# '***** This is version Xyce Release 7.10.0-opensource' (measured).
_XYCE_VERSION = re.compile(r'This is version (Xyce Release \S+)')


def read_xyce_version_from_text(text: Optional[str]) -> Optional[str]:
    """The Xyce RELEASE this launch reports, read from the SAME captured
    stdout transcript this launch produced — Xyce writes no
    separate log file the way ngspice does, so there is no path to read;
    the caller passes the stdout text its own runner already captured.
    Never invoked separately. ``None`` if unreadable/absent.
    """
    if not text:
        return None
    match = _XYCE_VERSION.search(text)
    return match.group(1) if match else None


def _launched_corner(corner: str, no_pdk_device: bool) -> dict:
    """The corner fields of a silk-launched deck, from the corner proof.

    A deck proven to instantiate no PDK device carries no ``.lib``, so no
    corner reached the simulator: corner none. Otherwise the corner silk
    selected for the ``.lib`` section is the value (design section 3.1).
    """
    if no_pdk_device:
        return {'corner': None, 'corner_kind': 'none',
                'corner_basis': 'proven-no-pdk-device'}
    return {'corner': corner, 'corner_kind': 'fixed',
            'corner_basis': 'configured-and-emitted'}


def observed_conditions(request: _TemperatureRequest, corner: str, raw: RawResult,
                        *, no_pdk_device: bool = False) -> OperatingConditions:
    """The snapshot of one silk launch.

    *corner* is the corner silk prepared the deck at; *no_pdk_device* is
    :func:`has_no_pdk_device` for the launched module, which decides whether
    that corner is the evidence (fixed) or only the run's label (none).
    """
    corner_fields = _launched_corner(corner, no_pdk_device)
    if request.sweep is not None:
        return OperatingConditions('silk', request.backend, temperature_kind='sweep',
                                   temperature_basis='configured',
                                   temperature_sweep=request.sweep, **corner_fields)
    observations = (raw.ambient_temperatures
                    if request.backend in TEMPERATURE_BACKENDS else ())
    if request.opaque:
        # Unrelated custom text must not erase an honest backend observation.
        observations = ()
    if request.temperature is not None:
        if not observations or any(
            not math.isclose(t, request.temperature, rel_tol=0, abs_tol=5.01e-7)
            for t in observations
        ):
            # One observation per solve point on some backends: report the
            # distinct values, so a long transient cannot bury the message.
            #
            # Deliberately a plain SimulationError, NOT a
            # SimulationConfigurationError (TASK-163): this is post-launch
            # evidence about what the run actually solved at, not a fact
            # knowable about the request beforehand -- the deck asked for a
            # temperature and the backend did something else. It stays a
            # per-test result, like any other way a launch can disappoint.
            distinct = tuple(dict.fromkeys(observations))[:8]
            raise SimulationError(
                f'Configured TEMP={request.temperature:g} C but this launch '
                f'observed {distinct or "no ambient TEMP confirmation"}. '
                'Check the simulator log and temperature overrides; no report was published.')
        # ngspice prints six fractional digits and Xyce eight; keep the
        # prepared numeric setting once independently confirmed to that
        # reporting precision.
        temperature, basis = request.temperature, request.basis
    elif observations and all(t == observations[0] for t in observations):
        temperature, basis = observations[0], 'backend-observed'
    else:
        return OperatingConditions('silk', request.backend, **corner_fields)
    return OperatingConditions('silk', request.backend, temperature_kind='fixed',
                               temperature_c=temperature, temperature_basis=basis,
                               **corner_fields)


def external_sweep(raw: Optional[RawResult]) -> Optional[ConditionSweep]:
    """Recognize an actual nutmeg temperature axis without reading a deck."""
    from silk.signals.rawfile import RawFile
    if raw is None or not isinstance(raw.native, RawFile):
        return None
    for plot in raw.native.plots:
        if not plot.vectors:
            continue
        axis = plot.vectors[0]
        if axis.name.lower() in ('temp-sweep', 'temperature'):
            values = axis.data.real
            if len(values):
                return ConditionSweep(float(values[0]), float(values[-1]),
                                        float(values[1] - values[0]) if len(values) > 1 else 0.0)
    return None


def confirm_conditions(conditions: OperatingConditions, *, corner=None,
                       temperature=None, voltage=None,
                       error_type: type[Exception] = CharacterizationContextError) -> None:
    """Optional declarations may confirm a snapshot, never fill its unknowns.

    Each declaration is compared with ``conditions.scalar(fact)``, so
    ``ConditionState.NONE`` confirms a none fact and a value never does.
    """
    if corner is not None and not isinstance(corner, ConditionState):
        corner = str(corner)
    for field, explicit, actual in (
        ('corner', corner if isinstance(corner, ConditionState)
         else (None if corner is None else str(corner)),
         conditions.scalar('corner')),
        ('temperature', temperature, conditions.scalar('temperature')),
        ('voltage', voltage, conditions.scalar('supply')),
    ):
        if explicit is not None and explicit != actual:
            raise error_type(
                f'{field}={explicit!r} conflicts with operating conditions '
                f'({actual!r}); declare conditions when running/evaluating '
                'new evidence, not when reusing or storing it.')


def storage_conditions(snapshot: OperatingConditions,
                       context: Optional[CharacterizationContext],
                       corner, temperature, voltage, *, label=None) -> tuple:
    """Validate persistence overrides and the independently attached descriptor.

    Returns the snapshot's condition scalars (``scalar(fact)``): a value,
    ``ConditionState.NONE`` / ``SWEEP``, or ``None`` for unknown.

    *label* is the corner the run was requested at (``SpecReport.corner``).
    An explicit *corner* equal to it names the run, not the evidence, so it
    is not confirmed against a none corner; any other corner must match
    ``snapshot.scalar('corner')``.
    """
    if (corner is not None and label is not None
            and not isinstance(corner, ConditionState) and str(corner) == str(label)):
        corner = None
    confirm_conditions(snapshot, corner=corner, temperature=temperature,
                       voltage=voltage, error_type=EvidenceMismatchError)
    if context is not None:
        confirm_conditions(snapshot, corner=context.corner,
                           temperature=context.temperature, voltage=context.voltage,
                           error_type=EvidenceMismatchError)
    return (snapshot.scalar('corner'), snapshot.scalar('temperature'),
            snapshot.scalar('supply'))


# ---------------------------------------------------------------------------
# Public proofs: "no PDK device anywhere" and "no supply" (TASK-404)
#
# Each takes a Module and returns a bool. True is a PROOF; False means "not
# proven", never "proven the opposite". silk's launch path and an external
# caller that runs a silk-generated deck itself use these same two functions.
# ---------------------------------------------------------------------------

def has_no_pdk_device(module) -> bool:
    """Prove that *module* instantiates no PDK device anywhere in its hierarchy.

    ``True`` means the emitted deck carries no PDK ``.lib``/``.include``, so
    no process corner can have reached the simulator: the corner is *none*,
    and a consumer running the deck itself needs no PDK files. Any element not
    positively recognised as ideal, and any ``add_custom_line`` text anywhere
    in the hierarchy, makes this ``False`` (corner not proven none).

    This is the inverse of the netlister's ``module_uses_pdk_devices``, not a
    copy of it.

    Parameters
    ----------
    module : Module
        The circuit to inspect.

    Returns
    -------
    bool
        ``True`` is a proof; ``False`` means not proven, never proven the
        opposite.

    Examples
    --------
    >>> import tempfile
    >>> from silk import Module, load_pdk
    >>> from silk.elements import elements
    >>> from silk.simulation import has_no_pdk_device
    >>> mod = Module('sky130', tempfile.mkdtemp(), 'r.cir', load_pdk('sky130'))
    >>> _ = mod.add(elements.resistor('r1', p='in', n='out', value=1e3))
    >>> has_no_pdk_device(mod)
    True
    """
    from silk.circuit.spice_netlister import module_uses_pdk_devices
    return not module_uses_pdk_devices(module)


def _literal_zero(value) -> bool:
    """True only for a literal zero ``dc``. A parameter reference or an
    unassigned variable is not a literal: silk does not evaluate SPICE
    expressions."""
    if isinstance(value, bool):
        return False
    try:
        return _parse_si(value) == 0
    except (TypeError, ValueError):
        return False


def _hierarchy(module):
    """Yield *module* and every child Module under it, each once."""
    seen, stack = set(), [module]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for table in ('elem_dict', 'module_dict'):
            for entry in (getattr(current, table, None) or {}).values():
                child = entry.get('device') if isinstance(entry, dict) else None
                if hasattr(child, 'sig_sources_dict') or hasattr(child, 'net_dict'):
                    stack.append(child)


def has_no_supply(module) -> bool:
    """Prove that *module* has no supply, by the rule of the operating-conditions
    design (section 3.3), over the whole hierarchy.

    ``True`` only when **all** hold:

    1. no supply is configured (``set_supply_voltage``), own or inherited;
    2. the corner is proven none (:func:`has_no_pdk_device`), because a
       transistor circuit or opaque custom SPICE always has something that
       powers it;
    3. no net is classified ``PortRole.POWER`` (declared ports, ``Port(role=)``
       or opt-in role inference, never a name list);
    4. every independent voltage or current source carries a waveform, or has
       a ``dc`` that is a literal zero. A parameter-reference ``dc`` fails.
       Controlled sources are not supplies and are ignored.

    ``False`` means the supply is unknown, not that one exists.

    Parameters
    ----------
    module : Module
        The circuit to inspect.

    Returns
    -------
    bool
        ``True`` is a proof that there is no supply; ``False`` means not
        proven.

    Examples
    --------
    >>> import tempfile
    >>> from silk import Module, load_pdk, sources
    >>> from silk.elements import elements
    >>> from silk.simulation import has_no_supply
    >>> mod = Module('sky130', tempfile.mkdtemp(), 'rc.cir', load_pdk('sky130'))
    >>> _ = mod.add(elements.resistor('r1', p='in', n='out', value=1e3))
    >>> _ = mod.add_source(sources.VoltageSource(
    ...     'vs', positive='in', negative='0', dc=0.0,
    ...     pulse=dict(v1=0, v2=1, td=0, tr=1e-9, tf=1e-9, pw=1e-6, per=2e-6)))
    >>> has_no_supply(mod)
    True
    """
    from silk.circuit.ports import PortRole
    from silk.sources import CurrentSource, VoltageSource
    if not has_no_pdk_device(module):
        return False
    for mod in _hierarchy(module):
        if getattr(mod, '_supply_voltage', None) is not None:
            return False
        for net in (getattr(mod, 'net_dict', None) or {}).values():
            if getattr(net, 'classification', None) is PortRole.POWER:
                return False
        for src in (getattr(mod, 'sig_sources_dict', None) or {}).values():
            if not isinstance(src, (VoltageSource, CurrentSource)):
                continue
            if _time_varying(src):
                continue
            if not _literal_zero(getattr(src, 'dc', None)):
                return False
    return True
