"""Port descriptors and enums for element kinds, directions and electrical roles."""

# Circuit port/element vocabulary — enums + the Port descriptor.
#
# Split out of circuit_elements.py: these are shared
# *vocabulary*, not the element manager. circuit_elements.py re-exports them
# for backward compatibility, so existing imports (including string-literal
# imports emitted into generated user code) keep working unchanged — the class
# objects are identical, so isinstance identity is preserved.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union


class ElementType(str, Enum):
    """Kind of a placed element: a leaf device or a sub-module instance.

    Stored under the ``'type'`` key of each entry in a module's element
    table; the values are the single letters ``'d'`` and ``'m'``.
    """
    DEVICE = 'd'
    MODULE = 'm'


class PortDir(str, Enum):
    """Direction of a module port.

    ``IN``, ``OUT`` and ``INOUT`` are signal-flow directions. ``POWER`` and
    ``GROUND`` state what the port carries and imply the matching
    :class:`PortRole` (see ``implied_role``).
    """
    IN     = 'input'
    OUT    = 'output'
    INOUT  = 'inout'
    POWER  = 'power'
    GROUND = 'ground'

    @property
    def implied_role(self) -> 'Optional[PortRole]':
        """The :class:`PortRole` a bare direction implies, or ``None``.

        ``PortDir.POWER`` and ``PortDir.GROUND`` are not directions in the
        signal-flow sense — they *are* a statement of what the port carries.
        So a port declared with just ``PortDir.POWER`` / ``PortDir.GROUND``
        is given ``PortRole.POWER`` / ``PortRole.GROUND`` at declaration
        , and every role-keyed consumer — the lint rules
        INT-002 / INT-004 / STRUCT-002 / STRUCT-007 / STRUCT-009,
        ``Net.classification``, the ``PortRoles`` constraint export, the
        annotation-coverage gate — sees the port as a supply. Signal
        directions (IN / OUT / INOUT) imply nothing: their role is unknown
        until stated or inferred. An explicit role always wins; the
        derivation happens only when none was given.
        """
        return _IMPLIED_ROLE.get(self)


class PortRole(str, Enum):
    """Semantic role of a module port: what kind of signal it carries.

    Roles drive the role-keyed lint rules, ``Net.classification`` and the
    annotation-coverage gate. A port without a role is unannotated.
    """
    ANALOG  = 'analog'
    DIGITAL = 'digital'
    CLOCK   = 'clock'
    POWER   = 'power'
    GROUND  = 'ground'


# Direction -> role a bare declaration implies. Keyed on PortDir, so the
# mapping is one object shared by ``PortDir.implied_role`` and anyone who
# needs to enumerate the supply directions.
_IMPLIED_ROLE = {
    PortDir.POWER:  PortRole.POWER,
    PortDir.GROUND: PortRole.GROUND,
}


@dataclass
class Port:
    """Rich port descriptor for :meth:`silk.circuit.Module.define_module_ports`.

    This is a transient argument bag consumed by :meth:`silk.circuit.Module.define_module_ports`.
    Fields are unpacked into :class:`silk.circuit.modules.ModulePin` and the
    ``Port`` instance is not retained after that call.

    Alternative to bare :class:`PortDir` or ``(PortDir, PortRole)`` tuple.
    Accepts an optional :class:`~silk.circuit.voltage_domain.VoltageDomain`
    for domain-aware checking. When ``role`` is omitted and ``dir`` is
    ``PortDir.POWER`` / ``PortDir.GROUND`` the matching role is derived at
    declaration (see ``PortDir.implied_role``).

    Prefer the convenience factories below over the raw constructor for
    the common cases — they cut boilerplate and remove the need to import
    ``PortDir`` / ``PortRole`` in user code::

        Port.analog_input(avdd)
        Port.analog_output(avdd)
        Port.supply(avdd)       # direction defaults to IN
        Port.ground()           # direction defaults to IN

    Parameters
    ----------
    dir : PortDir
        Port direction (required).
    role : PortRole, optional
        Semantic role of the port.
    domain : VoltageDomain, optional
        Voltage domain this port belongs to.
    bits : int, range or slice, optional
        Declares a bus instead of a scalar port.
    naming : {'bracketed', 'underscore'}, default 'bracketed'
        Per-bit pin spelling for a bus.
    """
    dir: 'PortDir'
    role: 'Optional[PortRole]' = None
    domain: 'Optional[Any]' = None   # VoltageDomain | None
    #: Bus width. ``None`` is a scalar port; an ``int`` N means bits
    #: ``0..N-1``; a ``range`` or ``slice`` (step 1) names explicit indices.
    bits: 'Optional[Union[int, range, slice]]' = None
    #: Per-bit pin spelling for a bus. ``'bracketed'`` gives ``d[3]``,
    #: ``'underscore'`` gives ``d_3``. Ignored for scalar ports.
    naming: 'str' = 'bracketed'

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------
    #
    # Signal-family factories (analog / digital / clock) take only a
    # ``domain`` and encode direction in the method name, because for
    # signals IN vs OUT is the distinguishing axis.
    #
    # Rail-family factories (``supply`` / ``ground``) take an optional
    # ``direction`` keyword (default IN) because supplies and grounds
    # are IN in the overwhelming majority of cases; override only for
    # bias / reference-generator modules that emit a rail.

    @classmethod
    def analog_input(cls, domain=None, *, bits=None, naming='bracketed'):
        """Analog signal port, direction IN.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain the port belongs to; ``None`` leaves it unchecked.
        bits : int, range or slice, optional
            Declares a bus. An ``int`` N means indices ``0..N-1``; ``range`` and
            ``slice`` must have step 1 (a slice needs an explicit stop). ``None``
            declares a scalar port.
        naming : {'bracketed', 'underscore'}, default 'bracketed'
            Per-bit pin spelling for a bus: ``d[3]`` or ``d_3``.

        Returns
        -------
        Port
            A new descriptor to pass in the mapping given to
            :meth:`silk.circuit.Module.define_module_ports`.

        Raises
        ------
        ValueError
            At declaration time (not here), when ``bits`` is an invalid
            value of an accepted type (for example a nonpositive width, or
            an empty, non-unit-step or unbounded range or slice) or
            ``naming`` is invalid.
        TypeError
            At declaration time (not here), when ``bits`` has an unsupported
            type.
        """
        return cls(dir=PortDir.IN, role=PortRole.ANALOG, domain=domain,
                   bits=bits, naming=naming)

    @classmethod
    def analog_output(cls, domain=None, *, bits=None, naming='bracketed'):
        """Analog signal port, direction OUT.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain the port belongs to; ``None`` leaves it unchecked.
        bits : int, range or slice, optional
            Declares a bus. An ``int`` N means indices ``0..N-1``; ``range`` and
            ``slice`` must have step 1 (a slice needs an explicit stop). ``None``
            declares a scalar port.
        naming : {'bracketed', 'underscore'}, default 'bracketed'
            Per-bit pin spelling for a bus: ``d[3]`` or ``d_3``.

        Returns
        -------
        Port
            A new descriptor to pass in the mapping given to
            :meth:`silk.circuit.Module.define_module_ports`.

        Raises
        ------
        ValueError
            At declaration time (not here), when ``bits`` is an invalid
            value of an accepted type (for example a nonpositive width, or
            an empty, non-unit-step or unbounded range or slice) or
            ``naming`` is invalid.
        TypeError
            At declaration time (not here), when ``bits`` has an unsupported
            type.
        """
        return cls(dir=PortDir.OUT, role=PortRole.ANALOG, domain=domain,
                   bits=bits, naming=naming)

    @classmethod
    def digital_input(cls, domain=None, *, bits=None, naming='bracketed'):
        """Digital signal port, direction IN.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain the port belongs to; ``None`` leaves it unchecked.
        bits : int, range or slice, optional
            Declares a bus. An ``int`` N means indices ``0..N-1``; ``range`` and
            ``slice`` must have step 1 (a slice needs an explicit stop). ``None``
            declares a scalar port.
        naming : {'bracketed', 'underscore'}, default 'bracketed'
            Per-bit pin spelling for a bus: ``d[3]`` or ``d_3``.

        Returns
        -------
        Port
            A new descriptor to pass in the mapping given to
            :meth:`silk.circuit.Module.define_module_ports`.

        Raises
        ------
        ValueError
            At declaration time (not here), when ``bits`` is an invalid
            value of an accepted type (for example a nonpositive width, or
            an empty, non-unit-step or unbounded range or slice) or
            ``naming`` is invalid.
        TypeError
            At declaration time (not here), when ``bits`` has an unsupported
            type.
        """
        return cls(dir=PortDir.IN, role=PortRole.DIGITAL, domain=domain,
                   bits=bits, naming=naming)

    @classmethod
    def digital_output(cls, domain=None, *, bits=None, naming='bracketed'):
        """Digital signal port, direction OUT.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain the port belongs to; ``None`` leaves it unchecked.
        bits : int, range or slice, optional
            Declares a bus. An ``int`` N means indices ``0..N-1``; ``range`` and
            ``slice`` must have step 1 (a slice needs an explicit stop). ``None``
            declares a scalar port.
        naming : {'bracketed', 'underscore'}, default 'bracketed'
            Per-bit pin spelling for a bus: ``d[3]`` or ``d_3``.

        Returns
        -------
        Port
            A new descriptor to pass in the mapping given to
            :meth:`silk.circuit.Module.define_module_ports`.

        Raises
        ------
        ValueError
            At declaration time (not here), when ``bits`` is an invalid
            value of an accepted type (for example a nonpositive width, or
            an empty, non-unit-step or unbounded range or slice) or
            ``naming`` is invalid.
        TypeError
            At declaration time (not here), when ``bits`` has an unsupported
            type.
        """
        return cls(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=domain,
                   bits=bits, naming=naming)

    @classmethod
    def clock_input(cls, domain=None, *, bits=None, naming='bracketed'):
        """Clock port, direction IN.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain the port belongs to; ``None`` leaves it unchecked.
        bits : int, range or slice, optional
            Declares a bus. An ``int`` N means indices ``0..N-1``; ``range`` and
            ``slice`` must have step 1 (a slice needs an explicit stop). ``None``
            declares a scalar port.
        naming : {'bracketed', 'underscore'}, default 'bracketed'
            Per-bit pin spelling for a bus: ``d[3]`` or ``d_3``.

        Returns
        -------
        Port
            A new descriptor to pass in the mapping given to
            :meth:`silk.circuit.Module.define_module_ports`.

        Raises
        ------
        ValueError
            At declaration time (not here), when ``bits`` is an invalid
            value of an accepted type (for example a nonpositive width, or
            an empty, non-unit-step or unbounded range or slice) or
            ``naming`` is invalid.
        TypeError
            At declaration time (not here), when ``bits`` has an unsupported
            type.
        """
        return cls(dir=PortDir.IN, role=PortRole.CLOCK, domain=domain,
                   bits=bits, naming=naming)

    @classmethod
    def clock_output(cls, domain=None, *, bits=None, naming='bracketed'):
        """Clock port, direction OUT.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain the port belongs to; ``None`` leaves it unchecked.
        bits : int, range or slice, optional
            Declares a bus. An ``int`` N means indices ``0..N-1``; ``range`` and
            ``slice`` must have step 1 (a slice needs an explicit stop). ``None``
            declares a scalar port.
        naming : {'bracketed', 'underscore'}, default 'bracketed'
            Per-bit pin spelling for a bus: ``d[3]`` or ``d_3``.

        Returns
        -------
        Port
            A new descriptor to pass in the mapping given to
            :meth:`silk.circuit.Module.define_module_ports`.

        Raises
        ------
        ValueError
            At declaration time (not here), when ``bits`` is an invalid
            value of an accepted type (for example a nonpositive width, or
            an empty, non-unit-step or unbounded range or slice) or
            ``naming`` is invalid.
        TypeError
            At declaration time (not here), when ``bits`` has an unsupported
            type.
        """
        return cls(dir=PortDir.OUT, role=PortRole.CLOCK, domain=domain,
                   bits=bits, naming=naming)

    @staticmethod
    def _parse_direction(d) -> 'PortDir':
        """Normalize a direction argument.

        Accepts a :class:`PortDir` enum or a string in either short
        (``'in'``, ``'out'``, ``'inout'``) or long (``'input'``,
        ``'output'``, ``'inout'``) form — so user code need not import
        ``PortDir`` just to override a default.
        """
        if isinstance(d, PortDir):
            return d
        if isinstance(d, str):
            alias = {'in': 'input', 'out': 'output', 'inout': 'inout'}
            return PortDir(alias.get(d.lower(), d.lower()))
        raise TypeError(
            f'direction must be a PortDir or str, got {type(d).__name__}'
        )

    @classmethod
    def supply(cls, domain=None, direction=PortDir.IN, *, bits=None, naming='bracketed'):
        """Supply-rail port (role POWER).

        Direction defaults to IN; override for bias generators that emit a
        rail.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain of the rail.
        direction : PortDir or str, default ``PortDir.IN``
            A :class:`PortDir` or one of ``'in'``, ``'out'``, ``'inout'``,
            ``'input'``, ``'output'`` (case-insensitive).
        bits : None
            Must stay ``None``; rails are scalar.
        naming : str, default 'bracketed'
            Stored on the descriptor; irrelevant for a scalar rail.

        Returns
        -------
        Port
            A descriptor with role ``PortRole.POWER``.

        Raises
        ------
        TypeError
            If ``bits`` is given, or ``direction`` is neither a
            :class:`PortDir` nor a string.
        ValueError
            If ``direction`` is a string that names no direction.
        """
        if bits is not None:
            raise TypeError(
                "Port.supply does not accept bits= — supplies are scalar rails. "
                "If you need a wide bias array, declare it via "
                "Port.analog_input(bits=N, role=PortRole.POWER) instead."
            )
        return cls(dir=cls._parse_direction(direction),
                   role=PortRole.POWER, domain=domain, naming=naming)

    @classmethod
    def ground(cls, domain=None, direction=PortDir.IN, *, bits=None, naming='bracketed'):
        """Ground port (role GROUND).

        Direction defaults to IN; override when a module generates its own
        ground reference.

        Parameters
        ----------
        domain : VoltageDomain, optional
            Voltage domain of the rail.
        direction : PortDir or str, default ``PortDir.IN``
            A :class:`PortDir` or one of ``'in'``, ``'out'``, ``'inout'``,
            ``'input'``, ``'output'`` (case-insensitive).
        bits : None
            Must stay ``None``; rails are scalar.
        naming : str, default 'bracketed'
            Stored on the descriptor; irrelevant for a scalar rail.

        Returns
        -------
        Port
            A descriptor with role ``PortRole.GROUND``.

        Raises
        ------
        TypeError
            If ``bits`` is given, or ``direction`` is neither a
            :class:`PortDir` nor a string.
        ValueError
            If ``direction`` is a string that names no direction.
        """
        if bits is not None:
            raise TypeError(
                "Port.ground does not accept bits= — grounds are scalar rails."
            )
        return cls(dir=cls._parse_direction(direction),
                   role=PortRole.GROUND, domain=domain, naming=naming)
