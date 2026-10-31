from dataclasses import dataclass, field
from typing import Any
from silk.lint.severity import Severity
from silk.lint.locations import (
    ViolationLocation, NetLocation, TerminalLocation,
)


@dataclass
class Violation:
    """One finding reported by a lint rule.

    :class:`InterfaceViolation` and :class:`JunctionViolation` subclass it and
    keep the field names of the pre-location-model API as properties. All
    three are plain dataclasses and are serialised by
    :class:`LintReport` using their class name.

    Attributes
    ----------
    rule_id : str
        Identifier of the rule that raised the finding, for example
        ``'STR-001'``.
    severity : Severity
        Severity of the finding. Normally the rule's ``severity``, which is
        its ``default_severity`` unless overridden at construction.
    message : str
        Human-readable description of the finding.
    location : ViolationLocation
        The design element the finding points at.
    context : dict
        Free-form extra facts the rule attached. Defaults to an empty dict.
        Report serialisation copies it into JSON, so values must be
        JSON-encodable.
    """
    rule_id: str
    severity: Severity
    message: str
    location: ViolationLocation
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterfaceViolation(Violation):
    """Port-interface-level violation (driver conflict, role mismatch, etc.).

    Adds no fields. It exists to expose the pre-location-model accessors
    ``net``, ``ports`` and ``kind`` as read-only properties derived from
    ``location`` and ``rule_id``. Constructor arguments and fields are those
    of :class:`Violation`.

    Attributes
    ----------
    rule_id, severity, message, location, context
        As for :class:`Violation`.
    """

    @property
    def net(self) -> str:
        """Name of the net the finding is on.

        Returns
        -------
        str
            ``location.net_name`` when the location is a
            :class:`NetLocation`, otherwise ``''``.
        """
        if isinstance(self.location, NetLocation):
            return self.location.net_name
        return ""

    @property
    def ports(self) -> list[str]:
        """Names of the ports involved in the finding.

        Returns
        -------
        list of str
            ``location.port_names`` (the location's own list, not a copy)
            when the location is a :class:`NetLocation`, otherwise a new
            empty list.
        """
        if isinstance(self.location, NetLocation):
            return self.location.port_names
        return []

    @property
    def kind(self) -> str:
        """Backward-compatible alias for ``rule_id``.

        Returns
        -------
        str
            The rule identifier.
        """
        return self.rule_id


@dataclass
class JunctionViolation(Violation):
    """Device junction overrange violation (post-simulation).

    ``worst_time`` is the independent-axis value at the worst-case sample,
    which is a time in seconds for the transient and behavioral paths but the
    sweep value for a ``.dc`` sweep (the axis the run actually swept). The
    message renders it with its axis name, so read the message rather than
    assuming seconds. AC and noise results never reach here — STR-001 declines
    them outright, see ``JunctionOverrangeRule.simulation_data_usable``.

    For a pair rated by one magnitude ``m``, ``measured_v`` and ``rated_max``
    are ``|V|`` at the worst sample and ``m``. For a pair rated by a signed
    range ``(lo, hi)`` they are the SIGNED worst voltage and the bound it
    crossed: ``hi`` (message ``exceeds <pair>_max``) or the negative ``lo``
    (message ``below <pair>_min``). A +1.0 V forward body bias on a sky130
    NMOS therefore reads ``measured_v=1.0, rated_max=0.3``, not 1.0 against
    1.95.

    Attributes
    ----------
    rule_id, severity, message, location, context
        As for :class:`Violation`.
    device_type : str
        Device type name the rating came from. Empty by default.
    measured_v : float
        Worst-case voltage in volts, as described above. Default ``0.0``.
    rated_max : float
        The rated bound that ``measured_v`` was compared with, in volts.
        Default ``0.0``.
    worst_time : float
        Independent-axis value at the worst sample: seconds for transient and
        behavioral runs, the sweep value for a ``.dc`` sweep. Default ``0.0``.
    """
    device_type: str = ""
    measured_v: float = 0.0
    rated_max: float = 0.0
    worst_time: float = 0.0

    @property
    def instance(self) -> str:
        """Hierarchical path of the instance the finding is on.

        Returns
        -------
        str
            ``location.instance_path`` when the location is a
            :class:`TerminalLocation`, otherwise ``''``.
        """
        if isinstance(self.location, TerminalLocation):
            return self.location.instance_path
        return ""

    @property
    def terminal_pair(self) -> tuple[str, ...]:
        """Terminals whose voltage difference was rated.

        Returns
        -------
        tuple of str
            ``location.terminal_pair`` when the location is a
            :class:`TerminalLocation`, otherwise ``()``.
        """
        if isinstance(self.location, TerminalLocation):
            return self.location.terminal_pair
        return ()

    @property
    def kind(self) -> str:
        """Backward-compatible alias for ``rule_id``.

        Returns
        -------
        str
            The rule identifier.
        """
        return self.rule_id
