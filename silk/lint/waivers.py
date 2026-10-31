"""File-based waiver system for silk lint.

Waivers suppress specific violations with required justification. They
are first-class audit artifacts: every waived violation is still reported
in the LintReport's `waivers_applied` section, not hidden.
"""
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional
import yaml
from silk.lint.violations import Violation
from silk.lint.locations import (
    NetLocation, PortLocation, InstanceLocation, TerminalLocation, ModuleLocation,
)


LOCATION_TYPE_TO_CLASS = {
    "net": NetLocation,
    "port": PortLocation,
    "instance": InstanceLocation,
    "terminal": TerminalLocation,
    "module": ModuleLocation,
}


@dataclass
class WaiverMatcher:
    """Structured predicate on a violation's location.

    Attributes
    ----------
    location_type : str
        One of ``"net"``, ``"port"``, ``"instance"``, ``"terminal"``,
        ``"module"``.
    fields : dict
        Maps attribute_name to the expected value on the location object.
        All fields must match for the waiver to apply. An empty dict matches
        every location of the given type.
    """
    location_type: str
    fields: dict = field(default_factory=dict)

    def matches(self, violation: Violation) -> bool:
        """Test whether *violation*'s location satisfies the predicate.

        Parameters
        ----------
        violation : Violation
            The finding to test; only its ``location`` is read.

        Returns
        -------
        bool
            ``True`` if ``location_type`` names a known type, the location is
            an instance of that location class, and every entry of ``fields``
            equals the location attribute of the same name (a missing
            attribute reads as ``None``). ``False`` otherwise, including for
            an unknown ``location_type``; no exception is raised.
        """
        expected_class = LOCATION_TYPE_TO_CLASS.get(self.location_type)
        if expected_class is None:
            return False
        if not isinstance(violation.location, expected_class):
            return False
        for field_name, expected_value in self.fields.items():
            actual = getattr(violation.location, field_name, None)
            if actual != expected_value:
                return False
        return True


@dataclass
class Waiver:
    """A suppression of a specific violation, with required justification.

    Waivers are first-class audit artifacts: every waived violation is
    still reported (in the ``waivers_applied`` section), not hidden.

    Attributes
    ----------
    rule_id : str
        Identifier of the rule whose findings this waiver can suppress.
    matcher : WaiverMatcher
        Predicate on the finding's location.
    reason : str
        Justification. Required; construction raises :class:`ValueError`
        when it is empty or only whitespace.
    author : str
        Who granted the waiver. Empty by default.
    expires : datetime.date, optional
        Last day the waiver applies. After that date it no longer matches.
        ``None`` (default) never expires.

    Raises
    ------
    ValueError
        If ``reason`` is empty or whitespace-only.
    """
    rule_id: str
    matcher: WaiverMatcher
    reason: str
    author: str = ""
    expires: Optional[date] = None

    def __post_init__(self):
        if not self.reason or not self.reason.strip():
            raise ValueError("Waiver requires a non-empty reason string")

    def matches(self, violation: Violation) -> bool:
        """Test whether this waiver suppresses *violation*.

        Parameters
        ----------
        violation : Violation
            The finding to test.

        Returns
        -------
        bool
            ``True`` if the rule ids are equal, the waiver has not expired
            (``expires`` earlier than today's date means expired; the
            expiry day itself still applies) and ``matcher`` matches the
            location. ``False`` otherwise.
        """
        if violation.rule_id != self.rule_id:
            return False
        if self.expires is not None and self.expires < date.today():
            return False
        return self.matcher.matches(violation)


@dataclass
class WaiverFile:
    """A set of waivers loaded from a YAML file.

    Attributes
    ----------
    waivers : list of Waiver
        The waivers, in file order. Empty by default.
    """
    waivers: list = field(default_factory=list)

    @classmethod
    def load(cls, path) -> "WaiverFile":
        """Read waivers from a YAML file.

        The file is a mapping with a ``waivers`` list. Each entry has
        ``rule_id`` and ``reason`` (both required), an optional ``matcher``
        mapping with ``location_type`` and ``fields``, and optional
        ``author`` and ``expires`` (an ISO ``YYYY-MM-DD`` date). An empty
        file, or one without ``waivers``, loads as no waivers.

        Parameters
        ----------
        path : str or os.PathLike
            The YAML file.

        Returns
        -------
        WaiverFile
            The parsed waivers.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If an entry has a missing or empty ``reason``, or an ``expires``
            value that is not an ISO date.
        KeyError
            If an entry has no ``rule_id``.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        waivers_raw = data.get("waivers", [])
        waivers = []
        for idx, entry in enumerate(waivers_raw):
            if "reason" not in entry or not entry["reason"]:
                raise ValueError(
                    f"Waiver #{idx} in {path} missing required 'reason' field"
                )
            matcher_data = entry.get("matcher", {})
            matcher = WaiverMatcher(
                location_type=matcher_data.get("location_type", ""),
                fields=matcher_data.get("fields", {}),
            )
            expires = entry.get("expires")
            if expires is not None and not isinstance(expires, date):
                expires = date.fromisoformat(str(expires))
            waivers.append(Waiver(
                rule_id=entry["rule_id"],
                matcher=matcher,
                reason=entry["reason"],
                author=entry.get("author", ""),
                expires=expires,
            ))
        return cls(waivers=waivers)


def apply_waivers(
    violations: list,
    waivers: list,
) -> tuple:
    """Split violations into those a waiver suppresses and those it does not.

    A violation is suppressed when any waiver's :meth:`Waiver.matches` is
    true for it. Order within each list follows the input order.

    Parameters
    ----------
    violations : list of Violation
        The findings to partition.
    waivers : list of Waiver
        The waivers to apply.

    Returns
    -------
    tuple of list
        ``(unsuppressed, suppressed)``, each a list of :class:`Violation`.
        Nothing is removed from the original list.
    """
    unsuppressed = []
    suppressed = []
    for v in violations:
        if any(w.matches(v) for w in waivers):
            suppressed.append(v)
        else:
            unsuppressed.append(v)
    return unsuppressed, suppressed
