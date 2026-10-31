from abc import ABC, abstractmethod
import typing as _typing
from typing import Iterator, Literal, Optional
from silk.lint.severity import Severity
from silk.lint.violations import Violation


# Valid categories — derived from RuleCategory Literal so they cannot drift
#: The rule categories silk recognises: ``"interface"``, ``"domain"``,
#: ``"stress"``, ``"structural"``, ``"isolation"`` and ``"solver"``. A
#: :class:`Rule` subclass's ``category`` must be one of them.
RuleCategory = Literal["interface", "domain", "stress", "structural", "isolation", "solver"]
#: The members of ``RuleCategory`` as a ``set`` of strings; the set that
#: :class:`Rule` validates ``category`` against.
VALID_CATEGORIES: set[str] = set(_typing.get_args(RuleCategory))


class Rule(ABC):
    """Base class for all analog lint rules.

    Subclass it, set the class attributes below and implement :meth:`check`;
    then add an instance to a :class:`RuleRegistry` with
    :meth:`RuleRegistry.add_rule`.

    Subclasses MUST set these class attributes (omitting one makes the
    constructor raise :class:`TypeError`):

    ``rule_id`` : str
        Stable identifier, for example ``"INT-001"``.
    ``category`` : str
        One of ``VALID_CATEGORIES``.
    ``default_severity`` : Severity
        Severity used unless overridden per instance. Defaults to
        ``Severity.WARNING``, so it is always present on the base class.
    ``applicable_to`` : tuple of type
        Types of scope the rule visits; the registry skips the rule when the
        scope is not an instance of one of them.
    ``rationale`` : str
        Prose explaining why the rule exists and how to fix a finding.

    Subclasses MAY set:

    ``requires_simulation`` : bool
        Default ``False``. When true the registry does not call
        :meth:`check` unless simulation data is available.
    ``config_schema`` : dict
        Mapping of configuration field name to type. Default ``{}``. Each
        instance gets its own copy; the registry does not validate ``config``
        against it.

    Subclasses MAY override :meth:`simulation_data_usable` to decline
    simulation data of a kind they cannot read, and MAY define
    ``check_with_context(scope, *, skipped=False)`` returning
    ``(violations, context)``; the registry prefers it over :meth:`check`
    when present.

    Parameters
    ----------
    config : dict, optional
        Rule-specific configuration. Copied into ``self.config``; ``None``
        gives an empty dict.
    severity_override : Severity, optional
        Severity for this instance. ``None`` uses ``default_severity``. The
        value is stored as ``self.severity``.

    Raises
    ------
    TypeError
        If the subclass leaves ``rule_id`` or ``rationale`` empty, sets a
        ``category`` outside ``VALID_CATEGORIES``, or leaves
        ``applicable_to`` empty.
    """
    rule_id: str = ""
    category: RuleCategory | str = ""
    default_severity: Severity = Severity.WARNING
    applicable_to: tuple[type, ...] = ()
    rationale: str = ""
    requires_simulation: bool = False
    config_schema: dict[str, type] = {}

    def __init__(
        self,
        config: Optional[dict] = None,
        severity_override: Optional[Severity] = None,
    ):
        if not self.rule_id:
            raise TypeError(f"{type(self).__name__} must set class attribute rule_id")
        if not self.rationale:
            raise TypeError(f"{type(self).__name__} must set class attribute rationale")
        if self.category not in VALID_CATEGORIES:
            raise TypeError(
                f"{type(self).__name__}.category={self.category!r} not in {VALID_CATEGORIES}"
            )
        if not self.applicable_to:
            raise TypeError(f"{type(self).__name__} must set class attribute applicable_to")

        self.config = dict(config) if config else {}
        self.severity = severity_override if severity_override is not None else self.default_severity
        self.config_schema = dict(type(self).config_schema)

    def simulation_data_usable(self, scope) -> bool:
        """Can this rule read the simulation data *scope* is carrying?

        Consulted by :meth:`RuleRegistry.run` only for rules that set
        ``requires_simulation``. Returning ``False`` records the visit as
        ``VisitOutcome.SKIPPED_NO_SIM`` — the same outcome as no simulation at
        all, since from the rule's point of view there is no data it can use —
        instead of dispatching :meth:`check`.

        ``has_simulation_data`` answers "was this module simulated at all?";
        this answers the narrower "and is it the *kind* of result I read?".
        STR-001, for instance, scores terminal voltages against PDK ratings in
        volts, which is meaningless for an AC or noise plot — a rule
        in that position must decline rather than report a confident wrong
        number, and must decline quietly, because a lint run over a PVT sweep
        or a multi-analysis testbench has to keep going.

        The default returns ``True``: a rule that does not care which analysis
        produced the data needs no opinion here.

        Parameters
        ----------
        scope : object
            The element being linted, as passed to :meth:`check`.

        Returns
        -------
        bool
            ``True`` if the rule can use the simulation data; ``False`` to
            decline it.
        """
        return True

    @abstractmethod
    def check(self, scope) -> Iterator[Violation]:
        """Yield zero or more :class:`Violation` objects for *scope*.

        Abstract: subclasses must override it. The registry calls it for each
        scope the rule applies to (and, for ``requires_simulation`` rules,
        only when usable simulation data exists), and records ``PASS`` when
        nothing is yielded and ``VIOLATION`` otherwise.

        Parameters
        ----------
        scope : object
            The element to lint; an instance of one of ``applicable_to``
            (typically a :class:`~silk.circuit.Module`).

        Yields
        ------
        Violation
            One finding per problem.

        Raises
        ------
        NotImplementedError
            From the base implementation, if a subclass calls it through
            ``super()``.
        """
        raise NotImplementedError(f"{type(self).__name__}.check must be implemented")
