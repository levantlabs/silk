import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional
from silk.lint.rule import Rule
from silk.lint.violations import Violation
from silk.lint.locations import ViolationLocation, ModuleLocation
from silk.lint.exclusions import ScopeExclusion


class VisitOutcome(Enum):
    """Result recorded for one rule dispatch in a :class:`VisitEvent`.

    The members, with their string values, are:

    ``PASS`` (``"pass"``)
        The rule ran and yielded nothing.
    ``INCOMPLETE`` (``"incomplete"``)
        The rule yielded nothing, but its stress-coverage context reports
        partially checked or unchecked devices
        (:meth:`RuleRegistry.run` sets this only for such rules).
    ``VIOLATION`` (``"violation"``)
        The rule yielded at least one finding.
    ``EXCLUDED`` (``"excluded"``)
        Reserved for a scope-excluded subtree. :meth:`RuleRegistry.run` never
        records it; coverage reports count it in their ``excluded`` column
        if a visit log carries it.
    ``SKIPPED_NO_SIM`` (``"skipped_no_sim"``)
        The rule needs simulation data and none usable was available.
    """
    PASS = "pass"
    INCOMPLETE = "incomplete"
    VIOLATION = "violation"
    EXCLUDED = "excluded"
    SKIPPED_NO_SIM = "skipped_no_sim"


@dataclass
class VisitEvent:
    """One entry per (rule × scope) dispatched by the registry in v1.

    In v1 rules receive the full scope; the registry records one event per
    (rule × scope) pair with the scope's location. When net/instance
    iteration moves into the rule dispatch loop in v2, this becomes one
    entry per (rule × element).

    Recorded as an event list (not aggregated counts) so v2 can merge
    across runs via concatenation without schema changes.

    Attributes
    ----------
    rule_id : str
        Identifier of the rule that was dispatched.
    location : ViolationLocation
        Location of the scope the rule ran on. In v1 the registry always
        records a :class:`ModuleLocation` named after the scope's ``name``
        (or ``'?'`` when it has none).
    outcome : VisitOutcome
        What happened to the dispatch.
    timestamp : float
        Wall-clock time of the dispatch, in seconds since the Unix epoch
        (``time.time()``).
    simulator : str, optional
        Name of the simulator behind the data. :meth:`RuleRegistry.run` leaves
        it ``None``.
    context : dict
        Extra facts from the rule's ``check_with_context``, for example the
        ``'stress_coverage'`` entry STR-001 provides; empty for rules without
        that hook.
    metric_results : dict
        Reserved for per-visit metric values; :meth:`RuleRegistry.run`
        leaves it empty.
    """
    rule_id: str
    location: ViolationLocation
    outcome: VisitOutcome
    timestamp: float
    simulator: Optional[str] = None
    context: dict = field(default_factory=dict)
    metric_results: dict = field(default_factory=dict)


@dataclass
class RegistryRunResult:
    """What :meth:`RuleRegistry.run` returns.

    Attributes
    ----------
    violations : list of Violation
        Every finding yielded by the rules that ran, in rule order. Waivers
        and exclusions have not been applied to this list.
    visit_log : list of VisitEvent
        One event per dispatched rule, in rule order.
    """
    violations: list[Violation] = field(default_factory=list)
    visit_log: list[VisitEvent] = field(default_factory=list)


# v1 profile registry — only "default" is supported
SUPPORTED_PROFILES = {"default"}


def module_has_simulation_data(module) -> bool:
    """Return True when *module* carries simulation data a rule can read.

    The single predicate behind ``has_simulation_data=`` for every entry point
    that runs simulation-requiring rules (``generate_report``,
    ``check_junction_overrange``). Two independent flags mean "this module has
    been simulated":

    - ``_behavioral_mode`` -- populated by ``simulate_behavioral()``;
    - ``sim_results_loaded`` -- populated by a real SPICE run (``simulate()``).

    Rules such as STR-001 read both data paths, so gating on only one of the
    flags silently records the rule as ``SKIPPED_NO_SIM`` on a module that was
    in fact measured. Callers must use this helper rather than re-deriving
    the condition.

    Parameters
    ----------
    module : object
        The module (or any object) to inspect. Missing attributes count as
        false, so no exception is raised for an object without the flags.

    Returns
    -------
    bool
        ``True`` if either flag is truthy on *module*.
    """
    return bool(getattr(module, "_behavioral_mode", False)
                or getattr(module, "sim_results_loaded", False))


class RuleRegistry:
    """Holds rules, dispatches them over a scope, records visit events.

    A registry also carries scope exclusions and waivers (loaded from YAML with
    :meth:`load_exclusions` and :meth:`load_waivers`). :meth:`run` itself
    applies neither; the report-generating code consults them afterwards (see
    :func:`apply_waivers` and :meth:`is_excluded`).

    Parameters
    ----------
    rules : iterable of Rule, optional
        Initial rules, kept in the order given. ``None`` gives an empty
        registry.
    exclusions : list of ScopeExclusion, optional
        Initial scope exclusions. ``None`` gives none.
    waivers : list of Waiver, optional
        Initial waivers. ``None`` gives none.

    Examples
    --------
    >>> from silk import RuleRegistry
    >>> registry = RuleRegistry.default()
    >>> len(registry.rules) > 0
    True
    """

    def __init__(
        self,
        rules: Optional[Iterable[Rule]] = None,
        exclusions: Optional[list] = None,
        waivers: Optional[list] = None,
    ):
        self._rules: list[Rule] = list(rules) if rules else []
        self._exclusions: list[ScopeExclusion] = list(exclusions) if exclusions else []
        self._waivers: list = list(waivers) if waivers else []

    @classmethod
    def default(cls, include_optional: bool = False) -> "RuleRegistry":
        """Build the registry with silk's built-in rule catalog.

        Parameters
        ----------
        include_optional : bool, optional
            When true, also register the opt-in rules from
            ``silk.lint.rules.OPTIONAL_CATALOG`` (see
            ``silk.lint.rules.FULL_CATALOG``). Default ``False``.

        Returns
        -------
        RuleRegistry
            A new registry holding fresh rule instances, with no exclusions
            or waivers.
        """
        from silk.lint.rules import DEFAULT_CATALOG, FULL_CATALOG
        return cls(rules=FULL_CATALOG() if include_optional else DEFAULT_CATALOG())

    def add_rule(self, rule: Rule) -> None:
        """Append *rule* to the registry.

        Parameters
        ----------
        rule : Rule
            The rule instance to register. It is not checked for a duplicate
            ``rule_id``; rules run in the order they were added.
        """
        self._rules.append(rule)

    def load_exclusions(self, path) -> None:
        """Load scope exclusions from a YAML file and append them.

        Parameters
        ----------
        path : str or os.PathLike
            YAML file with an ``exclusions:`` list; see
            :meth:`ExclusionFile.load` for the entry format.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If an entry has an empty or missing ``reason``.
        """
        from silk.lint.exclusions import ExclusionFile
        ef = ExclusionFile.load(path)
        self._exclusions.extend(ef.exclusions)

    def load_waivers(self, path) -> None:
        """Load waivers from a YAML file and append them.

        Parameters
        ----------
        path : str or os.PathLike
            YAML file with a ``waivers:`` list; see :meth:`WaiverFile.load`
            for the entry format.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If an entry has an empty or missing ``reason``, or an
            unparseable ``expires`` date.
        KeyError
            If an entry has no ``rule_id``.
        """
        from silk.lint.waivers import WaiverFile
        wf = WaiverFile.load(path)
        self._waivers.extend(wf.waivers)

    def is_excluded(self, instance_path: str) -> Optional[ScopeExclusion]:
        """Return the exclusion covering *instance_path*, if any.

        Parameters
        ----------
        instance_path : str
            Dot-separated hierarchical instance path.

        Returns
        -------
        ScopeExclusion or None
            The first loaded exclusion that matches the path or one of its
            ancestors (see :meth:`ScopeExclusion.matches_instance_path`);
            ``None`` if none does.
        """
        for e in self._exclusions:
            if e.matches_instance_path(instance_path):
                return e
        return None

    @property
    def rules(self) -> list[Rule]:
        """The registered rules, in dispatch order.

        Returns
        -------
        list of Rule
            A new list on each access; mutating it does not change the
            registry (use :meth:`add_rule`).
        """
        return list(self._rules)

    def run(
        self,
        scope,
        profile: str = "default",
        categories: Optional[set[str]] = None,
        has_simulation_data: bool = False,
    ) -> RegistryRunResult:
        """Dispatch rules against *scope*; return violations and visit log.

        Rules are skipped (no visit event recorded) when they are filtered out
        by the `categories` parameter or when scope type does not match the
        rule's ``applicable_to``. Rules that are dispatched produce exactly one
        visit event with outcome ``PASS``, ``VIOLATION``, ``INCOMPLETE`` or
        ``SKIPPED_NO_SIM``. Rules may implement
        ``check_with_context(scope, skipped=False)`` to return findings plus
        visit context in one pass; ``skipped=True`` must not measure.

        This means a ``run_category("structural", ...)`` call produces a visit
        log containing only structural rules. :class:`CoverageReport` must
        therefore be computed from a registry run that does NOT use category or
        ``applicable_to`` filtering, or it will under-count "checked" rules.

        Parameters
        ----------
        scope : object
            The element to lint, typically a :class:`~silk.circuit.Module`.
        profile : str, optional
            Rule profile. Only ``"default"`` is supported.
        categories : set of str, optional
            When given, only rules whose ``category`` is in the set run.
            ``None`` (default) runs every category.
        has_simulation_data : bool, optional
            Whether *scope* carries simulation results. Rules with
            ``requires_simulation`` are recorded as ``SKIPPED_NO_SIM`` when
            this is false (or the rule's ``simulation_data_usable`` declines).
            Use :func:`module_has_simulation_data` to derive it from a
            module. Default ``False``.

        Returns
        -------
        RegistryRunResult
            The violations and the visit log. Waivers and exclusions are not
            applied.

        Raises
        ------
        ValueError
            If *profile* is not ``"default"``.
        """
        if profile not in SUPPORTED_PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}; v1 supports {SUPPORTED_PROFILES}"
            )

        from silk.circuit.modules import Module
        if isinstance(scope, Module):
            from silk.circuit.wiring_consistency import assert_consistent
            assert_consistent(scope)

        result = RegistryRunResult()

        for rule in self._rules:
            if categories is not None and rule.category not in categories:
                continue
            if not isinstance(scope, rule.applicable_to):
                continue

            location_for_skip = self._default_location(scope)
            # A simulation-requiring rule is skipped when the module was not
            # simulated at all, and equally when it was simulated with an
            # analysis whose results the rule cannot read (TASK-81) — from the
            # rule's point of view both mean "no data I can use".
            context_check = getattr(rule, "check_with_context", None)
            if rule.requires_simulation and (
                    not has_simulation_data or not rule.simulation_data_usable(scope)):
                result.visit_log.append(VisitEvent(
                    rule_id=rule.rule_id,
                    location=location_for_skip,
                    outcome=VisitOutcome.SKIPPED_NO_SIM,
                    context=context_check(scope, skipped=True)[1] if context_check else {},
                    timestamp=time.time(),
                ))
                continue

            if context_check:
                violations_from_rule, context = context_check(scope)
            else:
                violations_from_rule, context = list(rule.check(scope)), {}
            outcome = VisitOutcome.VIOLATION if violations_from_rule else VisitOutcome.PASS
            stress = context.get('stress_coverage', {}).get('summary', {})
            if not violations_from_rule and (stress.get('partial') or stress.get('none')):
                outcome = VisitOutcome.INCOMPLETE
            result.visit_log.append(VisitEvent(
                context=context,
                rule_id=rule.rule_id,
                location=location_for_skip,
                outcome=outcome,
                timestamp=time.time(),
            ))
            result.violations.extend(violations_from_rule)

        return result

    def run_category(self, category: str, scope, **kwargs) -> RegistryRunResult:
        """Run only the rules in one category.

        Equivalent to :meth:`run` with ``categories={category}``.

        Parameters
        ----------
        category : str
            Category name, for example ``"structural"``.
        scope : object
            The element to lint, as for :meth:`run`.
        **kwargs
            Passed to :meth:`run` (``profile``, ``has_simulation_data``).
            Passing ``categories`` as well raises :class:`TypeError`.

        Returns
        -------
        RegistryRunResult
            As for :meth:`run`.

        Raises
        ------
        ValueError
            If ``profile`` is not ``"default"``.
        """
        return self.run(scope, categories={category}, **kwargs)

    @staticmethod
    def _default_location(scope) -> ViolationLocation:
        name = getattr(scope, "name", "") or getattr(scope, "__name__", "") or "?"
        return ModuleLocation(module_name=name)
