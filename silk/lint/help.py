"""help_rule and list_rules — interactive exploration of the rule catalog."""
from silk.lint.rules import DEFAULT_CATALOG, FULL_CATALOG


def help_rule(rule_id: str) -> str:
    """Return formatted metadata + rationale prose for a rule by ID.

    Covers every shipped rule — the default catalog and the opt-in rules
    from ``OPTIONAL_CATALOG`` — and says which one the rule belongs to.

    Parameters
    ----------
    rule_id : str
        Rule identifier, for example ``"STR-001"``.

    Returns
    -------
    str
        A text block with the rule id and class name, category, default
        severity, whether it requires simulation, whether it is in the default
        catalog, and the rationale.

    Raises
    ------
    KeyError
        If *rule_id* is not a shipped rule.
    """
    default_ids = {r.rule_id for r in DEFAULT_CATALOG()}
    for rule in FULL_CATALOG():
        if rule.rule_id == rule_id:
            in_default = rule.rule_id in default_ids
            membership = (
                "yes" if in_default
                else "no (opt-in: RuleRegistry.default(include_optional=True))"
            )
            return (
                f"{rule.rule_id} — {type(rule).__name__}\n"
                f"Category: {rule.category}\n"
                f"Default severity: {rule.default_severity.to_string()}\n"
                f"Requires simulation: {rule.requires_simulation}\n"
                f"In default catalog: {membership}\n"
                f"\n"
                f"{rule.rationale}\n"
            )
    raise KeyError(f"Unknown rule: {rule_id}")


def list_rules(include_optional: bool = False) -> list:
    """Return the rule IDs in the default catalog.

    Pass ``include_optional=True`` to append the opt-in rule IDs
    (``OPTIONAL_CATALOG``) so the list covers everything ``help_rule``
    resolves.

    Parameters
    ----------
    include_optional : bool, optional
        Also list the opt-in rules. Default ``False``.

    Returns
    -------
    list of str
        Rule identifiers in catalog order.
    """
    catalog = FULL_CATALOG() if include_optional else DEFAULT_CATALOG()
    return [r.rule_id for r in catalog]
