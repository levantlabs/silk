"""Name-based port-role inference.

Port roles are inferred from names only when a ruleset is supplied.

silk source contains only the **matching engine and zero patterns**. Patterns
are *data*: a user-defined, replaceable ruleset (build in Python or load YAML).
silk ships an **opt-in default** (:meth:`RoleRuleset.default`) — nothing infers
unless a ruleset is loaded/passed (memory ``feedback_port_role_inference``).

Public surface::

    rs = RoleRuleset.default()                          # silk's shipped opt-in default
    rs = RoleRuleset.from_yaml("house.yaml")            # a company's own (replaces default)
    rs = RoleRuleset().add("*vbias*", PortRole.ANALOG)  # build/extend in code; add() chainable
    rs.classify(name) -> Optional[PortRole]             # name → role or None

    infer_port_roles(module, ruleset) -> InferReport    # fill unset roles on module.pin_list
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import yaml

from silk.circuit.ports import PortRole


class RoleRulesetError(ValueError):
    """Raised at load time for a malformed ruleset — unknown ``role:``,
    neither/both of ``match:``/``regex:``, a bad ``version:``, or an invalid
    regex. Subclasses :class:`ValueError` so ``except ValueError`` still catches
    it (matching :class:`silk.pdk.registry.PDKRegistryError`'s contract)."""


class PortRoleInferenceWarning(UserWarning):
    """Legacy warning category retained for imports.

    Live module placements refresh inferred roles in their parent bindings,
    so inference no longer emits a stale-copy warning.
    """


# Only version 1 is defined today; anything else is a loud load-time error.
_SUPPORTED_VERSIONS = frozenset({1})

# Specificity metric. "Literal" chars are those OUTSIDE the relevant
# metacharacter set; a pattern with none of them is an *exact literal*.
#   glob  — metachars counted are only ``*`` and ``?``.
#   regex — the standard regex metacharacter set, so an anchored regex like
#           ``^d\[[0-9]+\]$`` is NOT mistaken for an exact literal.
_GLOB_META = frozenset("*?")
_REGEX_META = frozenset(r".^$*+?{}[]\|()")

# Located __file__-relative, matching silk's packaged-data convention
# (silk/pdk/registry.py:17). silk uses no importlib.resources.
_DEFAULT_YAML = os.path.join(
    os.path.dirname(__file__), "data", "port-roles.default.yaml"
)


@dataclass
class _Rule:
    """One ``(pattern, role, kind)`` rule with a precomputed specificity key."""
    pattern: str          # glob patterns are stored lowercased; regex kept raw
    role: PortRole
    kind: str             # 'glob' | 'regex'
    order: int            # declaration index (lower = earlier)
    regex: object = field(default=None, repr=False)  # compiled re when kind=='regex'

    def matches(self, name_lower: str, name_raw: str) -> bool:
        if self.kind == "glob":
            return fnmatch.fnmatchcase(name_lower, self.pattern)
        return self.regex.search(name_raw) is not None

    @property
    def _meta(self) -> frozenset:
        return _GLOB_META if self.kind == "glob" else _REGEX_META

    @property
    def is_exact(self) -> bool:
        return not any(c in self._meta for c in self.pattern)

    @property
    def literal_count(self) -> int:
        return sum(1 for c in self.pattern if c not in self._meta)

    @property
    def specificity(self) -> Tuple[bool, int, int]:
        # Higher tuple == more specific. Order: exact-literal beats any
        # patterned rule; else more literal chars; final tiebreak declaration
        # order (earlier wins → larger ``-order``).
        return (self.is_exact, self.literal_count, -self.order)


class RoleRuleset:
    """An ordered set of name→role rules. Pure/stateless w.r.t. any module."""

    def __init__(self) -> None:
        self._rules: List[_Rule] = []

    def __len__(self) -> int:
        """Number of rules in the set."""
        return len(self._rules)

    # -- construction -------------------------------------------------------

    def add(self, pattern: str, role) -> "RoleRuleset":
        """Add a **glob** rule (``fnmatch``-style, e.g. ``*vbias*``). Chainable.

        For regex rules, load a YAML ruleset with a ``regex:`` key — the code
        API is intentionally glob-only so contributors need no regex.

        Parameters
        ----------
        pattern : str
            Glob matched case-insensitively against a port name.
        role : PortRole or str
            The role to assign; a string is matched case-insensitively
            against the :class:`~silk.circuit.ports.PortRole` values.

        Returns
        -------
        RoleRuleset
            ``self``, so calls chain.

        Raises
        ------
        RoleRulesetError
            If ``role`` is not a valid role.
        """
        self._append(pattern=str(pattern), role=role, kind="glob")
        return self

    def _append(self, *, pattern: str, role, kind: str) -> None:
        role = self._coerce_role(role)
        if kind == "glob":
            rule = _Rule(pattern=pattern.lower(), role=role, kind="glob",
                         order=len(self._rules))
        elif kind == "regex":
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                raise RoleRulesetError(f"invalid regex {pattern!r}: {e}")
            rule = _Rule(pattern=pattern, role=role, kind="regex",
                         order=len(self._rules), regex=compiled)
        else:  # pragma: no cover - internal guard
            raise RoleRulesetError(f"internal: unknown rule kind {kind!r}")
        self._rules.append(rule)

    @staticmethod
    def _coerce_role(role) -> PortRole:
        if isinstance(role, PortRole):
            return role
        try:
            return PortRole(role.lower() if isinstance(role, str) else role)
        except (ValueError, KeyError):
            valid = ", ".join(r.value for r in PortRole)
            raise RoleRulesetError(
                f"unknown role {role!r}; must be one of: {valid}")

    # -- matching -----------------------------------------------------------

    def classify(self, name: str) -> Optional[PortRole]:
        """Return the most-specific matching rule's :class:`PortRole`, or
        ``None``. Case-insensitive; no bus-name stripping (v1)."""
        if name is None:
            return None
        name_lower = name.lower()
        best: Optional[_Rule] = None
        for rule in self._rules:
            if rule.matches(name_lower, name):
                if best is None or rule.specificity > best.specificity:
                    best = rule
        return best.role if best is not None else None

    # -- loaders ------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path) -> "RoleRuleset":
        """Load a ruleset from a YAML file. See ``docs/authoring-port-role-rules.md`` for the schema.

        A user's ruleset supersedes the shipped default. Raises
        ``RoleRulesetError`` (loudly) on any malformed rule.

        Parameters
        ----------
        path : str or os.PathLike
            YAML file to read.

        Returns
        -------
        RoleRuleset
            A new ruleset holding the file's rules.

        Raises
        ------
        RoleRulesetError
            If the file is not a mapping, has an unsupported ``version``, or
            contains a malformed rule, role or regex.
        FileNotFoundError
            If ``path`` does not exist.
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls._from_mapping(data, source=str(path))

    @classmethod
    def default(cls) -> "RoleRuleset":
        """silk's shipped **opt-in** default ruleset
        (``silk/circuit/data/port-roles.default.yaml``).

        Returns a fresh instance each call, so a caller may ``.add()`` house
        rules to it without mutating a shared singleton."""
        return cls.from_yaml(_DEFAULT_YAML)

    @classmethod
    def _from_mapping(cls, data, source: str) -> "RoleRuleset":
        if not isinstance(data, dict):
            raise RoleRulesetError(
                f"{source}: top-level YAML must be a mapping, got "
                f"{type(data).__name__}")
        version = data.get("version")
        if version not in _SUPPORTED_VERSIONS:
            raise RoleRulesetError(
                f"{source}: unsupported version {version!r}; supported: "
                f"{sorted(_SUPPORTED_VERSIONS)}")
        rules = data.get("rules") or []
        if not isinstance(rules, list):
            raise RoleRulesetError(
                f"{source}: 'rules' must be a list, got {type(rules).__name__}")
        rs = cls()
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise RoleRulesetError(
                    f"{source}: rule #{i} must be a mapping, got "
                    f"{type(rule).__name__}")
            has_match = "match" in rule
            has_regex = "regex" in rule
            if has_match == has_regex:   # neither (missing) or both (dual)
                raise RoleRulesetError(
                    f"{source}: rule #{i} must have exactly one of "
                    f"'match:' or 'regex:'")
            if "role" not in rule:
                raise RoleRulesetError(f"{source}: rule #{i} missing 'role:'")
            if has_match:
                rs._append(pattern=str(rule["match"]), role=rule["role"],
                           kind="glob")
            else:
                rs._append(pattern=str(rule["regex"]), role=rule["role"],
                           kind="regex")
        return rs


@dataclass
class InferReport:
    """Result of :func:`infer_port_roles <silk.circuit.infer_port_roles>`.

    Attributes
    ----------
    filled : list of tuple
        ``(port_name, role)`` for each port whose role this call set.
    unmatched : list of str
        Names of unset ports the ruleset did not classify — these stay
        ``None`` so the #9 annotation-coverage gate still flags them.
    """
    filled: List[Tuple[str, PortRole]]
    unmatched: List[str]


def infer_port_roles(module, ruleset) -> InferReport:
    """Fill each *unset* :class:`PortRole` on ``module.pin_list`` from *ruleset*.

    Roles are filled on the live definition after its ports are declared.
    Explicit roles are never overwritten; unmatched ports stay ``None``.
    Parents refresh placement metadata and net bindings from that definition,
    so inference before or after ``add_module`` reaches the parent graph.
    Filled roles carry ``role_inferred=True`` to distinguish name-pattern
    guesses from explicit declarations. Walk ``Module.iter_instances()`` to
    reach a hierarchy's definitions; repeated placements share one Module.

    Parameters
    ----------
    module : Module
        The module whose own ``pin_list`` is annotated in place.
    ruleset : RoleRuleset
        The name→role ruleset to classify with (e.g. ``RoleRuleset.default()``
        or a house ruleset).

    Returns
    -------
    InferReport
        ``filled`` and ``unmatched`` port names (see ``InferReport``).
    """
    filled: List[Tuple[str, PortRole]] = []
    unmatched: List[str] = []
    pin_list = getattr(module, "pin_list", None) or {}
    for port_name, pin in pin_list.items():
        if getattr(pin, "role", None) is not None:
            continue                                # NEVER clobber an explicit role
        role = ruleset.classify(port_name)
        if role is not None:
            pin.role = role                         # the PortRole enum (M4)
            pin.role_inferred = True                # a guess, not a statement
            filled.append((port_name, role))
        else:
            unmatched.append(port_name)
    return InferReport(filled=filled, unmatched=unmatched)


# ---------------------------------------------------------------------------
# CLI — ``python -m silk.circuit.port_roles``
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Check a port-role ruleset YAML: validate it loads, and optionally
    dry-run it against sample port names.

    Modes
    -----
    - default / ``--file PATH`` — load the ruleset (silk's shipped default if
      no ``--file``) and report OK + the rule count, or the load error.
    - ``--names NAME [NAME ...]`` — also classify each name, showing what the
      ruleset resolves it to (or ``(unmatched)``).

    Exit code ``0`` = ruleset valid, ``1`` = malformed ruleset content,
    ``2`` = file not found / unreadable — so it doubles as a CI / pre-commit
    gate for a house ruleset. See ``docs/authoring-port-role-rules.md``.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog='python -m silk.circuit.rolecheck',
        description='Check a port-role ruleset YAML (validate + optional '
                    'dry-run). See docs/authoring-port-role-rules.md.')
    parser.add_argument(
        '--file', default=None,
        help="ruleset YAML to check (default: silk's shipped "
             "port-roles.default.yaml).")
    parser.add_argument(
        '--names', nargs='+', default=None, metavar='NAME',
        help='port names to dry-run against the ruleset, showing what each '
             'classifies to.')
    args = parser.parse_args(argv)

    label = args.file or 'default (port-roles.default.yaml)'
    try:
        rs = RoleRuleset.from_yaml(args.file) if args.file else RoleRuleset.default()
    except FileNotFoundError as e:
        print(f'ERROR: cannot read {args.file!r}: {e}', file=sys.stderr)
        return 2
    except RoleRulesetError as e:
        print(f'INVALID: {label}: {e}', file=sys.stderr)
        return 1

    print(f'OK: {label}: {len(rs)} rule(s)')
    if args.names:
        for name in args.names:
            role = rs.classify(name)
            print(f'  {name:<16} -> '
                  f'{role.value if role is not None else "(unmatched)"}')
    return 0
