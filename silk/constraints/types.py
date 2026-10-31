"""Device-relational layout constraints.

Scope (per ``DESIGN.md`` )
----------------------------
This module holds **only** constraints that talk about a *set* of
devices — there is no single owner object that can carry the
information. Net-related budgets (IR drop, current density, max
length) live as fields on :class:`silk.circuit.net.Net`. Port
semantics (power, ground, clock) live in
:class:`silk.circuit.circuit_elements.PortRole`. Both round-trip through
the silk JSON serialization bridge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from silk.constraints._netlist_view import NetlistView
from silk.constraints.envelope import ConstraintEnvelope


# ---------------------------------------------------------------------------
# Symmetry & matching
# ---------------------------------------------------------------------------

class SymmetricBlocks(ConstraintEnvelope):
    """Mirror-symmetric placement along an axis.

    Each entry in ``pairs`` is either a 2-element list (two instances
    mirrored across the axis) or a 1-element list (a self-symmetric
    instance lying on the axis). This is a hard constraint by default.

    Attributes
    ----------
    constraint : str
        Type tag, always ``'SymmetricBlocks'``; it is the discriminator in
        the serialized JSON.
    pairs : list of list of str
        Instance-name groups, at least one. Each group has one name
        (self-symmetric) or two (mirrored).
    direction : str
        ``'H'`` or ``'V'``. silk stores it and does
        not interpret it. Required.
    severity : str
        ``'hard'`` or ``'soft'``; defaults to ``'hard'``.

    Raises
    ------
    ValueError
        On construction, if a field is invalid or unknown (pydantic's
        ``ValidationError``, a ``ValueError`` subclass).

    Examples
    --------
    >>> from silk.constraints import SymmetricBlocks
    >>> c = SymmetricBlocks(pairs=[['m1', 'm2'], ['m_tail']], direction='V')
    >>> c.constraint, c.severity
    ('SymmetricBlocks', 'hard')
    """

    constraint: Literal["SymmetricBlocks"] = "SymmetricBlocks"
    pairs: List[List[str]] = Field(min_length=1)
    direction: Literal["H", "V"]
    severity: Literal["hard", "soft"] = "hard"

    @field_validator("pairs")
    @classmethod
    def _check_pair_sizes(cls, pairs: List[List[str]]) -> List[List[str]]:
        for i, p in enumerate(pairs):
            if len(p) not in (1, 2):
                raise ValueError(
                    f"pairs[{i}] must have 1 (self-symmetric) or 2 "
                    f"(mirror) entries; got {len(p)}"
                )
        return pairs

    def validate_against(self, netlist) -> List[str]:
        """Check that every named instance exists in a netlist.

        Parameters
        ----------
        netlist : silk.circuit.Module
            The module whose instance names (the keys of its ``elem_dict``)
            are checked against; a pre-built name view is also accepted.

        Returns
        -------
        list of str
            One message per unknown instance, in the order of ``pairs``;
            empty when all exist.
        """
        view = NetlistView.of(netlist)
        all_refs = [name for pair in self.pairs for name in pair]
        return [
            f"SymmetricBlocks: unknown instance '{n}'"
            for n in view.missing_instances(all_refs)
        ]


class SameTemplate(ConstraintEnvelope):
    """Force named instances to share an identical layout template.

    Attributes
    ----------
    constraint : str
        Type tag, always ``'SameTemplate'``.
    instances : list of str
        Names of the instances; at least two.
    severity : str
        ``'hard'`` or ``'soft'``; defaults to ``'soft'``.

    Raises
    ------
    ValueError
        On construction, if a field is invalid or unknown (pydantic's
        ``ValidationError``, a ``ValueError`` subclass).

    Examples
    --------
    >>> from silk.constraints import SameTemplate
    >>> SameTemplate(instances=['m1', 'm2']).severity
    'soft'
    """

    constraint: Literal["SameTemplate"] = "SameTemplate"
    instances: List[str] = Field(min_length=2)
    severity: Literal["hard", "soft"] = "soft"

    def validate_against(self, netlist) -> List[str]:
        """Check that every named instance exists in a netlist.

        Parameters
        ----------
        netlist : silk.circuit.Module
            The module whose instance names (the keys of its ``elem_dict``)
            are checked against; a pre-built name view is also accepted.

        Returns
        -------
        list of str
            One message per unknown instance, in the order of ``instances``;
            empty when all exist.
        """
        view = NetlistView.of(netlist)
        return [
            f"SameTemplate: unknown instance '{n}'"
            for n in view.missing_instances(self.instances)
        ]


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class GroupBlocks(ConstraintEnvelope):
    """Force the named instances into a generated sub-hierarchy.

    The optional ``generator`` dict says *how* the group is laid out
    (e.g. ``{"name": "MOS", "parameters": {"pattern": "cc"}}`` for a
    common-centroid MOS pair). Generator semantics are
    backend-defined — the schema only stores the dict.

    Attributes
    ----------
    constraint : str
        Type tag, always ``'GroupBlocks'``.
    name : str
        Name of the generated group.
    instances : list of str
        Names of the instances in the group; at least one.
    generator : dict, optional
        How the group is laid out; stored as given and not interpreted by
        silk. Defaults to ``None``.
    severity : str
        ``'hard'`` or ``'soft'``; defaults to ``'hard'``.

    Raises
    ------
    ValueError
        On construction, if a field is invalid or unknown (pydantic's
        ``ValidationError``, a ``ValueError`` subclass).

    Examples
    --------
    >>> from silk.constraints import GroupBlocks
    >>> GroupBlocks(name='diff', instances=['m1', 'm2']).generator is None
    True
    """

    constraint: Literal["GroupBlocks"] = "GroupBlocks"
    name: str
    instances: List[str] = Field(min_length=1)
    generator: Optional[Dict[str, Any]] = None
    severity: Literal["hard", "soft"] = "hard"

    def validate_against(self, netlist) -> List[str]:
        """Check that every named instance exists in a netlist.

        Parameters
        ----------
        netlist : silk.circuit.Module
            The module whose instance names (the keys of its ``elem_dict``)
            are checked against; a pre-built name view is also accepted.

        Returns
        -------
        list of str
            One message per unknown instance, in the order of ``instances``;
            empty when all exist.
        """
        view = NetlistView.of(netlist)
        return [
            f"GroupBlocks[{self.name}]: unknown instance '{n}'"
            for n in view.missing_instances(self.instances)
        ]


# ---------------------------------------------------------------------------
# Proximity (silk extension — no ALIGN equivalent)
# ---------------------------------------------------------------------------

class ProximityConstraint(ConstraintEnvelope):
    """Keep two device instances within ``max_distance_um`` of each other.

    Used for thermal-coupling pairs (matched-resistor heaters,
    bandgap-reference Q-pair) and for fast-loop matching where the
    placer must not drift instances apart.

    silk-only; ALIGN has no equivalent. It is serialized with this type name verbatim;
    consumers must support it or filter it before import.

    Attributes
    ----------
    constraint : str
        Type tag, always ``'ProximityConstraint'``.
    instance_a : str
        Name of the first instance (text, no unit).
    instance_b : str
        Name of the second instance; must differ from ``instance_a``.
    max_distance_um : float
        Largest allowed separation, in um (microns); must be greater than 0.
    severity : str
        ``'hard'`` or ``'soft'``; defaults to ``'soft'``.

    Raises
    ------
    ValueError
        On construction, if a field is invalid or unknown (pydantic's
        ``ValidationError``, a ``ValueError`` subclass).

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> from silk.constraints import ProximityConstraint
    >>> pair = ProximityConstraint(instance_a='q1', instance_b='q2',
    ...                            max_distance_um=20.0)
    >>> pair.validate_against(SimpleNamespace(elem_dict={'q1': 1}, net_dict={}))
    ["ProximityConstraint: unknown instance 'q2'"]
    """

    constraint: Literal["ProximityConstraint"] = "ProximityConstraint"
    instance_a: str
    instance_b: str
    max_distance_um: float = Field(gt=0)
    severity: Literal["hard", "soft"] = "soft"

    @model_validator(mode="after")
    def _check_distinct(self) -> "ProximityConstraint":
        if self.instance_a == self.instance_b:
            raise ValueError(
                "ProximityConstraint requires two distinct instances; "
                f"got '{self.instance_a}' twice"
            )
        return self

    def validate_against(self, netlist) -> List[str]:
        """Check that both named instances exist in a netlist.

        Parameters
        ----------
        netlist : silk.circuit.Module
            The module whose instance names (the keys of its ``elem_dict``)
            are checked against; a pre-built name view is also accepted.

        Returns
        -------
        list of str
            One message per unknown instance (``instance_a`` first); empty
            when both exist.
        """
        view = NetlistView.of(netlist)
        return [
            f"ProximityConstraint: unknown instance '{n}'"
            for n in view.missing_instances([self.instance_a, self.instance_b])
        ]


__all__ = [
    "SymmetricBlocks",
    "SameTemplate",
    "GroupBlocks",
    "ProximityConstraint",
]
