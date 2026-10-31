"""silk.layout.matching — HalfCircuit declaration and net-equivalence classes.

Phase 1 public API:
    HalfCircuit    — frozen dataclass declaring a matched device pair/group
    LayoutInferenceError — raised for unrecoverable structural inconsistencies

build_equivalence_classes and compute_canonical_inputs are defined in this
module and used by silk.layout.estimator. They are not part of the public API.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from silk.layout._types import LayoutWarning

if TYPE_CHECKING:
    from silk.circuit.modules import Module
    from silk.circuit.net import Net
    from silk.layout.routing_model import RoutingModel


class LayoutInferenceError(Exception):
    """Raised when matching declarations produce an unrecoverable structural
    inconsistency that the estimator cannot work around.

    Examples: a declared HalfCircuit member has no port entries in elem_dict
    at estimation time, making mirror-net inference impossible.
    """


@dataclass(frozen=True)
class HalfCircuit:
    """Declaration that two ordered lists of devices are symmetric mirrors.

    Positional pairing: ``left[i]`` and ``right[i]`` are a matched pair.
    The estimator uses this to infer mirror-net pairs and symmetrize
    parasitic estimates across the matched branches — ensuring that
    equivalent nets on both sides get identical layout scalars.

    Parameters
    ----------
    name : str
        Identifier for this half-circuit (unique within the parent module).
    left : tuple of str
        Instance names of devices on the left branch, in order.
    right : tuple of str
        Instance names of devices on the right branch, in order.
        Must have the same length as ``left``.
    strategy : str, optional
        Reserved for future use: ``'common_centroid'``, ``'interdigitated'``, …
        Ignored in Phase 1.
    """
    name: str
    left: Tuple[str, ...]
    right: Tuple[str, ...]
    strategy: Optional[str] = None

    def __post_init__(self) -> None:
        # Coerce lists → tuples (frozen dataclass forbids direct mutation,
        # but we can use object.__setattr__ during __post_init__).
        object.__setattr__(self, 'left', tuple(self.left))
        object.__setattr__(self, 'right', tuple(self.right))

        if len(self.left) == 0:
            raise ValueError(
                f"HalfCircuit {self.name!r}: empty sides not allowed"
            )
        if len(self.left) != len(self.right):
            raise ValueError(
                f"HalfCircuit {self.name!r}: left has {len(self.left)} members, "
                f"right has {len(self.right)} — positional pairing requires equal lengths"
            )
        overlap = set(self.left) & set(self.right)
        if overlap:
            raise ValueError(
                f"HalfCircuit {self.name!r}: members {sorted(overlap)} appear on both sides"
            )


# ---------------------------------------------------------------------------
# Internal union-find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Path-compressed union-find for net-name strings."""

    def __init__(self, elements):
        self._parent = {e: e for e in elements}

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[rx] = ry

    def classes(self) -> Dict[str, List[str]]:
        """Return {root: [members]} mapping."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for e in self._parent:
            groups[self.find(e)].append(e)
        return dict(groups)


# ---------------------------------------------------------------------------
# Device-type-agnostic pin accessors
# ---------------------------------------------------------------------------

def _dev_pin_names(dev) -> set:
    """Return the set of pin names for a PDKDevice or AnalogElement."""
    if hasattr(dev, 'pin_list'):      # PDKDevice
        return set(dev.pin_list.keys())
    return set(getattr(dev, 'nets', {}).keys())


def _dev_pin_net(dev, pin_name: str):
    """Return the net connected to pin_name, or None if unconnected."""
    if hasattr(dev, 'pin_list'):      # PDKDevice
        pin = dev.pin_list.get(pin_name)
        return pin.net if pin else None
    return getattr(dev, 'nets', {}).get(pin_name)


# ---------------------------------------------------------------------------
# Equivalence-class construction
# ---------------------------------------------------------------------------

def build_equivalence_classes(
    module,
    warnings: list,
) -> Dict[str, List[str]]:
    """Build net-equivalence classes from module.half_circuits.

    Uses union-find over all nets in ``module.net_dict``.  Every net appears
    in exactly one class.  Mirror nets are inferred from positional pairing:
    for each HalfCircuit, left[i] and right[i] are matched, so ports with
    the same name on both devices contribute a union of their connected nets.

    Shared nets (same net on both sides of a pair) are not unioned — they
    are self-mirrors and already in the same class.

    Classes with more than two members emit a
    ``LayoutWarning(reason='matching_conflict_large_class')``.

    Parameters
    ----------
    module
        The module being estimated.
    warnings : list
        Mutable list; warnings are appended in-place.

    Returns
    -------
    dict
        ``{class_id: [net_name, ...]}`` — every net in ``module.net_dict``
        appears in exactly one list.  ``class_id`` is the union-find root
        (a net name string; treat as opaque).
    """
    uf = _UnionFind(module.net_dict.keys())

    for hc in module.half_circuits.values():
        for i in range(len(hc.left)):
            left_entry  = module.elem_dict.get(hc.left[i])
            right_entry = module.elem_dict.get(hc.right[i])
            if left_entry is None or right_entry is None:
                continue
            left_dev  = left_entry['device']
            right_dev = right_entry['device']

            left_ports  = _dev_pin_names(left_dev)
            right_ports = _dev_pin_names(right_dev)
            common      = left_ports & right_ports

            for port in common:
                left_net_name  = _dev_pin_net(left_dev, port)
                right_net_name = _dev_pin_net(right_dev, port)
                if (left_net_name and right_net_name
                        and left_net_name != right_net_name
                        and left_net_name in module.net_dict
                        and right_net_name in module.net_dict):
                    uf.union(left_net_name, right_net_name)

    classes = uf.classes()

    # Emit warnings for large classes (size > 2)
    for root, members in classes.items():
        if len(members) > 2:
            warnings.append(LayoutWarning(
                net_name=None,
                reason='matching_conflict_large_class',
                severity='warn',
                detail=(
                    f"Equivalence class has {len(members)} members: "
                    f"{sorted(members)[:8]}{'…' if len(members) > 8 else ''}. "
                    "Caused by overlapping HalfCircuit declarations sharing a device. "
                    "Estimator proceeds with max-across-members inputs."
                ),
            ))

    return classes


# ---------------------------------------------------------------------------
# Canonical inputs for one equivalence class
# ---------------------------------------------------------------------------

@dataclass
class _CanonicalInputs:
    """Aggregated inputs for one equivalence class (max across members)."""
    peak_current: Optional[float]   # None if all members have no current
    fanout: int
    classification: Optional[object]
    length: float   # statistical length in um (computed from wirelength model)


def compute_canonical_inputs(
    members: list,  # list of Net objects
    module,
    warnings: list,
) -> _CanonicalInputs:
    """Compute max-across-members canonical inputs for one equivalence class.

    For current: max of non-None values; None if all are None.
    For fanout: max across members.
    For classification: most common non-None value (majority vote); None if all None.
    For length: max of ``estimate_net_length`` called on each member with the
    canonical classification.

    Emits one ``no_current_data`` warning per member with no current data.
    """
    from silk.layout.wirelength_model import estimate_net_length

    # Peak current: max of non-None values
    currents = [n.peak_current for n in members if n.peak_current is not None]
    canonical_current = max(currents) if currents else None
    if canonical_current is None:
        for net in members:
            warnings.append(LayoutWarning(
                net_name=net.name,
                reason='no_current_data',
                severity='info',
                detail=(
                    f"Net '{net.name}' has no peak_current — "
                    "falling back to min_width floor. "
                    "Populate Net.peak_current (via SPICE or manual annotation) "
                    "for current-aware estimates."
                ),
            ))

    # Fanout: max across members
    canonical_fanout = max(n.fanout for n in members)

    # Classification: majority vote
    roles = [n.classification for n in members if n.classification is not None]
    canonical_classification = Counter(roles).most_common(1)[0][0] if roles else None

    # Length: max across members using canonical classification
    canonical_length = max(
        estimate_net_length(n, module, canonical_classification)
        for n in members
    )

    return _CanonicalInputs(
        peak_current=canonical_current,
        fanout=canonical_fanout,
        classification=canonical_classification,
        length=canonical_length,
    )
