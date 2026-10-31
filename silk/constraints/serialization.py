"""silk-canonical JSON round-trip.

The on-disk shape is a flat array of objects, each keyed by
``"constraint"``. Three kinds of entries cohabit the array:

1. **Real constraints** — typed instances of the four
   device-relational types in :mod:`silk.constraints.types`. Carry
   the full audit envelope (id, source, …). One entry per constraint.
2. **PortRoles overlay** — exactly one entry per module, carrying a
   ``{port_name: role_string}`` dict that captures every port with
   a non-None ``PortRole``. Mirrors silk's "PortRole is one field per
   port" data model. An additive ``inferred`` dict lists, sparsely,
   every port whose role is a guess rather than a statement —
   ``ModulePin.role_inferred``. A legacy file has no
   ``inferred`` key at all; every role in it imports as declared.
3. **NetIntent overlay** — one entry per net that has any silk-native
   layout-intent field set (``max_ir_drop_mV``, ``max_j_factor``,
   ``max_length_um``, ``is_critical=True``, or ``domain``). Only the
   set fields are emitted.

These entries use a flat array with a ``"constraint"`` discriminator and
an audit envelope on each device-relational constraint. Port and net
entries preserve silk's native state. Consumers with a different schema
need an adapter for the overlays and any unsupported constraint types.
"""
from __future__ import annotations

import json as _json
import warnings
from typing import Annotated, Any, Iterable, List, Tuple, Union

from pydantic import Field as _PField
from pydantic import TypeAdapter

from silk._json_codec import decode_json, encode_json
from silk.constraints.envelope import ConstraintEnvelope
from silk.constraints.types import (
    GroupBlocks,
    ProximityConstraint,
    SameTemplate,
    SymmetricBlocks,
)


ConstraintUnion = Annotated[
    Union[SymmetricBlocks, SameTemplate, GroupBlocks, ProximityConstraint],
    _PField(discriminator="constraint"),
]

_LIST_ADAPTER = TypeAdapter(List[ConstraintUnion])


# Net-intent fields that emit if non-default. ``is_critical`` defaults
# to False (omit when False); the others default to None (omit when None).
_NET_SCALAR_INTENT = (
    "max_ir_drop_mV",
    "max_j_factor",
    "max_length_um",
)
_NET_FLAG_INTENT = ("is_critical",)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _role_str(role) -> str:
    """PortRole values are str-Enums; ``str(role)`` returns
    ``'PortRole.POWER'`` rather than ``'power'``. Use ``.value`` (or
    fall through if the role was already stored as a plain string)."""
    return getattr(role, "value", role)


def _emit_port_roles(mod) -> List[dict]:
    """Single PortRoles entry containing every port with a non-None role.

    ``inferred`` rides beside ``roles`` as a second, sparse dict: only
    ports whose ``pin.role_inferred`` is True appear in it (mirrors
    ``roles`` itself only listing non-None ones). Additive and
    defaulted — a consumer that predates this field ignores the key, and
    the map is omitted entirely when nothing was inferred, so the
    common case (an all-declared module) emits byte-identical JSON to
    the legacy format.
    """
    pin_list = getattr(mod, "pin_list", None) or {}
    roles = {
        name: _role_str(pin.role) for name, pin in pin_list.items()
        if getattr(pin, "role", None) is not None
    }
    if not roles:
        return []
    inferred = {
        name: True for name in roles
        if getattr(pin_list[name], "role_inferred", False)
    }
    entry = {"constraint": "PortRoles", "roles": roles}
    if inferred:
        entry["inferred"] = inferred
    return [entry]


def _emit_net_intent(mod) -> List[dict]:
    """One NetIntent entry per net with any silk-native intent set.

    Only set fields are emitted; nets with no intent are omitted.
    Field ordering in each entry: scalar budgets, then is_critical,
    then domain (matches likely-read order)."""
    net_dict = getattr(mod, "net_dict", None) or {}
    out: List[dict] = []
    for net_name, net in net_dict.items():
        entry: dict = {}
        for attr in _NET_SCALAR_INTENT:
            val = getattr(net, attr, None)
            if val is not None:
                entry[attr] = val
        for attr in _NET_FLAG_INTENT:
            val = getattr(net, attr, False)
            if val:
                entry[attr] = True
        dom = getattr(net, "domain", None)
        if dom is not None:
            entry["domain"] = {
                "name": dom.name,
                "nominal": dom.nominal,
                "tolerance": getattr(dom, "tolerance", None),
            }
            if getattr(dom, "switchable", False):
                entry["domain"]["switchable"] = True
        if entry:
            out.append({"constraint": "NetIntent", "net": net_name, **entry})
    return out


def to_obj(mod) -> List[dict]:
    """Serialize a Module to the silk-canonical list-of-dicts form."""
    real = [c.model_dump(mode="json") for c in mod.list_constraints()]
    return real + _emit_port_roles(mod) + _emit_net_intent(mod)


def to_json(mod, *, indent: int = 2) -> str:
    """Pretty-printed silk-canonical JSON."""
    return _json.dumps(encode_json(to_obj(mod)), allow_nan=False, indent=indent, default=str)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

_OVERLAY_TYPES = {"PortRoles", "NetIntent"}


def _split(entries: Iterable[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """Partition raw payload into (real_constraints, port_overlays,
    net_overlays)."""
    real, ports, nets = [], [], []
    for e in entries:
        kind = e.get("constraint")
        if kind == "PortRoles":
            ports.append(e)
        elif kind == "NetIntent":
            nets.append(e)
        else:
            real.append(e)
    return real, ports, nets


def _apply_port_roles(mod, entries: List[dict]) -> None:
    """Restore each port's role, and — additively — whether
    that role was a guess. A port named in the entry's ``inferred`` map
    comes back ``pin.role_inferred = True``; every other port, including
    every port in a legacy file (which never has this key), imports
    as declared (``role_inferred = False``) — the same value a role sets
    to on the module that never called ``infer_port_roles``. Without
    this, a role that was inferred before export returned as declared
    after import and silently defeated INT-004's guessed-POWER guard
    ."""
    pin_list = getattr(mod, "pin_list", None) or {}
    for entry in entries:
        inferred_map = entry.get("inferred", {})
        for port_name, role in entry.get("roles", {}).items():
            pin = pin_list.get(port_name)
            if pin is None:
                warnings.warn(
                    f"PortRoles: port '{port_name}' is not declared on the "
                    f"module — ignoring",
                    stacklevel=3,
                )
                continue
            pin.role = role
            pin.role_inferred = bool(inferred_map.get(port_name, False))


def _apply_net_intent(mod, entries: List[dict]) -> None:
    """Restore each NetIntent entry to its Net's silk-native fields,
    using ``assign_net_domain`` for domain so the module's domain
    registry stays consistent."""
    from silk.circuit.voltage_domain import VoltageDomain
    net_dict = getattr(mod, "net_dict", None) or {}
    for entry in entries:
        net_name = entry["net"]
        net = net_dict.get(net_name)
        if net is None:
            warnings.warn(
                f"NetIntent: net '{net_name}' is not declared on the "
                f"module — ignoring (call mod.add_net() first)",
                stacklevel=3,
            )
            continue
        for attr in _NET_SCALAR_INTENT:
            if attr in entry:
                setattr(net, attr, entry[attr])
        for attr in _NET_FLAG_INTENT:
            if attr in entry:
                setattr(net, attr, entry[attr])
        if "domain" in entry:
            d = entry["domain"]
            mod.assign_net_domain(
                net_name,
                VoltageDomain(name=d["name"], nominal=d["nominal"],
                              tolerance=d.get("tolerance"),
                              switchable=bool(d.get("switchable", False))),
            )


def from_obj(payload_or_list: Any) -> List[ConstraintEnvelope]:
    """Parse the *real* (typed) constraints from a payload.

    Overlay entries (PortRoles, NetIntent) need a module to apply to,
    so they are silently dropped here. Use :func:`from_json` for the
    in-place form that handles them.
    """
    if isinstance(payload_or_list, (str, bytes, bytearray)):
        payload_or_list = decode_json(_json.loads(payload_or_list))
    real, _, _ = _split(payload_or_list)
    return _LIST_ADAPTER.validate_python(real)


def from_json(mod, payload) -> List[ConstraintEnvelope]:
    """Parse JSON and apply: real constraints attach via
    ``add_constraint``; overlays write back to PortRole / Net fields.

    Returns the list of real constraints attached. Raises
    pydantic ``ValidationError`` on unknown discriminator (other than
    overlays) or malformed fields — nothing is attached if validation
    fails.
    """
    raw = decode_json(_json.loads(payload)) if isinstance(payload, (str, bytes, bytearray)) else payload
    real, ports, nets = _split(raw)
    constraints = _LIST_ADAPTER.validate_python(real)
    for c in constraints:
        mod.add_constraint(c)
    _apply_port_roles(mod, ports)
    _apply_net_intent(mod, nets)
    return constraints


__all__ = [
    "ConstraintUnion",
    "to_obj", "to_json", "from_obj", "from_json",
]
