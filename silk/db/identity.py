"""Versioned, deterministic identities for live silk designs.

Structure v4 includes DUT-owned sources and retains v3's device parameters
as canonical PDK-native SPICE text, using emission's unit conversion and
rendering rules on extracted copies.
It retains v2's child port bindings at every depth. V1 omitted those bindings;
v1 and v2 both omitted parameter unit interpretation. Historical snapshots
and evidence stay untouched, but cannot authorize reuse or restoration into
v4 designs, even if digest bytes match. V3 also omitted sources. Rebuild the
live design in a new database or under a new circuit name and seal fresh evidence.

Fingerprint schemes are independent of the SQL schema version. Schema-v1 to
v2 copy migration preserves historical tags and evidence; it does not upgrade
their meaning. See docs/database.md for the preserve-and-rebuild procedure.
"""

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from silk._json_codec import encode_json
from silk.db.context import (
    RESERVED_PVT_PARAMETER_NAMES,
    CharacterizationContext,
)
from silk.db.models import Fingerprint, SnapshotIdentity
from silk.exceptions import (
    CharacterizationContextError,
    CircuitError,
    DuplicateNameError,
    EvidenceMismatchError,
    UnsupportedFingerprintSchemeError,
)


STRUCTURE_FINGERPRINT_SCHEME = "silk.structure/v4"
SPEC_FINGERPRINT_SCHEME = "silk.spec/v1"
# DB-02 (TASK-4): the characterization-profile identity scheme (design §3.1).
# Registering it here only makes ``_require_fingerprint_scheme`` accept the
# tag when a caller explicitly names ``PROFILE_FINGERPRINT_SCHEME`` as its
# ``expected_scheme`` — every comparison site still pins its own domain, so
# this can never let a profile digest compare equal against a structure or
# spec digest.
PROFILE_FINGERPRINT_SCHEME = "silk.profile/v1"
_SUPPORTED_FINGERPRINT_SCHEMES = frozenset(
    {
        STRUCTURE_FINGERPRINT_SCHEME,
        SPEC_FINGERPRINT_SCHEME,
        PROFILE_FINGERPRINT_SCHEME,
    }
)


@dataclass(frozen=True)
class _SpecDefinitionFingerprint:
    owner_path: str
    key: str
    fingerprint: Fingerprint
    parent_owner_path: Optional[str]
    parent_key: Optional[str]
    spec: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class _CanonicalCharacterizationProfile:
    """Canonical bytes persisted and hashed for one profile descriptor."""

    parameters_json: str
    payload_json: bytes
    fingerprint: Fingerprint


def _profile_parameter_error(
    path: str, detail: str
) -> CharacterizationContextError:
    return CharacterizationContextError(
        f"Characterization profile {path} {detail}"
    )
def _arbitrary_int_to_decimal(value: int) -> str:
    """Format an exact int without Python's process-wide digit limit."""
    if value == 0:
        return "0"
    sign = ""
    if value < 0:
        sign = "-"
        value = -value

    chunks = []
    while value:
        value, chunk = divmod(value, 1_000_000_000)
        chunks.append(chunk)
    most_significant = str(chunks.pop())
    remainder = "".join(
        f"{chunk:09d}" for chunk in reversed(chunks)
    )
    return sign + most_significant + remainder


def _decimal_to_arbitrary_int(value: str) -> int:
    """Parse canonical ASCII decimal without Python's digit limit."""
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if (
        not digits
        or (len(digits) > 1 and digits[0] == "0")
        or any(character < "0" or character > "9" for character in digits)
        or (negative and digits == "0")
    ):
        raise ValueError("not canonical decimal")

    first_width = len(digits) % 9 or 9
    result = int(digits[:first_width])
    for start in range(first_width, len(digits), 9):
        result = (
            result * 1_000_000_000
            + int(digits[start:start + 9])
        )
    return -result if negative else result




def _canonicalize_profile_parameter_value(
    value: Any,
    *,
    path: str,
    active_containers: set,
) -> Dict[str, Any]:
    """Convert one declared parameter to the closed v1 tagged-node form."""
    value_type = type(value)
    if value is None:
        return {"type": "none"}
    if value_type is bool:
        return {"type": "bool", "value": value}
    if value_type is int:
        return {"type": "int", "value": _arbitrary_int_to_decimal(value)}
    if value_type is float:
        if not math.isfinite(value):
            raise _profile_parameter_error(path, "must be a finite float")
        return {"type": "float", "value": value.hex()}
    if value_type is str:
        return {"type": "str", "value": value}

    if value_type not in (list, tuple, dict):
        raise _profile_parameter_error(
            path,
            "has unsupported type "
            f"{value_type.__name__}; expected None, bool, int, finite float, "
            "str, list/tuple, or exact dict",
        )

    container_id = id(value)
    if container_id in active_containers:
        raise _profile_parameter_error(path, "contains a recursive cycle")
    active_containers.add(container_id)
    try:
        if value_type in (list, tuple):
            return {
                "type": "sequence",
                "value": [
                    _canonicalize_profile_parameter_value(
                        item,
                        path=f"{path}[{index}]",
                        active_containers=active_containers,
                    )
                    for index, item in enumerate(value)
                ],
            }

        entries = []
        for key in value:
            if type(key) is not str:
                raise _profile_parameter_error(
                    path,
                    f"mapping keys must be exact strings; got {key!r}",
                )
        for key in sorted(value):
            entries.append(
                {
                    "name": key,
                    "value": _canonicalize_profile_parameter_value(
                        value[key],
                        path=f"{path}.{key}",
                        active_containers=active_containers,
                    ),
                }
            )
        return {"type": "mapping", "value": entries}
    finally:
        active_containers.remove(container_id)


def _canonical_profile_json(value: Any) -> str:
    return json.dumps(
        encode_json(value, finite_only=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonicalize_characterization_profile(
    context: CharacterizationContext,
) -> _CanonicalCharacterizationProfile:
    """Derive silk.profile/v1 identity from owner, test, and parameters.

    Process corner, temperature, and voltage deliberately do not enter these
    bytes; they select evidence beneath the stable profile identity.
    """
    if not isinstance(context, CharacterizationContext):
        raise TypeError(
            "context must be a CharacterizationContext, got "
            f"{type(context).__name__}"
        )

    parameter_entries = []
    for key in context.parameters:
        if type(key) is not str or not key:
            raise _profile_parameter_error(
                "parameters", "keys must be non-empty exact strings"
            )
        if key in RESERVED_PVT_PARAMETER_NAMES:
            raise _profile_parameter_error(
                f"parameters.{key}", "uses a reserved PVT field name"
            )
    for key in sorted(context.parameters):
        parameter_entries.append(
            {
                "name": key,
                "value": _canonicalize_profile_parameter_value(
                    context.parameters[key],
                    path=f"parameters.{key}",
                    active_containers=set(),
                ),
            }
        )

    parameters_json = _canonical_profile_json(parameter_entries)
    payload_json, fingerprint = _fingerprint_characterization_profile_fields(
        context.owner_path,
        context.test_name,
        parameter_entries,
    )
    return _CanonicalCharacterizationProfile(
        parameters_json=parameters_json,
        payload_json=payload_json,
        fingerprint=fingerprint,
    )


def _stored_profile_error(detail: str) -> EvidenceMismatchError:
    return EvidenceMismatchError(
        f"Stored characterization profile {detail}"
    )


def _decode_profile_parameter_node(node: Any, *, path: str) -> Any:
    if type(node) is not dict or type(node.get("type")) is not str:
        raise _stored_profile_error(
            f"parameters_json has an invalid tagged node at {path}"
        )

    tag = node["type"]
    if tag == "none":
        if set(node) != {"type"}:
            raise _stored_profile_error(
                f"parameters_json has an invalid none node at {path}"
            )
        return None

    if set(node) != {"type", "value"}:
        raise _stored_profile_error(
            f"parameters_json has invalid fields at {path}"
        )
    raw_value = node["value"]

    if tag == "bool":
        if type(raw_value) is not bool:
            raise _stored_profile_error(
                f"parameters_json has an invalid bool at {path}"
            )
        return raw_value
    if tag == "int":
        if type(raw_value) is not str:
            raise _stored_profile_error(
                f"parameters_json has an invalid int at {path}"
            )
        try:
            value = _decimal_to_arbitrary_int(raw_value)
        except ValueError as exc:
            raise _stored_profile_error(
                f"parameters_json has an invalid int at {path}"
            ) from exc
        if _arbitrary_int_to_decimal(value) != raw_value:
            raise _stored_profile_error(
                f"parameters_json has a non-canonical int at {path}"
            )
        return value
    if tag == "float":
        if type(raw_value) is not str:
            raise _stored_profile_error(
                f"parameters_json has an invalid float at {path}"
            )
        try:
            value = float.fromhex(raw_value)
        except ValueError as exc:
            raise _stored_profile_error(
                f"parameters_json has an invalid float at {path}"
            ) from exc
        if not math.isfinite(value) or value.hex() != raw_value:
            raise _stored_profile_error(
                f"parameters_json has a non-canonical float at {path}"
            )
        return value
    if tag == "str":
        if type(raw_value) is not str:
            raise _stored_profile_error(
                f"parameters_json has an invalid str at {path}"
            )
        return raw_value
    if tag == "sequence":
        if type(raw_value) is not list:
            raise _stored_profile_error(
                f"parameters_json has an invalid sequence at {path}"
            )
        return tuple(
            _decode_profile_parameter_node(item, path=f"{path}[{index}]")
            for index, item in enumerate(raw_value)
        )
    if tag == "mapping":
        return _decode_profile_parameter_entries(
            raw_value,
            path=path,
            top_level=False,
        )
    raise _stored_profile_error(
        f"parameters_json has unknown type tag {tag!r} at {path}"
    )


def _decode_profile_parameter_entries(
    entries: Any,
    *,
    path: str,
    top_level: bool,
) -> Mapping[str, Any]:
    if type(entries) is not list:
        raise _stored_profile_error(
            f"parameters_json must encode {path} as a list"
        )

    decoded = {}
    previous_name = None
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"name", "value"}:
            raise _stored_profile_error(
                f"parameters_json has an invalid entry at {path}"
            )
        name = entry["name"]
        if type(name) is not str or (top_level and not name):
            raise _stored_profile_error(
                f"parameters_json has an invalid name at {path}"
            )
        if top_level and name in RESERVED_PVT_PARAMETER_NAMES:
            raise _stored_profile_error(
                f"parameters_json uses reserved PVT name {name!r}"
            )
        if previous_name is not None and name <= previous_name:
            raise _stored_profile_error(
                "parameters_json names are not in canonical order"
            )
        previous_name = name
        decoded[name] = _decode_profile_parameter_node(
            entry["value"], path=f"{path}.{name}"
        )
    return MappingProxyType(decoded)


def _decode_characterization_profile_parameters(
    parameters_json: str,
) -> Tuple[Mapping[str, Any], Any]:
    if type(parameters_json) is not str:
        raise _stored_profile_error("parameters_json is not text")
    try:
        entries = json.loads(parameters_json)
        canonical_json = _canonical_profile_json(entries)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _stored_profile_error(
            "parameters_json is not valid canonical JSON"
        ) from exc
    if canonical_json != parameters_json:
        raise _stored_profile_error("parameters_json is not canonical")
    parameters = _decode_profile_parameter_entries(
        entries,
        path="parameters",
        top_level=True,
    )
    return parameters, entries


def _fingerprint_characterization_profile_fields(
    owner_path: str,
    test_name: str,
    parameter_entries: Any,
) -> Tuple[bytes, Fingerprint]:
    payload = {
        "owner_path": owner_path,
        "parameters": parameter_entries,
        "test_name": test_name,
    }
    payload_json = _canonical_profile_json(payload).encode("utf-8")
    digest = hashlib.sha256(
        b"silk.profile/v1\0profile\0" + payload_json
    ).hexdigest()
    return (
        payload_json,
        Fingerprint(PROFILE_FINGERPRINT_SCHEME, digest),
    )


def _canonical_sources(sources):
    """Validate source facts before the legacy serializers' default=str.

    Source facts are resolved JSON, never Variables, nonfinite tags or opaque
    object reprs. Keep the older device canonicalizers unchanged.
    """
    try:
        entries = sorted(sources, key=lambda item: item['name'])
        json.dumps(encode_json(entries, finite_only=True), allow_nan=False)
        return entries
    except (TypeError, ValueError, KeyError, RecursionError) as exc:
        raise CircuitError(f'Cannot fingerprint source facts: {exc}') from exc


def _fingerprint_module_structure(
    module_structure: Mapping[str, Any],
) -> Fingerprint:
    """Hash a v4 definition, retaining source-free canonical bytes."""
    def read_field(name, default, normalize):
        try:
            return normalize(module_structure.get(name, default))
        except (TypeError, AttributeError) as exc:
            raise CircuitError(
                f"Malformed module structure field {name!r}: {exc}"
            ) from exc

    sorted_devices = read_field(
        "devices", [], lambda values: sorted(values, key=lambda item: item.get("name", "")),
    )
    sorted_nets = read_field("nets", [], sorted)
    sorted_submodules = read_field(
        "submodules", {}, lambda values: dict(sorted(values.items())),
    )
    sorted_passives = read_field(
        "passives", [], lambda values: sorted(values, key=lambda item: item.get("name", "")),
    )
    payload = {
        "devices": sorted_devices,
        "nets": sorted_nets,
        "submodules": sorted_submodules,
        "passives": sorted_passives,
    }
    if module_structure.get('sources'):
        payload['sources'] = _canonical_sources(module_structure['sources'])
    canonical = json.dumps(
        encode_json(payload), allow_nan=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, digest)


def _fingerprint_structure(
    structure: Mapping[str, Any],
) -> Tuple[Fingerprint, Dict[str, Fingerprint]]:
    """Hash v4 structure, retaining source-free canonical bytes."""
    sorted_structure = copy.deepcopy(structure)

    if "transistors" in sorted_structure:
        sorted_structure["transistors"] = sorted(
            sorted_structure["transistors"],
            key=lambda item: item.get("name", ""),
        )
    if "nets" in sorted_structure:
        sorted_structure["nets"] = sorted(
            sorted_structure["nets"],
            key=lambda item: item.get("name", ""),
        )
    if "hierarchy" in sorted_structure:
        sorted_structure["hierarchy"] = sorted(
            sorted_structure["hierarchy"],
            key=lambda item: item.get("instance_name", ""),
        )
    if "parameters" in sorted_structure:
        sorted_structure["parameters"] = sorted(
            sorted_structure["parameters"],
            key=lambda item: (item.get("module", ""), item.get("name", "")),
        )
    if "passives" in sorted_structure:
        sorted_structure["passives"] = sorted(
            sorted_structure["passives"],
            key=lambda item: item.get("name", ""),
        )

    canonical_transistors = json.dumps(
        encode_json(sorted_structure.get("transistors", [])), allow_nan=False, sort_keys=True, default=str
    )
    canonical_nets = json.dumps(
        encode_json(sorted_structure.get("nets", [])), allow_nan=False, sort_keys=True, default=str
    )
    canonical_hierarchy = json.dumps(
        encode_json(sorted_structure.get("hierarchy", [])), allow_nan=False, sort_keys=True, default=str
    )
    canonical_passives = json.dumps(
        encode_json(sorted_structure.get("passives", [])), allow_nan=False, sort_keys=True, default=str
    )
    module_hashes = {
        path: data["hash"]
        for path, data in sorted_structure.get("modules", {}).items()
    }
    canonical_modules = json.dumps(encode_json(module_hashes), allow_nan=False, sort_keys=True)
    combined_data = (
        canonical_transistors
        + canonical_nets
        + canonical_hierarchy
        + canonical_passives
        + canonical_modules
    )
    if sorted_structure.get('sources'):
        combined_data += json.dumps(
            encode_json(_canonical_sources(sorted_structure['sources'])),
            allow_nan=False, sort_keys=True,
        )
    digest = hashlib.sha256(combined_data.encode()).hexdigest()
    module_fingerprints = {
        path: Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, module_digest)
        for path, module_digest in module_hashes.items()
    }
    return (
        Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, digest),
        module_fingerprints,
    )


def _escape_owner_path_segment(segment: str) -> str:
    if not isinstance(segment, str):
        raise TypeError(
            f"Module instance names must be strings, got {type(segment).__name__}"
        )
    return segment.replace("~", "~0").replace("/", "~1")


def _iter_spec_owners(module: Any, owner_path: str = "$") -> Iterator[Tuple[str, Any]]:
    yield owner_path, module
    manager = getattr(module, "_element_mgr", None)
    elem_dict = getattr(manager, "elem_dict", {}) or {}
    for instance_name, entry in elem_dict.items():
        child = entry.get("device") if isinstance(entry, dict) else entry
        if child is None:
            continue
        if not hasattr(child, "_specs") or not hasattr(child, "_element_mgr"):
            continue
        child_path = f"{owner_path}/{_escape_owner_path_segment(instance_name)}"
        yield from _iter_spec_owners(child, child_path)


def _validated_spec_key(spec: Any, owner_path: str) -> str:
    key = getattr(spec, "key", None)
    if not isinstance(key, str):
        raise TypeError(
            f"Spec at {owner_path} has a non-string key: {type(key).__name__}"
        )
    if not key.strip():
        raise ValueError(
            f"Spec at {owner_path} has an empty or whitespace-only key"
        )
    return key


def _canonical_spec_json(payload: Any) -> bytes:
    return json.dumps(
        encode_json(payload), allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _fingerprint_spec_definition_fields(
    *,
    owner_path: str,
    key: str,
    metric: str,
    op: str,
    target: Real,
    units: str,
    parent_owner_path: Optional[str],
    parent_key: Optional[str],
) -> Fingerprint:
    """Fingerprint one declaration from its identity-bearing fields."""
    if (parent_owner_path is None) != (parent_key is None):
        raise ValueError(
            "A spec parent requires both owner path and key, or neither"
        )
    parent_payload = (
        None
        if parent_owner_path is None
        else {"owner_path": parent_owner_path, "key": parent_key}
    )
    payload = {
        "owner_path": owner_path,
        "key": key,
        "metric": metric,
        "op": op,
        "target": float(target).hex(),
        "units": units,
        "parent": parent_payload,
    }
    definition_bytes = (
        b"silk.spec/v1\0definition\0" + _canonical_spec_json(payload)
    )
    return Fingerprint(
        SPEC_FINGERPRINT_SCHEME,
        hashlib.sha256(definition_bytes).hexdigest(),
    )


def _fingerprint_spec_set_records(
    records: Iterable[Tuple[str, str, Fingerprint]],
) -> Fingerprint:
    """Fingerprint persisted ``(owner_path, key, definition)`` records."""
    normalized = []
    seen = set()
    for owner_path, key, definition in records:
        _require_fingerprint_scheme(definition, SPEC_FINGERPRINT_SCHEME)
        location = (owner_path, key)
        if location in seen:
            raise DuplicateNameError(
                f"Spec key {key!r} is duplicated at owner path {owner_path!r}"
            )
        seen.add(location)
        normalized.append(
            {
                "owner_path": owner_path,
                "key": key,
                "definition_hash": definition.digest,
            }
        )
    normalized.sort(key=lambda item: (item["owner_path"], item["key"]))
    # This is the permanent domain separator for the v1 set canonicalizer.
    set_bytes = b"silk.spec/v1\0set\0" + _canonical_spec_json(normalized)
    return Fingerprint(
        SPEC_FINGERPRINT_SCHEME,
        hashlib.sha256(set_bytes).hexdigest(),
    )


def _fingerprint_spec_set(
    module: Any,
) -> Tuple[Fingerprint, Tuple[_SpecDefinitionFingerprint, ...]]:
    """Fingerprint every live spec declaration and their deterministic set."""
    collected = []
    locations_by_object = {}
    owners_by_object = {}

    # Pass 1: assign every live Spec an owner path/key and reject ambiguity.
    for owner_path, owner in _iter_spec_owners(module):
        local_keys = set()
        for spec in getattr(owner, "_specs", ()) or ():
            key = _validated_spec_key(spec, owner_path)
            if id(spec) in owners_by_object and owners_by_object[id(spec)] is not owner:
                raise CircuitError('The same Spec object belongs to two distinct modules.')
            owners_by_object[id(spec)] = owner
            if key in local_keys:
                raise DuplicateNameError(
                    f"Spec key {key!r} is duplicated at owner path {owner_path!r}"
                )
            local_keys.add(key)
            locations_by_object.setdefault(id(spec), []).append((owner_path, key))
            collected.append((owner_path, key, spec))

    # Pass 2: parent identities are now resolvable regardless of walk order.
    definitions = []
    for owner_path, key, spec in collected:
        parent = getattr(spec, "parent", None)
        if parent is None:
            parent_owner_path = None
            parent_key = None
        else:
            parent_locations = locations_by_object.get(id(parent))
            if parent_locations is None:
                raise CircuitError(
                    f"Spec {owner_path}/{key} has a parent outside the live "
                    "instance tree"
                )
            # A shared definition contributes specs at every placement. Its
            # parent spec resolves in the same instance subtree, using the
            # longest common owner prefix; never pick a sibling arbitrarily.
            def proximity(location):
                count = 0
                for left, right in zip(owner_path.split('/'), location[0].split('/')):
                    if left != right:
                        break
                    count += 1
                return count
            closest = max(proximity(location) for location in parent_locations)
            candidates = [location for location in parent_locations
                          if proximity(location) == closest]
            if len(candidates) != 1:
                raise CircuitError(
                    f"Spec {owner_path}/{key} has an ambiguous parent across "
                    f"instances: {candidates}")
            parent_owner_path, parent_key = candidates[0]

        definition_fingerprint = _fingerprint_spec_definition_fields(
            owner_path=owner_path,
            key=key,
            metric=spec.metric,
            op=spec.op,
            target=spec.target,
            units=spec.units,
            parent_owner_path=parent_owner_path,
            parent_key=parent_key,
        )
        definitions.append(
            _SpecDefinitionFingerprint(
                owner_path=owner_path,
                key=key,
                fingerprint=definition_fingerprint,
                parent_owner_path=parent_owner_path,
                parent_key=parent_key,
                spec=spec,
            )
        )

    definitions.sort(key=lambda item: (item.owner_path, item.key))
    set_fingerprint = _fingerprint_spec_set_records(
        (
            definition.owner_path,
            definition.key,
            definition.fingerprint,
        )
        for definition in definitions
    )
    return set_fingerprint, tuple(definitions)


def _require_fingerprint_scheme(
    fingerprint: Fingerprint,
    expected_scheme: str,
) -> None:
    if expected_scheme not in _SUPPORTED_FINGERPRINT_SCHEMES:
        raise UnsupportedFingerprintSchemeError(
            f"Unsupported expected fingerprint scheme {expected_scheme!r}"
        )
    if fingerprint.scheme != expected_scheme:
        if (fingerprint.scheme in ("silk.structure/v1", "silk.structure/v2", "silk.structure/v3")
                and expected_scheme == STRUCTURE_FINGERPRINT_SCHEME):
            raise UnsupportedFingerprintSchemeError(
                f"Unsupported fingerprint scheme {fingerprint.scheme!r}; expected "
                f"{STRUCTURE_FINGERPRINT_SCHEME!r}. V1 omitted child port "
                "bindings; v1/v2 omitted parameter unit interpretation; "
                "v1/v2/v3 omitted DUT-owned sources. "
                "Preserve the stored snapshot; rebuild the live Python design "
                "in a new database or under a new circuit name and seal fresh "
                "v4 evidence. Do not relabel or reuse v1/v2/v3 evidence. "
                "See docs/database.md."
            )
        raise UnsupportedFingerprintSchemeError(
            f"Unsupported fingerprint scheme {fingerprint.scheme!r}; "
            f"expected {expected_scheme!r}"
        )


def _fingerprints_equal(
    left: Fingerprint,
    right: Fingerprint,
    *,
    expected_scheme: str,
) -> bool:
    """Compare digests only after both scheme tags are accepted."""
    _require_fingerprint_scheme(left, expected_scheme)
    _require_fingerprint_scheme(right, expected_scheme)
    return left.digest == right.digest


__all__ = [
    "Fingerprint",
    "SnapshotIdentity",
    "STRUCTURE_FINGERPRINT_SCHEME",
    "SPEC_FINGERPRINT_SCHEME",
    "PROFILE_FINGERPRINT_SCHEME",
]
