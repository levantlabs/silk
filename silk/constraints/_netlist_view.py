"""Read-only view of the names a netlist exposes — used by validators.

Decouples constraint validation from the silk Module concrete shape so
tests can pass dict-like stubs and so constraints stay reusable from
external JSON loaders.

Only ``instances`` are referenced by the device-relational constraint
types in v1 (port semantics moved to ``PortRole``, net properties
moved to ``Net`` — see DESIGN.md ). ``nets`` is kept for use by the
serialization bridge, which walks them to serialize silk-native
JSON entries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class NetlistView:
    instances: frozenset
    nets: frozenset

    @classmethod
    def from_module(cls, mod) -> "NetlistView":
        instances = getattr(mod, "elem_dict", {}) or {}
        nets = getattr(mod, "net_dict", {}) or {}
        return cls(
            instances=frozenset(instances.keys()),
            nets=frozenset(nets.keys()),
        )

    @classmethod
    def of(cls, source) -> "NetlistView":
        if isinstance(source, cls):
            return source
        return cls.from_module(source)

    def has_instance(self, name: str) -> bool:
        return name in self.instances

    def missing_instances(self, names: Iterable[str]) -> list[str]:
        return [n for n in names if n not in self.instances]
