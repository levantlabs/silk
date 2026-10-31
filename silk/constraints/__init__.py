"""Device-relational layout intent with a provenance envelope.

Importing this package is a side-effect import: it attaches
``Module.add_constraint(constraint)`` and ``Module.list_constraints()`` to
:class:`silk.circuit.Module`. Without the import those methods do not exist.
``add_constraint`` stores and returns a :class:`ConstraintEnvelope` (it
raises ``TypeError`` for anything else) and ``list_constraints`` returns a
copy of the stored list.

What's in scope
---------------
Designer intent that talks about a *set* of devices and therefore
needs a typed, module-scope container. The four types are
:class:`SymmetricBlocks`, :class:`SameTemplate`, :class:`GroupBlocks`, and
:class:`ProximityConstraint`; each carries the audit fields of
:class:`ConstraintEnvelope`.

What's *not* here (and why)
---------------------------
* **Port semantics** (power / ground / clock) live on
  :class:`silk.circuit.PortRole` and the convenience
  factories ``Port.supply()`` / ``Port.ground()`` / ``Port.clock_input()``.
* **Per-net budgets** (IR drop, current density, max length) live as
  ``max_ir_drop_mV`` / ``max_j_factor`` / ``max_length_um`` fields on
  :class:`silk.circuit.net.Net`.

The constraint types follow the schema of the open-source ALIGN analog layout
project (BSD-3-Clause licence), so constraint files can be exchanged with it.
silk's implementation is its own. The shared device-relational entries use the
same type names and field shapes; silk's JSON also includes ``PortRoles`` and
``NetIntent`` overlays and ``ProximityConstraint``, which need adaptation for
consumers that accept only that schema. See ``DESIGN.md`` for the format.

Importing this package monkey-patches ``Module.add_constraint`` and
``Module.list_constraints`` onto :class:`silk.circuit.modules.Module`.
The patch is idempotent.
"""
from silk.constraints.envelope import ConstraintEnvelope
from silk.constraints.types import (
    GroupBlocks,
    ProximityConstraint,
    SameTemplate,
    SymmetricBlocks,
)
from silk.constraints._module_patch import install as _install

_install()

#: Tuple of every concrete constraint class (``SymmetricBlocks``,
#: ``SameTemplate``, ``GroupBlocks``, ``ProximityConstraint``), used by
#: serialization to build a constraint from its ``constraint`` tag.
ALL_CONSTRAINT_TYPES = (
    SymmetricBlocks,
    SameTemplate,
    GroupBlocks,
    ProximityConstraint,
)

__all__ = [
    "ConstraintEnvelope",
    "SymmetricBlocks", "SameTemplate", "GroupBlocks", "ProximityConstraint",
    "ALL_CONSTRAINT_TYPES",
]
