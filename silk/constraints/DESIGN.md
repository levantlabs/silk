# Layout intent in silk.constraints

`silk.constraints` describes relationships among device instances that a
layout consumer can use: symmetry (`SymmetricBlocks`), matching
(`SameTemplate`), grouping (`GroupBlocks`) and proximity
(`ProximityConstraint`). Each declaration carries a `ConstraintEnvelope`
with an identifier, source, timestamp, confidence, severity and rationale.
The package declares intent; checker implementations and placement belong
to consumers.

## Schema motivation and implementation

The open-source ALIGN analog layout project (BSD-3-Clause licence) provides
the schema that motivates the shared constraint type names and field shapes.
silk follows that schema so constraint files can be exchanged with it.
silk's implementation is its own, using Pydantic v2 models; the project is
not a runtime dependency.

The shared device-relational entries are `SymmetricBlocks`, `SameTemplate`
and `GroupBlocks`. `ProximityConstraint` has no equivalent in that schema.
Consumers must support this extension or filter it before import. silk's
port and net overlays also require adaptation for consumers that accept
only the shared schema; an entire silk JSON file is not automatically
compatible with such a consumer.

## Where intent belongs

A constraint belongs in the catalogue when it relates a set of devices
and no single device owns the relationship. Each constraint refers to
instances by name and stays independent of PDK constants. Consumers resolve
technology-specific information when checking or implementing the intent.

Port semantics belong on ports: `PortRole` and factories such as
`Port.supply()`, `Port.ground()` and `Port.clock_input()` express their roles.
Per-net properties belong on `Net`: `max_ir_drop_mV`, `max_j_factor`,
`max_length_um`, `is_critical` and `domain`. These fields remain the source
of truth for their respective objects rather than duplicate constraints.

## Module state and JSON snapshots

Importing `silk.constraints` attaches `Module.add_constraint()` and
`Module.list_constraints()` to `Module`. The installation is idempotent.
Declarations stored on the module are the source of truth; serialized
`*.const.json` files are snapshots for exchange and restoration.

`silk.constraints.serialization` uses silk's own JSON format: a flat array
of objects distinguished by a `"constraint"` key. It contains:

- Device-relational constraints, one object per declaration, with the audit
  envelope as flat fields on that object.
- A `PortRoles` overlay when ports have roles, holding a `{port: role}` map.
  An optional sparse `inferred` map preserves which roles were inferred.
- A `NetIntent` overlay for each net with intent set. Only set budgets,
  a true `is_critical` flag and an assigned domain are emitted.

`to_json(mod)` serializes this state. `from_json(mod, payload)` validates
the constraint declarations, attaches them through `add_constraint()` and
restores overlays to the corresponding ports and nets. Missing port or net
references in overlays produce warnings and are ignored. Domain restoration
uses `assign_net_domain()` so the module's domain registry stays consistent.
`from_obj(payload)` parses only device-relational constraints; overlays need
a module and are handled by `from_json()`.

An external consumer with a different schema needs an adapter for the
overlays and unsupported types. Such adapters should reflect the actual
consumer's requirements rather than change silk's native storage format.
