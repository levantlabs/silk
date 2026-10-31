"""Unportable ground-port interfaces and their shared predicate.

``ngspice-42`` folds the *node name* ``gnd`` (case-insensitively) onto the
global SPICE node ``0``; Xyce 7.10 folds nothing, at any spelling.
Measured 2026-09-04 with one byte-identical deck through both simulators:

* folded by ngspice: ``gnd``, ``GND``, ``Gnd``;
* NOT folded by ngspice: ``ground``, ``GROUND``, ``vss``, ``VSS``, ``gnd!``,
  ``agnd``.

For an *internal* net, ngspice merges it with ground while Xyce leaves it
as a separate node. That can silently produce wrong results.
Silk therefore rejects nets named gnd in Xyce decks.
Authored gnd! names are separately refused by the name validator.

For a subcircuit **port** it is silent and wrong. Node ``0`` is global in
SPICE and pierces subcircuit boundaries, so folding the port *name* discards
whatever the parent wired to that port position::

    .subckt blk a gnd / R1 a gnd 1k / .ends
    Xblk top mid blk        <- parent wires the gnd port to 'mid', not ground
    R2 mid 0 1k
    ngspice-42 :  I(V1) = -1.00 mA  V(mid) = 0.0   <- port ignored, tied to 0
    Xyce 7.10  :  I(V1) = -0.50 mA  V(mid) = 0.5   <- port honoured

Both runs "succeed" and neither warns. Two different circuits, no diagnostic.

The hazard therefore lives in the block's **interface**, not in any one
wiring of it: a reusable block is instantiated many times, possibly by code
its author does not control and possibly in another design entirely, so a
check keyed on the instantiation sites present in the current netlist could
never protect the block. The trigger here is consequently the *declaration* —
a port named ``gnd`` is rejected regardless of how it is or will be wired.
The formal name ``0`` has the same defect on every SPICE backend: it is
global ground itself, independently of any simulator's alias folding.

Declaration, lint, emission and static setup use the message built here:

* ``STRUCT-012`` (``silk.lint.rules.structural``) — lint time, no
  simulation needed, waivable like any other rule;
* ``define_module_ports`` refuses formal ``0`` with ``ValueError``;
  ``gnd`` retains its lint/emission-time refusal;
* defensive emission (including pure deck rendering) and static admission
  for SPICE and behavioral runs refuse ``0`` and case-insensitive ``gnd``
  with ``CircuitError``. The read-only setup check uses that same guard.
  Behavioral modeled-leaf interfaces are checked, but their replaced
  contents are not entered.

Deliberately NOT done here: canonicalising ``gnd`` to ``0`` at emission. Node
``0`` is global, so folding ``gnd`` inside a ``.subckt`` would tie that block's
ground to global ground even when the parent wired the port elsewhere —
replicating ngspice's bug into the backends that currently get it right, and
making a legitimately isolated-ground block unrepresentable.
"""

# The exact set ngspice folds, casefolded. One entry today; a set because the
# alias table is a measured property of a simulator, not a constant of SPICE.
FOLDED_GROUND_PORT_NAMES = frozenset({'gnd'})

# Verified NOT folded by ngspice-42, so it is a safe thing to suggest.
SAFE_GROUND_PORT_NAME = 'vss'


def _is_global_ground_port(port_name) -> bool:
    """Whether a formal name denotes SPICE's backend-independent global ground."""
    return isinstance(port_name, str) and port_name == '0'


def is_unportable_ground_port(port_name) -> bool:
    """True for global ``0`` or a formal name folded to it by ngspice.

    Case-insensitive over :data:`FOLDED_GROUND_PORT_NAMES`, so ``gnd``,
    ``GND`` and ``Gnd`` all match. ``gnd!``, ``agnd``, ``ground``, ``GROUND``,
    ``vss`` and ``VSS`` do not: ngspice leaves them alone and they are
    legitimately distinct nets (a negative supply, a quiet analog ground).
    """
    return (_is_global_ground_port(port_name)
            or (isinstance(port_name, str)
                and port_name.casefold() in FOLDED_GROUND_PORT_NAMES))


def _unportable_names(pin_list):
    """Check keys and stored names, including independently mutated pin records."""
    names = {}
    for key, pin in (pin_list or {}).items():
        for name in (key, getattr(pin, 'name', None)):
            if is_unportable_ground_port(name):
                names.setdefault(name, None)
    return list(names)


def unportable_ground_ports(module) -> list:
    """Declared port names on *module* that make its interface unportable.

    Reads the keys and stored names in ``module.pin_list`` and nothing else.
    ``net_dict`` is deliberately
    not consulted: the separate Xyce deck guard rejects internal ``gnd``
    nets. This predicate enforces the backend-independent port contract.
    """
    return _unportable_names(getattr(module, 'pin_list', None))


def block_name(module) -> str:
    """Best available name for *module* in a diagnostic."""
    for attr in ('module_name', 'circuit_name'):
        value = getattr(module, attr, None)
        if value:
            return str(value)
    return '<unnamed>'


def unportable_ground_port_message(block: str, port_name: str) -> str:
    """The one diagnostic for declaration, lint, emission and static admission."""
    prefix = f"block {block!r} declares a port named {port_name!r}: "
    remedy = (
        f" Rename the port to {SAFE_GROUND_PORT_NAME!r} and wire it to the "
        "parent's ground net when ground is intended; keep actual ground "
        "connections named '0'."
    )
    if _is_global_ground_port(port_name):
        return (prefix + "SPICE node 0 is global and pierces subcircuit boundaries, "
                "so a parent's wiring to this formal port is silently discarded."
                + remedy)
    return (prefix + "ngspice folds "
        f"this port name to the global SPICE node 0, so the block's interface "
        f"is not portable — a parent's wiring to that port is silently "
        f"discarded on ngspice, while Xyce honours it, and no "
        f"simulator warns. The declaration is the defect: a reusable block "
        f"cannot know how it will be wired later." + remedy)


def _require_portable_ground_ports(module, pin_list=None):
    """Refuse an unsafe interface, optionally checking its placement pins too.

    This checks one interface only; callers use their existing hierarchy
    walks, preserving the behavioral modeled-leaf replacement boundary.
    """
    from silk.exceptions import CircuitError
    offenders = dict.fromkeys((*unportable_ground_ports(module),
                               *_unportable_names(pin_list)))
    if offenders:
        raise CircuitError('; '.join(
            unportable_ground_port_message(block_name(module), name)
            for name in offenders))


def _require_entered_ground_interfaces(module):
    """Check raw immediate interfaces before graph views synchronize pins.

    Called only on entry to an existing visitor. This does not descend:
    visitors decide whether to enter children or replace them with a model.
    """
    from silk.circuit.ports import ElementType
    _require_portable_ground_ports(module)
    for entry in module._element_mgr.elem_dict.values():
        if entry['type'] == ElementType.MODULE:
            _require_portable_ground_ports(entry['device'], entry.get('pin_list'))
