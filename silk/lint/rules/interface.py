"""Interface rules: INT-001..INT-005.

The compatibility function ``check_interfaces`` runs these rules through
the registry.
"""
from collections import defaultdict
from typing import Iterator
from silk.lint.rule import Rule
from silk.lint.severity import Severity
from silk.lint.violations import InterfaceViolation
from silk.lint.locations import NetLocation
from silk.circuit.modules import Module


# The SPICE global ground node. Every SPICE simulator silk targets (ngspice,
# Xyce) reads the literal net name '0' as ground — a fact of the netlist
# language, not of any one backend — so the interface rules may know it
# here, in ONE place, without consulting ``module.simulator``. This is the
# only net name silk's lint layer recognises by spelling: 'gnd' / 'vss' /
# 'ground' are user naming conventions (ngspice folds 'gnd' to node 0; Xyce
# does not) and belong to a user ``RoleRuleset``, never to this file
# (TASK-85; the 'gnd' port-name hazard is TASK-36).
SPICE_GROUND_NODE = '0'


def _is_ground_node(net_name: str) -> bool:
    """True iff *net_name* is the simulator's global ground node ``'0'``.

    Backend-independent by construction: the rules call this and nothing
    else to decide whether a net is ground by name.
    """
    return net_name == SPICE_GROUND_NODE


# Binding kinds the interface rules reason about. 'submodule' and 'source'
# carry a direction; 'device' and 'passive' never do (an AnalogElement pin
# and a passive terminal have no declared direction — a MOSFET drain/source
# is bidirectional, a resistor end is either end), so they are kept as
# *presence* only: another party on the net, never a driver. 'lint_marker'
# (NoConnect) is not a party. Rules that need a driver ask for
# ``direction == 'output'``, which no device/passive binding ever has.
_RELEVANT_KINDS = ('submodule', 'source', 'device', 'passive')

# Kinds whose mere presence on a net is "the other party" for INT-004: a
# primitive at the same level, or the module's own port (see below).
_OTHER_PARTY_KINDS = ('device', 'passive', 'port')

# The module's own port pin counts as the other party only when the block
# receives the net from its parent: an IN / POWER / GROUND port is driven
# from outside, an INOUT port may be. An OUT port is the block's promise
# to drive, so it is NOT the other party — a net that reaches an OUT port
# with nothing driving it is exactly the floating output INT-004 exists
# to report.
_PORT_DIRS_DRIVEN_FROM_OUTSIDE = ('input', 'power', 'ground', 'inout')

# Which rules may read the ROLE on the module's own port binding (TASK-104).
# INT-002 does: a child's GROUND port wired onto the block's own POWER port
# is a supply short expressed through the interface, and only the own-port
# role can show it. INT-003 / INT-005 do not: they ask whether two CHILDREN
# disagree about what a net carries, and the block's own port is not a
# peer of its children — a DIGITAL port fanned into ANALOG-declared stdcell
# pins is the parent's abstraction boundary, not a wiring error (measured
# on the shipped ``sar_adc_spice``: seven such findings if they looked).
# Those two rules filter with this helper before comparing roles.
_ROLE_PEER_KINDS = ('submodule', 'source', 'device', 'passive')


def _role_peers(bindings):
    """The bindings INT-003 / INT-005 compare: every kind but ``'port'``."""
    return [b for b in bindings if b.kind in _ROLE_PEER_KINDS]


def _unwrap(value):
    """PortDir/PortRole enum or string -> lowercase string; None -> None."""
    if value is None:
        return None
    if hasattr(value, 'value'):
        value = value.value
    return str(value).lower()


def _own_port_binding(module, net_name):
    """The module's OWN port pin on *net_name*, as a synthetic binding, or None.

    Inside a hierarchical block the port name IS the internal net name (a
    ``.subckt`` header lists nodes), but nothing registers the port on
    ``Net.pins`` — the parent binds the *instance's* copy of the pin. So a
    pass-through net (the block's IN port fanned out to two children's
    inputs) had no driver as far as the interface rules could see, and
    INT-004 reported every such net inside every hierarchy.

    Returns a ``kind='port'`` :class:`PortBinding` carrying the declared
    direction — the module name (or ``'self'``) stands in as the instance
    — for a port driven from outside (:data:`_PORT_DIRS_DRIVEN_FROM_OUTSIDE`);
    ``None`` for an OUT port and for a net that is not a port.

    The port's ROLE rides along as the pin holds it: an explicit
    ``PortRole``, direction-implied supply role (a bare
    ``PortDir.POWER`` port is POWER — the pin already says so, nothing is
    derived here), or a role ``infer_port_roles`` filled — in which case
    ``role_inferred`` is carried too, so a consumer can tell the guess from
    the statement. first shipped this binding with
    ``role=None`` because carrying the role expanded INT-003 / INT-005 onto
    the block's own port (seven findings on the shipped ``sar_adc_spice``
    FSM blocks); those two rules now drop ``kind='port'`` bindings before
    comparing roles (:func:`_role_peers`), so the role reaches INT-002 —
    which could not see a child's GROUND port wired onto the block's own
    POWER port without it — and INT-004, whose presence rule already
    silences any net the block's own port is on. Never stored on the Net:
    STRUCT-011, netlisting and current annotation keep counting real pins
    only.
    """
    from silk.circuit.net import PortBinding
    pin = (getattr(module, 'pin_list', None) or {}).get(net_name)
    if pin is None:
        return None
    direction = _unwrap(getattr(pin, 'direction', None))
    if direction not in _PORT_DIRS_DRIVEN_FROM_OUTSIDE:
        return None
    return PortBinding(
        net=net_name,
        instance=getattr(module, 'module_name', None) or 'self',
        port=net_name,
        direction=direction,
        role=_unwrap(getattr(pin, 'role', None)),
        kind='port',
        role_inferred=bool(getattr(pin, 'role_inferred', False)),
    )


def _iter_module_net_bindings(module) -> dict[str, list]:
    """Yield (net_name, list_of_bindings) for each net with ≥2 bindings.

    Includes submodule port bindings, top-level source bindings
    (``VoltageSource``, ``CurrentSource``, VCVS/VCCS/CCVS/CCCS), same-level
    device and passive bindings, and the module's own port pin when the
    net is one of its ports driven from outside. Source bindings carry
    ``direction='output'`` on output terminals, so the NoDriverRule can
    recognise a top-level source as a driver of its connected net —
    previously a false positive when a single source drove a net with
    multiple submodule input sinks.

    Device and passive bindings carry ``direction=None`` (an
    ``AnalogElement`` pin has no declared direction) and are kept as
    presence only — "something else is on this net" — never as a driver:
    INT-001 still cannot see a device shorted against a submodule output,
    and that is deliberate until pins carry a direction. The
    module's own IN / POWER / GROUND / INOUT port is likewise the other
    party on a pass-through net inside a hierarchy, carrying its declared
    role for INT-002 (INT-003 / INT-005 drop it: :func:`_role_peers`); see
    :func:`_own_port_binding`.

    The ``>= 2`` filter is deliberate and stays: every rule built on this
    helper asks whether two things on ONE net conflict, which a lone
    binding cannot. It does mean a net with a single undriven input is
    invisible here — that shape (a net-name typo, a floating DUT input, a
    lone gate) is owned by STRUCT-011 SINGLE_CONNECTION_NET, which counts
    bindings of every kind and is in the default catalog. See.

    The one exception is the ground node ``'0'``: the simulator itself is
    always the other party on it, so a single binding there is kept — a
    lone POWER port wired to ``'0'`` must reach INT-002.
    """
    net_bindings: dict = defaultdict(list)
    net_dict = getattr(module, 'net_dict', {})
    for net_name, net in net_dict.items():
        for b in net.pins:
            if b.kind not in _RELEVANT_KINDS:
                continue
            net_bindings[net_name].append(b)
        own = _own_port_binding(module, net_name)
        if own is not None:
            net_bindings[net_name].append(own)
    return {n: bs for n, bs in net_bindings.items()
            if len(bs) >= 2 or _is_ground_node(n)}


def _net_loc(net_name: str, ports: list) -> NetLocation:
    """Build a NetLocation from a net name and binding list."""
    port_names = [f"{b.instance}.{b.port}" for b in ports]
    return NetLocation(net_name=net_name, port_names=port_names)


class DriverConflictRule(Rule):
    rule_id = "INT-001"
    category = "interface"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A net with more than one OUT-direction port is almost always a "
        "wiring mistake. Two drivers on the same net means the one with "
        "lower output impedance wins; at best this produces incorrect "
        "behavior, at worst it shorts two power paths through the drivers. "
        "Intentional dotted-OR connections should be annotated or waived "
        "with a reason string so the intent is documented."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, bindings in _iter_module_net_bindings(module).items():
            drivers = [b for b in bindings if b.direction == 'output']
            if len(drivers) < 2:
                continue
            names = ', '.join(f'{b.instance}.{b.port}' for b in drivers)
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"Net '{net_name}' has multiple drivers: {names}",
                location=_net_loc(net_name, drivers),
            )


class PowerGroundShortRule(Rule):
    rule_id = "INT-002"
    category = "interface"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A net with both POWER-role and GROUND-role ports creates a direct "
        "short between supply rails. This is a catastrophic silicon failure "
        "— the circuit draws unbounded current through the bonds and can "
        "physically destroy the die. There is no legitimate reason for a "
        "net to bind both a POWER and GROUND port simultaneously. Fix by "
        "moving one of the named ports to its own rail net; this rule is "
        "not meant to be waived.\n"
        "  The net named '0' is the simulator's global ground node, so a "
        "POWER-role port wired to '0' is a grounded supply even when no "
        "GROUND-role port shares the net (TASK-85).\n"
        "  Inside a hierarchical block the block's own POWER / GROUND port "
        "takes part: a child's GROUND port wired onto the net that carries "
        "the block's own POWER port (or POWER onto GROUND) is the same short, "
        "expressed through the interface, and is reported with the block's "
        "port named as a party (TASK-104)."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, bindings in _iter_module_net_bindings(module).items():
            has_power  = any(b.role == 'power'  for b in bindings)
            has_ground = any(b.role == 'ground' for b in bindings)
            if not has_power:
                continue
            if _is_ground_node(net_name):
                # '0' IS ground: the supply is grounded whether or not a
                # GROUND-role port happens to share the net.
                involved = [b for b in bindings if b.role in ('power', 'ground')]
                names = ', '.join(f'{b.instance}.{b.port}({b.role})' for b in involved)
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=(f"Net '{net_name}' is the simulator ground node "
                             f"— supply is grounded: {names}"),
                    location=_net_loc(net_name, involved),
                )
                continue
            if not has_ground:
                continue
            involved = [b for b in bindings if b.role in ('power', 'ground')]
            names = ', '.join(f'{b.instance}.{b.port}({b.role})' for b in involved)
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"Net '{net_name}' connects POWER and GROUND ports: {names}",
                location=_net_loc(net_name, involved),
            )


class RoleMismatchRule(Rule):
    rule_id = "INT-003"
    category = "interface"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A DIGITAL-role port connected to an ANALOG-role port indicates an "
        "abstraction boundary crossing without explicit interface logic. "
        "Digital signals have fast edges and rail-to-rail swings that inject "
        "noise into sensitive analog nodes. Analog signals have intermediate "
        "voltages that digital inputs cannot reliably interpret. If the "
        "connection is intentional (e.g., digital control of an analog "
        "switch), annotate one of the ports with role=MIXED and waive.\n"
        "  Compared child against child only: the block's own port is not a "
        "peer of the instances inside it, so a DIGITAL port fanned into "
        "ANALOG-declared children is not reported here (TASK-104)."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, all_bindings in _iter_module_net_bindings(module).items():
            bindings = _role_peers(all_bindings)
            roles = {b.role for b in bindings if b.role is not None}
            if 'digital' not in roles or 'analog' not in roles:
                continue
            involved = [b for b in bindings if b.role in ('digital', 'analog')]
            names = ', '.join(f'{b.instance}.{b.port}({b.role})' for b in involved)
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"Net '{net_name}' mixes DIGITAL and ANALOG ports: {names}",
                location=_net_loc(net_name, involved),
            )


class NoDriverRule(Rule):
    rule_id = "INT-004"
    category = "interface"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A net with one or more IN-direction ports but zero OUT-direction "
        "ports has nothing driving it. Downstream inputs read a floating "
        "voltage whose value depends on leakage, coupling, and history — "
        "effectively a random bit. In simulation this typically produces X "
        "propagation; in silicon it produces metastable behavior. Either "
        "add a driver or annotate the net with an intentional pull-up/down.\n"
        "  This rule needs at least two submodule/source bindings on the "
        "net to have anything to compare; a net whose ONLY binding is that "
        "one undriven input is reported by STRUCT-011 "
        "SINGLE_CONNECTION_NET instead (TASK-35). The net named '0' is the "
        "simulator's global ground node and is driven by definition; it is "
        "never reported here (TASK-85).\n"
        "  Supply nets are skipped, with one asymmetry (TASK-42): a "
        "GROUND-role net is skipped whether the role was declared or "
        "inferred, because nothing ever drives a return rail in the OUT "
        "direction (source negatives bind as inputs) so the rule has "
        "nothing to say about it; a POWER-role net is skipped only when the "
        "role was DECLARED. A real supply is driven — by a source at the "
        "top, by the parent through the port inside a hierarchy — so a "
        "declared POWER on an IN port is the user's statement that the "
        "drive is external, whereas a role filled by infer_port_roles from "
        "a name pattern ('*vdd*' on a monitor input such as 'vdd_sense') is "
        "a guess and buys no silence: the net is judged on its drivers like "
        "any signal net.\n"
        "  What counts as a driver (TASK-46): a submodule or source binding "
        "with direction OUT. What counts as the other party — enough to "
        "silence this rule, though never a driver for INT-001 — is the "
        "presence of a same-level primitive (a MOSFET drain, a resistor "
        "divider tap: device and passive pins carry no direction, and a "
        "drain or a resistor end is bidirectional, so presence is all that "
        "can be said) or the module's own IN / POWER / GROUND / INOUT port "
        "on a pass-through net inside a hierarchy (the parent drives it "
        "through the port). The module's own OUT port is not the other "
        "party: a net that reaches an OUT port with nothing driving it is a "
        "floating output and is reported."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, bindings in _iter_module_net_bindings(module).items():
            if _is_ground_node(net_name):
                continue
            annotated = [b for b in bindings if b.role is not None]
            if annotated and all(b.role in ('power', 'ground') for b in annotated):
                # A supply net — unless the only supply evidence is an
                # INFERRED POWER role, which is a guess (TASK-42).
                if any(b.role == 'ground'
                       or not getattr(b, 'role_inferred', False)
                       for b in annotated):
                    continue
            drivers = [b for b in bindings if b.direction == 'output']
            if drivers:
                continue
            # A same-level device/passive pin or the module's own
            # outside-driven port is the other party: nothing to say
            # (TASK-46). Presence only — never a driver for INT-001.
            if any(b.kind in _OTHER_PARTY_KINDS for b in bindings):
                continue
            names = ', '.join(f'{b.instance}.{b.port}({b.direction})' for b in bindings)
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"Net '{net_name}' has no driver: {names}",
                location=_net_loc(net_name, bindings),
            )


class ClockOnAnalogRule(Rule):
    rule_id = "INT-005"
    category = "interface"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A CLOCK-role port sharing a net with an ANALOG-role port means "
        "the full-swing clock edge is being injected into an analog signal "
        "path. Coupling and substrate noise from the switching edge will "
        "corrupt measurements and bias points. Use a level translator, "
        "buffer, or at minimum a clean routing barrier.\n"
        "  Compared child against child only: the block's own CLOCK port "
        "fanned into ANALOG-declared children is not reported here "
        "(TASK-104)."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, all_bindings in _iter_module_net_bindings(module).items():
            bindings = _role_peers(all_bindings)
            has_clock  = any(b.role == 'clock'  for b in bindings)
            has_analog = any(b.role == 'analog' for b in bindings)
            if not (has_clock and has_analog):
                continue
            involved = [b for b in bindings if b.role in ('clock', 'analog')]
            names = ', '.join(f'{b.instance}.{b.port}({b.role})' for b in involved)
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"Net '{net_name}' connects CLOCK and ANALOG ports: {names}",
                location=_net_loc(net_name, involved),
            )
