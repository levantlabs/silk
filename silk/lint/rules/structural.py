"""Structural rules — STRUCT-001..STRUCT-013 plus BUS-002.

STRUCT-009 is shipped but opt-in (see ``silk.lint.rules.OPTIONAL_CATALOG``);
everything else here, STRUCT-010 included is in the default
catalog.
"""
import fnmatch
from typing import Iterator
from silk.lint.rule import Rule
from silk.lint.severity import Severity
from silk.lint.violations import InterfaceViolation
from silk.lint.locations import NetLocation, PortLocation, InstanceLocation, ModuleLocation
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import ElementType
from silk.lint.rules.interface import _is_ground_node
from silk.circuit.ground_port import (
    block_name, unportable_ground_ports, unportable_ground_port_message,
)


class FloatingNetRule(Rule):
    rule_id = "STRUCT-003"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "A net has zero connections in its parent module. Usually this is "
        "a leftover from refactoring — a port was removed but the net "
        "declaration remains. Sometimes it is intentional (a probe point, "
        "a test tap, an unused differential output). For intentional "
        "floating nets, register a silk.lint_markers.NoConnect on the net "
        "via Module.add_no_connect(net_name, reason='...'). The rule will "
        "then skip the net and report it in the 'Intentionally Unconnected' "
        "section of the lint report."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, net in module.net_dict.items():
            pins = net.pins
            # Truly floating: zero pins
            if not pins:
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=f"Net '{net_name}' has no connections",
                    location=NetLocation(net_name=net_name, port_names=[]),
                )
                continue
            # All pins are lint markers (NoConnect) → intentional, skip
            if all(getattr(p, 'kind', None) == 'lint_marker' for p in pins):
                continue
            # Otherwise the net has real connections — not floating


class SingleConnectionNetRule(Rule):
    rule_id = "STRUCT-011"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "A net has exactly ONE connection in its parent module. One pin on "
        "a net is not a circuit: nothing drives it, nothing loads it, and "
        "the node floats to whatever the solver's initial condition and the "
        "device's leakage produce — ngspice happily simulates it and prints "
        "a plausible-looking voltage. This is the shape every common wiring "
        "mistake takes: a typo'd net name under the default "
        "strict_nets=False mode (the misspelling auto-creates a fresh net "
        "with the one pin on it), a DUT input nobody drives, a lone "
        "transistor gate, an output nothing reads.\n"
        "  Zero-pin nets belong to STRUCT-003 FLOATING_NET; a net that is a "
        "declared port of the module being checked is skipped, because its "
        "second connection is made by the parent when the block is "
        "instantiated. The net named '0' is also skipped: it is the "
        "simulator's global ground node, and the simulator itself is "
        "always the other party on it, so a lone binding there is never "
        "one-ended (pre-TASK-88 this rule reported 'Net 0 has only one "
        "connection: ...' for a lone GROUND or POWER port on '0' — a false "
        "positive, and for POWER a duplicate of INT-002's grounded-supply "
        "finding).\n"
        "  For an intentionally one-ended net (a probe point, a test tap, a "
        "monitor-only output), register a silk.lint_markers.NoConnect via "
        "Module.add_no_connect(net_name, reason='...'). The rule then skips "
        "the net and it is listed in the 'Intentionally Unconnected' "
        "section of the lint report."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        # Ports of THIS module are wired by the parent, not here.
        declared_ports = set(getattr(module, 'pin_list', None) or {})
        for net_name, net in module.net_dict.items():
            pins = net.pins
            # A NoConnect marker is the documented per-net suppression. It
            # is itself a PortBinding, so it has to be tested before the
            # count — otherwise a marker on a floating net would look like
            # that net's one connection.
            if any(getattr(p, 'kind', None) == 'lint_marker' for p in pins):
                continue
            if len(pins) != 1:
                continue    # 0 pins -> STRUCT-003; >= 2 -> a real node
            if net_name in declared_ports:
                continue
            if _is_ground_node(net_name):
                continue    # '0' always has the simulator as its other party
            binding = pins[0]
            port_name = f"{binding.instance}.{binding.port}"
            detail = port_name
            if binding.direction:
                detail = f"{port_name}({binding.direction})"
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    f"Net '{net_name}' has only one connection: {detail} — "
                    f"nothing else is wired to it"
                ),
                location=NetLocation(net_name=net_name,
                                     port_names=[port_name]),
            )


def _module_instances(module: Module):
    """Yield (instance_name, device) pairs for submodule instances in elem_dict.

    Filters elem_dict for entries whose type is ElementType.MODULE so that
    primitive devices (resistors, MOSFETs, etc.) are excluded.
    """
    for instance_name, entry in module.elem_dict.items():
        if entry.get('type') != ElementType.MODULE:
            continue
        device = entry.get('device')
        if device is None:
            continue
        yield instance_name, device, entry.get('pin_list', device.pin_list)


def _device_instances(module: Module):
    """Yield (instance_name, device) pairs for primitive device instances."""
    for instance_name, entry in module.elem_dict.items():
        if entry.get('type') != ElementType.DEVICE:
            continue
        device = entry.get('device')
        if device is None:
            continue
        yield instance_name, device


def _unwrap_role(role) -> str:
    """Convert a PortRole enum or string to a lowercase string."""
    if role is None:
        return ''
    if hasattr(role, 'value'):
        return str(role.value).lower()
    return str(role).lower()


class DanglingPortRule(Rule):
    rule_id = "STRUCT-001"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "A port declared on a submodule instance is not bound to any net "
        "inside the parent scope. Almost always an oversight or a leftover "
        "from refactoring: silicon will read a floating voltage at an input "
        "or leave an output disconnected. Intentional unconnected ports "
        "have two suppression paths:\n"
        "  (1) Per-pin waiver (preferred): "
        "Module.add_no_connect_port(instance, port, reason='...'). Forces "
        "a documented reason and survives downstream port renames.\n"
        "  (2) Name globs for device-library conventions: "
        "DanglingPortRule.ignore_patterns = ['nc', 'debug_*', 'dnu_*']. "
        "Use this only for PDK-mandated unused pins across many instances."
    )
    config_schema = {"ignore_patterns": list}

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        patterns = self.config.get("ignore_patterns", [])
        # Per-pin waivers from Module.add_no_connect_port(...). Stored as
        # NoConnectPort instances on module._no_connect_ports.
        waived = {(m.instance, m.port)
                  for m in getattr(module, '_no_connect_ports', [])}
        for instance_name, device, pins in _module_instances(module):
            pin_list = pins or {}
            for port_name, pin in pin_list.items():
                if (instance_name, port_name) in waived:
                    continue
                if any(fnmatch.fnmatch(port_name, p) for p in patterns):
                    continue
                if not getattr(pin, 'net', None):
                    yield InterfaceViolation(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"Submodule instance '{instance_name}' "
                            f"port '{port_name}' is not connected"
                        ),
                        location=PortLocation(
                            port_name=port_name,
                            module_name=instance_name,
                        ),
                    )


class PowerPortUnconnectedRule(Rule):
    rule_id = "STRUCT-002"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A submodule instance declares a POWER- or GROUND-role port, but "
        "the parent module does not bind that port to any net. A floating "
        "supply pin on an instantiated block means the block has no power "
        "and will not function; worse, some foundry macros draw leakage "
        "through ESD structures when unpowered, corrupting neighboring "
        "nets. Always wire power and ground pins of every instance."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for instance_name, device, pins in _module_instances(module):
            pin_list = pins or {}
            for port_name, pin in pin_list.items():
                role = _unwrap_role(getattr(pin, 'role', None))
                if role not in ('power', 'ground'):
                    continue
                if not getattr(pin, 'net', None):
                    yield InterfaceViolation(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"Instance '{instance_name}' "
                            f"port '{port_name}' (role={role}) is not connected"
                        ),
                        location=PortLocation(
                            port_name=port_name,
                            module_name=instance_name,
                        ),
                    )


class FloatingGateRule(Rule):
    rule_id = "STRUCT-004"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A MOSFET gate terminal is not bound to any net. A floating gate "
        "accumulates charge from leakage and has an indeterminate bias, "
        "causing the device to conduct or not conduct based on its history. "
        "In sensitive analog circuits this can produce wild offsets; in any "
        "circuit it is a latent reliability hazard. Either connect the gate "
        "to an explicit bias or tie it to a rail through a resistor. If the "
        "device is intentionally unused, delete it."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for instance_name, device in _device_instances(module):
            if not self._is_mosfet(device):
                continue
            gate_net = self._gate_net(device)
            if not gate_net:
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=f"MOSFET '{instance_name}' gate is not connected",
                    location=InstanceLocation(instance_path=instance_name),
                )

    @staticmethod
    def _is_mosfet(device) -> bool:
        type_name = type(device).__name__.lower()
        if any(k in type_name for k in ('mos', 'fet')):
            return True
        # Also check by pin set
        pin_keys = set()
        nets = getattr(device, 'nets', None)
        if isinstance(nets, dict):
            pin_keys = set(nets.keys())
        else:
            pin_list = getattr(device, 'pin_list', None) or {}
            pin_keys = set(pin_list.keys())
        return {'d', 'g', 's', 'b'}.issubset(pin_keys)

    @staticmethod
    def _gate_net(device):
        nets = getattr(device, 'nets', None)
        if isinstance(nets, dict):
            return nets.get('g') or nets.get('gate')
        pin_list = getattr(device, 'pin_list', None) or {}
        for pin_name in ('g', 'gate'):
            if pin_name in pin_list:
                return pin_list[pin_name].net
        return None


def _body_net(device):
    """Return the bound net for a MOSFET body/bulk terminal, or None."""
    nets = getattr(device, 'nets', None)
    if isinstance(nets, dict):
        return nets.get('b') or nets.get('bulk') or nets.get('body')
    pin_list = getattr(device, 'pin_list', None) or {}
    for pin_name in ('b', 'bulk', 'body'):
        if pin_name in pin_list:
            return pin_list[pin_name].net
    return None


#: Typed value alias per native passive class (silk.elements.passives). The
#: message must name the parameter in the spelling the author writes, and a
#: Resistor's value is written ``resistance=`` / ``.resistance``, never
#: ``value=``.
_PASSIVE_VALUE_NAMES = {
    'Resistor': 'resistance',
    'Capacitor': 'capacitance',
    'Inductor': 'inductance',
}


def _param_value(entry):
    """Current value of one declared parameter, unwrapped.

    *entry* is whatever the device's parameter mapping holds: a plain value,
    a ``ParameterConstraint`` (legacy ``PDKDevice.param_list``), or a
    ``Variable``. A constraint whose ``value`` is unset resolves to the
    catalog ``default`` — the same fallback
    :meth:`BasePDK.generate_netlist_entry` applies when it writes the deck,
    so the rule judges the device the simulator will actually see.
    """
    from silk.pdk.basepdk import ParameterConstraint   # deferred: circular import
    if isinstance(entry, ParameterConstraint):
        entry = entry.value if entry.value is not None else entry.default
    if entry is not None and not isinstance(entry, (int, float, str)) \
            and hasattr(entry, 'value'):   # Variable duck-type
        entry = entry.value
    return entry


def _declared_params(device):
    """``[(param_name, value), ...]`` for every parameter *device* declares.

    Names come from the device's OWN contract, never from a guess keyed on
    the class name, so the rule reads the same spelling the author writes:

    * native passives (:mod:`silk.elements.passives`) — the single ``value``,
      under its typed alias. Checked FIRST: these classes are
      themselves factory-built ``AnalogElement`` subclasses (``_param_names``
      is ``('value',)``, matching the next check below), but they still keep
      their own constructor's spelling ('resistance' etc.) for THIS message,
      exactly as before the shim;
    * factory-built :class:`AnalogElement` (every PDK catalog device, and a
      directly constructed ``silk.elements.elements.resistor`` etc.) —
      ``_param_names`` + the ``params`` dict (``w``, ``l``, ``m``, or
      ``value``);
    * legacy ``PDKDevice`` (the ``add_element`` path) — ``param_list``.

    Returns ``[]`` for a device that describes no parameters at all: the rule
    cannot tell an omission from an absent contract, and an ERROR it cannot
    substantiate is worse than silence.
    """
    value_name = _PASSIVE_VALUE_NAMES.get(type(device).__name__)
    if value_name is not None and hasattr(device, 'value'):
        return [(value_name, _param_value(device.value))]

    param_names = getattr(type(device), '_param_names', None)
    params = getattr(device, 'params', None)
    if param_names and isinstance(params, dict):
        rules = getattr(type(device), '_emission_spec', {}).get('paramList', {})
        defaults = getattr(type(device), '_param_defaults', {})
        values = []
        for p in param_names:
            value = params.get(p)
            if p not in params and rules.get(p, {}).get('emit_if_supplied'):
                value = defaults.get(p)
            values.append((p, _param_value(value)))
        return values

    param_list = getattr(device, 'param_list', None)
    if isinstance(param_list, dict):
        return [(p, _param_value(entry)) for p, entry in param_list.items()]

    return []


class BodyNotTiedRule(Rule):
    rule_id = "STRUCT-005"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "A MOSFET body/bulk terminal is not explicitly bound to a net. "
        "Depending on process, the body may need to be tied to the most "
        "negative rail (NMOS→gnd) or the most positive rail (PMOS→vdd, "
        "or nwell for triple-well devices). Leaving it unbound leaves "
        "behavior to the layout, which silk cannot verify from the "
        "schematic. For processes where the body is auto-tied at layout "
        "(bulk CMOS with ring contacts), silence this rule via waiver or "
        "config override — the goal is to make the intent explicit."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for inst_name, device in _device_instances(module):
            if not FloatingGateRule._is_mosfet(device):
                continue
            if _body_net(device):
                continue
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=f"MOSFET '{inst_name}' body/bulk terminal not bound",
                location=InstanceLocation(instance_path=inst_name),
            )


class MissingParamRule(Rule):
    rule_id = "STRUCT-006"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A device instance is missing a required parameter: MOSFETs "
        "without w or l, resistors without resistance, capacitors without "
        "capacitance, inductors without inductance. A device with a null "
        "parameter cannot be simulated or laid out — it is effectively "
        "not a real device. This is nearly always an oversight during "
        "instance construction; fix by supplying the missing argument. "
        "The parameters checked are the ones the device itself declares "
        "(the PDK catalog's paramList, or a passive's value), named in "
        "the spelling you write them in; a device that declares no "
        "parameters is not reported."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        # Native passives live in passives_dict, not elem_dict — without the
        # second pass an unsized R/C/L is invisible to this rule.
        instances = list(_device_instances(module))
        instances += list((getattr(module, 'passives_dict', None) or {}).items())
        for inst_name, device in instances:
            for param_name, value in _declared_params(device):
                if value is not None:
                    continue
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=(
                        f"Instance '{inst_name}' (type={type(device).__name__}) "
                        f"missing required parameter '{param_name}'"
                    ),
                    location=InstanceLocation(instance_path=inst_name),
                )


class SignalOnSupplyNetRule(Rule):
    rule_id = "STRUCT-007"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A net carries both a power/ground role port and a signal-role "
        "(analog, digital, or clock) port. This means the 'signal' is "
        "physically tied to a supply rail, which is almost always a "
        "wiring mistake — the signal cannot actually change. If the "
        "intent is to statically tie an input high or low, use an "
        "explicit bias element or mark the connection with a tie-off "
        "waiver; do not rely on sharing a supply net."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for net_name, net in module.net_dict.items():
            roles = {_unwrap_role(p.role) for p in net.pins if p.role is not None}
            has_supply = bool(roles & {'power', 'ground'})
            has_signal = bool(roles & {'analog', 'digital', 'clock'})
            if not (has_supply and has_signal):
                continue
            signal_pins = [p for p in net.pins if _unwrap_role(p.role) in ('analog', 'digital', 'clock')]
            port_names = [f"{p.instance}.{p.port}" for p in signal_pins]
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    f"Signal port(s) on supply net '{net_name}': "
                    f"{', '.join(port_names)}"
                ),
                location=NetLocation(net_name=net_name, port_names=port_names),
            )


class UnusedInstanceRule(Rule):
    rule_id = "STRUCT-008"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "An instance exists in the module but none of its terminals are "
        "bound to any net. This makes the instance unreachable — it has "
        "no electrical effect on the circuit. Usually this is a leftover "
        "from a refactor (the instance should be deleted) or a hooked-up "
        "placeholder that was never wired. Compare with the NoConnect "
        "marker, which is specifically for intentionally floating nets; "
        "an unused instance has no justification to remain in the design."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        for inst_name, entry in module.elem_dict.items():
            device = entry.get('device')
            if device is None:
                continue
            if getattr(device, 'is_lint_marker', False):
                continue
            pin_list = entry.get('pin_list', getattr(device, 'pin_list', None))
            nets_dict = getattr(device, 'nets', None)
            if isinstance(pin_list, dict):
                bound = any(getattr(p, 'net', None) for p in pin_list.values())
            elif isinstance(nets_dict, dict):
                bound = any(nets_dict.values())
            else:
                continue  # can't tell — assume bound
            if not bound:
                yield InterfaceViolation(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=f"Instance '{inst_name}' has no terminal bindings",
                    location=InstanceLocation(instance_path=inst_name),
                )


def _unwrap_direction(direction) -> str:
    """Convert a PortDir enum or string to a lowercase string."""
    if direction is None:
        return ''
    if hasattr(direction, 'value'):
        return str(direction.value).lower()
    return str(direction).lower()


def _module_has_devices(module: Module) -> bool:
    """True if module.elem_dict contains any primitive device entry."""
    for entry in module.elem_dict.values():
        if entry.get('type') == ElementType.DEVICE:
            return True
    return False


class PowerGroundPortRequired(Rule):
    rule_id = "STRUCT-009"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "Every subcircuit that contains transistor-level devices must "
        "declare at least one POWER-role and one GROUND-role port. Without "
        "declared supply ports the block is relying on hidden global nets, "
        "which most foundry flows forbid and which make the block unusable "
        "across different supply domains. Purely behavioral wrappers with "
        "no devices are skipped — they have nothing to power. Fix by "
        "adding explicit vdd/gnd ports to the subcircuit definition and "
        "wiring them at the parent level."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        # Only applies to subcircuit modules (those with a module_name set).
        if not getattr(module, 'module_name', None):
            return
        # Skip behavioral-only modules with no physical devices.
        if not _module_has_devices(module):
            return
        pin_list = getattr(module, 'pin_list', None) or {}
        declared_roles = {_unwrap_role(p.role) for p in pin_list.values()}
        if 'power' not in declared_roles:
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    f"Subcircuit '{module.module_name}' contains devices but "
                    f"declares no port with role POWER (missing POWER supply port)"
                ),
                location=ModuleLocation(module_name=module.module_name),
            )
        if 'ground' not in declared_roles:
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    f"Subcircuit '{module.module_name}' contains devices but "
                    f"declares no port with role GROUND (missing GROUND supply port)"
                ),
                location=ModuleLocation(module_name=module.module_name),
            )


# Directions under which a supply-role port RECEIVES its rail from the
# parent: an explicit IN, or the bare supply direction (``PortDir.POWER`` /
# ``PortDir.GROUND``), which is not a signal-flow direction at all but a
# statement of what the port carries (TASK-45; the examples' idiom).
_RECEIVING_DIRECTIONS = frozenset({'input', 'in', 'power', 'ground'})


class PortDirectionSanity(Rule):
    rule_id = "STRUCT-010"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "A port declared with role POWER or GROUND normally uses direction "
        "IN because a subcircuit receives its supplies from the parent. "
        "The one legitimate exception is a block that PRODUCES a rail from "
        "one it receives — a regulator, a charge pump, a header or footer "
        "switch, a bias generator: its regulated output is a POWER-role OUT "
        "port (Port.supply(direction='out')). The rule therefore exempts a "
        "supply-role OUT port whenever the block also declares a port of "
        "the SAME role that receives (direction IN, or the bare "
        "PortDir.POWER / PortDir.GROUND). A supply-role OUT port in a block "
        "with no such input has nothing to derive the rail from and almost "
        "certainly means the designer confused direction conventions, which "
        "will produce misleading hierarchical netlists. Direction INOUT is "
        "occasionally legitimate (a rail that both sources and sinks) but "
        "rare enough that an explicit warning is warranted so the intent is "
        "documented. Fix by changing the direction to IN, or — for a rail "
        "the block genuinely emits — declare the supply it is derived from "
        "as a same-role IN port; waive the rule if INOUT is truly intended."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        if not getattr(module, 'module_name', None):
            return
        pin_list = getattr(module, 'pin_list', None) or {}
        # Roles for which the block declares a receiving port. A block that
        # takes a POWER rail in may emit a POWER rail (regulator/converter);
        # one that takes a GROUND rail in may emit one (footer switch,
        # local-ground generator). The role must match: an inverter whose
        # author swapped 'vdd' to OUT still declares only a GROUND input, so
        # its POWER OUT port keeps being reported.
        received_roles = set()
        for pin in pin_list.values():
            role = _unwrap_role(getattr(pin, 'role', None))
            if role not in ('power', 'ground'):
                continue
            if _unwrap_direction(getattr(pin, 'direction', None)) in _RECEIVING_DIRECTIONS:
                received_roles.add(role)
        for port_name, pin in pin_list.items():
            role = _unwrap_role(getattr(pin, 'role', None))
            if role not in ('power', 'ground'):
                continue
            direction = _unwrap_direction(getattr(pin, 'direction', None))
            # PortDir enum values: 'input', 'output', 'inout', 'power', 'ground'.
            # A receiving direction is fine for a supply port.
            if direction in _RECEIVING_DIRECTIONS:
                continue
            if direction in ('output', 'out'):
                if role in received_roles:
                    # Regulated / derived output: the block receives a rail
                    # of this role and emits one (TASK-94 ruling).
                    continue
                message = (
                    f"Port '{port_name}' (role={role.upper()}) has direction "
                    f"OUT and the block declares no {role.upper()}-role input; "
                    f"supply ports should be IN unless the block derives the "
                    f"rail from a same-role supply it receives"
                )
            elif direction == 'inout':
                message = (
                    f"Port '{port_name}' (role={role.upper()}) has "
                    f"direction INOUT; supply ports should be IN"
                )
            else:
                continue
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=message,
                location=PortLocation(
                    port_name=port_name,
                    module_name=module.module_name,
                ),
            )


def _is_module_instance(entry: dict) -> bool:
    """True if elem_dict entry represents a submodule (not a primitive device).

    Accepts both the canonical ElementType.MODULE enum value and the plain
    string 'module' used in manual test stubs.
    """
    t = entry.get('type')
    if t is None:
        return False
    return t == ElementType.MODULE or str(t).lower() == 'module'


class BusPartialWiringRule(Rule):
    rule_id = "BUS-002"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (Module,)
    rationale = (
        "A bus port on a sub-instance has one or more bits unwired, with "
        "no NoConnectPort marker. To suppress, declare unused bits "
        "explicitly via Module.add_no_connect_port(instance, 'bus_name[i]', "
        "reason='...') or, for whole buses, pass the bus name (without "
        "index) to add_no_connect_port. See spec §2.5."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        # Build the set of (instance, bit_pin_name) pairs that have markers.
        marked = set()
        for marker in getattr(module, '_no_connect_ports', []):
            marked.add((marker.instance, marker.port))

        for instance_name, entry in module.elem_dict.items():
            if not _is_module_instance(entry):
                continue
            device = entry.get('device')
            if device is None:
                continue
            instance_buses = getattr(device, '_buses', {}) or {}
            pin_list = entry.get('pin_list', getattr(device, 'pin_list', None)) or {}
            for bus_name, spec in instance_buses.items():
                unwired = []
                for bit_name in spec.expanded_names():
                    pin = pin_list.get(bit_name)
                    if pin is None:
                        continue
                    pin_net = getattr(pin, 'net', None)
                    if pin_net in (None, ''):
                        if (instance_name, bit_name) not in marked:
                            unwired.append(bit_name)
                if unwired:
                    yield InterfaceViolation(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=(
                            f"Bus '{bus_name}' on instance '{instance_name}' "
                            f"has {len(unwired)} unwired bit(s): "
                            f"{', '.join(unwired)}. Mark intentional with "
                            f"add_no_connect_port."
                        ),
                        location=InstanceLocation(instance_path=instance_name),
                    )


class UnportableGroundPortRule(Rule):
    rule_id = "STRUCT-012"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A formal port named '0' is global SPICE ground on every backend: "
        "it pierces subcircuit boundaries and discards the parent's mapping. "
        "It is refused at declaration and rechecked for restored or mutated "
        "interfaces. Rename the formal to 'vss' and wire it to the parent's "
        "ground net; actual net and terminal values '0' remain valid.\n"
        "A module declares a port named 'gnd' (any case). ngspice folds that "
        "node name onto the global SPICE node 0; Xyce folds "
        "nothing, at any spelling. Node 0 is global and pierces subcircuit "
        "boundaries, so on ngspice the fold discards whatever the parent "
        "wired to that port position: '.subckt blk a gnd' instantiated as "
        "'Xblk top mid blk' ties the block's ground to global 0 instead of "
        "to 'mid'. Measured on one byte-identical deck: ngspice I(V1) = "
        "-1.0 mA with V(mid) = 0, Xyce I(V1) = -0.5 mA with V(mid) = 0.5, "
        "and ZERO warnings from either — two different circuits with no "
        "diagnostic anywhere.\n"
        "  The trigger is the DECLARATION, not the wiring. A reusable block "
        "is instantiated many times, possibly by code its author does not "
        "control and possibly in another design entirely, so a check keyed "
        "on the instantiation sites present in one netlist could never "
        "protect the block's interface. Every mapping of a 'gnd' port is "
        "suspect because the next one is out of reach.\n"
        "  Only PORTS are reported by this rule. Internal nets named 'gnd' "
        "can silently give wrong results on Xyce; silk rejects them with "
        "a separate Xyce deck guard. Safe port names include 'vss', 'VSS', 'agnd', "
        "'ground' and 'GROUND' as port names — ngspice does not fold "
        "any of them and they are legitimately distinct nets.\n"
        "  Fix by renaming the port, e.g. to 'vss' (verified not folded by "
        "ngspice). Do NOT canonicalise 'gnd' to '0' at emission: node 0 is "
        "global, so that would tie the block's ground to global ground even "
        "when the parent wired the port elsewhere, replicating ngspice's "
        "bug into the backends that get it right."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        block = block_name(module)
        for port_name in unportable_ground_ports(module):
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=unportable_ground_port_message(block, port_name),
                location=PortLocation(port_name=port_name, module_name=block),
            )


#: MOSFET terminal names that can only ever sit on a bias rail, never carry a
#: signal: the source and the body/bulk. Spelled the way the device contracts
#: spell them (``pdk.nmos1p8(s=..., b=...)``), like STRUCT-004's ``g`` and
#: STRUCT-005's ``b``. This is a *terminal* vocabulary, not a net-name one —
#: no rule here infers a role from what a net is called.
_BIAS_ONLY_MOS_TERMINALS = frozenset({'s', 'source', 'b', 'bulk', 'body'})


class HiddenSupplyNetRule(Rule):
    rule_id = "STRUCT-013"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (Module,)
    rationale = (
        "A subcircuit wires MOSFET source and body terminals to an internal "
        "net it does not declare as a port — a supply rail the author "
        "forgot to export. Nothing gates the net, no drain drives it, no "
        "passive or submodule conducts to it, and the parent cannot reach "
        "it: it is a hidden local supply, and the block simulates dead "
        "without a single diagnostic. Measured on a sky130 inverter whose "
        "body wires the PMOS to 'vdd' while the port block declares only "
        "vin/vout/gnd: ngspice-42 returns v(vout) = 0.0 and no error.\n"
        "  The trigger is structural, never a net's name. A net is reported "
        "only when EVERY pin on it is a MOSFET source or body terminal, "
        "because that is the one shape with no other reading — a gate, a "
        "drain, a resistor end, a submodule port or a source terminal all "
        "give the net a job other than 'rail'. Ports of the module are "
        "skipped (the parent wires them), and so is the global SPICE ground "
        "node '0', which the simulator itself always drives.\n"
        "  Fix by declaring the net as a supply port of the subcircuit "
        "(Port.supply() / Port.ground(), or the bare PortDir.POWER / "
        "PortDir.GROUND spelling) and wiring it at the parent level. If the "
        "block genuinely has no supply of that polarity — a PMOS-only "
        "current bank, a diode-connected bias reference, a resistor ladder, "
        "a cap-DAC switched between references — then every one of its "
        "device terminals already sits on a declared port and this rule "
        "stays silent; that is what 'supply-less by design' means here, and "
        "it needs no marker. STRUCT-009 (opt-in) is the stricter sibling "
        "that additionally requires POWER and GROUND ports to be declared.\n"
        "  Boundaries: a net with exactly one pin belongs to STRUCT-011 and "
        "one with none to STRUCT-003, so neither is repeated here. A hidden "
        "rail that also carries a decoupling capacitor or a back-annotated "
        "parasitic is NOT reported — the extra terminal removes the "
        "certainty, and a cap-DAC bottom plate (a cap plate plus two NMOS "
        "sources) is exactly that shape and is legitimate. This rule buys "
        "precision at that cost; run STRUCT-009 opt-in when you want the "
        "port declaration enforced regardless.\n"
        "  The suppression path is a waiver on this rule_id with a net "
        "matcher, NOT a NoConnect marker: NoConnect declares a net "
        "intentionally UNCONNECTED, and a hidden rail carries two or more "
        "real bindings, so find_stale_markers reports such a marker as "
        "stale. A marker does take the net out of scope here as a side "
        "effect — its binding is not a device terminal — but that is not "
        "the documented idiom and it leaves a stale marker in the report."
    )

    def check(self, module: Module) -> Iterator[InterfaceViolation]:
        # Only subcircuits: a testbench's supplies are sources, not ports.
        if not getattr(module, 'module_name', None):
            return
        if not _module_has_devices(module):
            return
        declared_ports = set(getattr(module, 'pin_list', None) or {})
        for net_name, net in module.net_dict.items():
            if net_name in declared_ports:
                continue
            if _is_ground_node(net_name):
                continue
            pins = net.pins
            # 0 pins -> STRUCT-003, 1 pin -> STRUCT-011.
            if len(pins) < 2:
                continue
            # No lint_marker branch: a NoConnect binding is not a device
            # terminal, so the all() below already takes such a net out of
            # scope. A real hidden rail is suppressed with a waiver — see
            # the rationale for why the marker is the wrong idiom here.
            if not all(getattr(p, 'kind', None) == 'device'
                       and str(p.port).lower() in _BIAS_ONLY_MOS_TERMINALS
                       for p in pins):
                continue
            terminals = [f"{p.instance}.{p.port}" for p in pins]
            yield InterfaceViolation(
                rule_id=self.rule_id,
                severity=self.severity,
                message=(
                    f"Subcircuit '{module.module_name}' wires "
                    f"{', '.join(terminals)} to net '{net_name}', which is "
                    f"not a declared port — hidden local supply"
                ),
                location=NetLocation(net_name=net_name, port_names=terminals),
            )
