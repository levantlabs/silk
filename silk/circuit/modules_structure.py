"""Structure / connectivity methods for :class:`~silk.circuit.modules.Module` — the circuit-building surface.

Split out of ``modules.py`` as the final and largest per-subsystem **part**. ``_StructureMethods`` is a pure mixin: no state, no separate identity. It carries the documented lazy ``from silk.circuit.modules import Module`` (in ``add_element``, ``add_module`` and ``_add_element_impl``) to avoid circular imports.
"""

import copy
import re
from types import MappingProxyType
from typing import TYPE_CHECKING

from silk.exceptions import (
    CircuitError, DuplicateNameError, UndeclaredNetError,
)
from silk.circuit.circuit_elements import ElementType
from silk.circuit.emitted_names import validate_instance_name, emitted_submodule_name
from silk.circuit._module_base import (
    _NetAccessor, ModulePin, SubcircuitInstance,
)
from silk.circuit.variable import _leaf_variables

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


# TASK-292.1: the .subckt-name validator. Neither the constructor's
# module_name= nor define_module_name() checked anything before this —
# a name SPICE cannot carry (whitespace, a leading digit, punctuation)
# used to surface much later as a mangled or rejected netlist line. This
# is also the exact set a Module(..., variants=True) auto-generated
# suffix must land inside: base_name already matches it, '_' plus lowercase
# hex digits do too, so the two checks compose without a second exemption.
# fullmatch, not match: '$' also matches before a trailing newline, so
# 'inv\n' passed and wrote a line break into the .subckt card (TASK-593).
_MODULE_NAME_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def _validate_module_name(name):
    """Refuse *name* as a module_name when SPICE cannot carry it as a
    ``.subckt`` identifier: must start with a letter or underscore and
    contain only letters, digits and underscores. Raises ``ValueError``
    naming the rejected value."""
    # A line break is a forbidden-text refusal, not a grammar error: the
    # shared check runs first so it raises DeckTextError (TASK-593).
    from silk.utils.deck_text import refuse_line_breaks
    refuse_line_breaks(name, who='module_name')
    if not isinstance(name, str) or not _MODULE_NAME_RE.fullmatch(name):
        raise ValueError(
            f"module_name {name!r} is not a valid SPICE subcircuit name: "
            f"it must match ^[A-Za-z_][A-Za-z0-9_]*$ (start with a letter "
            f"or underscore; letters, digits and underscores only)."
        )


def _resolve_port_mapping(parent, device, instance_name, wiring):
    """Expand wiring and reject overlapping original keys before placement."""
    from silk.circuit.net import Net
    from silk.circuit.bus_spec import _formal_bus, _port_targets

    expanded_wiring = {}
    claims = {}
    for original_key in wiring:
        for pin in _port_targets(device, original_key):
            if pin in claims:
                raise DuplicateNameError(
                    f"element {instance_name!r}: keys {claims[pin]!r} and "
                    f"{original_key!r} both claim formal pin {pin!r}. "
                    "Map each child pin once."
                )
            claims[pin] = original_key

    for port, val in wiring.items():
        # Reject multi-source list/tuple early — only raise for
        # iterables that are NOT strings (strings are valid net-name values).
        if isinstance(val, (list, tuple)) and not isinstance(val, str):
            raise CircuitError(
                f"port {port!r}: multi-source bus wiring is not supported "
                f"in v1 — Net.concat is deferred to v1.1. Pass a single "
                f"Net or slice instead. (Got a {type(val).__name__})"
            )

        spec = _formal_bus(device, port)
        if spec is not None:
            from silk.circuit.ports import PortDir

            string_bus = isinstance(val, str) and val in parent._bus_handles
            if string_bus:
                val = parent._bus_handles[val]
            elif isinstance(val, str):
                scalar_name = parent._element_mgr._canonical_net_name(val)
                if scalar_name not in parent.net_dict and val not in (parent.pin_list or {}):
                    raise UndeclaredNetError(
                        f"port {port!r} of element {instance_name!r}: net {val!r} "
                        "is not declared. Pass a declared bus name or handle; "
                        "declare a scalar with add_net first."
                    )
            parent_width = val.width if isinstance(val, Net) else 1
            parent_name = val.name if isinstance(val, Net) else val
            if (isinstance(val, Net) and parent_width == 1
                    and spec.width > 1 and val._underlying_pins):
                # A one-bit bus or slice names its bus, not its scalar bit.
                parent_name = val._underlying_pins[0].name
            broadcast = (
                parent_width == 1 and spec.width > 1
                and spec.direction in (PortDir.IN, PortDir.INOUT)
                and isinstance(val, (Net, str)) and not string_bus
            )
            if parent_width != spec.width and not broadcast:
                parent_module = (
                    parent.module_name or parent.circuit_name
                    or parent.netlistname.split('/')[-1]
                )
                raise CircuitError(
                    f"BUS_WIDTH_MISMATCH in parent module {parent_module!r} "
                    f"at element {instance_name!r} port {port!r} "
                    f"({spec.direction.value}): child bus port has width "
                    f"{spec.width}, parent net {parent_name!r} has width "
                    f"{parent_width}"
                )
            if broadcast:
                # Input/inout fanout reuses the scalar net at every bit position.
                for child_bit_name in spec.expanded_names():
                    expanded_wiring[child_bit_name] = parent_name
            elif isinstance(val, Net) and val._underlying_pins is not None:
                for child_bit_name, parent_net in zip(spec.expanded_names(), val):
                    expanded_wiring[child_bit_name] = parent_net.name
            else:
                # A scalar parent on a width-1 bus port connects to the
                # port's single expanded pin (any direction).
                (child_bit_name,) = spec.expanded_names()
                expanded_wiring[child_bit_name] = parent_name
        else:
            # Scalar port — Net → name normalization (current behavior).
            normalized = val.name if isinstance(val, Net) else val
            expanded_wiring[port] = normalized

    return expanded_wiring


def _child_spec_owner_path(owner_path, instance_name):
    from silk.db.identity import _escape_owner_path_segment

    return f"{owner_path}/{_escape_owner_path_segment(instance_name)}"


# ------------------------------------------------------------------
# Port declaration guards (TASK-37). SPICE folds node-name case exactly
# as it does for nets (G16, see ``CircuitGraph._canonical_net_name``), and
# a subcircuit's port list IS a list of node names: ``.subckt casedup Vin
# vin ...`` declares ONE node twice, so ngspice silently drops the second
# parent connection (measured 2026-09-02: i(Vb)=0 on a resistor subckt).
# Nets can be merged onto the first-seen spelling because they are
# internal; ports cannot — merging would change the subcircuit's
# signature and the caller's positional wiring — so the port path raises
# instead of folding.
# ------------------------------------------------------------------

def _port_casefold_index(module):
    """``{casefolded name: declared spelling}`` over the current pins."""
    return {name.casefold(): name for name in (module.pin_list or {})}


def _declare_pin(module, port_name, *, direction, role, domain, casefold_idx):
    """Declare one scalar port pin, enforcing the guards.

    ``casefold_idx`` is the caller-owned ``{casefold: spelling}`` map for
    the pins declared so far; it is updated in place so names introduced
    earlier in the same :meth:`define_module_ports` call collide too.
    """
    where = f"module {module.module_name or '<unnamed>'!r}"

    existing_name = casefold_idx.get(port_name.casefold())
    if existing_name is not None and existing_name != port_name:
        raise DuplicateNameError(
            f"port {port_name!r} differs only by case from already-declared "
            f"port {existing_name!r} in {where}. SPICE folds node-name case "
            f"(ngspice to lower, Xyce to upper), so a subcircuit declaring "
            f"both has ONE node under two port positions and the second "
            f"connection is silently dropped. Rename one of them."
        )

    existing = module.pin_list.get(port_name)
    if existing is not None:
        same = (existing.direction == direction
                and existing.role == role
                and (existing.domain is domain
                     or existing.domain == domain))
        if same:
            # Identical redeclaration: keep the existing pin so an
            # already-wired ``pin.net`` survives instead of being reset.
            return
        raise DuplicateNameError(
            f"port {port_name!r} is already declared in {where} as "
            f"(direction={existing.direction!r}, role={existing.role!r}, "
            f"domain={existing.domain!r}) and would be redeclared as "
            f"(direction={direction!r}, role={role!r}, domain={domain!r}). "
            f"Declare each port once; edit the original declaration "
            f"instead of overwriting it."
        )

    module.pin_list[port_name] = ModulePin(
        name=port_name, direction=direction, role=role, domain=domain
    )
    casefold_idx[port_name.casefold()] = port_name


# ------------------------------------------------------------------
# Lint-marker dedupe guard (TASK-89). ``add_no_connect``/
# ``add_no_connect_port`` are meant to register ONE suppression per
# target; nothing stopped a second call on the same net or (instance,
# port) from appending a second marker. STRUCT-003/011 and STRUCT-001
# only ever check "is there at least one marker", so the duplicate was
# invisible there — but TASK-51's ``find_stale_markers`` is deliberately
# per-marker, so once the target got wired the same pin/net produced TWO
# stale rows instead of one. Mirrors ``_declare_pin``'s identical-vs-
# conflicting split: same reason is a no-op (keep the first marker, same
# object identity other code may hold), different reason raises naming
# both so the caller sees the conflict instead of silently doubling up.
# ------------------------------------------------------------------

def _find_marker(markers, **target):
    """First marker in ``markers`` whose fields match ``target``, else ``None``."""
    for marker in markers:
        if all(getattr(marker, field) == value for field, value in target.items()):
            return marker
    return None


def _dedupe_marker(existing, reason, *, where, marker_cls_name):
    """Apply the dedupe contract given an already-found ``existing``.

    Returns ``existing`` unchanged for an identical reason (no-op — the
    caller must not construct or append a new marker). Returns ``None``
    when there is nothing to dedupe against (caller proceeds normally).
    Raises :class:`DuplicateNameError` naming both reasons on conflict.
    """
    if existing is None:
        return None
    if existing.reason == reason:
        return existing
    raise DuplicateNameError(
        f"{where} already has a {marker_cls_name} marker with reason "
        f"{existing.reason!r}; a second registration with reason "
        f"{reason!r} would silently double the marker (and, once the "
        f"target is wired, double the stale-marker row). Register a "
        f"{marker_cls_name} once per target; edit the existing marker's "
        f"reason instead of adding another."
    )


def _normalize_port_direction(port_name, direction):
    """Normalize a port direction through :meth:`Port._parse_direction`.

    Accepts a :class:`PortDir` or its string spelling (short ``'in'`` /
    ``'out'`` / ``'inout'`` or long ``'input'`` / ``'output'`` /
    ``'power'`` / ``'ground'``, any case). An unrecognized string raises
    ``ValueError``; a non-string, non-``PortDir`` raises ``TypeError`` —
    neither is stored verbatim any more.
    """
    from silk.circuit.circuit_elements import PortDir, Port

    try:
        return Port._parse_direction(direction)
    except ValueError:
        spellings = ['in', 'out', 'inout'] + [d.value for d in PortDir]
        accepted = "', '".join(dict.fromkeys(spellings))
        raise ValueError(
            f"port {port_name!r}: unknown direction {direction!r}. "
            f"Use a PortDir or one of '{accepted}'."
        ) from None
    except TypeError as exc:
        raise TypeError(f"port {port_name!r}: {exc}") from None


# Source terminals that actively drive their net. Direction mirrors the
# *actively-driving* terminal so lint rules can tell apart drivers from
# returns/sensors:
#   positive / out_pos   → 'output'  (actively drives the net)
#   negative / out_neg   → 'input'   (return path, typically gnd)
#   ctrl_pos / ctrl_neg  → 'input'   (sense only, VCVS/VCCS)
# Without this distinction, every source's `negative='gnd'` would be treated
# as a driver on gnd and produce spurious INT-001 multiple-driver reports.
_SOURCE_OUTPUT_TERMINALS = frozenset({'positive', 'out_pos'})


def _bind_source_terminal(module, name, terminal_name, net_name, *, undo=None):
    """Register one source terminal of *module* on the Net side.

    The single place a source terminal becomes a ``kind='source'``
    :class:`PortBinding` — ``add_source`` and ``connect_source`` both funnel
    through here, so a rewire is handled by
    :meth:`PyCircuitElements._bind_pin_to_net` exactly like a device rewire
    (the pin leaves the old ``Net.pins`` and joins the new one). A
    module-level function, not a method: ``Module``'s attribute names are
    frozen by the refactor golden (tests/refactor).
    """
    # Local import: silk.circuit.net imports Module under TYPE_CHECKING to
    # avoid a circular import at module load time.
    from silk.circuit.net import PortBinding
    direction = 'output' if terminal_name in _SOURCE_OUTPUT_TERMINALS else 'input'
    module._element_mgr._bind_pin_to_net(net_name, PortBinding(
        net=net_name,
        instance=name,
        port=terminal_name,
        direction=direction,
        role=None,
        kind='source',
    ), undo=undo)


def _device_add_replacement(device, name, nets, parameters, ports, pdk):
    """Render the factory form without constructing or mutating a device."""
    import keyword
    from silk.circuit.net import Net
    from silk.elements.base import AnalogElement
    from silk.pdk.basepdk import PDKDevice

    if pdk is None or not hasattr(pdk, '_devices'):
        return 'add(device)'

    # An existing instance must keep its identity; add accepts it directly.
    if isinstance(device, AnalogElement):
        return 'add(device)'
    if not (isinstance(device, PDKDevice) or
            isinstance(device, type) and issubclass(device, AnalogElement)):
        return 'add(device)'
    factory = ('pdk.' if isinstance(device, PDKDevice) or
               getattr(device, '_is_pdk_device', False) else 'elements.')
    factory += device.device_name
    try:
        wiring = dict(nets or {})
        wiring.update(ports)
        values = {key: value.name if isinstance(value, Net) else value
                  for key, value in wiring.items()}
        if (parameters and not isinstance(parameters, dict) and
                not isinstance(parameters[0], (tuple, list))):
            # Legacy dispatch uses the target module's catalog template,
            # even when the supplied class came from another PDK load.
            template = pdk._devices.get(device.device_name)
            if template is None:
                return 'add(device)'
            names = template.param_list
            if len(names) != len(parameters):
                return 'add(device)'
            values.update(zip(names, parameters))
        else:
            values.update(parameters or {})
    except (AttributeError, TypeError, ValueError):
        # Leave malformed arguments to the existing dispatch/validation path.
        return 'add(device)'
    args = [repr(name)]
    for key, value in values.items():
        if isinstance(key, str) and key.isidentifier() and not keyword.iskeyword(key):
            args.append(f'{key}={value!r}')
        else:
            args.append(f'**{{{key!r}: {value!r}}}')
    return f"add({factory}({', '.join(args)}))"


class _StructureMethods(_Base):
    """Circuit-building / connectivity methods mixed into ``Module``. Holds no state."""

    # ------------------------------------------------------------------
    # Element-graph read views (B2 boundary). The devices/nets/subckt-def
    # dicts are owned by the CircuitGraph (``self._element_mgr``); Module
    # exposes them as read-only mapping views so all mutation flows through
    # CircuitGraph / the add_*/remove_* methods and cannot silently bypass the
    # ``_bind_pin_to_net`` invariant. Reads (indexing, ``.get``, iteration,
    # ``.items()``) work unchanged; ``module.net_dict[x] = ...`` / ``del`` now
    # raise ``TypeError`` — use ``add_net`` / ``remove_net`` / ``add_element``.
    # ------------------------------------------------------------------
    @property
    def elem_dict(self):
        """Read-only view of the element (device/submodule instance) dict."""
        from silk.circuit.module_placement import sync_instance_pins
        sync_instance_pins(self)
        return MappingProxyType(self._element_mgr.elem_dict)

    @property
    def net_dict(self):
        """Read-only view of the net dict."""
        from silk.circuit.module_placement import sync_instance_pins
        sync_instance_pins(self)
        return MappingProxyType(self._element_mgr.net_dict)

    @property
    def module_dict(self):
        """Read-only view of the subcircuit-definition dict."""
        from silk.circuit.module_placement import refresh_definitions
        refresh_definitions(self)
        return MappingProxyType(self._element_mgr.module_dict)

    def remove_net(self, netname):
        """Remove a net (and its stale pin bindings) via the CircuitGraph.

        Delegates to ``CircuitGraph.remove_net``. Use this instead of
        ``del module.net_dict[name]`` (which now raises — the view is
        read-only). Raises ``KeyError`` if the net is absent.

        Parameters
        ----------
        netname : str
            Name of the net to remove, as stored in ``net_dict``.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If the net is absent.
        """
        return self._element_mgr.remove_net(netname)

    @property
    def nets(self):
        """Attribute-access view of this module's nets.

        ``tb.nets.avdd`` returns the :class:`Net` object for the net
        named ``'avdd'``; missing nets raise ``AttributeError`` at the
        call site, so typos surface before simulation.
        """
        return _NetAccessor(self)

    def define_module_name(self, name: str):
        """Name this module so it can be placed as a subcircuit.

        A module with a ``module_name`` emits a ``.subckt`` block; one
        without is a top-level circuit.

        Parameters
        ----------
        name : str
            The ``.subckt`` identifier: a letter or underscore followed by
            letters, digits and underscores.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``name`` is not a valid SPICE subcircuit name.
        """
        _validate_module_name(name)
        self.module_name = name

    def define_module_ports(self, ports: dict):
        """Declare subcircuit ports with optional direction, role, and width.

        Parameters
        ----------
        ports : dict
            Mapping of port name to one of:

            - :class:`PortDir` — direction only; the role is unset for
              IN / OUT / INOUT and **derived** for POWER / GROUND
              (``PortDir.POWER`` → ``PortRole.POWER``, ``PortDir.GROUND`` →
              ``PortRole.GROUND``, see ``PortDir.implied_role``)
            - a direction string (``'in'``/``'input'``, ``'out'``/``'output'``,
              ``'inout'``, ``'power'``, ``'ground'``), normalized to
              :class:`PortDir` (and the same role derivation applies)
            - ``(PortDir, PortRole)`` tuple — direction and semantic role
            - :class:`silk.circuit.circuit_elements.Port <silk.circuit.Port>` — full descriptor
              (may include ``bits=`` for buses); a ``Port``
              without a ``role`` gets the direction-implied one too
            - ``None`` — defaults to ``PortDir.INOUT``, no role

        Notes
        -----
        An explicitly given role is never overridden by the derivation:
        ``(PortDir.POWER, PortRole.ANALOG)`` is stored as written.

        Formal scalar and bus names use the authored identifier grammar,
        without the actual net's ``0`` exception. Rename a formal ``0`` to
        ``vss`` and wire it to the parent's ground net; actual ``0`` net
        connections remain valid. Formal ``gnd`` (any case) is accepted
        here but refused by lint, emission and static simulation setup.

        Raises
        ------
        DuplicateNameError
            If a port name differs only by case from one already declared
            (SPICE would fold them into a single node), or if a port is
            redeclared with a different direction, role, or domain. An
            identical redeclaration is a no-op.
        ValueError
            If a formal name is invalid (including ``'0'``), or a direction
            string names no known direction.
        TypeError
            If a formal name is not a string or Net-like name, or a direction
            is neither a :class:`PortDir` nor a string. Integer names are
            not coerced to strings.
        """
        from silk.circuit.circuit_elements import PortDir, Port
        from silk.circuit.bus_spec import (
            BusSpec, indices_from_bits, expand_bits, _bus_claims, _bus_collision,
        )

        if self.pin_list is None:
            self.pin_list = {}

        casefold_idx = _port_casefold_index(self)
        claims = _bus_claims(self._buses)

        from silk.elements.base import validate_net_token
        from silk.circuit.ground_port import (
            block_name, _is_global_ground_port, unportable_ground_port_message,
        )
        for port_name, value in ports.items():
            if _is_global_ground_port(port_name):
                raise ValueError(unportable_ground_port_message(block_name(self), port_name))
            validate_net_token(port_name, who='define_module_ports', kind='port')
            # Resolve to (direction, role, domain, bits, naming)
            if isinstance(value, Port):
                direction = value.dir
                role = value.role
                domain = value.domain
                bits = value.bits
                naming = value.naming
            elif isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(
                        f"Port '{port_name}': expected (PortDir, PortRole) 2-tuple, "
                        f"got {len(value)}-element tuple"
                    )
                direction, role = value
                domain = None
                bits = None
                naming = 'bracketed'
            else:
                direction = value
                role = None
                domain = None
                bits = None
                naming = 'bracketed'

            if direction is None:
                direction = PortDir.INOUT

            direction = _normalize_port_direction(port_name, direction)

            # TASK-45 contract: a supply *direction* with no stated role
            # carries the supply role. Explicit roles are left untouched.
            if role is None:
                role = direction.implied_role

            if bits is None:
                owner = claims.get(port_name.casefold())
                if owner is not None:
                    raise _bus_collision(self, owner[0], port_name, owner[1])
                # Scalar port — declared through the guarded chokepoint.
                _declare_pin(self, port_name, direction=direction, role=role,
                             domain=domain, casefold_idx=casefold_idx)
            else:
                # Bus port — expand to scalar pins and record BusSpec.
                indices = indices_from_bits(bits)
                expanded = expand_bits(port_name, bits, naming=naming)
                domain_name = (
                    domain.name if domain is not None and hasattr(domain, 'name')
                    else (domain if isinstance(domain, str) else None)
                )
                spec = BusSpec(port_name, indices, direction, role, domain_name, naming)
                existing = self._buses.get(port_name)
                if existing == spec and all(
                        bit in self.pin_list and
                        self.pin_list[bit].direction == direction and
                        self.pin_list[bit].role == role and
                        self.pin_list[bit].domain == domain
                        for bit in expanded):
                    continue
                adopting = (
                    existing is not None and existing.name == port_name
                    and existing.indices == indices and existing.naming == naming
                    and not any(bit in self.pin_list
                                for bit in existing.expanded_names())
                )
                # Adoption exposes this exact internal bus, so only its own
                # claims are exempt. Formal pins and other owners still refuse.
                # Validate the whole declaration before installing any pin.
                for name in (port_name, *expanded):
                    validate_net_token(name, who='define_module_ports', kind='port')
                    owner = claims.get(name.casefold())
                    if owner is not None and not (adopting and owner[0] is existing):
                        raise _bus_collision(self, owner[0], name, owner[1])
                    clash = casefold_idx.get(name.casefold())
                    if clash is not None:
                        raise DuplicateNameError(
                            f"bus {port_name!r} expansion collides with existing pin "
                            f"{clash!r} in module {self.module_name or '<unnamed>'!r}; "
                            f"conflicting declaration {name!r} claims a bus bit or alias."
                            + (f" {name!r} differs only by case from {clash!r}."
                               if name != clash else '')
                        )
                alias = next((name for name in self.net_dict
                              if name.casefold() == port_name.casefold()), None)
                if alias is not None:
                    raise _bus_collision(self, spec, alias, port_name)
                for bit_name in expanded:
                    _declare_pin(self, bit_name, direction=direction, role=role,
                                 domain=domain, casefold_idx=casefold_idx)
                self._buses[port_name] = spec
                claims.update(_bus_claims({port_name: spec}))
                if adopting:
                    # The existing handle already owns the materialized bits.
                    continue
                # A port declaration must not add unused graph nets: that
                # changes structure hashes and floating-net diagnostics.
                # Keep scalar Net objects in the handle until first use.
                from silk.circuit.net import Net
                underlying = tuple(
                    self.net_dict.get(self._element_mgr._canonical_net_name(bit))
                    or Net(name=bit, module=self) for bit in expanded
                )
                self._bus_handles[port_name] = Net._make_bus_handle(
                    name=port_name, underlying=underlying,
                    index_vector=tuple((port_name, i) for i in indices),
                )

    def define_module_nets(self, pin_dict):
        """Backward-compatible port declaration — all ports default to INOUT.

        Prefer :meth:`define_module_ports` for new code.

        Parameters
        ----------
        pin_dict : iterable of str
            Port names; each is declared ``PortDir.INOUT`` with no role.

        Returns
        -------
        None
        """
        from silk.circuit.circuit_elements import PortDir
        self.define_module_ports({p: PortDir.INOUT for p in pin_dict})

    def assign_net_domain(self, net_name: str, domain) -> None:
        """Tag an internal net with a voltage domain.

        Used at the parent/top level to declare which supply domain a net
        belongs to. The checker uses this for domain propagation and
        RAIL_MISMATCH detection.

        Writes to the resolved Net's domain, including through case aliases.
        If the net does not exist yet, it is created (honoring
        ``strict_nets`` via ``_ensure_net``).

        Parameters
        ----------
        net_name : str
            Name of the internal net.
        domain : VoltageDomain
            Domain to assign.

        Raises
        ------
        ValueError
            If the net is already assigned to an incompatible domain.
        """
        # Ensure the net exists (honors strict_nets).
        if net_name not in self.net_dict:
            net = self._element_mgr._ensure_net(net_name)
        else:
            net = self.net_dict[net_name]
        if net.domain is not None and not net.domain.is_compatible(domain):
            raise ValueError(
                f"Net '{net_name}': incompatible domain assignment. "
                f"Existing: '{net.domain.name}' ({net.domain.nominal}V), "
                f"New: '{domain.name}' ({domain.nominal}V)"
            )
        net.domain = domain
        if domain not in self._domains:
            self._domains.append(domain)

    # ===== UPDATED: Delegate to PyCircuitElements =====
    def add(self, *devices):
        """Add one or more factory-created elements to the module — an
        ``AnalogElement`` device (including a shimmed ``silk.elements.passives``
        instance), or a source (``VoltageSource``/``CurrentSource``,
        incl. the ``sources.vpulse(...)``-style factory output).

        Idempotent: re-adding the same instance is silently skipped. A
        different instance sharing the same name still raises.

        Raises
        ------
        TypeError
            *device* is neither a source nor an ``AnalogElement`` — named
            explicitly rather than surfacing as an ``AttributeError`` from
            deeper inside element registration.
        """
        from silk.circuit.device_view import _refuse_device_view
        for candidate in devices:
            _refuse_device_view(candidate, "Module.add()")
        from silk.sources import SourceBase
        from silk.elements.base import AnalogElement
        for device in devices:
            if isinstance(device, SourceBase):
                # A source is testbench stimulus — route to add_source (which is
                # itself idempotent). Checked BEFORE the elem_dict probe because
                # a source has no `.name` entry there and no _pin_names/nets for
                # _add_analog_element. Does NOT merge the dicts.
                self.add_source(device)
                continue
            if not isinstance(device, AnalogElement):
                raise TypeError(
                    f"Module.add() accepts an AnalogElement (a factory-created "
                    f"device or ideal element) or a source (VoltageSource/"
                    f"CurrentSource); got {type(device).__name__}."
                )
            validate_instance_name(device.name)
            entry = self._element_mgr.elem_dict.get(device.name)
            if entry is not None and entry.get('device') is device:
                continue
            self._element_mgr._add_analog_element(device)

    def add_element(self, elemDevice, elemName, nets=None, parameters=None,
                    **port_kwargs):
        """Deprecated alias — use :meth:`add_module` (a Module target) or
        :meth:`add` (a device target) instead.

        Kept with today's exact catch-all behaviour (device classes, device
        instances, legacy PDK devices, and Modules all still dispatch the
        same way); only a :class:`DeprecationWarning` naming the right
        replacement is new.

        See ``_add_element_impl`` for the full behaviour this delegates
        to unchanged.

        Parameters
        ----------
        elemDevice : Module or object
            The module to place, or a device class or instance.
        elemName : str
            Instance name.
        nets : dict, optional
            ``{pin: net}``; values may be :class:`Net` objects or names.
        parameters : dict, optional
            Device parameter values.
        **port_kwargs
            Port wiring as keywords, equivalent to ``nets``.

        Returns
        -------
        SubcircuitInstance or None
            A handle when ``elemDevice`` is a Module, else ``None``.

        Raises
        ------
        NotASubcircuitError
            If a Module target has no ``module_name`` or no declared ports.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(elemDevice, "add_element()")
        import warnings
        from silk.circuit.modules import Module
        target = ('add_module(module, name, nets=None, **ports)'
                   if isinstance(elemDevice, Module) else
                   _device_add_replacement(elemDevice, elemName, nets,
                                           parameters, port_kwargs, self.pdk_details))
        warnings.warn(
            f"Module.add_element() is deprecated; use {target} instead.",
            DeprecationWarning, stacklevel=2)
        return self._add_element_impl(elemDevice, elemName, nets=nets,
                                      parameters=parameters, **port_kwargs)

    def add_module(self, module, name, nets=None, **ports):
        """Place an instance of *module* — a hierarchy-verb sibling of
        :meth:`ModuleTestbench.add_module` (module_test.py), sharing this
        method's underlying implementation.

        ``module`` must be a :class:`~silk.circuit.modules.Module` with a
        ``module_name`` and declared ports (:meth:`define_module_name`,
        :meth:`define_module_ports`) — refused at the call with
        :class:`~silk.exceptions.NotASubcircuitError` otherwise.
        Port wiring is given as ``nets={'pin': net, ...}`` and/or
        ``**ports`` keyword form, exactly as :meth:`add_element` accepted
        it for a Module target.

        A parent bus name or bus handle expands bit-wise into a child bus
        port; different widths raise ``BUS_WIDTH_MISMATCH``. Bus ports
        declared with ``define_module_ports`` are available as ``self.nets``
        handles, without needing ``add_net`` first. A declared scalar net
        or scalar port, passed as a string or Net, broadcasts to every bit
        of an input or inout bus (for example, ``nets={'P': 'vss'}`` ties
        all port bits to ``vss``). Scalar connections to output buses still
        raise ``BUS_WIDTH_MISMATCH``, because tying output bits together
        would short them. An undeclared string for a multi-bit
        port raises ``UndeclaredNetError``; declare a scalar tie-off with
        ``add_net`` or pass a declared bus. A bare bus-port name cannot be
        used as a scalar device connection: select one of its bits.

        Returns
        -------
        SubcircuitInstance
            Handle onto the placed instance's internal nets and nested
            sub-instances.

        Raises
        ------
        TypeError
            *module* is not a Module — named explicitly rather than
            surfacing deep inside element registration; use :meth:`add` for
            a device.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(module, "add_module()")
        from silk.circuit.modules import Module
        if not isinstance(module, Module):
            raise TypeError(
                f"Module.add_module() accepts a Module (a subcircuit "
                f"instance); got {type(module).__name__}. Use add() to "
                f"place a device.")
        return self._add_element_impl(module, name, nets=nets, **ports)

    def _add_element_impl(self, elemDevice, elemName, nets=None,
                          parameters=None, **port_kwargs):
        """Add element to module — delegates to PyCircuitElements.

        The shared implementation behind both :meth:`add_element` (the
        deprecated catch-all alias) and :meth:`add_module` (Module-only).
        Port wiring can be specified in two equivalent ways; the kwarg
        form is preferred because typos become ``TypeError``/
        ``AttributeError`` at the call site instead of dangling-net
        errors at simulate time::

            # Preferred — port names as kwargs, nets as Net objects
            avdd = tb.add_net('avdd')
            fcamp_inst = tb.add_element(fcamp, 'fcamp',
                vinp=tb.nets.vinp, vout=tb.nets.vout,
                avdd_1p8=avdd,     gnd=tb.nets.gnd)

            # Escape hatch for reserved words / digit-starting port names
            tb.add_element(sub, 'u1', **{'in': tb.nets.a, '1p8': avdd})

            # Legacy dict form — still works
            tb.add_element(fcamp, 'fcamp', nets={'vinp': 'vinp', ...})

        Values in ``nets=`` or ``**port_kwargs`` may be either a
        :class:`~silk.circuit.net.Net` object or a net-name string;
        Net objects are resolved via ``.name``.

        When *elemDevice* is a Module (subcircuit instance), returns a
        :class:`SubcircuitInstance` handle so the caller can reach
        internal nets and nested sub-instances as attributes::

            fcamp_inst.vbiasn              # → Signal
            fcamp_inst.vbiasn.y[-1]        # final value

        For raw devices (non-Module), returns ``None`` for backward
        compatibility.

        Raises:
            NotASubcircuitError: If *elemDevice* is a Module with no
                ``module_name`` (a top-level circuit has no ``.subckt`` to
                reference), or if it has no declared ports at all and
                ``nets=``/port kwargs were given to wire it anyway. Refused
                here, where the mistake is made, rather than at the parent's
                ``generate_netlist()`` much later. A portless
                Module wired with nothing (e.g. a behavioral-model subcircuit
                with no pins) is unaffected.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(elemDevice, "_add_element_impl()")
        # Module imports this part, so resolve the class at call time.
        from silk.circuit.modules import Module
        if isinstance(elemDevice, Module):
            # Preflight before bus expansion can canonicalize parent nets.
            tokens = self._element_mgr._check_instance_name('elem_dict', elemName, elemDevice)
            if tokens is None:
                entry = self._element_mgr.elem_dict[elemName]
                return SubcircuitInstance(
                    root_tb=self, spice_name=emitted_submodule_name(elemName),
                    subcircuit_def=elemDevice,
                    port_map={p: pin.net for p, pin in (entry['pin_list'] or {}).items()},
                    parent=self)

        if not nets:
            wiring = {}
        elif isinstance(nets, dict):
            wiring = dict(nets)
        elif isinstance(nets, (list, tuple)) and isinstance(nets[0], (tuple, list)) \
                and len(nets[0]) == 2:
            # List of (pin, net) pairs — unambiguous, dict() handles it.
            wiring = dict(nets)
        else:
            # A bare positional list of net values (``nets=['out', 'in', ...]``).
            # This method never had pin-order information to zip it against
            # (unlike PyCircuitElements.add_element / _apply_connections,
            # which read the pin order off the constructed device) — it fed
            # straight into dict(nets), which for a list of plain strings
            # either crashed (measured: dict(['out', 'in', 'gnd', 'gnd']) ->
            # "ValueError: dictionary update sequence element #0 has length
            # 3; 2 is required") or, worse, silently mispaired 2-character
            # net names into a wrong {pin: net} map (dict(['ab', 'cd']) ==
            # {'a': 'b', 'c': 'd'}) instead of raising at all. Reject with a
            # message pointing at the two forms this method documents
            # (TASK-43 AC #2) rather than support a third, unadvertised one.
            raise CircuitError(
                f'ERROR: add_element({elemName!r}): positional net lists are '
                f'not supported here (got nets={nets!r}). Use nets='
                f'{{"pin": "net", ...}} or pass ports as keyword arguments, '
                f'e.g. add_element(dev, {elemName!r}, pin1=net1, pin2=net2).'
            )
        # Kwargs override dict entries when both forms are given.
        wiring.update(port_kwargs)

        # ------------------------------------------------------------------
        # Bus-aware expansion
        # ------------------------------------------------------------------
        # When the target is a Module with declared bus ports, expand any
        # matching kwarg from a bus-handle Net into N scalar kwargs.
        # Scalar kwargs and non-Module targets pass through unchanged.
        expanded_wiring = _resolve_port_mapping(self, elemDevice, elemName, wiring)

        # ------------------------------------------------------------------
        # Dispatch to element manager
        # ------------------------------------------------------------------
        # For Module subcircuit devices, the old-style path in
        # _element_mgr.add_element accesses pdk_details.model_names which
        # crashes when pdk_details is None. Handle Module→Module wiring
        # directly here to avoid that dependency.
        #
        validate_instance_name(elemName)
        if isinstance(elemDevice, Module):
            from silk.circuit.circuit_elements import (
                _apply_connections, _apply_parameters, _validate_connections)
            from silk.circuit.modules_netlist import require_composable_child
            # A top-level circuit (no module_name) cannot become a child
            # here — refuse at the call that makes the mistake rather than at
            # the parent's much-later generate_netlist() (TASK-175). A
            # Module with no declared ports at all is a legitimate zero-pin
            # (e.g. behavioral-model) subcircuit as long as nothing is wired
            # to it; only refuse that shape when wiring was actually passed,
            # which is what used to reach connect_element's pinlist.keys()
            # and raise a bare AttributeError.
            from silk.circuit.module_placement import check_placement
            check_placement(self, elemDevice)
            require_composable_child(elemDevice, elemName,
                                      require_ports=bool(expanded_wiring))
            # Every port value has to survive as one node token — checked
            # before the instance is registered and before the submodule's
            # .subckt definition is recorded, so a rejected port map leaves
            # nothing behind (TASK-75 AC #1).
            if expanded_wiring:
                _validate_connections(elemDevice, elemName, expanded_wiring,
                                      list(elemDevice.pin_list or {}))
            if self.debug:
                print('Element Name found in custom module list')
            # Track module definition (dedupe by name; fail loud on a
            # same-name/different-body collision — the shared chokepoint in
            # circuit_elements.py). The return value lets the except-block
            # below undo exactly this call if the binding step further down
            # rejects it (TASK-98).
            from silk.circuit.circuit_elements import register_module_definition
            module_registration = register_module_definition(
                self._element_mgr.module_dict, elemDevice)
            assigned_name = module_registration[0]

            parent_owner_path = getattr(self, '_spec_owner_path', '$')
            self._element_mgr.elem_dict[elemName] = {
                'type': ElementType.MODULE,
                'device': elemDevice,
                'typeName': elemDevice.device_name,
                'pin_list': copy.deepcopy(elemDevice.pin_list),
                'module_name': assigned_name,
                'spec_owner_path': _child_spec_owner_path(parent_owner_path, elemName),
            }
            undo = []
            try:
                if expanded_wiring:
                    _apply_connections(self._element_mgr, elemName, expanded_wiring, undo=undo)
                if parameters is not None:
                    _apply_parameters(self._element_mgr, elemName, parameters)
            except Exception:
                # A failure partway through (e.g. an undeclared net under
                # strict_nets) used to leave the instance registered with
                # whichever ports had bound before the failing one
                # (measured: elem_dict['u1'] stayed with vin='a' bound and
                # gnd=None). Purge it so the call leaves no partial
                # instance behind (TASK-43 AC #4).
                self._element_mgr._purge_partial_instance('elem_dict', elemName, elemDevice, undo)
                # The submodule's .subckt definition registered just above
                # used to stay in module_dict either way — a harmless
                # unused definition if this was its only use, but a
                # collision waiting to happen if a later, structurally
                # different module reused the same module_name (TASK-98).
                # Undo exactly what that call did: purge a freshly created
                # entry outright, or roll a deduped one back to what it was
                # before this call.
                from silk.circuit.circuit_elements import (
                    unregister_module_definition)
                unregister_module_definition(
                    self._element_mgr.module_dict, elemDevice,
                    module_registration)
                raise

            self._element_mgr._register_instance('elem_dict', elemName, elemDevice, tokens)

            # The handle reads the live definition through placement-local wiring.
            entry = self._element_mgr.elem_dict[elemName]
            stored = entry['device']
            pins = entry['pin_list'] or {}
            port_map = {pin_name: pin.net for pin_name, pin in pins.items()}
            return SubcircuitInstance(
                root_tb=self,
                spice_name=emitted_submodule_name(elemName),
                subcircuit_def=stored,
                port_map=port_map,
                parent=self,
            )

        # Non-Module device — use existing path (Net names already resolved).
        self._element_mgr.add_element(elemDevice, elemName,
                                      expanded_wiring or None, parameters)
        return None

    def iter_instances(self, *, include_self=False):
        """Yield immutable ModuleInstance records in depth-first placement order.

        Records expose ``path`` (user instance path), ``spice_path`` (rawfile
        instance path), ``module`` (live
        definition), ``owner_path``, ``module_name`` (resolved name), and
        ``placed`` (False for the root and legacy organizational registrations).
        Repeated placements yield distinct records pointing at the same
        module. With include_self=True the root has empty path and spice_path
        and owner_path '$'.
        This does not run tests or simulate the circuit.
        """
        from silk.circuit.module_placement import walk_instances
        return walk_instances(self, include_self=include_self)

    def add_net(self, netname, parameters=None, classification=None, *,
                bits=None, naming='bracketed'):
        """Add a net to the module.

        Delegates to ``PyCircuitElements.add_net``, which creates a
        :class:`Net` object (or is a no-op if the net already exists).

        Parameters
        ----------
        netname : str
            Name of the net to add. For ``bits != None``, this is the logical
            bus name; the per-bit nets are named via ``naming``.
        parameters : list, optional
            Legacy parameter list, stored on ``Net.parameters``.
        classification : PortRole, optional
            Explicit classification override. See
            :attr:`silk.circuit.net.Net.classification <silk.circuit.Net.classification>` for the
            auto-inference rules that apply when this is ``None``.
        bits : int, range or slice, optional
            If provided, declare a multi-bit net. Returns an aggregating
            bus-handle :class:`Net` covering N scalar Nets (one per bit).
            Bit names follow ``naming`` ('bracketed' → ``d[3]``;
            'underscore' → ``d_3``). A bus named like an existing port bus
            must include every port bit, and uses the port's naming style.
            The internal handle covers the requested bits; the formal port
            retains its own indices, direction, role and domain.
        naming : str, default 'bracketed'
            How to expand bit names. 'bracketed' or 'underscore'.

        Returns
        -------
        Net
            For scalar (bits=None): a regular scalar Net (width=1).
            For bus: an aggregating bus-handle Net (width=N) over the
            underlying scalar Nets.

        Raises
        ------
        silk.exceptions.CircuitError
            If ``bits`` omits any bit of a port bus of the same name.
        DuplicateNameError
            If a bus name or one of its bit names collides with an existing
            net, port or bus.
        TypeError
            If ``bits`` has an unsupported type (for example a float).
        ValueError
            If ``netname`` is not a single valid net token, or ``bits`` is an
            invalid value of an accepted type (for example a nonpositive
            width, or an empty, non-unit-step or unbounded range or slice),
            or ``naming`` is invalid.
        """
        from silk.circuit.bus_spec import (
            indices_from_bits, expand_bits, _format_pin_name,
        )
        from silk.circuit.net import Net
        from silk.elements.base import validate_net_token

        # Normalise (and reject) here as well as in the element manager: the
        # scalar branch re-reads `netname` through _canonical_net_name after
        # delegating, and the bus branch expands it into bit names — both want
        # a plain, single-token string (TASK-75 AC #4).
        netname = validate_net_token(netname, who='add_net', kind='net')

        if bits is None:
            # Scalar — unchanged behavior. The lookup goes through the case
            # canonicalizer: a name differing only by case from an existing
            # net merges onto it (G16), so 'vdd' after 'VDD' returns the
            # existing 'VDD' Net rather than KeyError-ing.
            self._element_mgr.add_net(netname, parameters, classification)
            return self.net_dict[self._element_mgr._canonical_net_name(netname)]

        indices = indices_from_bits(bits)

        # Keep the formal port BusSpec separate from the internal handle,
        # which may cover additional bits without exposing them as pins.
        declared = self._buses.get(netname)
        if declared is not None and all(
                bit in (self.pin_list or {}) for bit in declared.expanded_names()):
            if not set(declared.indices).issubset(indices):
                raise CircuitError(
                    f"add_net({netname!r}, bits=...) has indices "
                    f"{tuple(indices)} but port bus {netname!r} in module "
                    f"{self.module_name or '<unnamed>'!r} is declared with "
                    f"indices {tuple(declared.indices)} "
                    f"({declared.direction.value}); an internal bus named like a "
                    f"port bus must include every port bit"
                )
            bit_names = tuple(_format_pin_name(netname, i, declared.naming)
                              for i in indices)
            underlying = []
            for bit_name in bit_names:
                bit_name = self._element_mgr._canonical_net_name(bit_name)
                if bit_name not in self.net_dict:
                    self._element_mgr.add_net(bit_name, parameters,
                                              classification)
                underlying.append(self.net_dict[bit_name])
            existing = self._bus_handles.get(netname)
            index_vector = tuple((netname, i) for i in indices)
            if existing is not None and existing._index_vector == index_vector:
                return existing
            handle = Net._make_bus_handle(
                name=netname,
                underlying=tuple(underlying),
                index_vector=index_vector,
            )
            self._bus_handles[netname] = handle
            return handle

        expanded = expand_bits(netname, bits, naming=naming)

        from silk.circuit.bus_spec import BusSpec, _bus_claims, _bus_collision
        from silk.circuit.circuit_elements import PortDir
        spec = BusSpec(netname, indices, PortDir.INOUT, None, None, naming)
        if self._buses.get(netname) == spec:
            return self._bus_handles[netname]
        claims = _bus_claims(self._buses)
        scalar_names = {name.casefold(): name for name in
                        (*self.net_dict, *(self.pin_list or {}))}
        for name in (netname, *expanded):
            validate_net_token(name, who='add_net', kind='net')
            owner = claims.get(name.casefold())
            if owner is not None:
                raise _bus_collision(self, owner[0], name, owner[1])
            clash = scalar_names.get(name.casefold())
            if clash is not None:
                raise _bus_collision(self, spec, clash, name)

        underlying = []
        for bit_name in expanded:
            self._element_mgr.add_net(bit_name, parameters, classification)
            underlying.append(self.net_dict[bit_name])

        # Internal buses use the same metadata as ports, with INOUT direction.
        self._buses[netname] = spec

        handle = Net._make_bus_handle(
            name=netname,
            underlying=tuple(underlying),
            index_vector=tuple((netname, i) for i in indices),
        )
        self._bus_handles[netname] = handle
        return handle

    def add_no_connect(self, net_name: str, reason: str) -> 'NoConnect':
        """Register a NoConnect lint marker on the given net.

        Suppresses both STRUCT-003 FLOATING_NET (a net with zero
        connections) and STRUCT-011 SINGLE_CONNECTION_NET (a net with
        exactly one) for the net, by attaching a PortBinding with
        ``kind='lint_marker'``. The reason string is required and surfaces
        in the lint report.

        A second call on the same ``net_name`` is a no-op that
        returns the already-registered marker when ``reason`` matches
        exactly; a differing ``reason`` raises ``DuplicateNameError``
        naming both, since silently doubling the marker also doubles its
        row in ``find_stale_markers`` once the net is wired.

        Returns the NoConnect instance for optional caller bookkeeping.
        """
        from silk.lint_markers import NoConnect
        from silk.circuit.net import PortBinding
        existing = next((nc for _inst, net, nc in self._no_connects
                         if net == net_name), None)
        deduped = _dedupe_marker(existing, reason, where=f"net {net_name!r}",
                                 marker_cls_name='NoConnect')
        if deduped is not None:
            return deduped
        nc = NoConnect(reason=reason)
        instance_name = f"_nc_{len(self._no_connects)}"
        self._no_connects.append((instance_name, net_name, nc))
        self._element_mgr._bind_pin_to_net(net_name, PortBinding(
            net=net_name,
            instance=instance_name,
            port='tie',
            direction=None,
            role=None,
            domain=None,
            kind='lint_marker',
        ))
        return nc

    def add_no_connect_port(self, instance: str, port: str,
                            reason: str):
        """Register a ``NoConnectPort`` marker on a submodule-instance port.

        Suppresses STRUCT-001 DANGLING_PORT for ``(instance, port)``. The
        ``reason`` string is required and surfaces in the lint report.

        Use when a submodule port is deliberately left unwired in this
        parent (optional features tied off elsewhere, unused auxiliary
        outputs, debug taps wired only in a separate testbench). Prefer
        this over adding a glob to ``DanglingPortRule.ignore_patterns`` —
        the per-port marker forces documentation and is robust to later
        port renames at the subcircuit level.

        Bus support: if ``port`` names a bus on the instance's
        declared module, this fans out to one ``NoConnectPort`` per bit
        and returns a ``list`` of markers. Per-bit names like ``'d[3]'``
        continue to work and target a single bit.

        A second call on the same ``(instance, port)`` — including a bit
        already covered by a prior whole-bus call — is a no-op
        that returns the already-registered marker when ``reason`` matches
        exactly; a differing ``reason`` raises ``DuplicateNameError``
        naming both, since silently doubling the marker also doubles its
        row in ``find_stale_markers`` once the pin is wired.

        Parameters
        ----------
        instance : str
            Submodule instance name in this module (must be present in
            ``elem_dict``). Validated to catch typos early.
        port : str
            Port name on that submodule (validated against the
            instance's ``pin_list``). May also be a bus name if the
            instance's module declares a bus on that name.
        reason : str
            Non-empty explanation of why this port is deliberately
            unwired.

        Returns
        -------
        NoConnectPort or list of NoConnectPort
            Single marker for a scalar/per-bit port; list of markers for
            whole-bus form.

        Raises
        ------
        CircuitError
            If ``instance`` is not in ``elem_dict`` or ``port`` is not
            a declared port on that instance (raised as ``CircuitError``).
        ValueError, TypeError
            From the NoConnectPort constructor on empty/non-string
            ``reason``.
        """
        from silk.lint_markers import NoConnectPort
        entry = self.elem_dict.get(instance)
        if entry is None:
            raise CircuitError(
                f"add_no_connect_port: instance {instance!r} not found in "
                f"module {self.module_name or '<top-level>'!r}. Known "
                f"instances: {sorted(self.elem_dict)}.")
        device = entry.get('device')
        pin_list = getattr(device, 'pin_list', None) or {}
        from silk.circuit.bus_spec import _formal_bus, _port_targets

        # Whole-bus form: `port` matches a bus declared on the child
        if _formal_bus(device, port) is not None:
            markers = []
            for bit_name in _port_targets(device, port):
                existing = _find_marker(self._no_connect_ports,
                                        instance=instance, port=bit_name)
                deduped = _dedupe_marker(
                    existing, reason, where=f"port {instance}.{bit_name}",
                    marker_cls_name='NoConnectPort')
                if deduped is not None:
                    markers.append(deduped)
                    continue
                marker = NoConnectPort(instance=instance, port=bit_name, reason=reason)
                self._no_connect_ports.append(marker)
                markers.append(marker)
            return markers

        # Per-bit / scalar form (existing behavior)
        if port not in pin_list:
            raise CircuitError(
                f"add_no_connect_port: port {port!r} not declared on "
                f"instance {instance!r}. Declared ports: "
                f"{sorted(pin_list)}.")
        existing = _find_marker(self._no_connect_ports, instance=instance, port=port)
        deduped = _dedupe_marker(existing, reason, where=f"port {instance}.{port}",
                                 marker_cls_name='NoConnectPort')
        if deduped is not None:
            return deduped
        marker = NoConnectPort(instance=instance, port=port, reason=reason)
        self._no_connect_ports.append(marker)
        return marker

    def declare_half_circuit(self, name, left, right, strategy=None):
        """Register a half-circuit matching declaration.

        Parameters
        ----------
        name : str
            Unique identifier within this module.
        left : sequence of str
            Instance names of devices on the left branch (must exist in elem_dict).
        right : sequence of str
            Instance names of devices on the right branch (same length as left).
        strategy : str, optional
            Reserved: 'common_centroid', 'interdigitated', etc. Ignored in Phase 1.

        Returns
        -------
        HalfCircuit

        Raises
        ------
        ValueError
            If name is already declared, or any member is not in elem_dict.
        """
        from silk.layout.matching import HalfCircuit
        if name in self.half_circuits:
            raise ValueError(
                f"HalfCircuit {name!r} already declared on module {self.module_name!r}"
            )
        all_members = list(left) + list(right)
        for m in all_members:
            if m not in self.elem_dict:
                raise ValueError(
                    f"HalfCircuit {name!r}: member {m!r} not in elem_dict"
                )
        hc = HalfCircuit(
            name=name,
            left=tuple(left),
            right=tuple(right),
            strategy=strategy,
        )
        self.half_circuits[name] = hc
        return hc

    def declare_match_pair(self, name, a, b):
        """Sugar for a 1-per-side HalfCircuit.

        Parameters
        ----------
        name : str
            Unique identifier for this matched pair.
        a : str
            Instance name of the left device.
        b : str
            Instance name of the right device.

        Returns
        -------
        HalfCircuit
        """
        return self.declare_half_circuit(name=name, left=[a], right=[b])

    def connect_module(self, name, nets=None, **ports):
        """Rewire selected ports of an existing module placement atomically.

        Wiring uses the same dict, keyword, Net and bus forms as
        :meth:`add_module`. Omitted ports retain their connections. Net names
        belong to this parent module; a Net is reduced to its name. A refused
        call leaves wiring and live declaration metadata unchanged.

        Only the named placement is rewired. Held, unbound placement
        references follow the new wiring; a probe already bound to a net
        keeps that net. Use pin assignment for devices.

        Raises
        ------
        CircuitError
            The name is not a module placement or a port is unknown.
        UndeclaredNetError
            A target net is undeclared under the shared wiring rules.
        """
        from silk.circuit.circuit_elements import (
            _validate_connections, _apply_connections,
        )
        from silk.circuit.module_placement import (
            _placement_wiring_transaction, sync_instance_pins,
        )

        manager = self._element_mgr
        entry = manager.elem_dict.get(name)
        if entry is None or entry['type'] != ElementType.MODULE:
            placements = [key for key, value in manager.elem_dict.items()
                          if value['type'] == ElementType.MODULE]
            advice = (' Use device pin assignment (dev.p = net).'
                      if entry is not None else '')
            raise CircuitError(
                f"{name!r} is not a module placement in this module. "
                f"Available placements: {placements!r}.{advice}")
        definition = entry['device']
        if not nets:
            wiring = {}
        elif isinstance(nets, dict):
            wiring = dict(nets)
        elif (isinstance(nets, (list, tuple))
              and isinstance(nets[0], (tuple, list)) and len(nets[0]) == 2):
            wiring = dict(nets)
        else:
            raise CircuitError(
                f"connect_module({name!r}): positional net lists are not supported. "
                "Use nets={'port': net} or pass ports as keyword arguments.")
        wiring.update(ports)
        with _placement_wiring_transaction(self) as undo:
            expanded = _resolve_port_mapping(self, definition, name, wiring)
            declared = definition.pin_list or {}
            for port in expanded:
                if port not in declared:
                    raise CircuitError(
                        f"Unknown port {port!r} on module placement {name!r}. "
                        f"Declared ports: {list(declared)!r}.")
            _validate_connections(definition, name, expanded, list(declared))
            for value in expanded.values():
                manager._accept_net(value)
            sync_instance_pins(self)
            _apply_connections(manager, name, expanded, undo=undo)

    def update_element(self, elemName, params, paramValues, _skipCheck=False):
        """Update parameter values of a placed element.

        Parameters
        ----------
        elemName : str
            Name of an element already added to the module.
        params : list of str
            Parameter names.
        paramValues : list
            New values, one per name. Numbers are wrapped in a
            :class:`Variable`; a Variable is stored as given.
        _skipCheck : bool, default False
            Internal; skips the existence check.

        Returns
        -------
        None

        Raises
        ------
        silk.exceptions.CircuitError
            If the element does not exist, ``params`` and ``paramValues``
            differ in length, or the element is a module rather than a
            device.
        """
        return self._element_mgr.update_element(elemName, params, paramValues, _skipCheck)

    def add_passive(self, passive):
        """Add a passive element (Resistor, Capacitor, or Inductor) to the module.

        Each terminal funnels through ``PyCircuitElements._bind_pin_to_net``
        so that ``Net.pins`` reflects passive connectivity with ``kind='passive'``.

        Accepts either a legacy ``PassiveBase`` instance or the ideal two-terminal ``AnalogElement`` that ``silk.elements.passives.
        Resistor/Capacitor/Inductor`` now build under the hood — recognised by
        ``_is_ideal_passive`` (SPICE prefix R/C/L), the same test
        ``silk.circuit.modules_results`` uses for current annotation.

        Parameters
        ----------
        passive : object
            A ``Resistor``, ``Capacitor`` or
            ``Inductor`` instance, or the ideal ``AnalogElement`` it builds.

        Returns
        -------
        None
            Re-adding an instance with the same name returns without change.

        Raises
        ------
        silk.exceptions.CircuitError
            If ``passive`` is not a passive.
        DuplicateNameError
            If a different element already uses the name.
        UndeclaredNetError
            If a terminal net is undeclared and ``strict_nets`` is set.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(passive, "add_passive()")
        from silk.elements.passives import PassiveBase
        from silk.circuit.modules_results import _is_ideal_passive
        from silk.circuit.net import PortBinding
        is_legacy = isinstance(passive, PassiveBase)
        if not (is_legacy or _is_ideal_passive(passive)):
            raise CircuitError(
                'ERROR: add_passive() requires a PassiveBase instance (Resistor, Capacitor, or Inductor). '
                f'Got {type(passive).__name__}.'
            )

        validate_instance_name(passive.name)
        tokens = self._element_mgr._check_instance_name('passives_dict', passive.name, passive)
        if tokens is None:
            return

        if is_legacy:
            pos_net, neg_net = passive.pos, passive.neg
        else:
            # An ideal AnalogElement (silk.elements.elements.resistor/
            # capacitor/inductor) names its terminals from analog_devices.yaml
            # (pinList [p, n]) — read the nets off its own pin names, but keep
            # recording the PortBinding under add_passive()'s own 'pos'/'neg'
            # vocabulary (unchanged since before TASK-299) so existing
            # Net.pins consumers and tests keep matching on 'pos'/'neg'
            # regardless of which family built the object.
            pin_names = type(passive)._pin_names
            pos_net, neg_net = passive.nets[pin_names[0]], passive.nets[pin_names[1]]

        undo = []
        try:
            # Register terminals via the chokepoint. _ensure_net raises
            # UndeclaredNetError if strict_nets is set.
            for terminal_name, net_name in (('pos', pos_net), ('neg', neg_net)):
                self._element_mgr._bind_pin_to_net(net_name, PortBinding(
                    net=net_name,
                    instance=passive.name,
                    port=terminal_name,
                    direction=None,
                    role=None,
                    kind='passive',
                ), undo=undo)

        except Exception:
            self._element_mgr._purge_partial_instance('passives_dict', passive.name, passive, undo)
            raise

        self.passives_dict[passive.name] = passive
        self._element_mgr._register_instance(
            'passives_dict', passive.name, passive, tokens)

    # ==================================================

    # ===== Simulation methods (used when Module is a top-level netlist) =====

    def add_source(self, source):
        """Add a voltage/current source to the circuit.

        Each source terminal funnels through
        ``PyCircuitElements._bind_pin_to_net`` so that ``Net.pins``
        reflects source connectivity with ``kind='source'``. The source is
        then given a back-reference to this module, so a later direct
        terminal write (``src.positive = 'b'``) re-binds through the same
        path — see ``SourceBase.__setattr__``.

        Parameters
        ----------
        source : SourceBase
            A voltage, current, controlled or behavioral source.

        Returns
        -------
        None
            Re-adding the same instance returns without change.

        Raises
        ------
        silk.exceptions.CircuitError
            If ``source`` is not a source.
        DuplicateNameError
            If a different element already uses the name.
        UndeclaredNetError
            If a terminal net is undeclared and ``strict_nets`` is set.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(source, "add_source()")
        from silk.sources import SourceBase
        if not isinstance(source, SourceBase):
            raise CircuitError(
                f'add_source() requires a SourceBase instance. Got {type(source).__name__}.'
            )
        name = source.name
        validate_instance_name(name)
        tokens = self._element_mgr._check_instance_name('sig_sources_dict', name, source)
        if tokens is None:
            return
        undo = []
        try:
            for terminal_name, net_name in self._source_terminal_nets(source):
                _bind_source_terminal(self, name, terminal_name, net_name, undo=undo)
        except Exception:
            self._element_mgr._purge_partial_instance('sig_sources_dict', name, source, undo)
            raise

        self.sig_sources_dict[name] = source
        self._element_mgr._register_instance('sig_sources_dict', name, source, tokens)
        # Select the strong emission context. Supported sources rewire
        # through their weak holders; custom emitters use this single owner.
        object.__setattr__(source, '_module', self)
        if self.debug:
            print(f'Added source: {type(source).__name__} "{name}"')

    def _source_nets(self, source):
        """Return all net names referenced by a source (legacy helper)."""
        return [net_name for _, net_name in self._source_terminal_nets(source)]

    def _source_terminal_nets(self, source):
        """Return a list of (terminal_name, net_name) tuples for a source.

        Terminal names match the attribute names on SourceBase subclasses:

        - Simple two-terminal sources (VoltageSource, CurrentSource,
          BehavioralVoltage, BehavioralCurrent): ``positive``, ``negative``.
        - Voltage-controlled (VCVS, VCCS): ``out_pos``, ``out_neg``,
          ``ctrl_pos``, ``ctrl_neg``.
        - Current-controlled (CCVS, CCCS): ``out_pos``, ``out_neg`` only.
          These sources reference a controlling source by ``ctrl_vname``
          (a source instance name, not a node), so there is no
          node-level binding for the controlling current path.
        """
        from silk.sources import (VoltageSource, CurrentSource,
                                   VoltageControlledVoltageSource, VoltageControlledCurrentSource,
                                   CurrentControlledVoltageSource, CurrentControlledCurrentSource,
                                   BehavioralVoltage, BehavioralCurrent)
        if isinstance(source, (VoltageSource, CurrentSource, BehavioralVoltage, BehavioralCurrent)):
            return [('positive', source.positive), ('negative', source.negative)]
        elif isinstance(source, (VoltageControlledVoltageSource, VoltageControlledCurrentSource)):
            return [
                ('out_pos',  source.out_pos),
                ('out_neg',  source.out_neg),
                ('ctrl_pos', source.ctrl_pos),
                ('ctrl_neg', source.ctrl_neg),
            ]
        elif isinstance(source, (CurrentControlledVoltageSource, CurrentControlledCurrentSource)):
            # CCVS/CCCS reference a controlling source by ctrl_vname (a
            # source instance name, not a node), so only the output
            # terminals contribute node-level bindings.
            return [('out_pos', source.out_pos), ('out_neg', source.out_neg)]
        # Unknown SourceBase subclass — silently skip binding. Phase 1
        # is conservative here so we don't break any unusual source
        # types we haven't accounted for; Phase 2 may tighten this.
        return []

    def update_net(self, netname, parameters=None):
        """Update a net's legacy parameters list.

        The net must already exist. Replaces ``Net.parameters`` with the
        supplied list (copied). Use :meth:`add_net` to create a new net.

        Parameters
        ----------
        netname : str
            Name of an existing net.
        parameters : list, optional
            New legacy parameter list; ``None`` clears it.

        Returns
        -------
        None

        Raises
        ------
        silk.exceptions.CircuitError
            If the net does not exist.
        """
        if parameters is None:
            parameters = []
        if netname not in self.net_dict:
            raise CircuitError(f'Net "{netname}" does not exist.')
        self.net_dict[netname].parameters = list(parameters)

    def print_elements(self):
        """Print all defined elements."""
        print(f'Elements ({len(self.elem_dict)} devices, {len(self.sig_sources_dict)} sources):')
        for k, v in self.elem_dict.items():
            type_name = v.get('typeName', v.get('type', '?')) if isinstance(v, dict) else type(v).__name__
            print(f'  [{type_name}] {k}')
        for k, v in self.sig_sources_dict.items():
            src_type = type(v).__name__
            nets = []
            if hasattr(v, 'positive') and hasattr(v, 'negative'):
                nets = [v.positive, v.negative]
            elif hasattr(v, 'out_pos') and hasattr(v, 'out_neg'):
                nets = [v.out_pos, v.out_neg]
            print(f'  [{src_type}] {k}  nets={nets}')

    def print_nets(self):
        """Print all defined nets."""
        net_names = list(self.net_dict.keys())
        print(f'Nets ({len(net_names)}): {", ".join(net_names)}')

    def update_source(self, sourceName, params, paramValues, _skipCheck=False):
        """Update source parameters (supports both SourceBase and legacy dict sources).

        ``_skipCheck`` is accepted for signature parity with
        :meth:`update_element` (the refactor golden pins it) but has no
        effect: an unknown name raises ``CircuitError`` either way. Pre-fix ``_skipCheck=True`` skipped the guard and only moved the
        failure to a bare ``KeyError`` from the ``sig_sources_dict`` lookup
        on the next line — a source that does not exist cannot be updated,
        same as :meth:`connect_source`.

        Parameters
        ----------
        sourceName : str
            Name of a source already added to the module.
        params : list of str
            Attribute names to change (e.g. ``'dc'``, ``'ac'``); names the
            source lacks are ignored.
        paramValues : list
            New values, one per name.
        _skipCheck : bool, default False
            Accepted for signature parity; has no effect.

        Returns
        -------
        None

        Raises
        ------
        silk.exceptions.CircuitError
            If the source does not exist.
        """
        if sourceName not in self.sig_sources_dict:
            raise CircuitError(f'Source "{sourceName}" does not exist.')
        source = self.sig_sources_dict[sourceName]
        _SOURCE_PARAMS = ['dc', 'ac', 'pwl', 'pulse', 'sine']
        if isinstance(source, dict):
            # Legacy dict-based source
            src_params = source.get('params', [None] * len(_SOURCE_PARAMS))
            for p, v in zip(params, paramValues):
                if p in _SOURCE_PARAMS:
                    src_params[_SOURCE_PARAMS.index(p)] = v
            source['params'] = src_params
        else:
            # Validate the whole proposed state before writing any part of it.
            changes = {p: v for p, v in zip(params, paramValues) if hasattr(source, p)}
            apply = getattr(source, '_apply_update', None)
            if apply is not None:
                apply(changes)
                return
            validate = getattr(source, '_validate_update', None)
            if validate is not None:
                validate(changes)
            for p, v in zip(params, paramValues):
                if hasattr(source, p):
                    setattr(source, p, v)

    def connect_source(self, sourceName, pins, nets, _skipCheck=False):
        """Rewire source terminals (supports both SourceBase and legacy dict sources).

        ``pins`` are terminal names as spelled on the source class
        (``positive``/``negative``; ``out_pos``/``out_neg``/``ctrl_pos``/
        ``ctrl_neg`` for controlled sources). Every net must already be
        declared, regardless of ``strict_nets``. Each rewired terminal goes through
        ``_bind_source_terminal``, so ``Net.pins`` follows the rewire:
        the old net drops the binding and the new net gains it.

        Parameters
        ----------
        sourceName : str
            Name of a source already added to the module.
        pins : list of str
            Terminal names to rewire.
        nets : list
            Declared net names (or :class:`Net` objects), one per terminal.
        _skipCheck : bool, default False
            Accepted for signature parity; has no effect.

        Returns
        -------
        None

        Raises
        ------
        CircuitError
            Unknown source, a pins/nets length mismatch
            (naming both lengths), or a pin name that is not one of the
            source's terminals (the message lists the valid ones).
            Checked before anything is written, so a bad call leaves the
            source and every ``Net`` untouched (raised as ``CircuitError``).
        UndeclaredNetError
            A net that has not been declared.

        Notes
        -----
        ``_skipCheck`` is retained in the source API signature but has no
        effect: an unknown name raises ``CircuitError`` either way. Pre-fix ``_skipCheck=True`` only moved the failure to a bare
        ``KeyError`` from the ``sig_sources_dict`` lookup on the next line —
        a source that does not exist cannot be rewired.
        """
        if sourceName not in self.sig_sources_dict:
            raise CircuitError(f'Source "{sourceName}" does not exist.')
        # Checked before the per-net/per-pin validation below (TASK-43):
        # the write loop further down zips ``pins`` against ``nets``, which
        # silently TRUNCATES to the shorter of the two on a mismatch —
        # measured pre-fix: connect_source('v1', ['positive'], ['a', 'b'])
        # rewired 'positive' and dropped 'b' with no error at all.
        if len(pins) != len(nets):
            raise CircuitError(
                f'ERROR: connect_source({sourceName!r}): pins has {len(pins)} '
                f'entries but nets has {len(nets)}; lengths must match.'
            )
        # Normalize Net arguments before dictionary lookup; each holder binds
        # the name locally rather than adopting the supplied Net object.
        from silk.elements.base import validate_net_token
        nets = [validate_net_token(
            net, who=f'connect_source({sourceName!r}) terminal {pin!r}', kind='terminal')
                if not isinstance(net, str) and isinstance(getattr(net, 'name', None), str) else net
                for pin, net in zip(pins, nets)]
        from silk.circuit.bus_spec import _guard_scalar_alias
        for net in nets:
            if isinstance(net, str):
                _guard_scalar_alias(self, net, connection=True)
            if net not in self.net_dict:
                raise UndeclaredNetError(f'Net "{net}" is undefined.')
        source = self.sig_sources_dict[sourceName]
        _LEGACY_PINS = ('positive', 'negative')
        # Legacy dict sources have no class to ask; SourceBase names its own
        # node terminals (the same tuple __setattr__ validates against).
        valid = _LEGACY_PINS if isinstance(source, dict) else tuple(source._node_terminals)
        for p in pins:
            if p not in valid:
                raise CircuitError(
                    f'Pin "{p}" is not a terminal of source "{sourceName}"; '
                    f'valid terminals: {", ".join(valid)}.'
                )
        if isinstance(source, dict):
            pin_nets = source.get('pinNets', ['', ''])
            for p, net in zip(pins, nets):
                pin_nets[_LEGACY_PINS.index(p)] = net
            source['pinNets'] = pin_nets
        else:
            source._apply_update(dict(zip(pins, nets)))

    def variables(self):
        """Return the Variable objects this module and its placed children
        use, each once by identity, in first-use order.

        Each module is visited in this order: devices, sources (including
        controlled-source numeric fields and stored waveform fields),
        passives_dict insertion order, attached behavioral params key order,
        then placed children in placement order, depth first. Children placed
        with add_module or embedded in testbenches share the original Variables.
        A shared Variable takes its first occurrence; distinct same-name
        Variables remain distinct.

        Expressions, Normal.nominal/sigma, Uniform.low/high and stored
        dict keys/values, list, tuple, set, frozenset, object-bearing ndarray
        and structured numpy scalar fields are searched without resolving
        values, drawing samples, executing models or reading derived properties.
        Set/frozenset order is undefined; cyclic containers terminate.
        Unassigned Variables are included. Arbitrary object attributes,
        callable internals and SPICE strings are not searched.

        Deprecated add_submodule registrations, legacy add_element device
        param_list storage and legacy PassiveBase.value storage are excluded.
        """
        from silk.circuit.modules import Module
        from silk.sources.waveforms import _Waveform
        seen = set()
        result = []

        def take(val):
            for var in _leaf_variables(val):
                if id(var) not in seen:
                    seen.add(id(var))
                    result.append(var)

        def walk(mod, visiting):
            children = []
            for entry in mod._element_mgr.elem_dict.values():
                device = entry.get('device') if isinstance(entry, dict) else entry
                if isinstance(device, Module):
                    children.append(device)
                elif hasattr(device, 'params'):
                    for val in device.params.values():
                        take(val)
            for src in mod.sig_sources_dict.values():
                for param in getattr(src.__class__, '_param_names', ()):
                    take(getattr(src, param, None))
                for value in vars(src).values():
                    if isinstance(value, _Waveform):
                        take(vars(value))
            for passive in mod.passives_dict.values():
                for val in getattr(passive, 'params', {}).values():
                    take(val)
            take(mod._behavioral_params)
            for child in children:
                if id(child) not in visiting:
                    walk(child, visiting | {id(child)})

        walk(self, {id(self)})
        return result

    # ==================================================

    # ===== NEW: Test management methods =====
    def add_submodule(self, name, module):
        """Deprecated — registers *module* in ``module_dict`` only; it never
        places an instance (no copy, no ``.subckt`` registration, no port
        binding, no ``x`` line). Use :meth:`add_module` to actually place a
        child.

        Args:
            name: Name for the submodule in module_dict
            module: Module instance to add as a submodule

        Raises:
            ValueError: If trying to add a module to itself (cycle detection)

        Example:
            parent.add_submodule('diffpair', differential_pair_module)

        Note:
            Unlike ``add_module``/``add_element``, a module_name-less
            ``module`` is NOT refused here. This supports
            an existing, deliberate use: ``add_submodule``
            also serves as a purely organizational parent/child
            registration for the recursive test-walking tree
            (``run_all_tests(recursive=True)``,
            ``silk/circuit/modules_testbench.py``'s
            ``_run_test_entries``), where the registered module owns no
            ports and is never emitted as a subcircuit — pinned by
            ``tests/test_temperature_inheritance.py::test_recursive_runs_apply_the_nearest_setting``,
            which registers two module_name-less containers this way on
            purpose. See docs/known_issues.md for current limitations.
        """
        from silk.circuit.device_view import _refuse_device_view
        _refuse_device_view(module, "add_submodule()")
        import warnings
        warnings.warn(
            "Module.add_submodule() is deprecated: it only registers "
            "module_dict and never places an instance (no copy, no "
            "'.subckt' registration, no port binding, no 'x' line). Use "
            "add_module(module, name, **ports) to place a child.",
            DeprecationWarning, stacklevel=2)
        if module is self:
            raise ValueError(
                f"Cannot add module {self.module_name!r} to itself as a submodule. "
                f"This would create a circular reference."
            )
        # Write through the owner — module_dict is exposed read-only on Module.
        self._element_mgr.module_dict[name] = {'device': module}

    def export_schema(self) -> dict:
        """Return a JSON-serialisable schema for this module's external interface.

        Includes port names, directions, and device instance names.
        A machine-readable API for tools that inspect modules, render circuits,
        or generate silk code.

        Returns
        -------
        dict
            See :func:`silk.elements.schema.module_schema <silk.circuit.module_schema>` for format.

        Examples
        --------
        ::

            import json
            from silk._json_codec import encode_json
            from silk.circuit.circuit_elements import PortDir
            ldo.define_module_name('ldo')
            ldo.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
            schema = ldo.export_schema()
            print(json.dumps(encode_json(schema), allow_nan=False, indent=2))
        """
        from silk.elements.schema import module_schema
        return module_schema(self)
