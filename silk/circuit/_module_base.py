"""Shared module context, instance records and circuit construction helpers."""

# Shared, importable Module building blocks — relocated out of modules.py
# so the per-subsystem "part" mixins can import them
# without a circular dependency on modules.py. modules.py re-exports every
# name defined here, so external imports (silk.circuit.modules._active_module, ...)
# and the _active context singleton stay canonical.
#
# This is a clean import leaf: it references Module / Net / Signal / PortWiring
# / ElementType only lazily (inside methods) or via duck typing, never at
# import time.

from silk.circuit.emitted_names import emitted_submodule_name
import threading
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from silk.exceptions import SimulationError


# ----------------------------------------------------------------------
# Active-module stack — set by using a Module as a context manager
# (``with mod:``). While a module is on the (thread-local) stack,
# element / source / passive constructors auto-register the new
# instance with it.
# ----------------------------------------------------------------------

_active = threading.local()


def _active_module():
    """Return the innermost module currently being built, or None."""
    stack = getattr(_active, 'stack', None)
    return stack[-1] if stack else None


def _push_active(module):
    stack = getattr(_active, 'stack', None)
    if stack is None:
        stack = []
        _active.stack = stack
    stack.append(module)


def _pop_active():
    _active.stack.pop()


class _NetAccessor:
    """Attribute-access namespace proxy for a Module's nets.

    Exposes ``module.net_dict`` as attributes so callers can pass
    ``tb.nets.avdd`` instead of the net-name string. Typo → AttributeError
    at the call site. ``dir(tb.nets)`` enumerates the nets for IDE
    autocomplete. Declared names, including single-underscore names, take
    precedence over accessor fields; double-underscore attributes keep
    Python's normal lookup.
    """
    __slots__ = ('__silk_module__',)

    def __init__(self, module):
        self.__silk_module__ = module

    def __getattribute__(self, name):
        if not name.startswith('__'):
            module = object.__getattribute__(self, '__silk_module__')
            if name in module.net_dict:
                return module.net_dict[name]
            bus_handles = getattr(module, '_bus_handles', None)
            if bus_handles is not None and name in bus_handles:
                return bus_handles[name]
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        module = object.__getattribute__(self, '__silk_module__')
        raise AttributeError(
            f"No net {name!r} on module "
            f"{module.module_name or '<top-level>'!r}. "
            f"Declare it with module.add_net({name!r}) first."
        ) from None

    def __getitem__(self, name):
        """Bracket access for nets whose names are not valid Python
        identifiers (e.g. ``tb.nets['0']`` for the ngspice ground, or
        any digit-leading net name)."""
        module = object.__getattribute__(self, '__silk_module__')
        try:
            return module.net_dict[name]
        except KeyError:
            pass
        bus_handles = getattr(module, '_bus_handles', None)
        if bus_handles is not None and name in bus_handles:
            return bus_handles[name]
        raise KeyError(
            f"No net {name!r} on module "
            f"{module.module_name or '<top-level>'!r}"
        ) from None

    def __contains__(self, name):
        module = object.__getattribute__(self, '__silk_module__')
        if name in module.net_dict:
            return True
        bus_handles = getattr(module, '_bus_handles', None)
        return bus_handles is not None and name in bus_handles

    def __iter__(self):
        module = object.__getattribute__(self, '__silk_module__')
        bus_handles = getattr(module, '_bus_handles', {}) or {}
        return iter({**module.net_dict, **bus_handles})

    def __dir__(self):
        module = object.__getattribute__(self, '__silk_module__')
        bus_handles = getattr(module, '_bus_handles', {}) or {}
        return list({**module.net_dict, **bus_handles})

    def __repr__(self):
        module = object.__getattribute__(self, '__silk_module__')
        names = ', '.join(sorted(module.net_dict))
        return f"<nets on {module.module_name or '<top-level>'}: {names}>"


@dataclass
class ModulePin:
    """One declared port of a module, as stored in ``Module.pin_list``.

    Built by :meth:`silk.circuit.Module.define_module_ports` from the ports mapping; a
    placement copies the pin and fills ``net`` with the parent net it is
    wired to.

    Parameters
    ----------
    name : str
        Port name, as declared (a bus bit is its expanded name, ``d[3]``).
    net : str, optional
        Parent-level net the pin is wired to at a placement; ``None`` on the
        module's own declaration.
    direction : str, optional
        The :class:`~silk.circuit.ports.PortDir` value.
    role : str, optional
        The :class:`~silk.circuit.ports.PortRole` value, or ``None`` if the
        port is unannotated.
    domain : VoltageDomain, optional
        Voltage domain the port belongs to.
    role_inferred : bool, default False
        Whether ``role`` was guessed by ``infer_port_roles``.
    """
    #: Port name, as declared.
    name: str
    #: Parent-level net the pin is wired to at a placement; ``None`` on the
    #: module's own declaration.
    net: Optional[str] = None
    #: The ``PortDir`` value.
    direction: Optional[str] = None
    #: The ``PortRole`` value, or ``None`` if unannotated.
    role: Optional[str] = None
    #: Voltage domain the port belongs to, or ``None``.
    domain: Optional[Any] = None
    # Provenance of ``role``: True only when ``infer_port_roles`` filled it
    # from a name pattern (a guess); False for every role the user stated —
    # an explicit PortRole, a factory, or the TASK-45 direction-implied
    # supply role (a statement). Snapshotted into ``PortBinding`` at wiring so
    # INT-004 can weigh a guessed POWER role differently (TASK-42).
    #: ``True`` only when ``infer_port_roles`` guessed ``role`` from the
    #: port name; ``False`` for every role the user stated.
    role_inferred: bool = False


# ----------------------------------------------------------------------
# SubcircuitInstance — ergonomic handle returned by add_element() when
# instantiating a subcircuit. Gives attribute-access to internal nets
# and nested child instances, hiding the xfcamp/xxfcamp SPICE-prefix
# convention from user code.
#
# Usage:
#     fcamp_inst = tb.add_element(fcamp, 'fcamp', nets={...})
#     tb.simulate(corner='tt')
#     fcamp_inst.vbiasn                 # → Signal (tb.load_signal under the hood)
#     fcamp_inst.vbiasn.y[-1]
#     bias_inst.fcamp.vbiasn            # hierarchy: bias → fcamp → vbiasn
#
# How it resolves net paths:
#   * Ports: mapped to the parent-level net the user provided in the
#     nets={} dict at add_element time. Recurses up through nested
#     SubcircuitInstance parents to reach a top-level net name, because
#     ngspice aliases port nodes to their external net in the rawfile.
#   * Internal nets: hierarchical path 'x{instance}.{net}', composed by
#     walking up the parent chain.
#   * Child instances: attribute returns a new SubcircuitInstance for the
#     nested subcircuit, with its prefix correctly composed.
# ----------------------------------------------------------------------

class SubcircuitInstance:
    """Handle to a subcircuit instance inside a parent Module.

    Returned by :meth:`Module.add_element <silk.circuit.Module.add_element>` when the element being added is
    itself a Module (subcircuit). Provides attribute-access to internal
    nets and nested child instances so user code never has to build the
    ``xxfcamp.vbiasn`` string by hand.

    Declared items take precedence over the handle's public members. A
    shadowed tool is unavailable through that handle's dot shortcut.

    Silk constructs these; user code obtains them from ``add_element``.

    Parameters
    ----------
    root_tb : Module or ModuleTestbench
        The top-level design the instance lives in.
    spice_name : str
        The SPICE instance token, including the leading ``'x'``.
    subcircuit_def : Module
        The placed module definition.
    port_map : dict
        Port name to parent net (a plain net name or a ``PortWiring``).
    parent : SubcircuitInstance or Module, optional
        The enclosing handle; ``None`` or the root for a top-level placement.
    """

    def __init__(self, root_tb, spice_name, subcircuit_def, port_map, parent=None):
        # Names starting with '_' are real instance attributes so
        # __getattr__ won't shadow them.
        object.__setattr__(self, '_root_tb', root_tb)
        object.__setattr__(self, '_spice_name', spice_name)      # e.g. 'xfcamp'
        object.__setattr__(self, '_def', subcircuit_def)
        object.__setattr__(self, '_parent', parent)               # SubcircuitInstance | Module
        container = parent._def if isinstance(parent, SubcircuitInstance) else root_tb
        entries = [(key, entry) for key, entry in container.elem_dict.items()
                   if emitted_submodule_name(key) == spice_name and entry.get('device') is subcircuit_def]
        link = (container, *entries[0]) if len(entries) == 1 else None
        object.__setattr__(self, '_link', link)

        # Convert dict[str, str] port_map into bus-aware dict[str, PortWiring].
        from silk.circuit.port_binding import PortWiring
        instance_buses = getattr(subcircuit_def, '_buses', {}) or {}
        # Build reverse lookup: expanded bit pin name → (bus_name, bit_index)
        pin_to_bus = {}
        for bus_name, spec in instance_buses.items():
            for idx, bit_name in zip(spec.indices, spec.expanded_names()):
                pin_to_bus[bit_name] = (bus_name, idx)

        new_map = {}
        for pin_name, parent_net_value in port_map.items():
            if isinstance(parent_net_value, PortWiring):
                new_map[pin_name] = parent_net_value
            else:
                new_map[pin_name] = PortWiring(
                    parent_net=parent_net_value,
                    bus=pin_to_bus.get(pin_name),
                )
        object.__setattr__(self, '_port_map', new_map)  # {port_name: PortWiring}

    def __setattr__(self, name, value):
        if not name.startswith('_'):
            raise AttributeError(
                f"Placement handles are read-only: cannot assign {name!r}. "
                "To rewire a port, call connect_module on the parent module.")
        object.__setattr__(self, name, value)

    # ---- public properties -------------------------------------------

    @property
    def instance_name(self):
        """The SPICE instance token, including silk's leading 'x' (e.g.
        ``'xfcamp'``; ``'xx1<0>'`` for an array copy). It is the simulator's
        spelling, a lookup key, and is not converted."""
        return self._spice_name

    @property
    def subcircuit(self):
        """The subcircuit Module definition this handle points at."""
        return self._def

    # ---- path resolution ---------------------------------------------

    def _hierarchical_prefix(self):
        """Build the full ngspice hierarchical path prefix up to this instance.

        Example: for ``bias_inst.fcamp`` where ``bias_inst`` is at tb top-
        level, returns ``'xbias.xfcamp'``.
        """
        if isinstance(self._parent, SubcircuitInstance):
            return f'{self._parent._hierarchical_prefix()}.{self._spice_name}'
        return self._spice_name

    def _external_path(self, name):
        """Resolve a local net/port name to the path ngspice actually uses
        in the rawfile.

        - If ``name`` is a port of this subcircuit, walk up the parent chain
          to the top-level net it aliases to.
        - Otherwise, treat as an internal net and return the full
          hierarchical path ``prefix.name``.
        """
        if self.__dict__.get('_link') is not None:
            from silk.circuit.placement_reference import _resolve_path
            return _resolve_path(self._placement_chain(), name)
        if name in self._port_map:
            parent_net = self._port_map[name].parent_net
            if isinstance(self._parent, SubcircuitInstance):
                return self._parent._external_path(parent_net)
            # parent is the top-level Module — the mapped net is already
            # a top-level net name.
            return parent_net
        return f'{self._hierarchical_prefix()}.{name}'

    def path(self, name, *, raw=False):
        """Public form of ``_external_path`` — returns the path string
        you would pass to ``tb.load_signal``.

        Array instance parts are shown as ``[k]`` (``'xx1[0].g'``);
        ``load_signal`` accepts either spelling. ``raw=True`` returns the
        simulator's own spelling (``'xx1<0>.g'``) exactly. The bracket
        spelling is used only when it is collision-safe in the owning
        design's net inventory: a literal net ``xx1[0].g`` beside the copy's
        ``xx1<0>.g`` keeps the raw spelling for the whole collision group.
        A port resolves to the parent net, which is returned as it is.

        Parameters
        ----------
        name : str
            A net or port of this subcircuit.
        raw : bool, default False
            Return the simulator's own spelling instead of the bracket form.

        Returns
        -------
        str
            The path to pass to ``load_signal``.
        """
        path = self._external_path(name)
        if raw:
            return path
        from silk.circuit.modules_netlist import _save_list_inventory
        from silk.spice_names import designer_alias_map
        root = self._root_tb
        if getattr(root, 'net_dict', None) is None:
            root = getattr(root, 'netlist', root)
        try:
            inventory = list(_save_list_inventory(root))
        except Exception:                                  # no inventory: stay exact
            return path
        return designer_alias_map(inventory + [path]).get(path, path)

    # ---- attribute access --------------------------------------------

    def _child_entry(self, name):
        """Return the elem_dict entry for a nested child subcircuit named
        *name*, or None if not found."""
        from silk.circuit.circuit_elements import ElementType  # local: circular-import safety
        entry = self._def.elem_dict.get(name)
        if entry is not None and entry.get('type') == ElementType.MODULE:
            return entry
        return None

    def _make_child_handle(self, name, entry):
        child_def = entry['device']
        pins = entry.get('pin_list', child_def.pin_list) or {}
        port_map = {pin_name: pin.net for pin_name, pin in pins.items()}
        return SubcircuitInstance(
            root_tb=self._root_tb,
            spice_name=emitted_submodule_name(name),            # same convention as generate_netlist_entry
            subcircuit_def=child_def,
            port_map=port_map,
            parent=self,
        )

    def _placement_chain(self):
        from silk.exceptions import CircuitError
        chain = self._parent._placement_chain() if isinstance(self._parent, SubcircuitInstance) else ()
        link = self.__dict__.get('_link')
        if link is None or link[0].elem_dict.get(link[1]) is not link[2] or link[2]['device'] is not self._def:
            raise CircuitError('Placement was removed or replaced; obtain a new placement handle.')
        return (*chain, link)

    def net(self, name):
        """Return a structural scalar net/port reference, including reserved names.

        Parameters
        ----------
        name : str
            A scalar net or port declared on this subcircuit.

        Returns
        -------
        PlacementReference
            A deferred reference; bind it with ``tb.probe``.

        Raises
        ------
        CircuitError
            If the placement was removed or replaced, or ``name`` is not a
            scalar net or port (use :meth:`bus` for a bus).
        """
        from silk.circuit.placement_reference import PlacementReference
        from silk.exceptions import CircuitError
        chain = self._placement_chain()
        if name not in self._def.net_dict and name not in self._def.pin_list:
            raise CircuitError(f'No scalar net/port {name!r}; use bus(name) for a bus.')
        return PlacementReference(self._root_tb, chain, name)

    def bus(self, name):
        """Return a deferred bus reference indexed by declared bit number.

        Parameters
        ----------
        name : str
            A bus declared on this subcircuit.

        Returns
        -------
        PlacementReference
            A deferred reference; subscripts use the declared bit numbers.

        Raises
        ------
        CircuitError
            If the placement was removed or replaced, or no such bus exists.
        """
        from silk.circuit.placement_reference import PlacementReference
        from silk.exceptions import CircuitError
        chain = self._placement_chain()
        spec = self._def._buses.get(name)
        if spec is None:
            raise CircuitError(f'No bus {name!r} in this placement.')
        handle = self._def._bus_handles.get(name)
        if handle is not None and handle.width > spec.width:
            return PlacementReference(self._root_tb, chain, name,
                children=tuple(SubcircuitInstance.net(self, bit.name) for bit in handle),
                indices=tuple(i for _, i in handle._index_vector))
        return PlacementReference(self._root_tb, chain, name,
            children=tuple(SubcircuitInstance.net(self, bit) for bit in spec.expanded_names()),
            indices=spec.indices)

    def child(self, name):
        """Look up an actual child placement or declared-number placement array.

        Parameters
        ----------
        name : str
            Instance key of a child module, or the base name of an array of
            them (``'x'`` for ``x[0]``, ``x[1]``).

        Returns
        -------
        SubcircuitInstance or PlacementArray
            A handle for one child, or an array handle for a numbered group.

        Raises
        ------
        CircuitError
            If no such child exists, the name is both a scalar and an array,
            or the placement was removed or replaced.
        """
        from silk.circuit.circuit_elements import ElementType
        return self._lookup_element(name, ElementType.MODULE)

    def device(self, name):
        """Look up a primitive device or declared-number device array.

        Parameters
        ----------
        name : str
            Instance key of a device, or the base name of an array of them.

        Returns
        -------
        PlacementDeviceView or PlacementArray
            A read-only placement-specific view of the device, or an array
            of such views. A view is distinct from the original device and
            from a view obtained through another placement. Public attribute
            assignment/deletion and edits to its params, nets and layout
            mappings refuse; edit the original device on its holding module.
            Values inside the mappings, including Variables, stay shared.
            Reads and fault targets refuse after removal or replacement.

        Raises
        ------
        CircuitError
            If no such device exists, the name is both a scalar and an array,
            or the placement was removed or replaced.
        """
        from silk.circuit.circuit_elements import ElementType
        return self._lookup_element(name, ElementType.DEVICE)

    def _lookup_element(self, name, kind):
        from silk.circuit.circuit_elements import ElementType
        from silk.circuit.placement_reference import PlacementArray, array_entries
        from silk.exceptions import CircuitError
        self._placement_chain()
        entry = self._def.elem_dict.get(name)
        group = array_entries(self._def, name, kind)
        if entry is not None and entry.get('type') == kind and group:
            raise CircuitError(f'Ambiguous scalar/array name {name!r}; use an exact placement key.')
        if group:
            return PlacementArray(name, self, kind, group)
        if entry is None or entry.get('type') != kind:
            raise CircuitError(f'No {kind} placement {name!r}.')
        if kind == ElementType.MODULE:
            return self._make_child_handle(name, entry)
        from silk.circuit.device_view import PlacementDeviceView
        return PlacementDeviceView(self, name, entry)

    def __getattribute__(self, name):
        if name in ('bus', 'child', 'device', 'instance_name', 'net', 'path', 'subcircuit'):
            try:
                return SubcircuitInstance.__getattr__(self, name)
            except AttributeError:
                # Without a declared item, the handle's tool stays available.
                pass
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        from silk.circuit.circuit_elements import ElementType
        from silk.circuit.placement_reference import array_entries
        from silk.exceptions import CircuitError
        matches = []
        # Internal lookups must still call the tool when an item shadows it.
        if name in self._def.net_dict or name in self._def.pin_list:
            matches.append(SubcircuitInstance.net)
        if name in self._def._buses:
            matches.append(SubcircuitInstance.bus)
        entry = self._def.elem_dict.get(name)
        for kind, lookup in ((ElementType.MODULE, SubcircuitInstance.child),
                             (ElementType.DEVICE, SubcircuitInstance.device)):
            if (entry is not None and entry.get('type') == kind) or array_entries(self._def, name, kind):
                matches.append(lookup)
        if len(matches) > 1:
            raise CircuitError(f'Ambiguous placement attribute {name!r}: '
                               f'{[method.__name__ for method in matches]}; '
                               'use net(name), bus(name), child(name), or device(name).')
        if matches:
            return matches[0](self, name)
        raise AttributeError(f'{self._def.module_name!r} has no net or sub-instance {name!r}.')

    def __dir__(self):
        from silk.circuit.circuit_elements import ElementType
        nets = list(self._def.net_dict.keys())
        children = [n for n, e in self._def.elem_dict.items()
                    if e.get('type') == ElementType.MODULE]
        return list(super().__dir__()) + nets + children

    def __repr__(self):
        from silk.spice_names import designer_component
        return (f'SubcircuitInstance({designer_component(self._spice_name)!r} of '
                f'{self._def.module_name!r})')

    def __setstate__(self, state):
        """Pickle back-compat: accept both legacy (dict[str, str]) and new
        (dict[str, PortWiring]) port_map shapes. Legacy state is upgraded
        in place; bus membership cannot be reconstructed for legacy data."""
        import warnings
        from silk.circuit.port_binding import PortWiring

        port_map = state.get('_port_map', {})
        if port_map and not isinstance(next(iter(port_map.values())), PortWiring):
            warnings.warn(
                "SubcircuitInstance._port_map upgraded from legacy dict[str, str] "
                "shape; bus membership cannot be reconstructed for this instance. "
                "Re-pickle to refresh.",
                DeprecationWarning, stacklevel=2,
            )
            state['_port_map'] = {
                k: PortWiring(parent_net=v, bus=None)
                for k, v in port_map.items()
            }
        self.__dict__.update(state)
