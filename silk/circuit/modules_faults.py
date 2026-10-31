"""Fault-injection methods for :class:`~silk.circuit.modules.Module`.

Split out of ``modules.py`` as a per-subsystem **part**.
``_FaultMethods`` is a pure mixin: it holds no state and has no separate
identity — ``Module`` inherits it, so every ``self`` here is the ``Module``
instance. See ``silk/faults/`` for campaign execution and reporting.
"""
from typing import TYPE_CHECKING

from silk.circuit.circuit_elements import ElementType

if TYPE_CHECKING:  # pragma: no cover
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object


class _FaultMethods(_Base):
    """Fault-injection hooks mixed into ``Module``. Holds no state."""

    def _silk_fault_all_device_paths(self) -> list:
        """Enumerate every primitive-device instance path in this module's
        hierarchy using the ``x<instance>.<...>`` convention that matches
        ngspice's rawfile naming and ``silk.faults.FaultSet`` targets.
        """
        out: list = []
        for inst_name, entry in self.elem_dict.items():
            etype = entry.get('type')
            dev = entry.get('device')
            if etype == ElementType.DEVICE:
                out.append(inst_name)
            elif etype == ElementType.MODULE and dev is not None:
                nested = dev._silk_fault_all_device_paths()
                out.extend(f'x{inst_name}.{n}' for n in nested)
        return out

    def _silk_resolve_fault_target(self, target_path: str):
        """Resolve an authored placement path to its terminal device.

        Each ``x<name>`` hop follows a live placement; the trailing component
        names a primitive device. Invalid paths raise ``KeyError``.
        """
        return _resolve_fault_context(self, target_path)[0]

    def _silk_apply_fault_markers(self, fault_set, rng, manifest=None):
        """Realize temporary device copies, first fault per complete target.

        Resolve the entire request before creating markers so invalid later
        targets leave the authored graph untouched. The emission overlay
        selects each copy only within its owning placement.
        """
        from copy import copy
        from silk.elements.base import AnalogElement
        from silk.circuit.spice_netlister import _PDKDeviceView
        from silk.faults.application import _realize_fault

        faults = fault_set.realize(self, rng)
        resolved, seen = [], set()
        for fault in faults:
            device, name, owner = _resolve_fault_context(self, fault.target)
            if fault.target not in seen:
                seen.add(fault.target)
                resolved.append((device, name, owner, fault))
        marked = []
        for original, name, owner, fault in resolved:
            device = copy(original)
            device._silk_pending_fault = fault
            pdk = owner.netlister.pdk_details
            view = device if isinstance(device, AnalogElement) else _PDKDeviceView(
                device, name, pdk)
            device._silk_fault_realization = _realize_fault(view, fault, rng, pdk=pdk)
            if manifest is not None:
                manifest.extend(device._silk_fault_realization.applied)
            marked.append(device)
        return marked

    @staticmethod
    def _silk_clear_fault_markers(marked):
        for device in marked:
            for attr in ('_silk_pending_fault', '_silk_fault_realization'):
                if hasattr(device, attr):
                    try:
                        delattr(device, attr)
                    except AttributeError:
                        pass


def _resolve_fault_context(module, target_path):
    """Resolve the device, registry key and owning emission module."""
    parts = target_path.split('.')
    node = module
    for hop in parts[:-1]:
        if not hop.startswith('x'):
            raise KeyError(
                f'Fault target {target_path!r}: non-terminal component '
                f'{hop!r} must start with "x" (SPICE subckt instance).')
        inst_name = hop[1:]
        entry = node.elem_dict.get(inst_name)
        if entry is None or entry.get('type') != ElementType.MODULE:
            raise KeyError(
                f'Fault target {target_path!r}: no submodule instance '
                f'named {inst_name!r} at {node.module_name or "<top>"}.')
        node = entry['device']
    leaf = parts[-1]
    leaf_entry = node.elem_dict.get(leaf)
    if leaf_entry is None or leaf_entry.get('type') != ElementType.DEVICE:
        raise KeyError(
            f'Fault target {target_path!r}: no device named {leaf!r} in '
            f'{node.module_name or "<top>"}.')
    return leaf_entry['device'], leaf, node


def _prepare_fault_definitions(module, marked):
    """Build an emission-only placement tree without rendering any bodies."""
    from silk.circuit.module_placement import _definition_names, resolved_name

    root = {'children': {}, 'devices': {}}
    occupied = set((_definition_names.get() or {}).keys())
    visited = set()

    def inventory(mod):
        if id(mod) in visited:
            return
        visited.add(id(mod))
        if mod.module_name is not None:
            occupied.add(resolved_name(mod.module_name).casefold())
        # Includes legacy registrations, which placement validation does not
        # traverse. Reading the index performs the usual healthy refresh.
        for name, entry in mod.module_dict.items():
            occupied.add(resolved_name(name).casefold())
            inventory(entry['device'])
        for entry in mod.elem_dict.values():
            if entry['type'] == ElementType.MODULE:
                inventory(entry['device'])

    inventory(module)
    nodes = {}
    for device in marked:
        parts = device._silk_pending_fault.target.split('.')
        node, parent, path = root, module, []
        for hop in parts[:-1]:
            instance = hop[1:]
            path.append(hop)
            entry = parent.elem_dict[instance]
            child = node['children'].get(instance)
            if child is None:
                # module_dict has already refreshed the healthy placement names.
                child = {'children': {}, 'devices': {}, 'module': entry['device'],
                         'healthy': resolved_name(entry['module_name'])}
                node['children'][instance] = child
                nodes[tuple(path)] = child
            node, parent = child, entry['device']
        node['devices'][parts[-1]] = device
    for path, node in sorted(nodes.items()):
        stem = '__silk_fault_' + node['healthy'] + '_' + '.'.join(path).encode('utf-8').hex()
        name, suffix = stem, 0
        while name.casefold() in occupied:
            suffix += 1
            name = f'{stem}_{suffix}'
        occupied.add(name.casefold())
        node['name'] = name
    return root
