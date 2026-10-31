"""Instance wiring and emission-time identity for live module references."""
from silk.circuit.emitted_names import emitted_submodule_name
from dataclasses import dataclass, replace
from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps

from silk.circuit.ports import ElementType
from silk.exceptions import CircuitError


_definition_names = ContextVar('silk_definition_names', default=None)


# The current body has its own placement-local device and child selections.
_fault_node = ContextVar('silk_fault_node', default=None)


@contextmanager
def _fault_body(node):
    token = _fault_node.set(node)
    try:
        yield
    finally:
        _fault_node.reset(token)


def _fault_definition_paths():
    """Derive generated-definition diagnostic paths from the active overlay."""
    paths = {}

    def walk(node, prefix):
        for instance, child in node['children'].items():
            path = prefix + emitted_submodule_name(instance) + '.'
            paths[child['name'].casefold()] = path
            walk(child, path)

    node = _fault_node.get()
    if node:
        walk(node, '')
    return paths


def resolved_name(name):
    """Resolve a spelling in the current deck's case-insensitive namespace."""
    return (_definition_names.get() or {}).get(name.casefold(), name)


def with_definition_scope(fn):
    """Keep one name resolver for the complete recursive emission call."""
    @wraps(fn)
    def emit(module, *args, **kwargs):
        if _definition_names.get() is not None:
            return fn(module, *args, **kwargs)
        token = _definition_names.set({})
        try:
            _definition_names.set(validate_definitions(module))
            return fn(module, *args, **kwargs)
        finally:
            _definition_names.reset(token)
    return emit


def check_placement(parent, child):
    """Refuse reference cycles before committing a placement."""
    pending, seen = [child], set()
    while pending:
        current = pending.pop()
        if current is parent:
            raise CircuitError('Recursive module placement would create a circular circuit.')
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(entry['device'] for entry in current._element_mgr.elem_dict.values()
                       if entry['type'] == ElementType.MODULE)


@contextmanager
def _placement_wiring_transaction(module):
    """Undo binding edits and declaration synchronization without replacing objects."""
    manager = module._element_mgr
    declarations = []
    for entry in manager.elem_dict.values():
        if entry['type'] != ElementType.MODULE:
            continue
        pins = entry.get('pin_list')
        declarations.append((entry, pins, dict(pins) if pins is not None else None,
                             [(pin, dict(vars(pin))) for pin in (pins or {}).values()]))
    nets = [(net, net.pins, list(net.pins),
             [(binding, dict(vars(binding))) for binding in net.pins])
            for net in manager.net_dict.values()]
    reverse = dict(manager._pin_to_net)
    casefold = dict(manager._net_casefold)
    warned = set(manager._case_merge_warned)
    undo = []
    try:
        yield undo
    except BaseException:
        manager._undo_bindings(undo)
        for entry, pins, members, fields in declarations:
            if pins is None:
                entry.pop('pin_list', None)
            else:
                pins.clear()
                pins.update(members)
                entry['pin_list'] = pins
            for pin, state in fields:
                vars(pin).clear()
                vars(pin).update(state)
        for net, pins, members, fields in nets:
            pins[:] = members
            net.pins = pins
            for binding, state in fields:
                vars(binding).clear()
                vars(binding).update(state)
        manager._pin_to_net.clear()
        manager._pin_to_net.update(reverse)
        manager._net_casefold.clear()
        manager._net_casefold.update(casefold)
        manager._case_merge_warned.clear()
        manager._case_merge_warned.update(warned)
        raise


def sync_instance_pins(module):
    """Refresh port declarations while retaining each placement's wiring.

    Port direction/role/domain belong to the live definition. Only the net
    binding belongs to a placement. Refresh existing PortBindings in place
    so annotation provenance survives metadata changes.
    """
    import copy
    manager = module._element_mgr
    if not manager.module_dict:
        return
    # Validate removals before updating any wiring or pin metadata.
    for instance, entry in manager.elem_dict.items():
        if entry['type'] != ElementType.MODULE:
            continue
        declared = getattr(entry['device'], 'pin_list', None)
        if declared is None:
            continue
        for port, pin in entry.get('pin_list', {}).items():
            if port not in declared and pin.net is not None:
                raise CircuitError(
                    f"Instance {instance!r}: wired port {port!r} was removed "
                    f"from the placed definition (connected to {pin.net!r}). "
                    "Restore the port or explicitly rebuild the placement wiring.")
    for instance, entry in manager.elem_dict.items():
        if entry['type'] != ElementType.MODULE:
            continue
        declared = getattr(entry['device'], 'pin_list', None)
        if declared is None:
            continue
        pins = entry.setdefault('pin_list', {})
        for port in tuple(pins):
            if port not in declared:
                for net in manager.net_dict.values():
                    net.pins[:] = [binding for binding in net.pins
                                   if (binding.instance, binding.port) != (instance, port)]
                manager._pin_to_net.pop((instance, port), None)
                del pins[port]
        for port, source in declared.items():
            if port not in pins:
                pins[port] = copy.copy(source)
                pins[port].net = None
            pin = pins[port]
            changed = False
            for attr in ('direction', 'role', 'domain', 'role_inferred'):
                value = getattr(source, attr)
                if getattr(pin, attr) != value:
                    setattr(pin, attr, value)
                    changed = True
            if changed and pin.net in manager.net_dict:
                for binding in manager.net_dict[pin.net].pins:
                    if (binding.instance, binding.port) != (instance, port):
                        continue
                    for attr in ('direction', 'role', 'domain', 'role_inferred'):
                        value = getattr(pin, attr)
                        setattr(binding, attr, getattr(value, 'value', value))


def refresh_definitions(module):
    """Rebuild the immediate definition index from current placed objects."""
    from silk.circuit.circuit_elements import (
        register_module_definition, structural_definition_name)
    manager = getattr(module, '_element_mgr', None)
    if manager is None:
        return
    definitions = manager.module_dict
    registrations = {name: entry for name, entry in definitions.items()
                     if 'numberInstantiated' not in entry}
    for entry in module.elem_dict.values():
        if entry['type'] != ElementType.MODULE:
            continue
        name, _, _ = register_module_definition(registrations, entry['device'])
        entry['module_name'] = name
        entry['structural_name'] = structural_definition_name(registrations, name)
    definitions.clear()
    definitions.update(registrations)


def instance_line(name, entry):
    """Emit an X line using the placement's wiring and current definition."""
    pins = entry.get('pin_list', entry['device'].pin_list)
    line = emitted_submodule_name(name) + ' '
    for port, pin in pins.items():
        if pin.net is None:
            raise CircuitError(
                f'ERROR: In module instantiation {name}, net for pin {port} is undefined')
        line += pin.net + ' '
    from silk.utils.deck_text import refuse_line_breaks
    node = _fault_node.get()
    child = node['children'].get(name) if node else None
    definition = (child['name'] if child else
                  resolved_name(entry.get('module_name', entry['device'].module_name)))
    return line + refuse_line_breaks(
        definition,
        who=f'subcircuit name of instance {name!r}') + '\n'


def validate_definitions(module):
    """Check the whole deck's flat subcircuit namespace against live bodies."""
    from silk.circuit.circuit_elements import (
        register_module_definition, structural_definition_name)
    registry, paths, visited, active = {}, {}, set(), set()

    def walk(parent, path):
        if id(parent) in active:
            raise CircuitError(f'Recursive module placement at {path!r}; a circuit must be acyclic.')
        if id(parent) in visited:
            return
        active.add(id(parent))
        for inst, entry in parent.elem_dict.items():
            if entry['type'] != ElementType.MODULE:
                continue
            child = entry['device']
            child_path = f'{path}.x{inst}' if path else f'x{inst}'
            walk(child, child_path)
            try:
                name, _, _ = register_module_definition(registry, child, validate=True)
            except CircuitError as exc:
                previous = ', '.join(paths.values())
                raise CircuitError(
                    f'Instances {previous} and {child_path}: {exc}') from exc
            paths.setdefault(name, child_path)
            entry['module_name'] = name
            entry['structural_name'] = structural_definition_name(registry, name)
        active.remove(id(parent))
        visited.add(id(parent))

    walk(module, '')
    if module.module_name is not None:
        try:
            register_module_definition(registry, module, validate=True,
                                       definition_name=module.module_name)
        except CircuitError as exc:
            raise CircuitError(f"Root and instances {', '.join(paths.values())}: {exc}") from exc
    return {name.casefold(): name for name in registry}


@dataclass(frozen=True)
class ModuleInstance:
    """One placement; identity is immutable and ``module`` is a live handle.

    Yielded by :meth:`silk.circuit.Module.iter_instances`. Repeated placements of one
    module give distinct records that share the same ``module`` object.

    Parameters
    ----------
    path : str
        Designer instance path, dot-separated; ``''`` for the root.
    spice_path : str
        The same path as spelled in the emitted deck and rawfile (instance
        components carry SPICE spelling); ``''`` for the root.
    module : Module
        The live module placed here, not a copy.
    owner_path : str
        Slash-separated path identifying the owning placement; ``'$'`` for
        the root.
    module_name : str or None
        Resolved definition name; ``None`` for a module with no
        ``module_name``.
    placed : bool, default True
        ``False`` for the root and for legacy organizational registrations,
        ``True`` for placements made from an element entry.
    """

    #: Designer instance path, dot-separated; ``''`` for the root.
    path: str
    #: The path as spelled in the emitted deck and rawfile; ``''`` for the root.
    spice_path: str
    #: The live module placed here (not a copy).
    module: object
    #: Slash-separated owner path; ``'$'`` for the root.
    owner_path: str
    #: Resolved definition name, or ``None`` when the module has none.
    module_name: str | None
    #: ``False`` for the root and legacy organizational registrations.
    placed: bool = True


def instance_children(module):
    """The runner and public walk share the same ordered hierarchy edges."""
    refresh_definitions(module)
    for instance, entry in module.elem_dict.items():
        if entry['type'] == ElementType.MODULE:
            yield instance, entry['device'], entry['module_name'], True
    for instance, entry in module.module_dict.items():
        if 'numberInstantiated' not in entry:
            yield instance, entry['device'], entry['device'].module_name, False


def walk_instances(module, *, include_self=False):
    """Walk placements in depth-first insertion order, without deduplicating."""
    from silk.circuit.modules_structure import _child_spec_owner_path

    def walk(current, path, spice_path, owner_path, name, ancestors, placed):
        if id(current) in ancestors:
            raise CircuitError(f'Recursive module placement at {path!r}.')
        yield ModuleInstance(path, spice_path, current, owner_path, name, placed)
        for instance, child, child_name, child_placed in instance_children(current):
            child_path = f'{path}.{instance}' if path else instance
            child_spice_path = f'{spice_path}.{emitted_submodule_name(instance)}' if spice_path else emitted_submodule_name(instance)
            yield from walk(child, child_path, child_spice_path,
                            _child_spec_owner_path(owner_path, instance),
                            child_name, ancestors | {id(current)}, child_placed)

    instances = walk(module, '', '', '$', module.module_name, set(), False)
    if not include_self:
        next(instances)
    names = {}
    for instance in instances:
        if instance.module_name is not None and instance.placed:
            name = names.setdefault(instance.module_name.casefold(), instance.module_name)
            instance = replace(instance, module_name=name)
        yield instance
