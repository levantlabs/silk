"""Direct wiring subscribers for live device objects."""
from dataclasses import dataclass
import weakref
import warnings


@dataclass(frozen=True)
class Holder:
    registry: str
    slot: str
    aliases: tuple
    kind: str


def supported(device):
    if isinstance(device, dict) or 'to_spice' in vars(device):
        return False
    from silk.elements.base import AnalogElement
    from silk.sources.classes import SourceBase, _builtin_card_fields
    if isinstance(device, AnalogElement):
        emitter = getattr(type(device), 'to_spice', None)
        return getattr(emitter, '__module__', None) == 'silk.elements.factory'
    return isinstance(device, SourceBase) and bool(_builtin_card_fields(type(device)))


def register(module, registry, slot, device):
    if not supported(device):
        return
    from silk.elements.base import AnalogElement
    if registry == 'passives_dict':
        aliases = tuple(zip(type(device)._pin_names, ('pos', 'neg')))
        kind = 'passive'
    elif isinstance(device, AnalogElement):
        aliases = tuple((p, p) for p in device._pin_names)
        kind = 'device'
    else:
        aliases = tuple((p, p) for p in device._node_terminals)
        kind = 'source'
    if kind == 'source':
        source_context(module)
    holders = vars(device).get('_wiring_holders')
    if holders is None:
        holders = weakref.WeakKeyDictionary()
        object.__setattr__(device, '_wiring_holders', holders)
    holders[module] = Holder(registry, slot, aliases, kind)


def retire(module, registry, slot, device):
    holders = vars(device).get('_wiring_holders', {})
    record = holders.get(module)
    if record is not None and (record.registry, record.slot) == (registry, slot):
        del holders[module]


def change_pins(device, changes, commit):
    """Accept the complete edit, warn on direct sharing, then commit or undo."""
    holders = list(vars(device).get('_wiring_holders', {}).items())
    plans = []
    for module, holder in holders:
        aliases = dict(holder.aliases)
        for pin, value in changes.items():
            if pin in aliases:
                module._element_mgr._accept_net(value)
                plans.append((module, holder, aliases[pin], value))
    from silk.elements.base import AnalogElement
    for pin, value in changes.items():
        old = device.nets.get(pin) if isinstance(device, AnalogElement) else getattr(device, pin, None)
        if old != value and len(holders) > 1:
            warnings.warn(f'{device.name}.{pin}: {old!r} -> {value!r} changes '
                          f'all {len(holders)} modules holding this object.',
                          UserWarning, stacklevel=3)
    undos = []
    # Canonicalization caches are bookkeeping too: rollback must restore them.
    snapshots = [(m._element_mgr, m._element_mgr._snapshot_net_name_state())
                 for m, _ in holders]
    try:
        from silk.circuit.net import PortBinding
        from silk.circuit.modules_structure import _bind_source_terminal
        for module, holder, port, value in plans:
            undo = []
            undos.append((module._element_mgr, undo))
            if holder.kind == 'source':
                _bind_source_terminal(module, holder.slot, port, value, undo=undo)
            else:
                module._element_mgr._bind_pin_to_net(value, PortBinding(
                    net=value, instance=holder.slot, port=port,
                    direction=None, role=None, kind=holder.kind), undo=undo)
        commit()
    except BaseException:
        for manager, undo in reversed(undos):
            manager._undo_bindings(undo)
        for manager, snapshot in snapshots:
            manager._restore_net_name_state(snapshot)
        raise


def detached_state(device):
    state = dict(vars(device))
    state.pop('_wiring_holders', None)
    if supported(device):
        state.pop('_module', None)
    state.pop('_module_ref', None)
    return state


def restore_module(module):
    """Rebuild only subscribers actually contained in a restored module."""
    manager = module._element_mgr
    if manager._module_ref is not module:
        return  # an intermediate shallow emission view still holds live objects
    token = vars(module).get('_source_context')
    if token is not None:
        token.bind(module)
    for source in module.sig_sources_dict.values():
        if supported(source):
            context = vars(source).get('_restore_context')
            if context is not None:
                context.subscribe(source)
            if getattr(source, '_module', None) is None:
                object.__setattr__(source, '_module', module)
    for slot, entry in module.elem_dict.items():
        register(module, 'elem_dict', slot, entry['device'])
    for registry in ('passives_dict', 'sig_sources_dict'):
        for slot, device in getattr(module, registry).items():
            if not isinstance(device, dict):
                register(module, registry, slot, device)


class _SourceContext:
    """Copy-local identity for restoring a source's selected emission context."""
    def __init__(self):
        self.module = None
        self.waiting = weakref.WeakSet()

    def __reduce__(self):
        return type(self), ()

    def subscribe(self, source):
        self.waiting.add(source)
        if self.module is not None and self.module() is not None:
            object.__setattr__(source, '_module', self.module())

    def bind(self, module):
        self.module = weakref.ref(module)
        for source in self.waiting:
            object.__setattr__(source, '_module', module)


def source_context(module):
    token = vars(module).get('_source_context')
    if token is None:
        token = _SourceContext()
        module._source_context = token
    return token
