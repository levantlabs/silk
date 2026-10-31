"""Read-only agreement checks for supported authored wiring."""
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps

from silk.exceptions import CircuitError

_scope = ContextVar('silk_wiring_validation', default=None)


def with_wiring_scope(fn):
    """Share one visited set for a complete recursive emission invocation."""
    @wraps(fn)
    def checked(module, *args, **kwargs):
        if _scope.get() is not None:
            assert_consistent(module, _scope.get())
            return fn(module, *args, **kwargs)
        token = _scope.set(set())
        try:
            assert_consistent(module, _scope.get())
            return fn(module, *args, **kwargs)
        finally:
            _scope.reset(token)
    return checked


def _check_local(module):
    from silk.circuit.ports import ElementType
    from silk.elements.base import AnalogElement, validate_net_token
    from silk.circuit.object_holders import supported

    canonical = {name.casefold(): name for name in module.net_dict}
    expected = {}
    slots = set()
    def add(slot, pins, kind):
        slots.add(slot)
        for port, net in pins:
            if net is not None:
                validate_net_token(net, who=f'object {slot!r} pin {port!r}')
            expected[slot, port] = (canonical.get(net.casefold(), net) if net is not None else None, kind)

    for slot, entry in module.elem_dict.items():
        obj = entry['device']
        if entry['type'] == ElementType.MODULE:
            add(slot, ((p, pin.net) for p, pin in (entry.get('pin_list') or {}).items()), 'submodule')
        elif isinstance(obj, AnalogElement) and supported(obj):
            add(slot, ((p, obj.nets.get(p)) for p in obj._pin_names), 'device')
    for slot, obj in module.passives_dict.items():
        if isinstance(obj, AnalogElement) and supported(obj):
            add(slot, zip(('pos', 'neg'), (obj.nets.get(p) for p in obj._pin_names)), 'passive')
    for slot, obj in module.sig_sources_dict.items():
        if not isinstance(obj, dict) and supported(obj):
            add(slot, ((p, getattr(obj, p, None)) for p in obj._node_terminals), 'source')

    actual = defaultdict(list)
    for net_name, net in module.net_dict.items():
        validate_net_token(net.name, who=f'Module net {net_name!r}', kind='net')
        for pin in net.pins:
            if pin.instance in slots:
                actual[pin.instance, pin.port].append((net_name, net.name, pin.net, pin.kind))
    reverse = module._element_mgr._pin_to_net
    keys = set(expected) | set(actual) | {k for k in reverse if k[0] in slots}
    for key in sorted(keys):
        want, kind = expected.get(key, (None, None))
        indexed = reverse.get(key)
        pins = actual.get(key, [])
        valid = (key not in reverse and not pins) if want is None else (
            indexed == want and pins == [(want, want, want, kind)])
        if not valid:
            name = module.module_name or module.circuit_name or module.netlistname
            raise CircuitError(f'Module {name!r}: object {key[0]!r} pin {key[1]!r} '
                               f'has inconsistent wiring: authored={want!r}, '
                               f'_pin_to_net={indexed!r}, Net.pins={pins!r}. '
                               'Restore the binding before lint or deck emission.')


def assert_consistent(module, seen=None):
    from silk.circuit.ports import ElementType
    seen = set() if seen is None else seen
    if id(module) in seen:
        return
    seen.add(id(module))
    _check_local(module)
    for entry in module.elem_dict.values():
        if entry['type'] == ElementType.MODULE:
            assert_consistent(entry['device'], seen)
    for entry in module.module_dict.values():
        assert_consistent(entry['device'], seen)
