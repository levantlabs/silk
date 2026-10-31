"""Permissive source-to-ground wiring inspection, without an electrical solve.

All physical pins of each known element (including sense pins, capacitors,
transistors and instance ports) belong to one wiring group. This intentionally
overapproximates connectivity: passing says nothing about voltages, conduction,
singularity or the backend's ability to evaluate a circuit.
"""
from dataclasses import dataclass
from inspect import getattr_static
from types import MethodType

from silk.circuit.circuit_elements import ElementType
from silk.circuit.emitted_names import emitted_submodule_name
from silk.circuit.ground_port import _require_entered_ground_interfaces
from silk.elements.base import AnalogElement
from silk.exceptions import SimulationError
from silk.sources.classes import (
    SourceBase, _IndependentSource,
    CurrentControlledVoltageSource, CurrentControlledCurrentSource,
)
from silk.sources._layout import WAVEFORM_SLOTS


@dataclass(frozen=True)
class DisconnectedSource:
    """One authored source and its resolved, ungrounded wiring group."""
    source: str
    nets: tuple[str, ...]


@dataclass(frozen=True)
class TopologyInspection:
    """Immutable findings; an empty result is not electrical certification.

    ``unchecked`` records opaque content whose possible effects prevent
    refusal. Unbounded custom SPICE text can touch any group, including via
    global nets, so its exemption must be global rather than guessed from pins.
    """
    disconnected: tuple[DisconnectedSource, ...]
    unchecked: tuple[str, ...] = ()


def _custom_method(obj, name):
    """Inspect a renderer without invoking it or a descriptor."""
    method = getattr_static(obj, name, None)
    if method is None:
        return False
    method = getattr(method, '__func__', method)
    return not getattr(method, '__module__', '').startswith('silk.')


def _source_hook_override(source):
    """Built-in outer emission also requires built-in delegated decisions.

    Static lookup handles class overrides, bound instance methods and renderer
    properties without executing selection, layout or terminal callbacks.
    """
    hooks = {name: getattr_static(SourceBase, name) for name in
             ('_node', '_validate_terminal', 'instance_name')}
    if isinstance(source, _IndependentSource):
        hooks.update({name: getattr_static(_IndependentSource, name)
                      for name in ('selected_waveform', 'physical_layout')})
    if isinstance(source, (CurrentControlledVoltageSource, CurrentControlledCurrentSource)):
        hooks['_resolve_ctrl_vname'] = getattr_static(SourceBase, '_resolve_ctrl_vname')
    for name, expected in hooks.items():
        actual = getattr_static(source, name, None)
        if isinstance(actual, MethodType):
            if actual.__self__ is not source:
                return name
            actual = actual.__func__
        if actual is not expected:
            return name
    return None


def _custom_includes(pdk, backend):
    """Additional includes are opaque; shipped model associations are not.

    Compare manifest-relative include names without opening installed models,
    allowing ordinary installations at different PDK_ROOT locations.
    """
    if pdk is None:
        return False
    includes = getattr(pdk, 'extra_includes', ()) or ()
    key = 'extra_includes'
    if backend == 'xyce' and getattr(pdk, 'xyce_extra_includes', None) is not None:
        includes, key = pdk.xyce_extra_includes, 'xyce_extra_includes'
    if not includes:
        return False
    from silk.pdk import registry
    try:
        declared = registry.manifest(pdk.pdk_name)['library'].get(key, ())
    except (registry.PDKRegistryError, KeyError, TypeError):
        return True
    return any(not any(str(path) == name or str(path).endswith('/' + name)
                       for name in declared) for path in includes)


def _multiline(values):
    """Recognize literal extra cards without resolving parameters/callbacks."""
    return any((isinstance(value, str) and ('\n' in value or '\r' in value))
               or (isinstance(value, dict) and _multiline(value.values()))
               for value in values)


def inspect_topology(module):
    """Inspect the selected backend's wiring without emitting or running it.

    Visits actual placements, not the definition registry. Behavioral modeled
    leaves contribute their ports instead of their replaced physical contents.
    Every entered interface, including modeled leaves, refuses formal ``0``
    and case-insensitive ``gnd`` before resolving actual connected nets.
    Shipped PDK devices contribute pins without reading their model libraries.
    Custom emitters/text are not executed or parsed into invented connections.
    """
    backend = module.simulator
    supported = backend in ('ngspice', 'xyce', 'behavioral')
    parent, labels, source_groups, unchecked = {}, {}, [], []
    if not supported:
        unchecked.append(f'unsupported backend {backend!r}')
    ground = ((), '0')

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(nodes):
        nodes = list(nodes)
        if nodes:
            root = find(nodes[0])
            for node in nodes[1:]:
                parent[find(node)] = root
        return nodes

    def visit(mod, path, ports, ancestors):
        _require_entered_ground_interfaces(mod)
        if id(mod) in ancestors:
            unchecked.append('recursive module placement')
            return
        ancestors = (*ancestors, id(mod))
        if not supported:
            # The interface rule is backend-independent. Retain the topology
            # exemption: do not resolve nets or inspect any electrical object.
            for entry in mod._element_mgr.elem_dict.values():
                if entry['type'] == ElementType.MODULE:
                    visit(entry['device'], path, {}, ancestors)
            return

        def resolve(net):
            # Backend global names must be handled BEFORE hierarchy prefixes
            # and aliases. A GROUND role alone never makes vss global ground.
            folded = str(net).casefold()
            is_ground = folded == '0' or (
                backend == 'ngspice' and folded == 'gnd') or (
                backend == 'behavioral' and net in ('gnd', '0'))
            if is_ground:
                node = ground
            elif folded in ports:
                return ports[folded]
            else:
                node = (path, folded)
            labels.setdefault(node, '.'.join((*path, str(net))))
            if node == ground:
                labels[node] = '0'
            return node

        prefix = '.'.join(path)
        if backend != 'behavioral':
            from silk.circuit.spice_netlister import PySpiceNetlister
            from silk.circuit.xyce_netlister import XyceNetlister
            # Child emission uses the child's selected/cached renderer, not
            # necessarily the root dialect. A stale cache would be replaced.
            # Never trigger the lazy netlister getter during inspection.
            netlister = vars(mod).get('_netlister')
            if (netlister is not None and vars(mod).get('_netlister_sim') == mod.simulator
                    and type(netlister) not in (PySpiceNetlister, XyceNetlister)):
                unchecked.append(f'{prefix or "<root>"}: replacement netlister')
        if mod.custom_lines:
            unchecked.append(f'{prefix or "<root>"}: custom lines (unbounded touch set)')
        for method in ('generate_netlist', 'generate_subckt'):
            if _custom_method(mod, method):
                unchecked.append(f'{prefix or "<root>"}: custom Module.{method}')
        if _custom_method(mod.pdk_details, 'generate_netlist_entry'):
            unchecked.append(f'{prefix or "<root>"}: custom PDK emitter')
        if backend != 'behavioral' and _custom_includes(mod.pdk_details, backend):
            unchecked.append(f'{prefix or "<root>"}: external model include (unbounded touch set)')
        for analysis in mod.sim_analysis_dict.values():
            if _custom_method(analysis, 'to_spice'):
                unchecked.append(f'{prefix or "<root>"}: custom analysis emitter')
        if _multiline(mod.simulator_options.values()):
            unchecked.append(f'{prefix or "<root>"}: multiline option content')

        def element(name, obj, nets, is_source=False):
            who = '.'.join((*path, name))
            if _custom_method(obj, 'to_spice'):
                unchecked.append(f'{who}: custom emitter (unbounded touch set)')
                return
            if isinstance(obj, SourceBase):
                hook = _source_hook_override(obj)
                if hook is not None:
                    unchecked.append(f'{who}: custom {hook} hook (unbounded touch set)')
                    return
            if _multiline(vars(obj).values()):
                unchecked.append(f'{who}: multiline element content (unbounded touch set)')
                return
            # Never call selected_waveform(), to_spice() or value callbacks.
            for slot in WAVEFORM_SLOTS:
                wave = vars(obj).get(slot)
                if wave is not None and _custom_method(wave, 'to_spice'):
                    unchecked.append(f'{who}: custom waveform emitter (unbounded touch set)')
                    return
            nodes = join(resolve(net) for net in nets if net is not None)
            if is_source and nodes:
                source_groups.append((who, nodes[0]))

        for name, src in mod.sig_sources_dict.items():
            element(name, src, (getattr(src, pin) for pin in src._node_terminals), True)
        for name, obj in mod.passives_dict.items():
            nets = obj.nets.values() if isinstance(obj, AnalogElement) else (obj.pos, obj.neg)
            element(name, obj, nets)
        for name, entry in mod.elem_dict.items():
            obj = entry['device']
            if entry['type'] == ElementType.MODULE:
                pins = entry.get('pin_list', obj.pin_list) or {}
                aliases = {port.casefold(): resolve(pin.net) for port, pin in pins.items()
                           if pin.net is not None}
                join(aliases.values())
                if backend == 'behavioral' and obj._behavioral_model is not None:
                    continue
                visit(obj, (*path, emitted_submodule_name(name)), aliases, ancestors)
            elif isinstance(obj, AnalogElement):
                element(name, obj, obj.nets.values(), obj._spice_prefix in ('V', 'I', 'E', 'G', 'F', 'H', 'B'))
            elif isinstance(obj, SourceBase):
                element(name, obj, (getattr(obj, pin) for pin in obj._node_terminals), True)
            else:
                element(name, obj, (pin.net for pin in obj.pin_list.values()),
                        obj.spice_prefix in ('V', 'I', 'E', 'G', 'F', 'H', 'B'))

    visit(module, (), {}, ())
    # Unbounded text/emitters may introduce .global wiring anywhere. Exempt
    # every possibly touched group; known ordinary devices never do this.
    if unchecked:
        return TopologyInspection((), tuple(dict.fromkeys(unchecked)))
    grouped = {}
    for node in parent:
        grouped.setdefault(find(node), []).append(labels[node])
    root = find(ground)
    disconnected = tuple(DisconnectedSource(who, tuple(sorted(grouped[find(node)])))
                         for who, node in source_groups if find(node) != root)
    return TopologyInspection(disconnected)


def require_ground_connections(module):
    """Refuse unsafe ground interfaces, then non-exempt ungrounded source groups."""
    result = inspect_topology(module)
    if result.disconnected:
        from silk.spice_names import designer_component
        details = '; '.join(f"Source {designer_component(item.source)!r}: group "
                            f"({', '.join(map(designer_component, item.nets))}) "
                            'has no connection to ground' for item in result.disconnected)
        error = SimulationError(
            details + '. Wire the group to actual ground, directly or through '
            'a connected component such as a resistor or capacitor. '
            'This check concerns connectivity only.')
        error.topology = result
        raise error
    return result
