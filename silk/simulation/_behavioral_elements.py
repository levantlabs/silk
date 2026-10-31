"""Ideal-element approximations for the single-pass behavioral engine.

Classification is done on the flattened electrical connectivity, before any
model runs. R/L components with one driver carry that driver's waveform;
resistive dividers and multiply driven components require an electrical solve.
"""
from dataclasses import dataclass

from silk.sources.classes import CurrentSource

from silk.circuit.circuit_elements import PortDir
from silk.exceptions import NotSupportedError
from silk.utils._utils import _parse_si, _resolve_value


@dataclass
class AlgebraicModel:
    """Adapter for a wire or voltage equation in the existing scheduler."""
    gain: object = None
    outputs: tuple = ('out',)
    polarity: int = 1
    independent: bool = False
    path: str = ''
    terminals: tuple = ()
    waveform: object = None

    def _invoke_behavioral_model(self, inputs, t):
        if self.independent:
            return {'out': inputs['reference'] + self.polarity * self.waveform}
        if self.gain is None:
            return {port: inputs['input'] for port in self.outputs}
        value = (inputs['reference'] + _parse_si(_resolve_value(self.gain, 'VCVS gain'))
                 * self.polarity * (inputs['positive'] - inputs['negative']))
        return {'out': value}


def vcvs_block(source, resolve):
    return AlgebraicModel(source.gain), {
        'positive': (resolve(source.ctrl_pos), PortDir.IN),
        'negative': (resolve(source.ctrl_neg), PortDir.IN),
        'reference': (resolve(source.out_neg), PortDir.IN),
        'out': (resolve(source.out_pos), PortDir.OUT),
    }


def classify_passives(passives, blocks, sources, rails):
    """Return algebraic blocks and an ordered, per-element notice snapshot."""
    # Names may repeat across the legacy and element registries. Identity
    # follows the flattened entry, while diagnostics retain the user's name.
    identified = [(index, *entry) for index, entry in enumerate(passives)]
    driven = set(rails)
    driven.update(net for block in blocks.values()
                  for net, direction in block.pins.values()
                  if direction in (PortDir.OUT, PortDir.INOUT))
    # A resistive shunt on an undriven intermediate node is part of a
    # divider, not a load on an ideal driver's output.
    incidence = {}
    for key, path, kind, a, b in identified:
        if kind in ('R', 'L'):
            incidence.setdefault(a, []).append(path)
            incidence.setdefault(b, []).append(path)
    pending = []
    notices = {}
    for key, path, kind, a, b in identified:
        rail = a if a in rails else b if b in rails else None
        other = b if rail == a else a
        if kind in ('R', 'C') and rail is not None and (
                kind == 'C' or other in driven
                or len(incidence.get(other, ())) == 1):
            notices[key] = f"{path}: ignored {kind} load (ideal models do not load)"
        elif kind == 'C':
            refuse(path, 'series C needs an electrical solve')
        else:
            pending.append((key, path, kind, a, b))

    adjacency = {}
    for edge in pending:
        for net in edge[3:]:
            adjacency.setdefault(net, []).append(edge)
    visited = set()
    algebraic = []
    for first in pending:
        if first[0] in visited:
            continue
        edges, nets, todo = {}, set(), [first[3]]
        while todo:
            net = todo.pop()
            if net in nets:
                continue
            nets.add(net)
            for edge in adjacency[net]:
                edges[edge[0]] = edge
                todo.extend(edge[3:])
        visited.update(edges)
        drivers = nets & driven
        if len(drivers) != 1:
            refuse(', '.join(edge[1] for edge in edges.values()), 'divider or ambiguous R/L path needs an '
                   'electrical solve; expected one driver, found '
                   f'{len(drivers)} ({", ".join(sorted(drivers))})')
        root = next(iter(drivers))
        outputs = tuple(sorted(nets - {root}))
        # Port names must not collide with the adapter's input slot.
        pins = {'input': (root, PortDir.IN), **{
            f'out{i}': (net, PortDir.OUT) for i, net in enumerate(outputs)}}
        algebraic.append((first[0], first[1], AlgebraicModel(
            outputs=tuple(f'out{i}' for i in range(len(outputs)))), pins))
        for key, path, kind, _, _ in edges.values():
            notices[key] = f'{path}: wire ({kind}, voltage passes through)'
    return algebraic, tuple(notices[key] for key, *_ in identified)


def refuse(path, reason):
    from silk.spice_names import designer_component
    path = designer_component(path)       # display only: instance parts as [k]
    raise NotSupportedError(
        f"Element '{path}': {reason}; the behavioral engine cannot evaluate "
        "this topology. Give the enclosing block a behavioral model or use "
        "a SPICE backend.")


def source_refuse(path, positive, negative, reason):
    """A capability refusal is not a verdict about electrical validity."""
    refuse(path, f"source terminals positive={positive!r}, negative={negative!r}: "
           f"{reason}. Use one Python model for the enclosing network or SPICE")


def orient_voltage_sources(controls, blocks, sources, rails, passives):
    """Orient voltage equations from proven producers, never numeric defaults.

    The preliminary wire reachability only plans orientation. The ordinary
    passive classifier still decides whether a path really is a wire; final
    eligibility is checked again against those actual adapters below.
    """
    equations = []
    for path, src, positive, negative in sources:
        if not isinstance(src, CurrentSource):
            equations.append((path, AlgebraicModel(independent=True), {
                'reference': (negative, PortDir.IN),
                'out': (positive, PortDir.OUT)}))
    equations.extend(controls)
    owned = {'0', 'gnd'}
    modeled = {net for block in blocks.values()
               for net, direction in block.pins.values() if direction == PortDir.OUT}
    owned.update(modeled)
    for path, device, pins in equations:
        device.path = path
        device.terminals = (pins['out'][0], pins['reference'][0])
    proven = set(owned)
    pending = list(equations)
    ordered = []
    while pending:
        # A shunt to a declared rail is not a voltage-solving operation.
        changed = True
        while changed:
            changed = False
            for _, kind, a, b in passives:
                if kind not in ('R', 'L') or (kind == 'R' and (a in rails or b in rails)):
                    continue
                if a in proven or b in proven:
                    before = len(proven)
                    proven.update((a, b))
                    changed |= len(proven) != before
        ready = [item for item in pending if
                 any(item[2][port][0] in proven for port in ('out', 'reference'))]
        if not ready:
            path, device, _ = pending[0]
            source_refuse(path, *device.terminals,
                          'unresolved voltage reference requires an electrical solve')
        for path, device, pins in ready:
            positive, negative = device.terminals
            if positive in proven and negative in proven:
                source_refuse(path, positive, negative,
                              'E output terminals are both already driven' if not device.independent
                              else 'both terminals already owned; constraint validation '
                              'needs an electrical solve')
            if positive in proven:
                pins['out'] = (negative, PortDir.OUT)
                pins['reference'] = (positive, PortDir.IN)
                device.polarity = -1
            output = pins['out'][0]
            owned.add(output)
            proven.add(output)
            ordered.append((path, device, pins))
            pending.remove((path, device, pins))
    return ordered


def validate_voltage_references(blocks):
    """Carry reference eligibility through actual R/L wire adapters.

    Missing POWER/GROUND defaults, INOUT-only declarations and current-valued
    arrays are deliberately absent from this producer set.
    """
    proven = {'0', 'gnd'}
    for block in blocks.values():
        if not isinstance(block.device, AlgebraicModel) or block.device.terminals:
            proven.update(net for net, direction in block.pins.values()
                          if direction == PortDir.OUT)
    changed = True
    while changed:
        changed = False
        for block in blocks.values():
            if (isinstance(block.device, AlgebraicModel) and not block.device.terminals
                    and block.pins['input'][0] in proven):
                before = len(proven)
                proven.update(net for net, direction in block.pins.values()
                              if direction == PortDir.OUT)
                changed |= before != len(proven)
    writers = {}
    for block in blocks.values():
        for net, direction in block.pins.values():
            if direction in (PortDir.OUT, PortDir.INOUT):
                writers.setdefault(net, []).append(block)

    def require(net, source, seen):
        if net not in proven:
            source_refuse(source.path, *source.terminals,
                          f'reference/control {net!r} has no proven voltage producer; '
                          'an electrical solve is required')
        producers = writers.get(net, [])
        ground_written = net in ('0', 'gnd') and any(writers.get(n) for n in ('0', 'gnd'))
        if ground_written or len(producers) > 1:
            source_refuse(source.path, *source.terminals,
                          f'reference/control {net!r} has conflicting voltage ownership; '
                          'an electrical solve is required')
        if producers:
            producer = producers[0]
            if isinstance(producer.device, AlgebraicModel) and not producer.device.terminals:
                if net in seen:
                    source_refuse(source.path, *source.terminals,
                                  'reference wire cycle needs an electrical solve')
                require(producer.pins['input'][0], source, seen | {net})

    for block in blocks.values():
        device = block.device
        if isinstance(device, AlgebraicModel) and device.terminals:
            for net, direction in block.pins.values():
                if direction in (PortDir.IN, PortDir.OUT):
                    require(net, device, set())
