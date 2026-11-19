"""Graph-backed node and source addresses; raw acquisition stays in the backend."""
from dataclasses import dataclass
from types import SimpleNamespace
import warnings

from silk.circuit.placement_reference import LiteralVoltageName, _resolve_path
from silk.exceptions import CircuitError, SimulationError
from silk.vector_match import collect_vector_candidates, first_row


@dataclass(frozen=True)
class NodeCandidate:
    """One authored node or source in one placement, before address disambiguation."""
    components: tuple
    chain: tuple
    leaf: str
    source = None

    @property
    def designer(self):
        return '.'.join((*self.components, self.leaf))

    @property
    def address(self):
        if self.source is not None:
            layout = getattr(self.source, 'physical_layout', None)
            terms = layout().current_terms if layout is not None else ((self.source.instance_name, 1),)
            return tuple((_resolve_path(self.chain, card, follow_ports=False), sign)
                         for card, sign in terms)
        return LiteralVoltageName(_resolve_path(self.chain, self.leaf))


def node_candidates(module, *, sources=False, request=None):
    """Structural addresses; sources=None includes nodes and sources in one walk."""
    from silk.circuit.circuit_elements import ElementType
    if sources is not False:
        from silk.circuit.emitted_names import emitted_submodule_name
        from silk.sources._layout import source_instances

    def walk(mod, components=(), chain=(), active=(), remaining=request):
        if id(mod) in active:
            return
        if sources is not False:
            selected = (mod.sig_sources_dict if remaining is None else
                        {name: source for name, source in mod.sig_sources_dict.items()
                         if str(name).lower() == remaining.lower()})
            for name, source, _ in source_instances(selected) if selected else ():
                record = NodeCandidate(components, chain, name)
                object.__setattr__(record, 'source', source)
                yield record
                if remaining is not None:
                    break  # Preserve the existing case-equivalent source-key order.
        if sources is not True:
            for name in dict.fromkeys((*mod.net_dict, *(mod.pin_list or {}))):
                yield NodeCandidate(components, chain, name)
        # A complete source leaf cannot name a descendant placement.
        if sources is True and remaining is not None and '.' not in remaining:
            return
        elements = mod.elem_dict if sources is False else getattr(mod, 'elem_dict', {})
        for key, entry in elements.items():
            if entry.get('type') == ElementType.MODULE:
                tail = remaining
                if sources and remaining is not None:
                    prefixes = {str(key).lower(), 'x' + str(key).lower(),
                                emitted_submodule_name(str(key)).lower()}
                    tail = next((remaining[len(prefix) + 1:] for prefix in prefixes
                                 if remaining.lower().startswith(prefix + '.')), None)
                    if tail is None:
                        continue
                yield from walk(entry['device'], (*components, key),
                                (*chain, (mod, key, entry)), (*active, id(mod)), tail)

    normalized = request.replace(':', '.') if sources and request is not None else request
    return list(walk(module, remaining=normalized)) if module is not None else []


def _targets(records, *, ignore_unconnected=False):
    """Deduplicate by physical node, never by flattened designer name or definition."""
    targets = {}
    for record in records:
        try:
            address = record.address
        except CircuitError:
            if ignore_unconnected:
                continue  # An invalid designer reading cannot override raw data.
            raise
        # Literal leaf punctuation and hierarchy punctuation have different
        # Xyce forms even when their flattened dot spelling is identical.
        key = (tuple((card.casefold(), sign) for card, sign in address)
               if isinstance(address, tuple) else
               frozenset(form.casefold() for form in address.voltage_forms()))
        targets.setdefault(key, address)
    return list(targets.values())


def dut_hint(records, name):
    from silk.circuit.modules_results import DUT_ELEMENT_NAME
    suggested = f'{DUT_ELEMENT_NAME}.{name}'
    if any(record.designer == suggested for record in records):
        # A bare colon spelling enters device/terminal dispatch at probe().
        if ':' in suggested:
            suggested = f'v({suggested})'
        return (f' The DUT is placed as {DUT_ELEMENT_NAME!r}; '
                f'use probe({suggested!r}) for that node.')
    return ''


class NodeVoltageRequest(str):
    """Original scalar request carried through deferred/copy/selected Signal reads.

    No graph or samples are stored. Candidates are rebuilt against the reader's
    circuit, including for a retained corner or externally loaded rawfile.
    """
    def __new__(cls, name, *, known_designer=False):
        value = super().__new__(cls, name)
        value.known_designer = known_designer
        return value


def node_request(module, name):
    records = node_candidates(module)
    return NodeVoltageRequest(name, known_designer=any(r.designer == name for r in records))


def _warn(request, targets, *, signal_type='V'):
    if signal_type == 'I':
        alternatives = []
        for terms in targets:
            alternatives.append(' + '.join(
                ('' if sign == 1 else f'{sign} * ') + f'tb.probe({card!r}, \'I\')'
                for card, sign in terms))
        warnings.warn(
            f'Current request i({str(request).lower()}) also names a different source. '
            'The emitted reading is used; select the other current with '
            + ' or '.join(alternatives) + '.', UserWarning, stacklevel=4)
        return
    warnings.warn(
        f'Node request {str(request)!r} also matches designer spelling '
        f'{str(request)!r}. The emitted reading is used; select the designer '
        f'node with its emitted path {list(map(str, targets))!r}, or a placement net reference.',
        UserWarning, stacklevel=4)


def resolve_node_request(module, raw, request, analysis=None):
    """Resolve only typed node requests, then let the existing loader acquire data.

    Old raw ambiguity is passed untouched to that loader. Original text wins
    over a different designer target, including names found only in an external
    file. Structural addresses retain literal leaf and Xyce forms.
    """
    if not isinstance(request, NodeVoltageRequest):
        return request
    from silk.simulation.backend import select_plot
    plot = select_plot(raw.native, analysis)
    if plot is None:
        plot = getattr(raw.native, 'current_plot', raw.native)
    if not hasattr(plot, 'vectors'):
        return request
    from silk.circuit.modules_netlist import _save_list_inventory
    return resolve_node_inventory(plot, request, node_candidates(module),
                                  _save_list_inventory(module) if module is not None else [])


def resolve_node_inventory(plot, request, records, physical, *, signal_type='V', known_designer=False, _matches=None):
    """Scalar node semantics over already-enumerated circuit facts.

    The viewer uses this same proof without walking the graph for each alias.
    """
    def rows(target):
        if signal_type == 'I':
            terms = target if isinstance(target, tuple) else ((target, 1),)
            found = []
            first = first_row(plot)
            for card, sign in terms:
                key = (card, first)
                if _matches is None:
                    indices = collect_vector_candidates(plot, card, 'I', first=first)
                else:
                    if key not in _matches:
                        _matches[key] = collect_vector_candidates(plot, card, 'I', first=first)
                    indices = _matches[key]
                found.append((tuple(indices), sign))
            return tuple(found)
        return tuple(collect_vector_candidates(plot, target, first=first_row(plot)))

    first = first_row(plot)
    key = (str(request), first)
    if _matches is None:
        original = collect_vector_candidates(plot, str(request), signal_type, first=first)
    else:
        if key not in _matches:
            _matches[key] = collect_vector_candidates(plot, str(request), signal_type, first=first)
        original = _matches[key]
    if len(original) > 1:
        return request  # Preserve the backend's AmbiguousSignalError.
    matching = [r for r in records if r.designer == request]
    if original:
        designer = _targets(matching, ignore_unconnected=True)
        original_rows = ((tuple(original), 1),) if signal_type == 'I' else tuple(original)
        if designer and any(rows(target) != original_rows for target in designer):
            _warn(request, designer, signal_type=signal_type)
        return request

    # A structurally known emitted node still wins when omitted by a save list.
    inventory = SimpleNamespace(vectors=[SimpleNamespace(name=f'{signal_type.lower()}({a})') for a in physical])
    emitted = collect_vector_candidates(inventory, str(request), signal_type, first=0)
    designer = _targets(matching, ignore_unconnected=bool(emitted))
    if emitted:
        old = {str(physical[i]).casefold() for i in emitted}
        targets = ({str(card).casefold() for terms in designer for card, _ in terms}
                   if signal_type == 'I' else {str(a).casefold() for a in designer})
        if designer and old != targets:
            _warn(request, designer, signal_type=signal_type)
        return request
    # One structural current target needs no row-identity comparison: the
    # loader will check its operands once when acquiring them.
    if signal_type == 'I' and len(designer) == 1:
        return designer[0]
    # In a dot-separated result two structural forms may reach the same
    # physical row. Deduplicate that proven identity, retaining unsaved
    # candidates (absence is not evidence that two targets are identical).
    unique = {}
    for target in designer:
        matched = rows(target)
        complete = (all(indices for indices, _ in matched) if signal_type == 'I' else bool(matched))
        address = target if signal_type == 'I' else tuple(sorted(target.voltage_forms()))
        key = ('rows', matched) if complete else ('address', address)
        unique.setdefault(key, target)
    designer = list(unique.values())
    if len(designer) > 1:
        if signal_type == 'I':
            raise CircuitError(
                f'Source path {str(request)!r} is ambiguous: distinct physical current '
                f'terms {designer!r}. Use the explicit physical current cards.')
        raise CircuitError(
            f'Node path {str(request)!r} is ambiguous: distinct physical nodes '
            f'{list(map(str, designer))!r}. Use a placement net reference '
            f'(child(...).net(...)) or an emitted path.')
    if designer:
        return designer[0]
    if signal_type == 'I':
        if known_designer:
            raise SimulationError(f'Source {str(request)!r} is no longer present in this circuit; '
                               'declare a new source probe or use an explicit raw current.')
        return request
    hint = dut_hint(records, request)
    if hint:
        raise CircuitError(f'Node {str(request)!r} is absent from this bench.{hint}')
    if request.known_designer:
        raise CircuitError(f'Designer node {str(request)!r} is no longer present in this circuit; '
                           'obtain a new placement net reference or use an emitted path.')
    return request


def select_save_nodes(module, inventory, *, signals=None, exclude=None, max_depth=None):
    """Translate designer selectors to physical rows without widening old matches.

    Save inventories and selectors are bare node names. Exclusions match the
    full inventory, and globs/hierarchy selectors intentionally select sets.
    """
    from silk.signals.view.select import matches_selector, hier_parse, _selector_not_found_error
    from silk.signals.view.template import ViewError
    records = node_candidates(module)
    positions = {}
    for i, name in enumerate(inventory):
        positions.setdefault(name.casefold(), set()).add(i)
    where = "in this design's save inventory"

    def expand(selectors, kwarg):
        selected = set()
        for selector in selectors:
            emitted = {i for i, name in enumerate(inventory) if matches_selector(name, selector)}
            designer = set()
            for record in records:
                if matches_selector(record.designer, selector):
                    designer.update(positions.get(str(record.address).casefold(), ()))
            if emitted and designer and emitted != designer:
                _warn(selector, [inventory[i] for i in sorted(designer)])
            matches = emitted or designer
            if not matches:
                raise _selector_not_found_error(selector,
                    inventory + [r.designer for r in records], kwarg, where)
            selected.update(matches)
        return selected

    selected = set(range(len(inventory))) if signals is None else expand(signals, 'signals')
    if exclude:
        selected -= expand(exclude, 'exclude')
    if max_depth is not None:
        selected = {i for i in selected if len(hier_parse(inventory[i])['segments']) <= max_depth}
    if not selected:
        raise ViewError(f'signal selection (signals=/exclude=/max_depth=) matched zero '
                        f'signals {where} — nothing would be emitted')
    return [name for i, name in enumerate(inventory) if i in selected]


def _xyce_printable_nodes(module):
    """Proven electrical nodes with unambiguous Xyce print spelling.

    A bare add_net declaration or a port on an empty subcircuit does not
    create a simulator node. Follow primitive bindings through wired ports,
    retaining structural components so literal punctuation cannot turn into
    hierarchy or a differential-voltage expression.
    """
    import re
    token = re.compile(r'[A-Za-z0-9_!$+\-\[\]<>]+')
    supported = set()

    def include(address):
        hierarchy = getattr(address, '_hierarchy', ())
        prefix = '.'.join(hierarchy) + '.' if hierarchy else ''
        leaf = str(address)[len(prefix):]
        if (leaf.casefold() not in {'0', 'gnd', 'gnd!'}
                and all(token.fullmatch(part) for part in (*hierarchy, leaf))):
            supported.add(str(address).casefold())

    for record in node_candidates(module):
        owner = record.chain[-1][2]['device'] if record.chain else module
        net = owner.net_dict.get(record.leaf)
        if net is None or not any(pin.kind in {'device', 'passive', 'source'}
                                  for pin in net.pins):
            continue
        try:
            include(record.address)
        except CircuitError:
            continue  # Unconnected ports cannot prove a printable address.

    from silk.circuit.emitted_names import emitted_submodule_name
    from silk.circuit.ports import ElementType

    def generated(mod, hierarchy=()):
        for source in mod.sig_sources_dict.values():
            layout = getattr(source, 'physical_layout', None)
            node = layout().internal_node if layout is not None else None
            if node is not None:
                include(LiteralVoltageName('.'.join((*hierarchy, node)), hierarchy=hierarchy))
        for name, entry in mod.elem_dict.items():
            if entry.get('type') == ElementType.MODULE:
                generated(entry['device'], (*hierarchy, emitted_submodule_name(name)))

    generated(module)
    return supported


def default_output_nodes(module):
    """Default Xyce capture includes probe and declarative metric node requests.

    Resolve against a structural inventory using the same emitted/designer
    precedence as readback. An explicit save list remains authoritative.
    Unknown model-internal nodes cannot be proved printable and are refused.
    """
    nodes = list(module.net_dict)
    requests = [(request, f'Xyce node probe {str(request)!r}')
                for request in getattr(module, '_pending_node_probes', ())]
    requests.extend(getattr(module, '_pending_metric_nodes', ()))
    if module.simulator != 'xyce' or module._save_list is not None or not requests:
        return nodes
    from silk.circuit.modules_netlist import _save_list_inventory
    from silk.exceptions import NotSupportedError
    from silk.simulation import Noise
    analyses = list(module.sim_analysis_dict.values())
    if analyses and all(isinstance(analysis, Noise) for analysis in analyses):
        raise NotSupportedError(
            f'{requests[0][1]}: noise output contains spectra, '
            'not node-voltage vectors. Use a separate AC, DC, Op or Transient test.')
    inventory = _save_list_inventory(module)
    plot = SimpleNamespace(vectors=[SimpleNamespace(name=f'v({name})')
                                    for name in inventory], npoints=1)
    raw = SimpleNamespace(native=plot)
    printable = _xyce_printable_nodes(module)
    seen = {name.casefold() for name in nodes}
    for request, context in requests:
        try:
            target = resolve_node_request(module, raw, request)
            rows = collect_vector_candidates(plot, target, first=0)
            if len(rows) != 1 or inventory[rows[0]].casefold() not in printable:
                raise NotSupportedError(
                    'expected one connected design node with supported print syntax; found '
                    f'{[inventory[i] for i in rows]!r}. '
                    'Use a connected design node or an unambiguous placement reference; '
                    'unused declarations, opaque model internals, ground and literal punctuation '
                    'outside letters, digits, _, !, $, +, -, brackets and array angles '
                    'have no supported capture path.')
        except CircuitError as exc:
            raise NotSupportedError(f'{context}: {exc}') from exc
        name = inventory[rows[0]]
        if name.casefold() not in seen:
            nodes.append(name)
            seen.add(name.casefold())
    return nodes
