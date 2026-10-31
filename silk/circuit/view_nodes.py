"""Successful-result address association; no independent result lifecycle."""
from dataclasses import dataclass
import os
from types import SimpleNamespace
import warnings

from silk.circuit.node_paths import node_candidates, NodeVoltageRequest, resolve_node_inventory
from silk.circuit.placement_reference import LiteralVoltageName
from silk.exceptions import CircuitError
from silk.vector_match import collect_vector_candidates, first_row


@dataclass(frozen=True)
class _AddressRecord:
    candidates: tuple
    physical: tuple
    currents: tuple = ()
    logical_currents: tuple = ()


def _capture_addresses(module):
    from silk.circuit.modules_netlist import _save_list_inventory
    try:
        records = node_candidates(module)
        physical = tuple(_save_list_inventory(module))
        candidates = []
        for record in records:
            address = record.address
            candidates.append((record.designer, record.components, record.leaf,
                               str(address), tuple(address._hierarchy)))
        from silk.circuit.module_test import _branch_vector_name
        from silk.sources._layout import source_instances
        from silk.sources.classes import CurrentSource, VoltageSource
        from silk.spice_names import to_designer_name, to_spice_name
        currents = []
        logical_currents = []
        for authored, source, prefix in source_instances(module.sig_sources_dict, module.elem_dict):
            if not isinstance(source, (CurrentSource, VoltageSource)):
                continue
            cards = list(source.physical_instance_names())
            if isinstance(source, CurrentSource) and module.simulator == 'behavioral':
                if source.instance_name not in cards:
                    cards.append(source.instance_name)
            if len(source.physical_instance_names()) > 1:
                try:
                    resolved = _branch_vector_name(module.sig_sources_dict, authored, module.elem_dict)
                except CircuitError:
                    resolved = None
                terms = tuple((prefix + card, sign) for card, sign in source.physical_layout().current_terms)
                expected = terms[0][0] if len(terms) == 1 else terms
                if resolved == expected:
                    aliases = tuple((to_designer_name(
                        ('v.' if isinstance(source, VoltageSource) and prefix
                         and module.simulator == 'ngspice' else '') + card,
                        kind='branch'), sign) for card, sign in terms)
                    logical_currents.append((authored, aliases))
            for card in cards:
                emitted = prefix + card
                branch = emitted
                if isinstance(source, VoltageSource) and prefix and module.simulator == 'ngspice':
                    branch = 'v.' + emitted
                alias = to_designer_name(branch, kind='branch')
                try:
                    resolved = _branch_vector_name(module.sig_sources_dict, alias, module.elem_dict)
                except CircuitError:
                    continue
                if (not isinstance(resolved, str)
                        or to_spice_name(resolved, kind='branch').casefold() != branch.casefold()):
                    continue
                request = branch
                probe_request = resolved
                if isinstance(source, CurrentSource) and module.simulator == 'ngspice':
                    request = '@' + ('i.' if '.' in emitted else '') + emitted + '[current]'
                    probe_request = request
                elif isinstance(source, CurrentSource) and module.simulator == 'xyce':
                    # The loader canonicalizes current-source cards before
                    # reading; the collector treats ':' and '.' alike.
                    probe_request = request
                currents.append((alias, request, isinstance(source, CurrentSource), probe_request))
        return _AddressRecord(tuple(candidates), physical, tuple(currents), tuple(logical_currents))
    except CircuitError:
        # Optional viewer proof cannot make an otherwise supported run fail.
        return None


def _proof(raw, frozen):
    records = [SimpleNamespace(designer=name, components=components, leaf=leaf,
                               address=LiteralVoltageName(address, hierarchy=hierarchy))
               for name, components, leaf, address, hierarchy in frozen.candidates]
    ids, addresses, paths = {}, [], {}
    for record in records:
        key = tuple(sorted(x.casefold() for x in record.address.voltage_forms()))
        if key not in ids:
            ids[key] = len(addresses)
            addresses.append(record.address)
        paths.setdefault(record.designer, set()).add(ids[key])
    matches = [[collect_vector_candidates(p, a, first=first_row(p)) for a in addresses]
               for p in raw.plots]
    # A label must not borrow a conflicting spelling from any plot, including
    # excluded rows. Repeated rows of the same physical target across analyses
    # remain legitimate; identity is not (plot, row).
    def globally_safe(name, target):
        for pi, plot in enumerate(raw.plots):
            for row, vector in enumerate(plot.vectors):
                if vector.name.casefold() == name.casefold():
                    if row not in matches[pi][target]:
                        return False
        return True

    def resolves(plot, inner, row):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                resolved = resolve_node_inventory(plot, NodeVoltageRequest(inner), records, frozen.physical)
            return collect_vector_candidates(plot, resolved, first=first_row(plot)) == [row]
        except CircuitError:
            return False

    metadata, proofs = [], []
    for pi, plot in enumerate(raw.plots):
        from silk.signals.view.emit import _signal_start
        vectors = [{'name': v.name, 'selectable': i >= _signal_start(plot)}
                   for i, v in enumerate(plot.vectors)]
        metadata.append({'version': 1, 'vectors': vectors,
                         'paths': [{'name': name, 'targets': [{'id': t, 'rows': matches[pi][t]}
                                    for t in sorted(targets)]} for name, targets in paths.items()]})
        row_targets = {}
        for target, rows in enumerate(matches[pi]):
            for row in rows:
                row_targets.setdefault(row, set()).add(target)
        plot_proof = []
        for row, targets in row_targets.items():
            if not vectors[row]['selectable']:
                continue
            safe = []
            for record in records:
                own = paths[record.designer]
                if len(own) != 1 or own != targets:
                    continue
                target = next(iter(own))
                name = f'v({record.designer})'
                # Case-folded spellings must not conceal another target.
                folded = set().union(*(v for k,v in paths.items() if k.casefold()==record.designer.casefold()))
                if (folded == own and matches[pi][target] == [row]
                        and globally_safe(name, target) and resolves(plot, record.designer, row)):
                    safe.append((len(record.components), record.designer, name))
            label = min(safe)[2] if safe else None
            probe = label
            if probe is None:
                name = plot.vectors[row].name
                inner = name[2:-1] if name[:2].lower() == 'v(' and name.endswith(')') else name
                if resolves(plot, inner, row):
                    probe = f'v({inner})'
            plot_proof.append((row, label, probe))
        # Card probes use the same branch-name resolution captured above and
        # the same vector collector as the result reader. Never infer a
        # voltage-source current when a behavioral run has no such row.
        current_rows = {}
        from silk.signals.analysis_kind import plot_silk_analysis
        for alias, request, current_source, probe_request in frozen.currents:
            if current_source and plot_silk_analysis(plot) == 'ac':
                continue
            rows = collect_vector_candidates(plot, request, 'I', first=first_row(plot))
            if len(rows) != 1 or not vectors[rows[0]]['selectable']:
                continue
            if collect_vector_candidates(plot, probe_request, 'I', first=first_row(plot)) != rows:
                continue
            row = rows[0]
            name = f'i({alias})'
            # Hidden and cross-analysis literal aliases must not steal a label.
            if any(v.name.casefold() == name.casefold() and
                   collect_vector_candidates(p, request, 'I', first=first_row(p)) != [ri]
                   for p in raw.plots for ri, v in enumerate(p.vectors)):
                continue
            current_rows.setdefault(row, set()).add(name)
        for row, names in current_rows.items():
            if len(names) == 1:
                name, = names
                plot_proof.append((row, name, name))
        proofs.append(tuple(plot_proof))
    return metadata, proofs


def _source_current_proofs(raw, frozen, proofs):
    """Retain checked terms and genuinely missing saved cards separately."""
    from silk.signals.analysis_kind import plot_silk_analysis
    from silk.simulation.behavioral_backend import BEHAVIORAL_COMMAND
    from silk.signals.view.emit import _plot_kind
    cards = {alias: (request, current_source)
             for alias, request, current_source, _ in frozen.currents}
    out = []
    for plot, proof in zip(raw.plots, proofs):
        if _plot_kind(plot) == 'constants':
            out.append(())
            continue
        checked = {probe: row for row, _, probe in proof if probe}
        sources = []
        for name, terms in frozen.logical_currents:
            probe = f'i({name})'
            if any(v.name.casefold() == probe.casefold() for p in raw.plots for v in p.vectors):
                continue
            rows, missing = [], []
            for alias, sign in terms:
                if alias not in cards:
                    break
                request, current_source = cards[alias]
                if ((current_source and plot_silk_analysis(plot) == 'ac')
                        or (not current_source and plot.command == BEHAVIORAL_COMMAND)):
                    break
                if f'i({alias})' in checked:
                    rows.append((checked[f'i({alias})'], sign))
                elif not collect_vector_candidates(plot, request, 'I', first=first_row(plot)):
                    missing.append(alias)
                else:
                    # Present but ambiguous/unproved is not a missing operand.
                    break
            else:
                sources.append((probe, tuple(rows), tuple(missing)))
        out.append(tuple(sources))
    return tuple(out)


def view_node_context(owner):
    """Return checked viewer context for a successful Module or bench, else None.

    Uses the bench's existing result route, including SelectedTestbench scope.
    The returned source is a borrowed read-only snapshot of one disk read;
    obtain a new context for each emission. Unsupported/missing associations
    keep legacy viewer behavior. No external file can acquire graph proof here.

    Parameters
    ----------
    owner : Module, ModuleTestbench or object
        The object whose most recent successful run supplies the results: a
        module, a testbench, or a selected-testbench handle (a
        ``SelectedTestbench``).

    Returns
    -------
    ViewNodeContext or None
        The checked context, or ``None`` when the results file is missing,
        was not produced by this object's run, or its content no longer
        matches the retained fingerprint.

    Raises
    ------
    TypeError
        If ``owner`` is not a Module, ModuleTestbench or SelectedTestbench.
    """
    return _view_node_context(owner)


def _view_node_context(owner, checked_source=None):
    from silk.circuit.modules import Module
    from silk.circuit.module_test import ModuleTestbench
    from silk.circuit.test_placement import SelectedTestbench
    from silk.signals.rawfile import read_rawfile
    from silk.signals.view.node_context import ViewNodeContext
    if isinstance(owner, (ModuleTestbench, SelectedTestbench)):
        reader = owner._result_netlist()
    elif isinstance(owner, Module):
        reader = owner
    else:
        raise TypeError('view_node_context requires a Module or ModuleTestbench result context')
    raw = getattr(reader, '_raw', None)
    record = getattr(raw, '_view_addresses', None)
    source = getattr(reader, 'results_file', None)
    fingerprint = getattr(getattr(raw, 'native', None), '_content_sha256', None)
    if (record is None or not fingerprint or not source or not isinstance(reader, Module)
            or not raw.source_path or os.path.abspath(source) != os.path.abspath(raw.source_path)):
        return None
    if _capture_addresses(reader) != record:
        return None
    from silk.simulation.generation import read_retained
    checked = checked_source if checked_source is not None else read_retained(owner)
    if checked._content_sha256 != fingerprint:
        return None
    metadata, proofs = _proof(checked, record)
    return ViewNodeContext._checked(checked, metadata, proofs,
                                    _source_current_proofs(checked, record, proofs))
