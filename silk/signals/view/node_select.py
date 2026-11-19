"""Full-inventory emitted-precedence selectors for checked node metadata."""
from silk.signals.view.select import matches_selector, hier_parse, _selector_not_found_error
from silk.signals.view.template import AmbiguousSelectorError, EmptySelectionError, ViewError


def validate_node_paths(meta):
    if not isinstance(meta, dict) or meta.get('version') != 1:
        raise ViewError('invalid node_paths version')
    vectors, paths = meta.get('vectors'), meta.get('paths')
    if not isinstance(vectors, (list, tuple)) or not isinstance(paths, (list, tuple)):
        raise ViewError('node_paths requires vectors and paths arrays')
    for v in vectors:
        if (not isinstance(v, dict) or not isinstance(v.get('name'), str)
                or type(v.get('selectable')) is not bool):
            raise ViewError('invalid node_paths vector')
    names, targets = set(), {}
    for path in paths:
        if (not isinstance(path, dict) or not isinstance(path.get('name'), str)
                or not path['name'] or path['name'] in names
                or not isinstance(path.get('targets'), (list, tuple)) or not path['targets']):
            raise ViewError('invalid node_paths path')
        if 'kind' in path and path['kind'] != 'device':
            raise ViewError('invalid node_paths path kind')
        names.add(path['name'])
        seen = set()
        for target in path['targets']:
            if not isinstance(target, dict):
                raise ViewError('invalid node_paths target')
            ident, rows = target.get('id'), target.get('rows')
            if (type(ident) is not int or ident < 0 or ident in seen
                    or not isinstance(rows, (list, tuple))
                    or any(type(i) is not int or not 0 <= i < len(vectors) for i in rows)
                    or list(rows) != sorted(set(rows))):
                raise ViewError('invalid node_paths target rows')
            if ident in targets and targets[ident] != rows:
                raise ViewError('inconsistent node_paths target rows')
            targets[ident] = rows
            seen.add(ident)
    return meta


def _device_selector_match(name, selector):
    """A device matches through the ordinary selector grammar, by its full
    label (``dut.MN:gm``) or DUT-relative alias (``MN:gm``); no ``v(...)`` form."""
    return (matches_selector(name, selector)
            or matches_selector(name.partition('.')[2], selector))


def _device_selector_exact(name, selector):
    folded = selector.casefold()
    return folded in (name.casefold(), name.casefold().partition('.')[2])


def select_node_paths(plots, *, signals=None, exclude=None, max_depth=None, current_aliases=None):
    """Return masks in original row order and emitted-precedence diagnostics."""
    all_names = []
    for plot in plots:
        vectors = plot['vectors']
        meta = plot.get('node_paths')
        validate_node_paths(meta if meta is not None else {'version': 1, 'vectors': vectors, 'paths': []})
        if meta is not None and meta['vectors'] != vectors:
            raise ViewError('node_paths inventory does not match plot vectors')
        all_names.extend(v['name'] for v in vectors)
        all_names.extend(p['name'] if p.get('kind') == 'device' else f"v({p['name']})"
                         for p in (meta or {}).get('paths', []))
    diagnostics = []
    selected_current_aliases = {}

    def expand(selectors, keyword):
        chosen = set()
        for selector in selectors:
            emitted, designer, exact_targets = set(), set(), set()
            for pi, plot in enumerate(plots):
                for row, vector in enumerate(plot['vectors']):
                    if vector['selectable'] and matches_selector(vector['name'], selector):
                        emitted.add((pi, row))
                for path in (plot.get('node_paths') or {}).get('paths', []):
                    if path.get('kind') == 'device':
                        if _device_selector_exact(path['name'], selector):
                            exact_targets.update(t['id'] for t in path['targets'])
                        if _device_selector_match(path['name'], selector):
                            for target in path['targets']:
                                designer.update((pi, row) for row in target['rows']
                                                if plot['vectors'][row]['selectable'])
                        continue
                    full = f"v({path['name']})"
                    # Ambiguity includes all case-fold-equivalent exact spellings,
                    # even if the textual (lowercase) matcher would not pick one.
                    if selector.casefold() in (path['name'].casefold(), full.casefold()):
                        exact_targets.update(t['id'] for t in path['targets'])
                    if matches_selector(full, selector):
                        for target in path['targets']:
                            designer.update((pi, row) for row in target['rows']
                                            if plot['vectors'][row]['selectable'])
            # Source aliases are a fallback for the physical inventory, not
            # additional waveform rows. An emitted match already owns the name.
            if not emitted and current_aliases is not None:
                for pi, aliases in enumerate(current_aliases):
                    for name, terms, missing, single in aliases:
                        if single and not missing and matches_selector(name, selector):
                            row = terms[0][0]
                            emitted.add((pi, row))
                            if keyword == 'signals':
                                selected_current_aliases.setdefault((pi, row), name)
            if emitted:
                if designer and emitted != designer:
                    diagnostics.append({'kind': 'emitted-precedence', 'selector': selector,
                                        'keyword': keyword})
                found = emitted
            else:
                if len(exact_targets) > 1:
                    raise AmbiguousSelectorError(
                        f'{keyword}= selector {selector!r} is ambiguous: distinct physical nodes',
                        selector=selector, kwarg=keyword,
                    )
                found = designer
            if not found:
                raise _selector_not_found_error(selector, all_names, keyword)
            chosen.update(found)
        return chosen

    selected = ({(pi, row) for pi, p in enumerate(plots) for row,v in enumerate(p['vectors']) if v['selectable']}
                if signals is None else expand(signals, 'signals'))
    if exclude:
        selected -= expand(exclude, 'exclude')
    masks = [[(pi, row) in selected and (max_depth is None or len(hier_parse(v['name'])['segments']) <= max_depth)
              for row,v in enumerate(p['vectors'])] for pi,p in enumerate(plots)]
    if max_depth is not None:
        for (pi, row), name in selected_current_aliases.items():
            masks[pi][row] = ((pi, row) in selected
                              and len(hier_parse(name)['segments']) <= max_depth)
    if not any(any(m) for m in masks):
        raise EmptySelectionError('signal selection (signals=/exclude=/max_depth=) matched zero signals in this rawfile — nothing would be emitted')
    result = {'masks': masks, 'diagnostics': diagnostics}
    if current_aliases is not None:
        result['current_aliases'] = selected_current_aliases
    return result
