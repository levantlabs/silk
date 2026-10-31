"""Placement-order scheduling of declared behavioral feedback components."""
from collections import deque

from silk.exceptions import LoopError


def components(deps):
    """Return strongly connected components in input placement order."""
    reach = {}
    for path in deps:
        seen, todo = set(), [path]
        while todo:
            other = todo.pop()
            if other not in seen:
                seen.add(other)
                todo.extend(deps[other] - seen)
        reach[path] = seen
    remaining = set(deps)
    groups = []
    for path in deps:
        if path in remaining:
            group = [other for other in deps
                     if other in remaining and other in reach[path] and path in reach[other]]
            groups.append(group)
            remaining.difference_update(group)
    return groups


def _refuse_mixed(groups, attachments, labels=None):
    for group in groups:
        contracts = {attachments[path][2] if path in attachments else 'vector'
                     for path in group}
        if len(contracts) > 1:
            models = [labels[path] for path in group] if labels is not None else group
            raise LoopError(
                f"Feedback loop models {models} mix sample and whole-array calls. "
                "Give the loop models the same execution=, or put the loop in one function.")


def schedule(graph, attachments, *, dc):
    """Keep FIFO driver ordering, treating each allowed loop as one component."""
    writers = {}
    for path, info in graph.items():
        for net in info['writes']:
            writers[net] = path
    deps = {path: {writers[net] for net in info['reads']
                   if net in writers and writers[net] != path}
            for path, info in graph.items()}
    groups = components(deps)
    owners = {path: k for k, group in enumerate(groups) for path in group}
    # INOUT read connections participate only in admission. They never add
    # scheduler dependencies or change their existing placement-order notices.
    admission = {path: deps[path] | {
        writers[net] for net in info.get('inout_reads', ())
        if net in writers and writers[net] != path}
        for path, info in graph.items()}
    if not dc:
        _refuse_mixed(components(admission), attachments)
    needs = {k: {owners[other] for path in group for other in deps[path]
                 if owners[other] != k} for k, group in enumerate(groups)}
    queue = deque(k for k in needs if not needs[k])
    order = []
    while queue:
        k = queue.popleft()
        order.extend(groups[k])
        for other in needs:
            if k in needs[other]:
                needs[other].remove(k)
                if not needs[other]:
                    queue.append(other)
    return order


def feedback_reads(state, blocks, attachments, *, check_mixed=False):
    """Return undelayed read cycles; optionally refuse actual mixed loops.

    Mixed-contract detection includes explicit delays: a delayed mixed loop
    still cannot interleave sample and whole-array calls.
    """
    deps = {path: set() for path in blocks}
    all_deps = {path: set() for path in blocks} if check_mixed else None
    reads = []
    writers = {number: path for path, number in state._writer_ids.items()}
    for path, port, quantity, delay in sorted(state.read_events, key=repr):
        net = blocks[path].pins[port][0]
        if quantity == 'v':
            values = [state._voltage(net)]
        else:
            values = [state._currents[pin] for pin in state._net_pins[net]
                      if pin != (path, port)]
        for value in values:
            for writer in set(value.writers[value.assigned]):
                other = writers[int(writer)]
                if other in deps and other != path:
                    reads.append((path, port, quantity, other, delay))
                    if all_deps is not None:
                        all_deps[path].add(other)
                    if not delay:
                        deps[path].add(other)
    if all_deps is not None:
        _refuse_mixed(components(all_deps), attachments,
                      labels={path: block.path for path, block in blocks.items()})
    cycles = [group for group in components(deps)
              if len(group) > 1 and any(path in attachments for path in group)]
    members = {path for group in cycles for path in group}
    return [[blocks[path].path for path in group] for group in cycles], [
        (blocks[path].path, port, quantity, blocks[other].path)
        for path, port, quantity, other, delay in reads
        if not delay and path in members and other in members]
