"""silk.circuit.composition — post-run composition graphic.

*Draw what was built, annotated with what was measured.*

An analog design in silk is a tree of :class:`~silk.circuit.modules.Module`
instances wired together by shared nets. By the time an example finishes
running, the whole story — which block feeds which, what the operating point
on the connecting wire was, whether each block met its specs — already lives
in structures silk maintains (``Net.pins`` for connectivity, ``Net.voltage``
for behavioral operating points, ``Module._specs`` for pass/fail). Yet the only
place a reader can *see* that composition is a hand-drawn ASCII diagram in a
docstring that drifts the moment the netlist changes.

This module closes that gap. It is a **pure renderer** over existing state — it
adds no new data and mutates nothing:

- :func:`module_graph` walks a module one level deep (by default) into a plain,
  JSON-shaped ``dict`` of nodes (immediate children), edges (shared nets), and
  ports. Deeper hierarchy is summarized as a recursive device count inside each
  child box; pass ``depth=`` to expand more levels as nested subgraphs.
- :func:`graph_to_mermaid` turns that dict into a Mermaid ``flowchart LR``
  string — zero new dependencies, and GitHub / the silk HTML report render it
  natively.
- :func:`save_graph_png` draws the same graph as a matplotlib block diagram.
  Boxes carry named pin points on their sides (west receives, east drives) and
  every wire attaches pin-to-pin; layout is longest-path layering, so signal
  flows left-to-right in columns and feedback (a wire back to an earlier column)
  sweeps under the graph as a coloured curve. Numeric instance/port families
  (``mux0..mux9``) collapse to one box/pill with an ``xN`` badge, the canvas
  grows with the node count so boxes never overlap, and high-fanout supplies /
  clocks are summarized in a caption rather than drawn as a hairball. Named
  pin points on box sides connect the layered diagram.

**Honest limitations.** The graph is a *connectivity* view, not a schematic —
it has no notion of signal-flow direction beyond port in/out, and it does not
place devices the way a schematic would. The default depth-1 view is the useful
one for inspection; ``depth>1`` is supported in the Mermaid renderer (nested
subgraphs) but the PNG renderer always draws only the immediate children, since
a boxes-and-arrows layout stops being readable past a single level. Spec status
on a child box reflects measurements *already recorded* on that subtree's specs
— it never triggers reconciliation, so a block whose specs were never measured
shows ``–`` (unmeasured), not a spurious pass or fail.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# ElementType is a lightweight str-enum in ports.py (DEVICE / MODULE); importing
# it here is cycle-free. Module itself is imported lazily inside the helpers that
# need it (net.py imports Module only under TYPE_CHECKING for the same reason).
from silk.circuit.ports import ElementType


# ---------------------------------------------------------------------------
# Graph extraction — pure, no rendering imports
# ---------------------------------------------------------------------------

def module_graph(mod, depth: int = 1) -> dict:
    """Walk a module into a plain graph dict of nodes, edges, and ports.

    Parameters
    ----------
    mod : Module
        The module to describe. Its immediate children become nodes; the nets
        it owns become edges.
    depth : int, default 1
        How many levels of children to expand as boxes. At ``depth=1`` (the
        default) only the immediate children are nodes, each carrying a
        recursive ``device_count`` summarizing everything beneath it. At
        ``depth>1``, every submodule child additionally carries a ``subgraph``
        key holding ``module_graph(child, depth-1)``.

    Returns
    -------
    dict
        ``{"name", "nodes", "edges", "ports"}`` where

        - **nodes** — one per immediate child: ``{"name", "kind",
          "module_name", "device_count", "spec", "subgraph"}``. ``kind`` is one
          of ``'submodule' | 'device' | 'passive' | 'source'``. ``device_count``
          is the recursive count of leaf devices + passives (0 for non-submodule
          children). ``spec`` is a status dict for submodules (see
          :func:`_spec_status`) or ``None``.
        - **edges** — one per net with two or more bound pins:
          ``{"net", "endpoints", "voltage", "is_port", "port_name"}``.
          ``endpoints`` is a list of ``{"instance", "port"}``. ``voltage`` is
          the final value of the net's behavioral waveform, or ``None``.
        - **ports** — one per top-level port: ``{"name", "direction", "role",
          "net", "endpoints"}``, where ``endpoints`` are the child pins bound to
          the port's net (used to draw port stubs).
    """
    from silk.circuit.module_placement import refresh_definitions
    refresh_definitions(mod)
    name = getattr(mod, 'module_name', None) or 'unnamed'
    net_dict = getattr(mod, 'net_dict', None) or {}
    port_net_map = _net_port_map(mod)

    nodes: List[dict] = []
    for inst_name, kind, obj in _child_instances(mod):
        node = {
            'name': inst_name,
            'kind': kind,
            'module_name': (mod.elem_dict.get(inst_name, {}).get('module_name')
                            if kind == 'submodule' else None) or _label_for(kind, obj),
            'device_count': _count_leaf_elements(obj) if kind == 'submodule' else 0,
            'spec': _spec_status(obj) if kind == 'submodule' else None,
            'subgraph': None,
        }
        if kind == 'submodule' and depth > 1 and obj is not None:
            node['subgraph'] = module_graph(obj, depth=depth - 1)
        nodes.append(node)

    edges: List[dict] = []
    for net_name, net in net_dict.items():
        pins = getattr(net, 'pins', None) or []
        if len(pins) < 2:
            continue
        edges.append({
            'net': net_name,
            'endpoints': [_endpoint(b) for b in pins],
            'voltage': _tail_voltage(net),
            'is_port': net_name in port_net_map,
            'port_name': port_net_map.get(net_name),
        })

    ports: List[dict] = []
    for pname, pin in (getattr(mod, 'pin_list', None) or {}).items():
        net_name = getattr(pin, 'net', None) or pname
        net = net_dict.get(net_name)
        endpoints = [
            _endpoint(b) for b in (getattr(net, 'pins', None) or [])
        ] if net is not None else []
        ports.append({
            'name': pname,
            'direction': _as_str(getattr(pin, 'direction', None)),
            'role': _as_str(getattr(pin, 'role', None)),
            'net': net_name,
            'endpoints': endpoints,
        })

    return {'name': name, 'nodes': nodes, 'edges': edges, 'ports': ports}


def _endpoint(binding) -> dict:
    """One net endpoint for the graph dict, from a :class:`PortBinding`.

    Carries the pin's real ``direction``/``role`` (from the child's declared
    ports, unwrapped to strings by the binding) alongside instance/port. These
    are additive: existing consumers (mermaid, older tests) read only
    ``instance``/``port``. Renderers use ``direction`` to orient wires from real
    data rather than any pin-name convention.
    """
    return {
        'instance': binding.instance,
        'port': binding.port,
        'direction': _as_str(getattr(binding, 'direction', None)),
        'role': _as_str(getattr(binding, 'role', None)),
    }


def _child_instances(mod):
    """Yield ``(instance_name, kind, obj)`` for every immediate child of *mod*.

    Children can arrive through several doors, all of which this unifies:
    device/submodule instances live in ``elem_dict`` (keyed by instance name),
    passives in ``passives_dict``, sources in ``sig_sources_dict``, and
    ``add_submodule`` writes a thin definition-free entry into ``module_dict``.
    Definition entries in ``module_dict`` (those carrying ``numberInstantiated``,
    written by ``add_element``) are templates, not instances, and are skipped.

    An ideal R/C/L added via ``Module.add()`` lands in ``elem_dict``,
    not ``passives_dict`` — recognised here by what it emits
    (:func:`~silk.circuit.modules_results._is_ideal_passive`, SPICE prefix
    R/C/L) so it still yields kind ``'passive'``, exactly as the same object
    would via ``add_passive()``. Without this, the two build paths
    disagreed on node kind, which fed different Mermaid/PNG styling
    (:data:`_KIND_FACE`) for what is otherwise the identical circuit.
    """
    from silk.circuit.modules_results import _is_ideal_passive

    seen = set()

    elem_dict = getattr(mod, 'elem_dict', None) or {}
    for inst_name, entry in elem_dict.items():
        if not isinstance(entry, dict):
            continue
        obj = entry.get('device')
        if entry.get('type') == ElementType.MODULE:
            kind = 'submodule'
        elif _is_ideal_passive(obj):
            kind = 'passive'
        else:
            kind = 'device'
        seen.add(inst_name)
        yield inst_name, kind, obj

    # add_submodule path: module_dict entries keyed by instance name, with no
    # 'numberInstantiated'/'defined' bookkeeping (those mark add_element defs).
    module_dict = getattr(mod, 'module_dict', None) or {}
    for inst_name, entry in module_dict.items():
        if inst_name in seen or not isinstance(entry, dict):
            continue
        if 'numberInstantiated' in entry or 'defined' in entry:
            continue
        obj = entry.get('device')
        if obj is None:
            continue
        seen.add(inst_name)
        yield inst_name, 'submodule', obj

    for inst_name, obj in (getattr(mod, 'passives_dict', None) or {}).items():
        yield inst_name, 'passive', obj

    for inst_name, obj in (getattr(mod, 'sig_sources_dict', None) or {}).items():
        yield inst_name, 'source', obj


def _count_leaf_elements(mod) -> int:
    """Recursive count of leaf devices + passives beneath *mod*.

    Submodules recurse; devices and passives each count as one; sources are
    stimulus, not part of the composed device count, so they are excluded.
    """
    total = 0
    for _name, kind, obj in _child_instances(mod):
        if kind == 'submodule' and obj is not None:
            total += _count_leaf_elements(obj)
        elif kind in ('device', 'passive'):
            total += 1
    return total


def _spec_status(mod) -> dict:
    """Summarize the pass/fail state of every spec in *mod*'s subtree.

    Side-effect-free: reads measurements already recorded on each spec (and, for
    dotted-path metrics, resolves the current value) but never records
    measurements or fires calibration the way ``reconcile_specs`` does.

    Returns
    -------
    dict
        ``{"status", "pass", "fail", "unmeasured", "total"}``. ``status`` is
        ``'fail'`` if any spec fails, ``'pass'`` if at least one passes and none
        is unmeasured, else ``'unmeasured'`` (which also covers a module with no
        specs at all).
    """
    specs = _collect_specs(mod)
    n_pass = n_fail = n_unmeasured = 0
    for spec in specs:
        value = _spec_value(spec)
        if value is None:
            n_unmeasured += 1
            continue
        try:
            passed = bool(_evaluate_spec(spec, value))
        except Exception:
            n_unmeasured += 1
            continue
        if passed:
            n_pass += 1
        else:
            n_fail += 1

    if n_fail:
        status = 'fail'
    elif n_pass and n_unmeasured == 0:
        status = 'pass'
    else:
        status = 'unmeasured'
    return {
        'status': status,
        'pass': n_pass,
        'fail': n_fail,
        'unmeasured': n_unmeasured,
        'total': len(specs),
    }


def _collect_specs(mod) -> list:
    """All specs on *mod* and its nested submodule instances (pre-order)."""
    if hasattr(mod, '_iter_all_specs'):
        try:
            return list(mod._iter_all_specs())
        except Exception:
            pass
    return list(getattr(mod, '_specs', None) or [])


def _spec_value(spec):
    """The value to evaluate a spec against, or ``None`` if unmeasured.

    Prefers the most recent recorded measurement; for a dotted-path metric with
    no recording yet, resolves it against the owning module (a read).
    """
    latest = getattr(spec, 'latest', None)
    if callable(latest):
        m = latest()
        if m is not None:
            return getattr(m, 'value', None)
    metric = getattr(spec, 'metric', '')
    if isinstance(metric, str) and '.' in metric:
        try:
            from silk.specs.resolver import resolve_metric
            rr = resolve_metric(spec.module, metric)
            if getattr(rr, 'resolved', False) and rr.value is not None:
                return rr.value
        except Exception:
            return None
    return None


def _evaluate_spec(spec, value) -> bool:
    """Compare *value* against a spec's target using its declared operator."""
    from silk.specs.spec import Measurement
    return spec.evaluate(Measurement(source='user', value=value))


def _net_port_map(mod) -> Dict[str, str]:
    """Map net name -> top-level port name for *mod*'s own ports.

    A subcircuit exposes its boundary nodes by port name, so the port's net is
    the port's name unless a flow has explicitly bound ``ModulePin.net``.
    """
    out: Dict[str, str] = {}
    for pname, pin in (getattr(mod, 'pin_list', None) or {}).items():
        net_name = getattr(pin, 'net', None) or pname
        out.setdefault(net_name, pname)
    return out


def _label_for(kind: str, obj) -> str:
    """Human-readable type label for a child node."""
    if obj is None:
        return kind
    if kind == 'submodule':
        return getattr(obj, 'module_name', None) or getattr(obj, 'device_name', None) or 'module'
    if kind == 'device':
        return (getattr(obj, '_device_type_name', None)
                or getattr(obj, 'device_name', None)
                or type(obj).__name__)
    # passive / source: the concrete class name (Resistor, VoltageSource, ...)
    return type(obj).__name__


def _tail_voltage(net) -> Optional[float]:
    """Final value of a net's behavioral waveform as a scalar, or ``None``."""
    v = getattr(net, 'voltage', None)
    if v is None:
        return None
    try:
        import numpy as np
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            return None
        return float(arr[-1])
    except Exception:
        return None


def _as_str(value) -> Optional[str]:
    """Normalize an enum/str/None field to a plain string (or None)."""
    if value is None:
        return None
    return getattr(value, 'value', None) or str(value)


# ---------------------------------------------------------------------------
# Mermaid renderer
# ---------------------------------------------------------------------------

_STATUS_SYMBOL = {'pass': '✓', 'fail': '✗', 'unmeasured': '–'}

# (open, close) delimiters that give each kind a distinct Mermaid shape.
_NODE_SHAPE = {
    'submodule': ('["', '"]'),      # rectangle
    'device':    ('("', '")'),      # rounded
    'passive':   ('("', '")'),      # rounded
    'source':    ('[/"', '"/]'),    # parallelogram
}


def graph_to_mermaid(graph: dict) -> str:
    """Render a :func:`module_graph` dict as a Mermaid ``flowchart LR`` string.

    Submodule children render as rectangles labeled with instance name, module
    name, recursive device count, and spec status (``✓`` pass / ``✗``
    fail / ``–`` unmeasured). Top-level ports render as stadium nodes; nets
    render as labeled edges. A net touching more than two pins renders as a small
    junction node so the edge count stays linear rather than N². When a node
    carries a ``subgraph`` (``depth>1``), it renders as a nested Mermaid
    ``subgraph`` block.
    """
    lines: List[str] = ['flowchart LR']
    _emit_mermaid(graph, lines, prefix='', indent=1)
    return '\n'.join(lines)


def _emit_mermaid(graph: dict, lines: List[str], prefix: str, indent: int) -> None:
    pad = '    ' * indent
    id_of: Dict[str, str] = {}
    port_id_of: Dict[str, str] = {}

    for p in graph['ports']:
        pid = f'{prefix}p_{_sanitize(p["name"])}'
        port_id_of[p['name']] = pid
        lines.append(f'{pad}{pid}(["{_port_label(p)}"])')

    for node in graph['nodes']:
        sub = node.get('subgraph')
        if sub is not None:
            sgid = f'{prefix}sg_{_sanitize(node["name"])}'
            id_of[node['name']] = sgid
            lines.append(f'{pad}subgraph {sgid}["{_node_label(node)}"]')
            _emit_mermaid(sub, lines, prefix=f'{sgid}_', indent=indent + 1)
            lines.append(f'{pad}end')
        else:
            nid = f'{prefix}n_{_sanitize(node["name"])}'
            id_of[node['name']] = nid
            open_d, close_d = _NODE_SHAPE.get(node['kind'], ('["', '"]'))
            lines.append(f'{pad}{nid}{open_d}{_node_label(node)}{close_d}')

    edge_nets = set()
    for e in graph['edges']:
        edge_nets.add(e['net'])
        participants = [id_of[ep['instance']] for ep in e['endpoints']
                        if ep['instance'] in id_of]
        if e.get('is_port') and e.get('port_name') in port_id_of:
            participants.append(port_id_of[e['port_name']])
        participants = list(dict.fromkeys(participants))
        if len(participants) < 2:
            continue
        label = _edge_label(e)
        if len(participants) == 2:
            a, b = participants
            lines.append(f'{pad}{a} ---|"{label}"| {b}')
        else:
            jid = f'{prefix}j_{_sanitize(e["net"])}'
            lines.append(f'{pad}{jid}{{{{"{label}"}}}}')
            for part in participants:
                lines.append(f'{pad}{part} --- {jid}')

    # Ports whose net is not already a (>=2-pin) edge: draw a direct stub so a
    # single-fanout input/output still shows its connection.
    for p in graph['ports']:
        if p['net'] in edge_nets:
            continue
        pid = port_id_of[p['name']]
        targets = [id_of[ep['instance']] for ep in p['endpoints']
                   if ep['instance'] in id_of]
        for tgt in dict.fromkeys(targets):
            lines.append(f'{pad}{pid} ---|"{p["net"]}"| {tgt}')


def _node_label(node: dict) -> str:
    parts = [node['name']]
    if node['module_name'] and node['module_name'] != node['name']:
        parts.append(f'({node["module_name"]})')
    if node['kind'] == 'submodule':
        parts.append(f'{node["device_count"]} dev')
        spec = node.get('spec') or {}
        parts.append(_STATUS_SYMBOL.get(spec.get('status'), _STATUS_SYMBOL['unmeasured']))
    return '<br/>'.join(parts)


def _port_label(p: dict) -> str:
    return f'{p["name"]}'


def _edge_label(e: dict) -> str:
    label = e['net']
    if e.get('voltage') is not None:
        label += f' {_fmt_v(e["voltage"])}'
    return label


def _fmt_v(v: float) -> str:
    return f'{v:.3g}V'


def _sanitize(s) -> str:
    return re.sub(r'\W', '_', str(s)) or 'x'


# ---------------------------------------------------------------------------
# matplotlib renderer (v1.2)
#
# Draws the layered graph with matplotlib: named pin points on box sides,
# longest-path layering into signal-flow columns, numeric-family collapsing (mux0..9 -> one xN box),
# feedback drawn as back-edges under the graph, and semantic shapes for spec
# status / unconnected ports. Wire orientation comes from REAL declared pin
# direction (module_graph now carries direction/role on every endpoint) — there
# are no pin-name conventions here; where a pin's direction is unknown, flow
# falls back to declaration order and is resolved by the layering.
# ---------------------------------------------------------------------------

_KIND_FACE = {
    'submodule': '#dbe9ff',
    'device':    '#eeeeee',
    'passive':   '#d9f0dd',
    'source':    '#fff3cd',
}
_STATUS_EDGE = {'pass': '#2e7d32', 'fail': '#c62828', 'unmeasured': '#9e9e9e'}
_PORT_FACE, _PORT_EDGE = '#fdf6d8', '#b8a24a'
_OPEN_EDGE = '#c62828'          # a declared port no drawn wire reaches
_EDGE_C, _BACK_C = '#666666', '#c0564a'

# Geometry in pixel-like data units (y increases downward, SVG-style; the axis
# is inverted at draw time). Boxes are sized from their content — title lines
# and the number of pin rows — and columns are pitched by the widest box plus a
# gutter, so nothing can overlap regardless of node count.
_PIN_PITCH = 20.0
_TITLE_H = 42.0
_STUB = 9.0
_CHAR = 7.2                      # px per pin-label char
_TCHAR = 8.0                    # px per title char
_COL_GUTTER = 80.0
_ROW_GAP = 30.0
_MARGIN = 34.0
_HARD_FANOUT_CAP = 8            # nets past this many pins -> caption, not drawn
_SUPPLY_FANOUT_CAP = 3          # supplies/grounds get a stricter cap
_DPI = 100.0
_FIG_MIN, _FIG_MAX = 4.5, 48.0


def _family_key(name: str):
    """Split ``name`` into (base, index, suffix) when it carries a numeric
    family index (``mux0`` -> ('mux', 0, ''), ``P3p`` -> ('P', 3, 'p')), else
    ``None``. The (base, suffix) pair is the family identity."""
    m = re.fullmatch(r'([A-Za-z_]\w*?)(\d+)([A-Za-z_]\w*)?', name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), (m.group(3) or '')


def _drives(direction: str) -> bool:
    """Whether a pin with this declared direction drives its net (box east side).

    Uses the pin's REAL direction only — no pin-name conventions. ``None`` /
    ``inout`` do not count as drivers; such nets fall back to declaration order
    and the layering for orientation.
    """
    return 'out' in (direction or '').lower()


def _in_like(direction: str, role: str) -> bool:
    """A top-level port that drives INTO the circuit (an input/clock/bias)."""
    d, r = (direction or '').lower(), (role or '').lower()
    if 'out' in d:
        return False
    if 'in' in d:
        return True
    # no direction: fall back to role
    return r not in ('', 'analog')  # ground/power/clock/etc. drive in


def _agg_status(members) -> dict:
    statuses = [ (m.get('spec') or {}).get('status') for m in members ]
    statuses = [s for s in statuses if s]
    if 'fail' in statuses:
        st = 'fail'
    elif statuses and all(s == 'pass' for s in statuses):
        st = 'pass'
    else:
        st = 'unmeasured'
    return {'status': st}


def _collapse_instances(nodes):
    """Collapse numeric instance families (>2 members) into one render node.

    Returns (alias, meta, order): alias maps member instance name -> render id;
    meta maps render id -> dict(label, kind, module_name, device_count, spec,
    count, members); order is the render ids in first-appearance order.
    """
    fam = {}
    for nd in nodes:
        k = _family_key(nd['name'])
        if k:
            fam.setdefault((k[0], k[2]), []).append((k[1], nd))
    big = {key for key, mem in fam.items() if len(mem) > 2}

    alias, meta, order = {}, {}, []
    for nd in nodes:
        name = nd['name']
        k = _family_key(name)
        key = (k[0], k[2]) if k else None
        if key in big:
            base, suf = key
            rid = f'i:{base}#{suf}'
            alias[name] = rid
            if rid not in meta:
                members = [n for _i, n in sorted(fam[key], key=lambda t: t[0])]
                idxs = sorted(i for i, _n in fam[key])
                rep = members[0]
                label = f'{base}{idxs[0]}..{base}{idxs[-1]}{suf}'
                meta[rid] = {
                    'label': label, 'kind': rep['kind'],
                    'module_name': rep['module_name'],
                    'device_count': rep['device_count'],
                    'spec': _agg_status(members) if rep['kind'] == 'submodule' else None,
                    'count': len(members), 'members': [m['name'] for m in members],
                    'is_port': False,
                }
                order.append(rid)
        else:
            rid = f'i:{name}'
            alias[name] = rid
            meta[rid] = {
                'label': name, 'kind': nd['kind'], 'module_name': nd['module_name'],
                'device_count': nd['device_count'], 'spec': nd.get('spec'),
                'count': 1, 'members': [name], 'is_port': False,
            }
            order.append(rid)
    return alias, meta, order


def _collapse_ports(ports):
    """Collapse numeric port families (>2) into one pill. Returns
    (alias, meta, order) mirroring :func:`_collapse_instances`."""
    fam = {}
    for p in ports:
        k = _family_key(p['name'])
        if k:
            fam.setdefault((k[0], k[2]), []).append((k[1], p))
    big = {key for key, mem in fam.items() if len(mem) > 2}

    alias, meta, order = {}, {}, []
    for p in ports:
        name = p['name']
        k = _family_key(name)
        key = (k[0], k[2]) if k else None
        if key in big:
            base, suf = key
            rid = f'p:{base}#{suf}'
            alias[name] = rid
            if rid not in meta:
                idxs = sorted(i for i, _p in fam[key])
                label = f'{base}{idxs[0]}..{base}{idxs[-1]}{suf}'
                meta[rid] = {
                    'label': label, 'is_port': True,
                    'direction': p['direction'], 'role': p['role'],
                    'count': len(fam[key]),
                    'members': [q['name'] for _i, q in sorted(fam[key], key=lambda t: t[0])],
                }
                order.append(rid)
        else:
            rid = f'p:{name}'
            alias[name] = rid
            meta[rid] = {
                'label': name, 'is_port': True, 'direction': p['direction'],
                'role': p['role'], 'count': 1, 'members': [name],
            }
            order.append(rid)
    return alias, meta, order


def _rank_and_backedges(node_ids, dedup_edges):
    """Longest-path layering with DFS back-edge detection.

    ``dedup_edges`` is a list of unique (src, dst) structural pairs. Returns
    (rank, back) where rank maps node -> column index and back is the set of
    (src, dst) pairs that close a cycle (drawn as feedback). Deterministic:
    nodes and edges are processed in the given order.
    """
    adj = {n: [] for n in node_ids}
    for s, d in dedup_edges:
        if s != d and s in adj:
            adj[s].append(d)

    color = {n: 0 for n in node_ids}   # 0 white / 1 gray / 2 black
    back = set()

    def visit(root):
        stack = [(root, iter(adj[root]))]
        color[root] = 1
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if color[v] == 1:
                    back.add((u, v))
                elif color[v] == 0:
                    color[v] = 1
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
            if not advanced:
                color[u] = 2
                stack.pop()

    for n in node_ids:
        if color[n] == 0:
            visit(n)

    indeg = {n: 0 for n in node_ids}
    dag = {n: [] for n in node_ids}
    for s, d in dedup_edges:
        if s == d or (s, d) in back or d not in indeg:
            continue
        dag[s].append(d)
        indeg[d] += 1
    rank = {n: 0 for n in node_ids}
    queue = [n for n in node_ids if indeg[n] == 0]
    while queue:
        n = queue.pop(0)
        for m in dag[n]:
            rank[m] = max(rank[m], rank[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return rank, back


def _collapse_side_pins(pins):
    """Collapse a numbered pin family on one box side into a single row
    (``d0..d14``), like the boxes themselves. Returns (rows, member->row)."""
    fam = {}
    for p in pins:
        k = _family_key(p)
        if k:
            fam.setdefault((k[0], k[2]), []).append(k[1])
    big = {key for key, idxs in fam.items() if len(idxs) > 2}
    rows, member = [], {}
    for p in pins:
        k = _family_key(p)
        key = (k[0], k[2]) if k else None
        if key in big:
            idxs = sorted(fam[key])
            row = f'{key[0]}{idxs[0]}..{key[0]}{idxs[-1]}{key[1]}'
        else:
            row = p
        member[p] = row
        if row not in rows:
            rows.append(row)
    return rows, member


def _compute_png_layout(graph: dict) -> dict:
    """Pure geometry + structure pass for :func:`save_graph_png`.

    Collapses numeric families, orients and ranks the connection graph
    (longest-path layering; cycles become back-edges), lays nodes into columns
    by rank with content-sized boxes and named pin rows, and returns everything
    the drawing pass needs. Kept separate so the layout — box extents (no
    overlaps), ranks (monotonic forward edges), and family collapse — is
    testable without rendering.
    """
    inst_alias, inst_meta, inst_order = _collapse_instances(graph['nodes'])
    port_alias, port_meta, port_order = _collapse_ports(graph['ports'])
    net_role = {p['net']: (p.get('role') or '').lower() for p in graph['ports']}

    def _is_rail(net, participants):
        if participants > _HARD_FANOUT_CAP:
            return True
        return net_role.get(net) in ('power', 'ground') and participants > _SUPPLY_FANOUT_CAP

    # --- collect connection groups ----------------------------------------
    # A group is (net, [(inst_node, pin), ...], port_rid|None, voltage). Two
    # sources feed it: shared nets (>=2 pins) from graph['edges'], and each
    # top-level port's own fan-in/out from graph['ports'] — the latter matters
    # because module_graph only emits an edge for a >=2-pin net, so a
    # single-fanout port (a lane input, the SOUT tap) would otherwise never wire.
    edge_nets = {e['net'] for e in graph['edges']}
    groups = []                    # (net, [(inst_node, pin, direction)], port_rid, voltage)
    for e in graph['edges']:
        eps = [(inst_alias[ep['instance']], ep['port'], ep.get('direction'))
               for ep in e['endpoints'] if ep['instance'] in inst_alias]
        prid = port_alias.get(e['port_name']) if e.get('is_port') else None
        groups.append((e['net'], eps, prid, e.get('voltage')))
    for p in graph['ports']:
        if p['net'] in edge_nets:
            continue                          # already covered as a shared net
        eps = [(inst_alias[ep['instance']], ep['port'], ep.get('direction'))
               for ep in p['endpoints'] if ep['instance'] in inst_alias]
        if eps:
            groups.append((p['net'], eps, port_alias[p['name']], None))

    # --- orient each group into directed pin-to-pin edges ------------------
    # The driver is picked from REAL declared direction: an instance pin whose
    # direction is 'output', or a port that drives into the circuit (from its
    # own direction/role). When no endpoint declares an output direction, we do
    # NOT guess from pin names — we fall back to declaration order and let the
    # layering resolve columns and feedback.
    directed = []          # (s, d, s_pin, d_pin, net, voltage)
    suppressed = []
    used_ports = set()
    for net, eps, port_rid, volt in groups:
        nodes_here = {n for n, _p, _d in eps}
        participants = len(nodes_here) + (1 if port_rid else 0)
        if _is_rail(net, participants):
            suppressed.append(net)
            continue
        # endpoints as (node, pin, is_driver)
        cand = [(n, p, _drives(d)) for n, p, d in eps]
        if port_rid is not None:
            pm = port_meta[port_rid]
            cand.append((port_rid, '', _in_like(pm['direction'], pm['role'])))
        drivers = [c for c in cand if c[2]]
        src = drivers[0] if drivers else (cand[0] if cand else None)
        if src is None:
            continue
        if port_rid is not None:
            used_ports.add(port_rid)
        for n, p, _drv in cand:
            if (n, p) == (src[0], src[1]):
                continue                      # the source pin itself
            # (n == src[0] with a different pin is a real self-loop: a collapsed
            #  family whose members feed each other — kept, drawn arcing over.)
            directed.append((src[0], n, src[1], p, net, volt))

    # Collapsing families produces many identical wires (all 10 mux.Y->ff.D);
    # draw each distinct pin-to-pin connection once. Different pins between the
    # same pair (a PFD's UP and DN into a charge pump) stay distinct.
    seen_edge = set()
    deduped = []
    for tup in directed:
        key = (tup[0], tup[1], tup[2], tup[3])
        if key in seen_edge:
            continue
        seen_edge.add(key)
        deduped.append(tup)
    directed = deduped

    # --- node set + ranking ------------------------------------------------
    node_ids = []
    for rid in port_order:                     # declared ports first (inputs rank 0)
        if rid in used_ports:
            node_ids.append(rid)
    node_ids += inst_order
    # A declared port that no drawn wire reaches renders as a dashed hollow
    # pill — unless its net was summarized as a shared rail, in which case it
    # belongs in the caption, not as a dangling stub.
    supp_set = set(suppressed)
    unconnected_ports = []
    for rid in port_order:
        if rid in used_ports:
            continue
        if _port_net(port_meta[rid], graph) in supp_set:
            continue
        unconnected_ports.append(rid)
    node_ids += [rid for rid in unconnected_ports if rid not in node_ids]

    dedup = list(dict.fromkeys((s, d) for s, d, *_ in directed))
    rank, back = _rank_and_backedges(node_ids, dedup)

    # --- pin rows per node (collapse numbered pin families per side) -------
    west, east = {}, {}
    for s, d, sp, dp, _net, _v in directed:
        if s.startswith('i:') and sp:
            east.setdefault(s, [])
            if sp not in east[s]:
                east[s].append(sp)
        if d.startswith('i:') and dp:
            west.setdefault(d, [])
            if dp not in west[d]:
                west[d].append(dp)
    west_map, east_map = {}, {}
    for store, mp in ((west, west_map), (east, east_map)):
        for n, pins in list(store.items()):
            rows, member = _collapse_side_pins(pins)
            store[n] = rows
            mp[n] = member

    # --- geometry ----------------------------------------------------------
    def label_lines(rid):
        meta = inst_meta.get(rid) or port_meta.get(rid)
        if rid.startswith('p:'):
            return [meta['label']]
        lines = [meta['label']]
        mn = meta['module_name']
        badge = f' x{meta["count"]}' if meta['count'] > 1 else ''
        if mn and mn != meta['label']:
            lines.append(f'({mn}){badge}')
        elif badge:
            lines.append(badge.strip())
        if meta['kind'] == 'submodule':
            sym = _STATUS_SYMBOL.get((meta.get('spec') or {}).get('status'),
                                     _STATUS_SYMBOL['unmeasured'])
            lines.append(f'{meta["device_count"]} dev  {sym}')
        return lines

    def size_of(rid):
        lines = label_lines(rid)
        w_title = max(len(t) for t in lines) * _TCHAR + 22
        if rid.startswith('p:'):
            return w_title, 28.0
        wp, ep = west.get(rid, []), east.get(rid, [])
        rows = max(len(wp), len(ep), 1)
        w_pins = (max((len(t) for t in wp), default=0)
                  + max((len(t) for t in ep), default=0)) * _CHAR + 46
        h = _TITLE_H + rows * _PIN_PITCH + 8
        return max(w_title, w_pins), h

    wh = {rid: size_of(rid) for rid in node_ids}
    layers = {}
    for rid in node_ids:
        layers.setdefault(rank[rid], []).append(rid)

    x_of, y_of = {}, {}
    x_cursor = _MARGIN
    for r in sorted(layers):
        col = layers[r]
        col_w = max(wh[n][0] for n in col)
        y_cursor = _MARGIN
        for n in col:
            x_of[n] = x_cursor + (col_w - wh[n][0]) / 2.0
            y_of[n] = y_cursor
            y_cursor += wh[n][1] + _ROW_GAP
        x_cursor += col_w + _COL_GUTTER
    width = x_cursor + _MARGIN
    height = (max((y_of[n] + wh[n][1] for n in node_ids), default=100.0)
              + _MARGIN + 18)

    boxes = {n: (x_of[n], y_of[n], wh[n][0], wh[n][1]) for n in node_ids}

    # --- edge geometry: back-edge / self-loop flags ------------------------
    edges_out = []
    for idx, (s, d, sp, dp, net, volt) in enumerate(directed):
        if s not in rank or d not in rank:
            continue
        is_self = s == d
        is_back = is_self or (s, d) in back or rank[d] <= rank[s]
        edges_out.append({
            's': s, 'd': d, 's_pin': sp, 'd_pin': dp, 'net': net,
            'voltage': volt, 'back': is_back, 'self': is_self, 'i': idx,
            'rail': net_role.get(net) in ('power', 'ground'),
        })

    # Feedback curves sweep below the boxes; reserve room so they stay on-canvas.
    if any(e['back'] and not e['self'] for e in edges_out):
        height += 66
    if any(e['self'] for e in edges_out):
        height += 20                      # self-loops arc over the top

    fig_w = min(max(width / _DPI, _FIG_MIN), _FIG_MAX)
    fig_h = min(max(height / _DPI, _FIG_MIN), _FIG_MAX)

    return {
        'name': graph['name'],
        'node_ids': node_ids,
        'inst_meta': inst_meta, 'port_meta': port_meta,
        'boxes': boxes, 'rank': rank,
        'west': west, 'east': east, 'west_map': west_map, 'east_map': east_map,
        'edges': edges_out,
        'unconnected': set(unconnected_ports),
        'suppressed_rails': list(dict.fromkeys(suppressed)),
        'width': width, 'height': height, 'figsize': (fig_w, fig_h),
        'label_lines': {rid: label_lines(rid) for rid in node_ids},
    }


def _port_net(meta, graph):
    """First net a (possibly collapsed) port pill carries — for rail filtering."""
    members = set(meta.get('members', []))
    for p in graph['ports']:
        if p['name'] in members:
            return p['net']
    return None


def _pin_xy(L, rid, pin, side):
    """Fixed named point on a box side; every wire on (rid, pin) lands here."""
    x, y, w, h = L['boxes'][rid]
    if rid.startswith('p:') or not pin:
        return (x + (w if side == 'E' else 0), y + h / 2.0)
    rows = L['east'].get(rid, []) if side == 'E' else L['west'].get(rid, [])
    amap = L['east_map'].get(rid, {}) if side == 'E' else L['west_map'].get(rid, {})
    row = amap.get(pin, pin)
    i = rows.index(row) if row in rows else 0
    return (x + (w if side == 'E' else 0), y + _TITLE_H + (i + 0.5) * _PIN_PITCH)


def save_graph_png(graph: dict, path: str) -> str:
    """Render *graph* as a matplotlib block diagram (v1.2).

    Boxes carry named pin points on their sides (west = receives, east =
    drives); every wire attaches pin-to-pin. Layout is longest-path layering —
    signal flows left to right in columns — and numeric instance/port families
    (``mux0..mux9``) collapse to one box/pill with an ``xN`` badge. Feedback
    (a wire back to an earlier column) sweeps under the graph as a coloured
    curve; self-loops arc over the top. Spec-failing blocks get a distinct
    doubled red border, and a declared port no wire reaches is drawn hollow and
    dashed. High-fanout supplies/clocks are summarized in a caption. Only the
    immediate children are drawn; ``depth>1`` subgraphs are a Mermaid feature.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    L = _compute_png_layout(graph)
    W, H = L['width'], L['height']
    label_big = len(L['node_ids']) <= 16

    fig, ax = plt.subplots(figsize=L['figsize'])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)                 # invert y: top-down like the SVG model
    ax.axis('off')
    ax.set_title(f'Composition: {graph["name"]}', fontsize=12, fontweight='bold')

    # edges first (under the boxes)
    for e in L['edges']:
        sx, sy = _pin_xy(L, e['s'], e['s_pin'], 'E')
        dx, dy = _pin_xy(L, e['d'], e['d_pin'], 'W')
        # Every endpoint connects through a stub so the wire lands on a nub
        # OUTSIDE the box, never hidden under it. Instance pins draw their own
        # stub in _draw_inst; a port pill's stub is drawn here (zorder above the
        # pill) so EVERY pill visibly taps its net — no orphan-looking pills.
        if e['s'].startswith('p:'):
            ax.plot([sx, sx + _STUB], [sy, sy], color=_PORT_EDGE, lw=1.6, zorder=5)
        if e['d'].startswith('p:'):
            ax.plot([dx - _STUB, dx], [dy, dy], color=_PORT_EDGE, lw=1.6, zorder=5)
        sx += _STUB
        dx -= _STUB
        color = _BACK_C if e['back'] else _EDGE_C
        # net labels only on non-rail signal wires, and only when uncluttered
        label = '' if (e['rail'] or not label_big) else e['net']
        if e['voltage'] is not None and label:
            label = f'{label} {_fmt_v(e["voltage"])}'
        if e['self']:
            top = L['boxes'][e['s']][1] - 16 - (e['i'] % 3) * 12
            _bezier(ax, (sx, sy), (sx + 46, top), (dx - 46, top), (dx, dy), color)
        elif e['back']:
            # sweep clearly under every box the feedback spans
            lo, hi = sorted((L['rank'][e['s']], L['rank'][e['d']]))
            spanned = [b for rid, b in L['boxes'].items() if lo <= L['rank'][rid] <= hi]
            base = max((b[1] + b[3] for b in spanned),
                       default=L['boxes'][e['s']][1] + L['boxes'][e['s']][3])
            bot = base + 22 + (e['i'] % 3) * 14
            _bezier(ax, (sx, sy), (sx + 50, bot), (dx - 50, bot), (dx, dy), color)
            _edge_text(ax, (sx + dx) / 2, bot, label, color)
        else:
            ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3,
                                        shrinkA=0, shrinkB=0), zorder=1)
            _edge_text(ax, (sx + dx) / 2, (sy + dy) / 2, label, color)

    # boxes over the wires
    for rid in L['node_ids']:
        x, y, w, h = L['boxes'][rid]
        if rid.startswith('p:'):
            _draw_pill(ax, rid, L, x, y, w, h)
        else:
            _draw_inst(ax, rid, L, x, y, w, h)

    if L['suppressed_rails']:
        ax.text(_MARGIN, H - 6, 'shared rails (not drawn): '
                + ', '.join(L['suppressed_rails']),
                ha='left', va='bottom', fontsize=8, color='#888', style='italic')

    fig.savefig(path, dpi=_DPI, bbox_inches='tight')
    plt.close(fig)
    return path


def _draw_inst(ax, rid, L, x, y, w, h):
    from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
    meta = L['inst_meta'][rid]
    status = (meta.get('spec') or {}).get('status') if meta['kind'] == 'submodule' else None
    face = _KIND_FACE.get(meta['kind'], '#eeeeee')
    edge_c = _STATUS_EDGE.get(status, '#333') if meta['kind'] == 'submodule' else '#333'
    lw = 2.4 if status == 'fail' else (1.9 if meta['kind'] == 'submodule' else 1.2)
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0,rounding_size=5',
                         facecolor=face, edgecolor=edge_c, linewidth=lw, zorder=3)
    ax.add_patch(box)
    if status == 'fail':
        # distinct SHAPE (a doubled border), not merely a colour
        ax.add_patch(Rectangle((x + 3, y + 3), w - 6, h - 6, fill=False,
                               edgecolor=edge_c, linewidth=1.0, zorder=3))

    for j, line in enumerate(L['label_lines'][rid]):
        ax.text(x + w / 2, y + 14 + j * 13, line, ha='center', va='center',
                fontsize=9, fontweight='bold' if j == 0 else 'normal', zorder=4)
    ax.plot([x + 6, x + w - 6], [y + _TITLE_H - 6, y + _TITLE_H - 6],
            color=edge_c, lw=0.5, alpha=0.5, zorder=3)

    for i, pin in enumerate(L['west'].get(rid, [])):
        py = y + _TITLE_H + (i + 0.5) * _PIN_PITCH
        ax.plot([x - _STUB, x], [py, py], color=edge_c, lw=1.4, zorder=3)
        ax.add_patch(Circle((x, py), 2.4, facecolor=edge_c, edgecolor=edge_c, zorder=4))
        ax.text(x + 6, py, pin, ha='left', va='center', fontsize=7.5, zorder=4)
    for i, pin in enumerate(L['east'].get(rid, [])):
        py = y + _TITLE_H + (i + 0.5) * _PIN_PITCH
        ax.plot([x + w, x + w + _STUB], [py, py], color=edge_c, lw=1.4, zorder=3)
        ax.add_patch(Circle((x + w, py), 2.4, facecolor=edge_c, edgecolor=edge_c, zorder=4))
        ax.text(x + w - 6, py, pin, ha='right', va='center', fontsize=7.5, zorder=4)


def _draw_pill(ax, rid, L, x, y, w, h):
    from matplotlib.patches import FancyBboxPatch
    meta = L['port_meta'][rid]
    open_ = rid in L['unconnected']
    face = 'white' if open_ else _PORT_FACE
    edge_c = _OPEN_EDGE if open_ else _PORT_EDGE
    style = (0, (4, 3)) if open_ else 'solid'
    pill = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={h/2}',
                          facecolor=face, edgecolor=edge_c, linewidth=1.3,
                          linestyle=style, zorder=3)
    ax.add_patch(pill)
    badge = f' x{meta["count"]}' if meta.get('count', 1) > 1 else ''
    ax.text(x + w / 2, y + h / 2, meta['label'] + badge, ha='center', va='center',
            fontsize=9, color=edge_c if open_ else '#333', zorder=4)


def _bezier(ax, p0, p1, p2, p3, color):
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch(path=MplPath([p0, p1, p2, p3],
                                       [MplPath.MOVETO, MplPath.CURVE4,
                                        MplPath.CURVE4, MplPath.CURVE4]),
                          arrowstyle='-|>', mutation_scale=11, color=color,
                          lw=1.3, shrinkA=0, shrinkB=0, zorder=1)
    ax.add_patch(arr)


def _edge_text(ax, mx, my, label, color):
    if not label:
        return
    ax.text(mx, my, label, ha='center', va='center', fontsize=7, color='#333',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85),
            zorder=2)
