# tests/test_composition.py
"""Tests for silk.circuit.composition — the post-run composition graphic.

Builds a small synthetic hierarchy (no PDK devices) and exercises the pure
graph extraction, both renderers, and the CircuitReport integration hook.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir, PortRole
from silk.elements import elements
from silk.sources import VoltageSource
from silk.specs.spec import Measurement
from silk.circuit.report import generate_report
from silk.circuit import composition as C


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    return pdk


def _child(tmp_path, name, ports):
    m = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    m.define_module_name(name)
    m.define_module_ports(ports)
    return m


def _build_hierarchy(tmp_path, *, with_spec=True, spec_passes=True):
    """Parent with two child Modules sharing net MID, plus a passive + source.

    - ``blk_a`` (1 internal resistor) drives MID; ``blk_b`` (1 internal
      resistor) is driven by it — MID is the shared internal net.
    - A parent-level ``rload`` resistor also taps MID (making MID a 3-pin net —
      exercises the junction path).
    - A parent-level ``vin`` source drives IN.
    - Optionally a spec on ``blk_a`` with a recorded measurement. Declared and
      measured BEFORE add_element, because add_element deep-copies the child.
    """
    a = _child(tmp_path, 'blk_a', {'in': PortDir.IN, 'out': PortDir.OUT})
    a.add(elements.resistor('ra', p='in', n='out', value=1e3))
    if with_spec:
        sp = a.declare_spec('gain', 'gain_metric', '>', 5.0, units='')
        sp.record_measurement(Measurement(source='user',
                                          value=10.0 if spec_passes else 1.0))

    b = _child(tmp_path, 'blk_b', {'in': PortDir.IN, 'out': PortDir.OUT})
    b.add(elements.resistor('rb', p='in', n='out', value=2e3))

    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk(),
                 module_name='top')
    top.define_module_ports({
        'IN':  PortDir.IN,
        'OUT': PortDir.OUT,
        'VSS': (PortDir.IN, PortRole.GROUND),
    })
    top.add_net('MID')
    top.add_module(a, 'a', nets={'in': 'IN', 'out': 'MID'})
    top.add_module(b, 'b', nets={'in': 'MID', 'out': 'OUT'})
    top.add(elements.resistor('rload', p='MID', n='VSS', value=5e3))
    top.add_source(VoltageSource('vin', positive='IN', negative='VSS', dc=1.0))
    return top


def _build_chain(tmp_path, n):
    """A linear cascade of *n* stage Modules: st0 -> st1 -> ... -> st{n-1}.

    Each stage's ``out`` feeds the next stage's ``in`` through a private 2-pin
    net (a chain link); a shared ``VDD`` touches every stage (a high-fanout
    rail). IN feeds st0, OUT taps st{n-1}.
    """
    stage = _child(tmp_path, 'stage', {'in': PortDir.IN, 'out': PortDir.OUT,
                                       'vdd': PortDir.INOUT})
    top = Module('TEST', str(tmp_path), 'chain.cir', pdk_details=_pdk(),
                 module_name='chain')
    top.define_module_ports({'IN': PortDir.IN, 'OUT': PortDir.OUT,
                             'VDD': (PortDir.IN, PortRole.POWER)})
    for i in range(n - 1):
        top.add_net(f's{i}')
    for i in range(n):
        inp = 'IN' if i == 0 else f's{i-1}'
        outp = 'OUT' if i == n - 1 else f's{i}'
        top.add_module(stage, f'st{i}', nets={'in': inp, 'out': outp, 'vdd': 'VDD'})
    return top


def _build_wide(tmp_path, n):
    """*n* DISTINCT-named children (no numeric family, so no collapse) sharing a
    high-fanout input ``A`` and rail ``VDD``. Used to exercise overlap-free
    placement and rail suppression at scale."""
    leaf = _child(tmp_path, 'leaf', {'a': PortDir.IN, 'vdd': PortDir.INOUT,
                                     'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'wide.cir', pdk_details=_pdk(),
                 module_name='wide')
    top.define_module_ports({'A': PortDir.IN, 'VDD': (PortDir.IN, PortRole.POWER)})
    for i in range(n):
        name = 'blk' + chr(ord('a') + i)          # blka, blkb, ... (no digits)
        top.add_module(leaf, name, nets={'a': 'A', 'vdd': 'VDD',
                                          'out': f'o_{name}'})
    return top


def _build_reverse_driver(tmp_path):
    """A sink declared FIRST and a driver declared SECOND share net M. Real pin
    direction (drv.out is an output) must win over declaration order."""
    sink = _child(tmp_path, 'sink', {'in': PortDir.IN})
    drv = _child(tmp_path, 'drv', {'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'rev.cir', pdk_details=_pdk(),
                 module_name='rev')
    top.add_net('M')
    top.add_module(sink, 'sk', nets={'in': 'M'})     # declared first
    top.add_module(drv, 'dr', nets={'out': 'M'})     # declared second, drives M
    return top


def _build_tapped_output(tmp_path):
    """The FB bug shape: a net driven by one child and read by another is ALSO
    exported as an output port. The port pill must still get a visible tap."""
    drv = _child(tmp_path, 'drv', {'out': PortDir.OUT})
    snk = _child(tmp_path, 'snk', {'in': PortDir.IN})
    top = Module('TEST', str(tmp_path), 'tap.cir', pdk_details=_pdk(),
                 module_name='tap')
    top.define_module_ports({'FBP': PortDir.OUT})
    top.add_module(drv, 'd', nets={'out': 'FBP'})    # drv.out drives net FBP
    top.add_module(snk, 's', nets={'in': 'FBP'})     # snk.in reads net FBP
    return top


def _orphan_port_pills(layout):
    """Port pills that are neither tapped by an edge nor drawn as dashed-hollow
    'unconnected' — i.e. pills that look like they float."""
    touched = {x for e in layout['edges'] for x in (e['s'], e['d'])}
    return [rid for rid in layout['node_ids']
            if rid.startswith('p:') and rid not in touched
            and rid not in layout['unconnected']]


def _build_series(tmp_path, names):
    """Distinct-named children wired head-to-tail: IN -> c0 -> c1 -> ... -> OUT.
    A forward signal cascade with no families — for rank-monotonicity."""
    seg = _child(tmp_path, 'seg', {'in': PortDir.IN, 'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'series.cir', pdk_details=_pdk(),
                 module_name='series')
    top.define_module_ports({'IN': PortDir.IN, 'OUT': PortDir.OUT})
    mids = [f'n{i}' for i in range(len(names) - 1)]
    for nn in mids:
        top.add_net(nn)
    for i, nm in enumerate(names):
        inp = 'IN' if i == 0 else mids[i - 1]
        outp = 'OUT' if i == len(names) - 1 else mids[i]
        top.add_module(seg, nm, nets={'in': inp, 'out': outp})
    return top


# ---------------------------------------------------------------------------
# module_graph — nodes
# ---------------------------------------------------------------------------

def test_graph_name(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    assert g['name'] == 'top'


def test_node_set_and_kinds(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    by_name = {n['name']: n for n in g['nodes']}
    assert set(by_name) == {'a', 'b', 'rload', 'vin'}
    assert by_name['a']['kind'] == 'submodule'
    assert by_name['b']['kind'] == 'submodule'
    assert by_name['rload']['kind'] == 'passive'
    assert by_name['vin']['kind'] == 'source'


def test_node_module_labels(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    by_name = {n['name']: n for n in g['nodes']}
    assert by_name['a']['module_name'] == 'blk_a'
    assert by_name['b']['module_name'] == 'blk_b'
    assert by_name['rload']['module_name'] == 'resistor'
    assert by_name['vin']['module_name'] == 'VoltageSource'


def test_device_counts(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    by_name = {n['name']: n for n in g['nodes']}
    # Each submodule holds exactly one internal resistor.
    assert by_name['a']['device_count'] == 1
    assert by_name['b']['device_count'] == 1
    # Non-submodule children carry no recursive count.
    assert by_name['rload']['device_count'] == 0
    assert by_name['vin']['device_count'] == 0


def test_device_count_is_recursive(tmp_path):
    """A grandchild's devices roll up into the child's device_count."""
    leaf = _child(tmp_path, 'leaf', {'in': PortDir.IN, 'out': PortDir.OUT})
    leaf.add(elements.resistor('rl', p='in', n='out', value=1e3))

    mid = _child(tmp_path, 'mid', {'in': PortDir.IN, 'out': PortDir.OUT})
    mid.add(elements.resistor('rm', p='in', n='mnode', value=1e3))
    mid.add_net('mnode')
    mid.add_module(leaf, 'leaf1', nets={'in': 'mnode', 'out': 'out'})

    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk(),
                 module_name='top')
    top.define_module_ports({'IN': PortDir.IN, 'OUT': PortDir.OUT})
    top.add_module(mid, 'mid1', nets={'in': 'IN', 'out': 'OUT'})

    g = C.module_graph(top)
    mid_node = next(n for n in g['nodes'] if n['name'] == 'mid1')
    # rm (in mid) + rl (in leaf) == 2
    assert mid_node['device_count'] == 2


# ---------------------------------------------------------------------------
# module_graph — edges
# ---------------------------------------------------------------------------

def test_shared_net_edge_endpoints(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    mid = next(e for e in g['edges'] if e['net'] == 'MID')
    endpoints = {(ep['instance'], ep['port']) for ep in mid['endpoints']}
    assert ('a', 'out') in endpoints
    assert ('b', 'in') in endpoints
    assert ('rload', 'p') in endpoints
    # MID is internal, not a top-level port.
    assert mid['is_port'] is False


def test_single_pin_net_is_not_an_edge(tmp_path):
    """A net with fewer than two bound pins is not an edge."""
    g = C.module_graph(_build_hierarchy(tmp_path))
    # OUT is bound only by b.out — one pin — so it is not in edges.
    assert 'OUT' not in {e['net'] for e in g['edges']}


def test_port_net_edge_is_flagged(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    # IN is bound by a.in AND vin.positive (two pins) and is a top-level port.
    in_edge = next(e for e in g['edges'] if e['net'] == 'IN')
    assert in_edge['is_port'] is True
    assert in_edge['port_name'] == 'IN'


def test_edge_voltage_tail(tmp_path):
    """When a net carries a behavioral waveform, the edge reports its tail."""
    top = _build_hierarchy(tmp_path)
    top.net_dict['MID'].voltage = np.linspace(0.0, 1.234, 50)
    g = C.module_graph(top)
    mid = next(e for e in g['edges'] if e['net'] == 'MID')
    assert mid['voltage'] == pytest.approx(1.234, rel=1e-6)


def test_edge_voltage_none_without_waveform(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    mid = next(e for e in g['edges'] if e['net'] == 'MID')
    assert mid['voltage'] is None


# ---------------------------------------------------------------------------
# module_graph — ports
# ---------------------------------------------------------------------------

def test_port_list(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    ports = {p['name']: p for p in g['ports']}
    assert set(ports) == {'IN', 'OUT', 'VSS'}
    assert 'in' in ports['IN']['direction'].lower()
    assert 'out' in ports['OUT']['direction'].lower()
    assert ports['VSS']['role'] == 'ground'


def test_port_endpoints(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    ports = {p['name']: p for p in g['ports']}
    out_eps = {(ep['instance'], ep['port']) for ep in ports['OUT']['endpoints']}
    assert ('b', 'out') in out_eps


# ---------------------------------------------------------------------------
# spec annotation
# ---------------------------------------------------------------------------

def test_spec_status_pass(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path, spec_passes=True))
    a = next(n for n in g['nodes'] if n['name'] == 'a')
    assert a['spec']['status'] == 'pass'
    assert a['spec']['pass'] == 1
    assert a['spec']['fail'] == 0


def test_spec_status_fail(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path, spec_passes=False))
    a = next(n for n in g['nodes'] if n['name'] == 'a')
    assert a['spec']['status'] == 'fail'
    assert a['spec']['fail'] == 1


def test_spec_status_unmeasured(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path, with_spec=False))
    a = next(n for n in g['nodes'] if n['name'] == 'a')
    # No specs at all → unmeasured, non-fatally.
    assert a['spec']['status'] == 'unmeasured'
    assert a['spec']['total'] == 0


def test_spec_status_does_not_reconcile(tmp_path):
    """Reading spec status must not fire calibration side effects."""
    top = _build_hierarchy(tmp_path, spec_passes=True)
    # The stored child instance's behavioral params (if any) must be untouched.
    stored_a = top.elem_dict['a']['device']
    before = getattr(stored_a, '_behavioral_params', None)
    C.module_graph(top)
    assert getattr(stored_a, '_behavioral_params', None) is before


# ---------------------------------------------------------------------------
# graph_to_mermaid
# ---------------------------------------------------------------------------

def test_mermaid_header_and_nodes(tmp_path):
    mm = C.graph_to_mermaid(C.module_graph(_build_hierarchy(tmp_path)))
    assert mm.startswith('flowchart LR')
    # Submodule box carries name, module name, device count, and a status glyph.
    assert 'n_a["a<br/>(blk_a)<br/>1 dev<br/>' in mm
    # Source rendered with a distinct (parallelogram) shape.
    assert 'n_vin[/"vin<br/>(VoltageSource)"/]' in mm
    # Top-level ports are stadium nodes.
    assert 'p_IN(["IN"])' in mm


def test_mermaid_multi_pin_net_uses_junction(tmp_path):
    """MID touches three pins → a junction node, not N^2 edges."""
    mm = C.graph_to_mermaid(C.module_graph(_build_hierarchy(tmp_path)))
    assert 'j_MID{{"MID"}}' in mm
    assert 'n_a --- j_MID' in mm
    assert 'n_b --- j_MID' in mm


def test_mermaid_two_pin_net_is_labeled_edge(tmp_path):
    top = _build_hierarchy(tmp_path)
    mm = C.graph_to_mermaid(C.module_graph(top))
    # OUT port has a single fanout (b.out) → a direct labeled stub, not a junction.
    assert 'p_OUT ---|"OUT"| n_b' in mm


def test_mermaid_edge_voltage_label(tmp_path):
    top = _build_hierarchy(tmp_path)
    top.net_dict['MID'].voltage = np.full(10, 0.9)
    mm = C.graph_to_mermaid(C.module_graph(top))
    assert '0.9V' in mm


def test_mermaid_depth_two_subgraph(tmp_path):
    mm = C.graph_to_mermaid(C.module_graph(_build_hierarchy(tmp_path), depth=2))
    # blk_a expands into a nested subgraph exposing its internal resistor.
    assert 'subgraph sg_a[' in mm
    assert 'sg_a_n_ra' in mm
    assert 'end' in mm


# ---------------------------------------------------------------------------
# save_graph_png
# ---------------------------------------------------------------------------

def test_save_graph_png_writes_nonempty(tmp_path):
    g = C.module_graph(_build_hierarchy(tmp_path))
    out = tmp_path / 'comp.png'
    ret = C.save_graph_png(g, str(out))
    assert ret == str(out)
    assert out.exists()
    assert out.stat().st_size > 0
    # PNG magic number.
    assert out.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'


def test_save_graph_png_large_chain_writes(tmp_path):
    """A 25-stage chain (the v1.0 single-column collapse case) still renders."""
    g = C.module_graph(_build_chain(tmp_path, 25))
    out = tmp_path / 'chain.png'
    C.save_graph_png(g, str(out))
    assert out.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'


# ---------------------------------------------------------------------------
# v1.2 PNG layout — family collapse, layering, overlap-proof placement
# ---------------------------------------------------------------------------

def _boxes_overlap(a, b):
    # rectangles are (x, y, w, h) with (x, y) the top-left corner
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    return x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1


def _assert_no_node_overlap(layout):
    ids = list(layout['boxes'])
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = layout['boxes'][ids[i]], layout['boxes'][ids[j]]
            assert not _boxes_overlap(a, b), \
                f'boxes {ids[i]} and {ids[j]} overlap: {a} vs {b}'


def test_layout_no_overlap_at_25_nodes(tmp_path):
    """No box overlaps even at 25+ distinct (non-collapsing) nodes."""
    layout = C._compute_png_layout(C.module_graph(_build_wide(tmp_path, 25)))
    # 25 children + 2 ports (A/VDD are suppressed rails, so pills may drop) —
    # at least the 25 instance boxes are present and non-overlapping.
    inst_boxes = [rid for rid in layout['boxes'] if rid.startswith('i:')]
    assert len(inst_boxes) == 25
    _assert_no_node_overlap(layout)


def test_layout_no_overlap_mixed(tmp_path):
    """Submodules + a passive + a source: still no overlaps."""
    _assert_no_node_overlap(C._compute_png_layout(C.module_graph(_build_hierarchy(tmp_path))))


def test_family_collapse(tmp_path):
    """A numeric instance family (st0..st5) collapses to one xN render node."""
    layout = C._compute_png_layout(C.module_graph(_build_chain(tmp_path, 6)))
    inst_nodes = [rid for rid in layout['node_ids'] if rid.startswith('i:')]
    assert inst_nodes == ['i:st#']                 # the whole family is one box
    assert layout['inst_meta']['i:st#']['count'] == 6
    assert len(layout['inst_meta']['i:st#']['members']) == 6


def test_family_collapse_only_above_two(tmp_path):
    """A 2-member family is NOT collapsed (needs > 2)."""
    layout = C._compute_png_layout(C.module_graph(_build_chain(tmp_path, 2)))
    inst_nodes = sorted(rid for rid in layout['node_ids'] if rid.startswith('i:'))
    assert inst_nodes == ['i:st0', 'i:st1']        # two separate boxes


def test_port_family_collapse(tmp_path):
    """A numeric port family (P0..P4) collapses to one pill with a count."""
    seg = _child(tmp_path, 'seg', {'p': PortDir.IN, 'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'pf.cir', pdk_details=_pdk(),
                 module_name='pf')
    ports = {'OUT': PortDir.OUT}
    for i in range(5):
        ports[f'P{i}'] = PortDir.IN
    top.define_module_ports(ports)
    # one sink so the P pills are actually wired (not dropped as unconnected)
    top.add_net('mid')
    for i in range(5):
        top.add_module(seg, f's{i}', nets={'p': f'P{i}', 'out': 'OUT'})
    layout = C._compute_png_layout(C.module_graph(top))
    port_nodes = [rid for rid in layout['node_ids'] if rid.startswith('p:P')]
    assert port_nodes == ['p:P#']
    assert layout['port_meta']['p:P#']['count'] == 5


def test_forward_edges_are_rank_monotonic(tmp_path):
    """Every non-feedback edge points to a strictly higher rank (left-to-right)."""
    layout = C._compute_png_layout(
        C.module_graph(_build_series(tmp_path, ['alpha', 'bravo', 'charlie', 'delta'])))
    rank = layout['rank']
    for e in layout['edges']:
        if not e['back']:
            assert rank[e['d']] > rank[e['s']], \
                f"forward edge {e['s']}->{e['d']} not rank-increasing"
    # the cascade itself spans four increasing ranks
    seg_ranks = [rank[rid] for rid in layout['node_ids'] if rid.startswith('i:')]
    assert len(set(seg_ranks)) == 4


def test_feedback_is_a_back_edge(tmp_path):
    """A wire back to an earlier stage is flagged as a back-edge (feedback)."""
    # st0..st5 in a ring: series plus collapse makes st#->st# a self/back edge.
    layout = C._compute_png_layout(C.module_graph(_build_chain(tmp_path, 6)))
    assert any(e['back'] for e in layout['edges'])


def test_high_fanout_rail_is_suppressed(tmp_path):
    """A rail touching many distinct blocks is summarized, not drawn."""
    layout = C._compute_png_layout(C.module_graph(_build_wide(tmp_path, 10)))
    assert 'VDD' in layout['suppressed_rails']


def test_figure_grows_with_node_count(tmp_path):
    """More nodes -> a bigger canvas (never a fixed cramped size)."""
    small = C._compute_png_layout(C.module_graph(_build_wide(tmp_path, 3)))
    big = C._compute_png_layout(C.module_graph(_build_wide(tmp_path, 25)))
    assert (big['figsize'][0] * big['figsize'][1]
            > small['figsize'][0] * small['figsize'][1])


def test_layout_deterministic(tmp_path):
    """Same graph -> identical geometry (no randomness)."""
    g = C.module_graph(_build_series(tmp_path, ['a1', 'b2', 'c3']))
    assert C._compute_png_layout(g)['boxes'] == C._compute_png_layout(g)['boxes']


def test_spec_fail_status_reaches_layout(tmp_path):
    """A failing block's status flows to inst_meta so the renderer can give it
    the distinct (doubled-border) shape."""
    layout = C._compute_png_layout(
        C.module_graph(_build_hierarchy(tmp_path, spec_passes=False)))
    assert layout['inst_meta']['i:a']['spec']['status'] == 'fail'


def test_family_spec_status_aggregates(tmp_path):
    """A collapsed family reports 'fail' if any member fails."""
    members = [{'spec': {'status': 'pass'}, 'kind': 'submodule'}] * 2 + \
              [{'spec': {'status': 'fail'}, 'kind': 'submodule'}]
    assert C._agg_status(members)['status'] == 'fail'
    allpass = [{'spec': {'status': 'pass'}, 'kind': 'submodule'}] * 3
    assert C._agg_status(allpass)['status'] == 'pass'


# ---------------------------------------------------------------------------
# Real pin direction on graph endpoints (additive; drives orientation)
# ---------------------------------------------------------------------------

def test_edge_endpoints_carry_direction_and_role(tmp_path):
    """Each edge endpoint additively carries the pin's real direction/role."""
    g = C.module_graph(_build_hierarchy(tmp_path))
    mid = next(e for e in g['edges'] if e['net'] == 'MID')
    by = {(ep['instance'], ep['port']): ep for ep in mid['endpoints']}
    # a.out is a declared output, b.in a declared input (from the child ports)
    assert by[('a', 'out')]['direction'] == 'output'
    assert by[('b', 'in')]['direction'] == 'input'
    # additive: the original keys are still present and unchanged
    assert set(by[('a', 'out')]) >= {'instance', 'port', 'direction', 'role'}


def test_port_endpoints_carry_direction(tmp_path):
    """Port endpoints carry the child pin's real direction too."""
    g = C.module_graph(_build_hierarchy(tmp_path))
    out_port = next(p for p in g['ports'] if p['name'] == 'OUT')
    ep = next(e for e in out_port['endpoints'] if e['instance'] == 'b')
    assert ep['direction'] == 'output'          # b.out drives the OUT port


def test_orientation_uses_real_direction_not_declaration_order(tmp_path):
    """The driver is chosen from real pin direction, even when it is declared
    AFTER the sink (declaration-order fallback would pick the wrong end)."""
    layout = C._compute_png_layout(C.module_graph(_build_reverse_driver(tmp_path)))
    pairs = {(e['s'], e['d']) for e in layout['edges']}
    assert ('i:dr', 'i:sk') in pairs            # driver -> sink, per real direction
    assert ('i:sk', 'i:dr') not in pairs
    # and the driver ranks strictly before the sink (left-to-right)
    assert layout['rank']['i:dr'] < layout['rank']['i:sk']


def test_no_pin_name_heuristic_symbol(tmp_path):
    """The renderer must not carry a hardcoded pin-name->role table."""
    assert not hasattr(C, '_is_output_pin')
    assert not hasattr(C, '_OUTPUT_PIN_NAMES')


# ---------------------------------------------------------------------------
# Every port pill connects (no floating pills)
# ---------------------------------------------------------------------------

def test_exported_net_port_is_tapped(tmp_path):
    """The FB regression: a net that is both an inter-child edge AND an output
    port must give its pill a drawn tap (not a floating pill)."""
    layout = C._compute_png_layout(C.module_graph(_build_tapped_output(tmp_path)))
    touched = {x for e in layout['edges'] for x in (e['s'], e['d'])}
    assert 'p:FBP' in touched                       # the port pill is wired
    assert _orphan_port_pills(layout) == []


def test_no_orphan_port_pills_in_hierarchy(tmp_path):
    """No drawn port pill floats: each is either tapped or explicitly dashed."""
    layout = C._compute_png_layout(C.module_graph(_build_hierarchy(tmp_path)))
    assert _orphan_port_pills(layout) == []


def test_every_wired_port_reaches_an_edge(tmp_path):
    """Every top-level port that has child endpoints ends up tapped."""
    g = C.module_graph(_build_hierarchy(tmp_path))
    layout = C._compute_png_layout(g)
    touched = {x for e in layout['edges'] for x in (e['s'], e['d'])}
    wired = {p['name'] for p in g['ports'] if p['endpoints']}
    for rid in layout['node_ids']:
        if not rid.startswith('p:'):
            continue
        members = set(layout['port_meta'][rid]['members'])
        if members & wired:                          # this pill carries a wired port
            assert rid in touched, f'{rid} carries a wired port but has no tap'


# ---------------------------------------------------------------------------
# CircuitReport integration
# ---------------------------------------------------------------------------

def test_report_add_composition_diagram_markdown(tmp_path):
    top = _build_hierarchy(tmp_path)
    report = generate_report(top)
    report.add_composition_diagram(top, depth=1)

    md_path = tmp_path / 'report.md'
    report.save(str(md_path))
    md = md_path.read_text()
    assert '## Composition Diagram' in md
    assert '```mermaid' in md
    assert 'flowchart LR' in md
    assert 'n_a[' in md


def test_report_add_composition_diagram_html(tmp_path):
    top = _build_hierarchy(tmp_path)
    report = generate_report(top)
    report.add_composition_diagram(top, depth=1)

    html_path = tmp_path / 'report.html'
    report.save_html(str(html_path))
    html = html_path.read_text()
    assert 'Composition Diagram' in html
    assert 'pre class="mermaid"' in html


def test_report_without_composition_has_no_section(tmp_path):
    top = _build_hierarchy(tmp_path)
    report = generate_report(top)
    md = report._to_markdown()
    assert 'Composition Diagram' not in md


def test_report_json_still_serializes_with_composition(tmp_path):
    """The new fields are repr/compare/JSON-excluded — save_json must not choke."""
    top = _build_hierarchy(tmp_path)
    report = generate_report(top)
    report.add_composition_diagram(top, depth=1)
    json_path = tmp_path / 'report.json'
    report.save_json(str(json_path))  # must not raise
    assert json_path.exists()
