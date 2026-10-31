"""TASK-312.1: every passives_dict consumer treats an ideal R/C/L identically
whether it landed in ``elem_dict`` (``Module.add(elements.resistor(...))``) or
``passives_dict`` (``Module.add_passive(elements.resistor(...))``).

The two build paths are otherwise the SAME object type (``AnalogElement``,
``_is_ideal_passive`` true) -- only the dict it ends up registered in differs.
A consumer that keys behavior off "which dict" instead of "what it emits"
disagrees between the two for what is, semantically, the identical circuit.

Each test below builds the identical two-terminal circuit both ways and
checks one consumer's output. See TASK-312.1's implementation notes for the
full measurement table (including the consumers found to already be
family-independent, so they get no fix and a lighter smoke test here).
"""
import time

import pytest

from silk.circuit.modules import Module
from silk.elements import elements


def _build(pdk, tmp_dir, path, name='meas'):
    """*path*: 'add' registers in elem_dict, 'add_passive' in passives_dict."""
    m = Module('PDK_skywater130', tmp_dir, f'{name}_{path}.cir', pdk_details=pdk)
    m.add_net('a')
    m.add_net('b')
    m.add_net('gnd')
    r = elements.resistor(name='r1', p='a', n='b', value=1000)
    c = elements.capacitor(name='c1', p='b', n='gnd', value=1e-12)
    if path == 'add':
        m.add(r)
        m.add(c)
    else:
        m.add_passive(r)
        m.add_passive(c)
    return m


# ---------------------------------------------------------------------------
# structure_tracker — structure hash must not depend on which dict holds the
# ideal R/C/L (family-independent hash contract).
# ---------------------------------------------------------------------------

def test_structure_hash_identical_for_add_and_add_passive(pdk, tmp_dir):
    from silk.db.structure_tracker import CircuitStructureTracker
    from silk.db.circuit_db import Database

    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')

    tracker = CircuitStructureTracker(Database(':memory:'), None)
    s_add = tracker.extract_structure(m_add)
    s_ap = tracker.extract_structure(m_ap)

    assert (sorted(s_add['passives'], key=lambda d: d['name'])
            == sorted(s_ap['passives'], key=lambda d: d['name']))
    assert s_add['transistors'] == s_ap['transistors'] == []

    h_add, _ = tracker.compute_structure_hash(s_add)
    h_ap, _ = tracker.compute_structure_hash(s_ap)
    assert h_add == h_ap


# ---------------------------------------------------------------------------
# spice_netlister — the emitted element LINE (to_spice()) must be byte
# identical regardless of dict; deck line ORDER and comment banners are
# allowed to differ (recorded, not fixed -- see task notes) because add()
# emits the element in the device section (with a per-instance banner) while
# add_passive() emits it in the passives section (single section banner).
# ---------------------------------------------------------------------------

def test_netlister_element_line_bytes_identical(pdk, tmp_dir):
    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')

    assert (m_add.elem_dict['r1']['device'].to_spice()
            == m_ap.passives_dict['r1'].to_spice())
    assert (m_add.elem_dict['c1']['device'].to_spice()
            == m_ap.passives_dict['c1'].to_spice())


# ---------------------------------------------------------------------------
# modules_results — per-terminal current-annotation dispatch (TASK-308) must
# not depend on which dict resolved the element.
# ---------------------------------------------------------------------------

def test_current_row_identical_for_add_and_add_passive(pdk, tmp_dir):
    from silk.circuit.modules_results import _is_ideal_passive, _current_row

    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')

    r_dev = m_add.elem_dict['r1']['device']
    r_pas = m_ap.passives_dict['r1']
    assert _is_ideal_passive(r_dev) and _is_ideal_passive(r_pas)
    for port in ('p', 'n'):
        assert _current_row(r_dev, port) == _current_row(r_pas, port)


# ---------------------------------------------------------------------------
# lint/rules/structural — MissingParamRule (STRUCT-006) is the one rule that
# reads passives_dict directly (as a second pass alongside elem_dict); an
# unsized ideal R must be reported identically regardless of which dict holds
# it.
# ---------------------------------------------------------------------------

def test_missing_param_rule_identical_for_add_and_add_passive(pdk, tmp_dir):
    from silk.lint.rules.structural import MissingParamRule

    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')

    r_add = elements.resistor(name='r2', p='a', n='b', value=1000)
    r_add.params['value'] = None
    r_ap = elements.resistor(name='r2', p='a', n='b', value=1000)
    r_ap.params['value'] = None
    m_add.add(r_add)
    m_ap.add_passive(r_ap)

    rule = MissingParamRule()
    msgs_add = sorted(v.message for v in rule.check(m_add))
    msgs_ap = sorted(v.message for v in rule.check(m_ap))
    assert msgs_add == msgs_ap
    assert any('r2' in m and 'value' in m for m in msgs_add)


# ---------------------------------------------------------------------------
# composition — module_graph() node kind (and therefore Mermaid shape / PNG
# face color) must not depend on which dict holds the ideal R/C/L. THIS was
# the one real bug this task fixed: _child_instances used to yield kind
# 'device' for an elem_dict-resident ideal passive and 'passive' for a
# passives_dict one, giving the identical circuit two different node colors
# (#eeeeee vs #d9f0dd) depending only on which Module.add*() call built it.
# ---------------------------------------------------------------------------

def test_composition_node_kind_identical_for_add_and_add_passive(pdk, tmp_dir):
    from silk.circuit.composition import module_graph, _count_leaf_elements, _KIND_FACE

    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')

    def node_kinds(mod):
        g = module_graph(mod)
        return {n['name']: (n['kind'], n['module_name']) for n in g['nodes']}

    kinds_add = node_kinds(m_add)
    kinds_ap = node_kinds(m_ap)
    assert kinds_add == kinds_ap
    assert kinds_add['r1'][0] == 'passive'
    assert kinds_add['c1'][0] == 'passive'

    # Face color follows kind directly -- pin this down explicitly so a
    # future kind rename can't silently reintroduce the two-colors bug.
    assert _KIND_FACE[kinds_add['r1'][0]] == _KIND_FACE[kinds_ap['r1'][0]] == '#d9f0dd'

    assert _count_leaf_elements(m_add) == _count_leaf_elements(m_ap)


# ---------------------------------------------------------------------------
# layout/parasitics — inject_parasitics() only ever creates/removes its own
# ``__par_*`` passives_dict entries; its output must not depend on whether
# the ORIGINAL circuit's own R/C/L used add() or add_passive().
# ---------------------------------------------------------------------------

def test_inject_parasitics_identical_regardless_of_original_family(pdk, tmp_dir):
    from silk.layout.parasitics import inject_parasitics
    from silk.layout.estimator import LayoutEstimate, NetLayout

    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')

    est = LayoutEstimate(
        computation_mode='statistical', routing_model_id='meas',
        timestamp=time.time(),
        nets={
            'a': NetLayout(length=1.0, width=1.0, r=10.0, c=1.0, equivalence_class=None),
            'b': NetLayout(length=1.0, width=1.0, r=20.0, c=2.0, equivalence_class=None),
        },
        devices={}, bounding_box=(1.0, 1.0), equivalence_classes={}, warnings=(),
    )
    inject_parasitics(m_add, est, mode='rc')
    inject_parasitics(m_ap, est, mode='rc')

    def par_entries(mod):
        return {k: (type(v).__name__, v.value)
                for k, v in mod.passives_dict.items() if k.startswith('__par_')}

    assert par_entries(m_add) == par_entries(m_ap)
    assert par_entries(m_add)  # non-empty: the comparison isn't vacuous


# ---------------------------------------------------------------------------
# simulation/behavioral.py — BehavioralEngine._build_graph iterates elem_dict
# and used to refuse ANY ElementType.DEVICE entry as "a transistor-level
# device", which never saw an add_passive()-built ideal R/C/L (passives_dict
# only) but DOES see the identical add()-built one. Found while migrating
# tests off passives.* (TASK-312.3): a shared test fixture's rload resistor
# moved from add_passive to add() and a behavioral-simulation test started
# raising "Element 'rload' is a transistor-level device". Not one of
# TASK-312.1's 12 audited consumers.
# ---------------------------------------------------------------------------

def test_behavioral_engine_does_not_refuse_ideal_passive_via_add(pdk, tmp_dir):
    from silk import sources
    from silk.simulation.behavioral import BehavioralEngine

    m_add = _build(pdk, tmp_dir, 'add')
    m_ap = _build(pdk, tmp_dir, 'add_passive')
    for m in (m_add, m_ap):
        m.add_source(sources.VoltageSource('v1', positive='a', negative='gnd', dc=1.0))

    engine = BehavioralEngine()
    # Before the fix, engine.run(m_add, ...) raised SimulationError here.
    sig_add, t_add = engine.run(m_add, t_step=1e-6, t_stop=1e-5)
    sig_ap, t_ap = engine.run(m_ap, t_step=1e-6, t_stop=1e-5)

    assert set(sig_add) == set(sig_ap)
    for net in sig_add:
        assert sig_add[net] == pytest.approx(sig_ap[net])
    assert t_add == pytest.approx(t_ap)
