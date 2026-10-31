"""Structure hash must see passives (R/C/L) at every level."""
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import Port
from silk.elements.passives import Resistor, Capacitor
from silk.elements import elements as ideal
from silk.db.structure_tracker import (
    CircuitStructureTracker,
    _ideal_passive_canonical_entry,
)
from silk.db.circuit_db import Database


def _hash(mod):
    t = CircuitStructureTracker(Database(':memory:'), None)
    s = t.extract_structure(mod)
    h, _ = t.compute_structure_hash(s)
    return h


def _blk(pdk, tmp_dir, *, r=1000.0, with_cap=False, name='blk'):
    m = Module('PDK_skywater130', tmp_dir, f'{name}.cir',
               pdk_details=pdk, module_name=name)
    m.define_module_ports({'A': Port.analog_input(), 'B': Port.analog_output()})
    m.add_net('mid')
    m.add_passive(Resistor('r1', pos='A', neg='mid', resistance=r))
    # Always bind port B, independent of with_cap, so the two variants have
    # identical net topology — only the extra capacitor differs.
    m.add_passive(Resistor('r2', pos='mid', neg='B', resistance=1000.0))
    if with_cap:
        m.add_passive(Capacitor('c1', pos='mid', neg='B', capacitance=1e-12))
    return m


def test_hash_deterministic(pdk, tmp_dir):
    assert _hash(_blk(pdk, tmp_dir)) == _hash(_blk(pdk, tmp_dir))


def test_passive_value_changes_hash(pdk, tmp_dir):
    assert _hash(_blk(pdk, tmp_dir, r=1000.0)) != _hash(_blk(pdk, tmp_dir, r=4700.0))


def test_passive_presence_changes_hash(pdk, tmp_dir):
    assert _hash(_blk(pdk, tmp_dir)) != _hash(_blk(pdk, tmp_dir, with_cap=True))


def test_submodule_passive_changes_parent_hash(pdk, tmp_dir):
    def parent(r):
        p = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        p.add_net('x')
        p.add_module(_blk(pdk, tmp_dir, r=r), 'u1', nets={'A': 'x', 'B': '0'})
        return p
    assert _hash(parent(1000.0)) != _hash(parent(4700.0))


# ---------------------------------------------------------------------------
# TASK-299: hand-written passives.* and YAML-built elements.* share ONE
# silk.structure/v3 identity for the same R/C, and today's hand-written-only
# hash is unchanged (owner ruling 2026-09-26).
# ---------------------------------------------------------------------------

_HARDCODED_HASH = 'aeba187148f8018b4eab1fb03765304cf880f7c6465014ff8ca2c5cf5eb7bd65'


def _blk_hardcoded(pdk, tmp_dir, name='blk_hc'):
    m = Module('PDK_skywater130', tmp_dir, f'{name}.cir',
               pdk_details=pdk, module_name=name)
    m.define_module_ports({'A': Port.analog_input(), 'B': Port.analog_output()})
    m.add_net('mid')
    m.add_net('gnd')
    m.add_passive(Resistor('r1', pos='A', neg='mid', resistance=1000.0))
    m.add_passive(Capacitor('c1', pos='mid', neg='gnd', capacitance=1e-12))
    return m


def _blk_yaml(pdk, tmp_dir, name='blk_yaml'):
    m = Module('PDK_skywater130', tmp_dir, f'{name}.cir',
               pdk_details=pdk, module_name=name)
    m.define_module_ports({'A': Port.analog_input(), 'B': Port.analog_output()})
    m.add_net('mid')
    m.add_net('gnd')
    m.add(ideal.resistor('r1', p='A', n='mid', value=1000.0))
    m.add(ideal.capacitor('c1', p='mid', n='gnd', value=1e-12))
    return m


def test_hardcoded_passive_hash_unchanged(pdk, tmp_dir):
    """Regression pin: today's hand-written-passive hash must never move —
    only YAML-built identities are allowed to move to match it."""
    assert _hash(_blk_hardcoded(pdk, tmp_dir)) == _HARDCODED_HASH


def test_yaml_and_hardcoded_ideal_passive_share_identity(pdk, tmp_dir):
    """Same instance names/values/nets, built through passives.*+add_passive
    vs elements.*+add() — ONE silk.structure/v3 identity either way."""
    assert _hash(_blk_hardcoded(pdk, tmp_dir)) == _hash(_blk_yaml(pdk, tmp_dir))


def test_yaml_and_hardcoded_ideal_passive_share_identity_nested(pdk, tmp_dir):
    """Same equality holds recursively, through a submodule instance.

    Both variants use the SAME submodule ``module_name`` ('blk') so only the
    R/C construction path differs — a different name would itself change the
    hierarchy hash, independent of this fix.
    """
    def parent(builder, tag):
        p = Module('PDK_skywater130', tmp_dir, f'{tag}_parent.cir', pdk_details=pdk)
        p.add_net('x')
        p.add_net('y')
        p.add_module(builder(pdk, tmp_dir, name='blk'), 'u1', nets={'A': 'x', 'B': 'y'})
        return p
    assert _hash(parent(_blk_hardcoded, 'hc')) == _hash(parent(_blk_yaml, 'yaml'))


def test_yaml_ideal_passive_differs_from_pdk_device(pdk):
    """Guard: a real PDK device (not an ideal R/C/L) must never be routed
    into the passives canonicalization — only ideal, non-PDK R/C/L qualify."""
    mn = pdk.nmos1p8('mn1', d='d', g='g', s='s', b='b', w=1.0, l=0.15, m=1)
    assert _ideal_passive_canonical_entry('mn1', mn) is None
