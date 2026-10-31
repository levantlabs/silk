"""CircuitTracker identity resolution and compatibility hash convenience."""

import pytest

from silk.circuit.modules import Module
from silk.db.tracking import CircuitTracker


def _build_dut(pdk, tmp_dir, r2=2000.0):
    """Two-resistor divider block; identical calls produce identical structure."""
    from silk.elements import elements
    from silk.circuit.circuit_elements import Port
    m = Module('PDK_skywater130', tmp_dir, 'dut_blk.cir',
               pdk_details=pdk, module_name='dut_blk')
    m.define_module_ports({'A': Port.analog_input(), 'B': Port.analog_output()})
    m.add_net('mid')
    m.add(elements.resistor('r1', p='A', n='mid', value=1000.0))
    m.add(elements.resistor('r2', p='mid', n='B', value=r2))
    return m


def test_hash_is_deterministic_across_builds(pdk, tmp_dir):
    t = CircuitTracker(_build_dut(pdk, tmp_dir), ':memory:')
    h1 = t.compute_structure_hash_for(_build_dut(pdk, tmp_dir))
    h2 = t.compute_structure_hash_for(_build_dut(pdk, tmp_dir))
    assert h1 == h2


def test_hash_changes_with_parameter(pdk, tmp_dir):
    t = CircuitTracker(_build_dut(pdk, tmp_dir), ':memory:')
    h1 = t.compute_structure_hash_for(_build_dut(pdk, tmp_dir, r2=2000.0))
    h2 = t.compute_structure_hash_for(_build_dut(pdk, tmp_dir, r2=4700.0))
    assert h1 != h2


def test_resolve_creates_once_then_reuses(pdk, tmp_dir):
    module = _build_dut(pdk, tmp_dir)
    t = CircuitTracker(module, ':memory:')
    identity = t.compute_snapshot_identity_for(module)

    assert t.resolve_version('dut_blk', identity) is None
    v1 = t.ensure_version(module)
    v2 = t.resolve_version('dut_blk', identity)
    assert v1 == v2
    n = t.db.conn.execute(
        "SELECT COUNT(*) FROM circuit_versions WHERE circuit_name='dut_blk'"
    ).fetchone()[0]
    assert n == 1


def test_resolve_different_hash_new_version(pdk, tmp_dir):
    first = _build_dut(pdk, tmp_dir, r2=2_000.0)
    second = _build_dut(pdk, tmp_dir, r2=4_700.0)
    t = CircuitTracker(first, ':memory:')

    first_id = t.ensure_version(first)
    second_id = t.ensure_version(second)

    assert first_id != second_id


def test_resolve_no_create_returns_none(pdk, tmp_dir):
    module = _build_dut(pdk, tmp_dir)
    t = CircuitTracker(module, ':memory:')
    identity = t.compute_snapshot_identity_for(module)

    assert t.resolve_version('dut_blk', identity) is None
