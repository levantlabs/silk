"""Layout-budget fields on Net.

These three fields capture per-net design intent that used to live in
the constraint catalogue (MaxIRDrop, MaxCurrentDensity, MaxNetLength).
They moved onto Net itself because the net is the natural owner of
its own properties — see silk/constraints/DESIGN.md Q5.
"""
import pytest

from silk.circuit.net import Net


def _make_net(name='n1', **overrides):
    return Net(name=name, module=None, **overrides)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_layout_budget_fields_default_to_none():
    n = _make_net()
    assert n.max_ir_drop_mV is None
    assert n.max_j_factor is None
    assert n.max_length_um is None


# ---------------------------------------------------------------------------
# Construction-time and post-construction assignment
# ---------------------------------------------------------------------------

def test_max_ir_drop_mV_round_trip():
    n = _make_net(max_ir_drop_mV=50.0)
    assert n.max_ir_drop_mV == 50.0
    n.max_ir_drop_mV = 25.0
    assert n.max_ir_drop_mV == 25.0


def test_max_j_factor_round_trip():
    n = _make_net(max_j_factor=0.7)
    assert n.max_j_factor == 0.7
    n.max_j_factor = 0.5
    assert n.max_j_factor == 0.5


def test_max_length_um_round_trip():
    n = _make_net(max_length_um=200.0)
    assert n.max_length_um == 200.0
    n.max_length_um = 150.0
    assert n.max_length_um == 150.0


# ---------------------------------------------------------------------------
# Pairing with existing layout fields
# ---------------------------------------------------------------------------

def test_max_length_um_independent_of_estimated_length():
    """Budget (max_length_um) and measurement (estimated_length) coexist
    without interfering — the checker compares them later."""
    n = _make_net(max_length_um=200.0, estimated_length=120.0)
    assert n.max_length_um == 200.0
    assert n.estimated_length == 120.0


# ---------------------------------------------------------------------------
# Module-level: nets created via add_net inherit the same defaults
# ---------------------------------------------------------------------------

def test_module_added_net_starts_with_no_budgets(empty_netlist):
    empty_netlist.add_net('vdd')
    n = empty_netlist.net_dict['vdd']
    assert n.max_ir_drop_mV is None
    assert n.max_j_factor is None
    assert n.max_length_um is None


def test_module_net_budgets_can_be_set_after_add(empty_netlist):
    empty_netlist.add_net('vdd')
    n = empty_netlist.net_dict['vdd']
    n.max_ir_drop_mV = 50.0
    n.max_j_factor = 0.7
    n.max_length_um = 200.0
    rebound = empty_netlist.net_dict['vdd']
    assert rebound.max_ir_drop_mV == 50.0
    assert rebound.max_j_factor == 0.7
    assert rebound.max_length_um == 200.0
