"""Probe/regression tests P1/P1b/P2 (spec 2026-07-03 §9) — the ngspice/
savecurrents/sky130 assumptions the design rests on, against REAL ngspice.

P1 fired the gate (control-`write` aborts on zero-length savecurrents vectors);
§4.1 resolves it with flag-gated batch-write mode. These now assert the
resolved behaviour and stay as regressions.

Slow: real ngspice + real sky130 models.
"""
import os

import pytest

from silk.pdk.basepdk import BasePDK
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk import sources
from silk.elements import elements
from silk.simulation import analysis

_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_have_pdk = os.path.exists(_SKY130_LIB)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed'),
]


def _pdk():
    return BasePDK(_PDK_YAML, library_location=_SKY130_LIB)


def _hier_tb(tmp_path):
    """Two-level hierarchical tb: real sky130 nmos in a subckt, instantiated in
    a tb with a top-level resistor + sources + single transient."""
    pdk = _pdk()
    leaf = Module('PDK_skywater130', str(tmp_path), 'leaf.cir',
                  pdk_details=pdk, module_name='leafcell')
    leaf.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT,
                              'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with leaf:
        pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1.0, l=0.15)

    tb = Module('PDK_skywater130', str(tmp_path), 'p1_tb.cir',
                pdk_details=pdk, module_name=None)
    tb.add_module(leaf, 'dut',
                   nets={'in': 'vin', 'out': 'vout', 'vdd': 'vdd', 'vss': '0'})
    with tb:
        elements.resistor('rload', p='vout', n='0', value=10e3)
        sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)
        sources.VoltageSource('vin', positive='vin', negative='0', dc=0.9)
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=40e-9))
    return tb


# ---------------------------------------------------------------------------
# P1 (regression) — savecurrents through silk now PRODUCES a loadable rawfile
# (the control-`write` abort is gone; batch mode writes it).
# ---------------------------------------------------------------------------
def test_P1_savecurrents_produces_loadable_rawfile(tmp_path):
    tb = _hier_tb(tmp_path)
    tb.save_device_currents = True
    tb.simulate(corner='tt')                 # must not raise (was: no rawfile)
    vecs = tb.get_signal_names()
    assert vecs, "batch-mode savecurrents produced an empty/unloadable rawfile"


# ---------------------------------------------------------------------------
# P1b — the new batch-mode assumptions from §4.1.
# ---------------------------------------------------------------------------
def test_P1b_batch_rawfile_has_subckt_currents_and_node_voltages(tmp_path):
    tb = _hier_tb(tmp_path)
    tb.save_device_currents = True
    tb.simulate(corner='tt')
    vecs = tb.get_signal_names()

    current_vecs = [v for v in vecs if '@' in v]
    node_vecs = [v for v in vecs if '@' not in v]
    print(f"\n[P1b] {len(vecs)} vectors; {len(current_vecs)} current, "
          f"{len(node_vecs)} node/source")
    for v in current_vecs[:20]:
        print(f"      {v}")

    # (b) subckt-internal MOS drain current, hierarchically pathed + i(...) wrap
    subckt_mos = [v for v in current_vecs
                  if 'i(@m' in v.lower() and '.' in v and '[id]' in v.lower()]
    assert subckt_mos, (
        "(b) FAILED: no i(@m.x….…[id]) subckt-internal current vector; "
        f"current vectors: {current_vecs}")

    # (a) subckt-internal node voltages present → deck-level `.save all` honored
    #     in batch mode (sky130 wrapper exposes hierarchical internal nodes).
    subckt_nodes = [v for v in node_vecs if 'xdut' in v.lower()]
    assert subckt_nodes, (
        "(a) FAILED: no subckt-internal node voltage — `.save all` not honored "
        f"in batch mode. Node vectors: {node_vecs[:20]}")


def test_P1b_both_naming_dialects_load(tmp_path):
    """(c) The batch rawfile's `v(node)` / `i(@…)` names load end-to-end through
    the existing load_signal path (the naming crux from the P1 STOP)."""
    tb = _hier_tb(tmp_path)
    tb.save_device_currents = True
    tb.simulate(corner='tt')

    # node voltage: batch names it `v(vout)`; requesting bare 'vout' must resolve
    vsig = tb.load_signal('vout')
    assert vsig is not None and len(vsig.y) > 0

    # current: wrapped `i(@m…[id])`; requesting the inner name with type I resolves
    cur = next(v for v in tb.get_signal_names()
               if v.startswith('i(@m') and '[id]' in v)
    inner = cur[len('i('):-1]                 # strip the i( … ) wrapper
    isig = tb.load_signal(inner, signalType='I')
    assert isig is not None and len(isig.y) > 0


# ---------------------------------------------------------------------------
# P2 (§9) — signs + KCL end-to-end through annotate_currents against REAL
# ngspice. Known DC divider Vdd=1.0 -> R1=1k -> vmid -> R2=1k -> 0 (I=0.5 mA);
# plus an independent 1 mA source into r3. Locks every §6.3 row's into-net sign
# and the KCL self-check. This is the authority on signs (design premise).
# ---------------------------------------------------------------------------
def _pin(net, instance, port):
    return next(b for b in net.pins if b.instance == instance and b.port == port)


def _divider_tb(tmp_path):
    pdk = _pdk()
    tb = Module('PDK_skywater130', str(tmp_path), 'p2_tb.cir',
                pdk_details=pdk, module_name=None)
    with tb:
        elements.resistor('r1', p='vdd', n='vmid', value=1e3)
        elements.resistor('r2', p='vmid', n='0', value=1e3)
        sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.0)
        elements.resistor('r3', p='ni', n='0', value=1e3)
        sources.CurrentSource('iforce', positive='0', negative='ni', dc=1e-3)
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=20e-9))
    return tb


def test_P2_divider_signs_and_kcl(tmp_path):
    tb = _divider_tb(tmp_path)
    tb.save_device_currents = True
    tb.simulate(corner='tt')
    rpt = tb.annotate_currents()

    vmid = tb.net_dict['vmid']
    # R1 feeds vmid (neg pin): +0.5 mA;  R2 drains vmid (pos pin): -0.5 mA
    assert _pin(vmid, 'r1', 'n').avg_current == pytest.approx(0.5e-3, rel=1e-3)
    assert _pin(vmid, 'r2', 'p').avg_current == pytest.approx(-0.5e-3, rel=1e-3)
    # V-source positive pin feeds vdd: +0.5 mA into the net (i(v) is negative)
    assert _pin(tb.net_dict['vdd'], 'vdd', 'positive').avg_current == \
        pytest.approx(0.5e-3, rel=1e-3)
    # I-source negative pin injects into 'ni': +1 mA
    assert _pin(tb.net_dict['ni'], 'iforce', 'negative').avg_current == \
        pytest.approx(1e-3, rel=1e-2)

    # KCL self-check ≈ 0 at every fully-annotated interior net
    for name in ('vmid', 'vdd', 'ni'):
        assert abs(rpt.worst_kcl_residual[name]) < 1e-9

    # 2-pin net aggregate is exact: vmid carries 0.5 mA
    assert vmid.avg_current == pytest.approx(0.5e-3, rel=1e-3)


# ---------------------------------------------------------------------------
# P2-MOS (§9 / finding B) — MOS drain sign + KCL on an all-drain internal net,
# against REAL ngspice. CMOS stack: vdd -> MP(pmos) -> mid -> MN(nmos) -> 0,
# both fully on, same series current. 'mid' is internal (tb has no ports) with
# two DRAIN pins, so it is fully annotated and KCL-checked — the one place the
# drain-only limitation (docs/backlog/2026-07-03-...) still lets MOS self-check.
# ---------------------------------------------------------------------------
def _cmos_stack_tb(tmp_path):
    pdk = _pdk()
    tb = Module('PDK_skywater130', str(tmp_path), 'p2mos_tb.cir',
                pdk_details=pdk, module_name=None)
    tb.add_net('mid')
    tb.add(pdk.pmos1p8('MP', d='mid', g='0', s='vdd', b='vdd', w=2.0, l=0.15, m=1))
    tb.add(pdk.nmos1p8('MN', d='mid', g='vdd', s='0', b='0', w=1.0, l=0.15, m=1))
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=20e-9))
    return tb


def test_P2_mos_drain_signs_and_kcl(tmp_path):
    tb = _cmos_stack_tb(tmp_path)
    tb.save_device_currents = True
    tb.simulate(corner='tt')
    rpt = tb.annotate_currents()

    mid = tb.net_dict['mid']
    i_mp = _pin(mid, 'MP', 'd').avg_current   # pmos drain: into-net = +raw
    i_mn = _pin(mid, 'MN', 'd').avg_current   # nmos drain: into-net = -raw

    # pmos SOURCES current into mid (+), nmos SINKS it (-) — opposite signs
    assert i_mp > 0, f'pmos drain should feed mid, got {i_mp}'
    assert i_mn < 0, f'nmos drain should drain mid, got {i_mn}'
    # same series current, equal magnitude
    assert i_mp == pytest.approx(-i_mn, rel=1e-2)
    # KCL self-check closes on the all-drain internal net
    tol = max(1e-9, 1e-3 * (abs(i_mp) + abs(i_mn)))
    assert abs(rpt.worst_kcl_residual['mid']) < tol
