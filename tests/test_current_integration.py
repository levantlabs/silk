"""Integration tests for net current annotation (spec 2026-07-03 §9) — real
ngspice + sky130, driven through the example builders.

Covers the hierarchical read path end to end: recursion into a submodule
instance, the per-instance KCL self-check on genuinely-internal nets, the
worst-case D1 reduction onto the definition copy, and Flow B
(``report.sim_results.annotate_currents()``).

Note on interface nets: a subckt's port nets (VDD/VOUT/…) have device pins but
no PortBinding for the boundary current (ports are not pins), so they neither
close KCL nor let the ½·Σ aggregate reach the true supply current — both would
be missing the port term. Annotation marks them interface/partial; these tests
assert on internal nets, where the self-check is meaningful.

Slow: real ngspice + real sky130 models.
"""
import os

import pytest

from silk.pdk.basepdk import BasePDK
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk import sources
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


def _pin(net, instance, port):
    return next(b for b in net.pins if b.instance == instance and b.port == port)


# ---------------------------------------------------------------------------
# 5T OTA — recursion, internal-net KCL, tail split, Flow B (§9)
# ---------------------------------------------------------------------------
def test_ota_5t_currents_flow_b():
    """Hierarchical read path + Flow B on the 5T OTA. With the drain-only MOS
    limitation, source/gate/bulk pins are skipped, so nets touching them go
    partial (tail = source pins; vint = gate pins) and OTA interface nets are
    port-exempt — no OTA net closes KCL here. What we CAN validate end to end:
    recursion reached the internals, the drain currents extract with correct
    OPPOSITE polarity signs on the vout node (M2 nmos vs M4 pmos), and Flow B
    works. The MOS KCL self-check itself is covered by P2-MOS (all-drain net)."""
    from silk.examples.ota_5t_guide import build_5t_ota
    mod = build_5t_ota()
    mod.save_device_currents = True             # propagated to the tb by run()

    report = mod.run_test('tran', corner='tt')  # Flow B
    tb = report.sim_results                      # the tb Module (OTA is 'dut')
    rpt = tb.annotate_currents(tstart=1.5e-6)    # settled window (step at 1 µs)

    # recursion reached the OTA instance internals
    assert 'xdut' in rpt.instance_modules
    assert any(n.startswith('xdut.') for n in rpt.nets_annotated)

    # polarity sign, in a real hierarchical circuit: the two drains on the
    # output node are opposite in sign and ≈ equal in magnitude at balance
    dut = tb.elem_dict['dut']['device']
    vout = dut.net_dict['VOUT']
    i_m2 = _pin(vout, 'M2', 'd').avg_current    # nmos drain → -raw
    i_m4 = _pin(vout, 'M4', 'd').avg_current    # pmos drain → +raw
    assert i_m2 < 0 < i_m4                       # opposite signs (polarity fix)
    assert abs(i_m2) == pytest.approx(abs(i_m4), rel=0.05)

    # drain-only consequence: gate/source pins are skipped → those nets partial
    assert 'xdut.tail' in rpt.partial_nets      # M1.s/M2.s (source) skipped
    assert 'xdut.vint' in rpt.partial_nets      # M3.g/M4.g (gate) skipped
    assert ('xdut.tail', 'M1.s', 'unsupported-kind') in rpt.skips

    # any net that IS KCL-checked (top-level R/source nets) closes tightly
    for name, resid in rpt.worst_kcl_residual.items():
        assert abs(resid) < 1e-6, f'{name} KCL residual {resid} A too large'


def test_ota_5t_tail_current_via_ammeter():
    """The payoff: with the ammeter-wrapper method every MOS terminal is read,
    so the diff-pair TAIL net (M1.s + M2.s + M5.d — all source/drain terminals)
    is fully annotated and closes KCL, and the tail splits ~equally at balance.
    This is exactly what the drain-only savecurrents method could NOT do."""
    from silk.examples.ota_5t_guide import build_5t_ota
    mod = build_5t_ota()
    mod.save_device_currents = True
    mod.terminal_current_method = 'ammeter_wrapper'      # propagates to the tb

    report = mod.run_test('tran', corner='tt')
    tb = report.sim_results
    rpt = tb.annotate_currents(tstart=1.5e-6)

    dut = tb.elem_dict['dut']['device']
    tail = dut.net_dict['tail']
    # all three tail pins are annotated now (not skipped)
    i_m1s = _pin(tail, 'M1', 's').avg_current
    i_m2s = _pin(tail, 'M2', 's').avg_current
    i_m5d = _pin(tail, 'M5', 'd').avg_current
    assert None not in (i_m1s, i_m2s, i_m5d)

    # tail is now a COMPLETE internal net → KCL-checked and closes
    assert 'xdut.tail' in rpt.worst_kcl_residual
    assert 'xdut.tail' not in rpt.partial_nets
    assert abs(rpt.worst_kcl_residual['xdut.tail']) < 1e-7

    # diff pair splits ~equally at balance; M5 sources the whole tail
    assert i_m1s == pytest.approx(i_m2s, rel=0.15)
    assert abs(i_m5d) == pytest.approx(abs(i_m1s) + abs(i_m2s), rel=0.05)

    # tail aggregate is now populated (complete net, ½·Σ bound)
    assert tail.avg_current is not None and tail.avg_current > 0


# ---------------------------------------------------------------------------
# Reused-block hierarchy (§9 flash-ADC properties) — recursion + D1 worst-case
# reduction + per-instance KCL, against REAL ngspice. Two instances of one
# comparator-style subckt (each with an all-drain internal net 'mid' that DOES
# self-check under the drain-only limitation), wired to different rails so the
# two instances carry different currents and D1's max-selection is meaningful.
# Stands in for the 15-comparator flash (heavy SPICE; its real comparator
# internal nets mostly go partial under drain-only) while covering the same
# recursion / D1 / per-instance-KCL paths the spec calls out.
# ---------------------------------------------------------------------------
def _pdk():
    return BasePDK(_PDK_YAML, library_location=_SKY130_LIB)


def _comparator_subckt(pdk, tmp):
    comp = Module('PDK_skywater130', tmp, 'gmcomp.cir',
                  pdk_details=pdk, module_name='gmcomp')
    comp.define_module_ports({'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    comp.add_net('mid')
    comp.add(pdk.pmos1p8('MP', d='mid', g='vss', s='vdd', b='vdd', w=2.0, l=0.15, m=1))
    comp.add(pdk.nmos1p8('MN', d='mid', g='vdd', s='vss', b='vss', w=1.0, l=0.15, m=1))
    return comp


def test_reused_block_recursion_d1_and_per_instance_kcl(tmp_path):
    pdk = _pdk()
    comp = _comparator_subckt(pdk, str(tmp_path))
    tb = Module('PDK_skywater130', str(tmp_path), 'flash_like_tb.cir',
                pdk_details=pdk, module_name=None)
    tb.add_module(comp, 'C0', nets={'vdd': 'vdda', 'vss': '0'})
    tb.add_module(comp, 'C1', nets={'vdd': 'vddb', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vdda', positive='vdda', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vddb', positive='vddb', negative='0', dc=1.2))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=20e-9))
    tb.save_device_currents = True
    tb.simulate(corner='tt')

    rpt = tb.annotate_currents()

    # recursion reached BOTH reused instances
    assert 'xC0' in rpt.instance_modules and 'xC1' in rpt.instance_modules

    # per-instance KCL: each instance's all-drain 'mid' net closes independently
    r0 = rpt.worst_kcl_residual['xC0.mid']
    r1 = rpt.worst_kcl_residual['xC1.mid']
    c0 = rpt.instance_modules['xC0'].net_dict['mid']
    c1 = rpt.instance_modules['xC1'].net_dict['mid']
    i0 = abs(_pin(c0, 'MP', 'd').avg_current)
    i1 = abs(_pin(c1, 'MP', 'd').avg_current)
    assert abs(r0) < max(1e-9, 1e-3 * 2 * i0)
    assert abs(r1) < max(1e-9, 1e-3 * 2 * i1)

    # the two instances carry DIFFERENT currents (1.8 V vs 1.2 V rail)
    assert i0 > i1 > 0

    # D1: the shared definition copy gets the worst-case (larger, C0) instance
    assert 'gmcomp' in rpt.definition_reduced
    defn = tb.module_dict['gmcomp']['device']
    i_def = abs(_pin(defn.net_dict['mid'], 'MP', 'd').avg_current)
    assert i_def == pytest.approx(i0, rel=1e-6)   # worst-of-instances
