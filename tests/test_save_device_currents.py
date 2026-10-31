"""Tests for the save_device_currents flag → device-current save directive.

Netlist-string level (no ngspice); the end-to-end rawfile behaviour is the P1
probe. Spec: 2026-07-03-net-current-annotation-design.md §4.
"""
import os

import pytest

from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK
from silk import sources
from silk.simulation import analysis
from silk.simulators import load_simulator_config
from silk.exceptions import SimulationError

_YAML = os.path.join('silk', 'pdk', 'data', 'skywater_130_device_details.yaml')


def _pdk():
    return BasePDK(_YAML, library_location='/fake/sky130.lib.spice')


def _build_tb(tmp_path, simulator='ngspice'):
    pdk = _pdk()
    tb = Module('PDK_skywater130', str(tmp_path), 'tb.cir',
                pdk_details=pdk, module_name=None)
    tb.simulator = simulator
    with tb:
        pdk.nmos1p8('mn', d='vout', g='vin', s='0', b='0', w=1.0, l=0.15)
        sources.VoltageSource('vdd', positive='vout', negative='0', dc=1.8)
        sources.VoltageSource('vin', positive='vin', negative='0', dc=0.9)
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return tb


def _netlist_text(tb):
    tb_deck_path = tb.generate_netlist()
    with open(tb_deck_path) as fh:
        return fh.read()


def test_default_is_false():
    tb = Module('PDK_skywater130', '/tmp', 't.cir',
                pdk_details=_pdk(), module_name='t')
    assert tb.save_device_currents is False


def test_directive_absent_by_default(tmp_path):
    tb = _build_tb(tmp_path)
    assert 'savecurrents' not in _netlist_text(tb)


def test_directive_emitted_when_true(tmp_path):
    tb = _build_tb(tmp_path)
    tb.save_device_currents = True
    assert '.options savecurrents' in _netlist_text(tb)


def test_savecurrents_in_ngspice_known_options():
    cfg = load_simulator_config('ngspice')
    assert 'savecurrents' in cfg.known_options


def test_flag_on_batch_deck_drops_control_block(tmp_path):
    """Flag on (ngspice, single tran) → .options savecurrents + deck-level
    .save all, and NO .control block (batch-write mode; spec §4.1)."""
    tb = _build_tb(tmp_path)
    tb.save_device_currents = True
    text = _netlist_text(tb)
    assert '.options savecurrents' in text
    assert '.save all' in text
    assert '.control' not in text


def test_flag_off_keeps_control_block(tmp_path):
    """Flag off (default) → deck unchanged: keeps the .control block, no
    savecurrents (zero regression surface; spec §4.1)."""
    tb = _build_tb(tmp_path)
    text = _netlist_text(tb)
    assert '.control' in text
    assert 'savecurrents' not in text


def test_flag_on_requires_exactly_one_transient(tmp_path):
    """Guard: batch mode needs exactly one Transient analysis (spec §4.1)."""
    tb = _build_tb(tmp_path)                       # has one Transient
    tb.add_sim_type(analysis.AC('dec', 10, 1.0, 1e6))  # now two analyses
    tb.save_device_currents = True
    with pytest.raises(SimulationError, match='(?i)transient'):
        tb.generate_netlist(corner='tt')


def test_xyce_warns_and_omits_directive(tmp_path):
    """Xyce has no device_current_save_directive → warn, emit nothing, no
    malformed deck (spec §4 / G4)."""
    tb = _build_tb(tmp_path, simulator='xyce')
    tb.save_device_currents = True
    with pytest.warns(UserWarning, match='device_current_save_directive'):
        text = _netlist_text(tb)
    assert 'savecurrents' not in text


def test_run_propagates_flag_to_tb(monkeypatch):
    """ModuleTestbench.run copies the flag from the DUT module to test.netlist
    (the tb) before simulating (spec §4)."""
    pdk = _pdk()
    dut = Module('PDK_skywater130', '/tmp', 'dut.cir',
                 pdk_details=pdk, module_name='dut')
    from silk.circuit.circuit_elements import PortDir
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'gnd': PortDir.GROUND})
    with dut:
        pdk.nmos1p8('mn', d='b', g='a', s='gnd', b='gnd', w=1.0, l=0.15)
    dut.save_device_currents = True

    t = dut.create_test('startup')
    t.add_module_under_test({'a': 'a', 'b': 'b', 'vdd': 'vdd', 'gnd': '0'})

    seen = {}
    from silk.circuit.spec_report import SpecReport
    from silk.circuit.module_test import _RunSimResult

    # run() is now prepare -> simulate -> evaluate (TASK-301.2): the flag
    # must already be on spec.netlist by the time _simulate_run sees it.
    def simulate_run(spec):
        seen['flag'] = spec.netlist.save_device_currents
        return _RunSimResult(netlist=spec.netlist, corner_str=spec.corner_str,
                             conditions=None, invocation=spec.invocation,
                             cache_key=spec.cache_key)

    monkeypatch.setattr(t, '_simulate_run', simulate_run)
    monkeypatch.setattr(t, '_evaluate_run', lambda result: SpecReport('startup', 'dut', {}))
    t.run('tt')
    assert seen['flag'] is True
