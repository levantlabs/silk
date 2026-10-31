"""Editing a registered bench after running it is equivalent to fresh authoring.

The deck matrix needs no simulator. It deliberately uses public APIs shared
by released/main silk and the placement implementation, so it guards the
contract independently of how test execution owns its state.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench
from silk.elements import elements
from silk.simulation import Transient
from tests._env import needs_ngspice


def _bench(folder):
    dut = Module(None, str(folder), 'rc.cir', pdk_details=None, module_name='rc')
    dut.define_module_ports({'in': 'in', 'out': 'out', 'vss': 'ground'})
    dut.add(elements.resistor('r', p='in', n='out', value=1000.0))
    dut.add(elements.capacitor('c', p='out', n='vss', value=1e-9))
    tb = dut.create_test('step')
    tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vss': '0'})
    tb.netlist.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1.0))
    tb.netlist.add_sim_type(Transient(step=1e-9, stop=200e-9))
    tb.add_metric('vout_final', lambda ckt: float(ckt.load_signal('vout', 'V').y[-1]))
    return dut, tb


def _edit(tb, edit):
    bench = tb.netlist
    if edit == 'source value':
        bench.sig_sources_dict['vin'].dc = 2.0
    elif edit == 'add_source':
        bench.add_source(sources.VoltageSource('extra', positive='extra', negative='0', dc=0.3))
    elif edit == 'add element':
        bench.add(elements.resistor('load', p='vout', n='0', value=2000.0))
    elif edit == 'changed analysis':
        bench.add_sim_type(Transient(step=2e-9, stop=400e-9))
    elif edit == 'simulator options':
        bench.set_simulator_options(reltol=3e-4)
    elif edit == 'temperature':
        bench.set_temperature(85.0)
    elif edit == 'save list':
        bench.set_save_list(['vout'])
    else:
        raise AssertionError(edit)


def _only_report(reports):
    assert len(reports) == 1
    report, = reports.values()
    assert not isinstance(report, dict), report
    return report


@pytest.mark.parametrize('edit', [
    'source value', 'add_source', 'add element', 'changed analysis',
    'simulator options', 'temperature', 'save list',
])
@pytest.mark.parametrize('next_run', ['run_all_tests', 'run_test'])
def test_edited_bench_matches_fresh_deck(tmp_path, monkeypatch, edit, next_run):
    decks = []

    def capture_simulation(bench, corner=None, **kwargs):
        deck_path = bench.generate_netlist(corner=corner)
        decks.append(Path(deck_path).read_text())
        return bench

    def capture_evaluation(self, *args, **kwargs):
        return SimpleNamespace(deck=decks[-1], metrics={}, passed=True)

    monkeypatch.setattr(Module, 'simulate', capture_simulation)
    monkeypatch.setattr(ModuleTestbench, 'evaluate', capture_evaluation)
    edited, tb = _bench(tmp_path / 'edited')
    fresh, fresh_tb = _bench(tmp_path / 'fresh')
    edited.set_output_options(mode='overwrite')
    fresh.set_output_options(mode='overwrite')
    try:
        initial = _only_report(edited.run_all_tests()).deck
        _edit(tb, edit)
        if next_run == 'run_all_tests':
            actual = _only_report(edited.run_all_tests()).deck
        else:
            actual = edited.run_test('step').deck
        _edit(fresh_tb, edit)
        expected = fresh.run_test('step').deck
        # Directory spellings are the only permitted emission difference.
        normalize = lambda deck: deck.replace(str(tmp_path / 'edited'), '<output>').replace(
            str(tmp_path / 'fresh'), '<output>')
        assert normalize(actual) != normalize(initial)
        assert normalize(actual) == normalize(expected)
    finally:
        edited.close()
        fresh.close()


@needs_ngspice
def test_source_edit_reaches_next_run_all_tests(tmp_path):
    dut, tb = _bench(tmp_path)
    try:
        first = _only_report(dut.run_all_tests())
        assert first.metrics['vout_final'] == pytest.approx(1.0)
        tb.netlist.sig_sources_dict['vin'].dc = 2.0
        second = _only_report(dut.run_all_tests())
        assert second.metrics['vout_final'] == pytest.approx(2.0)
    finally:
        dut.close()
