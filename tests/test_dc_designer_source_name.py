"""TASK-345: DC(source=) accepts the designer's source name, not only 'Vvg'."""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.simulation import DC, Noise
from silk.simulation.runner import ngspice_executable
from silk.simulation.xyce_runner import _xyce_executable


def _bench(root, source, simulator='ngspice'):
    mod = Module('sky130', str(root), 'div.cir', load_pdk('sky130'),
                 module_name='divider')
    mod.simulator = simulator
    mod.define_module_ports({'vin': 'input', 'vout': 'output'})
    mod.add(elements.resistor('r1', p='vin', n='vout', value=1e3))
    mod.add(elements.resistor('r2', p='vout', n='0', value=1e3))
    tb = mod.create_test('div')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add(sources.VoltageSource('vg', positive='vin', negative='0', dc=0))
    tb.add_sim_type(DC(source, 0, 1, 0.1))
    return tb


def _deck(tb, simulator='ngspice'):
    tb.netlist.simulator = simulator
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    return Path(tb_netlist_deck_path).read_text()


@pytest.mark.parametrize('spelling', ['vg', 'VG'])
def test_designer_spellings_render_the_emitted_name(spelling, tmp_path):
    assert '.dc Vvg 0 1 0.1' in _deck(_bench(tmp_path, spelling))


@pytest.mark.parametrize('spelling', ['Vvg', 'vvg'])
def test_emitted_spelling_passes_through_unchanged(spelling, tmp_path):
    assert f'.dc {spelling} 0 1 0.1' in _deck(_bench(tmp_path, spelling))


def test_temp_is_left_alone(tmp_path):
    assert '.dc temp 0 1 0.1' in _deck(_bench(tmp_path, 'temp'))


def test_unknown_source_is_refused_before_the_run_listing_sources(tmp_path):
    tb = _bench(tmp_path, 'nope')
    with pytest.raises(ValueError, match=r"'nope' is not a source of this test.*'vg'"):
        tb.run()
    assert not Path(os.path.join(tb.netlist.dir_folder, tb.netlist.netlistname)).exists()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which(ngspice_executable()) is None,
                    reason='ngspice not installed')
def test_ngspice_sweeps_by_designer_name(tmp_path):
    tb = _bench(tmp_path, 'vg')
    tb.run()
    sig = tb.netlist.load_signal('vout')
    np.testing.assert_allclose(sig.y, sig.x / 2.0, atol=1e-6)
    assert len(sig.x) == 11


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(shutil.which(_xyce_executable()) is None,
                    reason='Xyce unavailable')
def test_xyce_sweeps_by_designer_name(tmp_path):
    tb = _bench(tmp_path, 'vg', simulator='xyce')
    tb.run()
    sig = tb.netlist.load_signal('vout')
    np.testing.assert_allclose(sig.y, sig.x / 2.0, atol=1e-6)
    assert len(sig.x) == 11


def _colliding_sources():
    # 'vg' is source 'vg' (emitted Vvg) AND source 'g' (emitted Vg).
    return {
        'g': sources.VoltageSource('g', positive='a', negative='0', dc=0, ac=1),
        'vg': sources.VoltageSource('vg', positive='b', negative='0', dc=0),
    }


def _net():
    from silk.circuit.spice_netlister import PySpiceNetlister
    return PySpiceNetlister.__new__(PySpiceNetlister)


def test_dc_collision_resolves_to_the_emitted_source_and_warns(tmp_path):
    from silk.simulation.conditions import prepare_supply
    tb = _bench(tmp_path, 'vg')
    tb.netlist.sig_sources_dict = _colliding_sources()
    tb.set_supply_voltage(0, source='g')
    with pytest.warns(UserWarning, match=r"DC.*ambiguous.*emitted name of source "
                      r"'g'.*designer name of source 'vg'.*Resolved to source 'g'.*'Vvg'") as notices:
        request = prepare_supply(tb.netlist)
        out = _net().generate_analysis_commands(
            {'a': DC('vg', 0, 1, 0.1)}, sig_sources_dict=_colliding_sources())
    assert len(notices) == 1
    assert request.sweep.stop == 1 and request.source == 'g'
    assert '.dc vg 0 1 0.1' in out          # emitted spelling passed through
    # Following the hint ('Vvg', the other source's emitted name) selects it.
    out = _net().generate_analysis_commands(
        {'a': DC('Vvg', 0, 1, 0.1)}, sig_sources_dict=_colliding_sources())
    assert '.dc Vvg 0 1 0.1' in out


def test_noise_collision_keeps_the_exact_designer_source_and_warns():
    # Noise's historic precedence: an exact designer name wins, so 'vg'
    # is source 'vg' (Vvg), not source 'g' (emitted Vg).
    with pytest.warns(UserWarning, match=r"Noise.*ambiguous.*emitted name of "
                      r"source 'g'.*designer name of source 'vg'.*Resolved to "
                      r"source 'vg'.*designer name 'g'"):
        out = _net().generate_analysis_commands(
            {'a': Noise('vout', '0', 'vg', 'dec', 10, 1, 1e9)},
            sig_sources_dict=_colliding_sources())
    assert '.noise v(vout,0) Vvg dec' in out
    # Following the hint ('g', its designer name) selects the other source.
    out = _net().generate_analysis_commands(
        {'a': Noise('vout', '0', 'g', 'dec', 10, 1, 1e9)},
        sig_sources_dict=_colliding_sources())
    assert '.noise v(vout,0) Vg dec' in out


def test_noise_emitted_spelling_passes_through_without_warning():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = _net().generate_analysis_commands(
            {'a': Noise('vout', '0', 'Vg', 'dec', 10, 1, 1e9)},
            sig_sources_dict=_colliding_sources())
    assert '.noise v(vout,0) Vg dec' in out


def test_collision_other_source_is_picked_by_its_emitted_spelling():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = _net().generate_analysis_commands(
            {'a': DC('Vvg', 0, 1, 0.1)}, sig_sources_dict=_colliding_sources())
    assert '.dc Vvg 0 1 0.1' in out


def test_no_warning_without_a_collision():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = _net().generate_analysis_commands(
            {'a': DC('g', 0, 1, 0.1)}, sig_sources_dict=_colliding_sources())
    assert '.dc Vg 0 1 0.1' in out
