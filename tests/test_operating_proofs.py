"""Public corner and supply proofs (TASK-404, design 2026-10-01 section 3)."""
import pytest

from silk import Module, load_pdk, sources
from silk.circuit.ports import PortRole
from silk.elements import elements
from silk.simulation import has_no_pdk_device, has_no_supply


def _rc(tmp_path, *, src=None, name='rc'):
    mod = Module('sky130', str(tmp_path), f'{name}.cir', load_pdk('sky130'))
    mod.add(elements.resistor('r1', p='in', n='out', value=1e3))
    mod.add(elements.capacitor('c1', p='out', n='0', value=1e-12))
    mod.add_source(src if src is not None else sources.VoltageSource(
        'vs', positive='in', negative='0', dc=0.0,
        pulse=dict(v1=0, v2=1, td=0, tr=1e-9, tf=1e-9, pw=1e-6, per=2e-6)))
    return mod


def _inverter(tmp_path):
    pdk = load_pdk('sky130')
    mod = Module('sky130', str(tmp_path), 'inv.cir', pdk)
    mod.add(pdk.nmos1p8('mn', d='out', g='in', s='0', b='0', w=1.0, l=0.35))
    mod.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    return mod


def test_measured_rc_proves_both_none(tmp_path):
    mod = _rc(tmp_path)
    assert has_no_pdk_device(mod) is True
    assert has_no_supply(mod) is True


def test_sky130_inverter_proves_neither(tmp_path):
    mod = _inverter(tmp_path)
    assert has_no_pdk_device(mod) is False
    assert has_no_supply(mod) is False


def test_custom_line_defeats_corner_proof_and_so_supply(tmp_path):
    mod = _rc(tmp_path)
    mod.add_custom_line('.param x=1')
    assert has_no_pdk_device(mod) is False
    assert has_no_supply(mod) is False


def test_configured_supply_fails(tmp_path):
    mod = _rc(tmp_path, src=sources.VoltageSource('vs', positive='in', negative='0', dc=1.0))
    assert has_no_supply(mod) is False
    mod = _rc(tmp_path)
    mod._supply_voltage = 1.2
    assert has_no_pdk_device(mod) is True
    assert has_no_supply(mod) is False


def test_power_net_fails(tmp_path):
    mod = _rc(tmp_path)
    mod.add_net('rail', classification=PortRole.POWER)
    assert has_no_pdk_device(mod) is True
    assert has_no_supply(mod) is False


@pytest.mark.parametrize('dc,expected', [(0, True), (0.0, True), ('0', True),
                                         (1.0, False), (-0.5, False), ('1m', False)])
def test_dc_level_literal_zero_passes(tmp_path, dc, expected):
    mod = _rc(tmp_path, src=sources.VoltageSource('vs', positive='in', negative='0', dc=dc))
    assert has_no_supply(mod) is expected


def test_parameter_reference_dc_fails(tmp_path):
    src = sources.VoltageSource('vs', positive='in', negative='0',
                                dc=sources.VoltageSource.dc.variable(name='vs', value=0.0))
    mod = _rc(tmp_path, src=src)
    assert has_no_supply(mod) is False


def test_waveform_source_passes_even_with_dc_level(tmp_path):
    src = sources.VoltageSource('vs', positive='in', negative='0', dc=0.9,
                                sine=dict(offset=0, amplitude=1, freq=1e3))
    assert has_no_supply(_rc(tmp_path, src=src)) is True


def test_current_source_dc_fails_and_zero_passes(tmp_path):
    assert has_no_supply(_rc(tmp_path, src=sources.CurrentSource(
        'i1', positive='in', negative='0', dc=1e-6))) is False
    assert has_no_supply(_rc(tmp_path, src=sources.CurrentSource(
        'i1', positive='in', negative='0', dc=0))) is True


def test_controlled_source_is_ignored(tmp_path):
    mod = _rc(tmp_path)
    mod.add_source(sources.VoltageControlledVoltageSource(
        'e1', out_pos='o2', out_neg='0', ctrl_pos='out', ctrl_neg='0', gain=2.0))
    assert has_no_supply(mod) is True


@pytest.mark.parametrize('dc,expected', [(1.0, False), (0, True)])
def test_hierarchy_is_walked(tmp_path, dc, expected):
    child = Module('sky130', str(tmp_path), 'child.cir', load_pdk('sky130'),
                   module_name='child')
    child.define_module_ports({'a': 'in', 'b': 'out'})
    child.add(elements.resistor('rc', p='a', n='b', value=1e3))
    child.add_source(sources.VoltageSource('vc', positive='a', negative='0', dc=dc))
    top = _rc(tmp_path)
    top.add_module(child, 'x1', nets={'a': 'in', 'b': 'out'})
    assert has_no_pdk_device(top) is True
    assert has_no_supply(top) is expected
