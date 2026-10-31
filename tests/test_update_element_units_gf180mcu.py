"""Catalog/emission validation of TASK-598, without models or a simulator.

GF180MCU lengths are metres and diode areas are square metres; the shipped
author mode is microns (and square microns). Expectations below deliberately
do not use BasePDK's conversion factors. Losing authored-parameter bookkeeping
in update_element must fail both the physical checks and count validation.
This is preview-catalog coverage, not real-model qualification.
"""
from pathlib import Path

import pytest
import yaml

from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.exceptions import CircuitError
from silk.pdk import load_pdk
from silk.utils._utils import _parse_si


_CATALOG = Path(__file__).resolve().parents[1] / 'silk/pdk/data/gf180mcu_device_details.yaml'
_SPECS = yaml.safe_load(_CATALOG.read_text())
_DEVICES = sorted(_SPECS)
_MODES = ('shipped', 'si', 'pdk')


@pytest.fixture(scope='module')
def pdks(tmp_path_factory):
    # An explicitly absent model file proves catalog emission needs no models.
    missing = tmp_path_factory.mktemp('gf180-model-free') / 'absent.lib'
    assert not missing.exists()
    return {mode: load_pdk('gf180mcu', library_location=str(missing),
                          **({} if mode == 'shipped' else {'user_units': mode}))
            for mode in _MODES}


def _build(pdk, tmp_path, name, params=None, *, legacy=False):
    cls = getattr(pdk, name)
    pins = {pin: f'n{i}' for i, pin in enumerate(cls._pin_names)}
    mod = Module('gf180mcu', str(tmp_path), 'unused.cir', pdk_details=pdk)
    if legacy:
        with pytest.warns(DeprecationWarning, match='add_element'):
            mod.add_element(cls, 'dut', nets=pins, parameters=params)
    else:
        mod.add(cls('dut', **pins, **(params or {})))
    return mod


def _line(mod):
    return mod.pdk_details.generate_netlist_entry('dut', mod.elem_dict['dut'])


def _factory_line(mod):
    return mod.elem_dict['dut']['device'].to_spice()


def _values(line):
    return {key: _parse_si(value) for token in line.split() if '=' in token
            for key, value in [token.split('=', 1)]}


def _target(param, mode):
    # Independent physical targets: 4 um width/perimeter, 2 um length,
    # 12 um^2 area, two instances, three fingers. GF180's native unit is SI.
    if param in ('m', 'nf'):
        return (2 if param == 'm' else 3), (2 if param == 'm' else 3)
    if param == 'area':
        return (12.0 if mode == 'shipped' else 12e-12), 12e-12
    if param in ('l', 'r_length', 'c_length'):
        return (2.0 if mode == 'shipped' else 2e-6), 2e-6
    assert param in ('w', 'r_width', 'c_width', 'pj')
    return (4.0 if mode == 'shipped' else 4e-6), 4e-6


def test_catalog_coverage_is_complete(pdks):
    """No family or newly declared parameter may silently leave the matrix."""
    families = {
        'mos': [n for n in _DEVICES if n.startswith(('nmos', 'pmos'))],
        'resistor': [n for n in _DEVICES if n.startswith('res_')],
        'capacitor': [n for n in _DEVICES if n.startswith('cap_')],
        'diode': [n for n in _DEVICES if n.startswith('diode_')],
        'bipolar': [n for n in _DEVICES if n.startswith(('npn_', 'pnp_'))],
    }
    assert {k: len(v) for k, v in families.items()} == {
        'mos': 9, 'resistor': 22, 'capacitor': 11, 'diode': 9, 'bipolar': 10}
    assert sorted(n for group in families.values() for n in group) == _DEVICES
    assert set(pdks['shipped']._device_classes) == set(_DEVICES)
    assert pdks['shipped'].user_units == 'u'
    assert {p for spec in _SPECS.values() for p in spec['paramList']} == {
        'm', 'nf', 'w', 'l', 'r_width', 'r_length', 'c_width', 'c_length',
        'area', 'pj'}
    for spec in _SPECS.values():
        for param, data in spec['paramList'].items():
            if param in ('m', 'nf'):
                assert 'pdk_unit' not in data
                assert data['minval'] == 1 and data['resolution'] == 1
            else:
                assert data['pdk_unit'] == 'base'
                assert data.get('quantity', 'length') == (
                    'area' if param == 'area' else 'length')
                # No geometry bounds are declared: negative-geometry rejection
                # cannot be claimed from this catalog. Counts supply the
                # real below-minimum and integer checks below.
                assert data['minval'] == data['maxval'] == -1


@pytest.mark.parametrize('name', _DEVICES)
@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('supplied', (False, True), ids=('defaulted', 'supplied'))
def test_updates_match_physical_units_constructor_and_assignment(
        pdks, tmp_path, name, mode, supplied):
    pdk = pdks[mode]
    params = _SPECS[name]['paramList']
    initial = {p: (1 if p in ('m', 'nf') else _target(p, mode)[0] / 2)
               for p in params} if supplied else {}
    updated = _build(pdk, tmp_path, name, initial)
    expected = _values(_factory_line(updated))
    # Defaults are already native metres / m^2, including YAML string defaults.
    if not supplied:
        assert expected == pytest.approx({
            p: float(data.get('default', data['minval']))
            for p, data in params.items()}, rel=1e-12, abs=0)
    for param in params:
        value, physical = _target(param, mode)
        built = _build(pdk, tmp_path, name, {**initial, param: value})
        assigned = _build(pdk, tmp_path, name, initial)
        setattr(assigned.elem_dict['dut']['device'], param, value)
        # Update each field separately so all unrelated defaults are observed.
        for _ in range(2):
            updated.update_element('dut', [param], [value])
            expected[param] = physical
            assert _values(_factory_line(updated)) == pytest.approx(
                expected, rel=1e-12, abs=0)
        # Compare against a fresh module to keep prior updates out of the
        # constructor/assignment comparison.
        one = _build(pdk, tmp_path, name, initial)
        one.update_element('dut', [param], [value])
        assert _factory_line(one) == _factory_line(built) == _factory_line(assigned)


@pytest.mark.parametrize('name', _DEVICES)
@pytest.mark.parametrize('mode', _MODES)
def test_live_variables_keep_identity_and_physics(pdks, tmp_path, name, mode):
    mod = _build(pdks[mode], tmp_path, name)
    dev = mod.elem_dict['dut']['device']
    for param in _SPECS[name]['paramList']:
        value, physical = _target(param, mode)
        var = Variable(f'live_{param}', value=value)
        mod.update_element('dut', [param], [var])
        assert dev.params[param] is var
        assert _values(_factory_line(mod))[param] == pytest.approx(physical, rel=1e-12, abs=0)
        var.value = value * 2
        assert dev.params[param] is var
        assert _values(_factory_line(mod))[param] == pytest.approx(physical * 2, rel=1e-12, abs=0)


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('name,param', [
    (n, p) for n in _DEVICES for p in _SPECS[n]['paramList'] if p in ('m', 'nf')])
@pytest.mark.parametrize('bad,reason', [(0, 'below the PDK minimum'),
                                      (1.5, 'integer')])
def test_declared_count_constraints_match_assignment(
        pdks, tmp_path, name, param, mode, bad, reason):
    for path in ('update', 'assignment', 'constructor'):
        mod = _build(pdks[mode], tmp_path, name,
                     {param: bad} if path == 'constructor' else {})
        if path == 'update':
            mod.update_element('dut', [param], [bad])
        elif path == 'assignment':
            setattr(mod.elem_dict['dut']['device'], param, bad)
        with pytest.raises(CircuitError, match=reason) as error:
            _factory_line(mod)
        assert param in str(error.value) and 'dut' in str(error.value)
        mod.update_element('dut', [param], [2])
        assert _values(_factory_line(mod))[param] == 2


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('name,param,literal', [
    ('nmos3p3', 'w', '4u'), ('pmos3p3', 'l', '2u'),
    ('res_nplus_u', 'r_width', '4u'), ('cap_mim_1f0', 'c_length', '2u'),
    ('diode_nd2ps_3p3', 'area', '12p')])
def test_string_literals_stay_verbatim(pdks, tmp_path, mode, name, param, literal):
    mod = _build(pdks[mode], tmp_path, name)
    mod.update_element('dut', [param], [literal])
    assert mod.elem_dict['dut']['device'].params[param] == literal
    assert f'{param}={literal}' in _factory_line(mod).split()
    built = _build(pdks[mode], tmp_path, name, {param: literal})
    assert _factory_line(mod) == _factory_line(built)


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('name', ['nmos3p3', 'pmos6p0_dss', 'res_nplus_u',
                                  'cap_mim_1f0', 'diode_nd2ps_3p3', 'npn_10x10'])
def test_legacy_updates_remain_compatible(pdks, tmp_path, mode, name):
    pdk = pdks[mode]
    mod = _build(pdk, tmp_path, name, legacy=True)
    values = {p: _target(p, mode)[0] for p in _SPECS[name]['paramList']}
    expected = {p: _target(p, mode)[1] for p in values}
    for _ in range(2):
        mod.update_element('dut', list(values), list(values.values()))
        assert _values(_line(mod)) == pytest.approx(expected, rel=1e-12, abs=0)
    assert _line(mod) == _factory_line(_build(pdk, tmp_path, name, values))
    param = next(iter(values))
    var = Variable('legacy_live', value=values[param])
    mod.update_element('dut', [param], [var])
    assert mod.elem_dict['dut']['device'].param_list[param].value is var
    var.value *= 2
    assert _values(_line(mod))[param] == pytest.approx(expected[param] * 2, rel=1e-12, abs=0)
