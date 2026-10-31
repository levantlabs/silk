# tests/test_junction_check.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from silk.circuit.modules import Module
from silk.circuit.junction_check import check_junction_overrange, JunctionViolation
from silk.lint.severity import Severity


def _pdk_with_ratings():
    pdk = MagicMock()
    pdk.model_names = ['nmos1p8']
    device = MagicMock()
    device.junction_ratings = {'Vgs_max': 1.98, 'Vds_max': 1.98, 'Vgd_max': 1.98}
    device.pin_list = {
        'd': MagicMock(net='vout'),
        'g': MagicMock(net='vin'),
        's': MagicMock(net='gnd'),
        'b': MagicMock(net='gnd'),
    }
    pdk._devices = {'nmos1p8': device}
    return pdk


def _make_module(tmp_path, pdk, t, results):
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    m._behavioral_mode = True
    m._behavioral_t = t
    # Phase 2a: populate Net.voltage directly — the dict is gone.
    for net_name, waveform in results.items():
        m.add_net(net_name)
        m.net_dict[net_name].voltage = waveform
    return m


def test_no_violations_within_ratings(tmp_path):
    pdk = _pdk_with_ratings()
    t = np.linspace(0, 1e-9, 100)
    results = {
        'vin':  np.full(100, 1.5),
        'vout': np.full(100, 1.8),
        'gnd':  np.zeros(100),
    }
    m = _make_module(tmp_path, pdk, t, results)
    m._element_mgr.elem_dict['mn1'] = {
        'type': 'device',
        'typeName': 'nmos1p8',
        'device': pdk._devices['nmos1p8'],
    }
    violations = check_junction_overrange(m)
    assert violations == []


def test_vgs_overrange_detected(tmp_path):
    pdk = _pdk_with_ratings()
    t = np.linspace(0, 1e-9, 100)
    # Vgs = vin - gnd = 2.5V >> Vgs_max=1.98V
    results = {
        'vin':  np.full(100, 2.5),
        'vout': np.full(100, 1.8),
        'gnd':  np.zeros(100),
    }
    m = _make_module(tmp_path, pdk, t, results)
    m._element_mgr.elem_dict['mn1'] = {
        'type': 'device',
        'typeName': 'nmos1p8',
        'device': pdk._devices['nmos1p8'],
    }
    violations = check_junction_overrange(m)
    assert len(violations) >= 1
    v = violations[0]
    assert v.rule_id == 'STR-001'
    assert v.severity == Severity.ERROR
    assert v.instance == 'mn1'
    assert v.location.terminal == 'Vgs'
    assert v.measured_v > 1.98
    assert v.rated_max == 1.98


def test_no_violations_pre_simulation(tmp_path):
    pdk = _pdk_with_ratings()
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    # _behavioral_mode is False (default)
    violations = check_junction_overrange(m)
    assert violations == []


def test_device_without_ratings_skipped(tmp_path):
    pdk = MagicMock()
    device = MagicMock()
    device.junction_ratings = {}   # empty — no ratings defined
    device.pin_list = {'p': MagicMock(net='vp'), 'n': MagicMock(net='vn')}
    pdk._devices = {'resistor': device}
    t = np.linspace(0, 1e-9, 10)
    results = {'vp': np.full(10, 1.8), 'vn': np.zeros(10)}
    m = _make_module(tmp_path, pdk, t, results)
    m._element_mgr.elem_dict['r1'] = {'type': 'device', 'typeName': 'resistor', 'device': device}
    violations = check_junction_overrange(m)
    assert violations == []


def test_coverage_failure_and_partial_are_independent(tmp_path):
    from silk.circuit import junction_check
    pdk = _pdk_with_ratings()
    m = _make_module(tmp_path, pdk, np.array([0., 1e-9]), {
        'vin': np.array([2.5, 2.5]), 'vout': np.array([1.8, 1.8]),
    })
    m._element_mgr.elem_dict['mn1'] = {
        'type': 'device', 'typeName': 'nmos1p8', 'device': pdk._devices['nmos1p8'],
    }
    before = check_junction_overrange(m)
    assert hasattr(junction_check, 'junction_stress_coverage')
    coverage = junction_check.junction_stress_coverage(m)
    assert coverage.summary() == {'total': 1, 'full': 0, 'partial': 1, 'none': 0, 'failed': 1}
    device, = coverage.devices
    assert device.instance_path == 'mn1'
    assert device.type_name == 'nmos1p8'
    assert device.failed and device.coverage == 'partial'
    assert device.reasons == ('no_ratings',)
    assert device.pairs == {'Vgs': 'fail', 'Vds': 'pass', 'Vgd': 'pass',
                            'Vbs': 'unrated', 'Vbd': 'unrated'}
    assert check_junction_overrange(m) == before
