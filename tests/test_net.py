"""Net dataclass tests — Phase 1 structural + Phase 2a stored state.

Phase 2a replaced the Phase 1 voltage read-through property with a
stored field, and added domain / classification / is_critical to Net.
These tests cover the new stored-field contract.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from silk.circuit.net import Net, PortBinding
from silk.circuit.circuit_elements import PortRole


def _make_net(name='n1', **overrides):
    """Construct a Net without a real Module (pass None as owner for unit tests)."""
    return Net(name=name, module=None, **overrides)


# ---------------------------------------------------------------------------
# PortBinding
# ---------------------------------------------------------------------------

class TestPortBinding:

    def test_required_fields(self):
        """PortBinding requires net, instance, port; others default to None."""
        b = PortBinding(net='vout', instance='mn1', port='d')
        assert b.net == 'vout'
        assert b.instance == 'mn1'
        assert b.port == 'd'
        assert b.direction is None
        assert b.role is None
        assert b.domain is None
        assert b.kind is None

    def test_full_fields(self):
        """All fields can be populated."""
        b = PortBinding(
            net='vdd',
            instance='comp1',
            port='vdd_in',
            direction='power',
            role='power',
            domain=object(),
            kind='submodule',
        )
        assert b.direction == 'power'
        assert b.role == 'power'
        assert b.domain is not None
        assert b.kind == 'submodule'


# ---------------------------------------------------------------------------
# Net structural
# ---------------------------------------------------------------------------

class TestNet:

    def test_minimum_construction(self):
        """A Net requires name and module; parameters and pins default to empty."""
        fake_module = SimpleNamespace()
        net = Net(name='vout', module=fake_module)
        assert net.name == 'vout'
        assert net.module is fake_module
        assert net.parameters == []
        assert net.pins == []

    def test_parameters_can_be_passed(self):
        """Legacy parameters list is preserved as an attribute."""
        fake_module = SimpleNamespace()
        net = Net(name='vdd', module=fake_module, parameters=['max_voltage=1.8'])
        assert net.parameters == ['max_voltage=1.8']

    def test_parameters_independent_per_net(self):
        """Default parameter lists are not shared across instances."""
        fake_module = SimpleNamespace()
        a = Net(name='a', module=fake_module)
        b = Net(name='b', module=fake_module)
        a.parameters.append('x')
        assert b.parameters == []

    def test_fanout_counts_pins(self):
        """fanout returns len(pins)."""
        fake_module = SimpleNamespace()
        net = Net(name='vout', module=fake_module)
        assert net.fanout == 0
        net.pins.append(PortBinding(net='vout', instance='mn1', port='d'))
        net.pins.append(PortBinding(net='vout', instance='mp1', port='d'))
        assert net.fanout == 2


# ---------------------------------------------------------------------------
# Phase 2a: voltage stored field
# ---------------------------------------------------------------------------

class TestNetVoltageField:
    """Phase 2a: Net.voltage is a stored field, no longer a read-through property."""

    def test_voltage_defaults_to_none(self):
        n = _make_net()
        assert n.voltage is None

    def test_voltage_direct_assignment_roundtrip(self):
        n = _make_net()
        arr = np.array([0.1, 0.2, 0.3])
        n.voltage = arr
        assert n.voltage is arr

    def test_voltage_can_be_cleared_back_to_none(self):
        n = _make_net()
        n.voltage = np.array([1.0, 2.0])
        n.voltage = None
        assert n.voltage is None


class TestNetDomainField:
    """Phase 2a: Net.domain is a stored field, populated by Module.assign_net_domain."""

    def test_domain_defaults_to_none(self):
        n = _make_net()
        assert n.domain is None

    def test_domain_direct_assignment(self):
        n = _make_net()
        # Use a simple sentinel; Module.assign_net_domain is exercised in Task 3.
        class _FakeDomain:
            name = 'AVDD'
        d = _FakeDomain()
        n.domain = d
        assert n.domain is d


class TestNetIsCritical:
    """Phase 2a: Net.is_critical is a plain bool flag for layout annotation."""

    def test_is_critical_defaults_to_false(self):
        n = _make_net()
        assert n.is_critical is False

    def test_is_critical_toggleable(self):
        n = _make_net()
        n.is_critical = True
        assert n.is_critical is True
        n.is_critical = False
        assert n.is_critical is False


class TestNetClassificationProperty:
    """Phase 2a: Net.classification is a property — explicit override wins,
    otherwise inferred from bound pin roles.

    Priority: POWER > GROUND > CLOCK > DIGITAL > ANALOG.
    """

    def test_classification_none_when_no_pins_and_no_override(self):
        n = _make_net()
        assert n.classification is None

    def test_classification_none_when_pin_has_no_role(self):
        n = _make_net()
        n.pins.append(PortBinding(net='n1', instance='inst1', port='p1', role=None))
        assert n.classification is None

    def test_classification_infers_power_from_power_pin(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='vdd', instance='inst1', port='vdd', role=PortRole.POWER,
        ))
        assert n.classification == PortRole.POWER

    def test_classification_infers_ground_from_ground_pin(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='gnd', instance='inst1', port='gnd', role=PortRole.GROUND,
        ))
        assert n.classification == PortRole.GROUND

    def test_classification_infers_clock_from_clock_pin(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='clk', instance='inst1', port='clk', role=PortRole.CLOCK,
        ))
        assert n.classification == PortRole.CLOCK

    def test_classification_infers_digital(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='en', instance='inst1', port='en', role=PortRole.DIGITAL,
        ))
        assert n.classification == PortRole.DIGITAL

    def test_classification_infers_analog(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='vin', instance='inst1', port='vin', role=PortRole.ANALOG,
        ))
        assert n.classification == PortRole.ANALOG

    def test_classification_priority_power_beats_analog(self):
        """POWER wins when a power pin and an analog pin share a net."""
        n = _make_net()
        n.pins.append(PortBinding(
            net='vdd', instance='a', port='p', role=PortRole.ANALOG,
        ))
        n.pins.append(PortBinding(
            net='vdd', instance='b', port='p', role=PortRole.POWER,
        ))
        assert n.classification == PortRole.POWER

    def test_classification_priority_ground_beats_digital(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='gnd', instance='a', port='p', role=PortRole.DIGITAL,
        ))
        n.pins.append(PortBinding(
            net='gnd', instance='b', port='p', role=PortRole.GROUND,
        ))
        assert n.classification == PortRole.GROUND

    def test_classification_priority_clock_beats_digital(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='clk', instance='a', port='p', role=PortRole.DIGITAL,
        ))
        n.pins.append(PortBinding(
            net='clk', instance='b', port='p', role=PortRole.CLOCK,
        ))
        assert n.classification == PortRole.CLOCK

    def test_classification_priority_digital_beats_analog(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='en', instance='a', port='p', role=PortRole.ANALOG,
        ))
        n.pins.append(PortBinding(
            net='en', instance='b', port='p', role=PortRole.DIGITAL,
        ))
        assert n.classification == PortRole.DIGITAL

    def test_classification_explicit_override_wins(self):
        """A user-declared classification wins over inference."""
        n = _make_net()
        n.pins.append(PortBinding(
            net='special', instance='a', port='p', role=PortRole.ANALOG,
        ))
        n.classification = PortRole.CLOCK   # explicit override
        assert n.classification == PortRole.CLOCK

    def test_classification_clearing_override_reverts_to_inference(self):
        n = _make_net()
        n.pins.append(PortBinding(
            net='vdd', instance='a', port='p', role=PortRole.POWER,
        ))
        n.classification = PortRole.CLOCK   # override
        assert n.classification == PortRole.CLOCK
        n.classification = None              # clear
        assert n.classification == PortRole.POWER  # reverts to inference


import os
import tempfile
from silk.circuit.modules import Module
from silk.pdk.basepdk import BasePDK


def _make_module(tmp_path, name='m'):
    pdk_yaml = os.path.join(
        os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
        'skywater_130_device_details.yaml',
    )
    pdk = BasePDK(pdk_yaml, library_location='/fake/lib')
    return Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
                  pdk_details=pdk, module_name=name)


def test_add_net_accepts_classification_kwarg(tmp_path):
    m = _make_module(tmp_path)
    m.add_net('vdd', classification=PortRole.POWER)
    assert m.net_dict['vdd'].classification == PortRole.POWER


def test_add_net_default_classification_is_none(tmp_path):
    m = _make_module(tmp_path)
    m.add_net('some_internal_net')
    assert m.net_dict['some_internal_net'].classification is None


from silk.circuit.voltage_domain import VoltageDomain


def test_assign_net_domain_writes_to_net_domain_field(tmp_path):
    m = _make_module(tmp_path)
    m.add_net('vdd')
    avdd = VoltageDomain('AVDD', nominal=1.8, tolerance=0.1)
    m.assign_net_domain('vdd', avdd)
    assert m.net_dict['vdd'].domain is avdd


def test_assign_net_domain_tracks_module_domains(tmp_path):
    """Module._domains (the list of all domains in this module) still
    accumulates, even though _net_domains (the per-net dict) is gone."""
    m = _make_module(tmp_path)
    m.add_net('vdd')
    m.add_net('gnd')
    avdd = VoltageDomain('AVDD', nominal=1.8, tolerance=0.1)
    m.assign_net_domain('vdd', avdd)
    m.assign_net_domain('gnd', avdd)
    assert avdd in m._domains


def test_assign_net_domain_incompatible_raises(tmp_path):
    m = _make_module(tmp_path)
    m.add_net('vdd')
    avdd = VoltageDomain('AVDD', nominal=1.8, tolerance=0.1)
    dvdd = VoltageDomain('DVDD', nominal=3.3, tolerance=0.1)
    m.assign_net_domain('vdd', avdd)
    with pytest.raises(ValueError, match='incompatible'):
        m.assign_net_domain('vdd', dvdd)


def test_assign_net_domain_compatible_reassignment_noop(tmp_path):
    """Assigning the same compatible domain twice is a no-op, not an error."""
    m = _make_module(tmp_path)
    m.add_net('vdd')
    avdd = VoltageDomain('AVDD', nominal=1.8, tolerance=0.1)
    m.assign_net_domain('vdd', avdd)
    m.assign_net_domain('vdd', avdd)   # should not raise
    assert m.net_dict['vdd'].domain is avdd
