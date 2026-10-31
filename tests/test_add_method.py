"""Tests for the new ckt.add() API and direct AnalogElement instantiation."""

import pytest
from silk.elements.base import AnalogElement
from silk.circuit.circuit_elements import PyCircuitElements
from silk.exceptions import DuplicateNameError


# ---------------------------------------------------------------------------
# Direct instantiation via pdk.<device>(...)
# ---------------------------------------------------------------------------

class TestDirectInstantiation:

    def test_pdk_class_instantiates(self, pdk):
        """pdk.nmos1p8('mn1', ...) produces an AnalogElement instance."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        assert isinstance(mn1, AnalogElement)

    def test_instance_name(self, pdk):
        """Instance name is stored correctly."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        assert mn1.name == 'mn1'

    def test_instance_nets(self, pdk):
        """Net connections are stored in instance.nets."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        assert mn1.nets['d'] == 'vout'
        assert mn1.nets['g'] == 'vin'
        assert mn1.nets['s'] == 'gnd'
        assert mn1.nets['b'] == 'gnd'

    def test_instance_params(self, pdk):
        """Parameters are stored in instance.params."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=2.0, l=0.15, m=1)
        assert mn1.params['w'] == 2.0
        assert mn1.params['l'] == 0.15
        assert mn1.params['m'] == 1

    def test_to_spice_contains_nets(self, pdk):
        """to_spice() output contains net names."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        spice = mn1.to_spice()
        assert 'vout' in spice
        assert 'vin' in spice

    def test_to_spice_contains_pdk_model(self, pdk):
        """to_spice() output contains the PDK model name."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        assert 'sky130_fd_pr__nfet_01v8' in mn1.to_spice()

    def test_to_spice_starts_with_prefix(self, pdk):
        """to_spice() line starts with XMmn1."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        assert mn1.to_spice().startswith('XMmn1')

    def test_direct_attr_assignment_updates_nets(self, pdk):
        """Assigning to a pin name routes to nets dict via __setattr__."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        mn1.d = 'new_net'
        assert mn1.nets['d'] == 'new_net'

    def test_direct_attr_assignment_updates_params(self, pdk):
        """Assigning to a param name routes to params dict via __setattr__."""
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        mn1.w = 3.0
        assert mn1.params['w'] == 3.0


# ---------------------------------------------------------------------------
# ckt.add() method
# ---------------------------------------------------------------------------

class TestCktAdd:

    def test_add_single_device(self, pdk, empty_netlist):
        """ckt.add(device) registers the device in elem_dict."""
        ckt = empty_netlist
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        ckt.add(mn1)
        assert 'mn1' in ckt.elem_dict

    def test_add_multiple_devices(self, pdk, empty_netlist):
        """ckt.add(d1, d2, d3) registers all devices."""
        ckt = empty_netlist
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        mp1 = pdk.pmos1p8('mp1', d='vout', g='vin', s='vdd', b='vdd',
                           w=2.0, l=0.15, m=1)
        ckt.add(mn1, mp1)
        assert 'mn1' in ckt.elem_dict
        assert 'mp1' in ckt.elem_dict

    def test_add_stores_analog_element_instance(self, pdk, empty_netlist):
        """ckt.add() stores the AnalogElement instance itself in elem_dict."""
        from silk.elements.base import AnalogElement
        ckt = empty_netlist
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        ckt.add(mn1)
        stored = ckt.elem_dict['mn1']['device']
        assert stored is mn1                          # same object, not a copy
        assert isinstance(stored, AnalogElement)

    def test_add_registers_nets(self, pdk, empty_netlist):
        """ckt.add() auto-registers nets from the device connections."""
        ckt = empty_netlist
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        ckt.add(mn1)
        assert 'vout' in ckt.net_dict
        assert 'vin' in ckt.net_dict

    def test_add_duplicate_name_raises(self, pdk, empty_netlist):
        """ckt.add() raises DuplicateNameError for duplicate instance names."""
        ckt = empty_netlist
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        mn1b = pdk.nmos1p8('mn1', d='a', g='b', s='c', b='d',
                            w=1.0, l=0.15, m=1)
        ckt.add(mn1)
        with pytest.raises(DuplicateNameError):
            ckt.add(mn1b)

    def test_add_strict_nets_raises(self, pdk, tmp_dir):
        """ckt.add() with strict_nets=True raises for undeclared nets."""
        from silk.circuit.circuit_elements import PyCircuitElements
        from silk.exceptions import UndeclaredNetError
        mgr = PyCircuitElements(pdk, strict_nets=True)
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        with pytest.raises(UndeclaredNetError):
            mgr._add_analog_element(mn1)


# ---------------------------------------------------------------------------
# Netlisting of new-style elements
# ---------------------------------------------------------------------------

class TestNewStyleNetlisting:

    def test_generate_device_instances_new_style(self, pdk, empty_netlist):
        """generate_device_instances() uses to_spice() for AnalogElement instances."""
        from silk.circuit.spice_netlister import PySpiceNetlister
        ckt = empty_netlist
        mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='gnd', b='gnd',
                           w=1.0, l=0.15, m=1)
        ckt.add(mn1)
        netlister = PySpiceNetlister(pdk)
        result = netlister.generate_device_instances(ckt.elem_dict)
        assert 'XMmn1' in result
        assert 'sky130_fd_pr__nfet_01v8' in result
        assert 'vout' in result

    def test_mixed_old_new_style(self, pdk, empty_netlist):
        """generate_device_instances() handles both old-style and new-style elements."""
        from silk.circuit.spice_netlister import PySpiceNetlister
        ckt = empty_netlist
        # Old-style
        ckt.add_element(pdk._devices['nmos1p8'], 'mn_old',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1.0, 'l': 0.15, 'm': 1})
        # New-style
        mn2 = pdk.nmos1p8('mn_new', d='out2', g='in2', s='gnd', b='gnd',
                           w=2.0, l=0.15, m=1)
        ckt.add(mn2)
        netlister = PySpiceNetlister(pdk)
        result = netlister.generate_device_instances(ckt.elem_dict)
        assert 'XMmn_old' in result
        assert 'XMmn_new' in result
