"""
Tests for PyCircuitElements — the shared mixin managing elem_dict, net_dict, and module_dict.

Covers add_element(), add_net(), connect_module(), update_element(), deep copy isolation,
and error paths (duplicate names, invalid pins, nonexistent elements).
"""

import pytest
import copy

from silk.exceptions import CircuitError


# ---------------------------------------------------------------------------
# add_element basics
# ---------------------------------------------------------------------------

class TestAddElement:

    def test_add_device_element_populates_elem_dict(self, pdk, empty_netlist):
        """Adding a device element stores it in elem_dict with correct type."""
        ckt = empty_netlist
        ckt.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15},
        )
        assert 'mn1' in ckt.elem_dict
        assert ckt.elem_dict['mn1']['type'] == 'd'

    def test_add_device_element_stores_device_name(self, pdk, empty_netlist):
        """The stored element records the device type name from the PDK."""
        ckt = empty_netlist
        ckt.add_element(pdk.pmos1p8, 'mp1',
                       nets={'d': 'out', 'g': 'in', 's': 'vdd', 'b': 'vdd'},
                       parameters={'w': 2, 'l': 0.15})
        assert ckt.elem_dict['mp1']['typeName'] == pdk.pmos1p8.device_name

    def test_add_multiple_elements(self, basic_netlist):
        """basic_netlist fixture adds two elements; both should be present."""
        assert 'mn1' in basic_netlist.elem_dict
        assert 'mp1' in basic_netlist.elem_dict
        assert len(basic_netlist.elem_dict) == 2

    def test_duplicate_element_name_exits(self, pdk, empty_netlist):
        """Adding an element with a duplicate name raises CircuitError."""
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15})
        with pytest.raises(CircuitError):
            ckt.add_element(pdk.nmos1p8, 'mn1',
                           nets={'d': 'out2', 'g': 'in2', 's': 'gnd', 'b': 'gnd'},
                           parameters={'w': 1, 'l': 0.15})

    def test_unknown_device_type_exits(self, pdk, empty_netlist):
        """Adding an element with an unrecognized device type raises CircuitError."""
        class FakeDevice:
            device_name = 'nonexistent_device_xyz'
        with pytest.raises(CircuitError):
            empty_netlist.add_element(FakeDevice(), 'bad1',
                                     nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'})

    def test_add_module_element(self, pdk, empty_netlist, basic_module):
        """Adding a module element stores it with type 'm' and populates module_dict."""
        ckt = empty_netlist
        ckt.add_element(
            basic_module, 'xinv1',
            nets={'in': 'sig_in', 'out': 'sig_out', 'vdd': 'vdd', 'vss': 'gnd'},
        )
        assert 'xinv1' in ckt.elem_dict
        assert ckt.elem_dict['xinv1']['type'] == 'm'
        assert basic_module.module_name in ckt.module_dict

    def test_add_module_increments_instantiation_count(self, pdk, empty_netlist, basic_module):
        """Instantiating the same module twice increments numberInstantiated."""
        ckt = empty_netlist
        ckt.add_element(basic_module, 'xinv1',
                       nets={'in': 'a', 'out': 'b', 'vdd': 'vdd', 'vss': 'gnd'})
        ckt.add_element(basic_module, 'xinv2',
                       nets={'in': 'c', 'out': 'd', 'vdd': 'vdd', 'vss': 'gnd'})
        assert ckt.module_dict[basic_module.module_name]['numberInstantiated'] == 2

    def test_module_target_rejected_by_raw_circuitgraph_leaves_no_orphan(
            self, pdk, basic_module):
        """LOUD, dead branch deleted rather than fixed (TASK-108). Only
        Module.add_element (modules_structure.py) is reachable from any
        current caller with a module-typed elemDevice — every Module sets
        device_name = 'module', and Module.add_element intercepts a Module
        target before ever delegating to CircuitGraph.add_element (grepped:
        no caller in silk/ or tests/ does otherwise). CircuitGraph.add_element
        used to have its own copy of the same MODULE branch anyway, calling
        register_module_definition with no rollback if the binding step
        rejected the call — the defect TASK-98 fixed on the live path, but
        never exercised here. Pre-fix (this exact call, same rejection):
        raised ValueError from _validate_connections on the empty 'vss' pin,
        but only AFTER orphaning module_dict['inv_cell'] (numberInstantiated
        1, present) and stamping basic_module._instantiation_count to 1,
        even though 'xinv1' never made it into elem_dict. Deleting the
        branch (rather than duplicating TASK-98's rollback for code nothing
        calls) makes a raw CircuitGraph reject any module-typed elemDevice
        outright, before register_module_definition ever runs, so nothing
        is left to roll back."""
        from silk.circuit.circuit_elements import PyCircuitElements
        mgr = PyCircuitElements(pdk)
        with pytest.raises(CircuitError, match='not found in PDK'):
            mgr.add_element(
                basic_module, 'xinv1',
                nets={'in': 'sig_in', 'out': 'sig_out', 'vdd': 'vdd', 'vss': ''},
            )
        assert mgr.module_dict == {}
        assert mgr.elem_dict == {}
        assert getattr(basic_module, '_instantiation_count', 0) == 0


# ---------------------------------------------------------------------------
# add_net
# ---------------------------------------------------------------------------

class TestAddNet:

    def test_add_net_creates_entry(self, empty_netlist):
        """add_net creates a new key in net_dict."""
        empty_netlist.add_net('vdd')
        assert 'vdd' in empty_netlist.net_dict

    def test_add_duplicate_net_is_noop(self, empty_netlist):
        """Adding the same net twice does not raise and keeps original parameters."""
        empty_netlist.add_net('vdd', [1.8])
        empty_netlist.add_net('vdd', [3.3])
        # Original parameters should be preserved on Net.parameters
        assert empty_netlist.net_dict['vdd'].parameters == [1.8]

    def test_net_dict_populated_by_basic_netlist(self, basic_netlist):
        """basic_netlist fixture explicitly adds vdd, gnd, in, out nets."""
        for net in ['vdd', 'gnd', 'in', 'out']:
            assert net in basic_netlist.net_dict

    def test_addnet_default_parameters_not_shared(self, empty_netlist):
        """Each net added without explicit parameters gets its own independent list.

        Regression test for mutable default argument bug: if parameters=[] is used
        as a default, all calls without parameters would share the same list object.
        """
        empty_netlist.add_net('net_a')  # No parameters passed
        empty_netlist.add_net('net_b')  # No parameters passed

        a = empty_netlist.net_dict['net_a']
        b = empty_netlist.net_dict['net_b']
        # Each net should have its own parameters list object
        assert a.parameters is not b.parameters

        # Mutating one should not affect the other
        a.parameters.append('max_voltage=3.3')
        assert a.parameters == ['max_voltage=3.3']
        assert b.parameters == []

    def test_addnet_explicit_parameters_work(self, empty_netlist):
        """Passing explicit parameters should work as before (stored on Net.parameters)."""
        params = ['max_voltage=1.8', 'min_voltage=0']
        empty_netlist.add_net('vdd', params)
        assert empty_netlist.net_dict['vdd'].parameters == params


# ---------------------------------------------------------------------------
# Auto-creation of nets (lazy net creation)
# ---------------------------------------------------------------------------

class TestLazyNetCreation:

    def test_connecting_element_auto_creates_nets(self, pdk, empty_netlist):
        """When add_element is called with nets that don't exist yet, they are auto-created."""
        ckt = empty_netlist
        # Don't call add_net first — nets should be auto-created
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'auto_out', 'g': 'auto_in', 's': 'auto_gnd', 'b': 'auto_gnd'})
        assert 'auto_out' in ckt.net_dict
        assert 'auto_in' in ckt.net_dict
        assert 'auto_gnd' in ckt.net_dict


# ---------------------------------------------------------------------------
# Placement and factory rewiring
# ---------------------------------------------------------------------------

class TestRewire:

    def test_factory_pin_assignment_updates_net(self, pdk, empty_netlist):
        dev = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        empty_netlist.add(dev)
        dev.d = 'new_out'
        assert dev.nets['d'] == 'new_out'
        assert empty_netlist._element_mgr._pin_to_net['mn1', 'd'] == 'new_out'

    def test_connect_nonexistent_module_refuses(self, empty_netlist):
        from silk.exceptions import CircuitError
        with pytest.raises(CircuitError, match='does_not_exist'):
            empty_netlist.connect_module('does_not_exist', d='out')

    def test_connect_wrong_module_port_refuses(self, empty_netlist, basic_module):
        from silk.exceptions import CircuitError
        empty_netlist.add_module(basic_module, 'stage', nets={'in': 'in', 'out': 'out'})
        with pytest.raises(CircuitError, match='nonexistent_pin'):
            empty_netlist.connect_module('stage', nonexistent_pin='some_net')


# ---------------------------------------------------------------------------
# update_element
# ---------------------------------------------------------------------------

class TestUpdateElement:

    def test_update_element_changes_parameter(self, pdk, empty_netlist):
        """update_element modifies parameter values on the stored device."""
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15})
        ckt.update_element('mn1', ['w'], [5])
        param_list = ckt.elem_dict['mn1']['device'].param_list
        assert param_list['w'].value.value == 5

    def test_update_nonexistent_element_exits(self, empty_netlist):
        """Updating a nonexistent element raises CircuitError."""
        with pytest.raises(CircuitError):
            empty_netlist.update_element('ghost', ['w'], [1])


# ---------------------------------------------------------------------------
# Deep copy isolation
# ---------------------------------------------------------------------------

class TestDeepCopyIsolation:

    def test_elements_are_deep_copied(self, pdk, empty_netlist):
        """Modifying the original PDK device after add_element doesn't affect the stored copy."""
        ckt = empty_netlist
        original_device = pdk.nmos1p8
        ckt.add_element(original_device, 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15})
        # The stored device should be a separate copy
        stored_device = ckt.elem_dict['mn1']['device']
        assert stored_device is not original_device

    def test_two_instances_independent(self, pdk, empty_netlist):
        """Two instances of the same device type have independent parameters."""
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'out1', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15})
        ckt.add_element(pdk.nmos1p8, 'mn2',
                       nets={'d': 'out2', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 10, 'l': 0.3})
        w1 = ckt.elem_dict['mn1']['device'].param_list['w'].value.value
        w2 = ckt.elem_dict['mn2']['device'].param_list['w'].value.value
        assert w1 != w2
        assert w1 == 1
        assert w2 == 10


# ---------------------------------------------------------------------------
# elem_dict / net_dict contents after operations
# ---------------------------------------------------------------------------

class TestDictContents:

    def test_elem_dict_keys_match_added_elements(self, basic_netlist):
        """elem_dict keys exactly match the set of added element names."""
        assert set(basic_netlist.elem_dict.keys()) == {'mn1', 'mp1'}

    def test_net_dict_contains_all_connected_nets(self, basic_netlist):
        """net_dict contains all nets that were explicitly added or auto-created."""
        expected = {'vdd', 'gnd', 'in', 'out'}
        assert expected.issubset(set(basic_netlist.net_dict.keys()))

    def test_element_has_device_key(self, basic_netlist):
        """Each element entry has a 'device' key with the device object."""
        for name, elem in basic_netlist.elem_dict.items():
            assert 'device' in elem
            assert 'type' in elem
            assert 'typeName' in elem


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------

class TestDebugMode:

    def test_add_element_debug_prints(self, pdk, tmp_dir, capsys):
        """add_element with debug=True prints debug messages (old-style PDKDevice path)."""
        from silk.circuit.modules import Module
        ckt = Module('PDK_skywater130', tmp_dir, 'dbg.cir',
                           pdk_details=pdk, debug=True)
        # Use PDKDevice directly (old-style) to exercise the debug print branch
        ckt.add_element(pdk._devices['nmos1p8'], 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15, 'm': 1})
        captured = capsys.readouterr()
        assert 'Element Name found in pdk' in captured.out

    def test_add_module_debug_prints(self, pdk, tmp_dir, basic_module, capsys):
        """Adding a module element with debug=True prints module debug messages."""
        from silk.circuit.modules import Module
        ckt = Module('PDK_skywater130', tmp_dir, 'dbg.cir',
                           pdk_details=pdk, debug=True)
        ckt.add_element(basic_module, 'xinv',
                       nets={'in': 'a', 'out': 'b', 'vdd': 'vdd', 'vss': 'gnd'})
        captured = capsys.readouterr()
        assert 'Element Name found in custom module list' in captured.out

    def test_add_net_debug_duplicate(self, pdk, tmp_dir, capsys):
        """add_net with debug=True prints message for duplicate net."""
        from silk.circuit.circuit_elements import PyCircuitElements
        mgr = PyCircuitElements(pdk, debug=True)
        mgr.add_net('vdd')
        mgr.add_net('vdd')  # duplicate
        captured = capsys.readouterr()
        assert 'already exists' in captured.out

    def test_connect_debug_auto_net_warning(self, pdk, tmp_dir, capsys):
        """connect_element with debug=True warns about auto-created nets."""
        from silk.circuit.circuit_elements import PyCircuitElements
        mgr = PyCircuitElements(pdk, debug=True)
        mgr.add_element(pdk.nmos1p8, 'mn1')
        # Connect to a net that doesn't exist yet
        mgr.connect_element('mn1', ['d'], ['auto_new_net'])
        captured = capsys.readouterr()
        assert 'undefined' in captured.out or 'add_net' in captured.out


# ---------------------------------------------------------------------------
# List-style nets and parameters
# ---------------------------------------------------------------------------

class TestListStyleArgs:

    def test_add_element_with_list_nets(self, pdk, empty_netlist):
        """add_element accepts list-of-tuples style nets."""
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets=[('d', 'out'), ('g', 'in'), ('s', 'gnd'), ('b', 'gnd')],
                       parameters={'w': 1, 'l': 0.15})
        pin_list = ckt.elem_dict['mn1']['device'].pin_list
        assert pin_list['d'].net == 'out'
        assert pin_list['g'].net == 'in'

    def test_add_element_with_list_params(self, pdk, empty_netlist):
        """add_element accepts list-of-tuples style parameters."""
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters=[('w', 2), ('l', 0.3)])
        param_list = ckt.elem_dict['mn1']['device'].param_list
        assert param_list['w'].value.value == 2
        assert param_list['l'].value.value == 0.3


# ---------------------------------------------------------------------------
# Primitive element type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# update_element edge cases
# ---------------------------------------------------------------------------

class TestUpdateElementEdgeCases:

    def test_params_length_mismatch_exits(self, pdk, empty_netlist):
        """update_element exits when params and paramValues have different lengths."""
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15})
        with pytest.raises(CircuitError, match='params length'):
            ckt.update_element('mn1', ['w', 'l'], [5])  # 2 params, 1 value

    def test_update_element_with_module_variable(self, pdk, empty_netlist):
        """update_element accepts a Variable directly as paramValue."""
        from silk.circuit.variable import Variable
        ckt = empty_netlist
        ckt.add_element(pdk.nmos1p8, 'mn1',
                       nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
                       parameters={'w': 1, 'l': 0.15})
        mv = Variable('w', value=7.5)
        ckt.update_element('mn1', ['w'], [mv])
        param_list = ckt.elem_dict['mn1']['device'].param_list
        assert param_list['w'].value is mv


# ---------------------------------------------------------------------------
# New-style class add_element — class not in PDK raises error (line 60)
# ---------------------------------------------------------------------------

class TestNewStyleClassDeviceNotInPDK:

    def test_new_style_class_not_in_pdk_raises(self, pdk, empty_netlist):
        """add_element with a new-style class whose type is not in PDK raises CircuitError (line 60)."""
        from silk.elements.factory import make_device_class
        from silk.elements.base import AnalogElement

        spec = {
            'PDKDevice': 'nonexistent_pdk_device',
            'pinList': ['a', 'b'],
            'paramList': {},
        }
        FakeCls = make_device_class('totally_fake_device', spec, is_pdk=True)
        # FakeCls is a class (subclass of AnalogElement) whose type is not in PDK
        with pytest.raises(CircuitError):
            empty_netlist.add_element(FakeCls, 'fake1',
                                     nets={'a': 'na', 'b': 'nb'})


# ---------------------------------------------------------------------------
# New-style AnalogElement instance path (lines 82-83)
# ---------------------------------------------------------------------------

class TestNewStyleInstancePath:

    def test_add_analog_element_instance_directly(self, pdk, empty_netlist):
        """add_element with an AnalogElement instance goes through _add_analog_element (lines 82-83)."""
        from silk.elements.factory import make_device_class
        from silk.elements.base import AnalogElement

        spec = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
            'paramList': {
                'w': {'minval': 1.0, 'maxval': 10.0, 'unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 1.0, 'unit': 'u'},
            },
        }
        NMOSCls = make_device_class('nmos1p8', spec, is_pdk=True)
        instance = NMOSCls('mn_inst', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15)

        # Pass an instance (not the class) — should call _add_analog_element
        empty_netlist.add_element(instance, 'mn_inst')
        assert 'mn_inst' in empty_netlist.elem_dict
        # Stored device is the AnalogElement instance itself (not a copy).
        assert empty_netlist.elem_dict['mn_inst']['device'] is instance
        assert isinstance(empty_netlist.elem_dict['mn_inst']['device'], AnalogElement)


# ---------------------------------------------------------------------------
# Positional list parameters in _apply_parameters (lines 341-342)
# ---------------------------------------------------------------------------

class TestPositionalListParameters:

    def test_positional_list_parameters(self, pdk, empty_netlist):
        """add_element with a plain positional list for parameters applies them in order (lines 341-342)."""
        ckt = empty_netlist
        # pin_list order for nmos1p8 is d, g, s, b
        # param_list order is w, l (and m if present)
        # Use positional list: values in param_list key order
        # First let's figure out param order
        device_copy = pdk._devices.get('nmos1p8')
        import copy
        dev_copy = copy.deepcopy(device_copy)
        param_order = list(dev_copy.param_list.keys())

        # Build positional values matching order
        param_vals = []
        defaults = {'w': 3.0, 'l': 0.15, 'm': 1}
        for p in param_order:
            param_vals.append(defaults.get(p, 1))

        ckt.add_element(
            pdk.nmos1p8, 'mn_pos',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters=param_vals,  # positional list, not dict or tuples
        )
        assert 'mn_pos' in ckt.elem_dict


# ---------------------------------------------------------------------------
# update_element on MODULE type → "not complete" (lines 265-267)
# ---------------------------------------------------------------------------

class TestUpdateElementAnalogElementInstance:

    def test_update_analog_element_instance_params(self, pdk, empty_netlist):
        """update_element on an AnalogElement instance updates its .params dict directly."""
        from silk.elements.factory import make_device_class
        from silk.elements.base import AnalogElement

        spec = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
            'paramList': {
                'w': {'minval': 1.0, 'maxval': 10.0, 'unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 1.0, 'unit': 'u'},
            },
        }
        NMOSCls = make_device_class('nmos1p8', spec, is_pdk=True)
        mn1 = NMOSCls('mn_ns', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.15)

        # Add via add_element(instance) — stored as the AnalogElement itself.
        empty_netlist.add_element(mn1, 'mn_ns')
        assert isinstance(empty_netlist.elem_dict['mn_ns']['device'], AnalogElement)

        # update_element takes the AnalogElement branch via isinstance dispatch.
        empty_netlist.update_element('mn_ns', ['w'], [5.0])
        assert empty_netlist.elem_dict['mn_ns']['device'].params['w'] == 5.0


class TestFallbackGetAttr:

    def test_getattr_fallback_when_device_not_in_devices_dict(self, pdk, empty_netlist):
        """add_element uses getattr(pdk, elemType) fallback when _devices.get() returns None (line 122).

        We patch the _devices dict to return None for a specific device, forcing the fallback
        to use getattr(pdk_details, elemType).
        """
        from silk.circuit.circuit_elements import PyCircuitElements

        # Get the real old-style PDKDevice for nmos1p8
        real_old_device = pdk._devices['nmos1p8']
        # Set up a custom PDK-like object where _devices.get() returns None
        # but getattr() still works
        import copy

        class FakePDK:
            """Minimal PDK-like that has nmos1p8 as attr but _devices returns None for it."""
            model_names = ['nmos1p8']
            module_names = ['module']
            primitiveNames = []
            _devices = {}  # empty — so .get() returns None

        fake_pdk = FakePDK()
        # Set the attribute directly so getattr(fake_pdk, 'nmos1p8') works
        setattr(fake_pdk, 'nmos1p8', copy.deepcopy(real_old_device))

        mgr = PyCircuitElements(fake_pdk)
        mgr.add_element(
            real_old_device,  # old-style PDKDevice
            'mn_fallback',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15, 'm': 1},
        )
        assert 'mn_fallback' in mgr.elem_dict


class TestUpdateElementModuleType:

    def test_update_module_type_raises_error(self, pdk, tmp_dir, basic_module):
        """update_element on MODULE-typed element raises error (not supported)."""
        from silk.circuit.circuit_elements import PyCircuitElements, ElementType

        mgr = PyCircuitElements(pdk)
        mgr.elem_dict['xinv'] = {
            'type': ElementType.MODULE,
            'device': basic_module,
            'typeName': 'module',
        }
        with pytest.raises(CircuitError, match='only supports DEVICE'):
            mgr.update_element('xinv', ['some_param'], [42])


# ---------------------------------------------------------------------------
# Phase 1 net refactor: module back-ref and reverse index
# ---------------------------------------------------------------------------

class TestNetRefactorInfrastructure:

    def test_module_backref_is_set(self, empty_netlist):
        """Module.__init__ wires _element_mgr._module_ref to the module itself."""
        assert empty_netlist._element_mgr._module_ref is empty_netlist

    def test_pin_to_net_index_starts_empty(self, empty_netlist):
        """_pin_to_net reverse index is initialized as an empty dict."""
        assert empty_netlist._element_mgr._pin_to_net == {}

    def test_ensure_net_creates_new_net(self, empty_netlist):
        """_ensure_net creates a new Net when the name is missing."""
        from silk.circuit.net import Net
        mgr = empty_netlist._element_mgr
        net = mgr._ensure_net('vout')
        assert isinstance(net, Net)
        assert net.name == 'vout'
        assert net.module is empty_netlist
        assert 'vout' in mgr.net_dict
        assert mgr.net_dict['vout'] is net

    def test_ensure_net_returns_existing_net(self, empty_netlist):
        """_ensure_net returns the existing Net if one is already registered."""
        mgr = empty_netlist._element_mgr
        first = mgr._ensure_net('vdd')
        second = mgr._ensure_net('vdd')
        assert first is second

    def test_ensure_net_raises_if_strict(self, pdk, tmp_dir):
        """When strict_nets=True, _ensure_net raises for unknown names."""
        from silk.circuit.modules import Module
        from silk.exceptions import UndeclaredNetError

        strict = Module('PDK_skywater130', tmp_dir, 'strict.cir',
                        pdk_details=pdk, strict_nets=True)
        with pytest.raises(UndeclaredNetError):
            strict._element_mgr._ensure_net('never_declared')

    def test_bind_pin_to_net_appends_binding(self, empty_netlist):
        """_bind_pin_to_net appends a PortBinding to net.pins."""
        from silk.circuit.net import PortBinding
        mgr = empty_netlist._element_mgr
        binding = PortBinding(
            net='vout', instance='mn1', port='d',
            direction=None, role=None, kind='device',
        )
        net = mgr._bind_pin_to_net('vout', binding)
        assert len(net.pins) == 1
        assert net.pins[0] is binding
        assert mgr._pin_to_net[('mn1', 'd')] == 'vout'

    def test_bind_pin_to_net_creates_net_if_missing(self, empty_netlist):
        """_bind_pin_to_net calls _ensure_net, so new nets are auto-created."""
        from silk.circuit.net import PortBinding
        mgr = empty_netlist._element_mgr
        mgr._bind_pin_to_net('newnet', PortBinding(
            net='newnet', instance='mn1', port='d'
        ))
        assert 'newnet' in mgr.net_dict

    def test_bind_pin_to_net_rewire_removes_stale_binding(self, empty_netlist):
        """Rebinding a pin removes the stale binding from the old net."""
        from silk.circuit.net import PortBinding
        mgr = empty_netlist._element_mgr
        # Initial bind: (mn1, d) -> 'old_net'
        mgr._bind_pin_to_net('old_net', PortBinding(
            net='old_net', instance='mn1', port='d'
        ))
        assert len(mgr.net_dict['old_net'].pins) == 1

        # Rebind: (mn1, d) -> 'new_net'
        mgr._bind_pin_to_net('new_net', PortBinding(
            net='new_net', instance='mn1', port='d'
        ))
        # Stale binding removed from old_net
        assert len(mgr.net_dict['old_net'].pins) == 0
        # New binding on new_net
        assert len(mgr.net_dict['new_net'].pins) == 1
        assert mgr.net_dict['new_net'].pins[0].instance == 'mn1'
        # Reverse index updated
        assert mgr._pin_to_net[('mn1', 'd')] == 'new_net'

    def test_bind_pin_to_net_same_net_noop(self, empty_netlist):
        """Rebinding the same (instance, port) to the same net is a no-op.

        The chokepoint returns early when the reverse-index lookup shows
        the pin is already bound to the target net, preventing duplicate
        PortBinding entries that would otherwise inflate fanout counts.
        """
        from silk.circuit.net import PortBinding
        mgr = empty_netlist._element_mgr
        mgr._bind_pin_to_net('vout', PortBinding(
            net='vout', instance='mn1', port='d'
        ))
        mgr._bind_pin_to_net('vout', PortBinding(
            net='vout', instance='mn1', port='d'
        ))
        # Only one binding — second call is a no-op.
        assert len(mgr.net_dict['vout'].pins) == 1

    def test_bind_pin_to_net_strict_propagates(self, pdk, tmp_dir):
        """_bind_pin_to_net raises UndeclaredNetError via _ensure_net when strict."""
        from silk.circuit.modules import Module
        from silk.circuit.net import PortBinding
        from silk.exceptions import UndeclaredNetError

        strict = Module('PDK_skywater130', tmp_dir, 'strict_bind.cir',
                        pdk_details=pdk, strict_nets=True)
        with pytest.raises(UndeclaredNetError):
            strict._element_mgr._bind_pin_to_net(
                'never_declared',
                PortBinding(net='never_declared', instance='mn1', port='d'),
            )

    def test_add_net_creates_net_object(self, empty_netlist):
        """add_net stores a Net object in net_dict."""
        from silk.circuit.net import Net
        empty_netlist.add_net('vdd', ['max_voltage=1.8'])
        value = empty_netlist.net_dict['vdd']
        assert isinstance(value, Net)
        assert value.parameters == ['max_voltage=1.8']

    def test_add_net_default_parameters_independent(self, empty_netlist):
        """Default parameters are still independent lists across nets."""
        empty_netlist.add_net('a')
        empty_netlist.add_net('b')
        assert empty_netlist.net_dict['a'].parameters is not empty_netlist.net_dict['b'].parameters

    def test_update_net_replaces_parameters(self, empty_netlist):
        """update_net updates the Net.parameters list in place of the old list."""
        empty_netlist.add_net('vdd', ['a'])
        empty_netlist.update_net('vdd', ['b', 'c'])
        assert empty_netlist.net_dict['vdd'].parameters == ['b', 'c']

    def test_connect_element_populates_net_pins(self, pdk, empty_netlist):
        """After add_element (which calls _apply_connections -> connect_element),
        net_dict[net].pins contains one PortBinding per connected pin."""
        empty_netlist.add_element(
            pdk.nmos1p8, 'mn1',
            nets={'d': 'out', 'g': 'in', 's': 'gnd', 'b': 'gnd'},
            parameters={'w': 1, 'l': 0.15},
        )
        out_pins = empty_netlist.net_dict['out'].pins
        assert len(out_pins) == 1
        assert out_pins[0].instance == 'mn1'
        assert out_pins[0].port == 'd'
        assert out_pins[0].kind == 'device'

    def test_factory_rewire_updates_net_pins(self, pdk, empty_netlist):
        dev = pdk.nmos1p8('mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        empty_netlist.add(dev)
        empty_netlist.add_net('out2')
        dev.d = 'out2'
        assert len(empty_netlist.net_dict['out'].pins) == 0
        new_pins = empty_netlist.net_dict['out2'].pins
        assert len(new_pins) == 1
        assert new_pins[0].instance == 'mn1'
        assert new_pins[0].port == 'd'

    def test_connect_module_has_submodule_kind(self, pdk, empty_netlist, basic_module):
        """Submodule pin bindings get kind='submodule' so checkers can filter."""
        empty_netlist.add_element(
            basic_module, 'xinv1',
            nets={'in': 'sig_in', 'out': 'sig_out', 'vdd': 'vdd', 'vss': 'gnd'},
        )
        sig_out_pins = empty_netlist.net_dict['sig_out'].pins
        assert len(sig_out_pins) == 1
        assert sig_out_pins[0].kind == 'submodule'

    def test_add_analog_element_instance_populates_net_pins(self, pdk, empty_netlist):
        """_add_analog_element (via Module.add) populates Net.pins with device bindings."""
        dev = pdk.nmos1p8(name='mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        empty_netlist.add(dev)

        out_pins = empty_netlist.net_dict['out'].pins
        assert len(out_pins) == 1
        assert out_pins[0].instance == 'mn1'
        assert out_pins[0].port == 'd'
        assert out_pins[0].kind == 'device'

    def test_add_analog_element_sets_module_backref(self, pdk, empty_netlist):
        """_add_analog_element sets device._module to the parent Module."""
        dev = pdk.nmos1p8(name='mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        empty_netlist.add(dev)
        assert dev._module is empty_netlist

    def test_add_passive_populates_net_pins(self, empty_netlist):
        """add_passive registers pos/neg as PortBindings on the affected nets."""
        from silk.elements.passives import Resistor
        r = Resistor('r1', pos='va', neg='vb', resistance=1000)
        empty_netlist.add_passive(r)

        va_pins = empty_netlist.net_dict['va'].pins
        vb_pins = empty_netlist.net_dict['vb'].pins

        assert len(va_pins) == 1
        assert va_pins[0].instance == 'r1'
        assert va_pins[0].port == 'pos'
        assert va_pins[0].kind == 'passive'

        assert len(vb_pins) == 1
        assert vb_pins[0].instance == 'r1'
        assert vb_pins[0].port == 'neg'
        assert vb_pins[0].kind == 'passive'

    def test_add_source_populates_net_pins(self, empty_netlist):
        """add_source registers source terminals in Net.pins with kind='source'."""
        from silk import sources
        empty_netlist.add_net('vdd')
        empty_netlist.add_net('gnd')
        src = sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8)
        empty_netlist.add_source(src)

        vdd_pins = empty_netlist.net_dict['vdd'].pins
        gnd_pins = empty_netlist.net_dict['gnd'].pins

        # Locate the source binding (nets may hold other bindings in complex tests).
        vdd_src_bindings = [b for b in vdd_pins if b.instance == 'vsup']
        gnd_src_bindings = [b for b in gnd_pins if b.instance == 'vsup']

        assert len(vdd_src_bindings) == 1
        assert vdd_src_bindings[0].port == 'positive'
        assert vdd_src_bindings[0].kind == 'source'

        assert len(gnd_src_bindings) == 1
        assert gnd_src_bindings[0].port == 'negative'
        assert gnd_src_bindings[0].kind == 'source'
