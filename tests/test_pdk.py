"""
Tests for BasePDK and related dataclasses.

Covers YAML loading, device attributes, model_names, Pin/ParameterConstraint/PDKDevice
dataclasses, generate_netlist_entry(), deep copy independence, and error paths.
"""

import copy
import pytest

from silk.pdk.basepdk import BasePDK, PDKDevice, Pin, ParameterConstraint, ValidationError
from silk.exceptions import CircuitError


# ---------------------------------------------------------------------------
# Constructor / YAML loading
# ---------------------------------------------------------------------------

class TestPDKLoading:

    def test_constructor_loads_yaml(self, pdk):
        """PDK constructor loads YAML without raising."""
        assert pdk is not None

    def test_bad_yaml_path_raises(self, tmp_dir):
        """Passing a non-existent YAML path raises an error."""
        with pytest.raises(Exception):
            BasePDK('/nonexistent/path.yaml', library_location='/fake')

    def test_model_names_populated(self, pdk):
        """model_names list is populated after loading."""
        assert isinstance(pdk.model_names, list)
        assert len(pdk.model_names) > 0

    def test_model_names_contains_expected_devices(self, pdk):
        """model_names includes core device types from YAML."""
        expected = ['nmos1p8', 'pmos1p8', 'res_hpo', 'cap_mim1']
        for name in expected:
            assert name in pdk.model_names, f"{name} not in model_names"


# ---------------------------------------------------------------------------
# Device attributes
# ---------------------------------------------------------------------------

class TestPDKDeviceAttributes:

    def test_nmos1p8_attribute_exists(self, pdk):
        """pdk.nmos1p8 attribute exists and is an AnalogElement subclass (Phase 2+)."""
        from silk.elements.base import AnalogElement
        assert hasattr(pdk, 'nmos1p8')
        assert isinstance(pdk.nmos1p8, type)
        assert issubclass(pdk.nmos1p8, AnalogElement)

    def test_pmos1p8_attribute_exists(self, pdk):
        """pdk.pmos1p8 attribute exists and is an AnalogElement subclass (Phase 2+)."""
        from silk.elements.base import AnalogElement
        assert hasattr(pdk, 'pmos1p8')
        assert isinstance(pdk.pmos1p8, type)
        assert issubclass(pdk.pmos1p8, AnalogElement)

    def test_device_has_pin_names(self, pdk):
        """Each device class has correct _pin_names tuple."""
        cls = pdk.nmos1p8
        assert set(cls._pin_names) == {'d', 'g', 's', 'b'}

    def test_device_has_param_names(self, pdk):
        """Each device class has correct _param_names tuple."""
        cls = pdk.nmos1p8
        assert 'w' in cls._param_names
        assert 'l' in cls._param_names
        assert 'm' in cls._param_names

    def test_device_pdk_device_name(self, pdk):
        """Device class stores correct _pdk_device string."""
        assert pdk.nmos1p8._pdk_device == 'sky130_fd_pr__nfet_01v8'
        assert pdk.pmos1p8._pdk_device == 'sky130_fd_pr__pfet_01v8'

    def test_resistor_has_terminal_pin_names(self, pdk):
        """res_hpo device class carries its a/b terminals (plus the body pin
        `c` added in the PV3 arity fix — the sky130 subckt is 3-terminal)."""
        cls = pdk.res_hpo
        assert 'a' in cls._pin_names
        assert 'b' in cls._pin_names
        assert 'c' in cls._pin_names

    def test_internal_devices_still_pdk_device(self, pdk):
        """pdk._devices still holds PDKDevice objects for backward compat."""
        assert isinstance(pdk._devices['nmos1p8'], PDKDevice)


# ---------------------------------------------------------------------------
# Pin dataclass
# ---------------------------------------------------------------------------

class TestPin:

    def test_pin_has_name(self):
        """Pin dataclass has name attribute."""
        p = Pin(name='d', net_name='drain')
        assert p.name == 'd'

    def test_pin_has_net_name(self):
        """Pin dataclass has net_name attribute."""
        p = Pin(name='g', net_name='gate')
        assert p.net_name == 'gate'

    def test_pin_net_defaults_to_none(self):
        """Pin.net defaults to None."""
        p = Pin(name='s', net_name='source')
        assert p.net is None

    def test_pin_net_assignable(self):
        """Pin.net can be assigned a net string."""
        p = Pin(name='d', net_name='drain', net='vdd')
        assert p.net == 'vdd'


# ---------------------------------------------------------------------------
# ParameterConstraint dataclass
# ---------------------------------------------------------------------------

class TestParameterConstraint:

    def test_has_minval_maxval_value(self):
        """ParameterConstraint has minval, maxval, value attributes."""
        pc = ParameterConstraint(minval='0.15', maxval='-1', value=None, pdk_unit='u')
        assert pc.minval == '0.15'
        assert pc.maxval == '-1'
        assert pc.value is None

    def test_unit_extracted_from_minval(self):
        """Unit is auto-extracted from minval suffix when not provided."""
        pc = ParameterConstraint(minval='0.42u', maxval='-1', value=None)
        assert pc.pdk_unit == 'u'

    def test_numeric_values_normalized(self):
        """Numeric minval/maxval are normalized to strings."""
        pc = ParameterConstraint(minval=1, maxval=100, value=None)
        assert pc.minval == '1'
        assert pc.maxval == '100'


# ---------------------------------------------------------------------------
# PDKDevice.from_dict
# ---------------------------------------------------------------------------

class TestPDKDeviceFromDict:

    def test_from_dict_creates_device(self):
        """PDKDevice.from_dict constructs a device from a minimal dict."""
        data = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'paramList': {
                'w': {'minval': '0.42', 'maxval': '-1', 'pdk_unit': 'u'},
                'l': {'minval': '0.15', 'maxval': '-1', 'pdk_unit': 'u'},
            },
            'pinList': {
                'd': 'drain',
                'g': 'gate',
                's': 'source',
                'b': 'bulk',
            }
        }
        dev = PDKDevice.from_dict('test_nmos', data)
        assert dev.device_name == 'test_nmos'
        assert dev.pdk_device == 'sky130_fd_pr__nfet_01v8'
        assert 'w' in dev.param_list
        assert 'l' in dev.param_list
        assert isinstance(dev.pin_list['d'], Pin)
        assert dev.pin_list['d'].net_name == 'drain'

    def test_from_dict_pin_objects(self):
        """from_dict creates Pin objects with correct attributes."""
        data = {
            'PDKDevice': 'test_dev',
            'paramList': {'m': {'minval': '1', 'maxval': '-1'}},
            'pinList': {'a': 'anode', 'k': 'cathode'},
        }
        dev = PDKDevice.from_dict('diode', data)
        assert dev.pin_list['a'].name == 'a'
        assert dev.pin_list['a'].net_name == 'anode'
        assert dev.pin_list['a'].net is None

    def test_from_dict_spice_prefix_defaults_to_XM(self):
        """A catalog entry that says nothing about its prefix is a subckt
        call, spelled XM<name> — the same default the factory path applies."""
        from silk.pdk.basepdk import DEFAULT_PDK_SPICE_PREFIX
        data = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'paramList': {'w': {'minval': '0.42', 'maxval': '-1'}},
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
        }
        dev = PDKDevice.from_dict('test_nmos', data)
        assert dev.spice_prefix == 'XM' == DEFAULT_PDK_SPICE_PREFIX

    def test_from_dict_spice_prefix_from_catalog(self):
        """`spice_prefix: D` in the catalog rides on the PDKDevice (TASK-23):
        a .model primitive must not be emitted as a subcircuit call."""
        data = {
            'PDKDevice': 'diode_nd2ps_03v3',
            'spice_prefix': 'D',
            'paramList': {'m': {'minval': '1', 'maxval': '-1'}},
            'pinList': {'a': 'anode', 'k': 'cathode'},
        }
        dev = PDKDevice.from_dict('diode', data)
        assert dev.spice_prefix == 'D'
        # Survives the deepcopy add_element performs on the template.
        assert copy.deepcopy(dev).spice_prefix == 'D'


# ---------------------------------------------------------------------------
# Deep copy independence
# ---------------------------------------------------------------------------

class TestDeviceDeepCopy:

    def test_deep_copy_independent(self, pdk):
        """Two deep copies of the same PDK device are independent."""
        dev1 = copy.deepcopy(pdk._devices['nmos1p8'])
        dev2 = copy.deepcopy(pdk._devices['nmos1p8'])

        # Modify pin on dev1
        dev1.pin_list['d'].net = 'out'

        # dev2 should not be affected
        assert dev2.pin_list['d'].net is None

    def test_deep_copy_param_independent(self, pdk):
        """Modifying param on one copy doesn't affect the other."""
        dev1 = copy.deepcopy(pdk._devices['nmos1p8'])
        dev2 = copy.deepcopy(pdk._devices['nmos1p8'])

        dev1.param_list['w'].value = 42
        assert dev2.param_list['w'].value is None


# ---------------------------------------------------------------------------
# generate_netlist_entry
# ---------------------------------------------------------------------------

class TestGenerateNetlistEntry:

    def test_returns_string(self, pdk):
        """generate_netlist_entry returns a string."""
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        # Set up pin connections
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        # Set parameter values (need Variable-like objects)
        from unittest.mock import MagicMock
        for param_name, constraint in dev.param_list.items():
            mock_val = MagicMock()
            mock_val.value = 1.0
            constraint.value = mock_val

        details = {'device': dev}
        result = pdk.generate_netlist_entry('N1', details)
        assert isinstance(result, str)

    def test_contains_instance_name(self, pdk):
        """generate_netlist_entry output contains the instance name."""
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        from unittest.mock import MagicMock
        for param_name, constraint in dev.param_list.items():
            mock_val = MagicMock()
            mock_val.value = 1.0
            constraint.value = mock_val

        details = {'device': dev}
        result = pdk.generate_netlist_entry('N1', details)
        assert 'XMN1' in result

    def test_contains_pdk_device_name(self, pdk):
        """generate_netlist_entry output contains the PDK device name."""
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        from unittest.mock import MagicMock
        for param_name, constraint in dev.param_list.items():
            mock_val = MagicMock()
            mock_val.value = 1.0
            constraint.value = mock_val

        details = {'device': dev}
        result = pdk.generate_netlist_entry('N1', details)
        assert 'sky130_fd_pr__nfet_01v8' in result


# ---------------------------------------------------------------------------
# Catalog spice_prefix on the legacy add_element path (TASK-23)
# ---------------------------------------------------------------------------

_PREFIX_YAML = """\
nmos1p8:
  PDKDevice: sky130_fd_pr__nfet_01v8
  paramList:
    w: {minval: 0.42, maxval: -1, pdk_unit: u}
    l: {minval: 0.15, maxval: -1, pdk_unit: u}
    m: {minval: 1, maxval: -1}
  pinList: {d: drain, g: gate, s: source, b: bulk}

diode_nd2ps_3p3:
  PDKDevice: diode_nd2ps_03v3
  spice_prefix: D
  paramList:
    m: {minval: 1, maxval: -1}
  pinList: {a: anode, k: cathode}
"""


class TestSpicePrefixLegacyPath:
    """The factory path honoured the catalog ``spice_prefix:`` key; the
    ``add_element`` path and every instance-name consumer spelled ``XM``. A
    PDK whose devices are ``.model`` primitives (GF180MCU's diodes) could not
    be emitted correctly on the legacy path. Both paths now read the one
    declaration."""

    @pytest.fixture
    def prefix_pdk(self, tmp_path):
        yaml_path = tmp_path / 'prefix.yaml'
        yaml_path.write_text(_PREFIX_YAML)
        return BasePDK(str(yaml_path), library_location='/fake/path')

    def test_both_paths_read_the_same_declaration(self, prefix_pdk):
        for name, dev in prefix_pdk._devices.items():
            assert dev.spice_prefix == getattr(prefix_pdk, name)._spice_prefix, name
        assert prefix_pdk._devices['nmos1p8'].spice_prefix == 'XM'
        assert prefix_pdk._devices['diode_nd2ps_3p3'].spice_prefix == 'D'

    def test_generate_netlist_entry_uses_declared_prefix(self, prefix_pdk):
        dev = copy.deepcopy(prefix_pdk._devices['diode_nd2ps_3p3'])
        dev.pin_list['a'].net = 'p'
        dev.pin_list['k'].net = 'n'
        line = prefix_pdk.generate_netlist_entry('D1', {'device': dev})
        assert line.startswith('DD1 p n diode_nd2ps_03v3 '), line
        assert 'XM' not in line

    def test_emitted_instance_name_uses_declared_prefix(self, prefix_pdk):
        from silk.circuit.emitted_names import emitted_instance_name
        assert emitted_instance_name(
            prefix_pdk._devices['diode_nd2ps_3p3'], 'D1') == 'DD1'
        assert emitted_instance_name(
            prefix_pdk._devices['nmos1p8'], 'M1') == 'XMM1'

    def test_add_element_deck_line_is_a_D_primitive(self, prefix_pdk, tmp_path):
        """End to end through Module.add_element → generate_netlist: the
        deck carries a `D` line for the diode and an `XM` line for the MOS,
        from the same circuit source."""
        from silk.circuit.modules import Module
        from silk.circuit.circuit_elements import PortDir
        from silk.pdk.basepdk import PDKDevice as _PDKDevice
        m = Module('prefix', str(tmp_path), 'prefix.cir',
                   pdk_details=prefix_pdk, module_name='sub')
        m.define_module_ports({'p': PortDir.IN, 'n': PortDir.OUT,
                               'g': PortDir.GROUND})
        m.add_element(prefix_pdk.diode_nd2ps_3p3, 'D1', nets={'a': 'p', 'k': 'n'},
                      parameters={'m': 1})
        m.add_element(prefix_pdk.nmos1p8, 'M1',
                      nets={'d': 'n', 'g': 'p', 's': 'g', 'b': 'g'},
                      parameters={'w': 1.0, 'l': 0.15, 'm': 1})
        assert isinstance(m.elem_dict['D1']['device'], _PDKDevice)
        # A module with module_name returns its .subckt text (the parent
        # writes the file); that text is the deck the diode line lands in.
        deck = m.generate_netlist(corner='tt').splitlines()
        assert any(l.startswith('DD1 p n diode_nd2ps_03v3 ') for l in deck), deck
        assert any(l.startswith('XMM1 n p g g sky130_fd_pr__nfet_01v8 ') for l in deck), deck
        assert not any(l.startswith('XMD1') for l in deck), deck


# ---------------------------------------------------------------------------
# BasePDK __init__ paths — UNIT_SI notice, debug=True
# ---------------------------------------------------------------------------

class TestBasePDKInit:

    def test_unit_pdk_prints_notice(self, pdk_yaml_path, capsys):
        """UNIT_PDK mode prints a startup notice about native units."""
        from silk.pdk.basepdk import BasePDK, UNIT_PDK

        BasePDK(pdk_yaml_path, library_location='/fake/lib', user_units=UNIT_PDK)
        captured = capsys.readouterr()
        # Diagnostic banner goes to stderr, not stdout (keeps stdout clean JSON
        # for consumers like `silk.elements.schema --pdk`).
        err = captured.err
        assert 'User units' in err   # the banner is deterministic; no weak disjuncts

    def test_unit_si_no_pdk_notice(self, pdk_yaml_path, capsys):
        """UNIT_SI mode does not print the PDK-native units startup notice."""
        from silk.pdk.basepdk import BasePDK, UNIT_SI

        BasePDK(pdk_yaml_path, library_location='/fake/lib', user_units=UNIT_SI)
        out = capsys.readouterr().out
        # The PDK-native notice is only printed for UNIT_PDK
        assert 'User units: PDK defaults' not in out

    def test_debug_true_prints_device_summary(self, pdk_yaml_path, capsys):
        """debug=True prints the loaded device summary after YAML parsing."""
        from silk.pdk.basepdk import BasePDK

        BasePDK(pdk_yaml_path, library_location='/fake/lib', debug=True)
        out = capsys.readouterr().out
        # _parse_standard_yaml prints 'PDK loaded: N devices (...)' when debug=True
        assert 'PDK loaded' in out
        assert 'nmos1p8' in out

    def test_debug_true_prints_raw_config(self, pdk_yaml_path, capsys):
        """debug=True also prints the raw config_data dict."""
        from silk.pdk.basepdk import BasePDK

        BasePDK(pdk_yaml_path, library_location='/fake/lib', debug=True)
        out = capsys.readouterr().out
        # config_data is a dict; its repr contains 'PDKDevice'
        assert 'PDKDevice' in out


# ---------------------------------------------------------------------------
# generate_netlist_entry — UNIT_SI conversion and dimensionless integer param
# ---------------------------------------------------------------------------

class TestGenerateNetlistEntryUnitSI:

    def _make_details(self, pdk, param_values):
        """Helper: deep-copy nmos1p8, set pin nets and param values."""
        import copy
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        for p, val in param_values.items():
            dev.param_list[p].value = val
        return {'device': dev}

    def test_unit_si_w_converted_to_um(self, pdk_yaml_path, capsys):
        """UNIT_SI mode multiplies W (metres) by 1e6 to get µm."""
        from silk.pdk.basepdk import BasePDK, UNIT_SI

        pdk_si = BasePDK(pdk_yaml_path, library_location='/fake/lib',
                                 user_units=UNIT_SI)
        # 0.42e-6 m == 0.42 µm — expected converted value is 0.42
        details = self._make_details(pdk_si, {'w': 0.42e-6, 'l': 0.15e-6, 'm': 1})
        result = pdk_si.generate_netlist_entry('N1', details)
        # 0.42e-6 * 1e6 = 0.42
        assert 'w=0.42' in result

    def test_unit_si_l_converted_to_um(self, pdk_yaml_path, capsys):
        """UNIT_SI mode multiplies L (metres) by 1e6 to get µm."""
        from silk.pdk.basepdk import BasePDK, UNIT_SI

        pdk_si = BasePDK(pdk_yaml_path, library_location='/fake/lib',
                                 user_units=UNIT_SI)
        details = self._make_details(pdk_si, {'w': 0.42e-6, 'l': 0.15e-6, 'm': 1})
        result = pdk_si.generate_netlist_entry('N1', details)
        # 0.15e-6 * 1e6 = 0.15
        assert 'l=0.15' in result

    def test_unit_pdk_no_conversion(self, pdk_yaml_path):
        """UNIT_PDK mode does not scale W/L values."""
        from silk.pdk.basepdk import BasePDK, UNIT_PDK

        pdk_pdk = BasePDK(pdk_yaml_path, library_location='/fake/lib',
                                  user_units=UNIT_PDK)
        details = self._make_details(pdk_pdk, {'w': 1.0, 'l': 0.15, 'm': 1})
        result = pdk_pdk.generate_netlist_entry('N1', details)
        # Asserted on the *value*, not on '1.0': every value on this path now
        # goes through _fmt, the factory path's renderer, so an integral float
        # writes 'w=1' here exactly as it does there (TASK-73). What this test
        # guards is that 1.0 µm is not rescaled, and 'w=1' says that.
        assert result.split().count('w=1') == 1
        assert 'l=0.15' in result


class TestGenerateNetlistEntryDimensionless:

    def test_dimensionless_integer_param_no_warning(self, pdk, capsys):
        """Dimensionless param 'm' with integer value produces no warning."""
        import copy
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        dev.param_list['w'].value = 1.0
        dev.param_list['l'].value = 0.15
        dev.param_list['m'].value = 2   # integer — valid

        details = {'device': dev}
        pdk.generate_netlist_entry('N1', details)
        out = capsys.readouterr().out
        assert 'WARNING' not in out

    def test_dimensionless_param_in_netlist(self, pdk):
        """Dimensionless param 'm' appears in generated netlist string."""
        import copy
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        dev.param_list['w'].value = 1.0
        dev.param_list['l'].value = 0.15
        dev.param_list['m'].value = 3

        details = {'device': dev}
        result = pdk.generate_netlist_entry('N1', details)
        assert 'm=3' in result


# ---------------------------------------------------------------------------
# generate_netlist_entry debug=True path
# ---------------------------------------------------------------------------

class TestGenerateNetlistEntryDebug:

    def test_debug_prints_param_names_and_netlist(self, pdk_yaml_path, capsys):
        """debug=True in generate_netlist_entry prints param names and the entry."""
        import copy
        from silk.pdk.basepdk import BasePDK

        pdk_debug = BasePDK(pdk_yaml_path, library_location='/fake/lib',
                                    debug=True)
        capsys.readouterr()  # clear init output

        dev = copy.deepcopy(pdk_debug._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        dev.param_list['w'].value = 1.0
        dev.param_list['l'].value = 0.15
        dev.param_list['m'].value = 1

        details = {'device': dev}
        pdk_debug.generate_netlist_entry('N1', details)
        out = capsys.readouterr().out
        # debug=True should print param names ('w', 'l', 'm') and the entry line
        assert 'w' in out
        assert 'XMN1' in out


# ---------------------------------------------------------------------------
# ParameterConstraint.validate_value — warning paths
# ---------------------------------------------------------------------------

class TestParameterConstraintValidateValue:

    def test_maxval_bounded_no_warning_within_range(self, capsys):
        """No warning when value is within bounded maxval."""
        pc = ParameterConstraint(minval='0.15', maxval='10.0', value=None, pdk_unit='u')
        pc.validate_value(5.0, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' not in out

    def test_maxval_bounded_warns_above_max(self, capsys):
        """Warning printed when value exceeds bounded maxval != '-1'."""
        pc = ParameterConstraint(minval='0.15', maxval='2.0', value=None, pdk_unit='u')
        pc.validate_value(5.0, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert 'exceeds maximum' in out

    def test_maxval_minus1_unbounded_no_warning(self, capsys):
        """No warning when maxval='-1' (unbounded) regardless of value."""
        pc = ParameterConstraint(minval='0.15', maxval='-1', value=None, pdk_unit='u')
        pc.validate_value(999.0, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' not in out

    def test_dimensionless_non_integer_warns(self, capsys):
        """Dimensionless param (no unit) with non-integer value prints warning."""
        pc = ParameterConstraint(minval='1', maxval='-1', value=None, pdk_unit=None)
        pc.validate_value(1.5, 'm')
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert 'integer' in out

    def test_dimensionless_integer_no_warn(self, capsys):
        """Dimensionless param with exact integer value does not warn."""
        pc = ParameterConstraint(minval='1', maxval='-1', value=None, pdk_unit=None)
        pc.validate_value(2, 'm')
        out = capsys.readouterr().out
        assert 'WARNING' not in out

    def test_below_minval_warns(self, capsys):
        """Warning printed when value is below minval."""
        pc = ParameterConstraint(minval='0.42', maxval='-1', value=None, pdk_unit='u')
        pc.validate_value(0.1, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert 'below minimum' in out

    def test_at_minval_no_warn(self, capsys):
        """No warning when value equals minval exactly."""
        pc = ParameterConstraint(minval='0.42', maxval='-1', value=None, pdk_unit='u')
        pc.validate_value(0.42, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' not in out

    def test_non_numeric_value_warns(self, capsys):
        """Non-numeric value prints a validation warning."""
        pc = ParameterConstraint(minval='0.15', maxval='-1', value=None, pdk_unit='u')
        pc.validate_value('not_a_number', 'w')
        out = capsys.readouterr().out
        assert 'WARNING' in out

# ---------------------------------------------------------------------------
# ParameterConstraint.validate_value — maxval bound
# ---------------------------------------------------------------------------

class TestParameterConstraintMaxval:

    def test_above_maxval_warns(self, capsys):
        """Warning printed when value exceeds maxval."""
        pc = ParameterConstraint(minval='0.15', maxval='10.0', value=None, pdk_unit='u')
        pc.validate_value(15.0, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert 'exceeds maximum' in out

    def test_at_maxval_no_warn(self, capsys):
        """No warning when value equals maxval exactly."""
        pc = ParameterConstraint(minval='0.15', maxval='10.0', value=None, pdk_unit='u')
        pc.validate_value(10.0, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' not in out

    def test_maxval_minus_one_unbounded(self, capsys):
        """maxval='-1' means unbounded — no warning for any large value."""
        pc = ParameterConstraint(minval='0.15', maxval='-1', value=None, pdk_unit='u')
        pc.validate_value(1000.0, 'w')
        out = capsys.readouterr().out
        assert 'WARNING' not in out


# ---------------------------------------------------------------------------
# BasePDK — UNIT_SI mode and debug output
# ---------------------------------------------------------------------------

class TestBasePDKUnitMode:

    def test_unit_pdk_prints_notice(self, pdk_yaml_path, capsys):
        """UNIT_PDK mode prints a unit notice on construction."""
        from silk.pdk.basepdk import UNIT_PDK
        from silk.pdk.basepdk import BasePDK
        BasePDK(pdk_yaml_path, library_location='/fake', user_units=UNIT_PDK)
        # Diagnostic banner goes to stderr (keeps stdout clean for JSON consumers).
        err = capsys.readouterr().err
        assert 'PDK' in err or 'unit' in err.lower()

    def test_unit_si_no_notice(self, pdk_yaml_path, capsys):
        """UNIT_SI mode does not print a PDK unit notice."""
        from silk.pdk.basepdk import UNIT_SI
        from silk.pdk.basepdk import BasePDK
        BasePDK(pdk_yaml_path, library_location='/fake', user_units=UNIT_SI)
        captured = capsys.readouterr()
        assert 'PDK defaults' not in captured.out
        assert 'PDK defaults' not in captured.err

    def test_debug_prints_device_list(self, pdk_yaml_path, capsys):
        """debug=True prints loaded device names during construction."""
        from silk.pdk.basepdk import BasePDK
        BasePDK(pdk_yaml_path, library_location='/fake', debug=True)
        out = capsys.readouterr().out
        assert 'PDK loaded' in out

    def test_unit_si_scales_width(self, pdk_yaml_path):
        """UNIT_SI mode converts W in metres to PDK microns in netlist output."""
        import copy
        from silk.pdk.basepdk import UNIT_SI
        from silk.pdk.basepdk import BasePDK

        pdk_si = BasePDK(pdk_yaml_path, library_location='/fake', user_units=UNIT_SI)
        device = copy.deepcopy(pdk_si._devices['nmos1p8'])
        for pin_name, pin in device.pin_list.items():
            pin.net = 'net_' + pin_name
        # Set W to 2e-6 m (2 µm in SI), plain values matching existing test pattern
        device.param_list['w'].value = 2e-6
        device.param_list['l'].value = 0.15e-6
        device.param_list['m'].value = 1

        result = pdk_si.generate_netlist_entry('mn1', {'device': device})
        # 2e-6 m * 1e6 = 2 µm. Asserted on the value, not on '2.0': a converted
        # value now goes through _fmt, because 0.15e-6 * 1e6 lands on
        # 0.15000000000000002 in binary floating point and reading that in a
        # deck is worse than reading '0.15'.
        # Split on whitespace: the line no longer carries a trailing space, so
        # ' l=0.15 ' (the last token) would not match (TASK-73).
        toks = result.split()
        assert 'w=2' in toks
        assert 'l=0.15' in toks


# ---------------------------------------------------------------------------
# Coverage gaps in basepdk.py
# ---------------------------------------------------------------------------

class TestParameterConstraintValidateValueEdgeCases:
    """Target lines 101-102 and 111-112: ValueError in minval/maxval float()."""

    def test_minval_non_numeric_string_no_crash(self, capsys):
        """validate_value does not crash when minval is a non-numeric string (line 101-102)."""
        pc = ParameterConstraint(minval='auto', maxval='-1', value=None, pdk_unit='u')
        # Should not raise — the except ValueError branch on line 101-102 silently passes
        pc.validate_value(0.5, 'w')
        # No crash is the assertion; no WARNING for minval since minval not parseable

    def test_maxval_non_numeric_string_no_crash(self, capsys):
        """validate_value does not crash when maxval is a non-numeric string (line 111-112)."""
        pc = ParameterConstraint(minval='0.15', maxval='unbounded', value=None, pdk_unit='u')
        # Should not raise — the except ValueError branch on line 111-112 silently passes
        pc.validate_value(999.0, 'w')
        # No crash is the assertion


class TestParseStandardYamlErrorPaths:
    """Target lines 207-210 (YAMLError, KeyError) and 214 (duplicate device)."""

    def test_yaml_error_raises_validation_error(self, tmp_path):
        """_parse_standard_yaml raises ValidationError for invalid YAML (line 207-208)."""
        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text(': : : invalid: yaml: {{{\n')
        from silk.pdk.basepdk import BasePDK, ValidationError as PDKValidationError
        with pytest.raises(PDKValidationError, match='Error parsing YAML'):
            BasePDK(str(bad_yaml), library_location='/fake')

    def test_missing_key_raises_validation_error(self, tmp_path):
        """_parse_standard_yaml raises ValidationError for missing YAML key (line 209-210)."""
        # Valid YAML but missing required 'PDKDevice' key
        incomplete_yaml = tmp_path / 'incomplete.yaml'
        incomplete_yaml.write_text(
            'mydevice:\n'
            '  paramList:\n'
            '    w:\n'
            '      minval: "0.42"\n'
            '      maxval: "-1"\n'
            '  pinList:\n'
            '    d: drain\n'
            # Note: 'PDKDevice' key intentionally missing
        )
        from silk.pdk.basepdk import BasePDK, ValidationError as PDKValidationError
        with pytest.raises(PDKValidationError, match='Missing required field'):
            BasePDK(str(incomplete_yaml), library_location='/fake')


class TestGenerateNetlistEntrySymbolicParam:
    """Target lines 262-263: except (ValueError, TypeError) in SI conversion."""

    def test_symbolic_param_skips_si_conversion(self, pdk_yaml_path):
        """SI conversion is skipped for non-numeric (symbolic) param values (line 262-263)."""
        import copy
        from silk.pdk.basepdk import BasePDK, UNIT_SI

        pdk_si = BasePDK(pdk_yaml_path, library_location='/fake', user_units=UNIT_SI)
        dev = copy.deepcopy(pdk_si._devices['nmos1p8'])
        for pin_name, pin in dev.pin_list.items():
            pin.net = 'net_' + pin_name
        # Use a symbolic string that cannot be converted to float
        dev.param_list['w'].value = '{w_param}'
        dev.param_list['l'].value = 0.15e-6
        dev.param_list['m'].value = 1

        # Should not raise — ValueError/TypeError branch at line 262-263 catches it
        result = pdk_si.generate_netlist_entry('N1', {'device': dev})
        assert isinstance(result, str)
        assert '{w_param}' in result


class TestDuplicateDeviceInYaml:
    """PDK YAMLs with duplicate device definitions must fail loudly."""

    def test_duplicate_device_raises_pdk_error(self, tmp_path):
        """Loading a PDK YAML with two entries for the same device name raises PDKError.

        Silk uses a strict YAML loader that raises on duplicate mapping keys,
        so a malformed PDK YAML with two ``nmos1p8:`` entries (e.g. from a
        merge conflict or copy-paste) fails at load time instead of silently
        using the second definition.
        """
        from silk.pdk.basepdk import BasePDK
        from silk.exceptions import PDKError

        yaml_text = """\
nmos1p8:
  PDKDevice: sky130_fd_pr__nfet_01v8
  paramList:
    w: {minval: 0.42, maxval: -1, pdk_unit: u}
    l: {minval: 0.15, maxval: -1, pdk_unit: u}
    m: {minval: 1, maxval: -1}
  pinList: {d: drain, g: gate, s: source, b: bulk}

nmos1p8:
  PDKDevice: totally_different_device
  paramList:
    w: {minval: 999, maxval: -1, pdk_unit: u}
    l: {minval: 999, maxval: -1, pdk_unit: u}
    m: {minval: 1, maxval: -1}
  pinList: {d: drain, g: gate, s: source, b: bulk}
"""
        yaml_path = tmp_path / 'dup_devices.yaml'
        yaml_path.write_text(yaml_text)

        with pytest.raises(PDKError, match='device defined multiple times'):
            BasePDK(str(yaml_path), library_location='/fake/path')


# ---------------------------------------------------------------------------
# BasePDK library config tests
# ---------------------------------------------------------------------------

class TestBasePDKLibraryConfig:

    def test_basepdk_library_config_defaults(self, pdk):
        # legacy fixture built via the direct constructor → empty/defaulted config
        assert pdk.corners == {}
        assert pdk.default_corner == 'tt'
        assert pdk.extra_includes == []
        assert pdk.pdk_name is None

    def test_basepdk_accepts_library_config(self, pdk_yaml_path):
        from silk.pdk.basepdk import BasePDK
        p = BasePDK('silk/pdk/data/skywater_130_device_details.yaml',
                            library_location='/dev/null',
                            corners={'tt': {'kind': 'process', 'desc': 'typical'}},
                            default_corner='tt', extra_includes=['/x.spice'],
                            pdk_name='sky130')
        assert p.default_corner == 'tt' and p.extra_includes == ['/x.spice']
        assert p.pdk_name == 'sky130' and 'tt' in p.corners
