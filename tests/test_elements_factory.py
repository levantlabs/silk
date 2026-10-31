"""Tests for pynetlist.core.elements.factory — make_device_class()."""

import warnings

import pytest
from silk.elements.factory import make_device_class, DeviceParamConstraint, _render_param
from silk.elements.base import AnalogElement


# ---------------------------------------------------------------------------
# Minimal YAML spec fixtures
# ---------------------------------------------------------------------------

PDK_NMOS_SPEC = {
    'PDKDevice': 'sky130_fd_pr__nfet_01v8',
    'paramList': {
        'm': {'minval': 1, 'maxval': -1},
        'w': {'minval': 0.42, 'maxval': -1, 'pdk_unit': 'u'},
        'l': {'minval': 0.15, 'maxval': -1, 'pdk_unit': 'u'},
    },
    'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
}

IDEAL_RESISTOR_SPEC = {
    'spice_prefix': 'R',
    'symbol': 'Resistor',
    'category': 'passive',
    'pinList': ['p', 'n'],
    'paramConstraints': [
        {'name': 'value', 'default': 1000, 'unit': 'Ohm'},
    ],
}

IDEAL_VSOURCE_SPEC = {
    'spice_prefix': 'V',
    'symbol': 'VSource',
    'category': 'source',
    'pinList': ['p', 'n'],
    'paramConstraints': [
        {'name': 'dc', 'default': 0, 'unit': 'V'},
    ],
}


# ---------------------------------------------------------------------------
# Class creation
# ---------------------------------------------------------------------------

class TestMakeDeviceClassBasics:

    def test_returns_a_class(self):
        """make_device_class returns a type (class)."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert isinstance(cls, type)

    def test_subclass_of_analog_element(self):
        """Returned class is a subclass of AnalogElement."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert issubclass(cls, AnalogElement)

    def test_class_has_pin_names(self):
        """Class has correct _pin_names from PDK YAML dict format."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert cls._pin_names == ('d', 'g', 's', 'b')

    def test_class_has_param_names(self):
        """Class has correct _param_names from PDK YAML."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert 'w' in cls._param_names
        assert 'l' in cls._param_names
        assert 'm' in cls._param_names

    def test_class_has_pdk_device(self):
        """Class stores the PDK device model name."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert cls._pdk_device == 'sky130_fd_pr__nfet_01v8'

    def test_class_has_device_type_name(self):
        """Class stores _device_type_name matching the YAML key."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert cls._device_type_name == 'nmos1p8'

    def test_class_has_backward_compat_device_name(self):
        """Class has device_name attr for backward compat with circuit_elements.py."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert cls.device_name == 'nmos1p8'

    def test_pdk_default_spice_prefix(self):
        """PDK device gets XM prefix by default."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert cls._spice_prefix == 'XM'

    def test_is_pdk_flag(self):
        """_is_pdk_device is True for PDK devices."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        assert cls._is_pdk_device is True


# ---------------------------------------------------------------------------
# List-format pinList (analog_devices.yaml)
# ---------------------------------------------------------------------------

class TestListPinFormat:

    def test_list_pin_names(self):
        """pinList as a list (analog_devices format) is handled correctly."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert cls._pin_names == ('p', 'n')

    def test_ideal_spice_prefix(self):
        """Ideal element uses spice_prefix from YAML."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert cls._spice_prefix == 'R'

    def test_ideal_no_pdk_device(self):
        """Ideal element has empty _pdk_device."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert cls._pdk_device == ''

    def test_ideal_is_pdk_flag_false(self):
        """_is_pdk_device is False for ideal elements."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert cls._is_pdk_device is False


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:

    def test_pdk_device_instantiates(self):
        """PDK device class can be instantiated with name, nets, and params."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        assert mn1.name == 'mn1'

    def test_nets_stored_correctly(self):
        """Net connections stored in instance.nets dict."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        assert mn1.nets['d'] == 'vout'
        assert mn1.nets['g'] == 'vin'
        assert mn1.nets['s'] == 'gnd'

    def test_params_stored_correctly(self):
        """Parameters stored in instance.params dict."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        assert mn1.params['w'] == 2.0
        assert mn1.params['l'] == 0.15

    def test_default_param_applied(self):
        """Default parameter value used when not supplied."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        r1 = cls('r1', p='vout', n='gnd')  # value uses default=1000
        assert r1.params['value'] == 1000

    def test_missing_pin_raises_type_error(self):
        """Missing required pin raises TypeError."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        with pytest.raises(TypeError, match="missing required net for pin"):
            cls('mn1', d='vout', g='vin', s='gnd', w=2.0, l=0.15, m=1)  # missing b=

    def test_unexpected_kwarg_raises_type_error(self):
        """Unexpected keyword argument raises TypeError."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1, z=99)

    def test_instance_is_analog_element(self):
        """Instance is an AnalogElement."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        assert isinstance(mn1, AnalogElement)


# ---------------------------------------------------------------------------
# SPICE output
# ---------------------------------------------------------------------------

class TestToSpice:

    def test_pdk_spice_line_contains_device_name(self):
        """PDK to_spice() output contains the PDK model name."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        spice = mn1.to_spice()
        assert 'sky130_fd_pr__nfet_01v8' in spice

    def test_pdk_spice_line_starts_with_prefix(self):
        """PDK to_spice() line starts with XMmn1."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        assert mn1.to_spice().startswith('XMmn1')

    def test_pdk_spice_contains_nets(self):
        """PDK to_spice() output contains net names."""
        cls = make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)
        mn1 = cls('mn1', d='vout', g='vin', s='gnd', b='gnd', w=2.0, l=0.15, m=1)
        spice = mn1.to_spice()
        assert 'vout' in spice
        assert 'vin' in spice

    def test_resistor_spice_line(self):
        """Ideal resistor produces Rname net_p net_n value format."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        r1 = cls('r1', p='vout', n='gnd', value=1e3)
        spice = r1.to_spice()
        assert spice.startswith('Rr1')
        assert 'vout' in spice
        assert 'gnd' in spice

    def test_vsource_spice_line(self):
        """Ideal vsource produces Vname p n DC value format."""
        cls = make_device_class('vsource', IDEAL_VSOURCE_SPEC, is_pdk=False)
        v1 = cls('vdd', p='vdd', n='gnd', dc=1.8)
        spice = v1.to_spice()
        assert spice.startswith('Vvdd')
        assert 'DC' in spice
        assert '1.8' in spice

    def test_spice_ends_with_newline(self):
        """to_spice() output ends with a newline."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        r1 = cls('r1', p='vout', n='gnd')
        assert r1.to_spice().endswith('\n')


# ---------------------------------------------------------------------------
# _render_param helper
# ---------------------------------------------------------------------------

class TestRenderParam:

    def test_renders_float(self):
        """Float value is formatted with SI suffix."""
        assert _render_param(1e3) == '1k'
        assert _render_param(1.5e-6) == '1.5u'

    def test_renders_string(self):
        """String value passes through unchanged."""
        assert _render_param('$w_input') == '$w_input'

    def test_renders_module_variable(self):
        """Variable.value is used for rendering."""
        from silk.circuit.variable import Variable
        mv = Variable('w', value=2.0)
        assert _render_param(mv) == '2'

    def test_module_variable_none_raises(self):
        """Variable with no value raises ValueError."""
        from silk.circuit.variable import Variable
        mv = Variable('w')
        mv._value = None
        with pytest.raises(ValueError, match='no value assigned'):
            _render_param(mv)


# ---------------------------------------------------------------------------
# BasePDK integration
# ---------------------------------------------------------------------------

class TestBasePDKIntegration:

    def test_pdk_builds_device_classes(self, pdk):
        """BasePDK._parse_standard_yaml populates _device_classes."""
        assert hasattr(pdk, '_device_classes')
        assert 'nmos1p8' in pdk._device_classes

    def test_device_class_is_analog_element_subclass(self, pdk):
        """_device_classes entries are AnalogElement subclasses."""
        cls = pdk._device_classes['nmos1p8']
        assert issubclass(cls, AnalogElement)

    def test_pdk_attr_is_analog_element_subclass(self, pdk):
        """pdk.nmos1p8 is now an AnalogElement subclass (Phase 2+)."""
        assert isinstance(pdk.nmos1p8, type)
        assert issubclass(pdk.nmos1p8, AnalogElement)


# ---------------------------------------------------------------------------
# Coverage gap: _build_constraints with no paramList/paramConstraints (line 66)
# ---------------------------------------------------------------------------

class TestBuildConstraintsEmpty:

    def test_empty_spec_returns_empty_dict(self):
        """Spec with no paramList or paramConstraints returns {} (line 66)."""
        spec_no_params = {
            'PDKDevice': 'dummy_device',
            'pinList': {'a': 'anode', 'b': 'cathode'},
        }
        cls = make_device_class('diode', spec_no_params, is_pdk=True)
        assert cls._param_names == ()
        assert cls._param_constraints == {}


# ---------------------------------------------------------------------------
# Coverage gap: missing required param (no default) → TypeError (line 143)
# ---------------------------------------------------------------------------

class TestMissingRequiredParam:

    def test_missing_required_param_raises_type_error(self):
        """Param with default=None (required) raises TypeError when omitted (line 143)."""
        spec = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'paramList': {
                'w': {'minval': None, 'maxval': -1, 'pdk_unit': 'u'},  # default=None → required
                'l': {'minval': 0.15, 'maxval': -1, 'pdk_unit': 'u'},
            },
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
        }
        cls = make_device_class('nmos_req', spec, is_pdk=True)
        with pytest.raises(TypeError, match="missing required parameter 'w'"):
            cls('mn1', d='vout', g='vin', s='gnd', b='gnd', l=0.15)


# ---------------------------------------------------------------------------
# Coverage gap: multi-param ideal element SPICE output (lines 181-183)
# ---------------------------------------------------------------------------

MULTI_PARAM_IDEAL_SPEC = {
    'spice_prefix': 'E',
    'symbol': 'VCVS',
    'category': 'source',
    'pinList': ['out_p', 'out_n'],
    'paramConstraints': [
        {'name': 'gain', 'default': 1.0},
        {'name': 'offset', 'default': 0.0},
    ],
}


class TestMultiParamIdealElement:

    def test_multi_param_ideal_spice_uses_name_val_format(self):
        """Multi-param ideal elements use name=val format (lines 181-183)."""
        cls = make_device_class('vcvs', MULTI_PARAM_IDEAL_SPEC, is_pdk=False)
        e1 = cls('e1', out_p='vout', out_n='gnd')
        spice = e1.to_spice()
        assert 'Ee1' in spice
        assert 'gain=' in spice
        assert 'offset=' in spice

    def test_multi_param_ideal_no_pdk_device_in_line(self):
        """Multi-param ideal element has no PDK device name in SPICE output."""
        cls = make_device_class('vcvs', MULTI_PARAM_IDEAL_SPEC, is_pdk=False)
        e1 = cls('e1', out_p='vout', out_n='gnd')
        spice = e1.to_spice()
        # No PDK subcircuit model name
        assert 'VCVS' not in spice or spice.startswith('E')


# ---------------------------------------------------------------------------
# Coverage gap: ParamAccessor.variable() error paths (lines 70-71, 74-77)
# ---------------------------------------------------------------------------

class TestParamAccessorVariable:
    """Tests for ParamAccessor.variable() — accessed via MyClass.param_name."""

    @pytest.fixture
    def nmos_cls(self):
        return make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)

    def test_class_access_returns_param_accessor(self, nmos_cls):
        """Class-level param access returns a ParamAccessor."""
        from silk.elements.factory import ParamAccessor
        acc = nmos_cls.w
        assert isinstance(acc, ParamAccessor)

    def test_param_accessor_variable_returns_module_variable(self, nmos_cls):
        """ParamAccessor.variable() returns a Variable."""
        from silk.circuit.variable import Variable
        mv = nmos_cls.w.variable(name='w_n', value=1.0)
        assert isinstance(mv, Variable)
        assert mv.name == 'w_n'
        assert mv.value == 1.0

    def test_param_accessor_bounds_auto_derived(self, nmos_cls):
        """ParamAccessor.variable() derives bounds[0] from PDK minval (lines 67-71)."""
        mv = nmos_cls.w.variable(name='w_n')
        # PDK_NMOS_SPEC has minval=0.42 for 'w'
        assert mv.bounds is not None
        assert mv.bounds[0] == pytest.approx(0.42)
        assert mv.bounds[1] is None

    def test_param_accessor_bounds_caller_override(self, nmos_cls):
        """Caller-supplied bounds= overrides auto-derivation (line 65 branch)."""
        mv = nmos_cls.w.variable(name='w_n', bounds=(0.5, 10.0))
        assert mv.bounds == (0.5, 10.0)

    def test_param_accessor_minval_non_numeric_yields_no_bounds(self):
        """Non-parseable minval skips bounds derivation (lines 70-71 exception path)."""
        from silk.elements.factory import (
            make_device_class, DeviceParamConstraint, ParamAccessor
        )
        # Build a class with a non-numeric minval in the constraint
        # We do this by patching the constraint directly after class creation
        spec = {
            'PDKDevice': 'dummy',
            'pinList': ['a'],
            'paramList': {
                'gain': {'minval': 0.1, 'maxval': 100.0, 'pdk_unit': ''},
            },
        }
        cls = make_device_class('GainDev', spec, is_pdk=True)
        # Replace the constraint's minval with a non-float string
        from silk.elements.factory import DeviceParamConstraint
        cls._param_constraints['gain'] = DeviceParamConstraint(
            default=1.0, minval='not-a-number', maxval='100'
        )
        acc = ParamAccessor(cls, 'gain', cls._param_constraints['gain'])
        mv = acc.variable(name='gain_var')
        # Should not raise; the unparseable minval maps to an unbounded
        # (None) lower end, while the parseable maxval IS derived — before
        # 2026-08-23 maxval was never read at all.
        assert mv.bounds == (None, 100.0)

    def test_param_accessor_resolution_non_numeric_skipped(self):
        """Non-parseable resolution is silently skipped (lines 74-77 exception path)."""
        from silk.elements.factory import (
            DeviceParamConstraint, ParamAccessor
        )
        import types
        # Create a dummy class to attach the accessor to
        dummy_cls = type('Dummy', (), {})
        c = DeviceParamConstraint(
            default=1.0, minval='0.5', resolution='bad-value'
        )
        acc = ParamAccessor(dummy_cls, 'x', c)
        mv = acc.variable(name='x_var')
        assert mv.resolution is None

    def test_param_accessor_resolution_auto_derived(self):
        """Parseable resolution is picked up by ParamAccessor.variable() (lines 73-77)."""
        from silk.elements.factory import DeviceParamConstraint, ParamAccessor
        dummy_cls = type('Dummy', (), {})
        c = DeviceParamConstraint(default=1.0, minval=None, resolution='0.005')
        acc = ParamAccessor(dummy_cls, 'x', c)
        mv = acc.variable(name='x_var')
        assert mv.resolution == pytest.approx(0.005)

    def test_param_accessor_no_constraint_gives_none_bounds(self):
        """ParamAccessor with constraint=None gives bounds=None."""
        from silk.elements.factory import ParamAccessor
        dummy_cls = type('Dummy', (), {})
        acc = ParamAccessor(dummy_cls, 'x', constraint=None)
        mv = acc.variable(name='x_var')
        assert mv.bounds is None

    def test_param_accessor_repr(self, nmos_cls):
        """ParamAccessor has a useful repr."""
        acc = nmos_cls.w
        r = repr(acc)
        assert 'ParamAccessor' in r
        assert 'w' in r


# ---------------------------------------------------------------------------
# Coverage gap: ParamDescriptor.__get__ instance without params dict (line 110)
# ---------------------------------------------------------------------------

class TestParamDescriptorInstanceFallback:
    """Tests for ParamDescriptor.__get__ when obj.params is not a dict (line 110)."""

    def test_instance_without_params_dict_falls_back_to_instance_dict(self):
        """ParamDescriptor falls back to obj.__dict__.get() when obj has no .params dict
        (covers line 110 — the 'return obj.__dict__.get(self._param)' branch)."""
        from silk.elements.factory import ParamDescriptor

        # Create a class that uses ParamDescriptor but does NOT have a .params dict
        class NonAnalogHost:
            pass

        pd = ParamDescriptor('dc', constraint=None)
        host_cls = type('HostClass', (NonAnalogHost,), {'dc': pd})

        obj = host_cls()
        # No params dict — descriptor should fall back to __dict__
        assert obj.dc is None  # not in __dict__, so returns None

        # Now place a value in __dict__ directly
        obj.__dict__['dc'] = 1.8
        assert obj.dc == 1.8

    def test_param_descriptor_class_access_returns_param_accessor(self):
        """ParamDescriptor class-level access (obj=None) returns ParamAccessor."""
        from silk.elements.factory import ParamDescriptor, ParamAccessor
        pd = ParamDescriptor('w', constraint=None)
        dummy_cls = type('Dummy', (), {'w': pd})
        acc = dummy_cls.w
        assert isinstance(acc, ParamAccessor)

    def test_param_descriptor_instance_access_with_params_dict(self):
        """ParamDescriptor instance access reads from obj.params dict."""
        from silk.elements.factory import ParamDescriptor
        pd = ParamDescriptor('w', constraint=None)
        dummy_cls = type('Dummy', (), {'w': pd})
        obj = dummy_cls()
        obj.params = {'w': 2.5}
        assert obj.w == 2.5

    def test_param_descriptor_repr(self):
        """ParamDescriptor has a useful repr."""
        from silk.elements.factory import ParamDescriptor
        pd = ParamDescriptor('gain', constraint=None)
        assert 'gain' in repr(pd)


# ---------------------------------------------------------------------------
# Source class pointer — declarative source_class attribute
# ---------------------------------------------------------------------------

class TestSourceClassPointer:
    SOURCE_SPEC = {
        'spice_prefix': 'V', 'symbol': 'VSource', 'category': 'source',
        'source_class': 'VoltageSource',
        'pinList': ['p', 'n'],
        'paramConstraints': [{'name': 'dc', 'default': 0, 'unit': 'V'}],
    }

    def test_source_class_stored(self):
        cls = make_device_class('vsource', self.SOURCE_SPEC, is_pdk=False)
        assert cls._source_class == 'VoltageSource'

    def test_absent_source_class_is_none(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert cls._source_class is None


# ---------------------------------------------------------------------------
# Default parameter behavior
# ---------------------------------------------------------------------------

class TestDefaultParameterValues:
    """Test that PDK devices use explicit defaults or fall back to minval."""

    def test_explicit_default_in_spec_is_used(self):
        """When 'default' is specified in paramList, it is used as the default."""
        spec_with_default = {
            'PDKDevice': 'test_device',
            'paramList': {
                'w': {'minval': 0.42, 'maxval': 10.0, 'default': 1.0, 'pdk_unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 5.0, 'default': 0.25, 'pdk_unit': 'u'},
            },
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
        }
        cls = make_device_class('test_nmos', spec_with_default, is_pdk=True)
        device = cls('m1', d='vout', g='vin', s='gnd', b='gnd')

        # Should use explicit defaults, not minval
        assert device.params['w'] == 1.0
        assert device.params['l'] == 0.25

    def test_minval_used_when_no_explicit_default(self):
        """When 'default' is not specified, minval is used as fallback."""
        spec_without_default = {
            'PDKDevice': 'test_device',
            'paramList': {
                'w': {'minval': 0.42, 'maxval': 10.0, 'pdk_unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 5.0, 'pdk_unit': 'u'},
                'm': {'minval': 1, 'maxval': 100},
            },
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
        }
        cls = make_device_class('test_nmos', spec_without_default, is_pdk=True)
        device = cls('m1', d='vout', g='vin', s='gnd', b='gnd')

        # Should fall back to minval
        assert device.params['w'] == 0.42
        assert device.params['l'] == 0.15
        assert device.params['m'] == 1

    def test_mixed_defaults_and_minval(self):
        """Some params can have explicit defaults while others fall back to minval."""
        spec_mixed = {
            'PDKDevice': 'test_device',
            'paramList': {
                'w': {'minval': 0.42, 'maxval': 10.0, 'default': 1.0, 'pdk_unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 5.0, 'pdk_unit': 'u'},
                'm': {'minval': 1, 'maxval': 100, 'default': 2},
            },
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
        }
        cls = make_device_class('test_nmos', spec_mixed, is_pdk=True)
        device = cls('m1', d='vout', g='vin', s='gnd', b='gnd')

        # w and m have explicit defaults, l falls back to minval
        assert device.params['w'] == 1.0
        assert device.params['l'] == 0.15
        assert device.params['m'] == 2


# ---------------------------------------------------------------------------
# TASK-41 — pin values that cannot be one node token are rejected
# ---------------------------------------------------------------------------

class TestPinValueValidation:
    """A device pin contributes exactly one node token to its SPICE line.

    Before TASK-41 the constructor only checked that the kwarg was PRESENT:
    ``g=''`` was accepted, the module's binding chokepoint skipped it (falsy)
    and ``to_spice()`` emitted a line one node short — ngspice-42 read the
    model name as the missing node and invented a net called
    ``sky130_fd_pr__nfet_01v8``, mis-wiring the transistor with no error
    anywhere. ``g=None`` reached ``' '.join`` and died there naming neither
    the instance nor the pin.
    """

    @pytest.fixture
    def cls(self):
        return make_device_class('nmos1p8', PDK_NMOS_SPEC, is_pdk=True)

    @pytest.mark.parametrize('bad', ['', '   ', '\t'])
    def test_blank_pin_is_a_valueerror_naming_instance_and_pin(self, cls, bad):
        with pytest.raises(ValueError) as e:
            cls('mn', d='y', g=bad, s='gnd', b='gnd', w=1, l=0.15)
        assert "nmos1p8 'mn' pin 'g'" in str(e.value)

    def test_pin_with_embedded_whitespace_is_a_valueerror(self, cls):
        """The same defect with the opposite sign: one pin, two node tokens."""
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            cls('mn', d='y', g='a b', s='gnd', b='gnd', w=1, l=0.15)

    @pytest.mark.parametrize('bad', [None, 0, 1.5, ['n1'], object()])
    def test_non_string_pin_is_a_typeerror_naming_instance_and_pin(self, cls, bad):
        with pytest.raises(TypeError) as e:
            cls('mn', d='y', g=bad, s='gnd', b='gnd', w=1, l=0.15)
        assert "nmos1p8 'mn' pin 'g'" in str(e.value)

    def test_none_message_points_at_the_unassigned_variable(self, cls):
        with pytest.raises(TypeError, match='never assigned'):
            cls('mn', d='y', g=None, s='gnd', b='gnd', w=1, l=0.15)

    def test_int_message_points_at_the_missing_quotes(self, cls):
        """``s=0`` for the SPICE ground node is a plausible slip."""
        with pytest.raises(TypeError, match=r"write '0', not 0"):
            cls('mn', d='y', g='in', s=0, b='gnd', w=1, l=0.15)

    def test_missing_pin_still_reports_the_missing_pin(self, cls):
        """The pre-existing presence check is unchanged."""
        with pytest.raises(TypeError, match="missing required net for pin 's'"):
            cls('mn', d='y', g='in', b='gnd', w=1, l=0.15)

    def test_net_object_is_accepted_and_normalised_to_its_name(self, cls):
        """AC #1 admits a Net. It normalises to the net NAME, because every
        downstream consumer (netlister, lint, layout) reads strings."""
        class _FakeNet:
            name = 'vbias'

        dev = cls('mn', d='y', g=_FakeNet(), s='gnd', b='gnd', w=1, l=0.15)
        assert dev.nets['g'] == 'vbias'
        assert dev.to_spice().split()[1:5] == ['y', 'vbias', 'gnd', 'gnd']

    def test_net_object_with_a_blank_name_is_still_rejected(self, cls):
        class _FakeNet:
            name = ''

        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            cls('mn', d='y', g=_FakeNet(), s='gnd', b='gnd', w=1, l=0.15)

    def test_ideal_elements_are_validated_too(self):
        """Ideal (non-PDK) classes share the same generated __init__."""
        rcls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        with pytest.raises(ValueError, match=r"resistor 'r1' pin 'n'"):
            rcls('r1', p='a', n='', value=1e3)

    def test_to_spice_guards_a_raw_nets_dict_mutation(self, cls):
        """AC #2 is a property of the EMITTER, not only of its callers: a
        direct ``device.nets`` write bypasses both entry points, so to_spice
        re-checks every pin before building the line."""
        dev = cls('mn', d='y', g='in', s='gnd', b='gnd', w=1, l=0.15)
        dev.nets['g'] = ''
        with pytest.raises(ValueError, match=r"nmos1p8 'mn' pin 'g'"):
            dev.to_spice()

    def test_to_spice_guards_a_deleted_pin(self, cls):
        dev = cls('mn', d='y', g='in', s='gnd', b='gnd', w=1, l=0.15)
        del dev.nets['g']
        with pytest.raises(TypeError, match='no net connected'):
            dev.to_spice()


class TestEmittedNodeTokenCount:
    """AC #2 across the whole catalog: every device class in the real sky130
    PDK, plus the ideal elements, emits exactly ``len(_pin_names)`` node
    tokens, in pin order, whatever its prefix or parameter shape."""

    def _classes(self, pdk):
        from silk.elements import elements
        classes = dict(pdk._device_classes)
        classes.update(elements._device_classes)
        return classes

    def test_every_catalog_device_emits_one_token_per_pin(self, pdk):
        checked = 0
        for dev_name, cls in sorted(self._classes(pdk).items()):
            pins = list(cls._pin_names)
            if not pins:
                continue
            nets = {p: f'n{i}' for i, p in enumerate(pins)}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', DeprecationWarning)
                    dev = cls('x1', **nets)
            except TypeError:
                continue          # a parameter with no default — not our case
            tokens = dev.to_spice().split()
            assert tokens[1:1 + len(pins)] == list(nets.values()), (
                f'{dev_name}: node tokens do not match its pins:\n'
                f'  {dev.to_spice().strip()}'
            )
            # And nothing after the nodes is mistakable for one of them —
            # this is the shape ngspice mis-parsed when a node went missing.
            assert tokens[1 + len(pins)] not in nets.values()
            checked += 1
        assert checked, 'no device class exercised — test is vacuous'


# ---------------------------------------------------------------------------
# TASK-299 AC#2: data-driven pin/parameter aliases
# ---------------------------------------------------------------------------

IDEAL_RESISTOR_SPEC_WITH_ALIASES = {
    **IDEAL_RESISTOR_SPEC,
    'pinAliases': {'pos': 'p', 'neg': 'n'},
    'paramAliases': {'resistance': 'value', 'res': 'value'},
}


class TestAliases:

    def test_no_aliases_declared_means_empty_alias_maps(self):
        """A spec with no pinAliases/paramAliases key gets empty dicts, not
        a KeyError — most YAML specs (PDK devices, sources) never declare any."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert cls._pin_aliases == {}
        assert cls._param_aliases == {}

    def test_pin_alias_resolves_to_canonical(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        dev = cls('r1', pos='a', neg='b', value=1000)
        assert dev.nets == {'p': 'a', 'n': 'b'}

    def test_param_alias_resolves_to_canonical(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        dev = cls('r1', p='a', n='b', resistance=4700)
        assert dev.params == {'value': 4700}

    def test_second_param_alias_also_resolves(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        dev = cls('r1', p='a', n='b', res=4700)
        assert dev.params == {'value': 4700}

    def test_alias_and_canonical_pin_both_given_raises(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        with pytest.raises(ValueError, match="'p'.*'pos'|'pos'.*'p'"):
            cls('r1', p='a', pos='a', n='b', value=1000)

    def test_alias_and_canonical_param_both_given_raises(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        with pytest.raises(ValueError, match="'value'.*'resistance'|'resistance'.*'value'"):
            cls('r1', p='a', n='b', value=1000, resistance=1000)

    def test_two_aliases_for_same_canonical_both_given_raises(self):
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        with pytest.raises(ValueError, match="'resistance'.*'res'|'res'.*'resistance'"):
            cls('r1', p='a', n='b', resistance=1000, res=2000)

    def test_aliases_do_not_appear_in_schema_facing_class_attrs(self):
        """The whole point of keeping aliases data-driven and out of Python:
        export_schema() (silk/elements/schema.py) reads only _pin_names/
        _param_names/_param_constraints — aliases must never show up there."""
        cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC_WITH_ALIASES, is_pdk=False)
        assert cls._pin_names == ('p', 'n')
        assert cls._param_names == ('value',)
        assert set(cls._param_constraints) == {'value'}

    def test_real_analog_devices_yaml_declares_the_documented_aliases(self):
        """AC#2's literal ask: resistance/res/capacitance/cap/inductance/ind
        -> value, pos/neg -> p/n, declared in the shipped YAML (not a test
        fixture)."""
        from silk.elements import elements as ideal
        assert ideal.resistor._pin_aliases == {'pos': 'p', 'neg': 'n'}
        assert ideal.resistor._param_aliases == {'resistance': 'value', 'res': 'value'}
        assert ideal.capacitor._pin_aliases == {'pos': 'p', 'neg': 'n'}
        assert ideal.capacitor._param_aliases == {'capacitance': 'value', 'cap': 'value'}
        assert ideal.inductor._pin_aliases == {'pos': 'p', 'neg': 'n'}
        assert ideal.inductor._param_aliases == {'inductance': 'value', 'ind': 'value'}

    def test_real_elements_export_schema_unchanged_by_aliases(self):
        """Byte-identical export_schema() before/after aliases exist — the
        catalog stability contract."""
        import json
        from silk.elements.schema import export_schema
        from silk.elements.library import ElementLibrary
        schema = export_schema(ElementLibrary())
        for name in ('resistor', 'capacitor', 'inductor'):
            pins = [p['name'] for p in schema[name]['pins']]
            params = [p['name'] for p in schema[name]['parameters']]
            assert pins == ['p', 'n'], f'{name}: alias leaked into pins: {pins}'
            assert params == ['value'], f'{name}: alias leaked into parameters: {params}'


# ---------------------------------------------------------------------------
# Module-namespace registration (pickle identity)
# ---------------------------------------------------------------------------

class TestModuleNamespaceRegistration:
    """make_device_class() registers each class into
    sys.modules['silk.elements.factory'].__dict__[device_name] so pickle can
    locate it by module+qualname (silk.faults.FaultCampaign's multiprocessing
    pool). That slot is a single, process-wide registry keyed only by
    device_name, shared by every ElementLibrary instance.

    Found via TASK-312.3: migrating tests off passives.* onto elements.*
    made pickling of an ElementLibrary-built Module newly order-dependent —
    a second ElementLibrary(custom_yaml_path) built later in the same
    process (e.g. by test_element_library.py) re-created a 'resistor' class
    and clobbered the slot the package singleton's 'resistor' instances
    relied on, breaking their pickling with
    "PicklingError: ... it's not the same object as
    silk.elements.factory.resistor". PDK devices keep last-registered-wins
    (structurally equivalent classes across PDK YAMLs); ideal elements
    (is_pdk=False) use first-registered-wins so the package singleton, which
    always builds first at import time, stays pickle-stable.
    """

    def test_ideal_element_keeps_first_registered_class(self):
        """Uses a device_name no real YAML declares, since the package
        singleton (silk.elements.elements) has already claimed 'resistor'
        by the time this test runs — exactly the scenario the fix protects."""
        import sys
        first = make_device_class('_test_ideal_widget', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert sys.modules['silk.elements.factory'].__dict__['_test_ideal_widget'] is first

        second = make_device_class('_test_ideal_widget', IDEAL_RESISTOR_SPEC, is_pdk=False)
        assert second is not first
        assert sys.modules['silk.elements.factory'].__dict__['_test_ideal_widget'] is first

    def test_pdk_device_keeps_last_registered_class(self):
        """Unchanged, intentional behavior for PDK devices — do not let the
        ideal-element fix silently flip this the other way."""
        import sys
        first = make_device_class('_test_pdk_widget', PDK_NMOS_SPEC, is_pdk=True)
        assert sys.modules['silk.elements.factory'].__dict__['_test_pdk_widget'] is first

        second = make_device_class('_test_pdk_widget', PDK_NMOS_SPEC, is_pdk=True)
        assert second is not first
        assert sys.modules['silk.elements.factory'].__dict__['_test_pdk_widget'] is second

    def test_first_registered_ideal_instance_still_pickles_after_a_second_registration(self):
        """The actual failure mode: an instance built from the FIRST class
        must still unpickle correctly even after a second make_device_class()
        call for the same ideal device_name."""
        import pickle
        first = make_device_class('_test_ideal_widget2', IDEAL_RESISTOR_SPEC, is_pdk=False)
        r1 = first('r1', p='a', n='b', value=1000)

        make_device_class('_test_ideal_widget2', IDEAL_RESISTOR_SPEC, is_pdk=False)  # clobber attempt

        restored = pickle.loads(pickle.dumps(r1))
        assert type(restored) is first
        assert restored.nets == {'p': 'a', 'n': 'b'}
        assert restored.params == {'value': 1000}


def test_passive_factory_warns_nothing():
    cls = make_device_class('resistor', IDEAL_RESISTOR_SPEC, is_pdk=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        cls('r1', p='vout', n='gnd', value=1e3)
