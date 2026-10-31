"""Tests for pynetlist.core.elements.base — AnalogElement base class."""

import pytest
from silk.elements.base import AnalogElement
from silk.elements import AnalogElement as AnalogElementFromPkg


class TestAnalogElementImport:

    def test_import_from_module(self):
        """AnalogElement imports from pynetlist.core.elements.base."""
        assert AnalogElement is not None

    def test_import_from_package(self):
        """AnalogElement re-exported from pynetlist.core.elements."""
        assert AnalogElementFromPkg is AnalogElement


class TestAnalogElementConstruction:

    def test_constructs_with_name(self):
        """AnalogElement can be constructed with just a name."""
        elem = AnalogElement('mn1')
        assert elem.name == 'mn1'

    def test_nets_default_empty(self):
        """nets dict is empty on construction."""
        elem = AnalogElement('mn1')
        assert elem.nets == {}

    def test_params_default_empty(self):
        """params dict is empty on construction."""
        elem = AnalogElement('mn1')
        assert elem.params == {}

    def test_layout_default_empty(self):
        """layout dict is empty on construction."""
        elem = AnalogElement('mn1')
        assert elem.layout == {}


class TestAnalogElementClassAttributes:

    def test_pin_names_default_empty(self):
        """_pin_names defaults to empty tuple."""
        assert AnalogElement._pin_names == ()

    def test_param_names_default_empty(self):
        """_param_names defaults to empty tuple."""
        assert AnalogElement._param_names == ()

    def test_spice_prefix_default(self):
        """_spice_prefix defaults to 'X'."""
        assert AnalogElement._spice_prefix == 'X'

    def test_pdk_device_default_empty(self):
        """_pdk_device defaults to empty string."""
        assert AnalogElement._pdk_device == ''

    def test_device_type_name_default_empty(self):
        """_device_type_name defaults to empty string."""
        assert AnalogElement._device_type_name == ''


class TestAnalogElementInterface:

    def test_to_spice_raises(self):
        """to_spice() raises NotImplementedError on base class."""
        elem = AnalogElement('mn1')
        with pytest.raises(NotImplementedError):
            elem.to_spice()

    def test_validate_does_not_raise(self):
        """validate() runs without error on base class (no-op)."""
        elem = AnalogElement('mn1')
        elem.validate()  # should not raise

    def test_scale_does_not_raise(self):
        """scale() runs without error on base class (stub)."""
        elem = AnalogElement('mn1')
        elem.scale(2.0)  # should not raise

    def test_repr(self):
        """__repr__ includes class name and instance name."""
        elem = AnalogElement('mn1')
        r = repr(elem)
        assert 'AnalogElement' in r
        assert 'mn1' in r


class TestAnalogElementSetattr:

    def test_direct_name_assignment(self):
        """name attribute can be set directly."""
        elem = AnalogElement('mn1')
        elem.name = 'mn2'
        assert elem.name == 'mn2'

    def test_unknown_attr_uses_object_setattr(self):
        """Unknown attribute falls through to normal object attribute."""
        elem = AnalogElement('mn1')
        elem.some_custom_attr = 42
        assert elem.some_custom_attr == 42

    def test_param_attr_routes_to_params_dict(self):
        """Setting a known param name routes to params dict on subclass."""
        # Create a minimal subclass with known _param_names
        class MyDevice(AnalogElement):
            _param_names = ('w', 'l')

        dev = MyDevice('mn1')
        dev.w = 2.0
        assert dev.params['w'] == 2.0

    def test_pin_attr_routes_to_nets_dict(self):
        """Setting a known pin name routes to nets dict on subclass."""
        class MyDevice(AnalogElement):
            _pin_names = ('d', 'g', 's', 'b')

        dev = MyDevice('mn1')
        dev.d = 'vout'
        assert dev.nets['d'] == 'vout'


# ---------------------------------------------------------------------------
# AnalogElement.variable() classmethod — lines 104-139
# ---------------------------------------------------------------------------

class TestAnalogElementVariable:
    """Tests for the AnalogElement.variable() classmethod."""

    @pytest.fixture
    def device_cls(self):
        """Minimal subclass with one constrained and one unconstrained param."""
        from silk.elements.factory import make_device_class, DeviceParamConstraint

        # Build a class with _param_constraints that include minval and resolution
        spec = {
            'PDKDevice': 'dummy_dev',
            'pinList': {'d': 'drain', 'g': 'gate'},
            'paramList': {
                'w': {'minval': 0.42, 'maxval': 20.0, 'unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 2.0, 'unit': 'u'},
            },
        }
        from silk.elements.factory import make_device_class
        cls = make_device_class('TestDev', spec, is_pdk=True)
        return cls

    def test_unknown_param_raises_value_error(self, device_cls):
        """variable() with an unknown param name raises ValueError (line 104-108)."""
        with pytest.raises(ValueError, match="unknown param 'z'"):
            device_cls.variable('z', name='z_var')

    def test_valid_param_returns_module_variable(self, device_cls):
        """variable() returns a Variable for a valid param."""
        from silk.circuit.variable import Variable
        mv = device_cls.variable('w', name='w_n', value=1.0)
        assert isinstance(mv, Variable)
        assert mv.name == 'w_n'
        assert mv.value == 1.0

    def test_bounds_auto_derived_from_constraint(self, device_cls):
        """variable() derives BOTH ends from the PDK constraint. The upper
        end used to be hardcoded None (maxval never read) — fixed
        2026-08-23; this fixture's maxval is a real 20."""
        mv = device_cls.variable('w', name='w_n')
        assert mv.bounds is not None
        assert mv.bounds[0] == pytest.approx(0.42)
        assert mv.bounds[1] == pytest.approx(20.0)

    def test_caller_supplied_bounds_override_pdk(self, device_cls):
        """Caller-supplied bounds= kwarg overrides PDK auto-derivation (line 117)."""
        mv = device_cls.variable('w', name='w_n', bounds=(1.0, 5.0))
        assert mv.bounds == (1.0, 5.0)

    def test_name_is_none_falls_back_to_param(self):
        """When name=None the param name is used as the variable name (lines 111-112)."""
        # Use the base AnalogElement class with a custom subclass that has _param_names
        class SimpleDevice(AnalogElement):
            _param_names = ('w',)
            _param_constraints = {}

        mv = SimpleDevice.variable('w', name=None)
        from silk.circuit.variable import Variable
        assert isinstance(mv, Variable)
        assert mv.name == 'w'

    def test_minval_non_numeric_skips_bounds(self):
        """minval that cannot be parsed as float yields bounds=None (lines 119-124)."""
        from silk.elements.factory import make_device_class, DeviceParamConstraint

        spec = {
            'PDKDevice': 'dummy',
            'pinList': ['a'],
            'paramList': {
                'gain': {'minval': 'N/A', 'maxval': 'N/A', 'unit': ''},
            },
        }
        cls = make_device_class('GainDev', spec, is_pdk=True)
        mv = cls.variable('gain', name='gain_var')
        assert mv.bounds is None

    def test_resolution_auto_derived(self):
        """variable() picks up resolution from DeviceParamConstraint (lines 128-136)."""
        from silk.elements.factory import make_device_class, DeviceParamConstraint
        from silk.elements.base import AnalogElement as AE

        # Inject a constraint with resolution via direct subclass manipulation
        class ResDevice(AE):
            _param_names = ('w',)
            _param_constraints = {
                'w': DeviceParamConstraint(default=1.0, minval='0.5', resolution='0.1'),
            }

        mv = ResDevice.variable('w', name='w_res')
        assert mv.resolution == pytest.approx(0.1)

    def test_resolution_non_numeric_skips(self):
        """resolution that cannot be parsed as float is silently skipped (lines 130-134)."""
        from silk.elements.factory import DeviceParamConstraint
        from silk.elements.base import AnalogElement as AE

        class ResDevice(AE):
            _param_names = ('w',)
            _param_constraints = {
                'w': DeviceParamConstraint(default=1.0, minval=None, resolution='bad'),
            }

        mv = ResDevice.variable('w', name='w_res')
        assert mv.resolution is None

    def test_caller_supplied_resolution_overrides_pdk(self):
        """Caller-supplied resolution= kwarg overrides PDK auto-derivation (line 128)."""
        from silk.elements.factory import DeviceParamConstraint
        from silk.elements.base import AnalogElement as AE

        class ResDevice(AE):
            _param_names = ('w',)
            _param_constraints = {
                'w': DeviceParamConstraint(default=1.0, minval='0.5', resolution='0.1'),
            }

        mv = ResDevice.variable('w', name='w_res', resolution=0.5)
        assert mv.resolution == pytest.approx(0.5)

    def test_no_constraint_for_param_gives_none_bounds(self):
        """When _param_constraints has no entry for param, bounds defaults to None."""
        class BareDevice(AnalogElement):
            _param_names = ('x',)
            _param_constraints = {}

        mv = BareDevice.variable('x', name='x_var')
        assert mv.bounds is None

    def test_error_message_lists_valid_params_when_none(self):
        """Error message says '(none)' when _param_names is empty (line 105)."""
        with pytest.raises(ValueError, match=r'\(none\)'):
            AnalogElement.variable('z', name='z_var')


# ---------------------------------------------------------------------------
# Phase 1 net refactor: __setattr__ propagates rewire to the manager
# ---------------------------------------------------------------------------

class TestPinAttributeRewirePropagation:

    def test_pin_attribute_assignment_before_add_is_local(self, pdk):
        """Setting a pin attribute before adding to a module only updates device.nets."""
        dev = pdk.nmos1p8(name='mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        # Not yet added — _module is None
        assert getattr(dev, '_module', None) is None
        dev.d = 'other'
        assert dev.nets['d'] == 'other'   # local update works

    def test_pin_attribute_assignment_after_add_propagates(self, pdk, empty_netlist):
        """After adding to a module, assigning a pin attribute updates Net.pins."""
        dev = pdk.nmos1p8(name='mn1', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15)
        empty_netlist.add(dev)

        # Sanity: 'out' has one pin binding
        assert len(empty_netlist.net_dict['out'].pins) == 1

        # Rewire via attribute assignment
        empty_netlist.add_net('out2')
        dev.d = 'out2'

        # Old net lost the binding
        assert len(empty_netlist.net_dict['out'].pins) == 0
        # New net has it
        out2_pins = empty_netlist.net_dict['out2'].pins
        assert len(out2_pins) == 1
        assert out2_pins[0].instance == 'mn1'
        assert out2_pins[0].port == 'd'
        assert out2_pins[0].kind == 'device'
        # Owner-side state also updated
        assert dev.nets['d'] == 'out2'


# ---------------------------------------------------------------------------
# TASK-41 — __setattr__ rejects a pin value that cannot be one node token
# ---------------------------------------------------------------------------

class TestSetattrPinValidation:
    """``mn.g = ''`` is the same slip as ``g=''`` at construction, one rewire
    later. It must fail the same way, and must not half-apply."""

    @pytest.fixture
    def dev(self):
        class MyDevice(AnalogElement):
            _pin_names = ('d', 'g', 's', 'b')

        d = MyDevice('mn1')
        d.d, d.g, d.s, d.b = 'out', 'in', 'gnd', 'gnd'
        return d

    @pytest.mark.parametrize('bad', ['', '  ', 'a b'])
    def test_blank_or_split_rewire_is_a_valueerror(self, dev, bad):
        with pytest.raises(ValueError, match=r"MyDevice 'mn1' pin 'g'"):
            dev.g = bad

    @pytest.mark.parametrize('bad', [None, 0, ['n']])
    def test_non_string_rewire_is_a_typeerror(self, dev, bad):
        with pytest.raises(TypeError, match=r"MyDevice 'mn1' pin 'g'"):
            dev.g = bad

    def test_rejected_rewire_leaves_the_old_binding_intact(self, dev):
        with pytest.raises(ValueError):
            dev.g = ''
        assert dev.nets['g'] == 'in'

    def test_net_object_rewire_normalises_to_its_name(self, dev):
        class _FakeNet:
            name = 'vbias'

        dev.g = _FakeNet()
        assert dev.nets['g'] == 'vbias'

    def test_params_are_not_affected(self, dev):
        """Only pin names go through the check — a param may legitimately be
        0, None-defaulted, or a Variable."""
        class MyDevice(AnalogElement):
            _pin_names = ('d',)
            _param_names = ('w',)

        d = MyDevice('mn1')
        d.w = 0
        assert d.params['w'] == 0
