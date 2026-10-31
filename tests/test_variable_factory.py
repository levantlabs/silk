"""Tests for the ParamDescriptor / ParamAccessor attribute-access variable API.

These tests verify the new attribute-style API:

    w_n = pdk.nmos1p8.w.variable(name='w_n', value=1.0, bounds=(0.42, 20.0))
    vdd = sources.VoltageSource.dc.variable(name='vdd', value=1.8, bounds=(1.6, 2.0))

while ensuring existing string-based and instance access continues to work.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.elements.factory import ParamAccessor, ParamDescriptor, DeviceParamConstraint
from silk.circuit.variable import Variable
from silk import sources


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pdk():
    """Real BasePDK instance from conftest-compatible factory."""
    from silk.pdk.basepdk import BasePDK
    yaml_path = os.path.join(
        os.path.dirname(__file__), '..',
        'silk', 'pdk', 'data', 'skywater_130_device_details.yaml',
    )
    return BasePDK(yaml_path, library_location='/fake/path/sky130.lib.spice')


# ---------------------------------------------------------------------------
# TestParamDescriptor
# ---------------------------------------------------------------------------

class TestParamDescriptor:

    # -----------------------------------------------------------------------
    # Class-level access → ParamAccessor
    # -----------------------------------------------------------------------

    def test_class_access_returns_param_accessor(self, pdk):
        """pdk.nmos1p8.w returns a ParamAccessor, not a plain value."""
        accessor = pdk.nmos1p8.w
        assert isinstance(accessor, ParamAccessor)

    def test_param_accessor_has_correct_param_name(self, pdk):
        """ParamAccessor carries the correct param name."""
        accessor = pdk.nmos1p8.w
        assert accessor._param == 'w'

    def test_param_accessor_has_correct_owner_class(self, pdk):
        """ParamAccessor carries the owner class."""
        accessor = pdk.nmos1p8.w
        assert accessor._cls is pdk.nmos1p8

    # -----------------------------------------------------------------------
    # .variable() → Variable
    # -----------------------------------------------------------------------

    def test_variable_name_required(self, pdk):
        """Omitting name= raises TypeError immediately."""
        import pytest
        with pytest.raises(TypeError):
            pdk.nmos1p8.w.variable(value=1.0)

    def test_variable_returns_module_variable(self, pdk):
        """pdk.nmos1p8.w.variable(name="w", value=1.0) returns a Variable."""
        mv = pdk.nmos1p8.w.variable(name="w", value=1.0)
        assert isinstance(mv, Variable)

    def test_variable_default_name_is_param_name(self, pdk):
        """When name is not supplied, Variable name defaults to the param name."""
        mv = pdk.nmos1p8.w.variable(name="w", value=1.0)
        assert mv.name == 'w'

    def test_variable_explicit_name_is_used(self, pdk):
        """Explicit name= argument is stored in Variable.name."""
        mv = pdk.nmos1p8.w.variable(name='my_w', value=1.0)
        assert mv.name == 'my_w'

    def test_variable_value_is_stored(self, pdk):
        """value= argument is stored in Variable._value."""
        mv = pdk.nmos1p8.w.variable(name="w", value=2.5)
        assert mv.value == 2.5

    def test_variable_resolution_inherited_from_pdk(self, pdk):
        """.variable() inherits the catalog's manufacturing grid.

        The sky130 device-details YAML declares ``resolution: 0.005`` for w/l;
        ``.variable()`` reads ``c.resolution`` and must carry it onto the
        Variable. (Before the _build_constraints fix the field was dropped,
        so this silently came back None and PDK variables sampled off-grid.)
        """
        mv = pdk.nmos1p8.w.variable(name="w", value=1.0)
        assert mv.resolution == pytest.approx(0.005)

    def test_variable_caller_bounds_override_pdk(self, pdk):
        """Caller-supplied bounds override any PDK-derived bounds."""
        mv = pdk.nmos1p8.w.variable(name="w", value=1.0, bounds=(1.0, 10.0))
        assert mv.bounds == (1.0, 10.0)

    def test_variable_pdk_minval_becomes_lower_bound(self, pdk):
        """When no bounds are supplied, PDK minval populates bounds lower limit."""
        mv = pdk.nmos1p8.w.variable(name="w", value=1.0)
        # nmos1p8 w minval=0.42; upper=None since maxval=-1 is not a valid float bound
        assert mv.bounds is not None
        assert mv.bounds[0] == pytest.approx(0.42)

    def test_variable_l_pdk_minval(self, pdk):
        """l param minval=0.15 is picked up correctly."""
        mv = pdk.nmos1p8.l.variable(name="l", value=0.15)
        assert mv.bounds is not None
        assert mv.bounds[0] == pytest.approx(0.15)

    # -----------------------------------------------------------------------
    # Instance-level access → param value from params dict
    # -----------------------------------------------------------------------

    def test_instance_w_returns_scalar(self, pdk):
        """mn.w returns the stored param value, not a ParamAccessor."""
        mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=2.0, l=0.15)
        assert mn.w == 2.0

    def test_instance_access_is_not_param_accessor(self, pdk):
        """mn.w is NOT a ParamAccessor on an instance."""
        mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1.5, l=0.15)
        assert not isinstance(mn.w, ParamAccessor)

    def test_instance_w_after_update(self, pdk):
        """Setting mn.w = 3.0 updates params dict and mn.w reflects the new value."""
        mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.15)
        mn.w = 3.0
        assert mn.w == 3.0

    # -----------------------------------------------------------------------
    # VoltageSource class-level descriptor
    # -----------------------------------------------------------------------

    def test_voltage_source_dc_class_returns_param_accessor(self):
        """VoltageSource.dc returns a ParamAccessor at class level."""
        accessor = sources.VoltageSource.dc
        assert isinstance(accessor, ParamAccessor)

    def test_voltage_source_dc_accessor_param_name(self):
        """VoltageSource.dc accessor has param name 'dc'."""
        accessor = sources.VoltageSource.dc
        assert accessor._param == 'dc'

    def test_voltage_source_dc_variable_returns_module_variable(self):
        """VoltageSource.dc.variable(name="dc", value=1.8, bounds=(0.5, 3.3)) returns Variable."""
        mv = sources.VoltageSource.dc.variable(name="dc", value=1.8, bounds=(0.5, 3.3))
        assert isinstance(mv, Variable)
        assert mv.value == 1.8
        assert mv.bounds == (0.5, 3.3)

    def test_voltage_source_dc_variable_name(self):
        """VoltageSource.dc.variable(name='vdd') uses explicit name."""
        mv = sources.VoltageSource.dc.variable(name='vdd', value=1.8, bounds=(1.6, 2.0))
        assert mv.name == 'vdd'

    def test_voltage_source_ac_class_returns_param_accessor(self):
        """VoltageSource.ac returns a ParamAccessor at class level."""
        accessor = sources.VoltageSource.ac
        assert isinstance(accessor, ParamAccessor)

    # -----------------------------------------------------------------------
    # Source instance — dc is a plain value, not a ParamAccessor
    # -----------------------------------------------------------------------

    def test_source_instance_dc_is_scalar(self):
        """src.dc on an instance returns the scalar DC value, not a ParamAccessor."""
        src = sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
        assert src.dc == 1.8
        assert not isinstance(src.dc, ParamAccessor)

    def test_source_instance_ac_is_scalar(self):
        """src.ac on an instance returns the scalar AC value."""
        src = sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8, ac=0.1)
        assert src.ac == 0.1

    # -----------------------------------------------------------------------
    # CurrentSource descriptors
    # -----------------------------------------------------------------------

    def test_current_source_dc_class_returns_param_accessor(self):
        """CurrentSource.dc returns a ParamAccessor at class level."""
        accessor = sources.CurrentSource.dc
        assert isinstance(accessor, ParamAccessor)

    def test_current_source_dc_variable(self):
        """CurrentSource.dc.variable(name="dc", ...) returns a configured Variable."""
        mv = sources.CurrentSource.dc.variable(name='ibias', value=100e-6,
                                                bounds=(10e-6, 1e-3))
        assert isinstance(mv, Variable)
        assert mv.name == 'ibias'
        assert mv.value == pytest.approx(100e-6)

    def test_current_source_instance_dc_is_scalar(self):
        """ibias.dc on a CurrentSource instance is the plain DC float value."""
        ibias = sources.CurrentSource('ibias', positive='vb', negative='gnd', dc=100e-6)
        assert ibias.dc == pytest.approx(100e-6)
        assert not isinstance(ibias.dc, ParamAccessor)

    # -----------------------------------------------------------------------
    # __repr__
    # -----------------------------------------------------------------------

    def test_param_accessor_repr(self, pdk):
        """ParamAccessor.__repr__ includes class name and param name."""
        accessor = pdk.nmos1p8.w
        r = repr(accessor)
        assert 'ParamAccessor' in r
        assert 'nmos1p8' in r
        assert 'w' in r

    def test_param_descriptor_repr(self):
        """ParamDescriptor.__repr__ includes the param name."""
        d = ParamDescriptor('w')
        r = repr(d)
        assert 'ParamDescriptor' in r
        assert 'w' in r

    # -----------------------------------------------------------------------
    # pmos descriptors work too
    # -----------------------------------------------------------------------

    def test_pmos_w_class_access(self, pdk):
        """pdk.pmos1p8.w returns a ParamAccessor."""
        accessor = pdk.pmos1p8.w
        assert isinstance(accessor, ParamAccessor)

    def test_pmos_w_variable(self, pdk):
        """pdk.pmos1p8.w.variable(...) returns a correctly named Variable."""
        mv = pdk.pmos1p8.w.variable(name='w_p', value=2.0, bounds=(0.42, 20.0))
        assert mv.name == 'w_p'
        assert mv.value == 2.0
        assert mv.bounds == (0.42, 20.0)
