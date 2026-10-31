"""
Tests for PyModules.Module — subcircuit definition, element management,
netlist generation, and hierarchical composition.
"""

import os
from collections.abc import Mapping
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from silk.circuit.modules import Module, ModulePin
from silk.circuit.variable import Variable
from silk import sources
from silk.exceptions import CircuitError
from silk.simulation.analysis import Transient, AC, DC, Noise


# ---------------------------------------------------------------------------
# Fixture with all params (w, l, m) set — needed for netlist generation
# ---------------------------------------------------------------------------

@pytest.fixture
def full_param_module(pdk, tmp_dir):
    """Module with all device parameters set including 'm'.

    Ground port named 'vss': this is a ``.subckt`` interface, and since
    TASK-36 a port named 'gnd' is rejected at emission (ngspice folds the
    port name onto global node 0, silently discarding the parent's wiring).
    """
    mod = Module('PDK_skywater130', tmp_dir, 'mod_full.cir', pdk_details=pdk)
    mod.define_module_name('inv_cell')
    mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'vss': None})

    mod.add(pdk.nmos1p8('mn', d='out', g='in', s='vss', b='vss', w=1, l=0.15, m=1))
    mod.add(pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2, l=0.15, m=1))
    return mod


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

class TestPyModuleConstructor:

    def test_instance_attributes(self, pdk, tmp_dir):
        """Constructor sets device_name, dir_folder, and empty dicts."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)

        assert mod.device_name == 'module'
        assert mod.dir_folder == tmp_dir
        assert mod.module_name is None
        assert mod.pin_list is None
        assert isinstance(mod.elem_dict, Mapping)
        assert len(mod.elem_dict) == 0

    def test_pdk_stored(self, pdk, tmp_dir):
        """PDK details are stored on the instance."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        assert mod.pdk_details is pdk


# ---------------------------------------------------------------------------
# define_module_name tests
# ---------------------------------------------------------------------------

class TestDefineModuleName:

    def test_sets_module_name(self, pdk, tmp_dir):
        """define_module_name() sets the module_name attribute."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_name('my_inv')
        assert mod.module_name == 'my_inv'

    def test_overwrite_module_name(self, pdk, tmp_dir):
        """Calling define_module_name() again overwrites the previous name."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_name('first')
        mod.define_module_name('second')
        assert mod.module_name == 'second'


# ---------------------------------------------------------------------------
# define_module_nets tests
# ---------------------------------------------------------------------------

class TestDefineModuleNets:

    def test_pin_list_populated(self, pdk, tmp_dir):
        """define_module_nets() creates pin_list with correct port names."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_nets({'in': None, 'out': None, 'vdd': None, 'gnd': None})

        assert mod.pin_list is not None
        assert set(mod.pin_list.keys()) == {'in', 'out', 'vdd', 'gnd'}

    def test_pins_are_modulepin_instances(self, pdk, tmp_dir):
        """Each pin in pin_list is a ModulePin dataclass."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_nets({'a': None, 'b': None})

        for pin in mod.pin_list.values():
            assert isinstance(pin, ModulePin)
            assert pin.net is None  # no net assigned yet

    def test_pin_name_matches_key(self, pdk, tmp_dir):
        """ModulePin.name matches its dictionary key."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_nets({'x': None})
        assert mod.pin_list['x'].name == 'x'


# ---------------------------------------------------------------------------
# add_element inside module tests
# ---------------------------------------------------------------------------

class TestModuleAddElement:

    def test_element_in_module_elem_dict(self, basic_module, pdk):
        """Elements added to the module appear in its elem_dict."""
        # basic_module already has 'mn' and 'mp'
        assert 'mn' in basic_module.elem_dict
        assert 'mp' in basic_module.elem_dict

    def test_element_type_is_device(self, basic_module):
        """Elements from PDK are typed as 'd' (device)."""
        assert basic_module.elem_dict['mn']['type'] == 'd'
        assert basic_module.elem_dict['mp']['type'] == 'd'

    def test_add_additional_element(self, basic_module, pdk):
        """Can add more elements after initial fixture setup."""
        basic_module.add(pdk.nmos1p8('mn2', d='out', g='in', s='gnd', b='gnd', w=0.5, l=0.15))
        assert 'mn2' in basic_module.elem_dict


# ---------------------------------------------------------------------------
# generate_netlist_entry tests
# ---------------------------------------------------------------------------

class TestGenerateNetlistEntry:

    def test_starts_with_x(self, basic_module):
        """generate_netlist_entry() output starts with 'x' (subcircuit instance prefix)."""
        # First connect the module pins to nets
        for pin_name in basic_module.pin_list:
            basic_module.pin_list[pin_name].net = pin_name

        entry = basic_module.generate_netlist_entry('inst1')
        assert entry.startswith('x')

    def test_contains_instance_name(self, basic_module):
        """Output contains the instance name."""
        for pin_name in basic_module.pin_list:
            basic_module.pin_list[pin_name].net = pin_name

        entry = basic_module.generate_netlist_entry('inst1')
        assert 'inst1' in entry

    def test_contains_module_name(self, basic_module):
        """Output contains the module name."""
        for pin_name in basic_module.pin_list:
            basic_module.pin_list[pin_name].net = pin_name

        entry = basic_module.generate_netlist_entry('inst1')
        assert basic_module.module_name in entry

    def test_contains_net_names(self, basic_module):
        """Output contains the connected net names."""
        basic_module.pin_list['in'].net = 'sig_in'
        basic_module.pin_list['out'].net = 'sig_out'
        basic_module.pin_list['vdd'].net = 'vdd'
        basic_module.pin_list['vss'].net = 'gnd'

        entry = basic_module.generate_netlist_entry('u1')
        assert 'sig_in' in entry
        assert 'sig_out' in entry

    def test_undefined_pin_exits(self, basic_module):
        """If any pin has no net assigned, generate_netlist_entry raises CircuitError."""
        # Leave pins unconnected (net = None)
        basic_module.pin_list['in'].net = None
        basic_module.pin_list['out'].net = 'out'
        basic_module.pin_list['vdd'].net = 'vdd'
        basic_module.pin_list['vss'].net = 'gnd'

        with pytest.raises(CircuitError):
            basic_module.generate_netlist_entry('inst_bad')


# ---------------------------------------------------------------------------
# generate_netlist(subckt=True) tests
# ---------------------------------------------------------------------------

class TestGenerateNetlistSubckt:

    def test_contains_subckt_keyword(self, full_param_module):
        """Output contains '.subckt' when subckt=True."""
        output = full_param_module.generate_netlist(subckt=True)
        assert '.subckt' in output

    def test_contains_ends_keyword(self, full_param_module):
        """Output contains '.ends' when subckt=True."""
        output = full_param_module.generate_netlist(subckt=True)
        assert '.ends' in output

    def test_contains_module_name(self, full_param_module):
        """Output contains the module name in the .subckt line."""
        output = full_param_module.generate_netlist(subckt=True)
        assert full_param_module.module_name in output

    def test_contains_pin_names(self, full_param_module):
        """Output contains the pin names in the .subckt header."""
        output = full_param_module.generate_netlist(subckt=True)
        for pin_name in full_param_module.pin_list:
            assert pin_name in output


# ---------------------------------------------------------------------------
# generate_subckt tests
# ---------------------------------------------------------------------------

class TestGenerateSubckt:

    def test_subckt_start_line(self, basic_module):
        """generate_subckt(0) produces .subckt header with module name."""
        line = basic_module.generate_subckt(0)
        assert line.startswith('.subckt')
        assert basic_module.module_name in line

    def test_subckt_end_line(self, basic_module):
        """generate_subckt(1) produces .ends line."""
        line = basic_module.generate_subckt(1)
        assert '.ends' in line


# ---------------------------------------------------------------------------
# Module used in parent PyNetlist tests
# ---------------------------------------------------------------------------

class TestModuleInParentNetlist:

    def test_module_instance_in_parent_elem_dict(self, basic_module, pdk, tmp_dir):
        """After adding a module to a PyNetlist, elem_dict contains the instance."""
        from silk.circuit.modules import Module

        # Connect module pins
        for pin_name in basic_module.pin_list:
            basic_module.pin_list[pin_name].net = pin_name

        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        parent.add_net('in')
        parent.add_net('out')
        parent.add_net('vdd')
        parent.add_net('gnd')

        parent.add_module(
            basic_module, 'u1',
            nets={'in': 'in', 'out': 'out', 'vdd': 'vdd', 'vss': 'gnd'},
        )

        assert 'u1' in parent.elem_dict
        assert parent.elem_dict['u1']['type'] == 'm'

    def test_module_appears_in_parent_netlist_output(self, full_param_module, pdk, tmp_dir):
        """Module subcircuit definition appears in the parent's generated netlist."""
        from silk.circuit.modules import Module

        for pin_name in full_param_module.pin_list:
            full_param_module.pin_list[pin_name].net = pin_name

        parent = Module('PDK_skywater130', tmp_dir, 'parent2.cir', pdk_details=pdk)
        parent.add_net('in')
        parent.add_net('out')
        parent.add_net('vdd')
        parent.add_net('gnd')

        parent.add_module(
            full_param_module, 'u1',
            nets={'in': 'in', 'out': 'out', 'vdd': 'vdd', 'vss': 'gnd'},
        )

        parent.add_source(sources.VoltageSource('vsup', positive='vdd', negative='gnd', dc=1.8))
        parent.add_sim_type(Transient(step=1e-9, stop=1e-7))
        parent_deck_path = parent.generate_netlist(corner='tt')

        content = open(parent_deck_path).read()
        assert '.subckt' in content
        assert full_param_module.module_name in content


# ---------------------------------------------------------------------------
# ModulePin dataclass tests
# ---------------------------------------------------------------------------

class TestModulePin:

    def test_default_net_is_none(self):
        """ModulePin net defaults to None."""
        pin = ModulePin(name='a')
        assert pin.name == 'a'
        assert pin.net is None

    def test_net_assignable(self):
        """ModulePin net can be assigned."""
        pin = ModulePin(name='a')
        pin.net = 'vdd'
        assert pin.net == 'vdd'


# ---------------------------------------------------------------------------
# Variable tests
# ---------------------------------------------------------------------------



class TestVariable:

    def test_basic_creation(self):
        """Variable stores name and value."""
        mv = Variable('width', value=1.0)
        assert mv.name == 'width'
        assert mv.value == 1.0

    def test_repr(self):
        """Variable __repr__ contains name and value."""
        mv = Variable('length', value=0.15)
        r = repr(mv)
        assert 'length' in r
        assert '0.15' in r

    def test_value_setter_assigns(self):
        mv = Variable('w', value=1.0)
        mv.value = 5.0
        assert mv.value == 5.0

    def test_range_argument_removed(self):
        with pytest.raises(TypeError):
            Variable('w', 2.0, 0.5)
        with pytest.raises(TypeError):
            Variable('w', value=2.0, range=0.5)
        assert not hasattr(Variable('w', value=2.0), 'range')

    def test_sweep_attributes_removed(self):
        mv = Variable('w', value=1.0)
        assert not hasattr(mv, 'sweep_values')
        assert not hasattr(mv, 'sweep_index')

    # ------------------------------------------------------------------
    # New constructor fields
    # ------------------------------------------------------------------

    def test_bounds_stored(self):
        mv = Variable('w', value=1.0, bounds=(0.5, 4.0))
        assert mv.bounds == (0.5, 4.0)

    def test_var_type_integer(self):
        mv = Variable('nfin', value=4, var_type='integer')
        assert mv.var_type == 'integer'

    def test_distribution_log_uniform(self):
        mv = Variable('w', value=1.0, bounds=(0.1, 10.0),
                            scale='log')
        assert mv.scale == 'log'

    def test_resolution_controls_integer_assignment(self):
        mv = Variable('nfin', value=4, bounds=(1, 20),
                            var_type='integer', resolution=1)
        mv.value = 5
        assert mv.value == 5

    def test_categories_stored(self):
        mv = Variable('corner', var_type='categorical',
                            categories=['tt', 'ss', 'ff'])
        assert mv.categories == ['tt', 'ss', 'ff']

    # ------------------------------------------------------------------
    # validate()
    # ------------------------------------------------------------------

    def test_validate_within_bounds(self):
        mv = Variable('w', bounds=(0.5, 4.0))
        assert mv.validate(1.0) is True

    def test_validate_at_boundary(self):
        mv = Variable('w', bounds=(0.5, 4.0))
        assert mv.validate(0.5) is True
        assert mv.validate(4.0) is True

    def test_validate_outside_bounds(self):
        mv = Variable('w', bounds=(0.5, 4.0))
        assert mv.validate(5.0) is False

    def test_validate_no_bounds_always_true(self):
        mv = Variable('w', value=99.0)
        assert mv.validate(99.0) is True
        assert mv.validate(-1e6) is True

    def test_validate_categorical_valid(self):
        mv = Variable('corner', var_type='categorical',
                            categories=['tt', 'ss', 'ff'])
        assert mv.validate('tt') is True

    def test_validate_categorical_invalid(self):
        mv = Variable('corner', var_type='categorical',
                            categories=['tt', 'ss', 'ff'])
        assert mv.validate('fs') is False

    # ------------------------------------------------------------------
    # clip()
    # ------------------------------------------------------------------

    def test_clip_within_bounds_unchanged(self):
        mv = Variable('w', bounds=(0.5, 4.0))
        assert mv.clip(1.5) == 1.5

    def test_clip_below_min(self):
        mv = Variable('w', bounds=(0.5, 4.0))
        assert mv.clip(0.1) == 0.5

    def test_clip_above_max(self):
        mv = Variable('w', bounds=(0.5, 4.0))
        assert mv.clip(10.0) == 4.0

    def test_clip_integer_rounds(self):
        mv = Variable('nfin', bounds=(1, 20), var_type='integer')
        assert mv.clip(3.7) == 4
        assert isinstance(mv.clip(3.7), int)

    def test_clip_no_bounds_unchanged(self):
        mv = Variable('w')
        assert mv.clip(999.0) == 999.0

    # ------------------------------------------------------------------
    # sample() — count and range
    # ------------------------------------------------------------------

    def test_sample_random_count(self):
        mv = Variable('w', bounds=(1.0, 5.0))
        s = mv.sample(10, method='random')
        assert len(s) == 10

    def test_sample_random_in_bounds(self):
        mv = Variable('w', bounds=(1.0, 5.0))
        s = mv.sample(50, method='random')
        assert all(1.0 <= v <= 5.0 for v in s)

    def test_sample_grid_endpoints(self):
        mv = Variable('w', bounds=(1.0, 5.0))
        s = mv.sample(5, method='grid')
        assert s[0] == pytest.approx(1.0)
        assert s[-1] == pytest.approx(5.0)
        assert len(s) == 5

    def test_sample_log_grid_positive(self):
        mv = Variable('w', bounds=(1e-9, 1e-6), scale='log')
        s = mv.sample(4, method='grid')
        assert len(s) == 4
        assert all(9.9e-10 <= v <= 1.01e-6 for v in s)

    def test_sample_lhs_count_and_range(self):
        mv = Variable('w', bounds=(0.0, 10.0))
        s = mv.sample(8, method='lhs')
        assert len(s) == 8
        assert all(0.0 <= v <= 10.0 for v in s)

    def test_sample_halton_count_and_range(self):
        mv = Variable('w', bounds=(0.0, 1.0))
        s = mv.sample(8, method='halton')
        assert len(s) == 8
        assert all(0.0 <= v <= 1.0 for v in s)

    def test_sample_integer_returns_ints(self):
        mv = Variable('nfin', bounds=(1, 20), var_type='integer')
        s = mv.sample(10, method='random')
        assert all(isinstance(v, int) for v in s)
        assert all(1 <= v <= 20 for v in s)

    def test_sample_categorical(self):
        mv = Variable('corner', var_type='categorical',
                            categories=['tt', 'ss', 'ff'])
        s = mv.sample(6)
        assert len(s) == 6
        assert all(v in ['tt', 'ss', 'ff'] for v in s)

    def test_sample_no_bounds_raises(self):
        mv = Variable('w', value=1.0)
        with pytest.raises(ValueError, match='bounds'):
            mv.sample(5)

    def test_sample_unknown_method_raises_not_silently_random(self):
        # A mistyped method must fail loud, not degrade to random (the caller
        # asked for a specific scheme and would never see it was ignored).
        mv = Variable('w', bounds=(0.0, 10.0))
        with pytest.raises(ValueError, match='Unknown sampling method'):
            mv.sample(5, method='latin')      # typo for 'lhs'

    def test_sample_categorical_unknown_method_raises(self):
        mv = Variable('corner', var_type='categorical',
                            categories=['tt', 'ss', 'ff'])
        with pytest.raises(ValueError, match='Unknown sampling method'):
            mv.sample(3, method='latin')

    def test_sample_sobol_raises_with_halton_hint(self):
        mv = Variable('w', bounds=(0.0, 10.0))
        with pytest.raises(ValueError, match='halton'):
            mv.sample(5, method='sobol')      # renamed 2026-08-23

    def test_sample_categorical_no_categories_raises(self):
        mv = Variable('x', var_type='categorical')
        with pytest.raises(ValueError, match='categories'):
            mv.sample(3)

    def test_sample_lhs_stratified(self):
        """LHS should cover all strata — at least one point per stratum."""
        mv = Variable('w', bounds=(0.0, 1.0))
        n = 10
        s = sorted(mv.sample(n, method='lhs'))
        for i in range(n):
            lo, hi = i / n, (i + 1) / n
            assert any(lo <= v < hi + 1e-12 for v in s), f"stratum [{lo},{hi}] not covered"

    def test_sample_integer_grid_saturates(self):
        mv = Variable('nfin', bounds=(1, 5), var_type='integer', resolution=1)
        with pytest.warns(UserWarning, match='nfin.*100.*5'):
            s = mv.sample(100, method='grid')
        assert set(s) == {1, 2, 3, 4, 5}

    def test_sample_log_uniform_random(self):
        """Log-uniform random samples should cluster more at small values."""
        mv = Variable('w', bounds=(1e-3, 1.0),
                            scale='log')
        s = mv.sample(200, method='random')
        assert all(1e-3 <= v <= 1.0 for v in s)


# ---------------------------------------------------------------------------
# Resolution validation in Variable constructor
# ---------------------------------------------------------------------------

class TestVariableResolutionValidation:
    """Test that Variable validates resolution at construction time."""

    def test_resolution_zero_raises_valueerror(self):
        """resolution=0 should raise ValueError at construction."""
        with pytest.raises(ValueError, match='resolution must be None or positive'):
            Variable('w', value=None, bounds=(0.15, 20.0),
                          resolution=0, enforce_resolution=True)

    def test_resolution_negative_raises_valueerror(self):
        """Negative resolution should raise ValueError at construction."""
        with pytest.raises(ValueError, match='resolution must be None or positive'):
            Variable('w', value=None, bounds=(0.15, 20.0),
                          resolution=-0.5, enforce_resolution=True)

    def test_resolution_zero_point_zero_raises_valueerror(self):
        """resolution=0.0 should also raise ValueError."""
        with pytest.raises(ValueError, match='resolution must be None or positive'):
            Variable('l', value=1.0, bounds=(0.15, 5.0),
                          resolution=0.0, enforce_resolution=True)

    def test_resolution_positive_accepted(self):
        """Positive resolution should be accepted."""
        mv = Variable('w', value=1.0, bounds=(0.15, 20.0),
                           resolution=0.005, enforce_resolution=True)
        assert mv.resolution == 0.005
        assert mv.value == 1.0

    def test_resolution_none_accepted(self):
        """resolution=None should be accepted."""
        mv = Variable('w', value=1.0, bounds=(0.15, 20.0),
                           resolution=None, enforce_resolution=True)
        assert mv.resolution is None
        assert mv.value == 1.0

    def test_resolution_small_positive_accepted(self):
        """Very small positive resolution should be accepted."""
        mv = Variable('w', value=1.0,
                           resolution=1e-9, enforce_resolution=True)
        assert mv.resolution == 1e-9


# ---------------------------------------------------------------------------
# Test management on Module
# ---------------------------------------------------------------------------

class TestModuleTestManagement:

    def test_has_tests_initially_false(self, basic_module):
        """Module has no tests initially."""
        assert basic_module.has_tests() is False

    def test_list_tests_initially_empty(self, basic_module):
        """list_tests() returns empty list initially."""
        assert basic_module.list_tests() == []

    def test_create_test_returns_test(self, basic_module):
        """create_test() returns a ModuleTestbench instance."""
        from silk.circuit.module_test import ModuleTestbench
        test = basic_module.create_test('my_test')
        assert isinstance(test, ModuleTestbench)

    def test_has_tests_after_create(self, basic_module):
        """has_tests() returns True after create_test()."""
        basic_module.create_test('t1')
        assert basic_module.has_tests() is True

    def test_list_tests_after_create(self, basic_module):
        """list_tests() includes created test name."""
        basic_module.create_test('tran_test')
        assert 'tran_test' in basic_module.list_tests()

    def test_run_test_nonexistent_raises(self, basic_module):
        """run_test() raises ValueError for unknown test name."""
        with pytest.raises(ValueError, match="No test 'missing'"):
            basic_module.run_test('missing')

    def test_run_test_calls_test_run(self, basic_module, monkeypatch):
        """run_test() delegates to the test's run() method."""
        test_obj = basic_module.create_test('my_test')
        called = {}
        monkeypatch.setattr(test_obj, 'run', lambda **kw: called.update(kw) or {'status': 'ok'})
        result = basic_module.run_test('my_test', corner='ff')
        assert result == {'status': 'ok'}
        assert called.get('corner') == 'ff'

    def test_run_all_tests_captures_exception(self, basic_module, monkeypatch):
        """run_all_tests() catches exceptions and records error."""
        basic_module.define_module_name('test_mod')
        test_obj = basic_module.create_test('fail_test')
        monkeypatch.setattr(test_obj, 'run', lambda **kw: (_ for _ in ()).throw(RuntimeError('boom')))
        results = basic_module.run_all_tests(recursive=False)
        key = 'test_mod.fail_test'
        assert key in results
        assert 'error' in results[key]
        assert 'boom' in results[key]['error']

    def test_run_all_tests_recursive_with_submodule(self, pdk, tmp_dir, monkeypatch):
        """run_all_tests(recursive=True) includes submodule tests."""
        from silk.circuit.modules import Module

        # Create a submodule with a test
        sub = Module('PDK_skywater130', tmp_dir, 'sub.cir', pdk_details=pdk)
        sub.define_module_name('sub_mod')
        sub.define_module_nets({'a': None, 'b': None})
        test_obj = sub.create_test('sub_test')
        monkeypatch.setattr(test_obj, 'run', lambda **kw: {'passed': True})

        # Create parent and add submodule as an element
        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        parent.define_module_name('parent_mod')
        parent.define_module_nets({'a': None, 'b': None})
        parent.add_module(sub, 'xsub', nets={'a': 'net_a', 'b': 'net_b'})

        results = parent.run_all_tests(recursive=True)
        assert 'sub_mod.sub_test' in results
        assert results['sub_mod.sub_test']['passed'] is True

    def test_run_all_tests_preserves_registered_submodule(
        self, pdk, tmp_dir, monkeypatch
    ):
        """Registration-only add_submodule children retain legacy recursion."""
        from silk.circuit.modules import Module

        child = Module('PDK_skywater130', tmp_dir, 'child.cir', pdk_details=pdk)
        child.define_module_name('child_mod')
        child_test = child.create_test('child_test')
        monkeypatch.setattr(child_test, 'run', lambda **kw: {'passed': True})

        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        parent.define_module_name('parent_mod')
        parent.add_submodule('registered_child', child)

        results = parent.run_all_tests(recursive=True)

        assert results['child_mod.child_test'] == {'passed': True}

    def test_run_all_tests_disambiguates_repeated_live_instances(
        self, pdk, tmp_dir, monkeypatch
    ):
        """Repeated module types run once per live instance without overwrite."""
        from silk.circuit.modules import Module

        sub = Module('PDK_skywater130', tmp_dir, 'sub.cir', pdk_details=pdk)
        sub.define_module_name('sub_mod')
        sub.define_module_nets({'a': None, 'b': None})
        sub.create_test('sub_test')

        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        parent.define_module_name('parent_mod')
        parent.define_module_nets({'a': None, 'b': None})
        parent.add_module(sub, 'x1', nets={'a': 'net_a', 'b': 'net_b'})
        parent.add_module(sub, 'x2', nets={'a': 'net_a', 'b': 'net_b'})

        first = parent.elem_dict['x1']['device']
        second = parent.elem_dict['x2']['device']
        monkeypatch.setattr(
            first._tests['sub_test'],
            'run',
            lambda **kw: {'owner_path': first._tests['sub_test'].owner_path},
        )
        monkeypatch.setattr(
            second._tests['sub_test'],
            'run',
            lambda **kw: {'owner_path': second._tests['sub_test'].owner_path},
        )

        results = parent.run_all_tests(recursive=True)

        assert results == {
            '$/x1:sub_mod.sub_test': {'owner_path': '$/x1'},
            '$/x2:sub_mod.sub_test': {'owner_path': '$/x2'},
        }

    def test_get_all_testable_modules_with_submodule(self, pdk, tmp_dir):
        """get_all_testable_modules() finds testable submodules."""
        from silk.circuit.modules import Module

        sub = Module('PDK_skywater130', tmp_dir, 'sub.cir', pdk_details=pdk)
        sub.define_module_name('sub_mod')
        sub.define_module_nets({'a': None, 'b': None})
        sub.create_test('sub_test')

        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        parent.define_module_name('parent_mod')
        parent.define_module_nets({'a': None, 'b': None})
        parent.add_module(sub, 'xsub', nets={'a': 'net_a', 'b': 'net_b'})

        testable = parent.get_all_testable_modules(include_self=True)
        module_names = [t['module'] for t in testable]
        assert 'sub_mod' in module_names


# ---------------------------------------------------------------------------
# ModuleTestbench API — add_passive() and add_sim_type() typed form
# ---------------------------------------------------------------------------

class TestPyModuleTestAPI:

    def _make_test(self, basic_module):
        """Create a bare ModuleTestbench without adding a module under test."""
        from silk.circuit.module_test import ModuleTestbench
        t = ModuleTestbench(basic_module, 'api_test')
        t._create_netlist()
        return t

    def test_addPassive_stores_in_netlist(self, basic_module):
        """add_passive() delegates to the underlying netlist's passives_dict."""
        from silk.elements import passives
        t = self._make_test(basic_module)
        cap = passives.Capacitor('cload', pos='vout', neg='0', capacitance=100e-12)
        t.add_passive(cap)
        assert 'cload' in t.netlist.passives_dict

    def test_addPassive_returns_self(self, basic_module):
        """add_passive() returns self for chaining."""
        from silk.elements import passives
        t = self._make_test(basic_module)
        result = t.add_passive(passives.Resistor('rbias', pos='a', neg='b', resistance=1e3))
        assert result is t

    def test_addPassive_invalid_type_raises(self, basic_module):
        """add_passive() with a non-passive raises CircuitError."""
        t = self._make_test(basic_module)
        with pytest.raises(CircuitError):
            t.add_passive('not_a_passive')

    def test_addSimType_typed_transient(self, basic_module):
        """add_sim_type() accepts a Transient object."""
        from silk.simulation import analysis
        t = self._make_test(basic_module)
        t.add_sim_type(analysis.Transient(step=10e-9, stop=50e-6))
        assert 'transient' in t.netlist.sim_analysis_dict

    def test_addSimType_typed_ac(self, basic_module):
        """add_sim_type() accepts an AC object."""
        from silk.simulation import analysis
        t = self._make_test(basic_module)
        t.add_sim_type(analysis.AC(variation='dec', points=10, fstart=1, fstop=1e9))
        assert 'ac' in t.netlist.sim_analysis_dict

    def test_addSimType_returns_self(self, basic_module):
        """add_sim_type() returns self for chaining."""
        from silk.simulation import analysis
        t = self._make_test(basic_module)
        result = t.add_sim_type(analysis.Transient(step=1e-9, stop=1e-6))
        assert result is t


# ---------------------------------------------------------------------------
# add_sim_type() on a subcircuit Module must raise, not silently no-op
# ---------------------------------------------------------------------------

class TestAddSimTypeOnSubcircuit:

    def test_raises_on_subcircuit_typed_form(self, pdk, tmp_dir):
        """Calling add_sim_type() on a Module with module_name set (i.e. a
        subcircuit) must raise — previously was a silent no-op dead end."""
        from silk.simulation import analysis
        sub = Module('PDK_skywater130', tmp_dir, 'sub.cir',
                       pdk_details=pdk, module_name='sub')
        with pytest.raises(ValueError, match='subcircuit'):
            sub.add_sim_type(analysis.Transient(step=1e-9, stop=1e-6))

    def test_raises_on_subcircuit_typed_form_too(self, pdk, tmp_dir):
        """The subcircuit guard fires regardless of analysis object type."""
        from silk.simulation import analysis
        sub = Module('PDK_skywater130', tmp_dir, 'sub.cir',
                       pdk_details=pdk, module_name='sub')
        with pytest.raises(ValueError, match='subcircuit'):
            sub.add_sim_type(analysis.AC('dec', 10, 1, 1e6))

    def test_testbench_still_accepts_sim_type(self, pdk, tmp_dir):
        """Sanity guard: a top-level testbench (no module_name) is
        unaffected and still stores the analysis."""
        from silk.simulation import analysis
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        tb.add_sim_type(analysis.Transient(step=1e-9, stop=1e-6))
        assert 'transient' in tb.sim_analysis_dict


# ---------------------------------------------------------------------------
# Module delegation methods (add_net, update_element)
# ---------------------------------------------------------------------------

class TestModuleDelegation:

    def test_add_net_delegation(self, basic_module):
        """add_net() on module delegates to _element_mgr."""
        basic_module.add_net('new_net')
        assert 'new_net' in basic_module._element_mgr.net_dict


    def test_update_element_delegation(self, basic_module):
        """update_element() on module delegates to _element_mgr."""
        basic_module.update_element('mn', ['w'], [5])
        param_list = basic_module.elem_dict['mn']['device'].param_list
        assert param_list['w'].value.value == 5


# ---------------------------------------------------------------------------
# generate_netlist subckt=False guard
# ---------------------------------------------------------------------------

class TestGenerateNetlistGuard:

    def test_raises_when_subckt_false(self, pdk, tmp_dir):
        """generate_netlist() raises ValueError when called without subckt=True."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_name('inv')
        mod.define_module_nets({'a': None, 'b': None})
        with pytest.raises(ValueError, match='subckt=True'):
            mod.generate_netlist(subckt=False)

    def test_default_arg_generates_subckt(self, pdk, tmp_dir):
        """generate_netlist() with no args auto-generates subckt when module_name is set."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_name('inv')
        mod.define_module_nets({'a': None, 'b': None})
        output = mod.generate_netlist()
        assert '.subckt' in output


# ---------------------------------------------------------------------------
# Module.add() — AnalogElement instance method (lines 83-84)
# ---------------------------------------------------------------------------

class TestPyModuleAdd:

    def test_add_analog_element_instance(self, pdk, tmp_dir):
        """Module.add() stores a factory-created AnalogElement in elem_dict."""
        from silk.elements.factory import make_device_class

        mod = Module('PDK_skywater130', tmp_dir, 'mod_add.cir', pdk_details=pdk)
        mod.define_module_name('test_add')
        mod.define_module_nets({'a': None, 'b': None})

        spec = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
            'paramList': {
                'w': {'minval': 1.0, 'maxval': 10.0, 'unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 1.0, 'unit': 'u'},
            },
        }
        NMOSCls = make_device_class('nmos1p8', spec, is_pdk=True)
        mn1 = NMOSCls('mn1', d='a', g='b', s='gnd', b='gnd', w=1.0, l=0.15)

        mod.add(mn1)
        assert 'mn1' in mod.elem_dict

    def test_add_multiple_instances(self, pdk, tmp_dir):
        """Module.add(*devices) accepts multiple devices at once."""
        from silk.elements.factory import make_device_class

        mod = Module('PDK_skywater130', tmp_dir, 'mod_add2.cir', pdk_details=pdk)
        mod.define_module_name('test_add2')
        mod.define_module_nets({'a': None, 'b': None})

        spec = {
            'PDKDevice': 'sky130_fd_pr__nfet_01v8',
            'pinList': {'d': 'drain', 'g': 'gate', 's': 'source', 'b': 'bulk'},
            'paramList': {
                'w': {'minval': 1.0, 'maxval': 10.0, 'unit': 'u'},
                'l': {'minval': 0.15, 'maxval': 1.0, 'unit': 'u'},
            },
        }
        NMOSCls = make_device_class('nmos1p8', spec, is_pdk=True)
        mn1 = NMOSCls('mn_a', d='a', g='b', s='gnd', b='gnd', w=1.0, l=0.15)
        mn2 = NMOSCls('mn_b', d='b', g='a', s='gnd', b='gnd', w=2.0, l=0.15)

        mod.add(mn1, mn2)
        assert 'mn_a' in mod.elem_dict
        assert 'mn_b' in mod.elem_dict


# ---------------------------------------------------------------------------
# Module.add_passive() (lines 109-124)
# ---------------------------------------------------------------------------

class TestPyModuleAddPassive:

    def test_addPassive_stores_resistor(self, pdk, tmp_dir):
        """add_passive() on Module stores a Resistor in passives_dict."""
        from silk.elements.passives import Resistor

        mod = Module('PDK_skywater130', tmp_dir, 'mod_pass.cir', pdk_details=pdk)
        mod.define_module_name('mod_with_passives')
        mod.define_module_nets({'in': None, 'out': None})

        r1 = Resistor('rload', pos='out', neg='gnd', resistance=10e3)
        mod.add_passive(r1)
        assert 'rload' in mod.passives_dict
        assert mod.passives_dict['rload'] is r1

    def test_addPassive_auto_registers_nets(self, pdk, tmp_dir):
        """add_passive() on Module auto-registers nets in net_dict."""
        from silk.elements.passives import Capacitor

        mod = Module('PDK_skywater130', tmp_dir, 'mod_pass2.cir', pdk_details=pdk)
        mod.define_module_name('mod_cap')
        mod.define_module_nets({'in': None, 'out': None})

        cap = Capacitor('cload', pos='out', neg='gnd', capacitance=1e-12)
        mod.add_passive(cap)
        assert 'out' in mod.net_dict
        assert 'gnd' in mod.net_dict

    def test_addPassive_duplicate_name_raises(self, pdk, tmp_dir):
        """add_passive() on Module raises DuplicateNameError for duplicate name."""
        from silk.elements.passives import Resistor
        from silk.exceptions import DuplicateNameError

        mod = Module('PDK_skywater130', tmp_dir, 'mod_dup.cir', pdk_details=pdk)
        mod.define_module_name('mod_dup')
        mod.define_module_nets({'a': None})

        mod.add_passive(Resistor('r1', pos='a', neg='gnd', resistance=1e3))
        with pytest.raises(DuplicateNameError):
            mod.add_passive(Resistor('r1', pos='a', neg='gnd', resistance=2e3))

    def test_addPassive_invalid_type_raises(self, pdk, tmp_dir):
        """add_passive() with non-passive raises CircuitError."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod_bad.cir', pdk_details=pdk)
        mod.define_module_name('mod_bad')
        mod.define_module_nets({'a': None})

        with pytest.raises(CircuitError):
            mod.add_passive('not_a_passive')

    def test_addPassive_passive_in_netlist_output(self, pdk, tmp_dir):
        """add_passive() on Module includes passive in generate_netlist output."""
        from silk.elements.passives import Resistor

        mod = Module('PDK_skywater130', tmp_dir, 'mod_netlist.cir', pdk_details=pdk)
        mod.define_module_name('mod_with_r')
        mod.define_module_nets({'in': None, 'out': None})

        mod.add(pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1, l=0.15, m=1))
        mod.add_passive(Resistor('rload', pos='out', neg='gnd', resistance=10e3))

        netlist_str = mod.generate_netlist(subckt=True)
        assert 'Rrload' in netlist_str


# ---------------------------------------------------------------------------
# Variable — uncovered distribution branches
# ---------------------------------------------------------------------------

class TestVariableGridAndLogGridEdgeCases:

    def test_grid_n_equals_1_returns_lo(self):
        """_grid_samples with n=1 returns [lo] (line 457)."""
        mv = Variable('w', bounds=(1.0, 5.0))
        s = mv.sample(1, method='grid')
        assert s == [1.0]

    def test_log_grid_n_equals_1_returns_lo(self):
        """_log_grid_samples with n=1 returns [lo] (line 465)."""
        mv = Variable('w', bounds=(1e-9, 1e-6), scale='log')
        s = mv.sample(1, method='grid')
        assert len(s) == 1
        assert abs(s[0] - 1e-9) < 1e-20

    def test_log_grid_lo_zero_raises(self):
        """_log_grid_samples with lo <= 0 raises ValueError (line 463)."""
        mv = Variable('w', bounds=(0.0, 1.0), scale='log')
        with pytest.raises(ValueError, match='log'):
            mv.sample(5, method='grid')

    def test_clip_categorical_returns_val_unchanged(self):
        """clip() for CATEGORICAL type returns val unchanged (line 376)."""
        mv = Variable('corner', var_type='categorical',
                            categories=['tt', 'ss', 'ff'])
        # CATEGORICAL clip just returns the value
        assert mv.clip('tt') == 'tt'
        assert mv.clip('outside_cats') == 'outside_cats'


class TestVariableDistributions:

    def test_normal_distribution_removed(self):
        with pytest.raises(TypeError, match='distribution'):
            Variable('w', bounds=(1.0, 5.0), distribution='normal')

    def test_log_uniform_random_samples_in_range(self):
        """_random_samples with LOG_UNIFORM distribution produces values in range."""
        mv = Variable('w', bounds=(1e-9, 1e-6), scale='log')
        s = mv.sample(50, method='random')
        assert len(s) == 50
        assert all(9.9e-10 <= v <= 1.01e-6 for v in s)

    def test_lhs_log_uniform_samples_in_range(self):
        """_lhs_samples with LOG_UNIFORM distribution produces values in range (lines 476-477)."""
        mv = Variable('w', bounds=(1e-3, 1.0), scale='log')
        s = mv.sample(20, method='lhs')
        assert len(s) == 20
        assert all(9.9e-4 <= v <= 1.01 for v in s)

    def test_halton_log_uniform_samples_in_range(self):
        """_halton_samples with LOG_UNIFORM distribution produces values in range (lines 491-492)."""
        mv = Variable('w', bounds=(1e-3, 1.0), scale='log')
        s = mv.sample(8, method='halton')
        assert len(s) == 8
        assert all(9.9e-4 <= v <= 1.01 for v in s)


# ---------------------------------------------------------------------------
# Variable.validate() with categorical None categories (lines 442-444)
# ---------------------------------------------------------------------------

class TestVariableValidateCategoricalEdge:

    def test_validate_categorical_with_none_categories_returns_false(self):
        """validate() for CATEGORICAL with categories=None returns False (line 442-444)."""
        mv = Variable('corner', var_type='categorical', categories=None)
        # categories is None → condition `self.categories is not None and val in self.categories`
        # → False (short-circuits on None)
        assert mv.validate('tt') is False


# ---------------------------------------------------------------------------
# add_submodule() — cycle detection
# ---------------------------------------------------------------------------

class TestAddSubmodule:

    def test_add_submodule_succeeds_with_different_module(self, pdk, tmp_dir):
        """add_submodule() successfully adds a different module as a submodule."""
        parent = Module('PDK_skywater130', tmp_dir, 'parent.cir', pdk_details=pdk)
        parent.define_module_name('parent_module')

        child = Module('PDK_skywater130', tmp_dir, 'child.cir', pdk_details=pdk)
        child.define_module_name('child_module')

        parent.add_submodule('sub_child', child)

        assert 'sub_child' in parent.module_dict
        assert parent.module_dict['sub_child']['device'] is child

    def test_add_submodule_rejects_self_reference(self, pdk, tmp_dir):
        """add_submodule() raises ValueError when trying to add a module to itself."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_name('self_ref_module')

        with pytest.raises(ValueError, match='Cannot add module.*to itself'):
            mod.add_submodule('self_ref', mod)

    def test_add_submodule_prevents_infinite_recursion(self, pdk, tmp_dir):
        """Preventing self-reference via add_submodule() avoids run_all_tests() infinite recursion."""
        mod = Module('PDK_skywater130', tmp_dir, 'mod.cir', pdk_details=pdk)
        mod.define_module_name('test_module')

        # Create a test so has_tests() returns True (required for recursion in run_all_tests)
        mod.create_test('dummy_test')

        # Verify we can't add the module to itself
        with pytest.raises(ValueError):
            mod.add_submodule('self_ref', mod)

        # Verify module_dict is unchanged (no self-reference added)
        assert len(mod.module_dict) == 0

        # Now run_all_tests() won't infinite loop because self-reference was prevented
        results = mod.run_all_tests(recursive=True)
        # Should complete without RecursionError
        assert isinstance(results, dict)


# ---------------------------------------------------------------------------
# Old netlist methods (for coverage of legacy code paths)
# ---------------------------------------------------------------------------
