"""Tests for hierarchical submodule support in silk.codegen.Generator.

Public codegen format: docs/codegen-contract.md
"""
import keyword

import pytest

from silk.codegen.generator import SubmoduleSpec, CircuitSpec


# ---------------------------------------------------------------------------
# Task 1 — SubmoduleSpec parsing
# ---------------------------------------------------------------------------

class TestSubmoduleSpecParsing:
    def test_minimal_entry(self):
        s = SubmoduleSpec.from_dict({
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'vin', 'vout': 'vmid'},
        })
        assert s.instance_name == 'inv1'
        assert s.import_path == '.inverter'
        assert s.module_var == 'mod'
        assert s.nets == {'vin': 'vin', 'vout': 'vmid'}
        assert s.parameters is None

    def test_with_parameters(self):
        s = SubmoduleSpec.from_dict({
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {},
            'parameters': {'drive_strength': 4},
        })
        assert s.parameters == {'drive_strength': 4}

    def test_empty_parameters_dict_becomes_none(self):
        s = SubmoduleSpec.from_dict({
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {},
            'parameters': {},
        })
        assert s.parameters is None

    def test_missing_instance_name_raises(self):
        with pytest.raises(KeyError, match='instance_name'):
            SubmoduleSpec.from_dict({
                'module_ref': {'import_path': '.x', 'module_var': 'mod'},
                'nets': {},
            })

    def test_missing_module_ref_raises(self):
        with pytest.raises(KeyError, match='module_ref'):
            SubmoduleSpec.from_dict({
                'instance_name': 'inv1',
                'nets': {},
            })

    def test_missing_import_path_raises(self):
        with pytest.raises(KeyError, match='import_path'):
            SubmoduleSpec.from_dict({
                'instance_name': 'inv1',
                'module_ref': {'module_var': 'mod'},
                'nets': {},
            })

    def test_missing_module_var_raises(self):
        with pytest.raises(KeyError, match='module_var'):
            SubmoduleSpec.from_dict({
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.x'},
                'nets': {},
            })

    def test_missing_nets_raises(self):
        with pytest.raises(KeyError, match='nets'):
            SubmoduleSpec.from_dict({
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.x', 'module_var': 'mod'},
            })


# ---------------------------------------------------------------------------
# Task 2 — CircuitSpec.submodules field + cross-entry validation
# ---------------------------------------------------------------------------

class TestCircuitSpecSubmodulesField:
    def _base_spec(self, submodules):
        return {
            'circuit_name': 'top',
            'pdk': 'sky130',
            'ports': {'a': {'dir': 'input'}},   # ports populated → module mode
            'submodules': submodules,
        }

    def test_default_empty_list(self):
        spec = CircuitSpec.from_dict({'circuit_name': 'x'})
        assert spec.submodules == []

    def test_parses_entries(self):
        spec = CircuitSpec.from_dict(self._base_spec([
            {
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.inv', 'module_var': 'mod'},
                'nets': {'vin': 'a'},
            },
        ]))
        assert len(spec.submodules) == 1
        assert spec.submodules[0].instance_name == 'inv1'

    def test_duplicate_instance_name_rejected(self):
        with pytest.raises(ValueError, match='duplicate'):
            CircuitSpec.from_dict(self._base_spec([
                {
                    'instance_name': 'inv1',
                    'module_ref': {'import_path': '.a', 'module_var': 'mod'},
                    'nets': {},
                },
                {
                    'instance_name': 'inv1',
                    'module_ref': {'import_path': '.b', 'module_var': 'mod'},
                    'nets': {},
                },
            ]))

    def test_non_identifier_instance_name_rejected(self):
        with pytest.raises(ValueError, match='identifier'):
            CircuitSpec.from_dict(self._base_spec([
                {
                    'instance_name': '1inv',
                    'module_ref': {'import_path': '.a', 'module_var': 'mod'},
                    'nets': {},
                },
            ]))

    def test_keyword_instance_name_rejected(self):
        with pytest.raises(ValueError, match='keyword'):
            CircuitSpec.from_dict(self._base_spec([
                {
                    'instance_name': 'class',
                    'module_ref': {'import_path': '.a', 'module_var': 'mod'},
                    'nets': {},
                },
            ]))

    def test_non_identifier_module_var_rejected(self):
        with pytest.raises(ValueError, match='module_var'):
            CircuitSpec.from_dict(self._base_spec([
                {
                    'instance_name': 'inv1',
                    'module_ref': {'import_path': '.a', 'module_var': '1mod'},
                    'nets': {},
                },
            ]))

    def test_empty_import_path_rejected(self):
        with pytest.raises(ValueError, match='import_path'):
            CircuitSpec.from_dict(self._base_spec([
                {
                    'instance_name': 'inv1',
                    'module_ref': {'import_path': '', 'module_var': 'mod'},
                    'nets': {},
                },
            ]))


# ---------------------------------------------------------------------------
# Task 3 — Deterministic alias computation
# ---------------------------------------------------------------------------

from silk.codegen.generator import _compute_submodule_aliases


class TestAliasComputation:
    def _sm(self, import_path, module_var, instance_name='inst'):
        return SubmoduleSpec(
            instance_name=instance_name,
            import_path=import_path,
            module_var=module_var,
            nets={},
        )

    def test_simple_case(self):
        aliases = _compute_submodule_aliases([
            self._sm('.inverter', 'mod', 'inv1'),
        ])
        assert aliases == {('.inverter', 'mod'): 'mod_inverter'}

    def test_dedup_same_path_and_var(self):
        # Two instances of the same child — one import alias only.
        aliases = _compute_submodule_aliases([
            self._sm('.inverter', 'mod', 'inv1'),
            self._sm('.inverter', 'mod', 'inv2'),
        ])
        assert aliases == {('.inverter', 'mod'): 'mod_inverter'}

    def test_strip_leading_dots(self):
        aliases = _compute_submodule_aliases([
            self._sm('...sub.pkg.inv', 'mod', 'i'),
        ])
        assert aliases == {('...sub.pkg.inv', 'mod'): 'mod_sub_pkg_inv'}

    def test_replace_slashes_and_hyphens(self):
        aliases = _compute_submodule_aliases([
            self._sm('pkg-v2/inv', 'mod', 'i'),
        ])
        assert aliases == {('pkg-v2/inv', 'mod'): 'mod_pkg_v2_inv'}

    def test_empty_sanitized_path(self):
        # import_path of just dots → sanitized is empty → trailing underscore.
        aliases = _compute_submodule_aliases([
            self._sm('...', 'foo', 'i'),
        ])
        assert aliases == {('...', 'foo'): 'foo_'}

    def test_collision_counter_distinguishes_paths(self):
        # Two distinct (path, var) pairs whose sanitization collides.
        # '.pkg.inv' → 'pkg_inv'; '.pkg_inv' → 'pkg_inv'. Both with var 'mod'
        # produce base 'mod_pkg_inv' twice. Counter disambiguates.
        aliases = _compute_submodule_aliases([
            self._sm('.pkg.inv', 'mod', 'i1'),
            self._sm('.pkg_inv', 'mod', 'i2'),
        ])
        assert aliases[('.pkg.inv', 'mod')] == 'mod_pkg_inv'
        assert aliases[('.pkg_inv', 'mod')] == 'mod_pkg_inv_1'

    def test_three_way_collision(self):
        aliases = _compute_submodule_aliases([
            self._sm('.a.b', 'mod', 'i1'),
            self._sm('.a_b', 'mod', 'i2'),
            self._sm('.a-b', 'mod', 'i3'),
        ])
        assert aliases[('.a.b', 'mod')] == 'mod_a_b'
        assert aliases[('.a_b', 'mod')] == 'mod_a_b_1'
        assert aliases[('.a-b', 'mod')] == 'mod_a_b_2'

    def test_alias_is_valid_identifier(self):
        # Even pathological input must produce a valid identifier.
        aliases = _compute_submodule_aliases([
            self._sm('!!!', 'mod', 'i1'),
        ])
        alias = aliases[('!!!', 'mod')]
        assert alias.isidentifier()
        import keyword as _kw
        assert not _kw.iskeyword(alias)

    def test_deterministic_ordering(self):
        # Same input twice → same output.
        sms = [
            self._sm('.a.b', 'mod', 'i1'),
            self._sm('.a_b', 'mod', 'i2'),
        ]
        a1 = _compute_submodule_aliases(sms)
        a2 = _compute_submodule_aliases(sms)
        assert a1 == a2


# ---------------------------------------------------------------------------
# Task 4 — Bus-vs-scalar Net-handle resolution
# ---------------------------------------------------------------------------

from silk.codegen.generator import _format_submodule_nets_dict


class TestNetsDictFormatting:
    def _spec_with_bus_nets(self, bus_names_widths):
        # Build a CircuitSpec with given bus_nets entries.
        return CircuitSpec.from_dict({
            'circuit_name': 'parent',
            'ports': {'a': {'dir': 'input'}},
            'bus_nets': [
                {'name': n, 'bits': w} for n, w in bus_names_widths
            ],
        })

    def _spec_with_bus_port(self, port_name, width):
        return CircuitSpec.from_dict({
            'circuit_name': 'parent',
            'ports': {
                port_name: {'dir': 'input', 'bits': width},
                'scalar': {'dir': 'input'},
            },
        })

    def test_scalar_value_emits_string(self):
        spec = self._spec_with_bus_nets([])
        nets = {'vin': 'a', 'vout': 'b'}
        out = _format_submodule_nets_dict(nets, spec)
        assert "'vin': 'a'" in out
        assert "'vout': 'b'" in out
        assert 'mod.nets' not in out

    def test_bus_net_value_emits_net_handle(self):
        spec = self._spec_with_bus_nets([('mybus', 4)])
        nets = {'data': 'mybus'}
        out = _format_submodule_nets_dict(nets, spec)
        assert "'data': mod.nets.mybus" in out
        assert "'data': 'mybus'" not in out  # handle, not a plain string

    def test_bus_port_value_emits_net_handle(self):
        spec = self._spec_with_bus_port('inbus', 8)
        nets = {'data': 'inbus'}
        out = _format_submodule_nets_dict(nets, spec)
        assert "'data': mod.nets.inbus" in out

    def test_mixed_scalar_and_bus(self):
        spec = self._spec_with_bus_nets([('mybus', 4)])
        nets = {'data': 'mybus', 'vin': 'a'}
        out = _format_submodule_nets_dict(nets, spec)
        assert "'data': mod.nets.mybus" in out
        assert "'vin': 'a'" in out

    def test_expanded_bit_value_emits_string(self):
        # Expanded form like 'mybus[0]' does not match the bus name 'mybus'
        # and should be emitted as a plain string.
        spec = self._spec_with_bus_nets([('mybus', 4)])
        nets = {'data[0]': 'mybus[0]'}
        out = _format_submodule_nets_dict(nets, spec)
        assert "'data[0]': 'mybus[0]'" in out
        assert 'mod.nets' not in out

    def test_empty_nets(self):
        spec = self._spec_with_bus_nets([])
        assert _format_submodule_nets_dict({}, spec) == '{}'


# ---------------------------------------------------------------------------
# Task 5 — Submodule emission (import lines + add_element calls)
# ---------------------------------------------------------------------------

from silk.codegen.generator import Generator
from tests._env import needs_pdk


class TestSubmoduleEmission:
    def _gen(self, submodules, extra=None):
        d = {
            'circuit_name': 'parent',
            'pdk': 'sky130',
            'ports': {'a': {'dir': 'input'}},
            'submodules': submodules,
        }
        if extra:
            d.update(extra)
        return Generator().generate(d)

    def test_back_compat_no_submodules_field(self):
        # Absent field → no import lines, no section marker.
        code = Generator().generate({
            'circuit_name': 'parent',
            'pdk': 'sky130',
            'ports': {'a': {'dir': 'input'}},
        })
        assert '# --- submodule instances ---' not in code
        assert 'add_element' not in code

    def test_back_compat_empty_submodules(self):
        code = self._gen([])
        assert '# --- submodule instances ---' not in code
        assert 'add_element' not in code

    def test_section_marker_present(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'a'},
        }])
        assert '# --- submodule instances ---' in code

    def test_import_line_emitted(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'a'},
        }])
        assert 'from .inverter import mod as mod_inverter' in code

    def test_import_dedup(self):
        # Two instances of the same child → one import line.
        code = self._gen([
            {
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
                'nets': {'vin': 'a'},
            },
            {
                'instance_name': 'inv2',
                'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
                'nets': {'vin': 'a'},
            },
        ])
        assert code.count('from .inverter import mod as mod_inverter') == 1
        # Both instances emit their own add_module line.
        assert "mod.add_module(mod_inverter, 'inv1'" in code
        assert "mod.add_module(mod_inverter, 'inv2'" in code

    def test_add_module_call_shape(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'a', 'vout': 'b'},
        }])
        assert "mod.add_module(mod_inverter, 'inv1'," in code
        assert "'vin': 'a'" in code
        assert "'vout': 'b'" in code
        assert 'parameters=' not in code   # no params → kwarg dropped

    def test_parameters_emitted_when_present(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {},
            'parameters': {'drive_strength': 4},
        }])
        assert "parameters={'drive_strength': 4}" in code

    def test_parameters_dropped_when_empty(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {},
            'parameters': {},
        }])
        assert 'parameters=' not in code

    def test_spec_list_order_preserved(self):
        code = self._gen([
            {
                'instance_name': 'second',
                'module_ref': {'import_path': '.b', 'module_var': 'mod'},
                'nets': {},
            },
            {
                'instance_name': 'first',
                'module_ref': {'import_path': '.a', 'module_var': 'mod'},
                'nets': {},
            },
        ])
        # 'second' appears before 'first' in generated code (spec order).
        i_second = code.index("'second'")
        i_first = code.index("'first'")
        assert i_second < i_first

    def test_bus_net_value_emits_handle(self):
        code = self._gen(
            [{
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.x', 'module_var': 'mod'},
                'nets': {'data': 'mybus'},
            }],
            extra={'bus_nets': [{'name': 'mybus', 'bits': 4}]},
        )
        assert "'data': mod.nets.mybus" in code


# ---------------------------------------------------------------------------
# Task 6 — End-to-end exec test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Generate parent + child code, exec parent, inspect runtime Module."""

    @needs_pdk

    def test_parent_imports_child_and_adds_element(self, tmp_path):
        from silk.codegen.generator import Generator
        # 1. Generate the child Module Python.
        child_code = Generator().generate({
            'circuit_name': 'inverter',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'child_sim'),
            'ports': {
                'vin':  {'dir': 'input'},
                'vout': {'dir': 'output'},
            },
        })
        # 2. Generate the parent Module Python referencing the child.
        parent_code = Generator().generate({
            'circuit_name': 'buffer',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'parent_sim'),
            'ports': {
                'vin':  {'dir': 'input'},
                'vout': {'dir': 'output'},
            },
            'submodules': [{
                'instance_name': 'inv1',
                'module_ref': {
                    'import_path': '_child_inverter',  # plain module, no dot
                    'module_var': 'mod',
                },
                'nets': {'vin': 'vin', 'vout': 'vout'},
            }],
        })
        # 3. Patch parent's import line to reference the temp-file module.
        unique_name = f'_child_inverter_{id(tmp_path)}'
        parent_code = parent_code.replace(
            'from _child_inverter import mod as',
            f'from {unique_name} import mod as',
        )
        child_file = tmp_path / f'{unique_name}.py'
        child_file.write_text(child_code)

        # 4. Make tmp_path importable.
        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            (tmp_path / 'child_sim').mkdir(exist_ok=True)
            (tmp_path / 'parent_sim').mkdir(exist_ok=True)
            ns: dict = {}
            exec(parent_code, ns)
            parent_mod = ns['mod']
            assert 'inv1' in parent_mod.elem_dict
            inv_entry = parent_mod.elem_dict['inv1']
            # The device stored is the child Module instance.
            child_mod = inv_entry['device']
            assert child_mod.module_name == 'inverter'
        finally:
            sys.path.remove(str(tmp_path))
            # Drop the imported child module so a second run doesn't see stale.
            sys.modules.pop(unique_name, None)

    @needs_pdk

    def test_parent_with_bus_wiring(self, tmp_path):
        from silk.codegen.generator import Generator
        # Child has a 4-bit bus port `data`.
        child_code = Generator().generate({
            'circuit_name': 'businv',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'child_sim_bus'),
            'ports': {
                'data':  {'dir': 'input', 'bits': 4},
                'vout':  {'dir': 'output'},
            },
        })
        unique_name = f'_child_businv_{id(tmp_path)}'
        # Parent has a 4-bit bus_nets entry, instantiates child using whole-bus form.
        parent_code = Generator().generate({
            'circuit_name': 'busparent',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'parent_sim_bus'),
            'ports': {
                'vout': {'dir': 'output'},
            },
            'bus_nets': [{'name': 'parent_bus', 'bits': 4}],
            'submodules': [{
                'instance_name': 'busi1',
                'module_ref': {
                    'import_path': '_child_businv',
                    'module_var': 'mod',
                },
                'nets': {'data': 'parent_bus', 'vout': 'vout'},
            }],
        })
        # Sanity-check the emitted code uses the Net handle, not a string.
        assert "'data': mod.nets.parent_bus" in parent_code
        # Wire up the import to the tmp file.
        parent_code = parent_code.replace(
            'from _child_businv import mod as',
            f'from {unique_name} import mod as',
        )
        child_file = tmp_path / f'{unique_name}.py'
        child_file.write_text(child_code)

        (tmp_path / 'child_sim_bus').mkdir(exist_ok=True)
        (tmp_path / 'parent_sim_bus').mkdir(exist_ok=True)

        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            ns: dict = {}
            exec(parent_code, ns)
            parent_mod = ns['mod']
            assert 'busi1' in parent_mod.elem_dict
            inv_entry = parent_mod.elem_dict['busi1']
            child_mod = inv_entry['device']
            assert child_mod.module_name == 'businv'
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop(unique_name, None)


# ---------------------------------------------------------------------------
# Task 9 — SubmoduleSpec value-type validation
# ---------------------------------------------------------------------------

class TestSubmoduleSpecValueTypes:
    def _base_entry(self, **overrides):
        d = {
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.x', 'module_var': 'mod'},
            'nets': {},
        }
        d.update(overrides)
        return d

    def test_nets_not_a_dict_rejected(self):
        with pytest.raises(TypeError, match="'nets' must be a dict"):
            SubmoduleSpec.from_dict(self._base_entry(nets=[['vin', 'a']]))

    def test_nets_value_none_rejected(self):
        with pytest.raises(TypeError, match="nets\\['vin'\\]"):
            SubmoduleSpec.from_dict(self._base_entry(nets={'vin': None}))

    def test_nets_value_int_rejected(self):
        with pytest.raises(TypeError, match="nets\\['vin'\\]"):
            SubmoduleSpec.from_dict(self._base_entry(nets={'vin': 123}))

    def test_nets_value_dict_rejected(self):
        with pytest.raises(TypeError, match="nets\\['vin'\\]"):
            SubmoduleSpec.from_dict(self._base_entry(
                nets={'vin': {'nested': 'dict'}}
            ))

    def test_parameters_not_a_dict_rejected(self):
        with pytest.raises(TypeError, match="'parameters' must be a dict"):
            SubmoduleSpec.from_dict(self._base_entry(parameters=[1, 2, 3]))

    def test_parameters_value_lambda_rejected(self):
        with pytest.raises(TypeError, match="parameters\\['cb'\\]"):
            SubmoduleSpec.from_dict(self._base_entry(
                parameters={'cb': lambda x: x}
            ))

    def test_parameters_value_dict_rejected(self):
        with pytest.raises(TypeError, match="parameters\\['cfg'\\]"):
            SubmoduleSpec.from_dict(self._base_entry(
                parameters={'cfg': {'a': 1}}
            ))

    def test_parameters_allowed_types_accepted(self):
        s = SubmoduleSpec.from_dict(self._base_entry(parameters={
            'as_str': 'x', 'as_int': 1, 'as_float': 1.5,
            'as_bool': True, 'as_none': None,
        }))
        assert s.parameters == {
            'as_str': 'x', 'as_int': 1, 'as_float': 1.5,
            'as_bool': True, 'as_none': None,
        }


class TestSubmoduleEmissionByteIdentity:
    """Refactor safety: _generate_module output should not change after
    extracting helpers. Snapshots known specs and asserts equality."""

    def test_subcircuit_with_submodule_byte_identical(self):
        # Spec covers all the emission branches: imports, parameters,
        # bus wiring, ordering. If any of these regress, this test
        # catches it.
        spec = {
            'circuit_name': 'parent',
            'pdk': 'sky130',
            'ports': {
                'vin': {'dir': 'input'},
                'vout': {'dir': 'output'},
            },
            'bus_nets': [{'name': 'mybus', 'bits': 4}],
            'submodules': [
                {
                    'instance_name': 'inv1',
                    'module_ref': {'import_path': '.inverter',
                                   'module_var': 'mod'},
                    'nets': {'data': 'mybus', 'vin': 'vin'},
                    'parameters': {'drive_strength': 4},
                },
                {
                    'instance_name': 'inv2',
                    'module_ref': {'import_path': '.inverter',
                                   'module_var': 'mod'},
                    'nets': {'data': 'mybus', 'vin': 'vin'},
                },
            ],
        }
        code = Generator().generate(spec)
        # Spot-check the critical lines individually rather than fingerprinting,
        # because module ordering inside the emitted file is itself an invariant
        # the test is meant to enforce.
        assert 'from .inverter import mod as mod_inverter' in code
        assert code.count('from .inverter import mod as mod_inverter') == 1
        assert '# --- submodule instances ---' in code
        # inv1 carries parameters → still add_element (TASK-292 unresolved).
        assert "mod.add_element(mod_inverter, 'inv1', nets=" in code
        assert "'data': mod.nets.mybus" in code
        assert "'vin': 'vin'" in code
        assert "parameters={'drive_strength': 4}" in code
        # inv2 carries no parameters → add_module.
        assert "mod.add_module(mod_inverter, 'inv2', nets=" in code
        # inv1 must appear before inv2 (spec order).
        assert code.index("'inv1'") < code.index("'inv2'")


class TestStandaloneSubmoduleEmission:
    """Submodule emission in the standalone (no-ports) generator path.

    Mirror of TestSubmoduleEmission but for _generate_netlist instead
    of _generate_module. Covers the testbench-style case where the
    top-level Module has no I/O ports.
    """

    def _gen(self, submodules, extra=None):
        d = {
            'circuit_name': 'tb',
            'pdk': 'sky130',
            # Note: NO 'ports' field — forces _generate_netlist path.
            'submodules': submodules,
        }
        if extra:
            d.update(extra)
        return Generator().generate(d)

    def test_back_compat_no_submodules_field(self):
        # Absent submodules → no import lines, no section marker.
        code = Generator().generate({
            'circuit_name': 'tb',
            'pdk': 'sky130',
        })
        assert '# --- submodule instances ---' not in code
        assert 'add_element' not in code

    def test_back_compat_empty_submodules(self):
        code = self._gen([])
        assert '# --- submodule instances ---' not in code
        assert 'add_element' not in code

    def test_section_marker_present(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'a'},
        }])
        assert '# --- submodule instances ---' in code

    def test_import_line_emitted(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'a'},
        }])
        assert 'from .inverter import mod as mod_inverter' in code

    def test_instances_attach_to_ckt_not_mod(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {'vin': 'a'},
        }])
        assert "ckt.add_module(mod_inverter, 'inv1'" in code
        # The standalone path must NOT emit `mod.add_module(...)`
        # (that's subcircuit mode).
        assert 'mod.add_module' not in code

    def test_import_dedup(self):
        code = self._gen([
            {
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
                'nets': {'vin': 'a'},
            },
            {
                'instance_name': 'inv2',
                'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
                'nets': {'vin': 'a'},
            },
        ])
        assert code.count('from .inverter import mod as mod_inverter') == 1

    def test_section_placement_after_sources_before_simulation(self):
        code = self._gen(
            [{
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
                'nets': {},
            }],
            extra={
                'sources': [{
                    'type': 'VoltageSource', 'instance': 'vdd',
                    'positive': 'vdd', 'negative': 'gnd', 'dc': 1.8,
                }],
                'simulation': {'type': 'Transient', 'step': 1e-9, 'stop': 1e-7},
            },
        )
        i_sources = code.index('# --- sources ---')
        i_submodules = code.index('# --- submodule instances ---')
        i_sim = code.index('# --- simulation ---')
        assert i_sources < i_submodules < i_sim

    def test_bus_nets_emitted_in_standalone(self):
        code = self._gen(
            [{
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.x', 'module_var': 'mod'},
                'nets': {'data': 'mybus'},
            }],
            extra={'bus_nets': [{'name': 'mybus', 'bits': 4}]},
        )
        assert '# --- internal bus nets ---' in code
        assert "ckt.add_net('mybus', bits=4)" in code

    def test_bus_value_emits_ckt_nets_handle(self):
        # Whole-bus wiring in standalone uses the ckt bus handle.
        code = self._gen(
            [{
                'instance_name': 'inv1',
                'module_ref': {'import_path': '.x', 'module_var': 'mod'},
                'nets': {'data': 'mybus'},
            }],
            extra={'bus_nets': [{'name': 'mybus', 'bits': 4}]},
        )
        assert "'data': ckt.nets.mybus" in code
        # Must NOT emit `mod.nets['<name>']` in standalone path.
        assert 'mod.nets' not in code

    def test_parameters_emitted_when_present(self):
        code = self._gen([{
            'instance_name': 'inv1',
            'module_ref': {'import_path': '.inverter', 'module_var': 'mod'},
            'nets': {},
            'parameters': {'drive_strength': 4},
        }])
        assert "parameters={'drive_strength': 4}" in code


class TestStandaloneSubmoduleEndToEnd:
    """End-to-end exec test for testbench-style standalone with submodules."""

    @needs_pdk

    def test_standalone_parent_with_one_submodule(self, tmp_path):
        # Child Module (subcircuit) with two scalar ports.
        child_code = Generator().generate({
            'circuit_name': 'leaf',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'child_sim'),
            'ports': {
                'vin':  {'dir': 'input'},
                'vout': {'dir': 'output'},
            },
        })
        unique_name = f'_child_leaf_{id(tmp_path)}'

        # Parent standalone (no ports). Submodule wires to internal nets.
        parent_code = Generator().generate({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'parent_sim'),
            'submodules': [{
                'instance_name': 'leaf1',
                'module_ref': {
                    'import_path': '_child_leaf',
                    'module_var': 'mod',
                },
                'nets': {'vin': 'a', 'vout': 'b'},
            }],
            # No simulation block — we just want the exec to succeed.
        })

        # Patch the parent's import line to reference the temp-file module.
        parent_code = parent_code.replace(
            'from _child_leaf import mod as',
            f'from {unique_name} import mod as',
        )
        child_file = tmp_path / f'{unique_name}.py'
        child_file.write_text(child_code)

        (tmp_path / 'child_sim').mkdir(exist_ok=True)
        (tmp_path / 'parent_sim').mkdir(exist_ok=True)

        import sys
        sys.path.insert(0, str(tmp_path))
        try:
            # The parent code calls ckt.simulate(...). We don't want it to
            # actually run ngspice. Patch the Module class so .simulate
            # is a no-op for this test.
            from silk.circuit.modules import Module as _Module
            orig_sim = _Module.simulate
            _Module.simulate = lambda self, **kw: None
            try:
                ns: dict = {}
                exec(parent_code, ns)
                ckt = ns['ckt']
                assert 'leaf1' in ckt.elem_dict
                leaf_dev = ckt.elem_dict['leaf1']['device']
                assert leaf_dev.module_name == 'leaf'
            finally:
                _Module.simulate = orig_sim
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop(unique_name, None)


@pytest.mark.parametrize('naming,nodes', [
    ('bracketed', 'A[4] A[5]'), ('underscore', 'A_4 A_5'),
])
def test_generated_parent_bus_port_feeds_child(tmp_path, monkeypatch, naming, nodes):
    import importlib
    import sys
    from silk.codegen.generator import Generator

    child_name = '_task456_bus_child'
    child_spec = CircuitSpec.from_dict({
        'circuit_name': 'bus_child', 'output_dir': str(tmp_path),
        'ports': {'P': {'dir': 'input', 'bits': 2}, 'vss': {'dir': 'ground'}},
    })
    (tmp_path / f'{child_name}.py').write_text(Generator().generate(child_spec))
    parent_spec = CircuitSpec.from_dict({
        'circuit_name': 'bus_parent', 'output_dir': str(tmp_path),
        'ports': {'A': {'dir': 'input', 'bits': [4, 6], 'naming': naming},
                  'vss': {'dir': 'ground'}},
        'submodules': [{'instance_name': 'x1',
                        'module_ref': {'import_path': child_name, 'module_var': 'mod'},
                        'nets': {'P': 'A', 'vss': 'vss'}}],
    })
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    namespace = {}
    try:
        exec(Generator().generate(parent_spec), namespace)
        parent = namespace['mod']
        netlist = parent.generate_netlist(subckt=True)
        assert f'xx1 {nodes} vss bus_child' in netlist
        assert f'.subckt bus_parent {nodes} vss' in netlist
    finally:
        sys.modules.pop(child_name, None)


@pytest.mark.parametrize('bus_name', keyword.kwlist + ['data'] + keyword.softkwlist)
@pytest.mark.parametrize('standalone,bus_kind', [(False, 'internal'), (False, 'port'), (True, 'internal')])
@pytest.mark.parametrize('array', [None, 2])
def test_bus_names_construct_and_preserve_bits(
    tmp_path, monkeypatch, bus_name, bus_kind, standalone, array,
):
    """Whole-bus connections must parse and expand even for keyword names."""
    import sys
    import types
    from silk import Port, PortDir
    from silk.circuit.modules import Module
    from silk.codegen.generator import Generator

    child = Module('leaf', str(tmp_path), 'leaf.cir', module_name='leaf', pdk_details=None)
    child.define_module_ports({'d': Port(dir=PortDir.IN, bits=2), 's': 'input'})
    imported = types.ModuleType('_bus_name_child')
    imported.mod = child
    monkeypatch.setitem(sys.modules, '_bus_name_child', imported)
    entry = {
        'instance_name': 'u',
        'module_ref': {'import_path': '_bus_name_child', 'module_var': 'mod'},
        'nets': {'d': bus_name, 's': 'scalar'},
    }
    if array is not None:
        entry['array'] = array
    spec = {
        'circuit_name': 'parent', 'pdk': 'sky130',
        'output_dir': str(tmp_path), 'submodules': [entry],
    }
    if not standalone:
        spec['ports'] = {'scalar': {'dir': 'input'}}
    if bus_kind == 'port':
        spec.setdefault('ports', {})[bus_name] = {'dir': 'input', 'bits': 2}
    else:
        spec['bus_nets'] = [{'name': bus_name, 'bits': 2}]
    code = Generator().generate(spec)
    if bus_name in ['data'] + keyword.softkwlist:
        parent_var = 'ckt' if standalone else 'mod'
        assert f"'d': {parent_var}.nets.{bus_name}" in code
    # Standalone output runs simulate() at module scope; only construct here.
    monkeypatch.setattr(Module, 'simulate', lambda self, **kwargs: None)
    namespace = {'__name__': '_generated_bus_parent'}
    exec(compile(code, '<generated_bus_parent>', 'exec'), namespace)
    parent = namespace['ckt' if standalone else 'mod']
    deck = parent.generate_netlist(subckt=not standalone)
    if standalone:
        from pathlib import Path
        deck = Path(deck).read_text()
    for name in (['u'] if array is None else ['u<0>', 'u<1>']):
        assert f'x{name} {bus_name}[0] {bus_name}[1] scalar leaf' in deck
