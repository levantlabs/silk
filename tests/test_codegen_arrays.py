"""Tests for the arrayed-instance emission feature.

"""
import pytest

from tests._env import needs_pdk


# ---------------------------------------------------------------------------
# Task 1 — _rewrite_template
# ---------------------------------------------------------------------------

class TestRewriteTemplate:
    def test_no_placeholder_returns_unchanged_and_false(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('vdd')
        assert s == 'vdd'
        assert has is False

    def test_bare_dollar_i_rewrites_to_curly(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('vref[$i]')
        assert s == 'vref[{i}]'
        assert has is True

    def test_parens_i_rewrites_same_as_bare(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('vref[$(i)]')
        assert s == 'vref[{i}]'
        assert has is True

    def test_addition_offset_rewrites(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('tap[$(i+1)]')
        assert s == 'tap[{i+1}]'
        assert has is True

    def test_subtraction_offset_rewrites(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('tap[$(i-3)]')
        assert s == 'tap[{i-3}]'
        assert has is True

    def test_whitespace_around_operator_tolerated(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('tap[$( i + 1 )]')
        assert s == 'tap[{i+1}]'
        assert has is True

    def test_multiple_placeholders_in_one_string(self):
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('tap_$i_to_$(i+1)')
        assert s == 'tap_{i}_to_{i+1}'
        assert has is True

    def test_dollar_outside_token_passes_through(self):
        # A literal '$' not followed by 'i' or '(' is left alone.
        from silk.codegen.generator import _rewrite_template
        s, has = _rewrite_template('cost_$5')
        assert s == 'cost_$5'
        assert has is False


# ---------------------------------------------------------------------------
# Task 2 — array + naming parsing
# ---------------------------------------------------------------------------

def _spec_with_submodule_array(array, naming=None):
    """Helper: minimal valid spec with one arrayed submodule."""
    sm = {
        'instance_name': 'U',
        'module_ref': {'import_path': '.x', 'module_var': 'mod'},
        'nets': {},
        'array': array,
    }
    if naming is not None:
        sm['naming'] = naming
    return {
        'circuit_name': 'tb',
        'pdk': 'sky130',
        'ports': {'a': {'dir': 'input'}},
        'submodules': [sm],
    }


class TestArrayFieldParsing:
    def test_submodule_array_int(self):
        from silk.codegen.generator import CircuitSpec
        spec = CircuitSpec.from_dict(_spec_with_submodule_array(8))
        sm = spec.submodules[0]
        assert sm.array == 8
        assert sm.naming == 'bracketed'   # default

    def test_submodule_array_range(self):
        from silk.codegen.generator import CircuitSpec
        spec = CircuitSpec.from_dict(_spec_with_submodule_array([1, 16]))
        sm = spec.submodules[0]
        assert sm.array == (1, 16)        # stored as tuple after normalization

    def test_submodule_naming_underscore(self):
        from silk.codegen.generator import CircuitSpec
        spec = CircuitSpec.from_dict(
            _spec_with_submodule_array(4, naming='underscore'))
        assert spec.submodules[0].naming == 'underscore'

    def test_submodule_array_absent(self):
        from silk.codegen.generator import CircuitSpec
        spec = CircuitSpec.from_dict({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'ports': {'a': {'dir': 'input'}},
            'submodules': [{
                'instance_name': 'U1',
                'module_ref': {'import_path': '.x', 'module_var': 'mod'},
                'nets': {},
            }],
        })
        assert spec.submodules[0].array is None
        assert spec.submodules[0].naming == 'bracketed'

    def test_submodule_array_zero_rejected(self):
        from silk.codegen.generator import CircuitSpec
        with pytest.raises(ValueError, match='array'):
            CircuitSpec.from_dict(_spec_with_submodule_array(0))

    def test_submodule_array_negative_rejected(self):
        from silk.codegen.generator import CircuitSpec
        with pytest.raises(ValueError, match='array'):
            CircuitSpec.from_dict(_spec_with_submodule_array(-3))

    def test_submodule_array_inverted_range_rejected(self):
        from silk.codegen.generator import CircuitSpec
        with pytest.raises(ValueError, match='array'):
            CircuitSpec.from_dict(_spec_with_submodule_array([5, 3]))

    def test_submodule_array_bad_list_length_rejected(self):
        from silk.codegen.generator import CircuitSpec
        with pytest.raises(ValueError, match='array'):
            CircuitSpec.from_dict(_spec_with_submodule_array([1, 2, 3]))

    def test_submodule_array_bad_type_rejected(self):
        from silk.codegen.generator import CircuitSpec
        with pytest.raises((TypeError, ValueError), match='array'):
            CircuitSpec.from_dict(_spec_with_submodule_array("eight"))

    def test_submodule_naming_invalid_rejected(self):
        from silk.codegen.generator import CircuitSpec
        with pytest.raises(ValueError, match='naming'):
            CircuitSpec.from_dict(_spec_with_submodule_array(4, naming='dotted'))

    def test_device_array(self):
        from silk.codegen.generator import CircuitSpec
        spec = CircuitSpec.from_dict({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'devices': [{
                'type': 'nmos1p8',
                'instance': 'M',
                'array': 4,
                'nets': {'d': 'vd[$i]'},
                'params': {'w': 2.0},
            }],
        })
        assert spec.devices[0].array == 4
        assert spec.devices[0].naming == 'bracketed'

    def test_source_array(self):
        from silk.codegen.generator import CircuitSpec
        spec = CircuitSpec.from_dict({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'sources': [{
                'type': 'VoltageSource',
                'instance': 'VS',
                'array': [0, 3],
                'positive': 'vref[$i]',
                'negative': 'gnd',
                'dc': 1.0,
            }],
        })
        assert spec.sources[0].array == (0, 3)


# ---------------------------------------------------------------------------
# Task 3 — submodule array emission
# ---------------------------------------------------------------------------

class TestSubmoduleArrayEmission:
    def _gen_subcircuit(self, sm_extra):
        from silk.codegen.generator import Generator
        sm = {
            'instance_name': 'U',
            'module_ref': {'import_path': '.child', 'module_var': 'mod'},
        }
        sm.update(sm_extra)
        return Generator().generate({
            'circuit_name': 'parent',
            'pdk': 'sky130',
            'ports': {'a': {'dir': 'input'}},
            'submodules': [sm],
        })

    def test_int_array_emits_list_comprehension(self):
        code = self._gen_subcircuit({
            'array': 4,
            'nets': {'vin': 'in[$i]'},
        })
        assert 'for i in range(4): mod.add_module(mod_child, ' in code
        assert "f'U[{i}]'" in code
        assert "f'in[{i}]'" in code or "'vin': f'in[{i}]'" in code
        assert 'for i in range(4)' in code

    def test_range_array_uses_range_a_b(self):
        code = self._gen_subcircuit({
            'array': [1, 16],
            'nets': {'vm': 'vref[$i]'},
        })
        assert 'for i in range(1, 16): mod.add_module(' in code

    def test_underscore_naming(self):
        code = self._gen_subcircuit({
            'array': 4,
            'naming': 'underscore',
            'nets': {},
        })
        assert "f'U_{i}'" in code
        assert "f'U[{i}]'" not in code

    def test_broadcast_nets_emit_plain_strings(self):
        code = self._gen_subcircuit({
            'array': 4,
            'nets': {'vdd': 'vdd', 'gnd': 'gnd', 'vp': 'vh_buf'},
        })
        assert "'vdd': 'vdd'" in code
        assert "'gnd': 'gnd'" in code
        assert "'vp': 'vh_buf'" in code
        assert "f'vdd'" not in code

    def test_arithmetic_offset_in_nets(self):
        code = self._gen_subcircuit({
            'array': 3,
            'nets': {'in': 'n[$i]', 'out': 'n[$(i+1)]'},
        })
        assert '{i+1}' in code
        assert "f'n[{i}]'" in code or '{i}' in code

    def test_mixed_broadcast_and_per_instance(self):
        code = self._gen_subcircuit({
            'array': 4,
            'nets': {'vp': 'vh_buf', 'vm': 'vref[$i]', 'vdd': 'vdd'},
        })
        assert "'vp': 'vh_buf'" in code
        assert "'vm': f'vref[{i}]'" in code
        assert "'vdd': 'vdd'" in code

    def test_non_arrayed_submodule_unchanged(self):
        code = self._gen_subcircuit({
            'nets': {'vin': 'in', 'vout': 'out'},
        })
        assert 'U = [' not in code
        assert "mod.add_module(mod_child, 'U'," in code

    def test_back_compat_no_array_in_spec_byte_identical(self):
        from silk.codegen.generator import Generator
        spec = {
            'circuit_name': 'parent',
            'pdk': 'sky130',
            'ports': {'a': {'dir': 'input'}},
            'submodules': [{
                'instance_name': 'U1',
                'module_ref': {'import_path': '.child', 'module_var': 'mod'},
                'nets': {'vin': 'in', 'vout': 'out'},
            }],
        }
        code = Generator().generate(spec)
        assert "U1 = " not in code
        assert "mod.add_module(mod_child, 'U1'," in code


# ---------------------------------------------------------------------------
# Task 4 — device + ideal_element array emission
# ---------------------------------------------------------------------------

class TestDeviceArrayEmission:
    def _gen_standalone(self, devices):
        from silk.codegen.generator import Generator
        return Generator().generate({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'devices': devices,
        })

    def test_device_array_emits_comprehension(self):
        code = self._gen_standalone([{
            'type': 'nmos1p8',
            'instance': 'M',
            'array': 4,
            'nets': {'d': 'vd[$i]', 'g': 'vin', 's': 'gnd', 'b': 'gnd'},
            'params': {'w': 2.0, 'l': 0.18},
        }])
        assert 'ckt.add(*[pdk.nmos1p8(' in code
        assert "f'M[{i}]'" in code
        assert "d=f'vd[{i}]'" in code or "f'vd[{i}]'" in code
        assert "g='vin'" in code              # broadcast
        assert 'for i in range(4)' in code
        # Inline splat, no variable bound to the instance name
        assert 'M = ' not in code

    def test_device_array_underscore(self):
        code = self._gen_standalone([{
            'type': 'nmos1p8',
            'instance': 'M',
            'array': 3,
            'naming': 'underscore',
            'nets': {'d': 'vd[$i]'},
            'params': {'w': 2.0},
        }])
        assert "f'M_{i}'" in code
        assert "f'M[{i}]'" not in code

    def test_non_array_devices_unchanged(self):
        code = self._gen_standalone([{
            'type': 'nmos1p8',
            'instance': 'm1',
            'nets': {'d': 'vd', 'g': 'vin', 's': 'gnd', 'b': 'gnd'},
            'params': {'w': 2.0},
        }])
        # Inline add; the instance name is never a variable
        assert "ckt.add(pdk.nmos1p8('m1'," in code
        assert 'm1 = ' not in code
        assert 'M = [' not in code

    def test_mixed_flat_and_array_devices(self):
        code = self._gen_standalone([
            {'type': 'nmos1p8', 'instance': 'm1',
             'nets': {'d': 'vd', 'g': 'vin', 's': 'gnd', 'b': 'gnd'},
             'params': {'w': 2.0}},
            {'type': 'nmos1p8', 'instance': 'M',
             'array': 3,
             'nets': {'d': 'vd[$i]'},
             'params': {'w': 2.0}},
        ])
        # Both appear in `# --- PDK devices ---` section
        assert "ckt.add(pdk.nmos1p8('m1'," in code
        assert 'ckt.add(*[pdk.nmos1p8(' in code
        assert 'm1 = ' not in code and 'M = ' not in code

    def test_arithmetic_offset_in_device_nets(self):
        code = self._gen_standalone([{
            'type': 'nmos1p8',
            'instance': 'R',
            'array': 4,
            'nets': {'d': 'n[$i]', 'g': 'n[$(i+1)]', 's': 'gnd', 'b': 'gnd'},
            'params': {'w': 2.0},
        }])
        # Either {i+1} or {i + 1} — accept the regex-normalized form
        assert '{i+1}' in code


class TestIdealElementArrayEmission:
    def _gen_standalone(self, ideals):
        from silk.codegen.generator import Generator
        return Generator().generate({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'ideal_elements': ideals,
        })

    def test_ideal_element_array(self):
        code = self._gen_standalone([{
            'type': 'resistor',
            'instance': 'R',
            'array': 16,
            'nets': {'p': 'tap[$i]', 'n': 'tap[$(i+1)]'},
            'params': {'value': 10e3},
        }])
        assert 'ckt.add(*[elements.resistor(' in code
        assert "f'R[{i}]'" in code
        assert 'for i in range(16)' in code
        assert 'R = ' not in code


# ---------------------------------------------------------------------------
# Task 5 — source array emission
# ---------------------------------------------------------------------------

class TestSourceArrayEmission:
    def _gen(self, sources):
        from silk.codegen.generator import Generator
        return Generator().generate({
            'circuit_name': 'tb',
            'pdk': 'sky130',
            'sources': sources,
        })

    def test_source_array_emits_list_then_for_loop(self):
        code = self._gen([{
            'type': 'VoltageSource',
            'instance': 'VS',
            'array': 4,
            'positive': 'vref[$i]',
            'negative': 'gnd',
            'dc': 1.0,
        }])
        assert 'ckt.add(*[sources.vdc(' in code
        assert "f'VS[{i}]'" in code
        assert "positive=f'vref[{i}]'" in code
        assert "negative='gnd'" in code
        assert 'for i in range(4)' in code
        # Registered inline via a splat; no variable, no for-loop (TASK-351)
        assert 'VS = ' not in code
        assert 'for v in' not in code

    def test_source_array_underscore_naming(self):
        code = self._gen([{
            'type': 'VoltageSource',
            'instance': 'VS',
            'array': 2,
            'naming': 'underscore',
            'positive': 'a[$i]',
            'negative': 'gnd',
            'dc': 1.0,
        }])
        assert "f'VS_{i}'" in code

    def test_non_array_source_unchanged(self):
        code = self._gen([{
            'type': 'VoltageSource',
            'instance': 'vdd',
            'positive': 'vdd',
            'negative': 'gnd',
            'dc': 1.8,
        }])
        # Existing flat emission
        assert "ckt.add(sources.vdc('vdd'," in code
        assert 'vdd = ' not in code
        assert 'VS = [' not in code


# ---------------------------------------------------------------------------
# Task 6 — End-to-end exec test for the arrayed submodule path
# ---------------------------------------------------------------------------

class TestArrayedSubmoduleEndToEnd:
    @needs_pdk
    def test_arrayed_submodule_execs_and_registers(self, tmp_path):
        """Generate a child Module, then a parent that uses array-of-4
        instantiations of the child, exec() the parent, and assert the
        parent's elem_dict carries all four bracketed instance names.
        """
        from silk.codegen.generator import Generator
        import os, sys

        unique = f'_child_arr_{id(tmp_path)}'

        child_code = Generator().generate({
            'circuit_name': 'leaf',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'leaf_sim'),
            'ports': {
                'a': {'dir': 'input'},
                'b': {'dir': 'output'},
            },
        })

        parent_code = Generator().generate({
            'circuit_name': 'parent',
            'pdk': 'sky130',
            'output_dir': str(tmp_path / 'parent_sim'),
            'submodules': [{
                'instance_name': 'U',
                'module_ref': {'import_path': '_child_arr',
                               'module_var': 'mod'},
                'array': 4,
                'nets': {'a': 'in[$i]', 'b': 'out[$i]'},
            }],
        })
        parent_code = parent_code.replace(
            'from _child_arr import mod as',
            f'from {unique} import mod as',
        )
        (tmp_path / 'leaf_sim').mkdir(exist_ok=True)
        (tmp_path / 'parent_sim').mkdir(exist_ok=True)
        (tmp_path / f'{unique}.py').write_text(child_code)

        # Patch simulate() to a no-op (we only care about elem_dict).
        from silk.circuit.modules import Module as _Module
        orig_sim = _Module.simulate
        _Module.simulate = lambda self, **kw: None

        sys.path.insert(0, str(tmp_path))
        try:
            ns: dict = {}
            exec(parent_code, ns)
            # The parent is a SUBCIRCUIT (has ports) — variable is `mod`
            # not `ckt`. Check both.
            parent_mod = ns.get('mod') or ns.get('ckt')
            assert parent_mod is not None, (
                f"neither 'mod' nor 'ckt' found in exec namespace: "
                f"{sorted(ns.keys())}"
            )
            keys = list(parent_mod.elem_dict.keys())
            for i in range(4):
                assert f'U[{i}]' in keys, (
                    f'expected U[{i}] in {keys}'
                )
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop(unique, None)
            _Module.simulate = orig_sim


# TASK-466: exercise emitted additions against real Modules and complete decks.
_CONTROLLED_CASES = [
    ('VCVS', 'VoltageControlledVoltageSource', 'E', {'gain': 2}, '2'),
    ('VCCS', 'VoltageControlledCurrentSource', 'G', {'gm': 0.01}, '0.01'),
    ('CCVS', 'CurrentControlledVoltageSource', 'H', {'transresistance': 1000}, '1000'),
    ('CCCS', 'CurrentControlledCurrentSource', 'F', {'gain': 2}, '2'),
]


@pytest.mark.parametrize('alias,canonical,prefix,params,value', _CONTROLLED_CASES)
@pytest.mark.parametrize('array,indices', [(2, [0, 1]), ([2, 4], [2, 3])])
@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
@pytest.mark.parametrize('context', ['standalone', 'module', 'test', 'rig'])
@pytest.mark.parametrize('array_controller', [None, 'bracketed', 'underscore'])
def test_controlled_arrays_emit_decks(tmp_path, alias, canonical, prefix, params,
                                     value, array, indices, naming, context,
                                     array_controller):
    import ast
    from silk import sources
    from silk.circuit.modules import Module
    from silk.codegen.generator import Generator
    from silk.codegen.testbench_generator import generate_testbenches

    # Keep subtraction in both forms; zero-based arrays cannot subtract one.
    low_offset = 0 if isinstance(array, int) else 1
    entry = dict(type=alias if context in ('test', 'standalone') else canonical,
                 instance='readout', out_pos='out[$i]', out_neg=f'low[$(i-{low_offset})]',
                 array=array, naming=naming, **params)
    controller = dict(type='VoltageSource', instance='sense', positive='sense',
                      negative='0', dc=0)
    # Controller naming intentionally differs from the consumer's naming.
    if array_controller:
        controller.update(array=[0, 2] if array == 2 else array, naming=array_controller)
    if prefix in 'EG':
        entry.update(ctrl_pos='input[$(i+1)]', ctrl_neg='ref[$(i)]')
    else:
        entry['ctrl_vname'] = 'sense'
    entries = [entry, controller]  # forward reference
    mod = Module('arrays', str(tmp_path), 'arrays.cir', pdk_details=None)
    if context in ('standalone', 'module'):
        spec = dict(circuit_name='arrays', sources=entries)
        if context == 'module':
            spec['ports'] = {'a': {'dir': 'input'}}
        code = Generator().generate(spec)
        # Execute the actual emitted add statements, avoiding simulate().
        tree = ast.parse(code)
        additions = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Attribute)
                     and node.value.func.attr == 'add']
        exec(compile(ast.Module(body=additions, type_ignores=[]), '<generated>', 'exec'),
             dict(ckt=mod, mod=mod, sources=sources))
        netlist = mod
    else:
        mod.module_name = 'dut'
        mod.define_module_ports({'a': 'input'})
        test = dict(name='t', dut_portmap={'a': 'a'})
        spec = dict(circuit_name='arrays', tests=[test])
        if context == 'rig':
            spec['rigs'] = [dict(name='r', sources=entries)]
            test['applied_rigs'] = ['r']
        else:
            test['sources'] = entries
        ns = {}
        exec(generate_testbenches(spec), ns)
        ns['_attach_t'](mod)
        netlist = mod.get_test('t').netlist
    deck_path = netlist.generate_netlist()
    from pathlib import Path
    deck = Path(deck_path).read_text()
    cards = [line for line in deck.splitlines() if line.startswith(prefix + 'readout')]
    expected = []
    for i in indices:
        name = f'readout<{i}>' if naming == 'bracketed' else f'readout_{i}'
        control = (f'input[{i+1}] ref[{i}]' if prefix in 'EG' else
                   f'Vsense<{i}>' if array_controller == 'bracketed' else
                   f'Vsense_{i}' if array_controller else 'Vsense')
        expected.append(f'{prefix}{name} out[{i}] low[{i-low_offset}] {control} {value}')
    assert cards == expected


@pytest.mark.parametrize('context', ['circuit', 'test', 'rig'])
@pytest.mark.parametrize('consumer', ['CCVS', 'CCCS'])
@pytest.mark.parametrize('target_type,target_array,consumer_array', [
    ('VoltageSource', [1, 3], 2),
    ('VoltageSource', 2, None),
    ('CurrentSource', 2, 2),
])
def test_controlled_array_invalid_controllers(context, consumer, target_type,
                                               target_array, consumer_array):
    from silk.codegen.generator import CircuitSpec
    from silk.codegen.testbench_generator import generate_testbenches
    entry = dict(type=consumer, instance='readout', out_pos='o', out_neg='0',
                 ctrl_vname='sense', **({'gain': 2} if consumer == 'CCCS' else
                                       {'transresistance': 1000}))
    if consumer_array is not None:
        entry['array'] = consumer_array
    entries = [entry, dict(type=target_type, instance='sense', positive='s',
                           negative='0', array=target_array)]
    with pytest.raises(ValueError) as exc:
        if context == 'circuit':
            CircuitSpec.from_dict(dict(sources=entries))
        else:
            test = dict(name='t', dut_portmap={})
            spec = dict(circuit_name='arrays', tests=[test])
            if context == 'rig':
                spec['rigs'] = [dict(name='r', sources=entries)]
                test['applied_rigs'] = ['r']
            else:
                test['sources'] = entries
            generate_testbenches(spec)
    message = str(exc.value)
    for detail in ('readout', 'sense', 'range',
                   'scalar' if consumer_array is None else '[0, 2)',
                   '[1, 3)' if target_array == [1, 3] else '[0, 2)'):
        assert detail in message


@pytest.mark.parametrize('controller_type,prefix,params', [
    ('VCVS', 'E', dict(ctrl_pos='s', ctrl_neg='0', gain=1)),
    ('CCVS', 'H', dict(ctrl_vname='seed', transresistance=1)),
    ('BehavioralVoltage', 'B', dict(positive='s', negative='0', expression='0')),
])
def test_controlled_array_can_sense_other_voltage_branches(tmp_path, controller_type,
                                                          prefix, params):
    from pathlib import Path
    from silk import sources
    from silk.circuit.modules import Module
    from silk.codegen.generator import Generator
    controller = dict(type=controller_type, instance='sense', **params)
    if prefix != 'B':
        controller.update(out_pos='s[$i]', out_neg='0', array=2)
    spec = dict(circuit_name='arrays', sources=[
        dict(type='CCCS', instance='copy', out_pos='o[$i]', out_neg='0',
             ctrl_vname='sense', gain=2, array=2),
        controller,
        dict(type='VoltageSource', instance='seed', positive='s', negative='0'),
    ])
    code = Generator().generate(spec)
    mod = Module('arrays', str(tmp_path), 'branches.cir', pdk_details=None)
    for line in code.splitlines():
        if line.startswith('ckt.add('):
            exec(line, dict(ckt=mod, sources=sources))
    deck_path = mod.generate_netlist()
    deck = Path(deck_path).read_text()
    suffixes = ['', ''] if prefix == 'B' else ['<0>', '<1>']
    for i, suffix in enumerate(suffixes):
        assert f'Fcopy<{i}> o[{i}] 0 {prefix}sense{suffix} 2\n' in deck


@pytest.mark.parametrize('array,indices', [(3, [0, 1, 2]), ([4, 7], [4, 5, 6])])
@pytest.mark.parametrize('kind', ['submodules', 'devices', 'ideal_elements'])
def test_per_copy_nets_match_handwritten_loop(tmp_path, monkeypatch, array, indices, kind):
    import ast
    import sys
    import types
    from silk.codegen.generator import Generator
    from silk.circuit.modules import Module
    from silk.elements import ElementLibrary
    from silk.pdk import load_pdk

    child = Module('leaf', str(tmp_path), 'leaf.cir', module_name='leaf', pdk_details=None)
    from silk import Port, PortDir
    child.define_module_ports({'d': Port(dir=PortDir.IN, bits=4), 'n': 'output'})
    child.add(ElementLibrary().resistor('r', p='d[2]', n='n', value=1000))
    imported = types.ModuleType('_per_copy_child')
    imported.mod = child
    monkeypatch.setitem(sys.modules, '_per_copy_child', imported)
    nets = [{'p': net, 'n': '0'} for net in ['n3', 'n0', 'clk']]
    entry = dict(instance='amp1', type='resistor', params={'value': 1000},
                 array=array, nets=nets)
    if kind == 'submodules':
        nets = [{'d[0]': 'a', 'd[1]': 'b', 'd[2]': net, 'd[3]': 'c', 'n': '0'}
                for net in ['p', 'q', 'r']]
        entry = dict(instance_name='amp1', array=array, nets=nets,
                     module_ref={'import_path': '_per_copy_child', 'module_var': 'mod'})
    elif kind == 'devices':
        entry.update(type='nmos1p8', params={'w': 1, 'l': 0.15},
                     nets=[{'d': n['p'], 'g': 'bias', 's': '0', 'b': '0'} for n in nets])
    spec = dict(circuit_name='parent', pdk='sky130', ports={},
                output_dir=str(tmp_path), **{kind: [entry]})
    code = Generator().generate(spec)
    tree = ast.parse(code)
    assert sum(isinstance(n, ast.For) for n in ast.walk(tree)) == 1
    assert f"__silk_amp1_nets = {entry['nets']!r}" in code
    index = 'i' if isinstance(array, int) else 'i - (4)'
    range_text = 'range(3)' if isinstance(array, int) else 'range(4, 7)'
    if kind == 'submodules':
        call = f"mod.add_module(mod__per_copy_child, f'amp1[{{i}}]', nets=__silk_amp1_nets[{index}])"
    elif kind == 'devices':
        call = f"mod.add(pdk.nmos1p8(f'amp1[{{i}}]', w=1, l=0.15, **__silk_amp1_nets[{index}]))"
    else:
        call = f"mod.add(elements.resistor(f'amp1[{{i}}]', value=1000, **__silk_amp1_nets[{index}]))"
    assert f"for i in {range_text}: {call}" in code
    namespace = {}
    exec(code, namespace)
    actual = namespace['mod']
    expected = Module('parent', str(tmp_path), 'parent.cir', module_name='parent',
                      pdk_details=load_pdk('sky130'))
    expected.define_module_ports({})
    for i, connections in zip(indices, entry['nets']):
        if kind == 'submodules':
            expected.add_module(child, f'amp1[{i}]', nets=connections)
        else:
            factory = load_pdk('sky130').nmos1p8 if kind == 'devices' else ElementLibrary().resistor
            expected.add(factory(f'amp1[{i}]', **connections, **entry['params']))
    assert actual.generate_netlist(subckt=True) == expected.generate_netlist(subckt=True)


@pytest.mark.parametrize('kind', ['submodules', 'devices', 'ideal_elements'])
@pytest.mark.parametrize('nets,error', [
    ([{'p': 'a'}], ValueError),
    ([{'p': 'a'}, {'n': 'b'}], ValueError),
    ([{'p': 'a'}, 'b'], TypeError),
    ([{'p': 'a'}, {'p': 1}], TypeError),
    ([{'p': 'a'}, {'p': 'bad net'}], ValueError),
    ([{'p': 'a'}, {'p': 'n[$i]'}], ValueError),
    ([{'p': 'a'}, {'p': '__silk_reserved'}], ValueError),
    ([{'d[]': 'a'}, {'d[]': 'b'}], ValueError),
    ([{'d[-1]': 'a'}, {'d[-1]': 'b'}], ValueError),
    ([{'d[1:0]': 'a'}, {'d[1:0]': 'b'}], ValueError),
    ([{'d [2]': 'a'}, {'d [2]': 'b'}], ValueError),
    ([{'p': 'd[1:0]'}, {'p': 'b'}], ValueError),
])
def test_per_copy_nets_refuse_invalid_entries(kind, nets, error):
    from silk.codegen.generator import Generator
    entry = dict(instance='amp1', type='resistor', array=2, nets=nets)
    if kind == 'submodules':
        entry = dict(instance_name='amp1', array=2, nets=nets,
                     module_ref={'import_path': 'child', 'module_var': 'mod'})
    with pytest.raises(error, match='amp1'):
        Generator().generate(dict(circuit_name='parent', pdk='sky130', ports={}, **{kind: [entry]}))


@pytest.mark.parametrize('pin', ['__silk_x[2]', '__SILK_x[2]', 2, 'net_like'])
def test_per_copy_submodule_pin_token_refusals(pin):
    from unittest.mock import Mock
    from silk.codegen.generator import Generator

    if pin == 'net_like':
        pin = Mock()
        pin.name = 'd[2]'
    entry = dict(instance_name='amp1', array=2,
                 nets=[{pin: 'a'}, {pin: 'b'}],
                 module_ref={'import_path': 'child', 'module_var': 'mod'})
    with pytest.raises(ValueError, match='amp1'):
        Generator().generate(dict(circuit_name='parent', pdk='sky130', ports={},
                                  submodules=[entry]))


@pytest.mark.parametrize('kind', ['devices', 'ideal_elements'])
def test_per_copy_device_bit_pin_is_refused(kind):
    from silk.codegen.generator import Generator
    entry = dict(instance='amp1', type='resistor', array=2,
                 nets=[{'p[2]': 'a'}, {'p[2]': 'b'}])
    with pytest.raises(ValueError, match='amp1'):
        Generator().generate(dict(circuit_name='parent', pdk='sky130', ports={},
                                  **{kind: [entry]}))


def test_per_copy_nets_binding_cannot_overwrite_parameter_variable():
    from silk.codegen.generator import Generator
    spec = dict(circuit_name='parent', pdk='sky130', ports={},
                variables=[{'name': '__silk_amp1_nets', 'value': 1000}],
                ideal_elements=[dict(type='resistor', instance='amp1', array=2,
                                     nets=[{'p': 'n3', 'n': '0'}, {'p': 'clk', 'n': '0'}],
                                     params={'value': {'variable': '__silk_amp1_nets'}})])
    with pytest.raises(ValueError, match='__silk_amp1_nets'):
        Generator().generate(spec)


@pytest.mark.parametrize('name', ['', None])
def test_per_copy_nets_invalid_submodule_identity_uses_entry_error(name):
    from silk.codegen.generator import CircuitSpec, Generator, SubmoduleSpec
    spec = CircuitSpec(circuit_name='parent', pdk='sky130', ports={}, submodules=[
        SubmoduleSpec(instance_name=name, import_path='child', module_var='mod',
                      array=1, nets=[{'p': 'a'}])])
    with pytest.raises(ValueError):
        Generator().generate(spec)
