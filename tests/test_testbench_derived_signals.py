"""TASK-373: a TestbenchSpec test declares derived signals from silk's fixed
operation list (probe -> operation -> metric), emitted as plain silk Python
over tb.probe(...) and Signal arithmetic/methods."""
import ast
import json

import numpy as np
import pytest

from silk.codegen.testbench_generator import TestbenchSpecError, generate_testbenches
from silk.signals import export_signal_operations
from silk.signals.base import Signal
from tests._env import HAVE_NGSPICE

TRAN = {'kind': 'transient', 'step': 1e-9, 'stop': 1e-6}
AC = {'kind': 'ac', 'variation': 'dec', 'points': 10, 'fstart': 1, 'fstop': 1e6}


def _spec(derived, metrics=(), analyses=(TRAN,), **test):
    t = {'name': 't', 'dut_portmap': {'a': 'a', 'b': 'b', 'c': 'c'},
         'analyses': list(analyses), 'derived_signals': derived,
         'metrics': list(metrics)}
    t.update(test)
    return {'circuit_name': 'c', 'tests': [t]}


def _attach_body(code):
    return [ln.strip() for ln in code.splitlines() if ln.startswith('    ')]


# --- the exported operation list (AC #3) -------------------------------------

PINNED = {   # the reviewer-pinned vocabulary (task notes, 2026-10-01)
    'sum': (2, None, [], 'a + b (+ ...)'),
    'difference': (2, 2, [], 'a - b'),
    'product': (2, None, [], 'a * b (* ...)'),
    'quotient': (2, 2, [], 'a / b'),
    'mean': (2, None, [], '(a + b + ...) / n'),
    'scale': (1, 1, [('gain', 1.0), ('offset', 0.0)], 'gain * a + offset'),
    'abs': (1, 1, [], 'a.mag()'),
    'derivative': (1, 1, [], 'a.derivative()'),
    'integral': (1, 1, [], 'a.integral()'),
}


def test_operation_list_matches_the_pinned_vocabulary():
    ops = export_signal_operations()
    assert [o['name'] for o in ops] == list(PINNED)
    for o in ops:
        lo, hi, params, python = PINNED[o['name']]
        assert (o['min_inputs'], o['max_inputs']) == (lo, hi)
        assert [(p['name'], p['default']) for p in o['parameters']] == params
        assert all(p['type'] == 'float' and not p['required']
                   for p in o['parameters'])
        assert o['python'] == python
        assert set(o) == {'name', 'min_inputs', 'max_inputs', 'ordered',
                          'parameters', 'python', 'description'}


def test_operation_list_is_json_safe_and_a_copy():
    ops = export_signal_operations()
    assert json.loads(json.dumps(ops)) == ops
    ops[0]['name'] = 'mutated'
    assert export_signal_operations()[0]['name'] == 'sum'


def test_generator_validates_against_the_exported_list(monkeypatch):
    """The generator reads the same list: an operation taken out of the
    export is refused, and a required parameter added to it is enforced."""
    import silk.signals.operations as ops_mod
    trimmed = tuple(o for o in ops_mod._OPERATIONS if o['name'] != 'abs')
    monkeypatch.setattr(ops_mod, '_OPERATIONS', trimmed)
    with pytest.raises(TestbenchSpecError, match="unknown operation 'abs'"):
        generate_testbenches(_spec([{'name': 'x', 'operation': 'abs',
                                     'inputs': ['a']}]))
    required = tuple(
        dict(o, parameters=[{'name': 'k', 'type': 'float', 'default': None,
                             'required': True}])
        if o['name'] == 'derivative' else o for o in ops_mod._OPERATIONS)
    monkeypatch.setattr(ops_mod, '_OPERATIONS', required)
    with pytest.raises(TestbenchSpecError,
                       match=r"missing required param\(s\) \['k'\]"):
        generate_testbenches(_spec([{'name': 'x', 'operation': 'derivative',
                                     'inputs': ['a']}]))


# --- emission (AC #1, #2, #7) -------------------------------------------------

def test_emits_probe_math_and_function_form_metric():
    code = generate_testbenches(_spec(
        [{'name': 'vd', 'operation': 'difference', 'inputs': ['a', 'b']},
         {'name': 'vs', 'operation': 'scale', 'inputs': ['vd'],
          'params': {'gain': 2.0}}],
        metrics=[{'name': 'avg', 'type': 'average', 'signal': 'vs',
                  'params': {'start': 1e-7}},
                 {'name': 'plain', 'type': 'average', 'signal': 'a'}]))
    assert 'from silk import metrics\n' in code
    body = _attach_body(code)
    i = body.index("vd = tb.probe('a') - tb.probe('b')")
    assert body[i + 1] == 'vs = 2.0 * vd + 0.0'
    assert body[i + 2] == ("tb.add_metric('avg', metrics.average, sig=vs, "
                           "start=1e-07)")
    assert body[i + 3] == "tb.add_metric('plain', type='average', signal='a')"
    assert body.index('tb.add_sim_type(analysis.Transient(step=1e-09, stop=1e-06))') < i
    ast.parse(code)


def test_every_operation_emits_the_silk_python_of_the_table():
    cases = {
        'sum': (['a', 'b', 'c'], {}, "tb.probe('a') + tb.probe('b') + tb.probe('c')"),
        'difference': (['a', 'b'], {}, "tb.probe('a') - tb.probe('b')"),
        'product': (['a', 'b'], {}, "tb.probe('a') * tb.probe('b')"),
        'quotient': (['a', 'b'], {}, "tb.probe('a') / tb.probe('b')"),
        'mean': (['a', 'b', 'c'], {}, "(tb.probe('a') + tb.probe('b') + tb.probe('c')) / 3"),
        'scale': (['a'], {'offset': -0.5}, "1.0 * tb.probe('a') + -0.5"),
        'abs': (['a'], {}, "tb.probe('a').mag()"),
        'derivative': (['a'], {}, "tb.probe('a').derivative()"),
        'integral': (['a'], {}, "tb.probe('a').integral()"),
    }
    assert set(cases) == set(PINNED)
    for op, (inputs, params, expr) in cases.items():
        code = generate_testbenches(_spec([{'name': 'x', 'operation': op,
                                            'inputs': inputs, 'params': params}]))
        assert f'x = {expr}' in _attach_body(code), op


def test_derived_analysis_pins_its_probes():
    code = generate_testbenches(_spec(
        [{'name': 'vd', 'operation': 'difference', 'inputs': ['a', 'b'],
          'analysis': 'tran'}],
        metrics=[{'name': 'avg', 'type': 'average', 'signal': 'vd'}],
        analyses=(TRAN, AC)))
    assert ("vd = tb.probe('a', analysis='tran') - "
            "tb.probe('b', analysis='tran')") in _attach_body(code)


def test_multi_input_metric_binds_derived_and_path_slots():
    code = generate_testbenches(_spec(
        [{'name': 'vs', 'operation': 'scale', 'inputs': ['b'],
          'params': {'gain': 1.0}}],
        metrics=[{'name': 'tpd', 'type': 'propagation_delay',
                  'signals': {'sig_in': 'a', 'sig_out': 'vs'},
                  'params': {'in_edge': 'any'}}]))
    assert ("tb.add_metric('tpd', metrics.propagation_delay, "
            "sig_in=tb.probe('a'), sig_out=vs, in_edge='any')"
            ) in _attach_body(code)


def test_spec_without_derived_signals_is_byte_identical():
    base = {'circuit_name': 'c', 'circuit_builder': '_b', 'tests': [{
        'name': 't', 'dut_portmap': {'a': 'a'}, 'analyses': [TRAN],
        'metrics': [{'name': 'avg', 'type': 'average', 'signal': 'a'}],
        'specs': [{'name': 's', 'metric': 'avg', 'op': '<', 'target': 1.0}]}]}
    with_empty = json.loads(json.dumps(base))
    with_empty['tests'][0]['derived_signals'] = []
    expected = (
        "from silk.simulation import analysis\n"
        "\n"
        "def _attach_t(mod):\n"
        "    tb = mod.create_test('t')\n"
        "    tb.add_module_under_test({'a': 'a'})\n"
        "    tb.add_sim_type(analysis.Transient(step=1e-09, stop=1e-06))\n"
        "    tb.add_metric('avg', type='average', signal='a')\n"
        "    tb.add_spec('s', 'avg', '<', 1.0)\n"
        "\n"
        "def build_c_with_tests(pdk):\n"
        "    mod = _b(pdk)\n"
        "    _attach_t(mod)\n"
        "    return mod\n")
    assert generate_testbenches(base) == expected
    assert generate_testbenches(with_empty) == expected


def test_helper_alias_avoids_a_derived_signal_name():
    helper = {'instance_name': 'h1',
              'module_ref': {'import_path': 'lib', 'module_var': 'mod'},
              'nets': {'out': 'hx'}}
    code = generate_testbenches(_spec(
        [{'name': 'mod_lib', 'operation': 'abs', 'inputs': ['a']}],
        submodules=[helper]))
    assert 'from lib import mod as mod_lib_1' in code
    assert 'mod_lib = tb.probe' in code


# --- refusals (AC #4) ---------------------------------------------------------

@pytest.mark.parametrize('derived, match', [
    ([{'name': 'x', 'operation': 'db20', 'inputs': ['a']}],
     r"unknown operation 'db20'; the silk operations are \['sum'"),
    ([{'name': 'x', 'inputs': ['a']}], 'unknown operation None'),
    ([{'name': 'x', 'operation': 'difference', 'inputs': ['a']}],
     "operation 'difference' takes exactly 2 inputs; got 1"),
    ([{'name': 'x', 'operation': 'sum', 'inputs': ['a']}],
     "operation 'sum' takes 2 or more inputs; got 1"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['a', 'b']}],
     "operation 'abs' takes exactly 1 input; got 2"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': 'a'}],
     "'inputs' must be a list"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['']}], "unknown input ''"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': [3]}], 'unknown input 3'),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['y']},
      {'name': 'y', 'operation': 'abs', 'inputs': ['a']}],
     "unknown input 'y'; it names a derived signal declared later"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['x']}],
     "unknown input 'x'; it names this derived signal"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['a'], 'params': {'gain': 2}}],
     r"unknown param\(s\) \['gain'\] for operation 'abs'"),
    ([{'name': 'x', 'operation': 'scale', 'inputs': ['a'],
       'params': {'gain': '2'}}], "param 'gain' must be a number"),
    ([{'name': 'x', 'operation': 'scale', 'inputs': ['a'],
       'params': {'gain': True}}], "param 'gain' must be a number"),
    ([{'name': 'a', 'operation': 'abs', 'inputs': ['b']}],
     "name clashes .* with DUT port 'a'"),
    ([{'name': 'VDD', 'operation': 'abs', 'inputs': ['b']}],
     "name clashes .* with net 'vdd' of a source"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['a']},
      {'name': 'x', 'operation': 'abs', 'inputs': ['b']}],
     'duplicate derived signal name'),
    ([{'name': 'tb', 'operation': 'abs', 'inputs': ['a']}], 'reserved'),
    ([{'name': 'metrics', 'operation': 'abs', 'inputs': ['a']}], 'reserved'),
    ([{'name': 'v-d', 'operation': 'abs', 'inputs': ['a']}],
     'must be a Python identifier'),
    ([{'name': 'lambda', 'operation': 'abs', 'inputs': ['a']}],
     'must be a Python identifier'),
    ([{'operation': 'abs', 'inputs': ['a']}], "missing required 'name'"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['a'], 'expr': 'a*2'}],
     r"unknown key\(s\) \['expr'\]"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['a'], 'analysis': 'ac'}],
     "analysis='ac', but this test declares"),
    ([{'name': 'x', 'operation': 'abs', 'inputs': ['a'], 'analysis': 'bogus'}],
     'not a silk analysis'),
    ('nope', 'derived_signals must be a list'),
])
def test_refusals(derived, match):
    vdd = {'type': 'VoltageSource', 'instance': 'vsup', 'positive': 'vdd',
           'negative': '0', 'dc': 1.8}
    with pytest.raises(TestbenchSpecError, match=match):
        generate_testbenches(_spec(derived, sources=[vdd]))


def test_inputs_reading_different_analyses_are_refused():
    with pytest.raises(TestbenchSpecError, match='read different analyses'):
        generate_testbenches(_spec(
            [{'name': 'x', 'operation': 'abs', 'inputs': ['a'], 'analysis': 'tran'},
             {'name': 'y', 'operation': 'difference', 'inputs': ['x', 'b'],
              'analysis': 'ac'}], analyses=(TRAN, AC)))


def test_ambiguous_derived_metric_names_the_derived_signal():
    with pytest.raises(TestbenchSpecError,
                       match="Set \"analysis\" on derived signal 'x'"):
        generate_testbenches(_spec(
            [{'name': 'x', 'operation': 'abs', 'inputs': ['a']}],
            metrics=[{'name': 'm', 'type': 'average', 'signal': 'x'}],
            analyses=(TRAN, AC)))


def test_metric_analysis_must_agree_with_its_derived_signal():
    with pytest.raises(TestbenchSpecError,
                       match="set \"analysis\": 'ac' on the derived signal"):
        generate_testbenches(_spec(
            [{'name': 'x', 'operation': 'abs', 'inputs': ['a'],
              'analysis': 'tran'}],
            metrics=[{'name': 'm', 'type': 'average', 'signal': 'x',
                      'analysis': 'ac'}],
            analyses=(TRAN, AC)))


# --- generated result == silk Signal computation (AC #5) ----------------------

def _sig(name, x, y):
    return Signal(name, 'voltage', 'time', 'V', x, y)


X = np.linspace(0.0, 1e-6, 41)
DATA = {'a': _sig('a', X, np.sin(2e6 * np.pi * X) + 0.2),
        'b': _sig('b', X, 0.5 + X * 1e5),
        'c': _sig('c', X, np.cos(4e6 * np.pi * X) - 2.0)}


class _FakeBench:
    """Stands in for tb: probe(path) returns that path's data Signal."""
    def probe(self, path, analysis=None):
        return DATA[path]


A, B, C = DATA['a'], DATA['b'], DATA['c']
DIRECT = {   # the silk Signal computation each operation names
    'sum': (['a', 'b', 'c'], {}, lambda: A + B + C),
    'difference': (['a', 'b'], {}, lambda: A - B),
    'product': (['a', 'b', 'c'], {}, lambda: A * B * C),
    'quotient': (['a', 'b'], {}, lambda: A / B),
    'mean': (['a', 'b', 'c'], {}, lambda: (A + B + C) / 3),
    'scale': (['a'], {'gain': -3.0, 'offset': 0.25}, lambda: -3.0 * A + 0.25),
    'abs': (['a'], {}, lambda: A.mag()),
    'derivative': (['a'], {}, lambda: A.derivative()),
    'integral': (['a'], {}, lambda: A.integral()),
}


@pytest.mark.parametrize('op', list(PINNED))
def test_generated_result_equals_silk_signal_computation(op):
    inputs, params, direct = DIRECT[op]
    code = generate_testbenches(_spec(
        [{'name': 'x', 'operation': op, 'inputs': inputs, 'params': params},
         {'name': 'y', 'operation': 'scale', 'inputs': ['x'],
          'params': {'gain': 2.0, 'offset': 1.0}}]))
    ns = {'tb': _FakeBench()}
    for line in _attach_body(code):
        if line.startswith(('x = ', 'y = ')):
            exec(line, ns)
    want = direct()
    np.testing.assert_array_equal(ns['x'].x, want.x)
    np.testing.assert_array_equal(ns['x'].y, want.y)
    np.testing.assert_array_equal(ns['y'].y, 2.0 * want.y + 1.0)


def test_mean_and_scale_defaults_match_their_definitions():
    """Independent of Signal arithmetic: mean is (a + b) / n, scale's
    defaults are the identity, integral starts at 0, abs is |y|."""
    ns = {'tb': _FakeBench()}
    code = generate_testbenches(_spec(
        [{'name': 'm', 'operation': 'mean', 'inputs': ['a', 'b']},
         {'name': 's', 'operation': 'scale', 'inputs': ['a']},
         {'name': 'g', 'operation': 'integral', 'inputs': ['a']},
         {'name': 'k', 'operation': 'abs', 'inputs': ['c']}]))
    for line in _attach_body(code):
        if line[:4] in ('m = ', 's = ', 'g = ', 'k = '):
            exec(line, ns)
    np.testing.assert_allclose(ns['m'].y, (A.y + B.y) / 2)
    np.testing.assert_array_equal(ns['s'].y, A.y)
    assert ns['g'].y[0] == 0.0
    np.testing.assert_array_equal(ns['k'].y, np.abs(C.y))


# --- end to end on ngspice (AC #6) --------------------------------------------

@pytest.mark.slow
def test_slow_generated_bench_checks_a_metric_of_a_derived_signal(tmp_path):
    if not HAVE_NGSPICE:
        pytest.skip('needs the ngspice binary')
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.metrics import average

    spec = {
        'circuit_name': 'div', 'circuit_builder': '_build_div',
        'tests': [{
            'name': 'tran',
            'dut_portmap': {'vin': 'vin', 'vout': 'vout', 'vss': '0'},
            'sources': [{'type': 'VoltageSource', 'instance': 'vs',
                         'positive': 'vin', 'negative': '0', 'dc': 1.2}],
            'analyses': [{'kind': 'transient', 'step': 1e-8, 'stop': 1e-6}],
            'derived_signals': [
                {'name': 'vdrop', 'operation': 'difference',
                 'inputs': ['vin', 'vout']},
                {'name': 'vdrop2', 'operation': 'scale', 'inputs': ['vdrop'],
                 'params': {'gain': 2.0}}],
            'metrics': [{'name': 'drop2_avg', 'type': 'average',
                         'signal': 'vdrop2'}],
            'specs': [{'name': 'drop2 > 1.1', 'metric': 'drop2_avg',
                       'op': '>', 'target': 1.1},
                      {'name': 'drop2 < 1.3', 'metric': 'drop2_avg',
                       'op': '<', 'target': 1.3}],
        }]}
    code = generate_testbenches(spec)
    assert 'vdrop2 = 2.0 * vdrop + 0.0' in code

    def _build_div(pdk):
        mod = Module('div', str(tmp_path), 'div.cir', pdk_details=None,
                     module_name='div')
        mod.define_module_ports({'vin': PortDir.INOUT, 'vout': PortDir.INOUT,
                                 'vss': PortDir.GROUND})
        with mod:
            # 1k/1k divider: vin - vout = vin / 2 = 0.6 V, so 2x that = 1.2 V.
            elements.resistor('r1', p='vin', n='vout', value=1000)
            elements.resistor('r2', p='vout', n='vss', value=1000)
        return mod

    ns = {'_build_div': _build_div}
    exec(code, ns)
    mod = ns['build_div_with_tests'](None)
    test = mod._tests['tran']
    report = test.run('tt')
    assert report.passed, report
    got = report.metrics['drop2_avg']
    assert got == pytest.approx(1.2, rel=1e-6)
    # The metric is the silk computation on the run's own traces.
    vin = test.netlist.load_signal('vin')
    vout = test.netlist.load_signal('vout')
    assert got == pytest.approx(average(2.0 * (vin - vout)), rel=1e-12)
