"""TASK-606.3: a TestbenchSpec signal may be an ordered list of paths, written as
one ``tb.probe_bus([...])`` line; a metric over a bus becomes a Python for loop of
scalar metrics named ``name[k]`` (doc-1 Q33, Q39, Q40 option A)."""
import ast

import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.codegen.testbench_generator import TestbenchSpecError, generate_testbenches
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
from silk.simulation.conditions import ConditionState

TRAN = {'kind': 'transient', 'step': 1e-9, 'stop': 1e-6}
PORTS = {'a0': 'd[0]', 'a1': 'd[1]', 'a2': 'd[2]', 'a3': 'd[3]', 'vin': 'vin'}
D4 = ['d[0]', 'd[1]', 'd[2]', 'd[3]']


def _spec(derived=(), metrics=(), specs=(), portmap=None, **test):
    t = {'name': 't1', 'dut_portmap': dict(portmap or PORTS), 'analyses': [TRAN],
         'metrics': list(metrics)}
    if derived:
        t['derived_signals'] = list(derived)
    if specs:
        t['specs'] = list(specs)
    t.update(test)
    return {'circuit_name': 'adc', 'circuit_builder': '_b', 'tests': [t]}


def _body(code):
    return [ln.strip() for ln in code.splitlines() if ln.startswith('    ')]


def _run(spec, tmp_path, columns, times=None):
    """Generate, import and evaluate the spec against a recorded rawfile."""
    code = generate_testbenches(spec)
    ports = spec['tests'][0]['dut_portmap']

    def _b(pdk):
        m = Module('x', str(tmp_path), 'x.cir', pdk_details=None, module_name='x')
        m.define_module_ports({p: PortDir.INOUT for p in ports})
        return m

    ns = {'_b': _b}
    exec(code, ns)
    mod = ns['build_adc_with_tests'](None)
    t = np.array([0.0, 1e-6]) if times is None else times
    vecs = [RawVector('time', 'time', data=t)]
    for name, y in columns.items():
        vecs.append(RawVector(f'v({name})', 'voltage', data=np.asarray(y, float)))
    path = tmp_path / 'r.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis', 'title': 't'},
                                         vecs, npoints=len(t), real=True, padded=True)]))
    return mod._tests['t1'].evaluate(rawfile=path, corner=ConditionState.NONE).metrics


def _cols():
    # bit b is the line b+1 -> b+3 over the run; vin is zero
    out = {f'd[{b}]': (-(b + 1.0), -(b + 3.0)) for b in range(4)}
    out['vin'] = (0.0, 0.0)
    return out


# -- generated text ---------------------------------------------------------------

def test_derived_bus_is_one_probe_bus_line_and_the_metric_is_a_loop():
    code = generate_testbenches(_spec(
        [{'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
        [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}]))
    assert "    dmag = tb.probe_bus(['d[0]', 'd[1]', 'd[2]', 'd[3]']).mag()" in code
    assert code.count('probe_bus') == 1                      # one line, not one per bit
    assert ("    for __silk_k, __silk_bit in enumerate(dmag):\n"
            "        tb.add_metric(f'dc[{__silk_k}]', metrics.average, sig=__silk_bit)\n") in code
    assert 'from silk import metrics' in code
    ast.parse(code)


def test_direct_list_metric_imports_metrics_and_loops():
    code = generate_testbenches(_spec(metrics=[{'name': 'dc', 'type': 'average', 'signal': D4}]))
    assert 'from silk import metrics' in code
    assert ("    for __silk_k, __silk_bit in enumerate("
            "tb.probe_bus(['d[0]', 'd[1]', 'd[2]', 'd[3]'])):") in code
    assert "tb.add_metric(f'dc[{__silk_k}]', metrics.average, sig=__silk_bit)" in code
    ast.parse(code)


def test_scalar_only_spec_has_no_bus_code():
    code = generate_testbenches(_spec(
        [{'name': 'x', 'operation': 'abs', 'inputs': ['vin']}],
        [{'name': 'dc', 'type': 'average', 'signal': 'x'}]))
    assert 'probe_bus' not in code and '__silk_' not in code and 'enumerate' not in code
    assert "tb.add_metric('dc', metrics.average, sig=x)" in code


def test_loop_locals_avoid_authored_names_and_a_derived_enumerate_is_refused():
    code = generate_testbenches(_spec(
        [{'name': '__silk_k', 'operation': 'abs', 'inputs': ['vin']},
         {'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
        [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}]))
    assert 'for __silk_k1, __silk_bit in enumerate(dmag):' in code
    assert "f'dc[{__silk_k1}]'" in code
    with pytest.raises(TestbenchSpecError, match="'enumerate'.*bus metric loop"):
        generate_testbenches(_spec(
            [{'name': 'enumerate', 'operation': 'abs', 'inputs': ['vin']},
             {'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
            [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}]))
    # a scalar derived signal named enumerate stays legal when no loop is emitted
    generate_testbenches(_spec([{'name': 'enumerate', 'operation': 'abs', 'inputs': ['vin']}]))


# -- numbers ----------------------------------------------------------------------

def test_loop_metrics_equal_the_per_bit_scalar_metrics(tmp_path):
    got = _run(_spec([{'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
                     [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}]),
               tmp_path, _cols())
    assert sorted(got) == ['dc[0]', 'dc[1]', 'dc[2]', 'dc[3]']
    for b in range(4):
        assert got[f'dc[{b}]'] == pytest.approx(b + 2.0)       # |-(b+1) .. -(b+3)| averages b+2


def test_descending_list_counts_k_by_position(tmp_path):
    got = _run(_spec([{'name': 'dmag', 'operation': 'abs', 'inputs': [D4[::-1]]}],
                     [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}]),
               tmp_path, _cols())
    assert [got[f'dc[{k}]'] for k in range(4)] == pytest.approx([5.0, 4.0, 3.0, 2.0])


def test_offset_difference_pairs_by_position(tmp_path):
    ports = {'a1': 'd[11]', 'a0': 'd[10]', 'b1': 'q[1]', 'b0': 'q[0]', 'vin': 'vin'}
    cols = {'d[11]': (5.0, 5.0), 'd[10]': (7.0, 7.0), 'q[1]': (1.0, 1.0),
            'q[0]': (2.0, 2.0), 'vin': (0.0, 0.0)}
    got = _run(_spec([{'name': 'err', 'operation': 'difference',
                       'inputs': [['d[11]', 'd[10]'], ['q[1]', 'q[0]']]}],
                     [{'name': 'm', 'type': 'average', 'signal': 'err'}], portmap=ports),
               tmp_path, cols)
    assert got['m[0]'] == pytest.approx(5.0 - 1.0)             # d[11] - q[1]
    assert got['m[1]'] == pytest.approx(7.0 - 2.0)             # d[10] - q[0]


def test_one_element_list_is_a_width_one_bus(tmp_path):
    got = _run(_spec(metrics=[{'name': 'dc', 'type': 'average', 'signal': ['vin']}]),
               tmp_path, {'vin': (3.0, 3.0), **{f'd[{b}]': (0, 0) for b in range(4)}})
    assert got['dc[0]'] == pytest.approx(3.0)


def test_scalar_broadcast_through_a_derived_scale(tmp_path):
    got = _run(_spec([{'name': 's', 'operation': 'scale', 'inputs': [D4],
                       'params': {'gain': 2.0, 'offset': 1.0}}],
                     [{'name': 'dc', 'type': 'average', 'signal': 's'}]),
               tmp_path, _cols())
    assert [got[f'dc[{k}]'] for k in range(4)] == pytest.approx(
        [2.0 * -(b + 2.0) + 1.0 for b in range(4)])


def test_bus_slot_with_a_scalar_slot_propagation_delay(tmp_path):
    t = np.linspace(0.0, 1e-6, 1001)
    vin = np.interp(t, [0, 1e-7, 1.1e-7, 1e-6], [0, 0, 1.0, 1.0])
    cols = {'vin': vin}
    for b in range(4):
        start = 1.2e-7 + b * 1e-8
        cols[f'd[{b}]'] = np.clip((t - start) / 1e-8, 0.0, 1.0)
    spec = _spec([{'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
                 [{'name': 'tpd', 'type': 'propagation_delay',
                   'signals': {'sig_in': 'vin', 'sig_out': 'dmag'},
                   'params': {'in_edge': 'rising', 'in_threshold': 0.5,
                              'out_threshold': 0.5}}],
                 sources=[{'type': 'VoltageSource', 'instance': 'vin', 'positive': 'vin',
                           'negative': 'gnd', 'pwl': [0, 0, 1e-7, 0, 1.1e-7, 1.0, 1e-6, 1.0]}])
    got = _run(spec, tmp_path, cols, times=t)
    values = [got[f'tpd[{k}]'] for k in range(4)]
    assert all(np.isfinite(values))
    assert values == pytest.approx([2e-8, 3e-8, 4e-8, 5e-8], abs=2e-10)


# -- refusals ---------------------------------------------------------------------

def _refused(spec, match):
    with pytest.raises(TestbenchSpecError, match=match):
        generate_testbenches(spec)


def test_malformed_lists_name_the_entry_and_element():
    inputs = lambda value: [{'name': 'x', 'operation': 'abs', 'inputs': [value]}]
    _refused(_spec(inputs([])), r"derived signal 'x'.*input 0.*empty list")
    _refused(_spec(inputs(['d[0]', ''])), r"derived signal 'x'.*input 0 element 1")
    _refused(_spec(inputs(['d[0]', 3])), r"derived signal 'x'.*input 0 element 1")
    _refused(_spec(inputs(['vin', 'vin'])),
             r"input 0: elements 0\.\.1 are the same path 'vin'.*string form")
    _refused(_spec(inputs(['xdut.xstage<2>.rout[0]'])),
             r"element 0 'xdut\.xstage<2>\.rout\[0\]'.*designer spelling")
    _refused(_spec(inputs(['xdut.rout[0]'])), r"element 0 'xdut\.rout\[0\]'.*designer spelling")
    _refused(_spec(metrics=[{'name': 'm', 'type': 'average', 'signal': []}]),
             r"metric 'm'.*empty list")


def test_unequal_widths_are_refused_at_generation():
    _refused(_spec([{'name': 'e', 'operation': 'difference',
                     'inputs': [['d[0]', 'd[1]'], ['d[2]', 'd[3]', 'vin']]}]),
             r"derived signal 'e'.*input 1 has 3 elements; input 0 has 2")


def test_two_bus_slots_in_one_metric_are_refused():
    _refused(_spec(metrics=[{'name': 'm', 'type': 'propagation_delay',
                             'signals': {'sig_in': ['d[0]', 'd[1]'],
                                         'sig_out': ['d[2]', 'd[3]']}}]),
             r"metric 'm'.*'sig_in' and 'sig_out' are both buses; one slot may be a bus")


def test_spec_on_a_bus_metric_is_refused():
    _refused(_spec(metrics=[{'name': 'dc', 'type': 'average', 'signal': D4}],
                   specs=[{'name': 's', 'metric': 'dc', 'op': '>', 'target': 0.0}]),
             r"spec 's'.*metric 'dc' expands to dc\[0\]\.\.dc\[3\].*not supported")


def test_generated_metric_name_collision_is_refused():
    _refused(_spec(metrics=[{'name': 'dc', 'type': 'average', 'signal': D4},
                            {'name': 'dc[1]', 'type': 'average', 'signal': 'vin'}]),
             r"generated metric 'dc\[1\]' collides")


def test_scalar_names_beginning_with_the_prefix_stay_legal():
    generate_testbenches(_spec(
        [{'name': '__silk_avg', 'operation': 'abs', 'inputs': ['vin']}],
        [{'name': '__silk_avg2', 'type': 'average', 'signal': '__silk_avg'}]))


@pytest.mark.parametrize('path, local', [('.k', '__silk_k'), ('.bit', '__silk_bit')])
def test_loop_locals_avoid_helper_aliases(path, local):
    """A helper import alias is a global read inside _attach_<test>; a loop local
    of the same name would make it a local there (UnboundLocalError)."""
    spec = _spec([{'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
                 [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}],
                 submodules=[{'instance_name': 'h1',
                              'module_ref': {'import_path': path, 'module_var': '__silk'},
                              'nets': {'out': 'out'}}])
    code = generate_testbenches(spec)
    assert f'import __silk as {local}' in code            # the alias really is taken
    assert f"tb.add_module({local}, 'h1'" in code          # and is read in the function
    assert f'{local}1' in code                             # so the loop local moves aside
    body = code[code.index('def _attach_t1'):code.index('def build_')]
    tree = ast.parse(body)
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Store)}
    assert local not in assigned


def test_loop_locals_are_unsuffixed_when_nothing_collides():
    code = generate_testbenches(_spec(
        [{'name': 'dmag', 'operation': 'abs', 'inputs': [D4]}],
        [{'name': 'dc', 'type': 'average', 'signal': 'dmag'}],
        submodules=[{'instance_name': 'h1',
                     'module_ref': {'import_path': '.biasgen', 'module_var': 'mod'},
                     'nets': {'out': 'out'}}]))
    assert 'for __silk_k, __silk_bit in enumerate(dmag):' in code
