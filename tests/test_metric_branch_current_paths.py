"""TASK-449: a metric's string signal path reads a source current in the
contract's ``i(vs)`` spelling AND the raw vector spelling ``i(vvs)``, through
the same lookup ``tb.probe`` uses; the one ambiguous case raises."""
import shutil

import pytest

from silk.codegen.testbench_generator import generate_testbenches
from silk.exceptions import AmbiguousBranchCurrentError, MetricBindingError

HAVE_NGSPICE = shutil.which('ngspice') is not None


def _build(tmp_path, sources, metrics):
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements

    spec = {
        'circuit_name': 'div', 'circuit_builder': '_build_div',
        'tests': [{
            'name': 'dc',
            'dut_portmap': {'vin': 'vin', 'vout': 'vout', 'vss': '0'},
            'sources': [{'type': 'VoltageSource', 'instance': n,
                         'positive': p, 'negative': '0', 'dc': v}
                        for n, p, v in sources],
            'analyses': [{'kind': 'transient', 'step': 1e-8, 'stop': 1e-7}],
            'metrics': metrics,
        }]}

    def _build_div(pdk):
        mod = Module('div', str(tmp_path), 'div.cir', pdk_details=None,
                     module_name='div')
        mod.define_module_ports({'vin': PortDir.INOUT, 'vout': PortDir.INOUT,
                                 'vss': PortDir.GROUND})
        with mod:
            elements.resistor('r1', p='vin', n='vout', value=1000)
            elements.resistor('r2', p='vout', n='vss', value=1000)
        return mod

    ns = {'_build_div': _build_div}
    exec(generate_testbenches(spec), ns)
    return ns['build_div_with_tests'](None)._tests['dc']


def _avg(name, signal):
    return {'name': name, 'type': 'average', 'signal': signal}


pytestmark = pytest.mark.slow
needs = pytest.mark.skipif(not HAVE_NGSPICE, reason='needs the ngspice binary')
ONE = [('vs', 'vin', 1.2)]
TWO = [('vs', 'vin', 1.2), ('vvs', 'vout', 0.5)]


@needs
@pytest.mark.parametrize('path', ['i(vs)', 'i(vvs)'])
def test_both_spellings_read_the_branch_current(tmp_path, path):
    test = _build(tmp_path, ONE, [_avg('m', path), _avg('net', 'vin')])
    report = test.run('tt')
    assert report.metrics['m'] == pytest.approx(-6.0e-4, rel=1e-3)
    assert report.metrics['net'] == pytest.approx(1.2, rel=1e-6)


@needs
@pytest.mark.parametrize('path', ['i(vs)', 'i(vvs)'])
def test_function_form_str_slot_reads_the_branch_current(tmp_path, path):
    from silk.metrics import average
    test = _build(tmp_path, ONE, [])
    test.add_metric('f', average, sig=path)
    assert test.run('tt').metrics['f'] == pytest.approx(-6.0e-4, rel=1e-3)


@needs
def test_function_form_str_slot_ambiguous_raises(tmp_path):
    from silk.metrics import average
    test = _build(tmp_path, TWO, [])
    test.add_metric('f', average, sig='i(vvs)')
    with pytest.raises(AmbiguousBranchCurrentError):
        test.run('tt')


@needs
def test_unknown_current_lists_accepted_spellings(tmp_path):
    test = _build(tmp_path, ONE, [_avg('m', 'i(nope)')])
    with pytest.raises(Exception, match='Accepted spellings'):
        r = test.run('tt')
        assert 'Accepted spellings' in str(r.metrics['m'])
        raise Exception(str(r.metrics['m']))


@needs
def test_ambiguous_current_raises_in_metric_and_probe(tmp_path):
    test = _build(tmp_path, TWO, [_avg('m', 'i(vvs)')])
    with pytest.raises(AmbiguousBranchCurrentError) as ei:
        test.run('tt')
    assert "'vs'" in str(ei.value) and "'vvs'" in str(ei.value)
    with pytest.raises(AmbiguousBranchCurrentError):
        test.probe('i(vvs)')
    # the unambiguous one on the same bench still reads its own source
    test.probe('i(vs)')


def test_branch_vector_name_rules():
    from silk.circuit.module_test import _branch_vector_name
    from silk.sources.classes import VoltageSource
    import inspect  # noqa: F401
    src = {'vs': VoltageSource('vs', 'a', '0', dc=1.0)}
    assert _branch_vector_name(src, 'vs') == 'Vvs'
    assert _branch_vector_name(src, 'vvs') == 'vvs'   # raw, verbatim
    assert _branch_vector_name(src, 'other') == 'other'
