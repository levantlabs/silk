"""ModuleTestbench.probe_bus: a bus Signal over named probes (TASK-606.3, doc-1 Q40)."""
import numpy as np
import pytest

from silk.exceptions import BusPairingError, ConfigurationError
from tests._env import needs_ngspice
from tests.test_bus_signal_maths import block, write_raw, ys


def bench(tmp_path):
    tb = block(tmp_path, 'dut').create_test('read')
    tb.add_module_under_test({'out': 'out'})
    tb.netlist.add_net('d', bits=range(8, 11))
    tb.netlist.add_net('c', bits=range(0, 3))
    tb.netlist.add_net('vin')
    return tb


def columns():
    cols = {'vin': (0, 1)}
    for n in range(8, 11):
        cols[f'd[{n}]'] = (n, n + .5)
    for n in range(3):
        cols[f'c[{n}]'] = (100 * n, 100 * n + 1)
    return cols


def test_probe_bus_is_a_new_aggregate_numbered_from_zero_in_list_order(tmp_path):
    tb = bench(tmp_path)
    pb = tb.probe_bus(['d[10]', 'd[9]', 'd[8]'])
    assert len(pb) == 3 and pb._indices == (0, 1, 2)
    assert [leaf.name for leaf in pb] == ['d[10]', 'd[9]', 'd[8]']   # leaves keep their paths
    write_raw(tb, tmp_path, columns())
    assert ys(pb) == [[10.0, 10.5], [9.0, 9.5], [8.0, 8.5]]
    assert pb[0].name == 'd[10]' and pb[2].name == 'd[8]'


def test_reverse_then_index_keeps_the_labels(tmp_path):
    tb = bench(tmp_path)
    pb = tb.probe_bus(['d[10]', 'd[9]', 'd[8]'])
    rev = pb[::-1]
    assert [leaf.name for leaf in rev] == ['d[8]', 'd[9]', 'd[10]']
    assert rev._indices == (2, 1, 0)
    assert rev[0].name == 'd[10]'          # label 0 stays on the d[10] probe


def test_declared_bus_keeps_its_numbers_and_maths_renumbers(tmp_path):
    tb = bench(tmp_path)
    d = tb.probe(tb.bus('d'))
    assert d._indices == (8, 9, 10)
    rev = d[::-1]
    assert [leaf.name for leaf in rev] == ['d[10]', 'd[9]', 'd[8]']
    assert rev[8].name == 'd[8]'
    with pytest.raises(Exception):
        rev[0]
    c = tb.probe_bus(['c[0]', 'c[1]', 'c[2]'])
    e = d + c
    assert e._indices == (0, 1, 2)
    write_raw(tb, tmp_path, columns())
    assert ys(e)[0] == [8.0 + 0.0, 8.5 + 1.0]
    assert ys(e)[2] == [10.0 + 200.0, 10.5 + 201.0]


def test_probe_bus_math_matches_the_scalar_probes(tmp_path):
    tb = bench(tmp_path)
    got = tb.probe_bus(['d[8]', 'd[9]']).mag()
    want = [tb.probe('d[8]').mag(), tb.probe('d[9]').mag()]
    write_raw(tb, tmp_path, columns())
    for g, w in zip(got, want):
        np.testing.assert_allclose(g.y, w.y)


def test_width_mismatch_raises_when_the_expression_is_built(tmp_path):
    tb = bench(tmp_path)
    with pytest.raises(BusPairingError):
        tb.probe_bus(['d[8]', 'd[9]']) - tb.probe_bus(['c[0]', 'c[1]', 'c[2]'])


def test_one_element_list_is_a_width_one_bus(tmp_path):
    tb = bench(tmp_path)
    pb = tb.probe_bus(['vin'])
    assert len(pb) == 1 and pb._indices == (0,)
    with pytest.raises(BusPairingError):
        pb + tb.probe_bus(['d[8]', 'd[9]'])


def test_tuple_is_accepted_and_repeats_are_allowed_here(tmp_path):
    tb = bench(tmp_path)
    assert len(tb.probe_bus(('d[8]', 'd[8]'))) == 2


def test_bare_string_is_refused(tmp_path):
    tb = bench(tmp_path)
    with pytest.raises(TypeError, match="not the string 'd'; use probe"):
        tb.probe_bus('d')


@pytest.mark.parametrize('empty', [[], ()])
def test_empty_input_raises_configuration_error(tmp_path, empty):
    tb = bench(tmp_path)
    with pytest.raises(ConfigurationError, match='needs at least one name'):
        tb.probe_bus(empty)


def test_wrong_container_and_element_types(tmp_path):
    tb = bench(tmp_path)
    with pytest.raises(TypeError, match='list or tuple.*set'):
        tb.probe_bus({'d[8]'})
    with pytest.raises(TypeError, match=r'names\[1\] must be a string, got int'):
        tb.probe_bus(['d[8]', 3])
    with pytest.raises(TypeError, match=r'names\[0\] must be a string, got BenchBusReference'):
        tb.probe_bus([tb.bus('d')])


def test_probe_errors_propagate_unwrapped(tmp_path):
    tb = bench(tmp_path)
    with pytest.raises(Exception) as info:
        tb.probe_bus(['d[8]', 'nosuchnet'])
    assert 'nosuchnet' in str(info.value) and not isinstance(info.value, TypeError)


def test_analysis_is_passed_to_every_element(tmp_path):
    tb = bench(tmp_path)
    tb.probe_bus(['d[8]', 'd[9]'], analysis='tran')   # accepted spelling reaches probe
    with pytest.raises(Exception, match='analysis'):
        tb.probe_bus(['d[8]', 'd[9]'], analysis='bogus')


def test_selected_probe_bus_keeps_its_owner_and_reads_back_its_own_result(tmp_path):
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
    from tests.test_object_probes import selected_bus
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    two = top.get_test('bus', instance='two')
    names = ['xdut.xstage.d[8]', 'xdut.xstage.d[9]']
    p1, p2 = one.probe_bus(names), two.probe_bus(names)
    assert all(leaf._bench() is one for leaf in p1._walk_leaves())
    assert all(leaf._bench() is two for leaf in p2._walk_leaves())

    def result_file(label, base):
        vectors = [RawVector('time', 'time', data=np.array([0., 1.]))]
        for offset, name in enumerate(names):
            vectors.append(RawVector(f'v({name})', 'voltage',
                                     data=np.array([base + offset, base + offset + .5])))
        path = tmp_path / (label + '.raw')
        write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors,
                                             npoints=2, real=True, padded=True)]))
        return str(path)

    one.evaluate(rawfile=result_file('one', 10.))
    two.evaluate(rawfile=result_file('two', 20.))
    # read back after BOTH evaluations: each bus follows its own selected test
    assert ys(p1) == [[10.0, 10.5], [11.0, 11.5]]
    assert ys(p2) == [[20.0, 20.5], [21.0, 21.5]]


@pytest.mark.slow
@needs_ngspice
def test_wider_internal_bus_probes_wired_and_internal_bits(tmp_path):
    from silk import Module
    from silk.circuit.ports import Port
    from silk.elements import ElementLibrary
    from silk.sources import VoltageSource
    from silk.simulation import Op

    elements = ElementLibrary()
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'x': Port.analog_input(bits=4), 'vss': Port.ground()})
    x = child.add_net('x', bits=8)
    for k in range(4):
        child.add(elements.resistor(f'ra{k}', p=x[k], n=x[k + 4], value=1000))
        child.add(elements.resistor(f'rb{k}', p=x[k + 4], n='vss', value=1000))
    host = Module(None, str(tmp_path), 'host.cir', None, module_name='host')
    host.define_module_ports({'xb': Port.analog_input(bits=4), 'vss': Port.ground()})
    stage = host.add_module(child, 'stage', x=host.add_net('xb', bits=4), vss='vss')
    assert stage.x._indices == tuple(range(8))
    tb = host.create_test('read')
    tb.add_sim_type(Op())
    tbx = tb.netlist.add_net('tbx', bits=4)
    for k in range(4):
        tb.add_source(VoltageSource(f'v{k}', positive=tbx[k], negative='0', dc=.2 * (k + 1)))
    tb.add_module_under_test({'xb': tbx, 'vss': '0'})
    probe = tb.probe(stage.x)
    assert [sig.name for sig in probe] == ([f'tbx[{k}]' for k in range(4)] +
        [f'xdut.xstage.x[{k}]' for k in range(4, 8)])
    tb.run()
    np.testing.assert_allclose([sig.y[0] for sig in probe],
                               [.2, .4, .6, .8, .1, .2, .3, .4])
