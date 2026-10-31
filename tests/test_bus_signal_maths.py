"""Bus Signal maths: bit by bit, by position like numpy (TASK-606.1, doc-1 Q33)."""
import copy
import operator

import numpy as np
import pytest

from silk.exceptions import BusPairingError, CircuitError
from silk.signals.base import Signal
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
from tests.test_object_probes import block

X = np.array([0.0, 1.0])


def bit(name, v0, v1):
    return Signal(name, 'voltage', 'time', 'V', X, np.array([v0, v1], float))


def bus(name, numbers, base=1.0):
    """Bound bus whose bit n is the line n*base, n*base+0.5."""
    return Signal._make_bus_handle(
        name, tuple(bit(f'{name}[{n}]', n * base, n * base + .5) for n in numbers),
        indices=tuple(numbers))


def ys(b):
    return [list(s.y) for s in b]


def test_equal_length_buses_combine_by_position():
    d = bus('d', (7, 6, 5, 4))
    e = bus('e', (4, 5, 6, 7), base=10.0)           # other numbers, other order
    r = d + e
    assert r._indices == (0, 1, 2, 3)                # a two-bus result is numbered from 0
    assert r.name == 'd+e' and len(r) == 4
    # position i: d[7]=[7,7.5] pairs with e[4]=[40,40.5], not with e[7]
    assert ys(r) == [[47.0, 48.0], [56.0, 57.0], [65.0, 66.0], [74.0, 75.0]]
    assert ys(d - e)[0] == [-33.0, -33.0]
    assert ys(d * e)[1] == [300.0, 6.5 * 50.5]
    assert ys(e / d)[0] == [40 / 7, 40.5 / 7.5]


def test_numpy_style_slices_of_one_bus():
    c = bus('c', (0, 1, 2, 3), base=1.0)
    a, b = c[1:3], c[2:0:-1]                         # c1,c2 and c2,c1
    got = list(a - b)                                # iteration is positional
    c1, c2 = c[1].y, c[2].y
    np.testing.assert_allclose(got[0].y, c1 - c2)
    np.testing.assert_allclose(got[1].y, c2 - c1)


@pytest.mark.parametrize('expr,expect', [
    (lambda d: d + 1, [[8, 8.5], [7, 7.5]]),
    (lambda d: 1 + d, [[8, 8.5], [7, 7.5]]),
    (lambda d: d - 1, [[6, 6.5], [5, 5.5]]),
    (lambda d: 10 - d, [[3, 2.5], [4, 3.5]]),
    (lambda d: d * 2, [[14, 15], [12, 13]]),
    (lambda d: 2 * d, [[14, 15], [12, 13]]),
    (lambda d: d / 2, [[3.5, 3.75], [3, 3.25]]),
    (lambda d: 84 / d, [[12, 84 / 7.5], [14, 84 / 6.5]]),
    (lambda d: d ** 2, [[49, 7.5 ** 2], [36, 6.5 ** 2]]),
    (lambda d: -d, [[-7, -7.5], [-6, -6.5]]),
])
def test_scalar_broadcast_and_unary_minus(expr, expect):
    d = bus('d', (7, 6))
    r = expr(d)
    assert r._indices == (7, 6)
    np.testing.assert_allclose(ys(r), expect)


def test_scalar_signal_applies_to_every_bit_on_either_side():
    d = bus('d', (5, 4))
    ref = bit('vref', 1.0, 2.0)
    assert ys(d - ref) == [[4.0, 3.5], [3.0, 2.5]]
    assert ys(ref - d) == [[-4.0, -3.5], [-3.0, -2.5]]
    assert (ref + d)._indices == (5, 4)


def test_unary_methods_act_per_bit_and_keep_numbers():
    d = bus('d', (9, 8), base=-1.0)
    m = d.mag()
    assert m._indices == (9, 8) and ys(m) == [[9.0, 8.5], [8.0, 7.5]]
    for name in ('real', 'imag', 'db20', 'phase_deg', 'derivative'):
        out = getattr(d, name)()
        assert out._indices == (9, 8) and len(out) == 2
    sh = d.shift(1.0)
    assert sh._indices == (9, 8)
    with pytest.raises(TypeError, match='no bus-valued argument'):
        d.shift(bus('z', (9, 8)))


def test_chained_maths_keeps_numbers():
    d = bus('d', (7, 6))
    out = ((d * 2 + 1) - d).mag()
    assert out._indices == (0, 1)
    np.testing.assert_allclose(ys(out), [[8, 8.5], [7, 7.5]])


@pytest.mark.parametrize('other,words', [
    ((7, 6), ("'d' (width 4)", "'e' (width 2)")),
    ((7, 6, 5, 4, 3), ("'d' (width 4)", "'e' (width 5)")),
    ((9,), ("'d' (width 4)", "'e' (width 1)")),
])
def test_length_mismatch_raises_naming_both_lengths(other, words):
    d = bus('d', (7, 6, 5, 4))
    e = bus('e', other)
    for op in (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b,
               lambda a, b: a / b):
        with pytest.raises(BusPairingError) as info:
            op(d, e)
        for w in words:
            assert w in str(info.value)
    with pytest.raises(BusPairingError, match="'e' \\(width"):
        e + d
    assert issubclass(BusPairingError, ValueError)
    from silk.exceptions import ConfigurationError
    import silk.exceptions as exc
    assert issubclass(BusPairingError, ConfigurationError) and 'BusPairingError' in exc.__all__


def test_same_length_different_numbers_is_not_an_error():
    r = bus('d', (3, 2, 1, 0)) + bus('e', (7, 6, 5, 4))
    assert r._indices == (0, 1, 2, 3) and len(r) == 4


def test_unsupported_operand_is_not_swallowed():
    d = bus('d', (1, 0))
    with pytest.raises(TypeError):
        d + 'x'
    with pytest.raises(TypeError, match='real number'):
        d ** d


def test_unchanged_bus_behaviour():
    d = bus('d', (7, 6, 5, 4))
    assert len(d) == 4
    assert [s.name for s in d] == ['d[7]', 'd[6]', 'd[5]', 'd[4]']
    assert d[6].name == 'd[6]' and d[5:7]._indices == (6, 5)
    code = d.as_int()
    assert code.type == 'count'
    # unary methods on a scalar still return scalars
    assert not isinstance(d[7].mag(), type(None)) and len(d[7].mag()) == 1
    result = copy.deepcopy(d + 1)
    assert result._indices == (7, 6, 5, 4)


# -- deferred probes -------------------------------------------------------

def write_raw(tb, tmp_path, columns):
    vectors = [RawVector('time', 'time', data=np.array([0., 1.]))]
    for name, (a, b) in columns.items():
        vectors.append(RawVector(f'v({name})', 'voltage', data=np.array([a, b])))
    path = tmp_path / 'bus.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis', 'title': 't'},
                                         vectors, npoints=2, real=True, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()


def bench_with_buses(tmp_path):
    tb = block(tmp_path, 'dut').create_test('read')
    tb.add_module_under_test({'out': 'out'})
    tb.netlist.add_net('d', bits=range(4, 8))
    tb.netlist.add_net('r', bits=range(4, 8))
    tb.netlist.add_net('w', bits=range(0, 4))
    return tb


def columns():
    cols = {}
    for n in range(4, 8):
        cols[f'd[{n}]'] = (n, n + .5)
        cols[f'r[{n}]'] = (10 * n, 10 * n + 1)
    for n in range(4):
        cols[f'w[{n}]'] = (0, 1)
    return cols


def test_tb_bus_reference_and_deferred_equals_eager(tmp_path):
    tb = bench_with_buses(tmp_path)
    ref = tb.bus('d')
    assert len(ref) == 4 and ref[5].name == 'd[5]' and len(ref[5:7]) == 2
    assert [n.name for n in ref] == ['d[4]', 'd[5]', 'd[6]', 'd[7]']
    d, r = tb.probe(ref), tb.probe(tb.bus('r'))
    deferred = ((d - r) * 2 + 1).mag()
    assert deferred._indices == (0, 1, 2, 3) and not deferred.bound
    sub = tb.probe(ref[5:7])
    assert sub._indices == (5, 6)
    write_raw(tb, tmp_path, columns())
    eager_d = Signal._make_bus_handle('d', tuple(bit(f'd[{n}]', n, n + .5) for n in range(4, 8)),
                                      indices=(4, 5, 6, 7))
    eager_r = Signal._make_bus_handle('r', tuple(bit(f'r[{n}]', 10 * n, 10 * n + 1)
                                                 for n in range(4, 8)), indices=(4, 5, 6, 7))
    eager = ((eager_d - eager_r) * 2 + 1).mag()
    for got, want in zip(deferred.resolve(), eager):
        np.testing.assert_allclose(got.y, want.y)
    # the bench reference itself does bus maths through its own probe
    assert ys((tb.bus('d') + 1).resolve())[0] == [5.0, 5.5]
    # a deferred bus decodes exactly like an eager one
    assert list(d.as_int().y) == list(eager_d.as_int().y)


def test_tb_bus_length_mismatch_refusal(tmp_path):
    tb = bench_with_buses(tmp_path)
    with pytest.raises(BusPairingError) as info:
        tb.probe(tb.bus('d')) + tb.probe(tb.bus('w')[1:3])
    assert '(width 4)' in str(info.value) and '(width 2)' in str(info.value)
    assert len(tb.probe(tb.bus('d')) + tb.probe(tb.bus('w'))) == 4   # same length


def test_tb_bus_refusals(tmp_path):
    tb = bench_with_buses(tmp_path)
    with pytest.raises(CircuitError, match="no bus 'nope'.*'d', 'r', 'w'"):
        tb.bus('nope')
    with pytest.raises(CircuitError, match='declared'):
        tb.bus('d')[0]
    other = bench_with_buses(tmp_path)
    with pytest.raises(CircuitError, match='another test bench'):
        other.probe(tb.bus('d'))
    with pytest.raises(CircuitError, match='voltage'):
        tb.probe(tb.bus('d'), type='I')


def test_bus_on_an_arrayed_placement_copy(tmp_path):
    tb = block(tmp_path, 'dut').create_test('read')
    stage = tb.add_module(block(tmp_path), 'stage', {'out': 'parent'})
    for k in (2, 8):
        leaf = block(tmp_path, f'copy{k}')
        leaf.add_net('rout', bits=range(4, 8))
        stage.subcircuit.add_module(leaf, f'X[{k}]', {'out': 'mid'})
    probe = tb.probe(stage.X[8].bus('rout'))
    assert probe._indices == (4, 5, 6, 7)
    assert [s.name for s in probe][0] == 'xstage.xX<8>.rout[4]'
    other = tb.probe(stage.X[2].bus('rout'))
    assert (probe + other)._indices == (0, 1, 2, 3)
    sliced = tb.probe(stage.X[8].bus('rout')[5:7])
    with pytest.raises(BusPairingError, match=r'width 4.*width 2'):
        probe - sliced


def test_width_one_bus_maths():
    a, b = bus('a', (3,)), bus('b', (3,), base=2.0)
    assert ys(a + b) == [[9.0, 10.0]] and (a + b)._indices == (0,)
    assert ys((-a).mag()) == [[3.0, 3.5]]
    assert len(a + bus('c', (4,))) == 1                 # same length, other number
    with pytest.raises(BusPairingError, match=r'width 1.*width 2'):
        a + bus('c', (4, 5))


def selected_benches(tmp_path):
    from tests.test_object_probes import selected_bus
    top, tb, stage = selected_bus(tmp_path)
    tb.netlist.add_net('d', bits=range(4, 8))
    return top, tb, top.get_test('bus', instance='one'), top.get_test('bus', instance='two')


def test_selected_bench_bus_keeps_its_owner(tmp_path):
    from silk.circuit.placement_reference import BenchBitReference
    top, tb, one, two = selected_benches(tmp_path)
    ref = one.bus('d')
    for piece in (ref, ref[5:7], ref[5], *ref):
        with pytest.raises(Exception, match='another selected test'):
            two.probe(piece)
        with pytest.raises(CircuitError, match='selected test'):
            tb.probe(piece)
    assert isinstance(ref[5], BenchBitReference)
    p = one.probe(ref[5:7])
    assert all(leaf._bench() is one for leaf in p._walk_leaves())
    summed = ref + 1                       # implicit arithmetic uses the owner
    assert all(leaf._bench() is one for leaf in summed._walk_leaves())
    assert all(leaf._bench() is one for leaf in (ref[4].mag())._walk_leaves())
    other = two.bus('d') + 1
    assert all(leaf._bench() is two for leaf in other._walk_leaves())
    with pytest.raises(Exception, match='another'):
        one.probe(two.bus('d'))


def test_bench_bit_reference_foreign_bench_and_maths(tmp_path):
    tb = bench_with_buses(tmp_path)
    other = bench_with_buses(tmp_path)
    bit4 = tb.bus('d')[4]
    assert [str(b.name) for b in tb.bus('d')][0] == 'd[4]'
    for piece in (tb.bus('d'), tb.bus('d')[5:7], bit4, *tb.bus('d')):
        with pytest.raises(CircuitError, match='another test bench'):
            other.probe(piece)
    write_raw(tb, tmp_path, columns())
    np.testing.assert_allclose((bit4 * 2 + 1).y, [9.0, 10.0])
    np.testing.assert_allclose(bit4.mag().y, [4.0, 4.5])
    assert tb.probe(bit4).name == 'd[4]'


def test_offset_bus_result_is_numbered_from_zero_by_position():
    d = bus('d', (8, 9, 10))                      # d[8:11]
    c = bus('c', (0, 1, 2))
    for e, order in ((d + c, 'dc'), (c + d, 'cd')):
        assert e._indices == (0, 1, 2) and len(e) == 3
        for k in range(3):                         # e[0] = d[8] + c[0], ...
            np.testing.assert_allclose(e[k].y, d[8 + k].y + c[k].y)
        assert order
    # the operands keep their own numbers
    assert d._indices == (8, 9, 10) and d[8].name == 'd[8]'
    with pytest.raises(CircuitError, match='declared'):
        (d + c)[8]


def test_offset_bus_with_number_and_scalar_signal():
    d = bus('d', (8, 9, 10))
    ref = bit('vref', 1.0, 2.0)
    for r in (d * 2, 2 * d, d - ref, ref - d, -d, d.mag(), d ** 2):
        assert r._indices == (8, 9, 10)
    np.testing.assert_allclose((d * 2)[8].y, 2 * d[8].y)
    np.testing.assert_allclose((2 - d)[10].y, 2 - d[10].y)


def test_reversed_operand_pairs_by_position_with_zero_numbering():
    d = bus('d', (8, 9, 10))
    rev = d[10:7:-1]                               # d10, d9, d8
    e = d - rev
    assert e._indices == (0, 1, 2)
    np.testing.assert_allclose(e[0].y, d[8].y - d[10].y)
    np.testing.assert_allclose(e[1].y, 0 * d[9].y)
    np.testing.assert_allclose(e[2].y, d[10].y - d[8].y)


def test_python_slice_excludes_stop_for_three_bit_offset_bus():
    d = bus('d', (8, 9, 10))
    assert d[8:11]._indices == (8, 9, 10)
    assert d[8:10]._indices == (8, 9)


def test_deferred_transformed_bus_arithmetic_keeps_position_pairing():
    from tests.test_signal_transforms_deferred import bench_with, probe, run
    d, e = bus('d', (7, 6)), bus('e', (4, 5), base=10.)
    bench = bench_with(d, e)
    bench.netlist._signals.update({s.name: s for s in (*d, *e)})
    # Declare bus shape as real probes do, retaining the authored bit numbers.
    dp = Signal._make_bus_handle('d', tuple(probe(bench, f'd[{n}]') for n in (7, 6)), indices=(7, 6))
    ep = Signal._make_bus_handle('e', tuple(probe(bench, f'e[{n}]') for n in (4, 5)), indices=(4, 5))
    expr = dp.dbm() / ep.dbm()
    assert expr._indices == (0, 1)
    assert all(s.type is None for s in expr)
    run(bench)
    for result, a, b in zip(expr, d, e):
        np.testing.assert_allclose(result.y,
            (10 * np.log10(a.y ** 2 / .05)) / (10 * np.log10(b.y ** 2 / .05)))


def test_unknown_bus_bit_power_preserves_values_and_type():
    unknown = Signal('u[7]', None, 'time', 'V', X, np.array([2., 3.]))
    d = Signal._make_bus_handle('u', (unknown,), indices=(7,))
    squared = d ** 2
    assert squared._indices == (7,)
    np.testing.assert_allclose(squared[7].y, [4., 9.])
    assert squared[7].type is None


@pytest.mark.parametrize('numbers', [(8, 9, 10), (10, 9, 8)])
@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('operation', [
    lambda d, s: d + s, lambda d, s: s + d,
    lambda d, s: d - s, lambda d, s: s - d,
    lambda d, s: d * s, lambda d, s: s * d,
    lambda d, s: d / s, lambda d, s: s / d,
    lambda d, s: d ** 2, lambda d, s: -d,
    lambda d, s: d.mag(), lambda d, s: (d * 2).mag(),
])
@pytest.mark.parametrize('scalar_signal', [False, True])
def test_one_bus_results_preserve_declared_index_access(
        tmp_path, numbers, deferred, operation, scalar_signal):
    source = bus('d', numbers)
    scalar = bit('ref', 2., 2.) if scalar_signal else 2.
    if deferred:
        tb = block(tmp_path, 'dut').create_test('read')
        tb.add_module_under_test({'out': 'out'})
        tb.netlist.add_net('d', bits=range(8, 11))
        d = tb.probe(tb.bus('d'))
        if numbers[0] > numbers[-1]:
            d = d[10:7:-1]
        if scalar_signal:
            tb.netlist.add_net('ref')
            scalar = tb.probe('ref')
    else:
        d = source
    result = operation(d, scalar)
    assert len(result) == 3
    with pytest.raises(CircuitError, match='declared'):
        result[0]
    sliced = result[8:10]
    reversed_slice = result[10:7:-1]
    if deferred:
        write_raw(tb, tmp_path, {
            **{f'd[{n}]': (n, n + .5) for n in numbers},
            'ref': (2., 2.),
        })
    for number, leaf in zip(numbers, result):
        expected = operation(source[number], 2.)
        np.testing.assert_allclose(result[number].y, expected.y)
        np.testing.assert_allclose(leaf.y, expected.y)
    for leaf, number in zip(reversed_slice, reversed(numbers)):
        np.testing.assert_allclose(leaf.y, operation(source[number], 2.).y)
    decoded = (d * 1).as_int(threshold=9.25)
    expected_code = [4, 6] if numbers[0] == 8 else [1, 3]
    np.testing.assert_array_equal(decoded.y, expected_code)
    expected_order = [n for n in numbers if n in (8, 9)]
    assert len(sliced) == 2
    for leaf, number in zip(sliced, expected_order):
        np.testing.assert_allclose(leaf.y, operation(source[number], 2.).y)
    # All three transformed bits are above threshold 0 or below -100.
    np.testing.assert_array_equal(result.as_int(threshold=-100).y, [7, 7])
    np.testing.assert_array_equal(result.as_int(threshold=1000).y, [0, 0])


def test_two_buses_with_same_declared_numbers_still_start_at_zero():
    d = bus('d', (8, 9, 10))
    result = d + bus('c', (8, 9, 10))
    np.testing.assert_allclose(result[0].y, [16., 17.])
    with pytest.raises(CircuitError, match='declared'):
        result[8]


def test_deferred_as_int_preserves_nan_when_read(tmp_path):
    from silk.metrics.diagnostics import MetricWarning
    tb = bench_with_buses(tmp_path)
    decoded = tb.probe(tb.bus('d')).as_int()
    assert not decoded.bound
    cols = {f'd[{n}]': (0, 1 if n == 4 else 0) for n in range(4, 8)}
    cols['d[6]'] = (np.nan, 0)
    write_raw(tb, tmp_path, cols)
    with pytest.warns(MetricWarning, match='as_int'):
        np.testing.assert_array_equal(decoded.y, [np.nan, 1])
    with pytest.warns(MetricWarning, match='as_int'):
        np.testing.assert_array_equal(decoded.x, X)


@pytest.mark.parametrize('operation', [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize('other_bus', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_bus_arithmetic_refuses_inexact_bit_grid(operation, other_bus, reverse):
    d = bus('d', (9, 8))
    shifted = Signal('e[4]', 'voltage', 'time', 'V',
                     np.array([0., np.nextafter(1., 2.)]), np.array([1., 2.]))
    other = (Signal._make_bus_handle('e', (bit('e[5]', 1., 2.), shifted),
                                     indices=(5, 4)) if other_bus else shifted)
    with pytest.raises(ValueError, match='X-values'):
        operation(other, d) if reverse else operation(d, other)
    assert d._indices == (9, 8)
