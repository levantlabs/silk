"""TASK-527 code review: a raw current name reaches the ONE collector with its
as-given spelling; source physical names have one owner (__silk_pwl/__silk_dc)."""
import numpy as np
import pytest

from silk import sources
from silk.exceptions import (AmbiguousBranchCurrentError, AmbiguousSignalError,
                             SimulationError)
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile
from tests.test_metric_branch_current_paths import _avg, _build

VS = sources.VoltageSource


def _bench(tmp_path, names, srcs=(), data=None):
    """A bench with *srcs* declared and a synthetic rawfile holding i(<name>)
    for every name in *names* already loaded (no simulator)."""
    from silk.circuit.module_test import _load_metric_path  # noqa: F401
    tb = _build(tmp_path, [], [_avg('m', 'vin')])
    for s in srcs:
        tb.netlist.add_source(s)
    vecs = [RawVector('time', 'time', data=np.array([0., 1e-8]))]
    for i, n in enumerate(names):
        vals = data[i] if data else [float(i + 1)] * 2
        vecs.append(RawVector(f'i({n})', 'current', data=np.array(vals)))
    path = tmp_path / 's.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis', 'title': 't'},
                                         vecs, npoints=2, real=True, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    return tb


def _both(tb, spelling):
    """(probe y, metric y) for one i(...) spelling."""
    from silk.circuit.module_test import _load_metric_path
    return tb.probe(spelling).y, _load_metric_path(tb.netlist, spelling).y


# --- P2-1: as-given spelling reaches the collector --------------------------

def test_bracket_only_raw_current_resolves_through_probe_and_metric(tmp_path):
    tb = _bench(tmp_path, ['vvs[0]'], data=[[3., 3.]])
    for y in _both(tb, 'i(vvs[0])'):
        np.testing.assert_array_equal(y, [3., 3.])


def test_angle_only_raw_current_resolves_by_bracket_request(tmp_path):
    tb = _bench(tmp_path, ['vvs<0>'], data=[[4., 4.]])
    for spelling in ('i(vvs[0])', 'i(vvs<0>)'):
        for y in _both(tb, spelling):
            np.testing.assert_array_equal(y, [4., 4.])


@pytest.mark.parametrize('order', [('vvs[0]', 'vvs<0>'), ('vvs<0>', 'vvs[0]')])
@pytest.mark.parametrize('equal', [False, True])
def test_both_rows_are_ambiguous_never_missing(tmp_path, order, equal):
    data = [[5., 5.], [5., 5.]] if equal else None
    tb = _bench(tmp_path, list(order), data=data)
    from silk.circuit.module_test import _load_metric_path
    with pytest.raises(AmbiguousSignalError, match='ambiguous'):
        tb.probe('i(vvs[0])').y
    with pytest.raises(AmbiguousSignalError) as exc:
        _load_metric_path(tb.netlist, 'i(vvs[0])')
    assert 'not found' not in str(exc.value)
    assert 'Accepted spellings' not in str(exc.value)


def test_designer_name_and_raw_escape_controls(tmp_path):
    tb = _bench(tmp_path, ['vvs<0>'], [VS('vs[0]', 'a', '0', dc=1)], data=[[6., 6.]])
    for spelling in ('i(vs[0])', 'i(vvs<0>)', 'i(vvs[0])'):
        for y in _both(tb, spelling):
            np.testing.assert_array_equal(y, [6., 6.])


# --- P2-3: one owner of a source's physical instance names ------------------

def test_physical_names_voltage_pwl_is_two_cards():
    s = VS('bias[0]', 'out', '0', dc=.2, pwl=[(0, 0), (1e-9, 1)])
    assert s.physical_instance_names() == ('Vbias<0>__silk_pwl', 'Vbias<0>__silk_dc')


def test_physical_names_constant_and_waveform_cases():
    assert VS('a', 'o', '0', dc=1).physical_instance_names() == ('Va',)
    generic = VS('a', 'o', '0', waveform=sources.PWL([(0, 0), (1, 1)]))
    assert generic.physical_instance_names() == ('Va__silk_wave', 'Va__silk_dc')
    cur = sources.CurrentSource('a', 'o', '0', pwl=[(0, 0), (1, 1)])
    assert cur.physical_instance_names() == ('Ia__silk_pwl', 'Ia__silk_dc')
    bits = VS('a', 'o', '0', waveform=sources.BitPattern('1010', v1=0, v2=1, ui='1n', tr='10p', tf='10p'))
    assert bits.physical_instance_names() == ('Va__silk_wave', 'Va__silk_dc')


def test_physical_names_never_resolve_variables_or_render_spice():
    var = VS.dc.variable(name='lvl')              # truly unset: value is None
    assert var.value is None

    class Exploding:
        def to_spice(self):
            raise AssertionError('waveform callback must not be called')
    s = VS('a', 'o', '0', dc=var, waveform=Exploding())
    assert s.physical_instance_names() == ('Va__silk_wave', 'Va__silk_dc')
    p = VS('b', 'o', '0', dc=var, pwl=[(0, 0), (1, 1)])
    assert p.physical_instance_names() == ('Vb__silk_pwl', 'Vb__silk_dc')


def test_to_spice_uses_the_helper_and_stays_byte_identical():
    s = VS('bias[0]', 'out[0]', '0', dc=.2, pwl=[(0, 0), (1e-9, 1)])
    names = s.physical_instance_names()
    text = s.to_spice()
    assert [line.split()[0] for line in text.splitlines()] == list(names)
    assert text == ('Vbias<0>__silk_pwl out[0] __silk_src_Vbias[0] PWL(0.0 0.0 1e-09 1.0)\n'
                    'Vbias<0>__silk_dc __silk_src_Vbias[0] 0 DC 0.2\n')


def _pwl_neighbours(tmp_path, suffix):
    tb = _bench(tmp_path, [f'vbias<0>__silk{suffix}', f'vvbias{suffix}<0>'], [
        VS('bias[0]', 'a', '0', dc=.2, pwl=[(0, 0), (1, 1)]),
        VS(f'vbias{suffix}[0]', 'b', '0', dc=1)], data=[[1., 1.], [2., 2.]])
    return tb


@pytest.mark.parametrize('suffix', ['_pwl', '_dc'])
def test_pwl_dc_neighbor_resolves_through_probe_and_metric(tmp_path, suffix):
    tb = _pwl_neighbours(tmp_path, suffix)
    for y in _both(tb, f'i(vbias{suffix}[0])'):
        np.testing.assert_array_equal(y, [2., 2.])


@pytest.mark.parametrize('suffix', ['_pwl', '_dc'])
def test_pwl_dc_raw_cards_and_authored_neighbors_are_distinct(tmp_path, suffix):
    tb = _pwl_neighbours(tmp_path, suffix)
    first = tb.probe(f'i(vbias<0>__silk{suffix})').y
    second = tb.probe(f'i(vvbias{suffix}<0>)').y
    np.testing.assert_array_equal(first, [1., 1.])
    np.testing.assert_array_equal(second, [2., 2.])


def test_pwl_and_pulse_conflict_before_placement():
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match=r'bias.*pwl.*pulse'):
        VS('bias[0]', 'a', '0', pulse=sources.Pulse(0, 1, tr='1n', tf='1n', pw='1u', per='2u'),
           pwl=[(0, 0), (1, 1)])


def test_current_pwl_card_and_authored_neighbor_are_distinct(tmp_path):
    cur = sources.CurrentSource('bias[0]', 'a', '0', pwl=[(0, 0), (1, 1)])
    tb = _bench(tmp_path, ['@Ibias<0>__silk_pwl[current]', 'vibias_pwl<0>'],
                [cur, VS('ibias_pwl[0]', 'b', '0', dc=1)])
    np.testing.assert_array_equal(tb.probe('i(ibias_pwl[0])').y, [2., 2.])
    np.testing.assert_array_equal(tb.probe('i(Ibias<0>__silk_pwl)').y, [1., 1.])


def test_unresolved_variable_source_and_bitpattern_names(tmp_path):
    var = VS.dc.variable(name='lvl')              # truly unset: value is None
    assert var.value is None
    tb = _bench(tmp_path, ['va'], [VS('a', 'o', '0', dc=var),
                                    VS('b', 'p', '0', waveform=sources.BitPattern(
                                        '1010', v1=0, v2=1, ui='1n', tr='10p', tf='10p'))])
    np.testing.assert_array_equal(tb.probe('i(a)').y, [1., 1.])


import re


def _suggested(exc):
    return re.search(r'Write i\((.+?)\) for source', str(exc)).group(1)


@pytest.mark.parametrize('suffix', ['_pwl', '_dc'])
def test_pwl_physical_card_resolves_without_a_remedy(tmp_path, suffix):
    tb = _pwl_neighbours(tmp_path, suffix)
    for y in _both(tb, f'i(vbias[0]__silk{suffix})'):
        np.testing.assert_array_equal(y, [1., 1.])
    if suffix == '_pwl':
        np.testing.assert_array_equal(tb.probe('i(bias[0])').y, [1., 1.])
    else:
        with pytest.raises(SimulationError):
            tb.probe('i(bias[0])').y


def test_one_card_remedy_resolves(tmp_path):
    from silk.circuit.module_test import _load_metric_path
    tb = _bench(tmp_path, ['vvs<0>', 'vvvs<0>'], [VS('vs[0]', 'a', '0', dc=1),
                                                    VS('vvs[0]', 'b', '0', dc=1)],
                data=[[1., 1.], [2., 2.]])
    with pytest.raises(AmbiguousBranchCurrentError) as exc:
        tb.probe('i(vvs[0])')
    remedy = _suggested(exc.value)
    np.testing.assert_array_equal(tb.probe(f'i({remedy})').y, [1., 1.])
    np.testing.assert_array_equal(_load_metric_path(tb.netlist, f'i({remedy})').y, [1., 1.])


@pytest.mark.parametrize('name', ['vbias[0]_pwl', 'vbias[0]_dc', 'ibias[0]_pwl'])
def test_suffix_after_index_is_not_an_authored_source_name(name):
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='invalid instance name'):
        VS(name, 'out', '0', dc=1)
