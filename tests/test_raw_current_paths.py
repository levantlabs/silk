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
    """Probe, metric and direct current reads for one i(...) spelling."""
    from silk.circuit.module_test import _load_metric_path
    return (tb.probe(spelling).y, _load_metric_path(tb.netlist, spelling).y,
            tb.netlist.load_signal(spelling).y, tb.netlist.load_signal(spelling, 'I').y)


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
    with pytest.warns(UserWarning, match='emitted reading'):
        probe = tb.probe('i(vvs[0])')
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(probe.y, [1., 1.])
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(vvs[0])').y, [1., 1.])
    np.testing.assert_array_equal(tb.probe('i(vvvs[0])').y, [2., 2.])


@pytest.mark.parametrize('name', ['vbias[0]_pwl', 'vbias[0]_dc', 'ibias[0]_pwl'])
def test_suffix_after_index_is_not_an_authored_source_name(name):
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='invalid instance name'):
        VS(name, 'out', '0', dc=1)


def test_direct_current_source_card_collision_and_probe_physical_escape(tmp_path):
    tb = _bench(tmp_path, ['vvs', 'vvvs'],
                [VS('vs', 'a', '0', dc=1), VS('vvs', 'b', '0', dc=2)],
                data=[[3., 4.], [7., 8.]])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(vvs)').y, [3., 4.])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(vvs)', 'I').y, [3., 4.])
    from silk.circuit.module_test import _load_metric_path
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(tb.probe('i(vvs)').y, [3., 4.])
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(vvs)').y, [3., 4.])
    np.testing.assert_array_equal(tb.netlist.load_signal('vvs', 'I').y, [3., 4.])
    np.testing.assert_array_equal(tb.probe('i(vs)').y, [3., 4.])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(vs)').y, [3., 4.])


@pytest.mark.parametrize('mapped_saved', [False, True])
def test_direct_logical_source_overrides_unrelated_external_current_row(tmp_path, mapped_saved):
    names = ['sense', 'vsense'] if mapped_saved else ['sense']
    data = [[3., 4.], [7., 8.]] if mapped_saved else [[3., 4.]]
    tb = _bench(tmp_path, names, [VS('sense', 'a', '0', dc=99)], data=data)
    if mapped_saved:
        np.testing.assert_array_equal(tb.netlist.load_signal('i(sense)').y, [7., 8.])
    else:
        with pytest.raises(SimulationError, match='not found'):
            tb.netlist.load_signal('i(sense)')
    np.testing.assert_array_equal(tb.netlist.load_signal('sense', 'I').y, [3., 4.])


def test_direct_unrecognised_wrapped_current_keeps_exact_raw_row(tmp_path):
    tb = _bench(tmp_path, ['mystery'], data=[[3., 4.]])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(mystery)').y, [3., 4.])


@pytest.mark.parametrize('adapter_saved', [False, True])
def test_unmapped_current_wrapper_keeps_default_selector(tmp_path, adapter_saved):
    names = ['ibias', '@ibias[current]'] if adapter_saved else ['ibias']
    data = [[1., 1.], [2., 2.]] if adapter_saved else [[1., 1.]]
    tb = _bench(tmp_path, names, [sources.CurrentSource('bias', 'a', '0', dc=.1)], data=data)
    np.testing.assert_array_equal(tb.netlist.load_signal('i(ibias)').y, [1., 1.])
    if adapter_saved:
        np.testing.assert_array_equal(tb.netlist.load_signal('i(ibias)', 'I').y, [2., 2.])
        np.testing.assert_array_equal(tb.netlist.load_signal('i(bias)').y, [2., 2.])
    else:
        for name, selector in [('i(ibias)', 'I'), ('i(bias)', 'V')]:
            with pytest.raises(SimulationError, match='not found'):
                tb.netlist.load_signal(name, selector)


@pytest.mark.parametrize('wrapped_saved', [False, True])
def test_metric_unmapped_current_keeps_bare_row_contract(tmp_path, wrapped_saved):
    from silk.circuit.module_test import _load_metric_path
    from silk.signals.rawfile import RawVector
    tb = _bench(tmp_path, ['rawcurrent'] if wrapped_saved else [], data=[[2., 2.]])
    plot = tb.netlist._raw.native.plots[0]
    plot.vectors.append(RawVector('rawcurrent', 'current', data=np.array([1., 1.])))
    path = tmp_path / 'bare.raw'
    write_rawfile(path, RawFile([plot]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    if wrapped_saved:
        with pytest.raises(AmbiguousSignalError) as exc:
            _load_metric_path(tb.netlist, 'i(rawcurrent)')
        assert 'Accepted spellings' not in str(exc.value)
        np.testing.assert_array_equal(tb.netlist.load_signal('i(rawcurrent)').y, [2., 2.])
    else:
        np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(rawcurrent)').y, [1., 1.])
        with pytest.raises(SimulationError, match='not found'):
            tb.netlist.load_signal('i(rawcurrent)')


@pytest.mark.parametrize('path', ['i(vdd)', 'I(VDD)', '  i(vdd)  '])
@pytest.mark.parametrize('selector', ['V', 'I'])
def test_authored_current_scan_preserves_case_and_wrapper_spacing(tmp_path, path, selector):
    tb = _bench(tmp_path, ['vvdd'], [VS('VdD', 'a', '0', dc=99)], data=[[3., 4.]])
    np.testing.assert_array_equal(tb.netlist.load_signal(path, selector).y, [3., 4.])


@pytest.mark.parametrize('path', ['i(xdut[1].xcore[2].ref[3])',
                                 'I(XDUT[1]:XCORE[2]:REF[3])',
                                 'i(xdut[1].XCORE[2]:ref[3])'])
def test_authored_current_scan_walks_complete_indexed_hierarchy(tmp_path, path):
    from silk import Module
    core = Module(None, str(tmp_path), 'core.cir', None, module_name='core')
    core.define_module_ports({})
    core.add_source(VS('ref[3]', 'a', '0', dc=99))
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({})
    dut.add_module(core, 'CoRe[2]', nets={})
    tb = _bench(tmp_path, ['v.xdut<1>.xcore<2>.vref<3>'], data=[[3., 4.]])
    tb.netlist.add_module(dut, 'DuT[1]', nets={})
    np.testing.assert_array_equal(tb.netlist.load_signal(path).y, [3., 4.])


def test_authored_x_placement_requires_existing_logical_x_prefix(tmp_path):
    from silk import Module
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({})
    child.add_source(VS('SeNsE', 'a', '0', dc=99))
    tb = _bench(tmp_path, ['v.xxblock.vsense'], data=[[3., 4.]])
    tb.netlist.add_module(child, 'xBlock', nets={})
    np.testing.assert_array_equal(tb.netlist.load_signal('i(XXBLOCK:SENSE)').y, [3., 4.])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(xblock.sense)').y, [3., 4.])


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('selector', ['V', 'I'])
@pytest.mark.parametrize('saved', [False, True])
def test_case_variant_source_card_collision_keeps_base_card_route(tmp_path, nested, selector, saved):
    from silk import Module
    card = 'xbranch.vvs' if nested else 'vvs'
    names = [card, 'v.' + card] if nested else [card]
    tb = _bench(tmp_path, names if saved else [],
                data=[[3., 4.]] * len(names) if saved else None)
    owner = tb.netlist
    if nested:
        owner = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
        owner.define_module_ports({})
        tb.netlist.add_module(owner, 'Branch', nets={})
    owner.add_source(VS('vs', 'a', '0', dc=1))
    owner.add_source(VS('vvs', 'b', '0', dc=2))
    path = 'i(XBRANCH:VVS)' if nested else 'i(VVS)'
    if saved:
        np.testing.assert_array_equal(tb.netlist.load_signal(path, selector).y, [3., 4.])
    else:
        with pytest.raises(SimulationError) as caught:
            tb.netlist.load_signal(path, selector)
        assert not isinstance(caught.value, AmbiguousBranchCurrentError)
    from silk.circuit.module_test import _load_metric_path
    with pytest.warns(UserWarning, match='emitted reading'):
        if saved:
            np.testing.assert_array_equal(_load_metric_path(tb.netlist, path).y, [3., 4.])
        else:
            with pytest.raises(SimulationError, match='not found') as caught:
                _load_metric_path(tb.netlist, path)
            assert not isinstance(caught.value, AmbiguousBranchCurrentError)


def test_case_equivalent_source_keys_preserve_lookup_order(tmp_path):
    tb = _bench(tmp_path, ['vbias', '@ibias[current]'],
                [VS('Bias', 'a', '0', dc=99), sources.CurrentSource('bIAS', 'b', '0', dc=99)],
                data=[[3., 4.], [7., 8.]])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(BIAS)').y, [3., 4.])


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('metric', [False, True])
def test_logical_current_calls_shared_lookup_once(tmp_path, monkeypatch, nested, split, metric):
    from silk import Module
    from silk.circuit import module_test
    source = sources.CurrentSource('bias[0]', 'a', '0', dc=.2, pwl=[0, .5, 1, 1]) if split else VS('bias[0]', 'a', '0', dc=99)
    prefix = 'xbranch.' if nested else ''
    names = (['@i.' + prefix + 'ibias<0>__silk_pwl[current]', '@i.' + prefix + 'ibias<0>__silk_dc[current]'] if nested else ['@ibias<0>__silk_pwl[current]', '@ibias<0>__silk_dc[current]']) if split else ['v.' + prefix + 'vbias<0>' if nested else 'vbias<0>']
    tb = _bench(tmp_path, names, [] if nested else [source], data=[[.5, 1], [.2, .2]] if split else [[3., 4.]])
    if nested:
        child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
        child.define_module_ports({})
        child.add_source(source)
        tb.netlist.add_module(child, 'branch', nets={})
    original = module_test._branch_vector_name
    calls = []

    def lookup(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(module_test, '_branch_vector_name', lookup)
    path = 'i(' + prefix + 'bias[0])'
    signal = module_test._load_metric_path(tb.netlist, path) if metric else tb.netlist.load_signal(path)
    np.testing.assert_allclose(signal.y, [.7, 1.2] if split else [3., 4.])
    assert len(calls) == 1


@pytest.mark.parametrize('path,selector', [('a', 'V'), ('v(a)', 'V'), ('vbias', 'I'),
                                         ('i(vbias)', 'V'), ('i(vbias)', 'I'),
                                         ('i(unknown)', 'V')])
def test_existing_raw_reads_bypass_shared_lookup(tmp_path, monkeypatch, path, selector):
    from silk.circuit import module_test
    tb = _bench(tmp_path, ['vbias', 'unknown'], [VS('bias', 'a', '0', dc=99)], data=[[3., 4.], [7., 8.]])
    plot = tb.netlist._raw.native.plots[0]
    plot.vectors.append(RawVector('v(a)', 'voltage', data=np.array([1., 2.])))

    def forbidden(*args, **kwargs):
        pytest.fail('raw read called logical source lookup')

    monkeypatch.setattr(module_test, '_branch_vector_name', forbidden)
    np.testing.assert_array_equal(tb.netlist.load_signal(path, selector).y,
                                  [1., 2.] if path in ('a', 'v(a)') else [7., 8.] if path == 'i(unknown)' else [3., 4.])


@pytest.mark.parametrize('path', ['i(xbranch.vsense)', 'i(xbranch.unknown)', 'i(xbranch.sense<0>)'])
def test_real_parent_does_not_make_raw_current_leaf_authored(tmp_path, monkeypatch, path):
    from silk import Module
    from silk.circuit import module_test
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({})
    child.add_source(VS('sense[0]', 'a', '0', dc=99))
    tb = _bench(tmp_path, [path[2:-1]], data=[[3., 4.]])
    tb.netlist.add_module(child, 'branch', nets={})
    plot = tb.netlist._raw.native.plots[0]
    plot.vectors.append(RawVector(path[2:-1], 'current', data=np.array([7., 8.])))

    def forbidden(*args, **kwargs):
        pytest.fail('unmatched leaf entered logical lookup')

    monkeypatch.setattr(module_test, '_branch_vector_name', forbidden)
    np.testing.assert_array_equal(tb.netlist.load_signal(path).y, [3., 4.])


def test_typed_literal_and_empty_readers_bypass_current_scan(tmp_path, monkeypatch):
    from silk import Module
    from silk.circuit import module_test
    from silk.circuit.node_paths import NodeVoltageRequest
    from silk.circuit.placement_reference import LiteralVoltageName
    tb = _bench(tmp_path, ['bias'], [VS('bias', 'a', '0', dc=99)], data=[[3., 4.]])
    plot = tb.netlist._raw.native.plots[0]
    plot.vectors.append(RawVector('v(i(bias))', 'voltage', data=np.array([7., 8.])))

    def forbidden(*args, **kwargs):
        pytest.fail('typed/literal/raw reader called logical lookup')

    monkeypatch.setattr(module_test, '_branch_vector_name', forbidden)
    np.testing.assert_array_equal(tb.netlist.load_signal(NodeVoltageRequest('i(bias)')).y, [3., 4.])
    np.testing.assert_array_equal(tb.netlist.load_signal(LiteralVoltageName('i(bias)')).y, [7., 8.])
    empty = Module(None, str(tmp_path), 'empty.cir', None)
    empty.load_file(str(tmp_path / 's.raw'))
    empty.read_raw_file()
    np.testing.assert_array_equal(empty.load_signal('i(bias)').y, [3., 4.])


def test_rebuilt_current_reader_uses_compatible_external_file(tmp_path):
    tb = _bench(tmp_path, ['vsense'], [VS('sense', 'a', '0', dc=99)], data=[[3., 4.]])
    path = str(tmp_path / 's.raw')
    rebuilt = _build(tmp_path / 'rebuilt', [], [])
    rebuilt.add_source(VS('sense', 'a', '0', dc=99))
    report = rebuilt.evaluate(rawfile=path)
    np.testing.assert_array_equal(report.sim_results.load_signal('i(sense)').y, [3., 4.])
    with pytest.raises(SimulationError, match='not found'):
        report.sim_results.load_signal('sense', 'I')
    np.testing.assert_array_equal(rebuilt.netlist.load_signal('i(sense)').x, [0., 1e-8])


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('selector', ['V', 'I'])
def test_raw_physical_read_keeps_base_adapter_enumeration(tmp_path, monkeypatch, nested, selector):
    from silk import Module
    from silk.sources import _layout
    from silk.circuit.modules_results import _source_current_probe_name
    name = 'v.xbranch.vsense' if nested else 'vsense'
    source = VS('sense', 'a', '0', dc=99)
    tb = _bench(tmp_path, [name], [] if nested else [source], data=[[3., 4.]])
    if nested:
        child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
        child.define_module_ports({})
        child.add_source(source)
        tb.netlist.add_module(child, 'branch', nets={})
    calls = []
    instances, layout = _layout.source_instances, _layout.source_layout

    def count_instances(*args, **kwargs):
        calls.append('instances')
        yield from instances(*args, **kwargs)

    def count_layout(*args, **kwargs):
        calls.append('layout')
        return layout(*args, **kwargs)

    monkeypatch.setattr(_layout, 'source_instances', count_instances)
    monkeypatch.setattr(_layout, 'source_layout', count_layout)
    path = 'i(' + name + ')'
    module, raw = tb.netlist, tb.netlist._raw
    physical = _source_current_probe_name(module, raw, path, selector, 'tran')
    expected = module.backend.load_signal(raw, physical, type=selector, analysis='tran')
    baseline_calls = calls[:]
    calls.clear()
    actual = module.load_signal(path, selector, analysis='tran')
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(actual.y, expected.y)
    assert calls == baseline_calls


class _NoSourceKeyIteration(dict):
    def __iter__(self):
        raise AssertionError('known-card gate scanned authored source keys')

    def items(self):
        raise AssertionError('known-card read scanned authored source items')


@pytest.mark.parametrize('placement,leaf', [(None, 'vs'), ('Branch', 'vs'),
                                            ('Branch[1]', 'vs[2]')])
@pytest.mark.parametrize('selector', ['V', 'I'])
@pytest.mark.parametrize('reader_kind', ['module', 'retained', 'private'])
def test_registered_card_gate_uses_index_without_source_scan(
        tmp_path, monkeypatch, placement, leaf, selector, reader_kind):
    from silk import Module
    from silk.circuit import module_test
    from silk.circuit.test_placement import EvidenceResults
    from silk.simulation._setup_check import private_view
    physical = (('x' + placement.lower() + '.') if placement else '') + leaf
    angle = physical.replace('[', '<').replace(']', '>')
    tb = _bench(tmp_path, [angle, 'v.' + angle] if placement else [angle],
                data=[[3., 4.], [3., 4.]] if placement else [[3., 4.]])
    owner = tb.netlist
    if placement:
        owner = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
        owner.define_module_ports({})
        tb.netlist.add_module(owner, placement, nets={})
    owner.add_source(VS(leaf[1:], 'a', '0', dc=1))
    reader = tb.netlist
    if reader_kind == 'private':
        reader = private_view(reader)
        if placement:
            copied = reader._element_mgr.elem_dict[placement]['device']
            assert copied is not owner
            # The registry tuple still names the original child. Only the slot
            # entry reaches this index; a stale-object walk would miss it.
            owner._element_mgr._instance_names = {}
            owner = copied
    elif reader_kind == 'retained':
        reader = EvidenceResults(reader)
    owner.sig_sources_dict = _NoSourceKeyIteration(owner.sig_sources_dict)
    import inspect
    original_elements = Module.elem_dict
    def synchronized_elements(module):
        caller = inspect.currentframe().f_back
        assert caller.f_code.co_name != 'load_signal', 'gate refreshed public element pins'
        return original_elements.fget(module)
    monkeypatch.setattr(Module, 'elem_dict', property(synchronized_elements))
    monkeypatch.setattr(module_test, '_branch_vector_name',
                        lambda *a, **k: pytest.fail('known card used logical lookup'))
    for spelling in [physical, angle, physical.upper().replace('.', ':')]:
        np.testing.assert_array_equal(reader.load_signal(f'i({spelling})', selector).y, [3., 4.])
        if selector == 'I':
            np.testing.assert_array_equal(reader.load_signal(spelling, 'I').y, [3., 4.])


@pytest.mark.parametrize('kind', ['resistor', 'module'])
@pytest.mark.parametrize('saved', [False, True])
@pytest.mark.parametrize('selector', ['V', 'I'])
def test_non_source_card_precedes_competing_authored_source(tmp_path, kind, saved, selector):
    from silk import Module
    from silk.elements import ElementLibrary
    from silk.circuit.module_test import _load_metric_path
    card = 'rload' if kind == 'resistor' else 'xbranch'
    tb = _bench(tmp_path, [card, 'v' + card] if saved else ['v' + card],
                [VS(card, 'b', '0', dc=1)],
                data=[[3., 4.], [7., 8.]] if saved else [[7., 8.]])
    if kind == 'resistor':
        tb.netlist.add_passive(ElementLibrary().resistor('load', p='a', n='0', value=1000))
    else:
        child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
        child.define_module_ports({})
        tb.netlist.add_module(child, 'branch', nets={})
    if saved:
        np.testing.assert_array_equal(tb.netlist.load_signal(f'i({card})', selector).y, [3., 4.])
    else:
        with pytest.raises(SimulationError):
            tb.netlist.load_signal(f'i({card})', selector)
    with pytest.warns(UserWarning, match='emitted reading'):
        if saved:
            np.testing.assert_array_equal(_load_metric_path(tb.netlist, f'i({card})').y, [3., 4.])
        else:
            with pytest.raises(SimulationError, match='not found'):
                _load_metric_path(tb.netlist, f'i({card})')



@pytest.mark.parametrize("spelling,selector", [
    ('i(dut.core.ref)', 'V'), ('dut.core.ref', 'I'),
    ('i(DUT:CORE:REF)', 'I'), ('i(xdut.xcore.ref)', 'V'),
])
def test_designer_source_current_prototype(tmp_path, spelling, selector):
    from silk import Module
    from silk.circuit.module_test import _current_probe_names, _load_metric_path
    child = Module(None, str(tmp_path), 'core.cir', None, module_name='core')
    child.define_module_ports({'ref': 'output'})
    child.add_source(VS('ref', 'local', '0', dc=9))
    tb = _bench(tmp_path, ['v.xdut.xcore.vref'], data=[[3., 4.]])
    tb.netlist.elem_dict['dut']['device'].add_module(child, 'core', {'ref': 'wire'})
    if not spelling.startswith('i('):
        with pytest.raises(ValueError, match='physical'):
            tb.probe(spelling, selector)
        spelling = f'i({spelling})'
    probe = tb.probe(spelling, selector)
    np.testing.assert_array_equal(probe.y, [3., 4.])
    np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(dut.core.ref)').y, [3., 4.])
    np.testing.assert_array_equal(tb.netlist.load_signal('i(dut.core.ref)').y, [3., 4.])
    assert _current_probe_names(tb._probe_registry) == ('xdut.xcore.Vref',)


@pytest.mark.parametrize('wrapped', [False, True])
def test_current_probe_re_resolves_changed_source_layout(tmp_path, wrapped):
    from silk.circuit.module_test import _current_probe_names
    src = sources.CurrentSource('ibias', 'a', '0', dc=.2)
    tb = _bench(tmp_path, ['@iibias[current]', '@iibias__silk_pwl[current]',
                           '@iibias__silk_dc[current]'], [src],
                data=[[.2, .2], [.5, 1.], [.2, .2]])
    if not wrapped:
        with pytest.raises(ValueError, match='physical'):
            tb.probe('ibias', 'I')
    probe = tb.probe('i(ibias)', 'I')
    np.testing.assert_array_equal(probe.y, [.2, .2])
    replacement = sources.CurrentSource('ibias', 'a', '0', dc=.2, pwl=[(0, .5), (1e-8, 1.)])
    rebuilt = _bench(tmp_path / 'rebuilt', ['@iibias__silk_pwl[current]',
                     '@iibias__silk_dc[current]'], [replacement], data=[[.5, 1.], [.2, .2]])
    tb.netlist = rebuilt.netlist
    np.testing.assert_allclose(probe.y, [.7, 1.2])
    assert _current_probe_names(tb._probe_registry) == ('Iibias__silk_pwl', 'Iibias__silk_dc')
    tb.netlist.sig_sources_dict.pop('ibias')
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='ibias.*no longer present'):
        _ = probe.y


def test_current_probe_emitted_precedence_prototype(tmp_path):
    from silk.circuit.module_test import _load_metric_path
    tb = _bench(tmp_path, ['vvs', 'vvvs'],
                [VS('vs', 'a', '0', dc=1), VS('vvs', 'b', '0', dc=2)],
                data=[[3., 4.], [8., 9.]])
    with pytest.warns(UserWarning, match='emitted reading'):
        p = tb.probe('i(vvs)')
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(p.y, [3., 4.])
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(vvs)').y, [3., 4.])


def test_current_probe_external_row_precedence_prototype(tmp_path):
    from silk.circuit.module_test import _load_metric_path
    tb = _bench(tmp_path, ['sense', 'vsense'], [VS('sense', 'a', '0', dc=9)],
                data=[[3., 4.], [8., 9.]])
    p = tb.probe('i(sense)')
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(p.y, [3., 4.])
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(sense)').y, [3., 4.])



def test_physical_probe_read_does_not_reinterpret_card(tmp_path, monkeypatch):
    from silk.circuit import module_test
    tb = _bench(tmp_path, ['vvs'], [VS('vs', 'a', '0', dc=9)], data=[[3., 4.]])
    probe = tb.probe('i(vvs)')
    monkeypatch.setattr(module_test, '_branch_vector_name',
                        lambda *a, **k: pytest.fail('physical probe repeated logical lookup'))
    np.testing.assert_array_equal(probe.y, [3., 4.])



@pytest.mark.parametrize('function_form', [False, True])
def test_string_current_metric_keeps_missing_target_error(tmp_path, function_form):
    from silk.metrics import average
    from silk.exceptions import CircuitError
    tb = _bench(tmp_path, ['vsense'], [VS('sense', 'a', '0', dc=9)], data=[[3., 4.]])
    if function_form:
        tb.add_metric('current_mean', average, sig='i(sense)')
    else:
        tb.add_metric('current_mean', type='average', signal='i(sense)')
    metric = tb._metrics[-1]
    assert metric['extractor'](tb.netlist) == pytest.approx(3.5)
    tb.netlist.sig_sources_dict.pop('sense')
    with pytest.raises(CircuitError, match='sense.*not found'):
        metric['extractor'](tb.netlist)


@pytest.mark.parametrize('split', [False, True])
def test_nested_array_current_source_uses_shared_address_and_capture(tmp_path, split):
    from silk import Module
    from silk.circuit.module_test import _current_probe_names, _load_metric_path
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import Transient
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({})
    child.add_source(sources.CurrentSource('ib[0]', 'local', '0', dc=.2,
        **({'pwl': [(0, .5), (1e-8, 1.)]} if split else {})))
    cards = (['xdut.xcore<0>.Iib<0>__silk_pwl', 'xdut.xcore<0>.Iib<0>__silk_dc']
             if split else ['xdut.xcore<0>.Iib<0>'])
    tb = _bench(tmp_path, ['@i.' + card.lower() + '[current]' for card in cards],
                data=[[.5, 1.], [.2, .2]] if split else [[.2, .2]])
    tb.netlist.elem_dict['dut']['device'].add_module(child, 'core[0]', {})
    expected = [.7, 1.2] if split else [.2, .2]
    for spelling in ('i(dut.core[0].ib[0])', 'i(XDUT:XCORE[0]:IB[0])'):
        np.testing.assert_allclose(tb.probe(spelling).y, expected)
        np.testing.assert_allclose(_load_metric_path(tb.netlist, spelling).y, expected)
        np.testing.assert_allclose(tb.netlist.load_signal(spelling).y, expected)
        with pytest.raises(SimulationError, match='not found'):
            tb.netlist.load_signal(spelling[2:-1], 'I')
    with pytest.raises(ValueError, match='physical'):
        tb.probe('dut.core[0].ib[0]', 'I')
    assert _current_probe_names(tb._probe_registry) == tuple(cards)
    netlister = PySpiceNetlister(None)
    args = (str(tmp_path), {'tran': Transient(1e-9, 1e-8)})
    logical = netlister.generate_output_block(*args, save_list=['out'],
        current_probe_names=_current_probe_names(tb._probe_registry), source_instance_names=cards)
    physical = netlister.generate_output_block(*args, save_list=['out'],
        current_probe_names=cards, source_instance_names=cards)
    assert logical == physical


def test_distinct_structural_source_targets_name_current_remedy(tmp_path):
    from silk import Module
    from silk.exceptions import CircuitError
    tb = _bench(tmp_path, [])
    for name in ('core', 'xcore'):
        child = Module(None, str(tmp_path), name + '.cir', None, module_name=name)
        child.define_module_ports({})
        child.add_source(VS('sense', 'local', '0', dc=1))
        tb.netlist.add_module(child, name, {})
    with pytest.raises(CircuitError, match='Source path.*ambiguous.*physical current'):
        tb.probe('i(xcore.sense)')



@pytest.mark.parametrize('kind', ['voltage', 'current', 'split'])
def test_registered_physical_source_adapter_uses_card_owner(tmp_path, monkeypatch, kind):
    from silk import Module
    from silk.sources import _layout
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({})
    cls = VS if kind == 'voltage' else sources.CurrentSource
    child.add_source(cls('bias', 'local', '0', dc=.2,
                         **({'pwl': [(0,.5),(1e-8,1.)]} if kind == 'split' else {})))
    leaf = 'vbias' if kind == 'voltage' else 'ibias' + ('__silk_pwl' if kind == 'split' else '')
    row = 'v.xchild.' + leaf if kind == 'voltage' else '@i.xchild.' + leaf + '[current]'
    tb = _bench(tmp_path, [row], data=[[3.,4.]])
    tb.netlist.add_module(child, 'child', {})
    monkeypatch.setattr(_layout, 'source_instances',
                        lambda *a, **k: pytest.fail('physical current re-enumerated sources'))
    np.testing.assert_array_equal(tb.netlist.load_signal('xchild.' + leaf, 'I').y, [3.,4.])


def test_current_collision_warning_has_pasteable_alternative(tmp_path):
    import warnings
    tb = _bench(tmp_path, ['vvs', 'vvvs'],
                [VS('vs','a','0',dc=1),VS('vvs','b','0',dc=2)],data=[[3.,4.],[8.,9.]])
    with pytest.warns(UserWarning) as caught:
        tb.probe('i(Vvs)')
    message = str(caught[0].message)
    assert 'i(vvs)' in message and "tb.probe('Vvvs', 'I')" in message
    assert "(('" not in message and ', 1)' not in message
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        np.testing.assert_array_equal(tb.probe('i(Vvvs)').y, [8.,9.])


@pytest.mark.parametrize('path', ['bias', 'child.bias', 'XCHILD:BIAS'])
@pytest.mark.parametrize('receiver', ['module', 'retained', 'private'])
def test_bare_current_source_load_keeps_physical_meaning_on_miss(tmp_path, path, receiver):
    from silk import Module
    tb = _bench(tmp_path, ['vbias' if path == 'bias' else 'v.xchild.vbias'],
                [VS('bias','a','0',dc=1)] if path == 'bias' else [], data=[[3.,4.]])
    if path != 'bias':
        child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
        child.define_module_ports({})
        child.add_source(VS('bias','local','0',dc=1))
        tb.netlist.add_module(child, 'child', {})
    reader = tb.netlist
    if receiver == 'retained':
        from silk.circuit.test_placement import EvidenceResults
        reader = EvidenceResults(reader, _retained=True)
    elif receiver == 'private':
        from silk.simulation._setup_check import private_view
        reader = private_view(reader)
    with pytest.raises(SimulationError, match='not found'):
        reader.load_signal(path, 'I')


def test_bare_current_source_does_not_replace_existing_external_row(tmp_path):
    tb = _bench(tmp_path, ['sense','vsense'], [VS('sense','a','0',dc=1)],data=[[3.,4.],[8.,9.]])
    np.testing.assert_array_equal(tb.netlist.load_signal('sense','I').y,[3.,4.])


@pytest.mark.parametrize('split', [False, True])
def test_bare_current_source_load_never_uses_logical_lookup(tmp_path, monkeypatch, split):
    from silk.circuit import module_test
    source = sources.CurrentSource('bias', 'a', '0', dc=.2,
        **({'pwl': [(0, .5), (1e-8, 1.)]} if split else {}))
    names = ['@ibias__silk_pwl[current]', '@ibias__silk_dc[current]'] if split else ['@ibias[current]']
    tb = _bench(tmp_path, names, [source], data=[[.5, 1.], [.2, .2]] if split else [[.2, .2]])
    monkeypatch.setattr(module_test, '_branch_vector_name',
                        lambda *a, **k: pytest.fail('physical read used logical lookup'))
    with pytest.raises(SimulationError, match='not found'):
        tb.netlist.load_signal('bias', 'I')



def test_bare_current_source_keeps_ambiguous_external_rows(tmp_path):
    tb = _bench(tmp_path, ['sense[0]', 'sense<0>', 'vsense<0>'],
                [VS('sense[0]', 'a', '0', dc=1)], data=[[3.,4.],[5.,6.],[8.,9.]])
    with pytest.raises(AmbiguousSignalError):
        tb.netlist.load_signal('sense[0]', 'I')


@pytest.mark.parametrize('spelling,selector', [('vsense', 'I'), ('i(vsense)', 'V')])
def test_physical_probe_does_not_borrow_authored_source_after_rebuild(tmp_path, spelling, selector):
    from silk.exceptions import SimulationError
    tb = _bench(tmp_path / 'old', ['vsense'], [VS('sense', 'a', '0', dc=1)], data=[[3., 4.]])
    probe = tb.probe(spelling, selector)
    np.testing.assert_array_equal(probe.y, [3., 4.])
    rebuilt = _bench(tmp_path / 'new', ['vvsense'], [VS('vsense', 'a', '0', dc=2)], data=[[8., 9.]])
    tb.netlist = rebuilt.netlist
    with pytest.raises(SimulationError, match='not found'):
        _ = probe.y


@pytest.mark.parametrize('slot', ['string', 'probe'])
def test_removed_current_target_keeps_string_and_probe_contracts(tmp_path, slot):
    from silk.metrics import average
    from silk.exceptions import MetricBindingError
    tb = _bench(tmp_path, ['vsense'], [VS('sense', 'a', '0', dc=1)], data=[[3., 4.]])
    tb._metrics.clear()
    tb.add_metric('current_mean', average, sig='i(sense)' if slot == 'string' else tb.probe('i(sense)'))
    tb.netlist.sig_sources_dict.pop('sense')
    if slot == 'string':
        with pytest.raises(MetricBindingError, match='current_mean.*sense.*not found'):
            tb.evaluate(rawfile=tb.netlist.results_file)
    else:
        assert tb.evaluate(rawfile=tb.netlist.results_file).metrics['current_mean'] == pytest.approx(3.5)


def test_ac_collision_validates_winning_voltage_card_only(tmp_path):
    from silk.circuit.module_test import _load_metric_path
    from silk.exceptions import NotSupportedError
    tb = _bench(tmp_path, [], [VS('s', 'a', '0', dc=1), sources.CurrentSource('vs', 'a', '0', dc=.2)])
    path = tmp_path / 'ac.raw'
    values = np.array([3.+1j, 4.+2j])
    write_rawfile(path, RawFile([RawPlot({'plotname': 'AC Analysis'}, [
        RawVector('frequency', 'frequency', data=np.array([1., 2.])),
        RawVector('i(vs)', 'current', data=values)], npoints=2, real=False, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    with pytest.warns(UserWarning, match='emitted reading'):
        probe = tb.probe('i(vs)', analysis='ac')
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(probe.y, values)
    with pytest.warns(UserWarning, match='emitted reading'):
        np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(vs)', analysis='ac').y, values)
    np.testing.assert_array_equal(tb.probe('vs', 'I', analysis='ac').y, values)
    with pytest.raises(NotSupportedError):
        tb.netlist.load_signal('ivs', 'I', analysis='ac')


@pytest.mark.parametrize('name', ['ik', 'ibias', 'xdut.xcore.ib'])
def test_bare_authored_current_probe_requires_explicit_logical_spelling(tmp_path, name):
    from silk import Module
    if '.' in name:
        dut = Module(None, str(tmp_path / 'dut'), 'dut.cir', None, module_name='dut')
        child = Module(None, str(tmp_path / 'core'), 'core.cir', None, module_name='core')
        dut.define_module_ports({})
        child.define_module_ports({})
        child.add_source(sources.CurrentSource('ib', 'a', '0', dc=.2))
        dut.add_module(child, 'core', {})
        tb = dut.create_test('t')
        tb.add_module_under_test({})
    else:
        source = sources.CurrentSource(name, 'a', '0', dc=.2,
            **({'pwl': [(0, .5), (1e-8, 1.)]} if name == 'ibias' else {}))
        tb = _bench(tmp_path, [], [source])
    with pytest.raises(ValueError, match='physical.*i\\('):
        tb.probe(name, 'I')
    assert tb._probe_registry == []
    logical = tb.probe(f'i({name})')
    assert logical.__dict__['_source_request'][0] == name
    assert logical.__dict__['_source_request'][1]


def test_metric_declaration_keeps_runtime_source_lookup(tmp_path, monkeypatch):
    from silk.circuit import node_paths
    from silk.metrics import average
    tb = _bench(tmp_path, ['vsense'], [VS('sense', 'a', '0', dc=1)], data=[[3., 4.]])
    monkeypatch.setattr(node_paths, 'node_candidates',
                        lambda *a, **k: pytest.fail('metric declaration enumerated source inventory'))
    tb.add_metric('function', average, sig='i(sense)')
    tb.add_metric('registered', type='average', signal='i(sense)')
    assert all('source_requests' not in metric for metric in tb._metrics)


@pytest.mark.parametrize('kind', ['node', 'voltage-source', 'split-current'])
@pytest.mark.parametrize('saved', [False, True])
def test_removed_probe_target_uses_only_declared_retained_rows(tmp_path, kind, saved):
    from silk.metrics import average
    from silk.circuit.module_test import _current_probe_names
    split = kind == 'split-current'
    source = (sources.CurrentSource('sense', 'a', '0', dc=.5, pwl=[0, 3, 1e-8, 4])
              if split else VS('sense', 'a', '0', dc=1))
    tb = _bench(tmp_path, [], [] if kind == 'node' else [source])
    tb._metrics.clear()
    if kind == 'node':
        tb.netlist.add_net('gone')
        probe = tb.probe('gone')
        names = ['v(gone)']
        tb.netlist.remove_net('gone')
    else:
        probe = tb.probe('i(sense)')
        names = ['i(Isense__silk_pwl)', 'i(Isense__silk_dc)'] if split else ['i(Vsense)']
        tb.netlist.sig_sources_dict.pop('sense')
    tb.add_metric('mean', average, sig=probe)
    vectors = [RawVector('time', 'time', data=np.array([0., 1e-8]))]
    for i, name in enumerate(names if saved else []):
        vectors.append(RawVector(name, 'voltage' if kind == 'node' else 'current',
                                 data=np.array([3., 4.] if i == 0 else [.5, .5])))
    path = tmp_path / 'retained.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors,
                                         npoints=2, real=True, padded=True)]))
    if saved:
        assert tb.evaluate(rawfile=str(path)).metrics['mean'] == pytest.approx(4. if split else 3.5)
    else:
        from silk.exceptions import CircuitError
        tb.netlist.load_file(str(path))
        tb.netlist.read_raw_file()
        with pytest.raises(CircuitError, match='no longer present'):
            _ = probe.y
    if kind != 'node':
        # Reading an old result does not authorize capturing a removed source.
        with pytest.raises(SimulationError, match='no longer present'):
            _current_probe_names(tb._probe_registry)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('spelling', ['i(s)', 'i(vs)', 'vs'])
def test_ac_preparation_checks_physical_winner(tmp_path, simulator, reverse, spelling):
    import warnings
    from pathlib import Path
    from silk.simulation import AC
    from silk.exceptions import NotSupportedError
    srcs = [VS('s', 'a', '0', ac=1), sources.CurrentSource('vs', 'a', '0', ac=.2)]
    if reverse:
        srcs.reverse()
    tb = _bench(tmp_path, [], srcs)
    tb._metrics.clear()
    tb.netlist.sim_analysis_dict.clear()
    tb.add_sim_type(AC('lin', 2, 1, 2))
    tb.netlist.simulator = simulator
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tb.probe(spelling, 'I', analysis='ac')
        path = tb.prepare()
    assert any('emitted reading' in str(w.message) for w in caught) == (spelling != 'i(s)')
    assert Path(path).is_file()
    tb.probe('ivs', 'I', analysis='ac')
    with pytest.raises(NotSupportedError, match="CurrentSource 'vs'"):
        tb.prepare()


@pytest.mark.parametrize('reverse', [False, True])
def test_ac_preparation_refuses_physical_current_despite_logical_collision(tmp_path, reverse):
    from silk.simulation import AC
    from silk.exceptions import NotSupportedError
    srcs = [sources.CurrentSource('s', 'a', '0', ac=1),
            sources.CurrentSource('is', 'a', '0', ac=2)]
    if reverse:
        srcs.reverse()
    tb = _bench(tmp_path, [], srcs)
    tb._metrics.clear()
    tb.netlist.sim_analysis_dict.clear()
    tb.add_sim_type(AC('lin', 2, 1, 2))
    with pytest.warns(UserWarning, match='emitted reading'):
        tb.probe('i(is)', analysis='ac')
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(NotSupportedError, match="CurrentSource 's'"):
            tb.prepare()
    assert not caught


@pytest.mark.parametrize('name', ['sense', 'external'])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_external_bare_current_resolves_owner_and_card_once(tmp_path, monkeypatch, name, simulator):
    from silk.sources import _layout
    tb = _bench(tmp_path, [name, 'vsense'], [VS('sense', 'a', '0', dc=1)],
                data=[[3., 4.], [8., 9.]])
    tb.netlist.simulator = simulator
    calls = []
    original = _layout.current_source_card

    def lookup(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(_layout, 'current_source_card', lookup)
    # An external row keeps its own values, including beside an authored alias.
    np.testing.assert_array_equal(tb.netlist.load_signal(name, 'I').y, [3., 4.])
    assert calls == [name]


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('operands', ['complete', 'missing', 'ambiguous'])
def test_removed_source_probe_reads_native_hierarchical_rows(tmp_path, simulator, split, operands):
    from silk import Module
    from silk.circuit.module_test import _current_probe_names, _load_metric_path
    tb = _bench(tmp_path, [])
    tb.netlist.simulator = simulator
    leaf = Module(None, str(tmp_path), 'leaf.cir', None, module_name='leaf')
    leaf.add_source(VS('sense', 'a', '0', dc=0))
    leaf.add_source(sources.CurrentSource('bias', '0', 'a', dc=.0002,
                   **({'pwl': [(0, 0), (1e-8, .0003)]} if split else {})))
    tb.netlist.elem_dict['dut']['device'].add_module(leaf, 'core[0]', {})
    probes = {name: tb.probe(f'i(dut.core[0].{name})') for name in ('sense', 'bias')}
    if simulator == 'ngspice':
        names = ['i(v.xdut.xcore<0>.vsense)'] + (
            ['@i.xdut.xcore<0>.ibias__silk_pwl[current]',
             '@i.xdut.xcore<0>.ibias__silk_dc[current]'] if split else
            ['@i.xdut.xcore<0>.ibias[current]'])
    else:
        names = ['I(XDUT:XCORE<0>:VSENSE)'] + (
            ['I(XDUT:XCORE<0>:IBIAS__SILK_PWL)',
             'I(XDUT:XCORE<0>:IBIAS__SILK_DC)'] if split else
            ['I(XDUT:XCORE<0>:IBIAS)'])
    data = [[.0002, .0005 if split else .0002], [0, .0003], [.0002, .0002]] if split else [
        [.0002, .0002], [.0002, .0002]]
    vectors = [RawVector('time', 'time', data=np.array([0., 1e-8]))]
    vectors.extend(RawVector(name, 'current', data=np.array(values))
                   for name, values in zip(names, data))
    path = tmp_path / 'native.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors,
                                        npoints=2, real=True, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    expected = {name: probe.snapshot() for name, probe in probes.items()}
    leaf.sig_sources_dict.pop('sense')
    leaf.sig_sources_dict.pop('bias')
    if operands == 'missing':
        vectors.pop()
    elif operands == 'ambiguous':
        vectors.append(RawVector(names[-1], 'current', data=np.array([8., 9.])))
    path = tmp_path / 'after-removal.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors,
                                        npoints=2, real=True, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    got = probes['sense'].snapshot()
    np.testing.assert_array_equal(got.x, expected['sense'].x)
    np.testing.assert_array_equal(got.y, expected['sense'].y)
    if operands == 'complete':
        got = probes['bias'].snapshot()
        np.testing.assert_array_equal(got.x, expected['bias'].x)
        np.testing.assert_array_equal(got.y, expected['bias'].y)
        assert (got.xlabel, got.ylabel) == (expected['bias'].xlabel, expected['bias'].ylabel)
    else:
        with pytest.raises(AmbiguousSignalError if operands == 'ambiguous' else SimulationError):
            probes['bias'].snapshot()
    with pytest.raises(SimulationError):
        _load_metric_path(tb.netlist, 'i(dut.core[0].bias)')
    with pytest.raises(SimulationError, match='no longer present'):
        _current_probe_names(tb._probe_registry)


@pytest.mark.parametrize('case', ['single', 'split', 'raw-winner', 'removed'])
def test_current_leaf_resolves_plot_and_saved_rows_once(tmp_path, monkeypatch, case):
    import warnings
    from silk.circuit import node_paths
    from silk.simulation import backend
    from silk.signals import spice
    from silk import vector_match
    src = (sources.CurrentSource('sense', 'a', '0', dc=.5, pwl=[0, 3, 1e-8, 4])
           if case == 'split' else VS('sense', 'a', '0', dc=1))
    tb = _bench(tmp_path, [], [src])
    probe = tb.probe('i(sense)', analysis='tran')
    rows = ([('@Isense__silk_pwl[current]', [3., 4.]),
             ('@Isense__silk_dc[current]', [.5, .5])] if case == 'split'
            else [('i(Vsense)', [3., 4.])])
    if case == 'raw-winner':
        rows.append(('i(sense)', [8., 9.]))
    if case == 'removed':
        tb.netlist.sig_sources_dict.pop('sense')
    vectors = [RawVector('time', 'time', data=np.array([0., 1e-8]))]
    vectors += [RawVector(name, 'current', data=np.array(values)) for name, values in rows]
    path = tmp_path / 'one-resolution.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors,
                                        npoints=2, real=True, padded=True)]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    native = tb.netlist._raw.native
    real_plot = native.plots[0]
    original_select = backend.select_plot
    original_match = vector_match.collect_vector_candidates
    selections, matches = [], []
    def select(raw, analysis):
        selections.append((raw, analysis))
        return original_select(raw, analysis)
    def match(plot, name, signal_type='V', *, first=1, _requests=None):
        if plot is real_plot or plot is native:
            matches.extend((str(request), signal_type, first)
                           for request in ((name,) if _requests is None else _requests))
        if _requests is None:
            return original_match(plot, name, signal_type, first=first)
        return original_match(plot, name, signal_type, first=first, _requests=_requests)
    monkeypatch.setattr(backend, 'select_plot', select)
    monkeypatch.setattr(vector_match, 'collect_vector_candidates', match)
    monkeypatch.setattr(node_paths, 'collect_vector_candidates', match)
    monkeypatch.setattr(spice, 'collect_vector_candidates', match)
    before = dict(vars(probe))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        got = probe.snapshot()
    np.testing.assert_array_equal(got.x, [0., 1e-8])
    np.testing.assert_array_equal(got.y, [3.5, 4.5] if case == 'split' else
                                  [8., 9.] if case == 'raw-winner' else [3., 4.])
    assert len(selections) == 1
    assert len(matches) == len(set(matches)), matches
    assert bool(caught) == (case == 'raw-winner')
    assert vars(probe) == before  # The read leaves no selection or row state on the probe.


def test_current_path_keeps_custom_backend_loader_and_keyword_refusal(tmp_path, monkeypatch):
    from silk.circuit.module_test import _load_metric_path
    tb = _bench(tmp_path, ['Vsense'], [VS('sense', 'a', '0', dc=1)])
    with pytest.raises(TypeError, match='unexpected_keyword'):
        _load_metric_path(tb.netlist, 'i(sense)', unexpected_keyword=True)
    seen = []
    expected = tb.netlist.load_signal('Vsense', 'I')
    def custom_loader(raw, name, *, type='V', analysis=None):
        seen.append((name, type, analysis))
        return expected
    backend = tb.netlist.backend
    monkeypatch.setattr(backend, 'load_signal', custom_loader)
    monkeypatch.setattr(type(tb.netlist), 'backend', property(lambda self: backend))
    assert _load_metric_path(tb.netlist, 'i(sense)', analysis='tran') is expected
    assert seen == [('Vsense', 'I', 'tran')]


def test_external_current_metric_keeps_implicit_device_plot_selection(tmp_path):
    from silk.circuit.module_test import _load_metric_path
    tb = _bench(tmp_path, ['external'], data=[[3., 4.]])
    tb.netlist.save_device_params = True
    transient = tb.netlist._raw.native.plots[0]
    op = RawPlot({'plotname': 'Operating Point'},
                 [RawVector('i(external)', 'current', data=np.array([9.]))],
                 npoints=1, real=True, padded=True)
    path = tmp_path / 'op-and-transient.raw'
    write_rawfile(path, RawFile([op, transient]))
    tb.netlist.load_file(str(path)); tb.netlist.read_raw_file()
    np.testing.assert_array_equal(_load_metric_path(tb.netlist, 'i(external)').y, [3., 4.])


def test_current_collision_warns_once_per_explicit_read_not_capture(tmp_path):
    import warnings
    from silk.circuit.module_test import _current_probe_names, _load_metric_path
    tb = _bench(tmp_path, ['vvs', 'vvvs'],
                [VS('vs', 'a', '0', dc=1), VS('vvs', 'b', '0', dc=2)],
                data=[[3., 4.], [8., 9.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probe = tb.probe('i(vvs)')
    assert len(caught) == 1
    assert 'Current request i(vvs)' in str(caught[0].message)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert _current_probe_names(tb._probe_registry) == ('vvs',)
    assert not caught
    for read in (lambda: probe.y, lambda: _load_metric_path(tb.netlist, 'i(vvs)').y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            np.testing.assert_array_equal(read(), [3., 4.])
        assert len(caught) == 1
        assert 'Current request i(vvs)' in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning, match='emitted reading'):
            _ = probe.y


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_removed_source_multiplot_capability_after_saved_resolution(
        tmp_path, simulator, nested, split, missing):
    from silk import Module
    from silk.exceptions import NotSupportedError
    tb = _bench(tmp_path, [])
    mod = tb.netlist
    mod.simulator = simulator
    owner = mod
    prefix = ''
    request = 'bias'
    if nested:
        owner = Module(None, str(tmp_path), 'leaf.cir', None, module_name='leaf')
        mod.elem_dict['dut']['device'].add_module(owner, 'core[0]', {})
        prefix = 'xdut.xcore<0>.'
        request = 'dut.core[0].bias'
    owner.add_source(sources.CurrentSource('bias', 'a', '0', dc=.2,
                     **({'pwl': [(0, 0), (1, 1)]} if split else {})))
    owner.add_source(VS('sense', 'b', '0', dc=0))
    current = tb.probe(f'i({request})')
    voltage = tb.probe(f"i({request.replace('bias', 'sense')})")
    cards = owner.sig_sources_dict['bias'].physical_instance_names()
    names = []
    for card in cards:
        card = prefix + card
        names.append((f'@{"i." if nested else ""}{card}[current]' if simulator == 'ngspice'
                      else f'I({card.replace(".", ":")})'))
    sense = f'i(v.{prefix}Vsense)' if nested and simulator == 'ngspice' else f'I({prefix}Vsense)'
    if simulator == 'xyce':
        sense = sense.replace('.', ':')
    plots = []
    for analysis, title in [('tran', 'Transient Analysis'), ('ac', 'AC Analysis'), ('op', 'Operating Point')]:
        count = 1 if analysis == 'op' else 2
        vectors = [] if analysis == 'op' else [RawVector(
            'frequency' if analysis == 'ac' else 'time', 'frequency' if analysis == 'ac' else 'time',
            data=np.array([1., 2.]))]
        vectors.append(RawVector(sense, 'current', data=np.full(count, 7.)))
        for name in names[:-1] if missing else names:
            vectors.append(RawVector(name, 'current', data=np.full(count, 3.)))
        plots.append(RawPlot({'plotname': title}, vectors, npoints=count,
                             real=analysis != 'ac', padded=True))
    path = tmp_path / 'removed-multiplot.raw'
    write_rawfile(path, RawFile(plots))
    mod.load_file(str(path))
    mod.read_raw_file()
    owner.sig_sources_dict.pop('bias')
    owner.sig_sources_dict.pop('sense')
    for analysis in ('tran', 'ac', 'op'):
        np.testing.assert_array_equal(voltage.resolve(analysis=analysis).y,
                                      [7.] if analysis == 'op' else [7., 7.])
        if missing:
            with pytest.raises(SimulationError) as exc:
                current.resolve(analysis=analysis)
            assert f"Source {request!r} is no longer present" in str(exc.value)
        elif analysis == 'ac':
            with pytest.raises(NotSupportedError) as exc:
                current.resolve(analysis=analysis)
            assert f"CurrentSource {request!r}:" in str(exc.value)
        else:
            np.testing.assert_array_equal(current.resolve(analysis=analysis).y,
                                          np.full(1 if analysis == 'op' else 2, 6. if split else 3.))


@pytest.mark.parametrize('override', ['subclass', 'instance'])
def test_source_current_resolution_preserves_module_loader_override(tmp_path, monkeypatch, override):
    from types import MethodType
    tb = _bench(tmp_path, ['Vsense'], [VS('sense', 'a', '0', dc=0)])
    mod = tb.netlist
    probe = tb.probe('i(sense)')
    expected = mod.load_signal('Vsense', 'I') * 4
    def load_signal(self, name, signalType='V', **select):
        assert (name, signalType) == ('Vsense', 'I')
        return expected
    if override == 'subclass':
        monkeypatch.setattr(mod, '__class__', type('CustomReader', (type(mod),), {'load_signal': load_signal}))
    else:
        monkeypatch.setattr(mod, 'load_signal', MethodType(load_signal, mod))
    assert probe.resolve(module=mod) is expected


def test_source_current_resolution_refuses_attached_but_unloaded_data(tmp_path):
    tb = _bench(tmp_path, ['Vsense'], [VS('sense', 'a', '0', dc=0)])
    probe = tb.probe('i(sense)')
    assert tb.netlist._raw.native.plots
    tb.netlist.sim_results_loaded = False
    with pytest.raises(SimulationError, match='not Loaded'):
        probe.resolve(module=tb.netlist)


def test_source_current_resolution_op_includes_row_zero(tmp_path):
    tb = _bench(tmp_path, ['Vsense'], [VS('sense', 'a', '0', dc=0)], data=[[3., 4.]])
    probe = tb.probe('i(sense)')
    op = RawPlot({'plotname': 'Operating Point'},
                 [RawVector('i(Vsense)', 'current', data=np.array([9.]))],
                 npoints=1, real=True, padded=True)
    path = tmp_path / 'op-row-zero.raw'
    write_rawfile(path, RawFile([op, tb.netlist._raw.native.plots[0]]))
    tb.netlist.load_file(str(path))
    tb.netlist.read_raw_file()
    np.testing.assert_array_equal(probe.resolve(analysis='op').y, [9.])
    np.testing.assert_array_equal(probe.resolve(analysis='tran').y, [3., 4.])


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('missing', [False, True])
def test_removed_current_ac_preserves_different_emitted_winner(tmp_path, simulator, split, missing):
    tb = _bench(tmp_path, [], [sources.CurrentSource('bias', 'a', '0', dc=.2,
                **({'pwl': [(0, 0), (1, 1)]} if split else {}))])
    mod = tb.netlist
    mod.simulator = simulator
    probe = tb.probe('i(bias)')
    cards = mod.sig_sources_dict['bias'].physical_instance_names()
    rows = [RawVector('frequency', 'frequency', data=np.array([1., 2.])),
            RawVector('i(bias)', 'current', data=np.array([8.+1j, 9.+2j]))]
    for card in cards[:-1] if missing else cards:
        name = f'@{card}[current]' if simulator == 'ngspice' else f'I({card})'
        rows.append(RawVector(name, 'current', data=np.array([3.+0j, 3.+0j])))
    path = tmp_path / 'removed-winner.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'AC Analysis'}, rows,
                                        npoints=2, real=False, padded=True)]))
    mod.load_file(str(path))
    mod.read_raw_file()
    mod.sig_sources_dict.pop('bias')
    np.testing.assert_array_equal(probe.resolve(analysis='ac').y, [8.+1j, 9.+2j])
