"""Synthetic regressions for TASK-541 implementation review."""
import copy
import pickle
import numpy as np
import pytest

from tests._legacy_names import add_legacy_net

from tests.test_object_probes import bench, block, raw, selected_bus
from silk import sources
from silk.exceptions import SimulationError
from silk.circuit.module_test import _unbound_leaves, _metric_slot_leaves
from silk.metrics.statistics import average
from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile


@pytest.mark.parametrize('inventory', [
    ['xx1[0].mid'], ['xx1[0].mid', 'xx1<0>.mid'], ['xx1<0>.mid'],
])
def test_literal_parent_node_never_aliases_instance(tmp_path, inventory):
    tb, stage = bench(tmp_path)
    add_legacy_net(tb.netlist, 'xx1[0].mid')
    tb.netlist.elem_dict['stage']['pin_list']['out'].net = 'xx1[0].mid'
    tb.add_module(block(tmp_path, 'other_leaf'), 'x1[0]', {'out': 'other'})
    p = tb.probe(stage.out)
    raw(tb, tmp_path, inventory)
    if 'xx1[0].mid' in inventory:
        np.testing.assert_equal(p.y, [1., 1.5])
    else:
        with pytest.raises(SimulationError, match='not found'):
            _ = p.y


@pytest.mark.parametrize('method', ['deepcopy', 'pickle'])
def test_summed_current_is_one_probe_after_copy(tmp_path, method):
    tb, _ = bench(tmp_path)
    tb.add_source(sources.CurrentSource('stim', 'parent', '0', dc=.2,
                                      pulse=sources.Pulse(0., 1., tr=.1, tf=.1, pw=1., per=2.)))
    p = tb.probe('i(stim)')
    tb.add_metric('mean_i', average, sig=p)
    other = copy.deepcopy(tb) if method == 'deepcopy' else pickle.loads(pickle.dumps(tb))
    assert len(other._probe_registry) == 1
    logical = other._probe_registry[0]
    assert logical.__dict__.get('_logical_probe') is True
    assert len(list(_unbound_leaves(logical))) == 2
    assert len(list(_unbound_leaves(logical, logical_probes=True))) == 1
    assert all(x._bench() is other for x in _unbound_leaves(logical))
    vectors = [RawVector('time','time',data=np.array([0.,1.])),
               RawVector('@istim__silk_wave[current]','current',data=np.array([1.,2.])),
               RawVector('@istim__silk_dc[current]','current',data=np.array([.2,.2]))]
    path = tmp_path / 'current.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname':'Transient Analysis'},vectors,npoints=2,real=True,padded=True)]))
    other.netlist.load_file(str(path))
    other.netlist.read_raw_file()
    np.testing.assert_allclose(logical.y, [1.2, 2.2])
    import matplotlib.pyplot as plt
    fig = other.plot(show=False)
    try:
        assert len(fig.axes) == 1
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [1.2, 2.2])
    finally:
        plt.close(fig)


def test_selected_pickle_metric_owner(tmp_path):
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    one.add_metric('code', average, sig=one.probe(stage.d[8:10]).as_int())
    cloned = pickle.loads(pickle.dumps(top))
    selected = cloned.get_test('bus', instance='one')
    leaves = list(_metric_slot_leaves(selected._metrics))
    assert len(leaves) == 2
    assert all(leaf._bench() is selected for leaf in leaves)


@pytest.mark.parametrize('early_read', [False, True])
def test_failed_prerun_read_does_not_create_an_extra_context(tmp_path, early_read):
    from silk.exceptions import UnboundSignalError
    dut = block(tmp_path, 'context_dut')
    h = dut.add_module(block(tmp_path, 'inner_leaf'), 'stage', {'out':'out'})
    ref = h.mid
    if early_read:
        with pytest.raises(UnboundSignalError):
            _ = ref.y
    tb = dut.create_test('only')
    tb.add_module_under_test({'out':'wire'})
    raw(tb, tmp_path, ['xdut.xstage.mid'])
    np.testing.assert_equal(tb.probe(ref).y, [1., 1.5])
    np.testing.assert_equal(ref.y, [1., 1.5])


@pytest.mark.parametrize('method', ['unchanged', 'deepcopy', 'pickle', 'pickle_signal_first'])
def test_selected_pickle_readback_does_not_follow_sibling(tmp_path, method):
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    code = one.probe(stage.d[8:10]).as_int()
    one.add_metric('code', average, sig=code)
    cloned, code = (pickle.loads(pickle.dumps((top, code))) if method == 'pickle' else
                    copy.deepcopy((top, code)) if method == 'deepcopy' else (top, code))
    if method == 'pickle_signal_first':
        code, cloned = pickle.loads(pickle.dumps((code, top)))
    one = cloned.get_test('bus', instance='one')
    two = cloned.get_test('bus', instance='two')
    def result_file(label, value):
        vectors = [RawVector('time','time',data=np.array([0.,1.])),
                   RawVector('v(xdut.xstage.d[8])','voltage',data=np.array([value,value])),
                   RawVector('v(xdut.xstage.d[9])','voltage',data=np.array([value,value]))]
        path = tmp_path / (label + '.raw')
        write_rawfile(path, RawFile([RawPlot({'plotname':'Transient Analysis'},vectors,npoints=2,real=True,padded=True)]))
        return str(path)
    one.evaluate(rawfile=result_file('one', 1.))
    from silk.exceptions import MetricBindingError
    with pytest.raises(MetricBindingError):
        two.evaluate(rawfile=result_file('two', 0.))
    np.testing.assert_array_equal(code.y, [3., 3.])


@pytest.mark.parametrize('roundtrip', [False, True])
def test_scalar_selected_pickle_context(tmp_path, roundtrip):
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    selected.add_metric('mean', average, sig=selected.probe('xdut.xstage.mid'))
    cloned = pickle.loads(pickle.dumps(top)) if roundtrip else top
    one = cloned.get_test('bus', instance='one')
    two = cloned.get_test('bus', instance='two')
    for selected, value, label in ((one, 1., 'one'), (two, 0., 'two')):
        path = tmp_path / (label + '.raw')
        vectors = [RawVector('time', 'time', data=np.array([0., 1.])),
                   RawVector('v(xdut.xstage.mid)', 'voltage', data=np.array([value, value]))]
        write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'}, vectors,
                                            npoints=2, real=True, padded=True)]))
        if selected is two:
            from silk.exceptions import MetricBindingError
            with pytest.raises(MetricBindingError):
                selected.evaluate(rawfile=str(path))
        else:
            selected.evaluate(rawfile=str(path))
    np.testing.assert_array_equal(one._metrics[0]['slots']['sig'].y, [1., 1.])


def test_standalone_selected_signal_pickle_stays_detached(tmp_path):
    from silk.exceptions import UnboundSignalError
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    restored = pickle.loads(pickle.dumps(selected.probe(stage.d[8:10]).as_int()))
    with pytest.raises(UnboundSignalError, match='detached'):
        _ = restored.y


def test_selected_metric_pickle_without_owner_root_refuses(tmp_path):
    from silk.exceptions import UnboundSignalError
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    selected.add_metric('code', average, sig=selected.probe(stage.d[8:10]).as_int())
    restored = pickle.loads(pickle.dumps(tb))
    raw(restored, tmp_path, ['xdut.xstage.d[8]', 'xdut.xstage.d[9]'])
    with pytest.raises(UnboundSignalError, match='detached'):
        _ = restored._metrics[0]['slots']['sig'].y


def test_authored_metric_pickle_keeps_authored_binding(tmp_path):
    tb, stage = bench(tmp_path)
    signal = tb.probe(stage.mid)
    tb.add_metric('mean', average, sig=signal)
    restored, signal = pickle.loads(pickle.dumps((tb, signal)))
    assert signal._bench() is restored
    raw(restored, tmp_path, ['xstage.mid'])
    np.testing.assert_array_equal(signal.y, [1., 1.5])



def test_foreign_owner_root_pickle_never_promotes_selected_probe(tmp_path):
    from silk import Module
    from silk.exceptions import UnboundSignalError
    top, tb, stage = selected_bus(tmp_path)
    one = top.get_test('bus', instance='one')
    one.add_metric('code', average, sig=one.probe(stage.d[8:10]).as_int())
    foreign = Module(None, str(tmp_path), 'foreign.cir', None)
    foreign.add_module(tb.module, 'one', {'out': 'other'})
    restored = pickle.loads(pickle.dumps(foreign))
    selected = restored.get_test('bus', instance='one')
    raw(selected._authored, tmp_path, ['xdut.xstage.d[8]', 'xdut.xstage.d[9]'])
    with pytest.raises(UnboundSignalError, match='detached'):
        _ = selected._metrics[0]['slots']['sig'].y


def test_selected_metric_pickle_after_owner_expiry_stays_detached(tmp_path):
    import gc
    from silk.exceptions import UnboundSignalError
    top, tb, stage = selected_bus(tmp_path)
    selected = top.get_test('bus', instance='one')
    selected.add_metric('code', average, sig=selected.probe(stage.d[8:10]).as_int())
    leaves = list(_metric_slot_leaves(tb._metrics))
    del selected, top
    gc.collect()
    assert all(leaf._bench() is None for leaf in leaves)
    restored = pickle.loads(pickle.dumps(tb))
    raw(restored, tmp_path, ['xdut.xstage.d[8]', 'xdut.xstage.d[9]'])
    with pytest.raises(UnboundSignalError, match='detached'):
        _ = restored._metrics[0]['slots']['sig'].y
