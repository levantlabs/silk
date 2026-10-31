"""``silk.signals.analysis_kind`` — the one rawfile-plot classifier shared by
STR-001 and the report layer (TASK-81 / TASK-82).

The kinds are pinned against the TASK-74 fixtures (real ngspice-42 and Xyce
7.10 output) and the boolean STR-001 reads is asserted to be exactly
``kind in SMALL_SIGNAL_KINDS``, so the two consumers cannot disagree.
"""
from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from silk.lint.rules import stress
from silk.signals.analysis_kind import (
    SMALL_SIGNAL_KINDS, _spice_results_are_small_signal, rawfile_analysis_kind,
)
from silk.signals.rawfile import read_rawfile

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles'


@pytest.mark.parametrize('fixture, kind', [
    ('e_ng_tran.raw', 'transient'),
    ('e_xy_tran.raw', 'transient'),      # Xyce axis 'TIME', case-folded
    ('e_ng_ac.raw', 'ac'),
    ('e_xy_ac.raw', 'ac'),
    ('e_ng_dc.raw', 'dc'),               # axis 'v(v-sweep)'
    ('e_xy_dc.raw', 'dc'),               # axis 'sweep'
    ('e_ng_noise.raw', 'noise'),         # real plot, 'frequency' axis
    ('e_ng_ramp.raw', 'transient'),
    ('e_ng_op.raw', 'op'),               # plotname 'Operating Point'
    ('e_xy_op.raw', 'op'),               # DC plotname, 'sweep' axis, 1 point
])
def test_kind_of_real_simulator_output(fixture, kind):
    assert rawfile_analysis_kind(read_rawfile(FIXTURES / fixture)) == kind


@pytest.mark.parametrize('fixture, vector0', [
    ('e_ng_op.raw', 'v(in)'),
    ('e_xy_op.raw', 'sweep'),
])
def test_op_is_not_a_dc_sweep(fixture, vector0):
    """Pre-fix both .op fixtures classified 'dc': the vector-0 rule saw a
    real plot on a non-time, non-frequency axis. But an operating point has
    no axis — ngspice-42 writes the first node 'v(in)' as vector 0 under
    plotname 'Operating Point', Xyce 7.10 writes a placeholder 'sweep' (0.0,
    not the 1.5 V source) under plotname 'DC transfer characteristic' — and
    both hold exactly one point (TASK-102)."""
    raw = read_rawfile(FIXTURES / fixture)
    assert raw.npoints == 1
    assert raw.vectors[0].name == vector0
    assert rawfile_analysis_kind(raw) == 'op'
    assert 'op' not in SMALL_SIGNAL_KINDS


def test_op_keys_on_plotname_or_single_point():
    """The two keys, separately: a plot whose plotname says 'Operating
    Point' is op at any point count (ngspice), and a real one-point plot is
    op under any plotname and axis name (Xyce's .op is indistinguishable
    from a one-point '.dc V1 1.5 1.5 1' — measured: both 'DC transfer
    characteristic', sweep=[x], 1 point — so a one-point DC sweep classifies
    op too: nothing was swept). A swept DC plot stays 'dc'."""
    def plot(name, n, plotname=''):
        return types.SimpleNamespace(
            plotname=plotname, real=True, npoints=n,
            vectors=[types.SimpleNamespace(name=name, data=np.zeros(n))])
    assert rawfile_analysis_kind(plot('v(in)', 1, 'Operating Point')) == 'op'
    assert rawfile_analysis_kind(plot('v(in)', 3, 'operating point')) == 'op'
    assert rawfile_analysis_kind(plot('sweep', 1, 'DC transfer characteristic')) == 'op'
    assert rawfile_analysis_kind(plot('v(v-sweep)', 1)) == 'op'
    assert rawfile_analysis_kind(plot('v(v-sweep)', 2)) == 'dc'
    assert rawfile_analysis_kind(plot('sweep', 401)) == 'dc'
    # A plot carrying no npoints classifies by its vector-0 data length.
    stub = types.SimpleNamespace(vectors=[
        types.SimpleNamespace(name='v(out)', data=np.zeros(1))])
    assert rawfile_analysis_kind(stub) == 'op'
    stub.vectors[0].data = np.zeros(5)
    assert rawfile_analysis_kind(stub) == 'dc'


def test_kind_is_taken_from_the_current_plot():
    """A multi-plot file (AC then transient) classifies as the LAST plot,
    the one the reader hands ``vectors[0]`` from."""
    raw = read_rawfile(FIXTURES / 'd_multiplot_ac_tran.raw')
    assert raw.plots[0].real is False
    assert rawfile_analysis_kind(raw) == 'transient'


@pytest.mark.parametrize('raw', [
    None,
    types.SimpleNamespace(current_plot=None, vectors=[]),
    types.SimpleNamespace(vectors=[]),
])
def test_unclassifiable_is_none_not_a_kind(raw):
    assert rawfile_analysis_kind(raw) is None


def test_complex_dtype_classifies_as_ac_without_header_flag():
    """A stub with no ``real`` attribute but complex data is AC: the dtype
    walk backs up the header flag so a missing attribute can never make a
    complex vector look real-valued."""
    plot = types.SimpleNamespace(vectors=[
        types.SimpleNamespace(name='time', data=np.zeros(3)),
        types.SimpleNamespace(name='v(out)', data=np.zeros(3, dtype=complex)),
    ])
    assert rawfile_analysis_kind(plot) == 'ac'


def test_axis_only_stubs_classify_by_name():
    """The STR-001 tests mount stubs with an axis name and a flag and no
    data at all; those still classify."""
    def stub(name, real):
        plot = types.SimpleNamespace(
            vectors=[types.SimpleNamespace(name=name)], real=real)
        return types.SimpleNamespace(current_plot=plot)
    assert rawfile_analysis_kind(stub('time', True)) == 'transient'
    assert rawfile_analysis_kind(stub('frequency', False)) == 'ac'
    assert rawfile_analysis_kind(stub('frequency', True)) == 'noise'
    assert rawfile_analysis_kind(stub('v(v-sweep)', True)) == 'dc'
    assert rawfile_analysis_kind(stub('temp-sweep', True)) == 'dc'
    # No data and no npoints: the point count is unknown, so the one-point
    # op rule cannot fire and the axis name decides (TASK-102).
    assert rawfile_analysis_kind(stub('v(in)', True)) == 'dc'


@pytest.mark.parametrize('fixture', [
    'e_ng_tran.raw', 'e_ng_ac.raw', 'e_ng_dc.raw', 'e_ng_noise.raw',
    'e_ng_op.raw', 'e_xy_tran.raw', 'e_xy_ac.raw', 'e_xy_dc.raw', 'e_xy_op.raw',
])
def test_str001_boolean_is_the_shared_kind(fixture):
    """STR-001's predicate is ``kind in SMALL_SIGNAL_KINDS`` — the same
    object the report layer dispatches on, imported into stress.py under
    its TASK-81 name, so the lint rule and the summary read one rawfile the
    same way."""
    raw = read_rawfile(FIXTURES / fixture)
    module = types.SimpleNamespace(simResultsAll=raw)
    expected = rawfile_analysis_kind(raw) in SMALL_SIGNAL_KINDS
    assert _spice_results_are_small_signal(module) is expected
    assert stress._spice_results_are_small_signal is _spice_results_are_small_signal


def test_str001_unclassifiable_is_not_small_signal():
    """No rawfile / an empty plot -> False, so STR-001 behaves as it did
    before the guard existed rather than silently declining to check."""
    assert _spice_results_are_small_signal(types.SimpleNamespace()) is False
    empty = types.SimpleNamespace(simResultsAll=types.SimpleNamespace(vectors=[]))
    assert _spice_results_are_small_signal(empty) is False


@pytest.fixture(params=['transient', 'ac', 'op'])
def lazy_consumer_raw(request, tmp_path, monkeypatch):
    from silk.signals import rawfile
    kind = request.param
    count = 1 if kind == 'op' else 5
    names = ['v(a)', 'v(b)', 'v(c)'] if kind == 'op' else [
        'frequency' if kind == 'ac' else 'time', 'v(a)', 'v(b)', 'v(c)']
    vectors = [rawfile.RawVector(name, data=np.arange(count, dtype=float) + i * 10)
               for i, name in enumerate(names)]
    if kind == 'ac':
        for vector in vectors[1:]:
            vector.data = vector.data + 2j
    plot = rawfile.RawPlot({'plotname': 'Operating Point' if kind == 'op' else ''},
                           vectors, count, kind != 'ac', None)
    path = tmp_path / 'consumer.raw'
    rawfile.write_rawfile(path, plot)
    raw = rawfile.read_rawfile(path, lazy=True)
    calls = []
    materialize = rawfile._LazyStore.materialize

    def counted(store, slot, name):
        calls.append(slot)
        return materialize(store, slot, name)

    monkeypatch.setattr(rawfile._LazyStore, 'materialize', counted)
    return raw, calls, kind


def test_lazy_classification_materializes_nothing(lazy_consumer_raw):
    raw, calls, kind = lazy_consumer_raw
    assert rawfile_analysis_kind(raw) == kind
    assert calls == []


@pytest.mark.parametrize('ambiguous', [False, True])
def test_lazy_failed_signal_lookup_materializes_nothing(lazy_consumer_raw, ambiguous):
    from silk.exceptions import AmbiguousSignalError, SimulationError
    from silk.signals.spice import NgspiceSignal
    raw, calls, _ = lazy_consumer_raw
    if ambiguous:
        raw.vectors[2].name = 'v(a)'
        error, message, name = AmbiguousSignalError, 'ambiguous', 'a'
    else:
        error, message, name = SimulationError, 'not found', 'missing'
    with pytest.raises(error, match=message):
        NgspiceSignal.from_rawfile(raw, name)
    assert calls == []


def test_lazy_signal_reads_only_axis_and_selected_vector(lazy_consumer_raw):
    from silk.signals.spice import NgspiceSignal
    raw, calls, kind = lazy_consumer_raw
    signal = NgspiceSignal.from_rawfile(raw, 'a')
    expected = [(0, 0)] if kind == 'op' else [(0, 0), (0, 1)]
    assert calls == expected
    np.testing.assert_array_equal(signal.x, np.arange(raw.npoints))
    expected_y = np.arange(raw.npoints) + (0 if kind == 'op' else 10)
    if kind == 'ac':
        expected_y = expected_y + 2j
    np.testing.assert_array_equal(signal.y, expected_y)
    assert signal.xlabel == ('op' if kind == 'op' else raw.vectors[0].name)
    repeated = NgspiceSignal.from_rawfile(raw, 'a')
    np.testing.assert_array_equal(repeated.y, signal.y)
    assert calls == expected


def test_lazy_classifier_observes_replaced_arrays_without_loading_others(lazy_consumer_raw):
    raw, calls, _ = lazy_consumer_raw
    raw.current_plot.real = True
    raw.vectors[1].data = np.ones(raw.npoints, dtype=complex)
    assert rawfile_analysis_kind(raw) == 'ac'
    assert calls == []


@pytest.mark.parametrize('stub', [False, True])
def test_classifier_dtype_fallback_preserves_writable_arrays(stub):
    from silk.signals.rawfile import RawVector
    vector_type = types.SimpleNamespace if stub else RawVector
    axis = vector_type(name='time', data=np.arange(3))
    value = vector_type(name='v(a)', data=np.ones(3))
    plot = types.SimpleNamespace(real=True, vectors=[axis, value])
    assert rawfile_analysis_kind(plot) == 'transient'
    value.data = np.ones(3, dtype=complex)
    assert rawfile_analysis_kind(plot) == 'ac'
    value.data = np.ones(3)
    assert rawfile_analysis_kind(plot) == 'transient'


def test_signal_loader_keeps_no_axis_diagnostic():
    from silk.signals.rawfile import RawFile
    from silk.signals.spice import NgspiceSignal
    from silk.exceptions import SimulationError
    with pytest.raises(SimulationError, match='No x-axis vector found'):
        NgspiceSignal.from_rawfile(RawFile([]), 'missing')
