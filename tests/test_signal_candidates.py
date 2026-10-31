"""TASK-527: ONE candidate collector for NgspiceSignal.from_rawfile and
SimulatorBackend.load_signal (union before choosing, identity dedup)."""
import numpy as np
import pytest

from silk.exceptions import SimulationError
from silk.signals.rawfile import RawFile, RawPlot, RawVector
from silk.signals.spice import NgspiceSignal


def _raw(*names, data=None, plots=1):
    def plot(tag):
        vecs = [RawVector('time', 'time', data=np.array([0., 1.]))]
        for i, n in enumerate(names):
            vals = data[i] if data else [float(i), float(i)]
            vecs.append(RawVector(n, 'voltage' if n.startswith('v') else 'current',
                                  data=np.array(vals, dtype=float)))
        return RawPlot({'plotname': 'Transient Analysis', 'title': tag},
                       vecs, npoints=2, real=True, padded=True)
    return RawFile([plot(str(k)) for k in range(plots)])


def test_bracket_alias_resolves_nested_node():
    raw = _raw('v(xx1<0>.xy<1>.g[0])')
    a = NgspiceSignal.from_rawfile(raw, 'xx1[0].xy[1].g[0]', 'V')
    b = NgspiceSignal.from_rawfile(raw, 'xx1<0>.xy<1>.g[0]', 'V')
    np.testing.assert_array_equal(a.y, b.y)


def test_one_row_two_spellings_single_candidate():
    raw = _raw('v(xx1<0>.mid)')
    sig = NgspiceSignal.from_rawfile(raw, 'XX1<0>.mid', 'V')   # as-given == converted
    assert sig.y[0] == 0.0


def test_literal_toplevel_net_beside_hierarchical_is_ambiguous():
    raw = _raw('v(xx1[0].mid)', 'v(xx1<0>.mid)')
    with pytest.raises(SimulationError, match=r'(?s)ambiguous.*xx1\[0\]\.mid.*xx1<0>\.mid'):
        NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V')


def test_reversed_vector_order_same_refusal():
    raw = _raw('v(xx1<0>.mid)', 'v(xx1[0].mid)')
    with pytest.raises(SimulationError, match='ambiguous'):
        NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V')


def test_equal_valued_distinct_rows_are_ambiguous():
    raw = _raw('v(xx1[0].mid)', 'v(xx1<0>.mid)', data=[[1., 1.], [1., 1.]])
    with pytest.raises(SimulationError, match='ambiguous'):
        NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V')


def test_ambiguity_error_is_typed_and_names_request_plot_candidates():
    from silk.exceptions import AmbiguousSignalError
    raw = _raw('v(xx1[0].mid)', 'v(xx1<0>.mid)')
    with pytest.raises(AmbiguousSignalError) as exc:
        NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V', plot=raw.plots[0])
    assert issubclass(AmbiguousSignalError, SimulationError)
    msg = str(exc.value)
    assert "'xx1[0].mid'" in msg and 'Transient Analysis' in msg
    assert 'v(xx1[0].mid)' in msg and 'v(xx1<0>.mid)' in msg


def test_zero_matches_keeps_missing_signal_error():
    raw = _raw('v(out)')
    with pytest.raises(SimulationError, match=r"Signal 'nope' \(type='V'\) not found in rawfile"):
        NgspiceSignal.from_rawfile(raw, 'nope', 'V')


def test_multi_plot_collects_only_selected_plot():
    raw = _raw('v(xx1<0>.mid)', plots=2)
    raw.plots[1].vectors.append(RawVector('v(xx1[0].mid)', 'voltage', data=np.zeros(2)))
    # plot 0 has one candidate; plot 1 has two
    NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V', plot=raw.plots[0])
    with pytest.raises(SimulationError, match='ambiguous'):
        NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V', plot=raw.plots[1])


def test_current_type_converts_leaf_and_xyce_colon_equivalence_kept():
    raw = _raw('i(vvs<0>)')
    NgspiceSignal.from_rawfile(raw, 'vvs[0]', 'I')
    raw2 = RawFile([RawPlot({'plotname': 'Transient Analysis'},
        [RawVector('TIME', 'time', data=np.array([0., 1.])),
         RawVector('V(XDUT:XX1<0>:MID)', 'voltage', data=np.array([1., 2.]))],
        npoints=2, real=True, padded=True)])
    sig = NgspiceSignal.from_rawfile(raw2, 'xdut.xx1[0].mid', 'V')   # dot request, colon raw
    np.testing.assert_array_equal(sig.y, [1., 2.])


def test_rawfile_with_no_simulator_header_and_mixed_case():
    raw = _raw('v(XDUT.Xx1<0>.Mid)')
    NgspiceSignal.from_rawfile(raw, 'xdut.xx1[0].mid', 'V')


def test_backend_and_from_rawfile_share_the_collector():
    from silk.signals import spice
    from silk.simulation.backend import SimulatorBackend, RawResult
    assert hasattr(spice, 'collect_vector_candidates')
    raw = _raw('v(xx1[0].mid)', 'v(xx1<0>.mid)')
    backend = SimulatorBackend.get('ngspice')
    with pytest.raises(SimulationError, match='ambiguous'):
        backend.load_signal(RawResult(native=raw, source_path='x.raw'), 'xx1[0].mid')


def _stub_ckt(raw):
    class Ckt:
        sig_sources_dict = {}

        def load_signal(self, name, kind='V', **select):
            return NgspiceSignal.from_rawfile(raw, name, kind)
    return Ckt()


def test_raw_current_only_as_given_row_resolves_through_branch_path():
    """P2-1: only i(vvs[0]) exists; the as-given spelling must reach the collector."""
    from silk.circuit.module_test import _branch_vector_name, _load_metric_path
    raw = _raw('i(vvs[0])', data=[[3., 3.]])
    assert _branch_vector_name({}, 'vvs[0]') == 'vvs[0]'
    sig = _load_metric_path(_stub_ckt(raw), 'i(vvs[0])')
    np.testing.assert_array_equal(sig.y, [3., 3.])


def test_raw_current_both_spellings_present_is_ambiguous_not_a_silent_pick():
    from silk.circuit.module_test import _load_metric_path
    from silk.exceptions import SimulationError
    raw = _raw('i(vvs[0])', 'i(vvs<0>)')
    with pytest.raises(SimulationError, match='ambiguous'):
        _load_metric_path(_stub_ckt(raw), 'i(vvs[0])')


def test_raw_current_angle_spelling_still_passes_through():
    from silk.circuit.module_test import _branch_vector_name, _load_metric_path
    raw = _raw('i(vvs<0>)', data=[[4., 4.]])
    assert _branch_vector_name({}, 'vvs<0>') == 'vvs<0>'
    np.testing.assert_array_equal(_load_metric_path(_stub_ckt(raw), 'i(vvs[0])').y, [4., 4.])


def test_default_single_plot_ambiguity_names_the_real_plot():
    """P3-1: a plain RawFile call (no plot=) must still name the plot."""
    from silk.exceptions import AmbiguousSignalError
    raw = _raw('v(xx1[0].mid)', 'v(xx1<0>.mid)')
    with pytest.raises(AmbiguousSignalError) as exc:
        NgspiceSignal.from_rawfile(raw, 'xx1[0].mid', 'V')
    assert 'Transient Analysis' in str(exc.value)
    assert '(unnamed plot)' not in str(exc.value)
