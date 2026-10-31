"""Unbound Signals: the probe state on :class:`silk.signals.base.Signal`.

An unbound
Signal is represented by ABSENCE — no ``x``/``y``/``xlabel``/``ylabel`` stored
on it — and reading one of those four resolves it against its testbench's
current results through a ``__getattr__`` fallback.

The bench here is a small fake: the real ``ModuleTestbench.probe()`` arrives in
TASK-215.2's successor (215.5), and everything this file asserts is a property
of ``Signal`` alone, namely that a probe holds names and a weak reference and
asks the bench for data at read time.
"""

import copy
import gc
import pickle
import weakref

import numpy as np
import pytest

from silk.exceptions import UnboundSignalError
from silk.signals.base import _DETACHED_BENCH, Signal


# ---------------------------------------------------------------------------
# A fake bench: exactly the two things a probe asks of one — a name, and a
# netlist that says whether results exist and can load a signal.
# ---------------------------------------------------------------------------

class FakeNetlist:
    def __init__(self):
        self.sim_results_loaded = False
        self.calls = []

    def load_signal(self, name, signal_type='V'):
        self.calls.append((name, signal_type))
        if not self.sim_results_loaded:
            raise AssertionError('load_signal called before results exist')
        t = np.linspace(0.0, 1e-6, 5)
        return Signal(name, 'voltage', 'time', f'v({name})', t, np.arange(5.0))


class FakeBench:
    def __init__(self, name='cm', netlist=None):
        self.test_name = name
        self.netlist = FakeNetlist() if netlist is None else netlist


@pytest.fixture
def bench():
    return FakeBench()


def bound_signal(name='v(out)'):
    t = np.linspace(0.0, 1e-6, 5)
    return Signal(name, 'voltage', 'time', name, t, np.arange(5.0))


# ---------------------------------------------------------------------------
# Construction: absence, and the fields that are present
# ---------------------------------------------------------------------------

class TestUnboundConstruction:

    def test_no_samples_are_stored(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        for attr in ('x', 'y', 'xlabel', 'ylabel'):
            assert attr not in p.__dict__

    def test_fields(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        assert p.name == 'vout'
        assert p._selector == 'V'
        assert p.type == 'voltage'
        assert p._analysis is None
        assert p._recipe is None
        assert p._bench() is bench

    def test_current_selector_maps_to_current(self, bench):
        p = Signal._unbound('vvs', 'I', bench)
        assert p._selector == 'I'
        assert p.type == 'current'

    def test_selector_is_case_insensitive(self, bench):
        assert Signal._unbound('vvs', 'i', bench)._selector == 'I'

    def test_unknown_selector_refused(self, bench):
        with pytest.raises(ValueError, match="selector must be 'V'"):
            Signal._unbound('vout', 'X', bench)

    def test_bus_fields_are_scalar_defaults(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        assert p.width == 1
        assert p._underlying_signals is None

    def test_bench_is_held_weakly(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        assert isinstance(p._bench, weakref.ref)

    def test_bench_may_be_absent(self):
        p = Signal._unbound('vout')
        assert p._bench is None
        assert p.bound is False


# ---------------------------------------------------------------------------
# Reading before the run
# ---------------------------------------------------------------------------

class TestBeforeResults:

    @pytest.mark.parametrize('attr', ['x', 'y', 'xlabel', 'ylabel'])
    def test_read_raises_unbound_signal_error(self, bench, attr):
        p = Signal._unbound('vp', 'V', bench)
        with pytest.raises(UnboundSignalError) as exc:
            getattr(p, attr)
        msg = str(exc.value)
        assert "'vp'" in msg        # names the probe
        assert "'cm'" in msg        # names the bench
        assert 'has not run yet' in msg

    def test_error_is_not_an_attribute_error(self, bench):
        p = Signal._unbound('vp', 'V', bench)
        assert not issubclass(UnboundSignalError, AttributeError)
        with pytest.raises(UnboundSignalError):
            hasattr(p, 'y')
        with pytest.raises(UnboundSignalError):
            getattr(p, 'y', None)

    def test_no_bench_names_the_probe(self):
        p = Signal._unbound('vp')
        with pytest.raises(UnboundSignalError, match="probe 'vp'"):
            p.y

    def test_bench_without_a_netlist_is_an_orphan(self):
        orphan = FakeBench(name='orphan')
        orphan.netlist = None
        p = Signal._unbound('vp', 'V', orphan)
        assert p.bound is False
        with pytest.raises(UnboundSignalError) as exc:
            p.y
        assert 'has no netlist' in str(exc.value)
        assert "'orphan'" in str(exc.value)

    def test_dead_bench(self):
        bench = FakeBench()
        p = Signal._unbound('vp', 'V', bench)
        del bench
        gc.collect()
        assert p.bound is False
        with pytest.raises(UnboundSignalError, match='garbage-collected'):
            p.y


# ---------------------------------------------------------------------------
# Reading after the run
# ---------------------------------------------------------------------------

class TestAfterResults:

    def test_attributes_come_from_the_loader(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        bench.netlist.sim_results_loaded = True
        assert np.array_equal(p.y, np.arange(5.0))
        assert np.array_equal(p.x, np.linspace(0.0, 1e-6, 5))
        assert p.xlabel == 'time'
        assert p.ylabel == 'v(vout)'

    def test_selector_is_passed_to_the_loader(self, bench):
        p = Signal._unbound('vvs', 'I', bench)
        bench.netlist.sim_results_loaded = True
        p.y
        assert bench.netlist.calls == [('vvs', 'I')]

    def test_nothing_is_stored_back(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        bench.netlist.sim_results_loaded = True
        p.y
        p.y
        assert 'y' not in p.__dict__
        assert len(bench.netlist.calls) == 2      # every read resolves again

    def test_resolve_returns_a_bound_signal(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        bench.netlist.sim_results_loaded = True
        resolved = p.resolve()
        assert resolved is not p
        assert isinstance(resolved, Signal)
        assert 'y' in resolved.__dict__
        assert resolved.bound is True
        assert p.bound is True                    # the probe stays unbound…
        assert 'y' not in p.__dict__              # …and stays data-free

    def test_resolve_accepts_an_explicit_module(self, bench):
        other = FakeNetlist()
        other.sim_results_loaded = True
        p = Signal._unbound('vout', 'V', bench)
        resolved = p.resolve(other)
        assert other.calls == [('vout', 'V')]
        assert bench.netlist.calls == []
        assert np.array_equal(resolved.y, np.arange(5.0))

    def test_stale_results_are_impossible(self, bench):
        """A probe re-reads: after the bench's results are cleared it refuses
        again rather than handing back the previous run."""
        p = Signal._unbound('vout', 'V', bench)
        bench.netlist.sim_results_loaded = True
        assert p.y is not None
        bench.netlist.sim_results_loaded = False
        assert p.bound is False
        with pytest.raises(UnboundSignalError):
            p.y


# ---------------------------------------------------------------------------
# bound: answers, never raises
# ---------------------------------------------------------------------------

class TestBoundProperty:

    def test_false_then_true(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        assert p.bound is False
        bench.netlist.sim_results_loaded = True
        assert p.bound is True

    def test_true_on_a_signal_carrying_data(self):
        assert bound_signal().bound is True

    def test_never_raises(self):
        for probe in (Signal._unbound('a'),
                      Signal._unbound('b', 'V', FakeBench(netlist=object())),
                      Signal._unbound('c', 'V', FakeBench())):
            assert probe.bound is False


# ---------------------------------------------------------------------------
# Width-1 behaviour: today's errors, not stray reads
# ---------------------------------------------------------------------------

class TestScalarBehaviour:

    def test_len_is_the_scalar_width(self, bench):
        assert len(Signal._unbound('vout', 'V', bench)) == 1
        assert len(bound_signal()) == 1

    def test_iteration_gives_todays_error(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        with pytest.raises(TypeError) as exc:
            iter(p)
        assert str(exc.value) == (
            "Signal 'vout' is scalar (width=1); iteration is not supported."
        )

    def test_subscript_gives_todays_error(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        with pytest.raises(TypeError) as exc:
            p[0]
        assert str(exc.value) == "Signal 'vout' is scalar; subscript not supported."


# ---------------------------------------------------------------------------
# The __getattr__ fallback itself
# ---------------------------------------------------------------------------

class TestAttributeFallback:

    def test_underscore_names_raise_plain_attribute_error(self, bench):
        """The guard that keeps copy/pickle from recursing: a '_' name is
        answered with a plain AttributeError and no bench lookup."""
        p = Signal._unbound('vout', 'V', bench)
        for name in ('__getnewargs__', '__getnewargs_ex__', '__array_ufunc__',
                     '_secret', '_silk_private'):
            assert not hasattr(Signal, name), f'{name} is a real attribute'
            with pytest.raises(AttributeError) as exc:
                getattr(p, name)
            assert not isinstance(exc.value, UnboundSignalError)
            assert getattr(p, name, 'default') == 'default'

    def test_underscore_guard_does_not_consult_the_bench(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        with pytest.raises(AttributeError):
            p._secret
        assert bench.netlist.calls == []

    def test_unknown_public_name_matches_pythons_message(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        with pytest.raises(AttributeError) as exc:
            p.no_such_thing
        assert str(exc.value) == "'Signal' object has no attribute 'no_such_thing'"

    def test_bound_signal_attribute_error_unchanged(self):
        s = bound_signal()
        with pytest.raises(AttributeError) as exc:
            s.no_such_thing
        assert str(exc.value) == "'Signal' object has no attribute 'no_such_thing'"

    def test_bound_signals_never_reach_the_fallback(self, monkeypatch):
        s = bound_signal()

        def boom(self, name):
            raise AssertionError(f'__getattr__ reached for {name!r}')

        monkeypatch.setattr(Signal, '__getattr__', boom)
        assert s.xlabel == 'time'
        assert s.ylabel == 'v(out)'
        assert s.x is not None and s.y is not None
        assert s.name == 'v(out)' and s.type == 'voltage'


# ---------------------------------------------------------------------------
# Copying: names and a weak reference, never a bench (TASK-205 stays fixed)
# ---------------------------------------------------------------------------

class TestCopying:

    EXPECTED_KEYS = {
        'name', 'type', '_selector', '_analysis', '_bench', '_recipe',
        'width', '_underlying_signals',
    }

    def test_deepcopy_of_a_leaf(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        c = copy.deepcopy(p)
        assert set(c.__dict__) == self.EXPECTED_KEYS
        assert c.name == 'vout'
        assert c._selector == 'V'
        assert c.type == 'voltage'
        assert c._analysis is None
        assert c._recipe is None
        assert c.width == 1
        assert c._underlying_signals is None

    def test_deepcopy_drags_no_bench_graph(self, bench):
        """Walk the copy's ``__dict__``: strings, None, an int and a weak
        reference. No Module, no bench, no netlist object."""
        c = copy.deepcopy(Signal._unbound('vout', 'V', bench))
        for key, value in c.__dict__.items():
            if key == '_bench':
                assert isinstance(value, weakref.ref)
                continue
            assert isinstance(value, (str, int, type(None))), (key, value)
            assert not isinstance(value, (FakeBench, FakeNetlist))

    def test_deepcopied_weakref_is_atomic(self, bench):
        """copy deep-copies a weak reference by sharing it, so the copy points
        at the ORIGINAL bench. Rebinding to a copied bench is TASK-215.8's
        ``ModuleTestbench.__deepcopy__`` slice; pinned here because the growth
        regression depends on this atomicity."""
        c = copy.deepcopy(Signal._unbound('vout', 'V', bench))
        assert c._bench() is bench

    def test_shallow_copy(self, bench):
        c = copy.copy(Signal._unbound('vout', 'V', bench))
        assert set(c.__dict__) == self.EXPECTED_KEYS
        assert c._bench() is bench

    def test_deepcopy_of_a_bound_signal_still_carries_data(self):
        c = copy.deepcopy(bound_signal())
        assert np.array_equal(c.y, np.arange(5.0))
        assert c.bound is True


# ---------------------------------------------------------------------------
# Pickling: the weak reference is dropped and the probe says so (section 3.0)
#
# A weakref cannot be pickled, and a bench whose metrics are named functions
# pickles today (FaultCampaign pickles whole Modules) — so a probe must not be
# the thing that stops it. Rebinding a restored bench's probes to itself is
# TASK-215.8, not here: unpickled on its own, a probe is DETACHED.
# ---------------------------------------------------------------------------

class TestPickling:

    def test_round_trip_keeps_the_names(self, bench):
        p = Signal._unbound('vp', 'I', bench, analysis=None)
        q = pickle.loads(pickle.dumps(p))
        assert q.name == 'vp'
        assert q._selector == 'I'
        assert q.type == 'current'
        assert q._analysis is None
        assert q._recipe is None
        assert q.width == 1
        assert q._underlying_signals is None

    def test_the_weakref_is_dropped(self, bench):
        q = pickle.loads(pickle.dumps(Signal._unbound('vp', 'V', bench)))
        assert not isinstance(q._bench, weakref.ref)
        assert q._bench is _DETACHED_BENCH

    def test_unpickled_probe_is_detached(self, bench):
        q = pickle.loads(pickle.dumps(Signal._unbound('vp', 'V', bench)))
        assert q.bound is False
        with pytest.raises(UnboundSignalError) as exc:
            q.y
        assert str(exc.value) == (
            "probe 'vp' is detached from its bench (unpickled); "
            "create it again with tb.probe"
        )

    def test_detachment_is_not_confused_with_a_dead_bench(self, bench):
        """Two different facts, two different messages."""
        detached = pickle.loads(pickle.dumps(Signal._unbound('vp', 'V', bench)))
        dead_bench = FakeBench()
        dead = Signal._unbound('vp', 'V', dead_bench)
        del dead_bench
        gc.collect()
        with pytest.raises(UnboundSignalError) as d_exc:
            detached.y
        with pytest.raises(UnboundSignalError) as g_exc:
            dead.y
        assert 'unpickled' in str(d_exc.value)
        assert 'garbage-collected' in str(g_exc.value)

    def test_the_marker_survives_further_round_trips(self, bench):
        q = pickle.loads(pickle.dumps(Signal._unbound('vp', 'V', bench)))
        again = pickle.loads(pickle.dumps(q))
        assert again._bench is _DETACHED_BENCH
        assert copy.deepcopy(q)._bench is _DETACHED_BENCH
        assert again.bound is False

    def test_a_probe_pickles_inside_a_container(self, bench):
        """What actually gets pickled is a metric list, never a bare probe."""
        metrics = [{'name': 'v_avg', 'fn': len, 'slots': {'sig': Signal._unbound('vp', 'V', bench)}}]
        restored = pickle.loads(pickle.dumps(metrics))
        probe = restored[0]['slots']['sig']
        assert probe.name == 'vp'
        assert probe.bound is False

    def test_a_probe_still_resolves_on_the_side_that_was_not_pickled(self, bench):
        """Pickling a probe must not disturb the original."""
        p = Signal._unbound('vout', 'V', bench)
        pickle.dumps(p)
        bench.netlist.sim_results_loaded = True
        assert p.bound is True
        assert np.array_equal(p.y, np.arange(5.0))

    def test_copying_is_not_pickling(self, bench):
        """copy keeps the bench (and TASK-215.8 rebinds it); only pickling
        detaches. The two paths must not share a hook."""
        p = Signal._unbound('vout', 'V', bench)
        assert copy.deepcopy(p)._bench() is bench
        assert copy.copy(p)._bench() is bench
        assert pickle.loads(pickle.dumps(p))._bench is _DETACHED_BENCH

    def test_a_signal_carrying_data_pickles_as_before(self):
        s = bound_signal()
        r = pickle.loads(pickle.dumps(s))
        assert isinstance(r, Signal)
        assert r.name == 'v(out)'
        assert r.type == 'voltage'
        assert r.xlabel == 'time'
        assert r.ylabel == 'v(out)'
        assert np.array_equal(r.y, np.arange(5.0))
        assert np.array_equal(r.x, np.linspace(0.0, 1e-6, 5))
        assert r.bound is True

    def test_deepcopy_of_a_signal_carrying_data_is_independent(self):
        s = bound_signal()
        c = copy.deepcopy(s)
        c.y[0] = 99.0
        assert s.y[0] == 0.0

    def test_a_bus_handle_still_copies_and_pickles(self):
        bus = Signal._make_bus_handle('d', (bound_signal('d0'), bound_signal('d1')))
        for restored in (copy.deepcopy(bus), pickle.loads(pickle.dumps(bus))):
            assert restored.width == 2
            assert len(restored) == 2
            assert restored.y is None
            assert [b.name for b in restored] == ['d0', 'd1']

    def test_a_subclass_round_trips_as_itself(self):
        from silk.signals.spice import NgspiceSignal
        t = np.linspace(0.0, 1e-6, 5)
        s = NgspiceSignal('v(out)', 'voltage', 'time', 'v(out)', t, np.arange(5.0))
        assert type(copy.deepcopy(s)) is NgspiceSignal
        assert type(pickle.loads(pickle.dumps(s))) is NgspiceSignal
        u = Signal._unbound('vp', 'V', FakeBench())
        assert type(pickle.loads(pickle.dumps(u))) is Signal

    def test_a_probe_that_never_had_a_bench_says_so(self):
        """Pickling cannot detach what was never attached: the message stays
        the accurate one."""
        q = pickle.loads(pickle.dumps(Signal._unbound('vx')))
        assert q._bench is None
        assert q.bound is False
        with pytest.raises(UnboundSignalError, match='no testbench is attached'):
            q.y


# ---------------------------------------------------------------------------
# snapshot(): resolve() under the name of what the caller gets (TASK-356)
# ---------------------------------------------------------------------------

class TestSnapshot:

    def test_leaf_probe_matches_resolve(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        bench.netlist.sim_results_loaded = True
        snap, res = p.snapshot(), p.resolve()
        assert snap is not p and snap.bound is True
        assert 'y' in snap.__dict__
        assert np.array_equal(snap.x, res.x) and np.array_equal(snap.y, res.y)

    def test_derived_expression_matches_resolve(self, bench):
        vp = Signal._unbound('vp', 'V', bench)
        vm = Signal._unbound('vm', 'V', bench)
        bench.netlist.sim_results_loaded = True
        snap = ((vp + vm) / 2).snapshot()
        assert 'y' in snap.__dict__
        assert np.array_equal(snap.y, np.arange(5.0))

    def test_explicit_module_and_analysis_are_passed_through(self, bench,
                                                             monkeypatch):
        seen = []
        monkeypatch.setattr(
            Signal, 'resolve',
            lambda self, module=None, *, analysis=None, corner=None:
            seen.append((module, analysis, corner)) or 'r')
        p = Signal._unbound('vout', 'V', bench)
        assert p.snapshot('m', analysis='ac', corner='ff') == 'r'
        assert seen == [('m', 'ac', 'ff')]

    def test_corner_read_matches_resolve(self, bench):
        other = FakeNetlist()
        other.sim_results_loaded = True
        bench._sweep_table = {'tt': '/a.raw', 'ff': '/b.raw'}
        bench._corner_netlist = lambda corner, readers: other
        p = Signal._unbound('vout', 'V', bench)
        snap = p.snapshot(corner='ff')
        assert other.calls == [('vout', 'V')]
        assert np.array_equal(snap.y, p.resolve(corner='ff').y)

    def test_snapshot_keeps_run_one_after_the_bench_reruns(self, bench):
        p = Signal._unbound('vout', 'V', bench)
        bench.netlist.sim_results_loaded = True
        run1 = np.arange(5.0)
        snap = p.snapshot()

        def rerun(name, signal_type='V'):
            t = np.linspace(0.0, 1e-6, 5)
            return Signal(name, 'voltage', 'time', f'v({name})', t, 2 * run1)
        bench.netlist.load_signal = rerun
        assert np.array_equal(p.y, 2 * run1)        # the probe follows
        assert np.array_equal(snap.y, run1)         # the snapshot does not


@pytest.mark.parametrize('expression', [
    lambda p, r: p + 1, lambda p, r: 1 + p,
    lambda p, r: p - 1, lambda p, r: 1 - p,
    lambda p, r: p * 2, lambda p, r: 2 * p,
    lambda p, r: p / 2, lambda p, r: 2 / p,
    lambda p, r: p + r, lambda p, r: r + p,
    lambda p, r: p.mag(), lambda p, r: (r + p).mag(),
])
def test_recipe_bound_tracks_inputs_without_evaluation(bench, monkeypatch, expression):
    p = Signal._unbound('vout', 'V', bench)
    derived = expression(p, bound_signal())
    before = {id(s): dict(s.__dict__) for s in (p, derived)}

    def refuse(*args, **kwargs):
        pytest.fail('bound evaluated a signal')

    monkeypatch.setattr(Signal, 'resolve', refuse)
    monkeypatch.setattr(Signal, 'mag', refuse)
    monkeypatch.setattr(bench.netlist, 'load_signal', refuse)
    assert derived.bound is False
    bench.netlist.sim_results_loaded = True
    assert derived.bound is True
    bench.netlist.sim_results_loaded = False
    assert derived.bound is False
    for s in (p, derived):
        assert s.__dict__ == before[id(s)]
    assert bench.netlist.calls == []


def test_recipe_bound_requires_both_benches_and_bus_children(bench):
    other = FakeBench()
    p = Signal._unbound('a', bench=bench)
    q = Signal._unbound('b', bench=other)
    bus = Signal._make_bus_handle('d', (p, q))
    expressions = (p + q, q + p, bus, bus.as_int(), (bus + 1).mag())
    for ready_a, ready_b, expected in [(False, False, False), (True, False, False),
                                      (True, True, True), (False, True, False)]:
        bench.netlist.sim_results_loaded = ready_a
        other.netlist.sim_results_loaded = ready_b
        assert [s.bound for s in expressions] == [expected] * len(expressions)
    assert bench.netlist.calls == other.netlist.calls == []


def test_recipe_bound_keeps_detached_and_expired_leaves_unready(bench):
    bench.netlist.sim_results_loaded = True
    p = Signal._unbound('a', bench=bench)
    detached = pickle.loads(pickle.dumps(p))
    expired_bench = FakeBench()
    expired = Signal._unbound('b', bench=expired_bench)
    del expired_bench
    gc.collect()
    for missing in (detached, expired, Signal._unbound('c')):
        assert (p + missing).mag().bound is False
        assert (missing + p).bound is False


@pytest.mark.parametrize('selection', ['corner', 'point'])
def test_public_probe_readiness_and_explicit_selection(tmp_path, selection, monkeypatch):
    from silk import Module
    from silk.elements import elements
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile

    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'out': 'output'})
    dut.add(elements.resistor('r', p='out', n='0', value=1000))
    tb = dut.create_test('readiness')
    tb.add_module_under_test({'out': 'out'})
    p = tb.probe('out')
    expressions = (p, p + 1, p.mag(), (p + 1).mag())
    assert [s.bound for s in expressions] == [False] * 4
    path = tmp_path / 'ready.raw'
    vectors = [RawVector('time', 'time', data=np.array([0., 1.])),
               RawVector('v(out)', 'voltage', data=np.array([1., 1.]))]
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'},
                                       vectors, npoints=2, real=True, padded=True)]))
    tb.evaluate(rawfile=str(path))
    for s, expected in zip(expressions, ([1, 1], [2, 2], [1, 1], [2, 2])):
        np.testing.assert_array_equal(s.y, expected)
    reader = tb._result_netlist()
    reader_state = dict(reader.__dict__)
    signal_states = [dict(s.__dict__) for s in expressions]
    with monkeypatch.context() as patch:
        def refuse(*args, **kwargs):
            pytest.fail('bound loaded or evaluated public probe data')
        patch.setattr(Signal, 'resolve', refuse)
        patch.setattr(type(reader), 'load_signal', refuse)
        assert [s.bound for s in expressions] == [True] * 4
    assert tb._result_netlist() is reader
    assert reader.__dict__.keys() == reader_state.keys()
    assert all(reader.__dict__[key] is value for key, value in reader_state.items())
    for s, state in zip(expressions, signal_states):
        assert s.__dict__ == state
    selected = expressions[-1].resolve()
    keys = ['tt', 'ff'] if selection == 'corner' else [('tt', None, 1.), ('tt', None, 2.)]
    tb._sweep_table = dict.fromkeys(keys, str(path))
    assert [s.bound for s in expressions] == [False] * 4
    assert selected.bound is True
    chosen = getattr(expressions[-1], selection)(keys[0])
    assert chosen.bound is True
    assert (chosen + 1).bound is True
    np.testing.assert_array_equal(chosen.y, [2, 2])


def test_public_summed_current_probe_bound(tmp_path):
    from silk import Module, sources
    from silk.elements import elements
    from silk.signals.rawfile import RawFile, RawPlot, RawVector, write_rawfile

    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'out': 'output'})
    dut.add(elements.resistor('r', p='out', n='0', value=1000))
    tb = dut.create_test('current')
    tb.add_module_under_test({'out': 'out'})
    tb.add_source(sources.CurrentSource('stim', positive='out', negative='0', dc=.2,
                                      pulse=sources.Pulse(0., 1., tr=.1, tf=.1, pw=1., per=2.)))
    p = tb.probe('i(stim)')
    assert p.bound is False
    vectors = [RawVector('time', 'time', data=np.array([0., 1.])),
               RawVector('@istim__silk_wave[current]', 'current', data=np.array([1., 2.])),
               RawVector('@istim__silk_dc[current]', 'current', data=np.array([.2, .2]))]
    path = tmp_path / 'current.raw'
    write_rawfile(path, RawFile([RawPlot({'plotname': 'Transient Analysis'},
                                       vectors, npoints=2, real=True, padded=True)]))
    tb.evaluate(rawfile=str(path))
    np.testing.assert_allclose(p.y, [1.2, 2.2])
    assert p.bound is True
