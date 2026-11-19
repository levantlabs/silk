"""Probes across corners (TASK-218).

``run_corners()`` runs one simulation per corner, and every report it returns
retains a reader associated with that member's own run.
Before this task a probe could therefore only ever show that last corner, and
``vcm.plot()`` after a three-corner sweep silently drew one of them.

Four things are proved here.

**The sweep table.** The bench remembers its LATEST sweep as ``{corner:
rawfile path}`` (``tb.sweep_corners``). ``run()`` and ``evaluate()`` leave
exactly one entry; ``run_corners()`` installs the whole set once its corners
have completed — which is also why a metric's probes stay resolvable *during*
a sweep, since mid-sweep the table is never ambiguous. TASK-301.3 (BREAKING):
every corner of a ``run_corners()`` sweep now always writes to its own
``<testbench dir>/<corner>/`` folder, regardless of output mode, so the table
retains every corner unconditionally — the output mode no longer determines
how many corners survive a sweep (see ``TestOverwriteOutputModeNoLongerCollapsesASweep``).

**The refusal.** With more than one corner retained, ``.x``/``.y`` raise
``UnboundSignalError`` naming the corners, because there is no single "current
results" to mean. One entry, or none, behaves exactly as it did before.

**Selection and overlay.** ``vcm.corner('ff')`` returns a BOUND Signal
carrying ff's samples — equal, sample for sample, to a direct read of ff's
rawfile — and a derived expression resolves each of its leaves from that same
file. ``vcm.plot()`` overlays every retained corner with a legend, and
``tb.plot(vp, vm, corners=[...])`` draws one panel per signal and one labelled
trace per corner.

**The live netlist is untouched.** A corner read goes through a separate
``RawResult``; the bench's current results, and a metric re-evaluated from
them, are still the last corner's afterwards.
"""
from __future__ import annotations

import os
import pickle
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from silk import sources
from tests._env import needs_ngspice
from silk.circuit.circuit_elements import PortDir
from silk.circuit.module_test import _retained_sweep
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import UnboundSignalError
from silk.metrics.statistics import average
from silk.simulation import analysis

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)


# ---------------------------------------------------------------------------
# What the table retains — no simulator needed
# ---------------------------------------------------------------------------

class _Report:
    """Just enough SpecReport for :func:`_retained_sweep`."""

    def __init__(self, rawfile_path):
        self.rawfile_path = rawfile_path


class TestRetainedSweep:
    def test_distinct_paths_keep_every_corner_in_order(self):
        table = _retained_sweep({'tt': _Report('/o/a.raw'),
                                 'ff': _Report('/o/b.raw'),
                                 'ss': _Report('/o/c.raw')})
        assert table == {'tt': '/o/a.raw', 'ff': '/o/b.raw', 'ss': '/o/c.raw'}
        assert list(table) == ['tt', 'ff', 'ss']

    def test_one_shared_path_collapses_to_the_last_corner(self):
        """Overwrite mode: three runs, one file, and it holds ss."""
        table = _retained_sweep({'tt': _Report('/o/r.raw'),
                                 'ff': _Report('/o/r.raw'),
                                 'ss': _Report('/o/r.raw')})
        assert table == {'ss': '/o/r.raw'}

    def test_paths_are_compared_normalized(self):
        table = _retained_sweep({'tt': _Report('/o/./r.raw'),
                                 'ff': _Report('/o/r.raw')})
        assert list(table) == ['ff']

    def test_a_report_without_a_rawfile_contributes_nothing(self):
        table = _retained_sweep({'tt': _Report(None),
                                 'ff': _Report('/o/b.raw')})
        assert table == {'ff': '/o/b.raw'}

    def test_no_rawfiles_at_all_is_an_empty_table(self):
        assert _retained_sweep({'tt': _Report(None)}) == {}


# ---------------------------------------------------------------------------
# A wired bench, never simulated: the table's API and every refusal
# ---------------------------------------------------------------------------

@pytest.fixture
def tb(pdk, tmp_dir):
    dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                 module_name='dut')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
    test = dut.create_test('t')
    test.add_module_under_test({'a': 'vin', 'b': 'vout'})
    test.add_source(sources.VoltageSource('vs', positive='vin', negative='0',
                                          dc=0.9))
    test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return test


class TestTheTableAPI:
    def test_a_fresh_bench_has_retained_nothing(self, tb):
        assert tb.sweep_corners == {}

    def test_sweep_corners_is_a_copy(self, tb):
        tb._sweep_table = {'tt': '/o/a.raw'}
        snapshot = tb.sweep_corners
        snapshot['ff'] = '/o/b.raw'
        assert tb.sweep_corners == {'tt': '/o/a.raw'}

    def test_a_corner_read_needs_a_sweep(self, tb):
        with pytest.raises(UnboundSignalError) as e:
            tb.probe('vout').corner('tt')
        assert 'no results to select a corner from' in str(e.value)

    def test_an_unknown_corner_lists_the_retained_ones(self, tb):
        tb._sweep_table = {'tt': '/o/a.raw', 'ss': '/o/c.raw'}
        with pytest.raises(UnboundSignalError) as e:
            tb.probe('vout').corner('ff')
        message = str(e.value)
        assert "did not retain corner 'ff'" in message
        assert "'tt'" in message and "'ss'" in message

    def test_reading_y_refuses_naming_every_retained_corner(self, tb):
        """The bench 'ran' three corners; .y has nothing to choose with."""
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw',
                           'ss': '/o/c.raw'}
        tb.netlist.sim_results_loaded = True
        with pytest.raises(UnboundSignalError) as e:
            tb.probe('vout').y
        message = str(e.value)
        assert 'ran 3 corners' in message
        assert "'tt'" in message and "'ff'" in message and "'ss'" in message
        assert '.corner(' in message and '.plot()' in message

    def test_one_retained_corner_resolves_as_before(self, tb, monkeypatch):
        """A single entry is never ambiguous: the live results answer."""
        tb._sweep_table = {'tt': '/o/a.raw'}
        tb.netlist.sim_results_loaded = True
        monkeypatch.setattr(type(tb.netlist), 'load_signal',
                            lambda self, name, sel='V', **kw: name)
        assert tb.probe('vout').resolve() == 'vout'

    def test_bound_is_false_while_several_corners_are_retained(self, tb):
        tb.netlist.sim_results_loaded = True
        probe = tb.probe('vout')
        tb._sweep_table = {'tt': '/o/a.raw'}
        assert probe.bound is True
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw'}
        assert probe.bound is False

    def test_corner_on_a_signal_that_already_carries_data_is_refused(self):
        from silk.signals.base import Signal
        bound = Signal('vout', 'voltage', 'time', 'v',
                       np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError) as e:
            bound.corner('tt')
        assert 'already carries data' in str(e.value)

    def test_plot_corners_on_a_bound_signal_is_refused(self):
        from silk.signals.base import Signal
        bound = Signal('vout', 'voltage', 'time', 'v',
                       np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError):
            bound.plot(corners=['tt'])
        plt.close('all')

    def test_an_empty_corner_list_selects_nothing_and_says_so(self, tb):
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw'}
        with pytest.raises(ValueError) as e:
            tb._corners_to_draw([])
        assert 'selects nothing' in str(e.value)

    def test_a_single_name_is_accepted_as_a_string(self, tb):
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw'}
        assert tb._corners_to_draw('ff') == ['ff']

    def test_the_default_is_every_corner_only_when_several_are_retained(self, tb):
        assert tb._corners_to_draw(None) is None
        tb._sweep_table = {'tt': '/o/a.raw'}
        assert tb._corners_to_draw(None) is None
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw'}
        assert tb._corners_to_draw(None) == ['tt', 'ff']

    def test_the_copy_of_a_bench_retains_no_sweep(self, tb):
        """A composed replica must not read the TEMPLATE bench's files."""
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw'}
        assert copy.deepcopy(tb).sweep_corners == {}

    def test_an_unpickled_bench_keeps_its_sweep(self, tb):
        """Same bench, and the files it names are still on disk."""
        tb._sweep_table = {'tt': '/o/a.raw', 'ff': '/o/b.raw'}
        assert pickle.loads(pickle.dumps(tb)).sweep_corners == {
            'tt': '/o/a.raw', 'ff': '/o/b.raw'}

    def test_the_docstring_explains_successful_corners_are_retained_in_both_modes(self):
        """The sweep table's docstring says each successful run_corners() member keeps its result file in both output modes."""
        from silk.circuit.module_test import ModuleTestbench
        doc = ModuleTestbench.sweep_corners.__doc__
        assert 'Each successful ``run_corners()`` member retains its result file' in doc
        assert "``mode='new'`` (the default) and ``mode='overwrite'``" in doc


class TestTheCornerReader:
    @pytest.mark.parametrize('state', [None, {}])
    def test_missing_run_state_refuses_with_recovery(self, tb, state):
        from silk.exceptions import ConfigurationError
        tb._sweep_table = {'ff': '/o/b.raw'}
        tb._sweep_states = {} if state is None else {'ff': state}
        with pytest.raises(ConfigurationError, match="t.*ff.*Rerun.*reevaluate"):
            tb._corner_netlist('ff')
        with pytest.raises(ConfigurationError, match="t.*ff.*Rerun.*reevaluate"):
            tb.probe('vout').point('ff')

    def test_one_reader_is_shared_by_the_leaves_of_one_expression(self, reference_grid):
        tb, _, keys = reference_grid
        readers = {}
        first = tb._corner_netlist(keys[0], readers)
        assert tb._corner_netlist(keys[0], readers) is first
        assert tb._corner_netlist(keys[0], None) is not first
        assert first.load_signal('vout', analysis='tran').y[-1] == 1
        assert first.results_file == tb.sweep_corners[keys[0]]


# ---------------------------------------------------------------------------
# Real ngspice — the RC circuit of tests/test_probe.py, three corners
# ---------------------------------------------------------------------------

def _rc_bench(tmp_dir, mode):
    """The 1 kOhm / 1 nF RC low-pass, driven by a vpulse, in *mode*.

    An RC has no corner dependence at all — tt, ff and ss produce the same
    waveform — which is the point: every assertion below is about WHICH file
    was read and HOW MANY traces were drawn, never about a value that happens
    to differ.
    """
    from silk.pdk.basepdk import BasePDK
    pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=pdk,
                 module_name='rc')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    dut.set_output_options(mode=mode)
    test = dut.create_test('rc')
    test.add_module_under_test({'a': 'vin', 'b': 'vout'})
    test.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                            pw=5e-6, per=10e-6)))
    test.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))
    return test


def _direct_read(path, name, backend):
    """The corner's file read straight off disk, with nothing of silk's
    testbench machinery in between."""
    from silk.signals.rawfile import read_rawfile
    from silk.simulation.backend import RawResult
    return backend.load_signal(
        RawResult(native=read_rawfile(path), source_path=path), name)


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestThreeCornersPerRunMode:
    """``set_output_options(mode='new')``: a directory per run, so all three
    corners' rawfiles survive the sweep."""

    @pytest.fixture(scope='class')
    def swept(self, tmp_path_factory):
        tb = _rc_bench(str(tmp_path_factory.mktemp('rc_new')), 'new')
        probes = {'vout': tb.probe('vout'), 'vin': tb.probe('vin')}
        probes['vcm'] = (probes['vout'] + probes['vin']) / 2
        tb.add_metric('vout_avg', average, sig=probes['vout'])
        res = tb.run_corners(['tt', 'ff', 'ss'])
        return tb, probes, res

    def test_every_corner_is_retained_with_its_own_file(self, swept):
        tb, _, _ = swept
        assert list(tb.sweep_corners) == ['tt', 'ff', 'ss']
        assert len(set(tb.sweep_corners.values())) == 3
        for path in tb.sweep_corners.values():
            assert os.path.exists(path)

    def test_each_report_records_the_file_it_was_extracted_from(self, swept):
        """AC #6: the per-corner SpecReport carries its rawfile path."""
        tb, _, res = swept
        for corner, report in res.items():
            assert report.rawfile_path == tb.sweep_corners[corner]

    def test_reading_y_refuses_naming_the_three_corners(self, swept):
        _, probes, _ = swept
        with pytest.raises(UnboundSignalError) as e:
            probes['vout'].y
        message = str(e.value)
        assert 'ran 3 corners' in message
        assert "'tt'" in message and "'ff'" in message and "'ss'" in message

    def test_a_derived_expression_refuses_too(self, swept):
        _, probes, _ = swept
        with pytest.raises(UnboundSignalError):
            probes['vcm'].y

    def test_corner_equals_a_direct_read_of_that_corners_file(self, swept):
        """AC #1: .corner('ff').y IS ff's file, sample for sample."""
        tb, probes, _ = swept
        expected = _direct_read(tb.sweep_corners['ff'], 'vout',
                                tb.netlist.backend)
        got = probes['vout'].corner('ff')
        assert np.array_equal(got.y, expected.y)
        assert np.array_equal(got.x, expected.x)
        assert got.xlabel == expected.xlabel

    def test_each_corner_reads_its_own_file(self, swept):
        tb, probes, _ = swept
        for corner, path in tb.sweep_corners.items():
            expected = _direct_read(path, 'vout', tb.netlist.backend)
            assert np.array_equal(probes['vout'].corner(corner).y, expected.y)

    def test_the_selected_corner_is_a_bound_signal(self, swept):
        _, probes, _ = swept
        selected = probes['vout'].corner('tt')
        assert selected.bound is True
        assert 'y' in selected.__dict__          # carries data, not a recipe
        assert float(np.max(selected.y)) == pytest.approx(1.0, abs=0.02)

    def test_a_derived_expression_resolves_every_leaf_from_one_corner(self, swept):
        tb, probes, _ = swept
        vout = _direct_read(tb.sweep_corners['ff'], 'vout', tb.netlist.backend)
        vin = _direct_read(tb.sweep_corners['ff'], 'vin', tb.netlist.backend)
        assert np.allclose(probes['vcm'].corner('ff').y,
                           (vout.y + vin.y) / 2)

    def test_a_deferred_transform_survives_corner_selection(self, swept):
        tb, probes, _ = swept
        slew = probes['vout'].derivative()
        expected = _direct_read(tb.sweep_corners['ss'], 'vout',
                                tb.netlist.backend).derivative()
        assert np.allclose(slew.corner('ss').y, expected.y)

    def test_plot_overlays_every_corner_with_a_legend(self, swept):
        """AC #1: three labelled traces."""
        _, probes, _ = swept
        ax = probes['vout'].plot()
        assert len(ax.lines) == 3
        assert [line.get_label() for line in ax.lines] == ['tt', 'ff', 'ss']
        assert ax.get_legend() is not None
        plt.close('all')

    def test_a_derived_signal_overlays_its_corners_too(self, swept):
        """The design doc's own line: ``vcm.plot()`` after a sweep."""
        tb, probes, _ = swept
        ax = probes['vcm'].plot()
        assert len(ax.lines) == 3
        assert [line.get_label() for line in ax.lines] == ['tt', 'ff', 'ss']
        assert ax.get_legend() is not None
        for line, corner in zip(ax.lines, ['tt', 'ff', 'ss']):
            vout = _direct_read(tb.sweep_corners[corner], 'vout',
                                tb.netlist.backend)
            vin = _direct_read(tb.sweep_corners[corner], 'vin',
                               tb.netlist.backend)
            assert np.allclose(line.get_ydata(), (vout.y + vin.y) / 2)
        plt.close('all')

    def test_plot_can_select_a_subset(self, swept):
        _, probes, _ = swept
        ax = probes['vout'].plot(corners=['tt', 'ss'])
        assert [line.get_label() for line in ax.lines] == ['tt', 'ss']
        plt.close('all')

    def test_a_selected_corner_plots_as_an_ordinary_signal(self, swept):
        _, probes, _ = swept
        ax = probes['vout'].corner('ff').plot()
        assert len(ax.lines) == 1
        plt.close('all')

    def test_tb_plot_draws_a_panel_per_signal_and_a_trace_per_corner(self, swept):
        """AC #4: two probes, two corners, four traces."""
        tb, probes, _ = swept
        fig = tb.plot(probes['vout'], probes['vin'], corners=['tt', 'ss'],
                      show=False)
        panels = fig.axes
        assert len(panels) == 2
        assert [len(ax.lines) for ax in panels] == [2, 2]
        assert sum(len(ax.lines) for ax in panels) == 4
        for ax in panels:
            assert [line.get_label() for line in ax.lines] == ['tt', 'ss']
            assert ax.get_legend() is not None
        plt.close('all')

    def test_tb_plot_defaults_to_every_retained_corner(self, swept):
        tb, probes, _ = swept
        fig = tb.plot(probes['vout'], show=False)
        assert [line.get_label() for line in fig.axes[0].lines] == \
            ['tt', 'ff', 'ss']
        plt.close('all')

    def test_tb_plot_overlay_puts_every_corner_on_one_axis(self, swept):
        tb, probes, _ = swept
        fig = tb.plot(probes['vout'], probes['vin'], corners=['tt', 'ss'],
                      overlay=True, show=False)
        assert len(fig.axes) == 1
        labels = [line.get_label() for line in fig.axes[0].lines]
        assert labels == ['vout (tt)', 'vout (ss)', 'vin (tt)', 'vin (ss)']
        plt.close('all')

    def test_an_unknown_corner_is_refused_listing_the_retained_ones(self, swept):
        """AC #4: on the bench and on a Signal, one message."""
        tb, probes, _ = swept
        for call in (lambda: tb.plot(probes['vout'], corners=['tt', 'xx'],
                                     show=False),
                     lambda: probes['vout'].plot(corners=['xx']),
                     lambda: probes['vout'].corner('xx')):
            with pytest.raises(UnboundSignalError) as e:
                call()
            message = str(e.value)
            assert "'xx'" in message
            assert "'tt'" in message and "'ff'" in message and "'ss'" in message
        plt.close('all')

    def test_a_corner_read_leaves_the_live_results_alone(self, swept):
        """AC #3, by identity: no reload, no rebind, no path change."""
        tb, probes, _ = swept
        before_file = tb.netlist.results_file
        before_raw = tb.netlist._raw
        before_vout = tb.netlist.load_signal('vout').y.copy()
        probes['vout'].corner('tt').y
        probes['vcm'].corner('ff').y
        assert tb.netlist.results_file == before_file
        assert tb.netlist._raw is before_raw
        assert np.array_equal(tb.netlist.load_signal('vout').y, before_vout)

    def test_the_live_results_are_still_the_last_corners(self, swept):
        tb, _, _ = swept
        assert tb.netlist.results_file == tb.sweep_corners['ss']


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestReEvaluationAfterCornerReads:
    """AC #3's other half: a metric re-extracted from the live results after a
    corner read still measures the last corner.

    Its own bench, because ``evaluate()`` rewrites the sweep table to the one
    corner it evaluated — that is the documented contract, and a test that
    mutates it must not be sharing a bench with the ones above.
    """

    @pytest.fixture(scope='class')
    def swept(self, tmp_path_factory):
        tb = _rc_bench(str(tmp_path_factory.mktemp('rc_eval')), 'new')
        vout = tb.probe('vout')
        tb.add_metric('vout_avg', average, sig=vout)
        res = tb.run_corners(['tt', 'ff', 'ss'])
        return tb, vout, res

    def test_a_metric_re_evaluated_after_corner_reads_is_the_last_corners(
            self, swept):
        tb, vout, res = swept
        vout.corner('tt').y
        vout.corner('ff').y
        report = tb.evaluate('ss')
        assert report.metrics['vout_avg'] == res['ss'].metrics['vout_avg']
        assert report.rawfile_path == res['ss'].rawfile_path

    def test_evaluate_leaves_one_retained_corner(self, swept):
        """Ordered after the re-evaluation above: the table is now that one
        run, and .y resolves again."""
        tb, vout, res = swept
        assert tb.sweep_corners == {'ss': res['ss'].rawfile_path}
        assert np.array_equal(vout.y, vout.corner('ss').y)


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestAPlainRunAfterASweep:
    """AC #1's last clause: ``run()`` resets the table to a single entry."""

    @pytest.fixture(scope='class')
    def rerun(self, tmp_path_factory):
        tb = _rc_bench(str(tmp_path_factory.mktemp('rc_rerun')), 'new')
        vout = tb.probe('vout')
        tb.run_corners(['tt', 'ff', 'ss'])
        with pytest.raises(UnboundSignalError):
            vout.y
        report = tb.run('ff')
        return tb, vout, report

    def test_the_table_holds_only_that_run(self, rerun):
        tb, _, report = rerun
        assert tb.sweep_corners == {'ff': report.rawfile_path}

    def test_reading_y_resolves_again(self, rerun):
        tb, vout, _ = rerun
        assert np.array_equal(vout.y, tb.netlist.load_signal('vout').y)
        assert vout.bound is True

    def test_plot_draws_one_trace_again(self, rerun):
        _, vout, _ = rerun
        ax = vout.plot()
        assert len(ax.lines) == 1
        plt.close('all')


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestOverwriteOutputModeNoLongerCollapsesASweep:
    """TASK-301.3 (BREAKING): every corner of a ``run_corners()`` sweep now
    gets its own ``<testbench dir>/<corner>/`` folder unconditionally, so
    overwrite mode — the default — no longer collapses the sweep to the
    last corner's file the way it did before this task. Overwrite mode
    still governs a REPEATED plain ``run()``/``evaluate()`` at the SAME
    corner (unaffected, and not tested here); this class is specifically
    about a multi-corner ``run_corners()`` sweep, which now behaves exactly
    like :class:`TestThreeCornersPerRunMode` regardless of output mode."""

    @pytest.fixture(scope='class')
    def swept(self, tmp_path_factory):
        tb = _rc_bench(str(tmp_path_factory.mktemp('rc_over')), 'overwrite')
        vout = tb.probe('vout')
        res = tb.run_corners(['tt', 'ff', 'ss'])
        return tb, vout, res

    def test_every_corner_is_retained_with_its_own_file(self, swept):
        tb, _, _ = swept
        assert list(tb.sweep_corners) == ['tt', 'ff', 'ss']
        assert len(set(tb.sweep_corners.values())) == 3
        for path in tb.sweep_corners.values():
            assert os.path.exists(path)

    def test_every_report_names_a_distinct_file(self, swept):
        _, _, res = swept
        assert len({r.rawfile_path for r in res.values()}) == 3

    def test_reading_y_refuses_naming_the_three_corners(self, swept):
        _, vout, _ = swept
        with pytest.raises(UnboundSignalError) as e:
            vout.y
        assert 'ran 3 corners' in str(e.value)

    def test_plot_draws_three_traces(self, swept):
        _, vout, _ = swept
        ax = vout.plot()
        assert len(ax.lines) == 3
        plt.close('all')

    def test_every_corner_can_be_selected(self, swept):
        tb, vout, _ = swept
        for corner in ('tt', 'ff', 'ss'):
            expected = _direct_read(tb.sweep_corners[corner], 'vout',
                                    tb.netlist.backend)
            assert np.array_equal(vout.corner(corner).y, expected.y)


def test_point_probe_resolves_expression_with_exact_tuple_key(tb, monkeypatch):
    point = ('tt', 125.0, 1.62)
    tb._sweep_table = {point: '/retained/point.raw'}
    tb.netlist.sim_results_loaded = True
    from silk.exceptions import ConfigurationError
    assert tb._point_corner(point) is point
    with pytest.raises(ConfigurationError, match='recorded run state'):
        tb._corner_netlist(point)
    with pytest.raises(UnboundSignalError, match='point'):
        tb.probe('vout').corner('tt')
    from silk.signals import Signal
    seen = []
    original = Signal.resolve
    def resolve(signal, *args, **kwargs):
        seen.append(kwargs.get('corner'))
        return 'selected'
    monkeypatch.setattr(Signal, 'resolve', resolve)
    assert (tb.probe('vout') + tb.probe('vin')).point(point) == 'selected'
    assert seen == [point]
    monkeypatch.setattr(Signal, 'resolve', original)
    with pytest.raises(UnboundSignalError, match='point'):
        tb._point_corner(('tt', -40.0, 1.62))


def test_grid_readers_are_scoped_to_bench(tb):
    point=('tt',125.0,1.62)
    other=copy.deepcopy(tb)
    tb._sweep_table={point:'/first.raw'}
    other._sweep_table={point:'/second.raw'}
    readers={}
    from silk.exceptions import ConfigurationError
    for bench in (tb, other):
        with pytest.raises(ConfigurationError, match='recorded run state'):
            bench._corner_netlist(point, readers)


@pytest.mark.parametrize('other_name', ['out', 'other'])
def test_corner_expression_keeps_bench_readers_separate(tmp_path, other_name):
    from tests._mock_rawfile import write_complete_rawfile
    benches = []
    try:
        for name, value in [('first', 1.0), ('second', 9.0)]:
            mod = Module(None, str(tmp_path / name), name + '.cir', None, module_name=name)
            bench = mod.create_test('read')
            mod.define_module_ports({})
            bench.add_module_under_test({})
            raw = tmp_path / (name + '.raw')
            write_complete_rawfile(raw, 'test\n.tran 1n 10n\n.end\n',
                                   values={'v(out)': value, 'v(other)': value})
            bench.evaluate(corner='tt', rawfile=raw)
            benches.append(bench)
        a, b = benches[0].probe('out'), benches[1].probe(other_name)
        for left, right, difference in [(a, b, -8), (b, a, 8)]:
            np.testing.assert_allclose((left.corner('tt') + right.corner('tt')).y, 10)
            np.testing.assert_allclose((left + right).corner('tt').y, 10)
            np.testing.assert_allclose((left + right).point('tt').y, 10)
            np.testing.assert_allclose((left - right).corner('tt').y, difference)
            np.testing.assert_allclose((left - right).point('tt').y, difference)
        readers = {}
        first = benches[0]._corner_netlist('tt', readers)
        assert benches[0]._corner_netlist('tt', readers) is first
        assert benches[1]._corner_netlist('tt', readers) is not first
        assert benches[0]._corner_netlist('tt') is not first
        # A new resolution must use the currently retained path.
        benches[0].evaluate(corner='tt', rawfile=tmp_path / 'second.raw')
        np.testing.assert_allclose((a + b).corner('tt').y, 18)
        np.testing.assert_allclose((a + b).point('tt').y, 18)
    finally:
        for bench in benches:
            bench.module.close()


@needs_ngspice
@pytest.mark.parametrize('plotter', ['probe', 'bench', 'overlay'])
def test_grid_plot_preserves_point_data_and_labels(tb, plotter, monkeypatch):
    tb.set_supply_voltages([1.0, 2.0], source='vs')
    probe = tb.probe('vout', analysis='tran')
    expression = (probe + tb.probe('vin', analysis='tran')) / 2
    reports = tb.run()
    keys = [('tt', None, 1.0), ('tt', None, 2.0)]
    assert list(reports.reports) == keys
    paths = {k: open(p, 'rb').read() for k, p in tb.sweep_corners.items()}
    live = tb.netlist.results_file
    native = tb.netlist.simResultsAll
    monkeypatch.setattr(type(tb), '_simulate_run', staticmethod(
        lambda *a, **k: pytest.fail('plotting launched a simulation')))
    try:
        if plotter == 'probe':
            ax = expression.plot(mode='mag')
        else:
            fig = tb.plot(expression, probe, show=False, overlay=plotter == 'overlay')
            assert len(fig.axes) == (1 if plotter == 'overlay' else 2)
            ax = fig.axes[0]
        lines = ax.lines[:2]
        assert len(ax.lines) == (4 if plotter == 'overlay' else 2)
        for line, value in zip(lines, [1.0, 2.0]):
            np.testing.assert_allclose(line.get_ydata(), value)
            assert line.get_xdata()[-1] == pytest.approx(10e-9 if plotter == 'probe' else 10.0)
        labels = [line.get_label() for line in lines]
        if plotter == 'overlay':
            assert all(str(key) in label for key, label in zip(keys, labels))
        else:
            assert labels == [str(key) for key in keys]
        assert [t.get_text() for t in ax.get_legend().texts] == [l.get_label() for l in ax.lines]
        plt.close('all')
        assert len(probe.point(keys[0]).plot().lines) == 1
        with pytest.raises(ValueError, match='already carries data'):
            tb.plot(probe.point(keys[0]), show=False)
        assert tb.netlist.results_file == live
        assert tb.netlist.simResultsAll is native
        assert {k: open(p, 'rb').read() for k, p in tb.sweep_corners.items()} == paths
    finally:
        plt.close('all')


@pytest.fixture
def reference_grid(tb, tmp_path, monkeypatch):
    """Retained files exercise selection and resolution without a simulator."""
    from tests._mock_rawfile import write_complete_rawfile
    from silk.signals import Signal
    keys = [('tt', None, 1.0), ('tt', None, 2.0)]
    for index, key in enumerate(keys):
        raw = tmp_path / f'reference_{index}.raw'
        write_complete_rawfile(raw, 'test\n.tran 1n 10n\n.end\n',
                               values={'v(vout)': index + 1.0})
        tb.evaluate(rawfile=str(raw))
        from silk.circuit.test_placement import EvidenceResults
        reader = EvidenceResults(tb.netlist, _retained=True)
        if index == 0:
            states, paths = {}, {}
        states[key] = {name: value for name, value in vars(reader).items()
                       if name not in {'_module', '_evidence_owner', 'backend'}}
        paths[key] = str(raw)
    tb._sweep_table = paths
    tb._sweep_states = states
    monkeypatch.setattr(type(tb), '_simulate_run', staticmethod(
        lambda *a, **k: pytest.fail('selection launched a simulation')))
    reference = Signal('reference', 'voltage', 'time', 'voltage',
                       [0, 10e-9], [2, 2])
    return tb, reference, keys


@pytest.mark.parametrize('selector', ['point', 'sample'])
def test_reference_before_probe_selects_retained_data(reference_grid, selector):
    tb, reference, keys = reference_grid
    if selector == 'sample':
        from silk.circuit.module_test import _sample_label
        keys = [(key, 0) for key in keys]
        tb._mc_samples = {key: path for key, path in zip(keys, tb._sweep_table.values())}
        tb._sweep_states = {_sample_label((key, 0)): value
                            for key, value in tb._sweep_states.items()}
        tb._sweep_table = {_sample_label(key): path for key, path in tb._mc_samples.items()}
    probe = tb.probe('vout')
    for expression, expected in [((reference + probe).mag(), 3),
                                 (reference + probe, 3), (probe + reference, 3),
                                 (reference - probe, 1), (probe - reference, -1),
                                 ((reference + probe) * 2, 6)]:
        np.testing.assert_allclose(
            getattr(expression, selector)(keys[0], analysis='tran').y, expected)
    with pytest.raises(UnboundSignalError, match='point|sample'):
        getattr(reference + probe, selector)(('missing', 99))
    with pytest.raises(ValueError, match='already carries data'):
        getattr(reference, selector)(keys[0])
    from silk.signals import Signal
    incompatible = Signal('bad', 'voltage', 'time', 'voltage', [0, 1], [2, 2])
    with pytest.raises(ValueError, match='coordinates'):
        getattr(incompatible + probe, selector)(keys[0], analysis='tran')


def test_reference_first_plot_uses_retained_order(reference_grid):
    tb, reference, keys = reference_grid
    expression = reference + tb.probe('vout', analysis='tran')
    try:
        ax = expression.plot()
        assert [line.get_label() for line in ax.lines] == list(map(str, keys))
        for line, key, expected in zip(ax.lines, keys, [3, 4]):
            np.testing.assert_allclose(line.get_ydata(), expected)
            np.testing.assert_allclose(line.get_ydata(), expression.point(key).y)
    finally:
        plt.close('all')


@pytest.mark.parametrize('expired', [False, True])
def test_reference_does_not_skip_missing_deferred_owner(reference_grid, expired):
    import weakref
    from silk.signals import Signal
    tb, reference, keys = reference_grid
    class Owner:
        pass
    owner = Owner()
    missing = Signal._unbound('missing', bench=owner if expired else None)
    if expired:
        ref = weakref.ref(owner)
        del owner
        assert ref() is None
    expression = reference + missing + tb.probe('vout')
    for selector in ('point', 'sample'):
        with pytest.raises(UnboundSignalError, match='live testbench'):
            getattr(expression, selector)(keys[0])
    with pytest.raises(UnboundSignalError, match='live testbench'):
        expression._corners_to_draw(['tt'])


def test_reference_mixed_benches_keep_leaf_owners(reference_grid):
    tb, reference, keys = reference_grid
    other = copy.deepcopy(tb)
    try:
        other._sweep_table = dict(zip(keys[::-1], tb._sweep_table.values()))
        other._sweep_states = dict(zip(keys[::-1], tb._sweep_states.values()))
        expression = reference + tb.probe('vout') - other.probe('vout')
        assert expression._corners_to_draw(None) == keys
        assert (reference + other.probe('vout') - tb.probe('vout'))._corners_to_draw(None) == keys[::-1]
        np.testing.assert_allclose(expression.point(keys[0], analysis='tran').y, 1)
    finally:
        other.module.close()


@pytest.mark.slow
@needs_ngspice
@pytest.mark.parametrize('declared', [False, True])
def test_collection_members_keep_their_run_without_vector_caches(tmp_path, declared):
    from silk.circuit.test_placement import EvidenceResults
    tb = _rc_bench(str(tmp_path), 'overwrite')
    authored_analyses = tb.netlist.sim_analysis_dict
    authored_options = tb.netlist.simulator_options
    authored_transient = authored_analyses['transient']
    if declared:
        tb.set_corners(['tt', 'ss'])
        reports = tb.run()
    else:
        reports = tb.run_corners(['tt', 'ss'])
    assert tb.netlist.sim_analysis_dict is authored_analyses
    assert tb.netlist.simulator_options is authored_options
    assert tb.netlist.sim_analysis_dict['transient'] is authored_transient
    for key, report in reports.items():
        reader = report.sim_results
        assert type(reader) is EvidenceResults
        assert reader.results_file == report.rawfile_path
        expected = _direct_read(report.rawfile_path, 'vout', tb.netlist.backend)
        np.testing.assert_array_equal(reader.load_signal('vout').y, expected.y)
        assert reader._raw.native is reader.simResultsAll
        assert all(vec._data is None for plot in reader.simResultsAll.plots
                   for vec in plot.vectors)
        assert reader.operating_conditions == report.operating_conditions


@pytest.mark.slow
@needs_ngspice
@pytest.mark.parametrize('operation', ['pickle', 'copy'])
def test_direct_collection_copy_pickle_preserves_run_associations(tmp_path, operation, monkeypatch):
    from silk.signals.rawfile import read_rawfile
    tb = _rc_bench(str(tmp_path), 'overwrite')
    reports = tb.run_corners(['tt', 'ss'])
    convert = (lambda obj: pickle.loads(pickle.dumps(obj))) if operation == 'pickle' else copy.copy
    # Unpickling restores associations only; it must not perform artifact I/O.
    with monkeypatch.context() as patch:
        patch.setattr('silk.signals.rawfile.read_rawfile',
                      lambda *a, **k: pytest.fail('copy/unpickle opened an artifact'))
        copies = [convert(obj) for obj in (tb, tb.module, reports, reports['tt'])]
    bench, module, collection, member = copies
    for result in (collection['tt'], member):
        np.testing.assert_array_equal(result.sim_results.load_signal('vout').y,
                                      reports['tt'].sim_results.load_signal('vout').y)
        assert result.sim_results.results_file == reports['tt'].rawfile_path
    for restored in (bench, module.get_test('rc')):
        np.testing.assert_array_equal(restored.probe('vout').corner('tt').y,
                                      reports['tt'].sim_results.load_signal('vout').y)
    deep = copy.deepcopy(tb)
    assert deep.sweep_corners == deep._sweep_states == deep._sweep_generations == {}


def test_external_replacement_and_extraction_select_current_state(tb, tmp_path):
    from tests._mock_rawfile import write_complete_rawfile
    selected = []
    def extract(module):
        reader = tb._corner_netlist('tt')
        assert reader._retained is False
        selected.append(float(tb.probe('vout').corner('tt').y[-1]))
        return selected[-1]
    tb.add_metric('during', extract)
    for number in (1., 2.):
        path = tmp_path / f'external_{number}.raw'
        write_complete_rawfile(path, 'test\n.tran 1n 10n\n.end\n', values={'v(vout)': number})
        report = tb.evaluate(rawfile=str(path), corner='tt')
        assert report.sim_results is tb.netlist
        assert report.metrics['during'] == number
        assert tb.probe('vout').corner('tt').y[-1] == number
    assert selected == [1., 2.]


def test_scalar_execution_selector_accepts_memory_only_evidence(tb):
    from silk.circuit.test_placement import EvidenceResults
    from silk.signals.rawfile import RawFile, RawPlot, RawVector
    from silk.simulation.backend import RawResult
    native = RawFile([RawPlot({'plotname': 'Transient Analysis'}, [
        RawVector('time', 'time', data=np.array([0., 1.])),
        RawVector('v(vout)', 'voltage', data=np.array([4., 5.]))],
        npoints=2, real=True, padded=True)])
    tb.netlist._raw = RawResult(native, '<memory>')
    tb.netlist.simResultsAll, tb.netlist.sim_results_loaded = native, True
    tb.netlist.results_file = '<memory>'
    reader = EvidenceResults(tb.netlist)
    tb._sweep_table = {'tt': '<memory>'}
    tb._sweep_states = {'tt': {name: value for name, value in vars(reader).items()
                                if name not in {'_module', '_evidence_owner', 'backend'}}}
    np.testing.assert_array_equal(tb.probe('vout').corner('tt').y, [4., 5.])
    assert tb._corner_netlist('tt')._retained is False


def test_retained_reader_honors_analysis_selection(tb, tmp_path):
    from tests._mock_rawfile import write_complete_rawfile
    from silk.circuit.test_placement import EvidenceResults
    from silk.exceptions import SimulationError
    path = str(tmp_path / 'multiple.raw')
    write_complete_rawfile(path, 'test\n.tran 1n 10n\n.ac lin 3 1 3\n.end\n',
                           values={'v(vout)': 2.})
    tb.netlist._raw = tb.netlist.backend.load_raw(path)
    tb.netlist.simResultsAll = tb.netlist._raw.native
    tb.netlist.sim_results_loaded, tb.netlist.results_file = True, path
    reader = EvidenceResults(tb.netlist, _retained=True)
    assert reader.load_signal('vout', analysis='tran').xlabel == 'time'
    assert reader.load_signal('vout', analysis='ac').xlabel == 'frequency'
    with pytest.raises(SimulationError):
        reader.load_signal('vout')


@pytest.mark.parametrize('shape', ['corners', 'declared', 'all', 'conditions', 'grid', 'scalar'])
def test_collection_logical_current_reads_its_own_run(tmp_path, pdk, shape, monkeypatch):
    from pathlib import Path
    from tests._mock_rawfile import write_complete_rawfile
    dut = Module('sky130', str(tmp_path), 'owned.cir', pdk, module_name='owned')
    dut.define_module_ports({'b': PortDir.OUT})
    dut.add(elements.resistor('load', p='b', n='0', value=1000))
    test = dut.create_test('owned')
    test.add_module_under_test({'b': 'vout'})
    test.add_source(sources.VoltageSource('vs', 'vout', '0', dc=.2, pwl=[0, .5, 1, 1]))
    test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    test.add_metric('direct_current', lambda c: float(c.load_signal('i(vs)').y[-1]))
    test.add_metric('string_current', type='average', signal='i(vs)')
    current = test.probe('i(vs)')
    voltage = test.probe('vout')
    expression = voltage + voltage * 2
    backend = test.netlist.backend
    launches = []

    def launch(_backend, netlist_path, results_dir, **kwargs):
        launches.append(netlist_path)
        value = float(len(launches))
        path = Path(results_dir) / 'owned.raw'
        write_complete_rawfile(path, Path(netlist_path).read_text(), values={
            'v(vout)': value, 'i(vvs__silk_pwl)': -value / 1000})
        raw = backend.load_raw(str(path))
        if shape == 'conditions':
            raw.ambient_temperatures = (0. if len(launches) == 1 else 85.,)
        return raw

    monkeypatch.setattr(type(backend), 'run', launch)
    if shape == 'corners':
        reports = test.run_corners(['tt', 'ss'])
    elif shape in ('declared', 'all'):
        test.set_corners(['tt', 'ss'])
        reports = test.run() if shape == 'declared' else next(iter(dut.run_all_tests().values()))
    elif shape == 'conditions':
        test.set_temperatures([0, 85])
        test.add_source(sources.VoltageSource('rail', 'vdd', '0', dc=1.8))
        test.set_supply_voltages([1.8], source='rail')
        reports = test.run()
    elif shape == 'scalar':
        report = test.run()
        reports = {report.corner: report}
    else:
        from silk import Variable
        value = Variable('offset', value=.2)
        test.netlist.sig_sources_dict['vs'].dc = value
        test.set_variable_values(value, [.2, .3])
        reports = test.run()
    values = []
    retained = test.sweep_corners
    artifacts = {key: Path(path).read_bytes() for key, path in retained.items()}
    live = test.netlist.results_file, test.netlist.simResultsAll
    report_state = {key: (dict(report.metrics), report.rawfile_path, report.operating_conditions)
                    for key, report in reports.items()}
    for index, (key, report) in enumerate(reports.items(), 1):
        reader = report.sim_results
        physical = reader.load_signal('vvs__silk_pwl', 'I')
        logical = reader.load_signal('i(vs)')
        np.testing.assert_array_equal(logical.x, physical.x)
        np.testing.assert_array_equal(logical.y, physical.y)
        assert report.metrics['direct_current'] == pytest.approx(logical.y[-1])
        assert report.metrics['string_current'] == pytest.approx(logical.y[-1])
        for probe in (current, test.probe('i(vs)')):
            selected = probe.point(key)
            assert selected.bound
            np.testing.assert_array_equal(selected.x, logical.x)
            np.testing.assert_array_equal(selected.y, logical.y)
        expected = reader.load_signal('vout')
        np.testing.assert_array_equal(expected.y, np.full(expected.y.shape, float(index)))
        np.testing.assert_array_equal(logical.y, np.full(logical.y.shape, -index / 1000))
        for probe in (voltage, test.probe('vout')):
            selected = probe.point(key)
            assert selected.bound
            np.testing.assert_array_equal(selected.x, expected.x)
            np.testing.assert_array_equal(selected.y, expected.y)
        selected = expression.point(key)
        assert selected.bound
        np.testing.assert_array_equal(selected.x, expected.x)
        np.testing.assert_allclose(selected.y, expected.y * 3)
        values.append(float(logical.y[-1]))
        if shape != 'scalar':
            assert all(vec._data is None for plot in reader.simResultsAll.plots for vec in plot.vectors)
    assert len(set(values)) == (1 if shape == 'scalar' else 2)
    assert len(launches) == len(list(reports.items()))
    assert {key: (dict(report.metrics), report.rawfile_path, report.operating_conditions)
            for key, report in reports.items()} == report_state
    assert test.sweep_corners == retained
    assert test.netlist.results_file == live[0]
    assert test.netlist.simResultsAll is live[1]
    assert {key: Path(path).read_bytes() for key, path in retained.items()} == artifacts


@pytest.mark.parametrize('key', ['missing', None, ['tt'], 1, ('tt',)])
def test_point_refuses_unknown_or_unsupported_keys(tb, key):
    tb._sweep_table = {'tt': '/retained.raw'}
    with pytest.raises(UnboundSignalError) as error:
        tb.probe('vout').point(key)
    assert str(error.value) == (
        f"No retained deterministic point for key {key!r}; available keys: ['tt'].")


def test_point_empty_table_and_monte_carlo_first(tb):
    for key in ('tt', None, ('tt', 0)):
        with pytest.raises(UnboundSignalError, match=r'available keys: \[\]'):
            tb.probe('vout').point(key)
    tb._mc_samples = {('tt', 0): 'sample'}
    for key in ('tt', None, ['tt'], ('tt', 0)):
        with pytest.raises(UnboundSignalError, match='Monte Carlo.*sample'):
            tb.probe('vout').point(key)


def test_point_bound_and_expired_bench_errors(tb):
    from silk.signals import Signal
    import gc
    bound = Signal('vout', 'voltage', 'time', 'v', np.array([0., 1.]), np.array([1., 2.]))
    with pytest.raises(ValueError, match='already carries data'):
        bound.point('tt')
    other = copy.deepcopy(tb)
    probe = other.probe('vout')
    del other
    gc.collect()
    with pytest.raises(UnboundSignalError, match='No live testbench'):
        probe.point('tt')


def test_point_preserves_missing_artifact_error(reference_grid):
    from pathlib import Path
    from silk.signals.rawfile import RawfileError
    tb, _, keys = reference_grid
    Path(tb.sweep_corners[keys[0]]).unlink()
    with pytest.raises(RawfileError):
        tb.probe('vout').point(keys[0])
