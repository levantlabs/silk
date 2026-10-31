"""``ModuleTestbench.plot()`` — probes and derived Signals, after a run.

``tb.plot(*signals)`` draws probes and Signals derived from them by
delegating to ``Module.plot``'s auto-scaled multi-panel/overlay layout
(``silk/circuit/modules_results.py:566`` -> the shared
``_plot_loaded_signals`` helper this task factored out of it), reusing
``Signal.resolve()`` (TASK-215.4) so each argument is fetched exactly once
instead of re-triggering the unbound ``__getattr__`` fallback on every
attribute read. With no arguments it draws every probe ``tb.probe()`` has
minted on that bench, via the lightweight weak-reference registry ``probe()``
appends to (``ModuleTestbench._probe_registry`` / ``_live_probes()``).

Three things are proved here, on a real one-transient RC low-pass (the same
circuit ``test_probe.py``'s real-ngspice legs use):

**The figures are real.** ``tb.plot(vp, vm, vcm)`` and ``tb.plot()`` both
produce a ``matplotlib.figure.Figure`` with one axis per signal (or one axis
total under ``overlay=True``), each carrying real plotted data.

**Each argument is resolved exactly once.** Patching ``Signal.resolve`` to
count calls, ``tb.plot(...)``'s total call multiset matches calling
``.resolve()`` once, directly, on each argument — no extra traffic beyond
what ``resolve()``'s own (already-tested) recursion costs on a derived
Signal.

**The error paths are clear and come first.** Before the bench has run;
with no arguments and no probes ever created; with a non-``Signal`` argument.
"""
from __future__ import annotations

import os
from unittest import mock

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import UnboundSignalError
from silk.metrics.statistics import average
from silk.signals.base import Signal
from silk.simulation import analysis

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)


def _rc_test(tmp_dir):
    """A wired (unsimulated) RC low-pass testbench: one resistor, one cap."""
    from silk.pdk.basepdk import BasePDK
    real_pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=real_pdk,
                 module_name='rc')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    test = dut.create_test('rc')
    test.add_module_under_test({'a': 'vin', 'b': 'vout'})
    test.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                            pw=5e-6, per=10e-6)))
    test.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))
    return test


def _rc_ac_test(tmp_dir):
    """The same RC low-pass, wired for an AC sweep instead of a transient --
    its probe resolves to a COMPLEX Signal (TASK-230)."""
    from silk.pdk.basepdk import BasePDK
    real_pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
    dut = Module('PDK_skywater130', tmp_dir, 'rc_ac.cir', pdk_details=real_pdk,
                 module_name='rc_ac')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    test = dut.create_test('rc_ac')
    test.add_module_under_test({'a': 'vin', 'b': 'vout'})
    test.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0, ac=1.0))
    test.add_sim_type(analysis.AC(variation='dec', points=10, fstart=1,
                                  fstop=1e9))
    return test


# ---------------------------------------------------------------------------
# Error paths — no simulation needed
# ---------------------------------------------------------------------------

class TestErrorPaths:
    @pytest.fixture
    def wired_test(self, pdk, tmp_dir):
        """A wired-but-never-run testbench (the fast stub PDK is enough)."""
        dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                     module_name='dut')
        dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                                 'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
        with dut:
            pdk.nmos1p8('m1', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)
        test = dut.create_test('t')
        test.add_module_under_test({'a': 'vin', 'b': 'vout', 'vdd': 'vdd',
                                    'vss': '0'})
        test.add_source(sources.VoltageSource('vs', positive='vin',
                                               negative='0', dc=0.9))
        test.add_source(sources.VoltageSource('vdd', positive='vdd',
                                               negative='0', dc=1.8))
        test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        return test

    def test_before_run_raises_clearly(self, wired_test):
        vout = wired_test.probe('vout')
        with pytest.raises(UnboundSignalError, match='has not run yet'):
            wired_test.plot(vout, show=False)

    def test_before_run_with_no_arguments_also_raises(self, wired_test):
        wired_test.probe('vout')
        with pytest.raises(UnboundSignalError, match='has not run yet'):
            wired_test.plot(show=False)

    def test_no_netlist_at_all_raises(self, pdk, tmp_dir):
        dut = Module('PDK_skywater130', tmp_dir, 'dut2.cir', pdk_details=pdk,
                     module_name='dut2')
        test = dut.create_test('never_wired')
        with pytest.raises(UnboundSignalError, match='has not run yet'):
            test.plot(show=False)


# ---------------------------------------------------------------------------
# Real ngspice — figures, resolution count, and the no-argument registry
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestAgainstRealNgspice:
    @pytest.fixture
    def ran(self, tmp_path):
        test = _rc_test(str(tmp_path))
        vp = test.probe('vout')
        vm = test.probe('vin')
        test.run()
        return test, vp, vm

    def test_multi_signal_plot_has_one_panel_per_signal(self, ran):
        test, vp, vm = ran
        vcm = (vp + vm) / 2
        fig = test.plot(vp, vm, vcm, show=False)
        assert len(fig.axes) == 3
        for ax in fig.axes:
            assert len(ax.lines) == 1
            assert len(ax.lines[0].get_xdata()) > 0

    def test_overlay_puts_every_signal_on_one_axis(self, ran):
        test, vp, vm = ran
        vcm = (vp + vm) / 2
        fig = test.plot(vp, vm, vcm, overlay=True, show=False)
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 3

    def test_no_arguments_draws_every_probe_of_the_bench(self, ran):
        """Only ``vp``/``vm`` were minted via ``probe()`` on this bench; a
        third Signal built by arithmetic (never itself passed to ``probe()``)
        is not itself in the registry and so is not drawn by default."""
        test, vp, vm = ran
        _ = (vp + vm) / 2  # not a probe of its own; not counted below
        fig = test.plot(show=False)
        assert len(fig.axes) == 2

    def test_non_signal_argument_raises_type_error(self, ran):
        test, _, _ = ran
        with pytest.raises(TypeError, match='Signal'):
            test.plot('vout', show=False)

    def test_the_rc_data_is_real(self, ran):
        """The plotted line carries the same (auto-scaled) sample count as
        the probe's own resolved data; the RC output actually charged."""
        test, vp, vm = ran
        fig = test.plot(vp, show=False)
        y = fig.axes[0].lines[0].get_ydata()
        assert len(y) == len(vp.y)
        assert float(np.max(vp.y)) == pytest.approx(1.0, abs=0.02)

    def test_no_probes_created_raises_value_error(self, tmp_path):
        test = _rc_test(str(tmp_path))
        test.add_metric('v_avg', average, sig='vout')  # declarative — no probe()
        test.run()
        with pytest.raises(ValueError, match='no probes'):
            test.plot(show=False)

    def test_each_argument_is_resolved_exactly_once(self, ran):
        """``tb.plot(...)``'s total ``resolve()`` traffic matches calling
        ``.resolve()`` once, directly, on each argument — the reference
        pattern a caller who wanted the same data by hand would produce.
        Composite Signals (``vcm``) legitimately resolve their own leaves as
        part of that single call; this asserts ``plot()`` adds nothing on
        top of that, not that shared leaves are resolved only once globally.
        """
        test, vp, vm = ran
        vcm = (vp + vm) / 2

        calls = []
        orig_resolve = Signal.resolve

        # The double mirrors resolve()'s real signature -- analysis=
        # (TASK-212) and corner=/_readers= (TASK-218) included: plot()'s
        # recursion through a recipe passes every one of those keywords down,
        # so a two-argument stand-in would fail on the composite.
        def counting_resolve(self, module=None, *, analysis=None,
                             corner=None, _readers=None):
            calls.append(id(self))
            return orig_resolve(self, module, analysis=analysis,
                                corner=corner, _readers=_readers)

        with mock.patch.object(Signal, 'resolve', counting_resolve):
            test.plot(vp, vm, vcm, show=False)
        plot_calls = sorted(calls)

        calls.clear()
        with mock.patch.object(Signal, 'resolve', counting_resolve):
            vp.resolve()
            vm.resolve()
            vcm.resolve()
        manual_calls = sorted(calls)

        assert plot_calls == manual_calls

    def test_leaf_probe_resolved_exactly_once(self, ran):
        test, vp, _ = ran
        calls = []
        orig_resolve = Signal.resolve

        def counting_resolve(self, module=None, *, analysis=None,
                             corner=None, _readers=None):
            if self is vp:
                calls.append(1)
            return orig_resolve(self, module, analysis=analysis,
                                corner=corner, _readers=_readers)

        with mock.patch.object(Signal, 'resolve', counting_resolve):
            test.plot(vp, show=False)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# TASK-230, finding 1 -- a complex (AC) probe draws dB vs log-f, not a
# ComplexWarning cast to real.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestComplexProbePlot:
    @pytest.fixture
    def ran_ac(self, tmp_path):
        test = _rc_ac_test(str(tmp_path))
        vout = test.probe('vout')
        test.run()
        return test, vout

    def test_ac_probe_draws_db_vs_log_f_with_no_complex_warning(self, ran_ac):
        import warnings
        import numpy.exceptions

        test, vout = ran_ac
        with warnings.catch_warnings():
            warnings.simplefilter('error', numpy.exceptions.ComplexWarning)
            fig = test.plot(vout, show=False)
        try:
            ax = fig.axes[0]
            assert ax.get_xscale() == 'log'
            assert ax.get_ylabel() == 'Magnitude (dB)'
            assert len(ax.lines) == 1
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_ac_probe_panel_is_identical_to_signal_plot(self, ran_ac):
        """Same figure, drawn the other way: ``sig.resolve().plot()``
        directly. tb.plot()'s panel for a complex probe must match it
        exactly -- same x/y data, same axis labels, same log scale."""
        import matplotlib.pyplot as plt

        test, vout = ran_ac
        fig = test.plot(vout, show=False)
        ax = fig.axes[0]
        line = ax.lines[0]

        ref_fig, ref_ax = plt.subplots()
        vout.resolve().plot(ax=ref_ax)
        ref_line = ref_ax.lines[0]

        np.testing.assert_array_equal(line.get_xdata(), ref_line.get_xdata())
        np.testing.assert_array_equal(line.get_ydata(), ref_line.get_ydata())
        assert ax.get_xlabel() == ref_ax.get_xlabel()
        assert ax.get_ylabel() == ref_ax.get_ylabel()
        assert ax.get_xscale() == ref_ax.get_xscale()
        plt.close(fig)
        plt.close(ref_fig)

    def test_a_real_and_a_complex_probe_together_each_get_their_own_treatment(
            self, tmp_path):
        """A transient probe and an AC probe share one ``tb.plot()`` call
        (two panels, different analyses -- TASK-212). The real panel keeps
        its auto-scaled SI label; the complex one gets the dB/log-f
        treatment; neither disturbs the other's axis labels."""
        import warnings
        import numpy.exceptions
        import matplotlib.pyplot as plt

        test = _rc_ac_test(str(tmp_path))
        test.add_sim_type(analysis.Transient(step=1e-8, stop=1e-6))
        vout_t = test.probe('vout', analysis='tran')
        vout_ac = test.probe('vout', analysis='ac')
        test.run()

        with warnings.catch_warnings():
            warnings.simplefilter('error', numpy.exceptions.ComplexWarning)
            fig = test.plot(vout_t, vout_ac, show=False)
        try:
            real_ax, ac_ax = fig.axes
            assert ac_ax.get_xscale() == 'log'
            assert ac_ax.get_ylabel() == 'Magnitude (dB)'
            assert real_ax.get_xscale() == 'linear'
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# TASK-230, finding 2 -- an inline probe (no variable holding it) survives
# to a later no-argument tb.plot(), because the registry now holds probes
# strongly.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestInlineProbeRegistry:
    def test_inline_probe_with_no_kept_reference_survives_to_plot(
            self, tmp_path):
        """``tb.probe('vout')`` minted as a bare statement -- the ONLY
        reference is the bench's own registry. Before TASK-230 this probe
        was collected before ``tb.plot()`` ever ran (the registry held only
        a weakref) and the bench reported it had no probes."""
        import gc

        test = _rc_test(str(tmp_path))
        test.probe('vout')  # inline; no local variable holds it
        gc.collect()        # force collection of anything not strongly held
        test.run()

        fig = test.plot(show=False)
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 1
        assert len(fig.axes[0].lines[0].get_xdata()) > 0

    def test_two_inline_probes_both_survive(self, tmp_path):
        import gc

        test = _rc_test(str(tmp_path))
        test.probe('vout')
        test.probe('vin')
        gc.collect()
        test.run()

        fig = test.plot(show=False)
        assert len(fig.axes) == 2
