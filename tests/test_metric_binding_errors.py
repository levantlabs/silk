"""A metric's Signal binding is the CALLER's business, so it raises (TASK-215.7).

Before this task, every failure inside the metric-extraction loop of
``ModuleTestbench._simulate``'s tail was caught and stored as the string
``"Error: ..."`` in the report's ``metrics`` dict. That buried two failures
that are not about the circuit at all:

* a **binding** failure — a probe naming a signal the rawfile does not
  contain, a deferred transform whose arguments were wrong, a probe whose
  bench never ran;
* a **kind** failure — an AC metric handed a transient trace, which is the
  expected diagnostic is:
  ``"Error: sig must have xlabel='frequency'"``.

Both now raise :class:`~silk.exceptions.MetricBindingError`. It is a
:class:`~silk.exceptions.ConfigurationError`, which is the one class
``Module.run_all_tests`` re-raises instead of folding into that test's
``{'error': str}`` entry (``silk/circuit/modules_testbench.py``, TASK-163),
so a mis-bound metric stops a recursive tree run rather than repeating the
identical failure once per test.

Everything else inside a metric's BODY keeps today's report-string
convention; that convention is pre-existing and deliberately out of scope,
and it is pinned here so a later change to it cannot happen silently.

The fast legs stand the simulator in with the committed TASK-74 rawfile
fixture (real ngspice-42 output) exactly as ``tests/test_testbench_evaluate.py``
does, so ``run()`` exercises its real tail on real simulator output without
launching anything. One class runs real ngspice on the RC circuit.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from tests._temperature import mark_simulated
from tests._env import needs_ngspice, needs_pdk

from silk import sources
from silk.circuit.modules import Module
from silk.circuit.ports import PortDir
from silk.elements import elements
from silk.exceptions import ConfigurationError, MetricBindingError
from silk.metrics.ac import bandwidth_3dB, dc_gain_dB, gain_bw_product
from silk.metrics.statistics import average
from silk.metrics.power import dynamic_energy_per_cycle
from silk.pdk.basepdk import BasePDK
from silk.signals.kinds import SignalKindError
from silk.simulation import analysis


FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles' / 'e_ng_tran.raw'
_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')


@pytest.fixture
def pdk():
    return BasePDK(_PDK_YAML, library_location=_SKY130_LIB)


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Builders — the e_ng_tran fixture holds v(clk), v(out), v(step), v(tone) and
# i(v1). The DUT's 'a' port is deliberately wired to a net named 'absent',
# which IS a declared net of the testbench (so tb.probe() validates it) and is
# NOT in the rawfile (so resolution fails) — the exact shape of "a probe of a
# name absent from the rawfile".
# ---------------------------------------------------------------------------

def _dut(pdk, tmp_dir, name='dut'):
    dut = Module('PDK_skywater130', tmp_dir, f'{name}.cir', pdk_details=pdk,
                 module_name=name)
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with dut:
        pdk.nmos1p8('mn', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)
    return dut


def _bench(dut, name='bind'):
    tb = dut.create_test(name)
    tb.add_module_under_test({'a': 'absent', 'b': 'out', 'vdd': 'clk',
                              'vss': '0'})
    tb.add_source(sources.VoltageSource('vsup', positive='clk', negative='0',
                                        dc=1.8))
    return tb


def _stub(tb):
    """Stand ``tb.netlist.simulate`` in for the simulator with the fixture."""
    def fake_simulate(corner=None, **_):
        tb.netlist.load_file(str(FIXTURE))
        tb.netlist.read_raw_file()
        tb.netlist.results_file = str(FIXTURE)
        mark_simulated(tb.netlist, corner, voltage=1.8, supply_source='vsup')
        return tb.netlist
    return patch.object(tb.netlist, 'simulate', side_effect=fake_simulate)


# ---------------------------------------------------------------------------
# AC #1 — a probe of a name absent from the rawfile raises from run()
# ---------------------------------------------------------------------------

class TestResolutionFailureRaises:

    def test_missing_name_raises_metric_binding_error_from_run(self, pdk, tmp_dir):
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('vavg', average, sig=tb.probe('absent'))
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        message = str(excinfo.value)
        assert "'vavg'" in message            # the metric
        assert "'absent'" in message          # the probe that failed
        assert 'sig=' in message              # the slot

    def test_it_is_a_configuration_error(self, pdk, tmp_dir):
        """The class is the whole mechanism: ConfigurationError is what
        run_all_tests lets through."""
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('vavg', average, sig=tb.probe('absent'))
        with _stub(tb), pytest.raises(ConfigurationError):
            tb.run('tt')

    def test_the_loader_error_is_kept_as_the_cause(self, pdk, tmp_dir):
        from silk.exceptions import SimulationError
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('vavg', average, sig=tb.probe('absent'))
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        assert isinstance(excinfo.value.__cause__, SimulationError)

    def test_a_missing_string_path_slot_raises_the_same_way(self, pdk, tmp_dir):
        """A ``str`` slot value goes through ckt.load_signal exactly as the
        declarative form does, and fails through the same one point."""
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('vavg', average, sig='nowhere')
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        assert "path 'nowhere'" in str(excinfo.value)

    def test_a_deferred_transforms_argument_error_raises_at_resolution(
            self, pdk, tmp_dir):
        """``window``'s ``x2 > x1`` check is deferred to resolution (design
        §4), so it surfaces here as a binding failure, not a report string."""
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('vavg', average, sig=tb.probe('out').window(5e-6, 1e-6))
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        message = str(excinfo.value)
        assert 'expression' in message
        assert 'require x2 > x1' in message

    def test_a_probe_with_no_bench_raises_a_binding_error_not_an_attribute_error(
            self, pdk, tmp_dir):
        """An UnboundSignalError during resolution is a binding failure too."""
        from silk.signals.base import Signal
        tb = _bench(_dut(pdk, tmp_dir))
        orphan = Signal._unbound('out')          # no bench attached
        tb.add_metric('vavg', average, sig=orphan)
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        from silk.exceptions import UnboundSignalError
        assert isinstance(excinfo.value.__cause__, UnboundSignalError)

    def test_a_good_metric_alongside_is_irrelevant_the_run_stops(
            self, pdk, tmp_dir):
        """The binding error is raised, so no SpecReport comes back at all —
        it is not one bad entry among good ones."""
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('good', average, sig=tb.probe('out'))
        tb.add_metric('bad', average, sig=tb.probe('absent'))
        with _stub(tb), pytest.raises(MetricBindingError):
            tb.run('tt')


# ---------------------------------------------------------------------------
# AC #1 — and it PROPAGATES through run_all_tests(recursive=True)
# ---------------------------------------------------------------------------

class TestPropagationThroughRunAllTests:

    def test_it_propagates_out_of_a_recursive_tree_run(self, pdk, tmp_dir):
        """The bad bench is on a block that also owns a submodule, so the
        recursive walk is live; the error is not folded into {'error': str}."""
        block = _dut(pdk, tmp_dir, 'block')
        leaf = _dut(pdk, tmp_dir, 'leaf')
        block.add_module(leaf, 'u2', nets={'a': 'absent', 'b': 'out',
                                            'vdd': 'clk', 'vss': '0'})
        tb = _bench(block)
        tb.add_metric('vavg', average, sig=tb.probe('absent'))
        with _stub(tb), pytest.raises(MetricBindingError):
            block.run_all_tests(recursive=True)
        leaf.close()
        block.close()

    def test_a_composed_childs_bench_propagates_through_the_recursion(
            self, pdk, tmp_dir):
        """The bench is owned by a block the PARENT embeds, so the failure
        has to travel back up through the recursive walk.

        The slot is a ``str`` path rather than a probe on purpose: until
        TASK-215.8 rebinds probes in ``__deepcopy__``, a probe on a replica
        still points at the template bench, which would fail for a second
        reason and prove the wrong thing. A path fails for exactly the
        reason under test — the name is not in the results.
        """
        template = _dut(pdk, tmp_dir, 'child')
        tb = _bench(template, 'childtest')
        tb.add_metric('vavg', average, sig='nowhere')
        root = Module('PDK_skywater130', tmp_dir, 'root.cir', pdk_details=pdk)
        root.add_module(template, 'u1', nets={'a': 'na', 'b': 'nb',
                                               'vdd': 'nv', 'vss': '0'})
        live = root.elem_dict['u1']['device']._tests['childtest']
        with _stub(live), pytest.raises(MetricBindingError):
            root.run_all_tests(recursive=True)
        template.close()
        root.close()

    def test_an_ordinary_metric_failure_is_still_buried_per_test(
            self, pdk, tmp_dir):
        """Today's convention, pinned: an extractor that blows up for a
        reason that is NOT a binding failure still lands as {'error': str}
        and the tree run continues."""
        block = _dut(pdk, tmp_dir, 'block')
        tb = _bench(block)

        def explode(ckt):
            raise RuntimeError('boom')

        tb.add_metric('kaboom', explode)
        with _stub(tb):
            results = block.run_all_tests(recursive=True)
        report = results['block.bind']
        assert report.metrics['kaboom'].startswith('Error: boom')


# ---------------------------------------------------------------------------
# AC #2 — the kind guard is promoted, in BOTH bound-by-name forms
# ---------------------------------------------------------------------------

class TestKindGuardPromotion:
    """``dc_gain_dB`` on a transient trace.

    BEFORE this task (measured on main @4204130, both forms): the run
    completed and ``report.metrics['gain']`` was the string
    ``"Error: sig must have xlabel='frequency', got 'time'"``.
    AFTER: both forms raise ``MetricBindingError``. The plain-callable
    ``extractor(ckt)`` form is deliberately unchanged.
    """

    @pytest.mark.parametrize('metric', [dc_gain_dB, bandwidth_3dB, gain_bw_product])
    def test_function_form_raises(self, pdk, tmp_dir, metric):
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('gain', metric, sig=tb.probe('out'))
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        message = str(excinfo.value)
        assert "'gain'" in message
        assert metric.__name__ in message
        assert "xlabel='frequency'" in message
        assert type(excinfo.value.__cause__) is SignalKindError
        assert str(excinfo.value.__cause__) == (
            f"{metric.__name__}: sig must have xlabel='frequency', got 'time'"
        )

    def test_declarative_type_form_raises_too(self, pdk, tmp_dir):
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('gain', type='dc_gain_dB', signal='out')
        with _stub(tb), pytest.raises(MetricBindingError) as excinfo:
            tb.run('tt')
        message = str(excinfo.value)
        assert "'gain'" in message
        assert "type='dc_gain_dB'" in message
        assert type(excinfo.value.__cause__) is SignalKindError
        assert str(excinfo.value.__cause__) == (
            "dc_gain_dB: sig must have xlabel='frequency', got 'time'"
        )

    def test_the_kind_error_is_still_a_value_error(self, pdk, tmp_dir):
        """SignalKindError stays a ValueError subclass, so every existing
        ``except ValueError`` around a metric call is unaffected."""
        sig_time = _bench(_dut(pdk, tmp_dir))
        del sig_time
        with pytest.raises(ValueError):
            raise SignalKindError('x')

    def test_plain_callable_form_still_lands_as_a_report_string(
            self, pdk, tmp_dir):
        """The third form is untouched: it hands the loop an opaque
        callable, and nothing about its failure says 'binding'."""
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('gain', lambda ckt: dc_gain_dB(ckt.load_signal('out')))
        with _stub(tb):
            report = tb.run('tt')
        assert report.metrics['gain'].startswith(
            "Error: dc_gain_dB: sig must have xlabel='frequency'")

    def test_a_body_error_that_is_not_a_kind_error_is_a_report_string(
            self, pdk, tmp_dir):
        """Pinned convention: only the kind guard is promoted. Any other
        failure INSIDE the metric body still becomes 'Error: ...'."""
        tb = _bench(_dut(pdk, tmp_dir))
        tb.add_metric('e_cycle', dynamic_energy_per_cycle,
                      sig_v=tb.probe('out'), sig_i=tb.probe('i(v1)'),
                      period=1.0)          # longer than the whole trace
        with _stub(tb):
            report = tb.run('tt')
        assert isinstance(report.metrics['e_cycle'], str)
        assert report.metrics['e_cycle'].startswith('Error: ')


# ---------------------------------------------------------------------------
# AC #2 — one bench per recipe, refused at declaration (design §4)
# ---------------------------------------------------------------------------

class TestCrossBenchRefusal:

    def test_a_foreign_probe_is_refused_naming_both_benches(self, pdk, tmp_dir):
        dut = _dut(pdk, tmp_dir)
        here = _bench(dut, 'here')
        there = _bench(dut, 'there')
        with pytest.raises(TypeError) as excinfo:
            here.add_metric('vavg', average, sig=there.probe('out'))
        message = str(excinfo.value)
        assert "'here'" in message and "'there'" in message
        assert "'dut'" in message
        assert "'out'" in message

    def test_a_foreign_probe_inside_an_expression_is_refused(self, pdk, tmp_dir):
        dut = _dut(pdk, tmp_dir)
        here = _bench(dut, 'here')
        there = _bench(dut, 'there')
        mixed = (here.probe('out') + there.probe('clk')) / 2
        with pytest.raises(TypeError) as excinfo:
            here.add_metric('vcm', average, sig=mixed)
        message = str(excinfo.value)
        assert "'here'" in message and "'there'" in message
        assert "'clk'" in message          # the offending leaf, not the root

    def test_a_foreign_probe_inside_a_transform_is_refused(self, pdk, tmp_dir):
        dut = _dut(pdk, tmp_dir)
        here = _bench(dut, 'here')
        there = _bench(dut, 'there')
        with pytest.raises(TypeError) as excinfo:
            here.add_metric('slew', average,
                            sig=there.probe('out').derivative())
        assert "'here'" in message_of(excinfo) and "'there'" in message_of(excinfo)

    def test_a_foreign_probe_in_a_second_slot_is_refused(self, pdk, tmp_dir):
        from silk.metrics.power import average_power
        dut = _dut(pdk, tmp_dir)
        here = _bench(dut, 'here')
        there = _bench(dut, 'there')
        with pytest.raises(TypeError) as excinfo:
            here.add_metric('p', average_power, sig_v=here.probe('out'),
                            sig_i=there.probe('i(v1)'))
        assert 'sig_i=' in message_of(excinfo)

    def test_this_benchs_own_probes_are_accepted(self, pdk, tmp_dir):
        dut = _dut(pdk, tmp_dir)
        here = _bench(dut, 'here')
        here.add_metric('vcm', average,
                        sig=(here.probe('out') + here.probe('clk')) / 2)
        assert here._metrics[-1]['name'] == 'vcm'

    def test_a_bound_signal_is_not_a_bench_question(self, pdk, tmp_dir):
        """A Signal that already carries data belongs to no bench, so it is
        never refused — and it is not walked for leaves either."""
        from silk.signals.base import Signal
        dut = _dut(pdk, tmp_dir)
        here = _bench(dut, 'here')
        bound = Signal('v', 'voltage', 'time', 'v',
                       np.linspace(0, 1, 5), np.ones(5))
        here.add_metric('vavg', average, sig=bound)
        assert here._metrics[-1]['slots'] == {'sig': bound}


def message_of(excinfo):
    return str(excinfo.value)


# ---------------------------------------------------------------------------
# Real ngspice — the same two failures on a circuit that actually simulated
# ---------------------------------------------------------------------------

@needs_ngspice
@needs_pdk
@pytest.mark.slow
class TestAgainstRealNgspice:
    """A 1 kOhm / 1 nF RC low-pass, one transient, driven by a vpulse —
    the same circuit TASK-215.5 and 215.6 used."""

    @staticmethod
    def _rc(pdk, tmp_dir, metric_setup):
        dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=pdk,
                     module_name='rc')
        dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
        with dut:
            elements.resistor('r1', p='a', n='b', value=1e3)
            elements.capacitor('c1', p='b', n='0', value=1e-9)
        tb = dut.create_test('rc')
        tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
        tb.add_source(sources.VoltageSource(
            'vs', positive='vin', negative='0', dc=0.0,
            pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                                pw=5e-6, per=10e-6)))
        tb.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))
        metric_setup(tb)
        return tb

    def test_a_probe_absent_from_the_rawfile_raises_from_a_real_run(
            self, pdk, tmp_dir):
        def setup(tb):
            # 'vin'/'vout' are in the rawfile; this made-up path is not, and
            # it is not a plain net name so probe() does not pre-validate it.
            tb.add_metric('vavg', average, sig=tb.probe('xdut.nowhere'))
        tb = self._rc(pdk, tmp_dir, setup)
        with pytest.raises(MetricBindingError) as excinfo:
            tb.run()
        assert "'vavg'" in str(excinfo.value)

    def test_dc_gain_on_a_transient_probe_raises_from_a_real_run(
            self, pdk, tmp_dir):
        def setup(tb):
            tb.add_metric('gain', dc_gain_dB, sig=tb.probe('vout'))
        tb = self._rc(pdk, tmp_dir, setup)
        with pytest.raises(MetricBindingError) as excinfo:
            tb.run()
        assert isinstance(excinfo.value.__cause__, SignalKindError)

    def test_a_correctly_bound_metric_on_the_same_circuit_still_runs(
            self, pdk, tmp_dir):
        def setup(tb):
            tb.add_metric('vavg', average, sig=tb.probe('vout'))
        tb = self._rc(pdk, tmp_dir, setup)
        report = tb.run()
        expected = average(tb.netlist.load_signal('vout'))
        assert report.metrics['vavg'] == pytest.approx(expected, abs=1e-12)
        # And the number is the circuit's, not an artefact: the RC follows a
        # 50 %-duty 1 V pulse, so its mean sits near half a volt.
        assert 0.3 < report.metrics['vavg'] < 0.7
        assert tb.netlist.load_signal('vout').y.max() == pytest.approx(1.0, abs=0.02)
