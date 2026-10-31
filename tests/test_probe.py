"""``ModuleTestbench.probe()`` — naming a signal before the run (TASK-215.5).

``probe()`` is the public
door onto the unbound Signal state TASK-215.2 built: it returns a Signal that
holds a name, a loader selector and a **weak** reference to its testbench, and
nothing else.

Four things are proved here.

**The weak reference.** A probe's ``_bench`` is a ``weakref.ref`` — asserted by
type, not by behaviour — and no bench object appears anywhere in its
``__dict__``. That is what keeps a bench's metric list from dragging a bench
graph into a deep copy (TASK-205).

**What is checked, and what deliberately is not.** A plain net name is checked
against the testbench's own nets at the call, naming the net and the
alternatives. A hierarchical path and a current are not: whether
``xdut.mid`` reaches the rawfile is a property of the run, not of the netlist.

**The spellings agree with the loader.** ``'out'`` and ``'v(out)'`` are one
probe; ``'i(vs)'`` is the branch current of source ``vs``, with the
emitted-name mapping (``Vvs`` → ``vvs``) applied for the user, so it is the
same probe as the raw ``probe('vvs', 'I')``; ``'xdut.m1:d'`` is a device
terminal current through the 0 V ammeter wrappers, refused at declaration when
the three opt-ins that create that vector are not set. Two slow legs run real
ngspice and assert each of these against ``load_signal`` sample for sample.

**Nothing is stored and ignored.** ``analysis=`` was refused outright until
TASK-212 landed; now the word is stored and read, and only an unknown word is
refused at the call. Whether the test DECLARES that analysis is a run-time
fact, checked when the metric is evaluated — see
``tests/test_multi_analysis_plots.py`` §3c.
"""
from __future__ import annotations

import os
import weakref

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import UnboundSignalError
from silk.signals.base import Signal
from silk.simulation import analysis

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)


# ---------------------------------------------------------------------------
# Fast fixtures — a real testbench that is never simulated
# ---------------------------------------------------------------------------

@pytest.fixture
def tb(pdk, tmp_dir):
    """A wired testbench: an nmos DUT, two sources, one transient.

    Never simulated. Its nets are what ``probe()`` validates against, and its
    sources are what the ``i(…)`` mapping reads.
    """
    dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                 module_name='dut')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT,
                             'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
    with dut:
        pdk.nmos1p8('m1', d='b', g='a', s='vss', b='vss', w=1.0, l=0.15)

    test = dut.create_test('t')
    test.add_module_under_test({'a': 'vin', 'b': 'vout', 'vdd': 'vdd',
                                'vss': '0'})
    test.add_source(sources.VoltageSource('vs', positive='vin', negative='0',
                                          dc=0.9))
    test.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                          dc=1.8))
    test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
    return test


@pytest.fixture
def ammeter_tb(tb):
    """The same bench with the three ammeter-wrapper opt-ins set."""
    tb.module.save_device_currents = True
    tb.module.terminal_current_method = 'ammeter_wrapper'
    return tb


# ---------------------------------------------------------------------------
# What a probe IS: a name and a weak reference, nothing else
# ---------------------------------------------------------------------------

class TestTheProbeObject:
    def test_probe_returns_an_unbound_signal(self, tb):
        vout = tb.probe('vout')
        assert isinstance(vout, Signal)
        assert vout.name == 'vout'
        assert vout.type == 'voltage'
        assert vout._selector == 'V'
        assert vout._recipe is None
        assert vout.bound is False

    def test_probe_holds_a_weakref_not_the_bench(self, tb):
        """Asserted by TYPE: ``weakref.ref``, and no bench anywhere in the
        probe's state. A strong reference here would put a whole design graph
        behind every metric (TASK-205)."""
        vout = tb.probe('vout')
        assert type(vout.__dict__['_bench']) is weakref.ref
        assert vout.__dict__['_bench']() is tb
        assert not any(isinstance(v, type(tb)) or isinstance(v, Module)
                       for v in vout.__dict__.values())

    def test_probe_stores_no_samples(self, tb):
        vout = tb.probe('vout')
        assert 'y' not in vout.__dict__ and 'x' not in vout.__dict__
        assert 'xlabel' not in vout.__dict__ and 'ylabel' not in vout.__dict__

    def test_reading_y_before_the_run_raises_naming_probe_and_bench(self, tb):
        vout = tb.probe('vout')
        with pytest.raises(UnboundSignalError) as exc:
            vout.y
        assert 'vout' in str(exc.value) and "'t'" in str(exc.value)
        assert not isinstance(exc.value, AttributeError)

    def test_the_bench_does_not_keep_a_list_of_probes(self, tb):
        """``probe()`` is a pure factory in this slice — ``tb.plot()`` with no
        arguments is TASK-215.9's, and the bench gains its registry there."""
        before = dict(tb.__dict__)
        tb.probe('vout')
        assert dict(tb.__dict__) == before


# ---------------------------------------------------------------------------
# Net validation — and the three spellings that deliberately defer
# ---------------------------------------------------------------------------

class TestNetValidation:
    def test_misspelled_net_raises_naming_the_net_and_the_available_nets(self, tb):
        with pytest.raises(ValueError) as exc:
            tb.probe('vuot')
        message = str(exc.value)
        assert "'vuot'" in message
        for net in ('vin', 'vout', 'vdd'):
            assert repr(net) in message

    def test_every_real_net_is_accepted(self, tb):
        for net in tb.netlist.net_dict:
            assert tb.probe(net).name == net

    def test_validation_is_case_insensitive_like_spice(self, tb):
        assert tb.probe('VOUT').name == 'VOUT'

    def test_hierarchical_path_defers(self, tb):
        """``xdut.mid`` is not a net of the testbench and never will be —
        whether it reaches the rawfile is what ``.save all`` decides."""
        mid = tb.probe('xdut.mid')
        assert mid.name == 'xdut.mid' and mid._selector == 'V'
        assert mid.bound is False

    def test_a_current_is_not_validated_as_a_net(self, tb):
        assert tb.probe('i(vs)').bound is False

    def test_raw_current_vector_name_is_not_validated_as_a_net(self, tb):
        """``probe('vvs', 'I')`` names a source branch, not a net: the plain
        rule must not fire on it."""
        assert tb.probe('vvs', 'I').name == 'vvs'

    def test_no_netlist_yet_means_nothing_to_validate_against(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'd.cir', pdk_details=pdk,
                     module_name='d')
        bare = mod.create_test('bare')
        assert bare.netlist is None
        assert bare.probe('anything').name == 'anything'


# ---------------------------------------------------------------------------
# Spellings
# ---------------------------------------------------------------------------

class TestSpellings:
    def test_out_and_v_out_are_the_same_probe(self, tb):
        plain, wrapped = tb.probe('vout'), tb.probe('v(vout)')
        assert (plain.name, plain._selector) == (wrapped.name,
                                                 wrapped._selector)
        assert wrapped.type == 'voltage'

    def test_v_wrapper_is_validated_like_the_plain_name(self, tb):
        with pytest.raises(ValueError) as exc:
            tb.probe('v(vuot)')
        assert "'vuot'" in str(exc.value)

    def test_i_of_a_source_applies_the_emitted_name_mapping(self, tb):
        """``VoltageSource('vs', …)`` is emitted as ``Vvs``; the loader wants
        the inner name ``vvs``."""
        iprobe = tb.probe('i(vs)')
        assert iprobe.name == 'Vvs'
        assert iprobe._selector == 'I' and iprobe.type == 'current'

    def test_i_of_a_source_equals_the_raw_selector_form(self, tb):
        assert tb.probe('i(vs)').name.lower() == tb.probe('vvs', 'I').name.lower()

    def test_i_of_an_unknown_name_passes_through_verbatim(self, tb):
        """That pass-through is what makes ``probe('i(vvs)')`` — the raw
        vector spelling — work, and it is why currents are not validated."""
        assert tb.probe('i(vvs)').name == 'vvs'
        assert tb.probe('i(xdut.whatever)').name == 'xdut.whatever'

    def test_current_source_maps_to_its_own_letter(self, tb):
        tb.add_source(sources.CurrentSource('ibias', positive='vout',
                                            negative='0', dc=1e-6))
        assert tb.probe('i(ibias)').name == 'Iibias'

    def test_selector_argument_still_works_as_the_low_level_form(self, tb):
        raw = tb.probe('v.xdut.xmm1.vamm_d', 'I')
        assert raw.name == 'v.xdut.xmm1.vamm_d'
        assert raw._selector == 'I' and raw.type == 'current'

    def test_lowercase_selector_is_accepted(self, tb):
        assert tb.probe('vvs', 'i')._selector == 'I'

    def test_a_bad_selector_is_refused(self, tb):
        with pytest.raises(ValueError) as exc:
            tb.probe('vout', 'Q')
        assert "'Q'" in str(exc.value)

    def test_v_wrapper_with_the_current_selector_is_a_contradiction(self, tb):
        with pytest.raises(ValueError) as exc:
            tb.probe('v(vout)', 'I')
        assert 'contradict' in str(exc.value)

    def test_a_non_string_name_is_refused(self, tb):
        with pytest.raises(TypeError):
            tb.probe(None)
        with pytest.raises(TypeError):
            tb.probe('   ')


# ---------------------------------------------------------------------------
# Device terminal currents
# ---------------------------------------------------------------------------

class TestDeviceTerminalCurrents:
    def test_terminal_probe_builds_the_ammeter_vector(self, ammeter_tb):
        """The device line is emitted ``XMm1`` inside instance ``xdut``, so
        the ammeter branch is ``v.xdut.xmm1.vamm_d`` — the same string
        ``terminal_current_vector`` looks for."""
        probe = ammeter_tb.probe('xdut.m1:d')
        assert probe.name == 'v.xdut.xmm1.vamm_d'
        assert probe._selector == 'I' and probe.type == 'current'

    def test_the_built_name_matches_the_backend_construction(self, ammeter_tb):
        from silk.simulation.backend import NgspiceBackend
        name = ammeter_tb.probe('xdut.m1:d').name

        class _Vectors:
            nvars = 1
            vectors = [type('v', (), {'name': f'i({name})'})]

        raw = type('raw', (), {'native': _Vectors()})
        vector = NgspiceBackend().terminal_current_vector(
            raw, 'xdut', 'XMm1', 'd')
        assert vector == f'i({name})'

    @pytest.mark.parametrize('port', ['d', 'g', 's', 'b'])
    def test_every_terminal_resolves(self, ammeter_tb, port):
        assert ammeter_tb.probe(f'xdut.m1:{port}').name.endswith(
            f'.vamm_{port}')

    def test_device_name_is_matched_case_insensitively(self, ammeter_tb):
        assert ammeter_tb.probe('XDUT.M1:D').name == 'v.xdut.xmm1.vamm_d'

    def test_refused_without_save_device_currents(self, tb):
        tb.module.terminal_current_method = 'ammeter_wrapper'
        with pytest.raises(ValueError) as exc:
            tb.probe('xdut.m1:d')
        assert 'save_device_currents' in str(exc.value)

    def test_refused_without_the_ammeter_method(self, tb):
        tb.module.save_device_currents = True
        with pytest.raises(ValueError) as exc:
            tb.probe('xdut.m1:d')
        assert 'ammeter_wrapper' in str(exc.value)

    def test_refused_without_a_transient_analysis(self, pdk, tmp_dir):
        dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                     module_name='dut')
        dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
        with dut:
            pdk.nmos1p8('m1', d='b', g='a', s='0', b='0', w=1.0, l=0.15)
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        test = dut.create_test('ac_only')
        test.add_module_under_test({'a': 'vin', 'b': 'vout'})
        test.add_sim_type(analysis.AC(variation='dec', points=10, fstart=1,
                                      fstop=1e6))
        with pytest.raises(ValueError) as exc:
            test.probe('xdut.m1:d')
        message = str(exc.value)
        assert 'Transient' in message and "'ac'" in message

    def test_the_refusal_names_save_device_currents_and_the_transient(self, tb):
        with pytest.raises(ValueError) as exc:
            tb.probe('xdut.m1:d')
        message = str(exc.value)
        assert 'save_device_currents' in message
        assert 'Transient' in message

    def test_an_unknown_terminal_is_refused(self, ammeter_tb):
        with pytest.raises(ValueError) as exc:
            ammeter_tb.probe('xdut.m1:z')
        assert "'d'" in str(exc.value)

    def test_an_unknown_device_is_refused_naming_the_devices(self, ammeter_tb):
        with pytest.raises(ValueError) as exc:
            ammeter_tb.probe('xdut.m9:d')
        message = str(exc.value)
        assert "'m9'" in message and "'m1'" in message

    def test_an_unknown_instance_path_is_refused(self, ammeter_tb):
        with pytest.raises(ValueError) as exc:
            ammeter_tb.probe('xnope.m1:d')
        assert "'xnope'" in str(exc.value)

    def test_a_top_level_device_needs_no_path(self, pdk, tmp_dir):
        """``add(pdk.nmos1p8(...))`` on the testbench itself: no instance
        path, so the vector is ``v.xmmtop.vamm_d``."""
        dut = Module('PDK_skywater130', tmp_dir, 'dut.cir', pdk_details=pdk,
                     module_name='dut')
        dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        test = dut.create_test('top')
        test.add_module_under_test({'a': 'vin', 'b': 'vout'})
        test.netlist.add(pdk.nmos1p8('mtop', d='vout', g='vin', s='0', b='0',
                                     w=1.0, l=0.15))
        test.add_sim_type(analysis.Transient(step=1e-9, stop=1e-8))
        assert test.probe('mtop:d').name == 'v.xmmtop.vamm_d'


# ---------------------------------------------------------------------------
# analysis= is refused, never stored and ignored
# ---------------------------------------------------------------------------

class TestAnalysisRefusal:
    """``analysis=`` is a real field since TASK-212 landed.

    Before the integration it raised ``TypeError('not supported until
    TASK-212')`` — a refusal whose whole purpose was that nothing should
    silently guess a plot before there was a plot to choose. Now the word is
    stored and read; only an unknown word is refused, and only the WORD is
    checked here, because whether the test declares that analysis is settled
    at run time (a hand-built test may add its sim types after its probes).
    """

    @pytest.mark.parametrize('value', ['tran', 'ac', 'dc', 'noise', 'op'])
    def test_each_silk_analysis_is_stored(self, tb, value):
        # 'op' (TASK-330.2) is a real, selectable silk analysis word here —
        # module_test._SIM_KEY_ANALYSIS carries it — even though it is
        # deliberately absent from SILK_ANALYSES, the published
        # metric-schema vocabulary (see tests/test_signal_kinds.py).
        assert tb.probe('vout', analysis=value)._analysis == value

    @pytest.mark.parametrize('value', ['', 'transient', 'AC', 'tran1'])
    def test_an_unknown_word_is_refused_naming_the_five(self, tb, value):
        with pytest.raises(ValueError) as exc:
            tb.probe('vout', analysis=value)
        message = str(exc.value)
        assert 'not a silk analysis' in message
        for word in ("'ac'", "'dc'", "'noise'", "'op'", "'tran'"):
            assert word in message

    def test_none_is_the_accepted_value(self, tb):
        assert tb.probe('vout', analysis=None)._analysis is None

    def test_the_word_check_comes_before_every_other_check(self, tb):
        """A misspelled net AND a misspelled analysis: the analysis refusal
        wins, so a script that will have to change anyway is told the deeper
        thing."""
        with pytest.raises(ValueError) as exc:
            tb.probe('vuot', analysis='transient')
        assert 'not a silk analysis' in str(exc.value)

    def test_an_undeclared_analysis_is_not_checked_at_the_call(self, tb):
        """The bench declares only a Transient; naming 'ac' here does NOT raise.

        Whether the test declares an analysis is a run-time fact — a
        hand-built test may call add_sim_type after its probes — so it is
        checked when the metric is evaluated, not at probe().
        """
        assert tb.probe('vout', analysis='ac')._analysis == 'ac'


# ---------------------------------------------------------------------------
# Real ngspice — the RC circuit of TASK-212's notes
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestAgainstRealNgspice:
    """One transient run on a 1 kΩ / 1 nF RC low-pass driven by a vpulse.

    Every probe is cross-checked against ``netlist.load_signal`` sample for
    sample: that is what makes a wrong name or a wrong selector visible,
    rather than a plausible-looking trace from the wrong vector.
    """

    @pytest.fixture(scope='class')
    def ran(self, tmp_path_factory):
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
        tmp_dir = str(tmp_path_factory.mktemp('rc'))
        dut = Module('PDK_skywater130', tmp_dir, 'rc.cir', pdk_details=pdk,
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
        probes = {'vout': test.probe('vout'),
                  'vin': test.probe('vin'),
                  'ivs': test.probe('i(vs)')}
        test.run()
        return test, probes

    def test_probe_equals_load_signal_sample_for_sample(self, ran):
        test, probes = ran
        expected = test.netlist.load_signal('vout')
        assert np.array_equal(probes['vout'].y, expected.y)
        assert np.array_equal(probes['vout'].x, expected.x)
        assert probes['vout'].xlabel == expected.xlabel

    def test_the_rc_actually_charged(self, ran):
        _, probes = ran
        assert float(np.max(probes['vout'].y)) == pytest.approx(1.0, abs=0.02)
        assert float(np.min(probes['vout'].y)) < 0.05

    def test_branch_current_spelling_equals_the_raw_vector(self, ran):
        """``i(vs)`` and the raw ``load_signal('vvs__silk_wave', 'I')`` are one vector."""
        test, probes = ran
        expected = test.netlist.load_signal('vvs__silk_wave', 'I')
        assert np.array_equal(probes['ivs'].y, expected.y)
        assert float(np.max(np.abs(probes['ivs'].y))) > 0.0

    def test_a_probe_made_after_the_run_reads_the_same_data(self, ran):
        test, probes = ran
        assert np.array_equal(test.probe('vout').y, probes['vout'].y)

    def test_a_probe_is_bound_after_the_run_and_caches_nothing(self, ran):
        test, probes = ran
        vout = probes['vout']
        assert vout.bound is True
        vout.y
        assert 'y' not in vout.__dict__          # resolved on access, not stored

    def test_the_raw_selector_form_agrees_with_the_spelling(self, ran):
        test, _ = ran
        assert np.array_equal(test.probe('i(vs)').y, test.probe('vvs__silk_wave', 'I').y)


@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestDeviceCurrentAgainstRealNgspice:
    """One sky130 nmos as the DUT, ammeter wrappers on, one transient run."""

    @pytest.fixture(scope='class')
    def ran(self, tmp_path_factory):
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
        tmp_dir = str(tmp_path_factory.mktemp('amm'))
        dut = Module('PDK_skywater130', tmp_dir, 'nm.cir', pdk_details=pdk,
                     module_name='nm')
        dut.define_module_ports({'d': PortDir.INOUT, 'g': PortDir.IN,
                                 's': PortDir.GROUND})
        with dut:
            pdk.nmos1p8('m1', d='d', g='g', s='s', b='s', w=1.0, l=0.15)
        dut.save_device_currents = True
        dut.terminal_current_method = 'ammeter_wrapper'
        test = dut.create_test('amm')
        test.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0'})
        test.add_source(sources.VoltageSource('vd', positive='vd',
                                              negative='0', dc=1.0))
        test.add_source(sources.VoltageSource('vg', positive='vg',
                                              negative='0', dc=0.7))
        test.add_sim_type(analysis.Transient(step=1e-9, stop=10e-9))
        probe = test.probe('xdut.m1:d')
        test.run()
        return test, probe

    def test_terminal_probe_resolves_to_the_ammeter_vector(self, ran):
        test, probe = ran
        expected = test.netlist.load_signal('v.xdut.xmm1.vamm_d', 'I')
        assert np.array_equal(probe.y, expected.y)

    def test_the_drain_actually_conducts(self, ran):
        _, probe = ran
        assert abs(float(np.asarray(probe.y)[-1])) > 1e-9

    def test_kcl_closes_across_the_four_terminal_probes(self, ran):
        """All four terminals are reachable by the spelling, and they sum to
        zero — the check that the names are the right four vectors."""
        test, _ = ran
        total = sum(float(np.asarray(test.probe(f'xdut.m1:{p}').y)[-1])
                    for p in 'dgsb')
        assert abs(total) < 1e-12
