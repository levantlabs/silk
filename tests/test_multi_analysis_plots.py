"""TASK-212 — a test with two analyses keeps BOTH plots, and silk never
guesses which one a metric reads.

The defect this pins, measured on an RC low-pass with ``Transient`` + ``AC``
declared in one test: silk emitted both dot-analyses, ngspice ran both, and
the ``.control`` block's bare ``write <path>`` saved the CURRENT plot only —
whichever analysis ngspice happened to run last. Every metric then read that
one plot, so the AC metric failed on a time axis (as an ``'Error: ...'``
STRING in the report, which the spec flow does not raise on) and the
time-domain metric returned a real number by accident. Tick order changed
nothing.

Three layers refuse the same thing in the same words — generate time
(``testbench_generator._validate``), run time
(``ModuleTestbench._require_unambiguous_analyses``) and the loader
(``backend.select_plot``) — and each is pinned here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from silk.circuit.module_test import (
    _analysis_candidates, _declared_analyses, _resolve_metric_analysis,
)
from silk.circuit.spice_netlister import PySpiceNetlister, _ngspice_plot_names
from silk.exceptions import SimulationError
from silk.simulation.analysis import AC, DC, Noise, Transient
from silk.simulation.backend import select_plot
from silk.signals.analysis_kind import plot_silk_analysis
from silk.signals.rawfile import read_rawfile
from silk.signals.spice import NgspiceSignal

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'rawfiles'


# ---------------------------------------------------------------------------
# (1) EMIT — the write line names every declared analysis's plot
# ---------------------------------------------------------------------------

def _netlister():
    """A bare netlister: this half of the emitter touches no PDK (the same
    ``__new__`` allowance ``generate_osdi_block`` documents)."""
    return PySpiceNetlister.__new__(PySpiceNetlister)


def test_plot_names_follow_deck_order_not_ngspice_order():
    """``sim_analysis_dict`` is keyed by analysis class and iterates in
    add order; the write-argument order is what sets the plot order in the
    rawfile, so the file reads the way the user declared the test. (ngspice
    RUNS .ac before .tran whatever the deck says — measured.)"""
    tran = Transient(step=1e-8, stop=1e-5)
    ac = AC(variation='dec', points=10, fstart=1, fstop=1e9)
    assert _ngspice_plot_names({'transient': tran, 'ac': ac}) == ['tran1', 'ac1']
    assert _ngspice_plot_names({'ac': ac, 'transient': tran}) == ['ac1', 'tran1']


def test_every_analysis_kind_has_a_plot_name():
    """All four, as MEASURED on ngspice-42 with a four-analysis RC deck."""
    d = {'transient': Transient(step=1e-8, stop=1e-5),
         'ac': AC(variation='dec', points=10, fstart=1, fstop=1e9),
         'dc': DC(source='v1', start=0, stop=1, step=0.1),
         'noise': Noise(output='out', ref='0', source='v1', variation='dec',
                        points=10, fstart=1, fstop=1e9)}
    assert _ngspice_plot_names(d) == ['tran1', 'ac1', 'dc1', 'noise1']


def test_one_analysis_keeps_the_bare_write_byte_for_byte():
    """The whole single-analysis contract in one assertion: a deck that
    declares one analysis emits the control block it always did, so the deck
    goldens and the sky130 pins are
    untouched."""
    nl = _netlister()
    before = nl.generate_control_block('/run/dir')
    assert before == nl.generate_control_block('/run/dir', plots=['tran1'])
    assert 'write /run/dir/rawfile.raw\n' in before


def test_two_analyses_qualify_the_write():
    nl = _netlister()
    block = nl.generate_control_block('/run/dir', plots=['tran1', 'ac1'])
    assert 'write /run/dir/rawfile.raw tran1.all ac1.all\n' in block


def test_a_noise_only_deck_still_setplots():
    """``setplot noise1`` selects the spectral-density plot over the
    integrated-totals plot ngspice leaves current. A single-analysis noise
    deck is unchanged; a multi-analysis one names noise1 in the write instead,
    so the setplot would be redundant."""
    nl = _netlister()
    assert 'setplot noise1\n' in nl.generate_control_block('/run/dir', noise=True)
    multi = nl.generate_control_block('/run/dir', noise=True,
                                      plots=['tran1', 'noise1'])
    assert 'setplot noise1\n' not in multi
    assert 'write /run/dir/rawfile.raw tran1.all noise1.all\n' in multi


def test_output_block_passes_the_declared_analyses_through():
    nl = _netlister()
    d = {'transient': Transient(step=1e-8, stop=1e-5),
         'ac': AC(variation='dec', points=10, fstart=1, fstop=1e9)}
    out = nl.generate_output_block('/run/dir', d)
    assert 'write /run/dir/rawfile.raw tran1.all ac1.all\n' in out


# ---------------------------------------------------------------------------
# One vocabulary, authored once — the three tables that spell it
# ---------------------------------------------------------------------------

def test_every_analysis_word_table_agrees_with_the_vocabulary():
    """``silk/signals/kinds.py`` is the vocabulary's home, but it is a
    deliberate LEAF with a pinned importer list, so two tables spell the four
    words without importing it: ``analysis_kind.SELECTABLE_ANALYSES`` (what
    ``analysis=`` accepts) and ``module_test._SIM_KEY_ANALYSIS`` (what
    ``add_sim_type`` files an analysis under). A test may import the leaf, so
    the drift is pinned here instead of at runtime.

    REVIEWER BINDING (2026-09-27, TASK-330.2, commit 178250ed, amended same
    day): ``'op'`` (analysis.Op) is a real, selectable silk analysis but is
    deliberately kept OUT of ``SILK_ANALYSES`` — the published metric-schema
    analysis/domain vocabulary
    (``tran|ac|dc|noise`` exactly). 'op' lives on the SIMULATION/selection
    side only: ``Op()``, ``_KIND_ANALYSIS``/``SELECTABLE_ANALYSES``,
    ``select_plot``, ``NGSPICE_PLOT_NAMES``, and (functionally the same
    role as ``_KIND_ANALYSIS`` — declared-analysis-to-silk-word translation,
    not schema export) ``module_test._SIM_KEY_ANALYSIS``. So each of these
    now equals ``SILK_ANALYSES`` PLUS ``'op'``, not ``SILK_ANALYSES`` itself.
    """
    from silk.circuit.module_test import _SIM_KEY_ANALYSIS
    from silk.signals.analysis_kind import SELECTABLE_ANALYSES
    from silk.signals.kinds import SILK_ANALYSES

    assert set(SELECTABLE_ANALYSES) == set(SILK_ANALYSES) | {'op'}
    assert set(_SIM_KEY_ANALYSIS.values()) == set(SILK_ANALYSES) | {'op'}
    # ...and the deck's own plot table covers exactly those five.
    from silk.circuit.spice_netlister import NGSPICE_PLOT_NAMES
    assert len(NGSPICE_PLOT_NAMES) == len(SILK_ANALYSES) + 1


# ---------------------------------------------------------------------------
# (2) SELECT — plot_silk_analysis / select_plot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fixture, analysis', [
    ('e_ng_tran.raw', 'tran'),
    ('e_xy_tran.raw', 'tran'),
    ('e_ng_ac.raw', 'ac'),
    ('e_xy_ac.raw', 'ac'),
    ('e_ng_dc.raw', 'dc'),
    ('e_xy_dc.raw', 'dc'),
    ('e_ng_noise.raw', 'noise'),
])
def test_plot_silk_analysis_on_real_simulator_output(fixture, analysis):
    """Pinned against the TASK-74 fixtures — real ngspice-42 and Xyce 7.10
    output — so the plotname table cannot drift from what simulators write."""
    raw = read_rawfile(FIXTURES / fixture)
    assert plot_silk_analysis(raw.plots[0]) == analysis


def test_an_operating_point_is_selectable_on_ngspice():
    """TASK-330.2: analysis.Op() emits a real '.op' whose ngspice plot is
    named 'Operating Point', so it is selectable as analysis='op' like any
    other declared analysis."""
    raw = read_rawfile(FIXTURES / 'e_ng_op.raw')
    assert plot_silk_analysis(raw.plots[0]) == 'op'


def test_an_operating_point_is_dc_not_op_on_xyce():
    """Xyce has no '.print op' form (TASK-330.1): Op() renders as a
    single-point '.dc' sweep there, which is plotname-indistinguishable
    from a genuine one-point DC() (both are 'DC transfer characteristic') —
    plotname wins by design (analysis_kind._KIND_ANALYSIS), so this reads
    as 'dc', not 'op'. A known, documented Xyce limitation, not a bug."""
    raw = read_rawfile(FIXTURES / 'e_xy_op.raw')
    assert plot_silk_analysis(raw.plots[0]) == 'dc'


class _Plot:
    """A plot stub: the reader's shape, name and vectors only."""

    def __init__(self, plotname):
        self.plotname = plotname
        self.vectors = [_Vec('time')]
        self.real = True
        self.npoints = 8


class _Vec:
    def __init__(self, name):
        self.name = name
        self.data = np.arange(8, dtype=float)


class _Raw:
    """A RawFile stub: the plot list plus the ``current_plot`` proxy (the LAST
    plot), which is the shape every pre-TASK-212 consumer reads through."""

    def __init__(self, *plots):
        self.plots = list(plots)

    @property
    def current_plot(self):
        return self.plots[-1] if self.plots else None


def test_one_plot_and_no_analysis_is_the_old_path():
    """``None`` means "the reader's own default" — every rawfile silk has ever
    written loads by exactly the path it loaded by before."""
    assert select_plot(_Raw(_Plot('Transient Analysis')), None) is None


def test_a_native_result_with_no_plots_selects_nothing():
    """An in-process backend's own result shape has no ``plots``; the function
    must be safe to call on every load path."""
    assert select_plot(object(), None) is None


def test_more_than_one_plot_and_no_analysis_refuses_naming_the_plots():
    raw = _Raw(_Plot('Transient Analysis'), _Plot('AC Analysis'))
    with pytest.raises(SimulationError) as e:
        select_plot(raw, None)
    assert "'Transient Analysis' (tran)" in str(e.value)
    assert "'AC Analysis' (ac)" in str(e.value)
    assert 'does not guess' in str(e.value)


def test_naming_an_analysis_the_file_does_not_hold_refuses():
    raw = _Raw(_Plot('Transient Analysis'), _Plot('AC Analysis'))
    with pytest.raises(SimulationError, match="holds no 'noise' plot"):
        select_plot(raw, 'noise')


def test_an_unknown_analysis_word_refuses():
    raw = _Raw(_Plot('Transient Analysis'), _Plot('AC Analysis'))
    with pytest.raises(SimulationError, match='is not a silk analysis'):
        select_plot(raw, 'transient')      # the classifier's word, not silk's


def test_from_rawfile_reads_the_selected_plot():
    """The ``plot=`` parameter is the whole of the change to the frozen
    ``from_rawfile`` mechanism: one substitution, same body."""
    raw = read_rawfile(FIXTURES / 'e_ng_ac.raw')
    default = NgspiceSignal.from_rawfile(raw, raw.vectors[1].name.split('(')[1]
                                         .rstrip(')'), 'V')
    selected = NgspiceSignal.from_rawfile(
        raw, raw.vectors[1].name.split('(')[1].rstrip(')'), 'V',
        plot=raw.plots[0])
    assert selected.xlabel == default.xlabel
    np.testing.assert_array_equal(selected.y, default.y)


# ---------------------------------------------------------------------------
# (2b) The other consumers of the loader (the task's PATH SURVEY, item 5)
# ---------------------------------------------------------------------------

class _ACPlot(_Plot):
    def __init__(self):
        super().__init__('AC Analysis')
        self.real = False
        self.vectors = [_Vec('frequency')]


class _Mod:
    def __init__(self, raw):
        self.simResultsAll = raw


def test_a_tran_plot_beside_an_ac_plot_is_not_small_signal():
    """STR-001 declines AC/spectral results (TASK-81) by asking this. With
    several plots it has to ask about all of them: a tran+ac run holds real
    node voltages whichever analysis ngspice ran LAST, and declining it would
    decline on the strength of plot order."""
    from silk.signals.analysis_kind import _spice_results_are_small_signal

    assert not _spice_results_are_small_signal(
        _Mod(_Raw(_ACPlot(), _Plot('Transient Analysis'))))
    assert not _spice_results_are_small_signal(
        _Mod(_Raw(_Plot('Transient Analysis'), _ACPlot())))
    # ...and an ac-only file is still declined, one plot or two.
    assert _spice_results_are_small_signal(_Mod(_Raw(_ACPlot())))
    assert _spice_results_are_small_signal(_Mod(_Raw(_ACPlot(), _ACPlot())))


def test_str001_names_the_transient_plot_on_a_multi_analysis_rawfile():
    """``_resolve_waveform`` swallows every exception, so without the explicit
    retry a multi-plot refusal would read as "no waveform" and silently skip
    the device — findings would just stop appearing."""
    from silk.lint.rules.stress import _resolve_waveform

    calls = []

    class _Loader:
        def load_signal(self, name, type, *, analysis=None):
            calls.append(analysis)
            if analysis is None:
                raise SimulationError('holds 2 analyses')
            return f'<{name} from {analysis}>'

    assert _resolve_waveform(_Loader(), 'vout') == '<vout from tran>'
    assert calls == [None, 'tran']


# ---------------------------------------------------------------------------
# (3) TIE — the resolution rule, shared by all three layers
# ---------------------------------------------------------------------------

_ANY = [('sig', 'any')]                      # average, rms, value_at, ...
_FREQ = [('sig', 'frequency_response')]      # dc_gain_dB, gain_at_freq, ...
_TRAN = [('sig', 'transient')]               # rise_time, overshoot, ...


def test_fewer_than_two_declared_analyses_never_selects():
    """One analysis is one plot. Every test silk has ever run takes this
    branch, and takes it whatever kind its metrics declare."""
    assert _resolve_metric_analysis('m', _ANY, ['tran'], None) is None
    assert _resolve_metric_analysis('m', _FREQ, ['ac'], None) is None
    assert _resolve_metric_analysis('m', _ANY, [], None) is None


def test_exactly_one_producer_needs_no_field():
    """A frequency_response can come from .ac and from nothing else in the
    test, so asking the user would be noise (AC #3)."""
    assert _resolve_metric_analysis('m', _FREQ, ['tran', 'ac'], None) == 'ac'
    assert _resolve_metric_analysis('m', _TRAN, ['tran', 'ac'], None) == 'tran'


def test_more_than_one_producer_requires_the_field():
    with pytest.raises(ValueError) as e:
        _resolve_metric_analysis('m', _ANY, ['tran', 'ac'], None,
                                 signals=['vout'])
    msg = str(e.value)
    assert "'vout' could come from more than one analysis" in msg
    assert "'ac', 'tran'" in msg
    assert "analysis='ac' or analysis='tran'" in msg


def test_an_explicit_analysis_settles_it():
    assert _resolve_metric_analysis('m', _ANY, ['tran', 'ac'], 'tran') == 'tran'
    assert _resolve_metric_analysis('m', _ANY, ['tran', 'ac'], 'ac') == 'ac'


def test_an_undeclared_analysis_is_refused():
    with pytest.raises(ValueError, match='but this test declares'):
        _resolve_metric_analysis('m', _ANY, ['tran', 'ac'], 'noise')


def test_an_undeclared_analysis_is_refused_in_a_one_analysis_test_too():
    """Checked BEFORE the one-analysis short-circuit: naming an analysis the
    test does not declare is a caller error whether or not the choice would
    have mattered."""
    with pytest.raises(ValueError, match="this test declares 'tran'"):
        _resolve_metric_analysis('m', _ANY, ['tran'], 'ac')


def test_no_declared_analyses_leaves_the_answer_to_the_loader():
    """An empty declared list is not "the test declares nothing you may name",
    it is "there is nothing here to compare against" — evaluate(rawfile=...)
    on an externally produced file. The loader answers instead."""
    assert _resolve_metric_analysis('m', _ANY, [], 'ac') is None


def test_an_analysis_the_kind_cannot_come_from_is_refused():
    """A frequency_response never comes from .tran, so naming it is a caller
    error even though the test declares it."""
    with pytest.raises(ValueError, match='cannot feed this metric'):
        _resolve_metric_analysis('m', _FREQ, ['tran', 'ac'], 'tran')


def test_an_unknown_analysis_word_is_refused_before_anything_else():
    with pytest.raises(ValueError, match='is not a silk analysis'):
        _resolve_metric_analysis('m', _ANY, ['tran'], 'transient')


def test_a_slot_with_no_kind_constrains_nothing():
    """A schema exported by a pre-2.0 silk carries bare names. An absence of
    information is not a statement that an analysis cannot feed the slot."""
    assert _analysis_candidates([('sig', None)], ['tran', 'ac']) == ['tran', 'ac']


def test_domain_is_never_consulted():
    """``average``'s domain is 'tran', but its INPUT kind is 'any' — so in a
    dc+ac test it is ambiguous rather than silently reading a transient that
    is not there. Domain is a label (TASK-208); selection reads kinds."""
    with pytest.raises(ValueError, match='more than one analysis'):
        _resolve_metric_analysis('m', _ANY, ['dc', 'ac'], None)


# ---------------------------------------------------------------------------
# (3b) The same rule, through the public API
# ---------------------------------------------------------------------------

def _rc_test(tmp_path, pdk, *, order='tran_first', analysis='tran',
             extra_ac=True):
    """An RC low-pass DUT in a testbench with Transient + AC — the repro
    circuit from the task, built through the public API."""
    from silk.circuit.modules import Module
    from silk import sources
    from silk.elements import elements

    mod = Module('PDK_skywater130', str(tmp_path), 'rc.cir', pdk_details=pdk)
    mod.define_module_name('rc_lpf')
    mod.define_module_ports({'vin': 'input', 'vout': 'output'})
    mod.add(elements.resistor('r1', p='vin', n='vout',
                                      value=1e3))
    mod.add(elements.capacitor('c1', p='vout', n='0',
                                       value=1e-9))
    tb = mod.create_test('rc')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add(sources.vpulse('in', positive='vin', negative='0', v1=0, v2=1,
                          td=0, tr=1e-9, tf=1e-9, pw=4e-6, per=8e-6, ac=1))
    tran = Transient(step=1e-8, stop=1.6e-5)
    ac = AC(variation='dec', points=10, fstart=1, fstop=1e9)
    declared = (tran, ac) if order == 'tran_first' else (ac, tran)
    for a in (declared if extra_ac else (tran,)):
        tb.add_sim_type(a)
    tb.add_metric('gain', type='dc_gain_dB', signal='vout')
    tb.add_metric('vavg', type='average', signal='vout', analysis=analysis)
    return mod, tb


def test_declared_analyses_reads_the_live_dict():
    """Off ``sim_analysis_dict``, not off anything recorded when the metric was
    added — a hand-built test may add sim types AFTER its metrics."""
    class _Mod:
        sim_analysis_dict = {'transient': None, 'ac': None}
    assert _declared_analyses(_Mod()) == ['tran', 'ac']
    assert _declared_analyses(object()) == []


def test_analysis_is_refused_on_the_callable_form():
    from silk.circuit.module_test import ModuleTestbench
    tb = ModuleTestbench.__new__(ModuleTestbench)
    tb._metrics = []
    with pytest.raises(TypeError, match='ckt.load_signal'):
        tb.add_metric('m', lambda ckt: 0.0, analysis='tran')


def test_a_bad_analysis_word_is_refused_at_declaration():
    from silk.circuit.module_test import ModuleTestbench
    tb = ModuleTestbench.__new__(ModuleTestbench)
    tb._metrics = []
    with pytest.raises(ValueError, match='is not a silk analysis'):
        tb.add_metric('m', type='average', signal='vout', analysis='transient')


# ---------------------------------------------------------------------------
# (3c) THE SAME RULE ON A PROBE — TASK-212 x TASK-215
#
# The function form has no ``analysis=`` of its own: the probe carries it
# (``tb.probe('vout', analysis='tran')``, design §2.3). Every branch of
# ``_resolve_metric_analysis`` is reachable from there, so each is pinned
# through a probe as well as through the declarative field — a rule stated
# once must refuse identically wherever it is asked.
# ---------------------------------------------------------------------------

class _DeclaredMod:
    """Declared analyses and the flat node inventory used by probe binding."""

    def __init__(self, *words):
        keys = {'tran': 'transient', 'ac': 'ac', 'dc': 'dc',
                'noise': 'noise'}
        self.sim_analysis_dict = {keys[w]: None for w in words}
        self.net_dict = dict.fromkeys(('vout', 'vp', 'vm'))
        self.pin_list = None
        self.elem_dict = {}


def _bench(*declared):
    """A testbench with the fields probes and metrics touch.

    No PDK, no netlist Module, no simulator: the question these tests ask —
    which plot does this slot read — is answered from the declared analyses
    and the metric's own signal kinds, and nothing else.
    """
    from silk.circuit.module_test import ModuleTestbench
    tb = ModuleTestbench.__new__(ModuleTestbench)
    tb._metrics = []
    tb._probe_registry = []
    tb.test_name = 'rc'
    tb.module = None
    tb.netlist = _DeclaredMod(*declared) if declared else None
    return tb


def _slot_analyses(tb):
    from silk.circuit.module_test import (
        _declared_analyses, _function_form_slot_analyses,
    )
    return _function_form_slot_analyses(
        tb._metrics[-1], _declared_analyses(tb.netlist))


def test_a_probe_names_the_plot_for_its_slot():
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('vavg', average, sig=tb.probe('vout', analysis='tran'))
    assert _slot_analyses(tb) == {'sig': 'tran'}


def test_the_other_probe_of_the_same_net_names_the_other_plot():
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('vavg', average, sig=tb.probe('vout', analysis='ac'))
    assert _slot_analyses(tb) == {'sig': 'ac'}


def test_one_producer_needs_no_field_on_the_probe_either():
    """``dc_gain_dB`` takes a ``frequency_response``, and ``.ac`` is the only
    declared analysis that makes one. Nothing to ask, so nothing is asked."""
    from silk.metrics import dc_gain_dB
    tb = _bench('tran', 'ac')
    tb.add_metric('gain', dc_gain_dB, sig=tb.probe('vout'))
    assert _slot_analyses(tb) == {'sig': 'ac'}


def test_a_kind_any_probe_without_an_analysis_is_refused():
    """``average`` takes kind ``any``, so in a tran+ac test both plots could
    feed it. The refusal names both, and names the PROBE spelling that
    settles it — the declarative form's ``analysis=`` is not the fix here."""
    from silk.exceptions import MetricBindingError
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('vavg', average, sig=tb.probe('vout'))
    with pytest.raises(MetricBindingError) as exc:
        _slot_analyses(tb)
    message = str(exc.value)
    assert 'could come from more than one analysis' in message
    assert "'ac', 'tran'" in message
    assert "tb.probe(..., analysis='ac')" in message
    assert "tb.probe(..., analysis='tran')" in message
    assert "'vout'" in message


def test_the_refusal_is_a_configuration_error_so_it_propagates():
    """TASK-215's contract: a binding failure must reach the caller, not be
    folded into ``run_all_tests``'s per-test ``{'error': str}``."""
    from silk.exceptions import ConfigurationError, MetricBindingError
    assert issubclass(MetricBindingError, ConfigurationError)


def test_an_undeclared_analysis_on_a_probe_is_refused():
    from silk.exceptions import MetricBindingError
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('vavg', average, sig=tb.probe('vout', analysis='dc'))
    with pytest.raises(MetricBindingError) as exc:
        _slot_analyses(tb)
    assert "this test declares 'tran', 'ac'" in str(exc.value)


def test_an_analysis_the_kind_cannot_come_from_is_refused_on_a_probe():
    from silk.exceptions import MetricBindingError
    from silk.metrics import dc_gain_dB
    tb = _bench('tran', 'ac')
    tb.add_metric('gain', dc_gain_dB, sig=tb.probe('vout', analysis='tran'))
    with pytest.raises(MetricBindingError) as exc:
        _slot_analyses(tb)
    assert 'cannot feed' in str(exc.value)


def test_a_bad_analysis_word_is_refused_at_the_probe_call():
    tb = _bench('tran', 'ac')
    with pytest.raises(ValueError, match='not a silk analysis'):
        tb.probe('vout', analysis='transient')


def test_one_declared_analysis_selects_nothing_for_a_probe():
    """Every test silk has ever run takes this branch: one analysis is one
    plot, and ``load_signal`` is called with the argument list it always
    had."""
    from silk.metrics import average
    tb = _bench('tran')
    tb.add_metric('vavg', average, sig=tb.probe('vout'))
    assert _slot_analyses(tb) == {'sig': None}


def test_a_derived_expression_is_settled_once_for_the_whole_recipe():
    """``(vp + vm) / 2`` has two leaves and one slot; the slot's kind decides,
    and the answer is handed to both leaves at resolution."""
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    vcm = (tb.probe('vp', analysis='tran') + tb.probe('vm', analysis='tran')) / 2
    tb.add_metric('vcm_avg', average, sig=vcm)
    assert _slot_analyses(tb) == {'sig': 'tran'}


def test_a_derived_expression_of_bare_probes_is_refused():
    from silk.exceptions import MetricBindingError
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('vcm_avg', average,
                  sig=(tb.probe('vp') + tb.probe('vm')) / 2)
    with pytest.raises(MetricBindingError, match='more than one analysis'):
        _slot_analyses(tb)


def test_leaves_that_pin_different_analyses_are_left_to_themselves():
    """A deliberately mixed recipe: each leaf already names its own plot, so
    there is nothing left to choose and no ambiguity to refuse."""
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('mixed', average,
                  sig=tb.probe('vout', analysis='tran')
                      - tb.probe('vout', analysis='ac'))
    assert _slot_analyses(tb) == {'sig': None}


def test_a_string_slot_is_settled_by_its_kind_like_the_declarative_form():
    """The function form accepts a path string in a slot; it selects a plot
    on exactly the same rule as ``type=``/``signal=`` does."""
    from silk.exceptions import MetricBindingError
    from silk.metrics import average, dc_gain_dB
    tb = _bench('tran', 'ac')
    tb.add_metric('gain', dc_gain_dB, sig='vout')
    assert _slot_analyses(tb) == {'sig': 'ac'}
    tb.add_metric('vavg', average, sig='vout')
    with pytest.raises(MetricBindingError, match='more than one analysis'):
        _slot_analyses(tb)


def test_a_bound_signal_slot_selects_no_plot():
    """Samples already in hand: no plot is read, so none is chosen and none
    is demanded."""
    from silk.metrics import average
    from silk.signals.base import Signal
    tb = _bench('tran', 'ac')
    sig = Signal('vout', 'voltage', 'time', 'voltage',
                 np.linspace(0, 1, 8), np.zeros(8))
    tb.add_metric('vavg', average, sig=sig)
    assert _slot_analyses(tb) == {'sig': None}


def test_a_slot_that_reads_no_plot_is_never_ambiguous():
    """A slot bound to something that never reaches the loader — here a bare
    ndarray — cannot be ambiguous, so it must not be refused as such."""
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    tb.add_metric('vavg', average, sig=np.zeros(8))
    assert _slot_analyses(tb) == {'sig': None}


def test_the_pre_extraction_layer_checks_the_function_form_too():
    """``_require_unambiguous_analyses`` is layer 2 for BOTH bound forms: it
    runs before the extraction loop, which catches every exception into an
    ``'Error: ...'`` string."""
    from silk.exceptions import MetricBindingError
    from silk.metrics import average, dc_gain_dB
    tb = _bench('tran', 'ac')
    tb.add_metric('gain', dc_gain_dB, sig=tb.probe('vout'))
    tb._require_unambiguous_analyses()          # unambiguous: silent
    tb.add_metric('vavg', average, sig=tb.probe('vout'))
    with pytest.raises(MetricBindingError, match='more than one analysis'):
        tb._require_unambiguous_analyses()


def test_analysis_is_refused_on_the_function_form_of_add_metric():
    """The metric-level keyword belongs to the declarative form. On the
    function form the refusal names the probe spelling instead."""
    from silk.metrics import average
    tb = _bench('tran', 'ac')
    with pytest.raises(TypeError) as exc:
        tb.add_metric('vavg', average, sig=tb.probe('vout'), analysis='tran')
    assert 'tb.probe(<net>, analysis=' in str(exc.value)


def test_both_forms_refuse_the_same_ambiguity_in_the_same_sentence():
    """One rule, one diagnosis. Only the closing remedy differs, because only
    the place to write the answer differs."""
    from silk.circuit.module_test import _resolve_metric_analysis
    from silk.exceptions import MetricBindingError
    from silk.metrics import average

    tb = _bench('tran', 'ac')
    tb.add_metric('vavg', average, sig=tb.probe('vout'))
    with pytest.raises(MetricBindingError) as probe_exc:
        _slot_analyses(tb)
    with pytest.raises(ValueError) as field_exc:
        _resolve_metric_analysis("add_metric('vavg', type='average')",
                                 [('sig', 'any')], ['tran', 'ac'], None,
                                 signals=['vout'])

    head = "'vout' could come from more than one analysis in this test: 'ac', 'tran'."
    assert head in str(probe_exc.value)
    assert head in str(field_exc.value)
    assert str(field_exc.value).endswith(
        "Add analysis='ac' or analysis='tran' to the metric.")
    assert str(probe_exc.value).endswith(
        "Name the analysis on the probe: tb.probe(..., analysis='ac') or "
        "tb.probe(..., analysis='tran').")


# ---------------------------------------------------------------------------
# (4) END TO END on a real simulator — the task's repro, both tick orders
# ---------------------------------------------------------------------------

def _ngspice_missing():
    from silk.simulation.runner import ngspice_executable
    return shutil.which(ngspice_executable()) is None


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
@pytest.mark.parametrize('order', ['tran_first', 'ac_first'])
def test_rc_two_analyses_end_to_end(tmp_path, order):
    """THE repro. Both plots in the rawfile, both metrics real numbers, the
    same numbers in both tick orders."""
    from silk import load_pdk

    mod, tb = _rc_test(tmp_path, load_pdk('sky130'), order=order)
    report = tb.run()

    raw = tb.netlist._raw.native
    assert [p.plotname for p in raw.plots] == (
        ['Transient Analysis', 'AC Analysis'] if order == 'tran_first'
        else ['AC Analysis', 'Transient Analysis'])

    # The AC metric is a FLOAT in dB, not the "Error: sig must have
    # xlabel='frequency'" string the report used to carry.
    assert isinstance(report.metrics['gain'], float)
    assert abs(report.metrics['gain']) < 0.1          # unity gain at DC
    # ... and the time-domain metric reads the transient, not by accident.
    assert report.metrics['vavg'] == pytest.approx(0.5, abs=0.02)

    tran_plot = select_plot(raw, 'tran')
    vout = [v for v in tran_plot.vectors if v.name == 'v(vout)'][0].data
    assert report.metrics['vavg'] == pytest.approx(float(np.mean(vout)),
                                                   rel=0.02)


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_an_ambiguous_metric_raises_before_extraction(tmp_path):
    """It must RAISE, not land in the report as a string: the extraction loop
    catches everything, and the spec flow does not raise on a string metric —
    which is exactly how the wrong-plot read went unnoticed."""
    from silk import load_pdk

    mod, tb = _rc_test(tmp_path, load_pdk('sky130'), analysis=None)
    with pytest.raises(ValueError, match='more than one analysis'):
        tb.run()


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_the_callable_form_selects_a_plot_through_the_loader(tmp_path):
    """AC #2: the callable form has no ``analysis=`` of its own — it selects
    where it LOADS, with the same keyword, and both forms land on the same
    number from the same plot."""
    from silk import load_pdk
    from silk.metrics import average

    mod, tb = _rc_test(tmp_path, load_pdk('sky130'))
    tb.add_metric('vavg_lambda', lambda ckt: average(
        ckt.load_signal('vout', analysis='tran')))
    report = tb.run()
    assert report.metrics['vavg_lambda'] == pytest.approx(
        report.metrics['vavg'], rel=1e-12)


# ---------------------------------------------------------------------------
# (4b) END TO END through PROBES — TASK-212 x TASK-215, on the same repro
# ---------------------------------------------------------------------------

def _rc_probe_test(tmp_path, pdk, *, pin_average=True):
    """The repro circuit again, measured the taught way: probes and the
    function form instead of ``type=``/``signal=``.

    ``pin_average`` decides whether the kind-``any`` metric's probe names its
    analysis. With it off, the run must refuse rather than read a plot.
    """
    from silk import sources
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.metrics import average, dc_gain_dB

    mod = Module('PDK_skywater130', str(tmp_path), 'rc.cir', pdk_details=pdk)
    mod.define_module_name('rc_lpf')
    mod.define_module_ports({'vin': 'input', 'vout': 'output'})
    mod.add(elements.resistor('r1', p='vin', n='vout',
                                      value=1e3))
    mod.add(elements.capacitor('c1', p='vout', n='0',
                                       value=1e-9))
    tb = mod.create_test('rc')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add(sources.vpulse('in', positive='vin', negative='0', v1=0, v2=1,
                          td=0, tr=1e-9, tf=1e-9, pw=4e-6, per=8e-6, ac=1))
    tb.add_sim_type(Transient(step=1e-8, stop=1.6e-5))
    tb.add_sim_type(AC(variation='dec', points=10, fstart=1, fstop=1e9))

    vout_t = tb.probe('vout', analysis='tran') if pin_average else tb.probe('vout')
    vout_f = tb.probe('vout', analysis='ac')
    # No analysis on this one: dc_gain_dB takes a frequency response and .ac
    # is the only declared analysis that makes one, so silk reads it without
    # being told — the "unambiguous needs no field" half of the rule.
    gain_probe = tb.probe('vout')
    tb.add_metric('vavg', average, sig=vout_t)
    tb.add_metric('gain', dc_gain_dB, sig=gain_probe)
    return mod, tb, {'vout_t': vout_t, 'vout_f': vout_f, 'gain': gain_probe}


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_probes_of_two_analyses_read_the_right_plots_end_to_end(tmp_path):
    """The repro measured through probes. Two probes of the SAME net, one per
    analysis, each reading its own plot — and a third that names none and is
    settled by its metric's kind."""
    from silk import load_pdk

    mod, tb, probes = _rc_probe_test(tmp_path, load_pdk('sky130'))
    report = tb.run()

    raw = tb.netlist._raw.native
    assert [p.plotname for p in raw.plots] == ['Transient Analysis',
                                               'AC Analysis']

    # Reading the probes directly: each resolves against ITS plot, on access.
    assert probes['vout_t'].xlabel == 'time'
    assert probes['vout_f'].xlabel == 'frequency'

    tran_plot = select_plot(raw, 'tran')
    ac_plot = select_plot(raw, 'ac')
    tran_vout = [v for v in tran_plot.vectors if v.name == 'v(vout)'][0].data
    ac_vout = [v for v in ac_plot.vectors if v.name == 'v(vout)'][0].data
    assert np.array_equal(probes['vout_t'].y, np.asarray(tran_vout))
    assert np.array_equal(probes['vout_f'].y, np.asarray(ac_vout))

    # ... and the metrics built on them agree with those plots.
    assert report.metrics['vavg'] == pytest.approx(float(np.mean(tran_vout)),
                                                   rel=0.02)
    assert isinstance(report.metrics['gain'], float)
    assert abs(report.metrics['gain']) < 0.1        # unity gain at DC

    # tb.plot() resolves through the same rule: a pinned probe draws its own
    # plot rather than meeting the loader's multi-plot refusal. The two
    # panels do NOT share an x-axis — one is seconds, the other hertz — and
    # each is labelled from its own.
    import matplotlib.pyplot as plt
    fig = tb.plot(probes['vout_t'], probes['vout_f'], show=False)
    labels = [ax.get_xlabel() for ax in fig.axes]
    assert len(fig.axes) == 2
    assert all(labels), labels
    assert labels[0] != labels[1], labels
    plt.close(fig)

    # Overlay cannot reconcile them, and says so instead of handing
    # matplotlib a 1646-point x and a 91-point y.
    with pytest.raises(ValueError, match='one x-axis'):
        tb.plot(probes['vout_t'], probes['vout_f'], overlay=True, show=False)


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_a_bare_kind_any_probe_raises_from_run(tmp_path):
    """``average`` on a probe with no analysis, in a tran+ac test: refused
    before extraction, as a ``MetricBindingError`` — so it raises out of
    ``run()`` instead of landing in the report as a string, and propagates
    through ``run_all_tests`` instead of becoming that test's error entry."""
    from silk import load_pdk
    from silk.exceptions import MetricBindingError

    mod, tb, _ = _rc_probe_test(tmp_path, load_pdk('sky130'),
                                pin_average=False)
    with pytest.raises(MetricBindingError) as exc:
        tb.run()
    message = str(exc.value)
    assert 'could come from more than one analysis' in message
    assert "'ac', 'tran'" in message
    assert "tb.probe(..., analysis='tran')" in message


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_the_probe_form_and_the_declarative_form_agree(tmp_path):
    """The same two numbers, said two ways: ``analysis=`` on the metric and
    ``analysis=`` on the probe must select the same plots."""
    from silk import load_pdk

    _, declarative = _rc_test(tmp_path / 'declarative', load_pdk('sky130'))
    _, probed, _ = _rc_probe_test(tmp_path / 'probed', load_pdk('sky130'))
    a = declarative.run().metrics
    b = probed.run().metrics
    assert b['vavg'] == pytest.approx(a['vavg'], rel=1e-12)
    assert b['gain'] == pytest.approx(a['gain'], rel=1e-12)


def test_a_missing_plot_is_an_abort_sentinel():
    """The qualified write is ALL-OR-NOTHING: MEASURED on ngspice-42, a
    ``write <path> tran1.all dc1.all`` whose ``dc1`` never ran prints
    ``Warning from checkvalid: vector dc1.all is not available or has zero
    length`` and writes NO FILE, exit 0. That is fail-loud (silk's abort gate
    fires on the missing rawfile) but the reason has to name the missing plot,
    or the user is told only that nothing was written."""
    from silk.simulation.runner import NGSPICE_ABORT_SENTINELS

    line = ('Warning from checkvalid: vector dc1.all is not available or '
            'has zero length.')
    assert any(s in line for s in NGSPICE_ABORT_SENTINELS)


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_one_failing_analysis_aborts_loudly_and_says_why(tmp_path):
    """A test whose .dc names a source the deck has no copy of. Before
    TASK-352 ngspice ran the transient, failed the DC, wrote nothing and
    exited 0, and silk's abort gate raised SimulationAborted. Since TASK-352
    silk refuses the request before any file is written (SweepSourceError,
    naming the source), and the netlister refuses it again, so the simulator
    is never reached on this route. The simulator-side gate stays covered by
    test_a_missing_plot_is_an_abort_sentinel above and by
    tests/test_sim_abort_integration.py."""
    from silk import load_pdk
    from silk.exceptions import SweepSourceError
    from silk.elements import elements
    from silk import sources
    from silk.circuit.modules import Module

    mod = Module('PDK_skywater130', str(tmp_path), 'rc.cir',
                 pdk_details=load_pdk('sky130'))
    mod.define_module_name('rc_lpf')
    mod.define_module_ports({'vin': 'input', 'vout': 'output'})
    mod.add(elements.resistor('r1', p='vin', n='vout',
                                      value=1e3))
    mod.add(elements.capacitor('c1', p='vout', n='0',
                                       value=1e-9))
    tb = mod.create_test('rc')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add(sources.vpulse('in', positive='vin', negative='0', v1=0, v2=1,
                          td=0, tr=1e-9, tf=1e-9, pw=4e-6, per=8e-6, ac=1))
    tb.add_sim_type(Transient(step=1e-8, stop=1.6e-5))
    tb.add_sim_type(DC(source='vnope', start=0, stop=1, step=0.1))
    tb.add_metric('vavg', type='average', signal='vout', analysis='tran')
    with pytest.raises(SweepSourceError, match='vnope'):
        tb.run()
    assert not list(tmp_path.rglob('*.raw')), 'refused before simulating'


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_the_explorer_lists_both_plots(tmp_path):
    """OWNER's report — 'I could only plot one of the sim types' — is the same
    root cause, and needs NO explorer change: ``view()`` already plans one
    entry per ``raw.plots`` (``silk/signals/view/emit.py``), so writing every
    declared analysis's plot is the whole of the fix on that side too."""
    import json
    import re

    from silk import load_pdk

    mod, tb = _rc_test(tmp_path, load_pdk('sky130'))
    tb.run()
    page = tb.netlist.view(out_path=str(tmp_path / 'rc.html'), open=False)
    data = Path(page).with_suffix('').with_suffix('.data.js').read_text()
    payload = json.loads(re.search(r'var p=(\{.*\});', data, re.S).group(1))
    assert [p['name'] for p in payload['plots']] == ['Transient Analysis',
                                                     'AC Analysis']


@pytest.mark.slow
@pytest.mark.skipif(_ngspice_missing(), reason='ngspice not installed')
def test_the_loader_refuses_a_multi_plot_file_with_no_analysis(tmp_path):
    from silk import load_pdk

    mod, tb = _rc_test(tmp_path, load_pdk('sky130'))
    tb.run()
    with pytest.raises(SimulationError, match='holds 2 analyses'):
        tb.netlist.load_signal('vout')
    assert tb.netlist.load_signal('vout', analysis='ac').xlabel == 'frequency'
    assert tb.netlist.load_signal('vout', analysis='tran').xlabel == 'time'


@pytest.mark.parametrize('name,axis,analysis', [
    ('DC transfer characteristic', 'sweep', 'dc'),
    ('Transient Analysis', 'TIME', 'tran'),
    ('AC Analysis', 'FREQUENCY', 'ac'),
])
def test_xyce_step_prefix_uses_trailing_analysis_name(name, axis, analysis):
    from silk.signals.rawfile import RawFile, RawPlot, RawVector
    plot = RawPlot(
        {'plotname': 'Step Analysis: Step 1 of 1 params: name = ... ' + name},
        [RawVector(axis, axis, (), np.array([20e-6])),
         RawVector('V(G1)', 'voltage', (), np.array([.7])),
         RawVector('V(G2)', 'voltage', (), np.array([.7]))],
        1, True, None)
    assert plot_silk_analysis(plot) == analysis
    assert select_plot(RawFile([plot]), analysis) is plot


def test_xyce_step_parameter_names_do_not_override_the_analysis():
    from types import SimpleNamespace
    plot = SimpleNamespace(
        plotname='Step Analysis: Step 1 of 1 params: noise = AC Analysis DC transfer characteristic',
        real=True, npoints=1, vectors=[SimpleNamespace(name='sweep', data=np.array([20e-6]))])
    assert plot_silk_analysis(plot) == 'dc'


def test_nonstep_custom_prefix_keeps_axis_fallback():
    from types import SimpleNamespace
    plot = SimpleNamespace(
        plotname='custom label DC transfer characteristic', real=True, npoints=1,
        vectors=[SimpleNamespace(name='sweep', data=np.array([0.]))])
    assert plot_silk_analysis(plot) == 'op'


# Same-kind selection: real measured data and explicitly synthetic reader objects.
PLOT_SELECTION = FIXTURES.parent / 'plot_selection'


def _assert_duplicate_plot_error(exc, kind):
    assert type(exc) is SimulationError
    message = str(exc)
    assert f"2 {kind!r} plots" in message
    assert 'not currently supported by silk' in message
    assert 'future extension' in message
    assert 'separate rawfile' in message


@pytest.mark.parametrize('filename,values', [
    ('combined.raw', (1., 9.)), ('reversed.raw', (9., 1.)),
])
@pytest.mark.parametrize('lazy', [False, True])
def test_recorded_duplicate_transients_refuse_without_reading_vectors(filename, values, lazy):
    raw = read_rawfile(PLOT_SELECTION / filename, lazy=lazy)
    assert [p.plotname for p in raw.plots] == ['Transient Analysis'] * 2
    with pytest.raises(SimulationError) as exc:
        select_plot(raw, 'tran')
    _assert_duplicate_plot_error(exc.value, 'tran')
    if lazy:
        assert all(v._data is None for p in raw.plots for v in p.vectors)
    # Parsing preserves order; explicit RawPlot access remains available.
    for plot, value in zip(raw.plots, values):
        signal = NgspiceSignal.from_rawfile(raw, 'out', plot=plot)
        np.testing.assert_allclose(signal.y, value)


def _synthetic_selection_plot(name, axis='time', value=1.):
    """Construct labelled synthetic data with the real reader's plot/vector types."""
    from silk.signals.rawfile import RawPlot, RawVector
    return RawPlot({'plotname': name}, [
        RawVector(axis, axis, (), np.array([0., 1.])),
        RawVector('v(out)', 'voltage', (), np.array([value, value])),
        RawVector('onoise_spectrum', 'voltage', (), np.array([value, value])),
        RawVector('@m.xdut.xmm1.msky130[id]', 'current', (), np.array([value, value])),
    ], 2, True, None)


@pytest.mark.parametrize('kind,name,axis', [
    ('tran', 'Transient Analysis', 'time'),
    ('ac', 'AC Analysis', 'frequency'),
    ('dc', 'DC transfer characteristic', 'v(v-sweep)'),
    ('noise', 'Noise Spectral Density Curves', 'frequency'),
    ('op', 'Operating Point', 'v(in)'),
])
@pytest.mark.parametrize('values', [(1., 9.), (9., 1.), (1., 1.)])
@pytest.mark.parametrize('step', [False, True])
def test_duplicate_kind_refuses_even_identical_values_and_xyce_steps(kind, name, axis, values, step):
    from silk.signals.rawfile import RawFile
    plots = [_synthetic_selection_plot(
        f'Step Analysis: Step {i} of 2 params: bias = {v} {name}' if step else name,
        axis, v) for i, v in enumerate(values, 1)]
    # A prefixed op header retains the existing DC classification on this axis.
    selected_kind = 'dc' if step and kind == 'op' else kind
    assert [plot_silk_analysis(p) for p in plots] == [selected_kind, selected_kind]
    with pytest.raises(SimulationError) as exc:
        select_plot(RawFile(plots), selected_kind)
    _assert_duplicate_plot_error(exc.value, selected_kind)


@pytest.mark.parametrize('reverse', [False, True])
def test_spectral_and_integrated_noise_are_two_matches(reverse):
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import NgspiceBackend, RawResult
    plots = [_synthetic_selection_plot('Noise Spectral Density Curves', 'frequency'),
             _synthetic_selection_plot('Integrated Noise', 'onoise_total')]
    if reverse:
        plots.reverse()
    raw = RawFile(plots)
    with pytest.raises(SimulationError) as exc:
        NgspiceBackend().load_noise(RawResult(raw, ''), 'out', analysis='noise')
    _assert_duplicate_plot_error(exc.value, 'noise')


def test_unique_ac_still_reads_beside_duplicate_transients():
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import NgspiceBackend, RawResult
    recorded = read_rawfile(PLOT_SELECTION / 'combined.raw')
    ac = read_rawfile(FIXTURES / 'e_ng_ac.raw').plots[0]
    raw = RawFile([*recorded.plots, ac])
    assert select_plot(raw, 'ac') is ac
    expected = NgspiceSignal.from_rawfile(raw, ac.vectors[1].name, plot=ac)
    actual = NgspiceBackend().load_signal(RawResult(raw, ''), ac.vectors[1].name, analysis='ac')
    np.testing.assert_array_equal(actual.y, expected.y)


@pytest.mark.parametrize('filename', ['combined.raw', 'reversed.raw'])
@pytest.mark.parametrize('route', ['load', 'probe', 'metric', 'branch'])
def test_public_load_probe_and_metric_wrappers_refuse_duplicate_transients(tmp_path, filename, route):
    from silk.circuit.modules import Module
    from silk.circuit.module_test import _evaluate_function_form_metric, _load_metric_path
    from silk.elements import elements
    from silk.exceptions import MetricBindingError
    from silk.metrics import average
    from silk.simulation.backend import RawResult
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'out': 'output'})
    dut.add(elements.resistor('load', p='out', n='0', value=1e3))
    tb = dut.create_test('external')
    tb.add_module_under_test({'out': 'out'})
    tb.add_sim_type(Transient(step=1e-6, stop=1e-5))
    probe = tb.probe('out', analysis='tran')
    tb.add_metric('mean', average, sig=probe)
    raw = read_rawfile(PLOT_SELECTION / filename, lazy=True)
    tb.netlist._raw = RawResult(raw, str(PLOT_SELECTION / filename))
    tb.netlist.simResultsAll = raw
    tb.netlist.sim_results_loaded = True
    if route == 'metric':
        with pytest.raises(MetricBindingError) as exc:
            _evaluate_function_form_metric(tb.netlist, tb._metrics[-1])
        _assert_duplicate_plot_error(exc.value.__cause__, 'tran')
    elif route == 'branch':
        with pytest.raises(SimulationError, match='Accepted spellings for a branch current') as exc:
            _load_metric_path(tb.netlist, 'i(vbias)', analysis='tran')
        _assert_duplicate_plot_error(exc.value.__cause__, 'tran')
    else:
        with pytest.raises(SimulationError) as exc:
            if route == 'load':
                tb.netlist.load_signal('out', analysis='tran')
            else:
                _ = probe.y
        _assert_duplicate_plot_error(exc.value, 'tran')
    assert all(v._data is None for p in raw.plots for v in p.vectors)


@pytest.mark.parametrize('count', [0, 1, 2])
@pytest.mark.parametrize('route', ['backend', 'inventory'])
def test_op_inventory_distinguishes_absent_unique_and_duplicate_plots(tmp_path, count, route):
    from silk import load_pdk
    from silk.circuit.modules import Module
    from silk.circuit.modules_results import capture_op_device_records
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import NgspiceBackend, RawResult
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.add(pdk.nmos1p8('M1', d='out', g='in', s='0', b='0', w=1., l=.5))
    bench = Module('sky130', str(tmp_path), 'bench.cir', pdk_details=pdk)
    bench.add_module(dut, 'dut', nets={})
    plots = [_synthetic_selection_plot('Operating Point') for _ in range(count)]
    raw = RawResult(RawFile(plots or [_synthetic_selection_plot('Transient Analysis')]), '')
    backend = NgspiceBackend()
    def read():
        if route == 'backend':
            return backend.device_op_vectors(raw, 'xdut', 'xmm1')
        return capture_op_device_records(bench, backend, raw, pdk)

    if count == 2:
        with pytest.raises(SimulationError) as exc:
            read()
        _assert_duplicate_plot_error(exc.value, 'op')
    elif route == 'backend':
        values = read()
        if count == 0:
            assert values == {}
        else:
            assert values['id'] == 1.
    else:
        records = read()
        assert len(records) == 1
        if count == 0:
            assert all(np.isnan(v) for v in records[0]['op_values'].values())
        else:
            assert records[0]['op_values']['id'] == 1.


@pytest.mark.parametrize('filename', ['combined.raw', 'reversed.raw'])
def test_duplicate_transients_remain_indexed_in_viewer(filename):
    from silk.signals.view.emit import emit_view_data, pick_default_plot
    payload = emit_view_data(str(PLOT_SELECTION / filename))
    assert [p['name'] for p in payload['plots']] == ['Transient Analysis'] * 2
    assert payload['default_plot'] == pick_default_plot(payload['plots']) == 1
    # Plot-local signals keep separate, indexed identities even with equal headers.
    assert all(p['signals'] for p in payload['plots'])


@pytest.mark.parametrize('filename,kind', [
    ('e_ng_tran.raw', 'tran'), ('e_ng_ac.raw', 'ac'),
    ('e_ng_dc.raw', 'dc'), ('e_ng_noise.raw', 'noise'), ('e_ng_op.raw', 'op'),
])
@pytest.mark.parametrize('lazy', [False, True])
def test_concatenated_real_reader_plots_refuse_without_materializing(tmp_path, filename, kind, lazy):
    # Repeated recorded output remains ambiguous even with identical values.
    content = (FIXTURES / filename).read_bytes()
    path = tmp_path / 'repeated.raw'
    path.write_bytes(content + content)
    raw = read_rawfile(path, lazy=lazy)
    assert len(raw.plots) == 2
    with pytest.raises(SimulationError) as exc:
        select_plot(raw, kind)
    _assert_duplicate_plot_error(exc.value, kind)
    if lazy:
        assert all(v._data is None for p in raw.plots for v in p.vectors)


def _xyce_device_step(index, gm):
    """Synthetic Xyce reader data: two solves with a printed device column."""
    from silk.signals.rawfile import RawPlot, RawVector
    return RawPlot(
        {'plotname': f'Step Analysis: Step {index} of 2 params: bias = {index} DC transfer characteristic'},
        [RawVector('sweep', 'voltage', (), np.array([0., 1.])),
         RawVector('N(XDUT:XMM1:gm)', 'conductance', (), np.array([gm, gm]))],
        2, True, None)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('route', ['probe', 'inventory'])
def test_xyce_device_steps_refuse_both_orders(reverse, route):
    from types import SimpleNamespace
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import RawResult, XyceBackend
    plots = [_xyce_device_step(1, .001), _xyce_device_step(2, .009)]
    if reverse:
        plots.reverse()
    raw = RawResult(RawFile(plots), '')
    address = SimpleNamespace(instance_path='xdut', leaf='XMM1',
                              designer_name='M1', polarity='n', model='test')
    with pytest.raises(SimulationError) as exc:
        if route == 'probe':
            XyceBackend().load_device_probe(raw, address, 'gm', analysis='dc')
        else:
            XyceBackend().device_op_vectors(raw, 'xdut', 'XMM1')
    _assert_duplicate_plot_error(exc.value, 'dc')


@pytest.mark.parametrize('kind,name,axis', [
    ('dc', 'DC transfer characteristic', 'sweep'),
    ('tran', 'Transient Analysis', 'TIME'),
])
@pytest.mark.parametrize('explicit', [False, True])
def test_xyce_single_plot_device_reads_still_work(kind, name, axis, explicit):
    from types import SimpleNamespace
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import RawResult, XyceBackend
    plot = _xyce_device_step(1, .001)
    plot.header['plotname'] = name
    plot.vectors[0].name = axis
    raw = RawResult(RawFile([plot]), '')
    address = SimpleNamespace(instance_path='xdut', leaf='XMM1',
                              designer_name='M1', polarity='n', model='test')
    signal = XyceBackend().load_device_probe(raw, address, 'gm',
                                            analysis=kind if explicit else None)
    np.testing.assert_allclose(signal.y, [.001, .001])
    assert signal.name == 'M1:gm'
    if kind == 'dc':
        assert XyceBackend().device_op_vectors(raw, 'xdut', 'XMM1')['gm'] == .001


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('route', ['load', 'probe'])
def test_public_pending_xyce_device_probe_refuses_steps(tmp_path, reverse, route):
    from silk import load_pdk
    from silk.circuit.modules import Module
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import RawResult
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'d': 'output', 'g': 'input'})
    dut.add(pdk.nmos1p8('M1', d='d', g='g', s='0', b='0', w=1., l=.5))
    dut.simulator = 'xyce'
    tb = dut.create_test('external')
    tb.add_module_under_test({'d': 'out', 'g': 'in'})
    tb.netlist.simulator = 'xyce'
    tb.add_sim_type(DC(source='bias', start=0, stop=1, step=1))
    probe = tb.probe('M1:gm', analysis='dc')
    tb.netlist._pending_device_probes = {('M1', 'gm')}
    plots = [_xyce_device_step(1, .001), _xyce_device_step(2, .009)]
    if reverse:
        plots.reverse()
    tb.netlist.sim_results_loaded = True
    tb.netlist._raw = RawResult(RawFile(plots), '')
    with pytest.raises(SimulationError) as exc:
        if route == 'load':
            tb.netlist.load_signal('M1:gm', analysis='dc')
        else:
            _ = probe.y
    _assert_duplicate_plot_error(exc.value, 'dc')
    # The same public pending request still reads a single plot.
    tb.netlist._raw = RawResult(RawFile([plots[0]]), '')
    expected = .009 if reverse else .001
    signal = tb.netlist.load_signal('M1:gm', analysis='dc') if route == 'load' else probe
    np.testing.assert_allclose(signal.y, [expected, expected])


def test_ngspice_empty_rawfile_has_no_op_vectors():
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import NgspiceBackend, RawResult
    assert NgspiceBackend().device_op_vectors(RawResult(RawFile([]), ''), 'xdut', 'XMM1') == {}


def test_xyce_device_reads_use_unique_selected_plot_for_data_and_diagnostics():
    from types import SimpleNamespace
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import RawResult, XyceBackend
    dc = _xyce_device_step(1, .001)
    tran = _xyce_device_step(2, .009)
    tran.header['plotname'] = 'Transient Analysis'
    tran.vectors[0].name = 'TIME'
    raw = RawResult(RawFile([dc, tran]), '')
    address = SimpleNamespace(instance_path='xdut', leaf='XMM1',
                              designer_name='M1', polarity='n', model='test')
    backend = XyceBackend()
    np.testing.assert_allclose(backend.load_device_probe(raw, address, 'gm', analysis='dc').y,
                               [.001, .001])
    assert backend.device_op_vectors(raw, 'xdut', 'XMM1')['gm'] == .001
    tran.vectors[1].name = 'N(XDUT:XMM1:gds)'
    with pytest.raises(SimulationError, match=r"This run DID provide \['gm'\]"):
        backend.load_device_probe(raw, address, 'vgs', analysis='dc')


@pytest.mark.parametrize('empty', [False, True])
def test_xyce_absent_dc_plot_has_no_op_vectors(empty):
    from silk.signals.rawfile import RawFile
    from silk.simulation.backend import RawResult, XyceBackend
    plot = _xyce_device_step(1, .001)
    plot.header['plotname'] = 'Transient Analysis'
    plot.vectors[0].name = 'TIME'
    raw = RawResult(RawFile([] if empty else [plot]), '')
    assert XyceBackend().device_op_vectors(raw, 'xdut', 'XMM1') == {}
