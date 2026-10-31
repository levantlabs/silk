"""TASK-195 (T-C) — the signal-kind vocabulary, and what it honestly knows.

Three things are pinned here.

**1. The predicates.** One class per kind, on hand-built Signals: what
each kind accepts, what it rules out, and — for the frequency family —
what it *cannot* decide. The honesty pins matter as much as the
acceptances: ``noise_density`` may never return ``MATCHES`` from a real
sample on the frequency axis, because nothing observable on a Signal
separates a density from a real-valued frequency response (design §11,
owner ruling D6: a Signal carries no stamped kind).

**2. produced_by is true.** The one AUTHORED field in schema v2 claims
that a given silk analysis yields a given kind. That claim is checked
against REAL rawfiles from REAL simulator runs — the gen_e fixtures
(``tests/fixtures/rawfiles/gen_e.py``, ngspice AND Xyce), loaded through
the real path, the same corpus ``tests/metrics/test_rawfile_realism.py``
uses. The check is driven off :data:`SIGNAL_KINDS` itself, so a kind
added to the table without a reachable analysis fails here rather than
shipping an unverifiable claim.

**3. It is still a leaf.** T-C changes no schema and no metric: nothing
in silk imports the module yet, and the module imports nothing from silk
(the ``silk.signals`` ↔ ``silk.metrics`` cycle, ``_guards``' rule).

No simulator and no PDK are needed: every fixture is committed.
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests._silk_sources import silk_py_files

from silk.signals.base import Signal
from silk.signals.kinds import (
    ANY, CODE_TRANSITIONS, FREQUENCY_RESPONSE, NOISE_DENSITY,
    POWER_SWEEP, REFERENCE_WAVEFORM, SIGNAL_KINDS,
    SILK_ANALYSES,
    TRANSIENT, KindMatch, SignalKind, _check_kind, signal_kinds_meta,
)
from silk.signals.rawfile import read_rawfile
from silk.signals.spice import NgspiceSignal
from silk.simulation.backend import RawResult, SimulatorBackend

_REPO = Path(__file__).resolve().parents[1]
FIXTURES = _REPO / 'tests' / 'fixtures' / 'rawfiles'
BACKENDS = ('ngspice', 'xyce')
_PREFIX = {'ngspice': 'e_ng', 'xyce': 'e_xy'}


# ---------------------------------------------------------------------------
# Signals: hand-built, and real
# ---------------------------------------------------------------------------

def transient_sig() -> Signal:
    t = np.linspace(0, 1e-6, 64)
    return Signal('v(out)', 'voltage', 'time', 'v(out)', t, np.sin(2e7 * t))


def ac_sig() -> Signal:
    f = np.logspace(1, 6, 64)
    return Signal('v(out)', 'voltage', 'frequency', 'v(out)', f,
                  100.0 / (1 + 1j * f / 1e3))


def real_frequency_sig() -> Signal:
    """Real samples on a frequency axis — a density, a sideband and a
    ``group_delay`` output are all this shape. That is the point."""
    f = np.logspace(1, 6, 64)
    return Signal('noise_out', 'noise', 'frequency', 'noise_out', f,
                  1e-9 / np.sqrt(f))


def dc_sweep_sig() -> Signal:
    v = np.linspace(0, 1.8, 64)
    return Signal('v(out)', 'voltage', 'v(v-sweep)', 'v(out)', v, np.tanh(v))


def code_sig() -> Signal:
    c = np.arange(1, 8, dtype=float)
    return Signal('codes', 'voltage', 'code', 'transition', c, c * 0.125)


def power_sweep_sig() -> Signal:
    p_in = np.linspace(-30.0, 0.0, 31)
    return Signal('pout', 'power', 'power', 'p_out', p_in, p_in + 20.0)


def load(backend: str, family: str, name: str = 'out') -> Signal:
    """The real load path: reader → ``NgspiceSignal.from_rawfile``."""
    return NgspiceSignal.from_rawfile(
        read_rawfile(FIXTURES / f'{_PREFIX[backend]}_{family}.raw'), name, 'V')


def load_noise(backend: str) -> Signal:
    """The real noise path: ``SimulatorBackend.load_signal(raw, 'noise_out')``."""
    be = SimulatorBackend.get(backend)
    if backend == 'ngspice':
        raw = be.load_raw(FIXTURES / 'e_ng_noise.raw')
    else:
        # Xyce noise is CSV-only; the backend resolves noise.csv from here.
        raw = RawResult(native=None,
                        source_path=str(FIXTURES / 'e_xy_noise' / 'rawfile.raw'))
    return be.load_signal(raw, 'noise_out')


#: One real Signal per silk analysis, per backend. Keyed by the analysis
#: names ``produced_by`` uses, which is what makes the reachability test
#: below a check of the authored table rather than of a parallel list.
def real_signal(backend: str, analysis: str) -> Signal:
    if analysis == 'tran':
        return load(backend, 'tran')
    if analysis == 'ac':
        return load(backend, 'ac')
    if analysis == 'dc':
        return load(backend, 'dc')
    if analysis == 'noise':
        return load_noise(backend)
    if analysis == 'op':
        return load(backend, 'op', name='mid')
    raise AssertionError(f'no fixture for analysis {analysis!r}')


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

class TestVocabulary:

    def test_the_vocabulary_is_the_design_table(self):
        # The vocabulary is OPEN — adding a kind is a deliberate one-line
        # edit HERE and in the produced_by pin below, and nowhere else:
        # nothing in the module or these tests is written against its size.
        assert set(SIGNAL_KINDS) == {
            'any', 'transient', 'frequency_response', 'noise_density',
            'code_transitions', 'power_sweep',
            # Added by TASK-196 (T-D) on the reviewer's finding (2): the
            # ``goal`` slot of ``error_norm`` is a waveform the USER writes
            # down, not one a probe yields, and annotating it 'transient' or
            # 'any' would let a consumer wire two probes into it. The
            # deliberate one-line edit the module docstring describes.
            'reference_waveform',
        }

    def test_produced_by_is_the_design_table(self):
        # Producer mapping exported in the catalog; see docs/schema-export.md.
        # REVIEWER BINDING (2026-09-27, TASK-330.2, commit 178250ed): 'op'
        # must NOT join ANY's producers, even though an Op() run's node
        # voltages/source currents are single-point Signals like any other
        # -- widening produced_by widens the digest-pinned exported schema.
        assert {n: k.produced_by for n, k in SIGNAL_KINDS.items()} == {
            'any': ('tran', 'ac', 'dc', 'noise'),
            'transient': ('tran',),
            'frequency_response': ('ac',),
            'noise_density': ('noise',),
            'code_transitions': (),
            'power_sweep': (),
            'reference_waveform': (),
        }

    def test_analysis_vocabulary_is_silks_own(self):
        # OWNER RULING: the @metric(domain=...) values. REVIEWER BINDING
        # (2026-09-27, TASK-330.2, commit 178250ed, amended same day):
        # 'op' (analysis.Op) is deliberately NOT a member -- it is a
        # real, selectable silk analysis, but is kept out of the
        # published metric-schema vocabulary so it can never become a
        # legal SignalKind.produced_by / @metric(domain=...) value by
        # construction. It is selectable purely on the simulation side
        # (silk.signals.analysis_kind's SELECTABLE_ANALYSES, NGSPICE_PLOT_NAMES) --
        # see tests/test_multi_analysis_plots.py's agreement test.
        assert SILK_ANALYSES == ('tran', 'ac', 'dc', 'noise')

    def test_every_produced_by_entry_is_a_silk_analysis(self):
        for kind in SIGNAL_KINDS.values():
            assert set(kind.produced_by) <= set(SILK_ANALYSES), kind.name

    def test_a_kind_cannot_be_built_on_a_foreign_analysis_name(self):
        # 'transient' is silk.signals.analysis_kind's word, not a domain.
        with pytest.raises(ValueError, match='not silk analysis vocabulary'):
            SignalKind(name='bogus', produced_by=('transient',),
                       expectation='—', predicate=lambda sig: KindMatch.MATCHES)

    def test_meta_shape_is_json_safe(self):
        import json
        meta = signal_kinds_meta()
        assert meta['code_transitions'] == {'produced_by': []}
        assert meta['transient'] == {'produced_by': ['tran']}
        assert json.loads(json.dumps(meta)) == meta

    def test_kinds_are_frozen(self):
        with pytest.raises(Exception):
            TRANSIENT.name = 'something else'


# ---------------------------------------------------------------------------
# Coverage that follows the table
# ---------------------------------------------------------------------------

SAMPLES = {
    'transient': transient_sig(),
    'ac': ac_sig(),
    'real frequency': real_frequency_sig(),
    'dc sweep': dc_sweep_sig(),
    'code': code_sig(),
    'power sweep': power_sweep_sig(),
}


class TestEveryKindInTheTable:
    """Driven off :data:`SIGNAL_KINDS`, so a kind added tomorrow is
    exercised without anyone remembering to add a class for it."""

    @pytest.mark.parametrize('kind_name', sorted(SIGNAL_KINDS))
    @pytest.mark.parametrize('sample', sorted(SAMPLES))
    def test_the_predicate_answers_with_a_KindMatch(self, kind_name, sample):
        assert isinstance(SIGNAL_KINDS[kind_name].check(SAMPLES[sample]),
                          KindMatch)

    @pytest.mark.parametrize('kind_name', sorted(SIGNAL_KINDS))
    def test_the_predicate_is_total_and_stable(self, kind_name):
        # No sample may raise, and the answer may not depend on when it is
        # asked — a predicate that reads anything but the Signal would show
        # up here.
        kind = SIGNAL_KINDS[kind_name]
        for sig in SAMPLES.values():
            assert kind.check(sig) is kind.check(sig)

    @pytest.mark.parametrize('kind_name', sorted(SIGNAL_KINDS))
    def test_every_kind_states_what_it_expects(self, kind_name):
        # The expectation string is what a refusal message quotes.
        assert SIGNAL_KINDS[kind_name].expectation.strip()

    @pytest.mark.parametrize('kind_name', sorted(SIGNAL_KINDS))
    def test_at_least_one_sample_is_ruled_out_unless_the_kind_is_any(
            self, kind_name):
        # A kind that accepts everything is not a kind. 'any' is the one
        # deliberate exception, and it is exempted BY ITS OWN predicate
        # rather than by name-matching a fixed vocabulary.
        results = {SIGNAL_KINDS[kind_name].check(s) for s in SAMPLES.values()}
        if results == {KindMatch.MATCHES}:
            assert SIGNAL_KINDS[kind_name].produced_by == tuple(SILK_ANALYSES)


# ---------------------------------------------------------------------------
# One class per predicate
# ---------------------------------------------------------------------------

class TestAnyKind:

    @pytest.mark.parametrize('sig', [transient_sig(), ac_sig(), dc_sweep_sig(),
                                     real_frequency_sig(), code_sig(),
                                     power_sweep_sig()])
    def test_every_signal_is_of_kind_any(self, sig):
        assert ANY.check(sig) is KindMatch.MATCHES

    def test_even_an_empty_signal(self):
        # Emptiness is _check_nonempty's business (TASK-77), not a kind.
        empty = Signal('e', 'voltage', 'time', 'e', np.array([]), np.array([]))
        assert ANY.check(empty) is KindMatch.MATCHES


class TestTransientKind:

    def test_a_time_axis_matches(self):
        assert TRANSIENT.check(transient_sig()) is KindMatch.MATCHES

    @pytest.mark.parametrize('sig', [ac_sig(), real_frequency_sig(),
                                     dc_sweep_sig(), code_sig(),
                                     power_sweep_sig()])
    def test_any_other_axis_conflicts(self, sig):
        assert TRANSIENT.check(sig) is KindMatch.CONFLICTS

    def test_the_label_is_compared_exactly_as_the_existing_guard_does(self):
        # from_rawfile lower-cases the label (TASK-87), so an upper-case
        # 'TIME' is a hand-built Signal and silk.metrics.waveform's own
        # guard rejects it today. The kind must not be laxer.
        from silk.metrics.waveform import _check_transient as guard
        shouty = Signal('v(out)', 'voltage', 'TIME', 'v(out)',
                        np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert TRANSIENT.check(shouty) is KindMatch.CONFLICTS
        with pytest.raises(ValueError):
            guard(shouty, 'rise_time')


class TestFrequencyResponseKind:

    def test_complex_samples_settle_it(self):
        assert FREQUENCY_RESPONSE.check(ac_sig()) is KindMatch.MATCHES

    def test_real_samples_on_a_frequency_axis_are_undecidable(self):
        # psrr_db and group_delay return REAL frequency Signals, and a
        # noise density is real on the same axis (design §11).
        assert (FREQUENCY_RESPONSE.check(real_frequency_sig())
                is KindMatch.UNDECIDABLE)

    @pytest.mark.parametrize('sig', [transient_sig(), dc_sweep_sig(),
                                     code_sig(), power_sweep_sig()])
    def test_a_non_frequency_axis_conflicts(self, sig):
        assert FREQUENCY_RESPONSE.check(sig) is KindMatch.CONFLICTS


class TestNoiseDensityKind:

    def test_real_samples_on_a_frequency_axis_are_undecidable(self):
        assert NOISE_DENSITY.check(real_frequency_sig()) is KindMatch.UNDECIDABLE

    def test_complex_samples_conflict(self):
        # A density is a real magnitude; load_noise loads float64.
        assert NOISE_DENSITY.check(ac_sig()) is KindMatch.CONFLICTS

    @pytest.mark.parametrize('sig', [transient_sig(), dc_sweep_sig(),
                                     code_sig(), power_sweep_sig()])
    def test_a_non_frequency_axis_conflicts(self, sig):
        assert NOISE_DENSITY.check(sig) is KindMatch.CONFLICTS

    def test_it_never_matches_a_density_it_cannot_tell_from_a_response(self):
        # The honest limit: NOISE_DENSITY cannot confirm even the real
        # thing, because FREQUENCY_RESPONSE fits the same observables on a
        # real y (a psrr_db / group_delay output looks identical).
        for backend in BACKENDS:
            sig = load_noise(backend)
            assert NOISE_DENSITY.check(sig) is KindMatch.UNDECIDABLE
            assert FREQUENCY_RESPONSE.check(sig) is KindMatch.UNDECIDABLE


class TestCodeTransitionsKind:

    def test_a_code_axis_matches(self):
        assert CODE_TRANSITIONS.check(code_sig()) is KindMatch.MATCHES

    @pytest.mark.parametrize('sig', [transient_sig(), ac_sig(), dc_sweep_sig(),
                                     real_frequency_sig(), power_sweep_sig()])
    def test_any_other_axis_conflicts(self, sig):
        assert CODE_TRANSITIONS.check(sig) is KindMatch.CONFLICTS

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_the_predicate_matches_what_silk_actually_derives(self, backend):
        # produced_by is EMPTY for this kind — no ANALYSIS yields one —
        # but extract_code_transitions does, from two tran captures. Run
        # the real deriver on the real ramp fixture so the predicate is
        # grounded in a producer rather than invented.
        from silk.metrics.converter import extract_code_transitions
        ct = extract_code_transitions(load(backend, 'ramp', 'in'),
                                      load(backend, 'ramp', 'code'))
        assert CODE_TRANSITIONS.check(ct) is KindMatch.MATCHES

    def test_a_consumer_still_cannot_reach_it_from_an_analysis(self):
        assert CODE_TRANSITIONS.produced_by == ()


class TestReferenceWaveformKind:
    """``error_norm``'s ``goal`` slot (TASK-196, reviewer finding 2)."""

    @pytest.mark.parametrize('sig', [transient_sig(), dc_sweep_sig(),
                                     real_frequency_sig(), code_sig(),
                                     power_sweep_sig(),
                                     load('ngspice', 'tran'),
                                     load('xyce', 'tran')])
    def test_no_input_whatsoever_yields_MATCHES(self, sig):
        # A goal lives on whatever axis the measurement lives on, so no
        # observable can confirm one. Same honesty as the sideband.
        assert REFERENCE_WAVEFORM.check(sig) is not KindMatch.MATCHES

    @pytest.mark.parametrize('sig', [transient_sig(), dc_sweep_sig(),
                                     real_frequency_sig(), code_sig(),
                                     power_sweep_sig()])
    def test_a_real_valued_trace_on_any_axis_is_undecidable(self, sig):
        assert REFERENCE_WAVEFORM.check(sig) is KindMatch.UNDECIDABLE

    def test_complex_samples_conflict(self):
        # The published reference-waveform kind stays real-only; broader
        # numerical support in direct calls does not revise schema 2.0.
        assert REFERENCE_WAVEFORM.check(ac_sig()) is KindMatch.CONFLICTS

    def test_frozen_predicate_remains_stricter_than_direct_error_norm(self):
        from silk.metrics.statistics import error_norm
        sig = Signal('measured', 'voltage', 'time', 'V', [0, 1], [1, 1])
        goal = Signal('goal', 'voltage', 'time', 'V', [0, 1], [1 + 2j, 1 + 2j])
        assert REFERENCE_WAVEFORM.check(goal) is KindMatch.CONFLICTS
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            assert error_norm(sig, goal) == 2.0

    def test_a_consumer_cannot_reach_it_from_an_analysis(self):
        # THE POINT of the kind: T-G refuses to offer error_norm because
        # this slot can never be satisfied by binding an analysis output.
        assert REFERENCE_WAVEFORM.produced_by == ()


class TestPowerSweepKind:

    def test_a_power_axis_matches(self):
        assert POWER_SWEEP.check(power_sweep_sig()) is KindMatch.MATCHES

    @pytest.mark.parametrize('sig', [transient_sig(), ac_sig(), dc_sweep_sig(),
                                     real_frequency_sig(), code_sig()])
    def test_any_other_axis_conflicts(self, sig):
        assert POWER_SWEEP.check(sig) is KindMatch.CONFLICTS

    def test_the_predicate_agrees_with_p1db_s_own_guard(self):
        # Nothing produces a power sweep, so the predicate reflects the
        # input contract p1db already enforces (silk/metrics/rf.py).
        # Pinning them against each other is what stops the two drifting.
        from silk.metrics.rf import p1db
        p1db(power_sweep_sig())                    # accepted: no raise
        with pytest.raises(ValueError):
            p1db(transient_sig())
        assert POWER_SWEEP.check(transient_sig()) is KindMatch.CONFLICTS


# ---------------------------------------------------------------------------
# produced_by, checked against real simulator output
# ---------------------------------------------------------------------------

class TestProducedByIsReachable:
    """AC#2 — every non-empty ``produced_by`` names an analysis that
    really yields that kind."""

    _CASES = [(kind_name, analysis, backend)
              for kind_name, kind in sorted(SIGNAL_KINDS.items())
              for analysis in kind.produced_by
              for backend in BACKENDS]

    def test_the_case_list_covers_every_claim_in_the_table(self):
        # Driven off the table, so a kind added with a produced_by entry
        # cannot skip this check.
        claimed = {(n, a) for n, k in SIGNAL_KINDS.items() for a in k.produced_by}
        assert {(n, a) for n, a, _ in self._CASES} == claimed
        assert claimed, 'no kind claims a producer — the table is not loaded'

    @pytest.mark.parametrize('kind_name,analysis,backend', _CASES)
    def test_a_real_signal_from_that_analysis_is_not_refused(
            self, kind_name, analysis, backend):
        sig = real_signal(backend, analysis)
        result = SIGNAL_KINDS[kind_name].check(sig)
        assert result is not KindMatch.CONFLICTS, (
            f'{kind_name}.produced_by claims {analysis!r}, but a real '
            f'{backend} {analysis} Signal is ruled OUT of the kind')

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_the_decidable_claims_are_confirmed_outright(self, backend):
        # Where the vocabulary CAN decide, "not refused" is not enough:
        # the real analysis output must MATCH.
        assert TRANSIENT.check(real_signal(backend, 'tran')) is KindMatch.MATCHES
        assert (FREQUENCY_RESPONSE.check(real_signal(backend, 'ac'))
                is KindMatch.MATCHES)
        for analysis in SILK_ANALYSES:
            assert ANY.check(real_signal(backend, analysis)) is KindMatch.MATCHES

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_the_noise_claim_is_as_strong_as_it_can_honestly_be(self, backend):
        # noise_density.produced_by == ('noise',) is TRUE, and the check
        # still cannot confirm it: the same observables fit a sideband.
        # Recorded here so the weaker assertion above is not mistaken for
        # an oversight.
        assert (NOISE_DENSITY.check(real_signal(backend, 'noise'))
                is KindMatch.UNDECIDABLE)

    @pytest.mark.parametrize('backend', BACKENDS)
    def test_a_real_signal_is_refused_by_the_kinds_that_do_not_claim_it(
            self, backend):
        # The other half of the claim: a tran Signal is ruled out of every
        # kind that does not claim 'tran'. The exemption is DERIVED from the
        # table, so a kind added later — a transient sub-kind, say — exempts
        # itself instead of failing a hard-coded pair.
        tran = real_signal(backend, 'tran')
        for name, kind in SIGNAL_KINDS.items():
            if 'tran' in kind.produced_by:
                continue
            result = kind.check(tran)
            assert result is not KindMatch.MATCHES, name
            # Every kind in the table but one is pinned to a specific
            # x-axis, and a tran Signal is not on it. 'reference_waveform'
            # (TASK-196) is the deliberate exception, named rather than
            # derived so that adding another axis-agnostic kind is a
            # visible edit here: a GOAL waveform lives on the
            # MEASUREMENT's axis, so a time-domain goal is the common case
            # and refusing it would be wrong. What keeps it unwireable is
            # its empty produced_by, not its predicate.
            if name != 'reference_waveform':
                assert result is KindMatch.CONFLICTS, name


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

class TestCheckKind:

    def test_it_raises_only_on_a_conflict_and_names_the_metric(self):
        with pytest.raises(ValueError) as exc:
            _check_kind(ac_sig(), TRANSIENT, 'propagation_delay')
        msg = str(exc.value)
        assert 'propagation_delay' in msg
        assert 'transient' in msg
        assert "xlabel='frequency'" in msg
        assert 'complex y' in msg

    def test_it_blames_the_named_slot(self):
        with pytest.raises(ValueError, match='sig_out'):
            _check_kind(ac_sig(), TRANSIENT, 'propagation_delay',
                        slot='sig_out')

    def test_it_returns_MATCHES_when_the_kind_is_established(self):
        assert (_check_kind(transient_sig(), TRANSIENT, 'rise_time')
                is KindMatch.MATCHES)

    def test_it_accepts_an_undecidable_input_and_says_so(self):
        # It must NOT raise — refusing here rejects a legitimate density
        # on a guess — and the return value tells a caller that nothing
        # was actually verified.
        result = _check_kind(real_frequency_sig(), FREQUENCY_RESPONSE,
                             'gain_at_freq')
        assert result is KindMatch.UNDECIDABLE

    def test_the_noise_density_guard_still_catches_the_one_it_can(self):
        # design §3.4: a transient handed to a noise-domain metric is
        # ruled out on the axis alone. A complex AC sweep is the other
        # decidable row (test_it_raises_only_on_a_conflict_and_names_the_metric
        # above); a real frequency-domain Signal is the undecidable one.
        with pytest.raises(ValueError):
            _check_kind(transient_sig(), NOISE_DENSITY, 'one_over_f_corner')


# ---------------------------------------------------------------------------
# AC#3 — still a leaf
# ---------------------------------------------------------------------------

class TestStillALeaf:

    #: Every silk module allowed to import this one. T-C pinned this EMPTY
    #: ("changes no schema and no metric"). T-D (TASK-196) is the first
    #: consumer, and it enters through exactly one door: the
    #: ``Annotated[Signal, KIND]`` aliases in ``silk/signals/schema.py``,
    #: which is where kinds.py's own docstring said they should go. The
    #: metric modules import the ALIASES, never the kinds module, so the
    #: silk.signals <-> silk.metrics cycle gains no new edge. Widening this
    #: set is a deliberate edit, and one that has to re-answer the cycle
    #: question.
    #:
    #: WIDENED by T-E (TASK-197) — the deliberate edit, with the cycle
    #: question re-answered. T-E tightens three runtime guards to kind
    #: CHECKS, and a check is not an annotation: ``_check_kind`` and the
    #: two kind objects it is called with have no alias form in
    #: ``silk/signals/schema.py``, and giving them one would route a guard
    #: through the schema module purely to preserve a rule about doors.
    #: The cycle is unaffected, for the reason ``_guards.py`` states about
    #: itself: ``kinds.py`` imports NOTHING from silk (pinned by
    #: ``test_it_imports_nothing_from_silk`` right below), so it cannot
    #: re-enter either half and is fully executed before any importer
    #: touches it. ``silk/metrics/noise.py`` also already imports
    #: ``silk.signals.base`` and ``silk.signals.schema``, and the latter
    #: imports ``kinds`` itself, so by the time the new import runs the
    #: module is in ``sys.modules`` — no edge is added that the existing
    #: two do not already force.
    MAY_IMPORT_KINDS = [
        # TASK-215.7: the evaluation loop catches SignalKindError to
        # re-raise it as MetricBindingError (design §9.2).
        'silk/circuit/module_test.py',
        # TASK-215.7: _check_freq's wrong-axis refusal IS a kind
        # statement, so it raises the subclass.
        'silk/metrics/_guards.py',
        'silk/metrics/noise.py',
        'silk/signals/schema.py',
    ]

    def test_only_the_sanctioned_module_imports_it(self):
        importers = []
        for path in silk_py_files():
            if path.name == 'kinds.py' and path.parent.name == 'signals':
                continue
            if 'signals.kinds' in path.read_text() or 'signals import kinds' in path.read_text():
                importers.append(str(path.relative_to(_REPO)))
        assert importers == self.MAY_IMPORT_KINDS

    def test_it_imports_nothing_from_silk(self):
        # The _guards.py rule: a module either half of the
        # silk.signals <-> silk.metrics cycle may need must not import
        # back into either.
        src = (_REPO / 'silk' / 'signals' / 'kinds.py').read_text()
        tree = ast.parse(src)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert [m for m in imported if m.split('.')[0] == 'silk'] == []

    def test_it_imports_without_silk_signals_being_importable_first(self):
        # Proof the leaf property is real: import the module by file,
        # with no silk package import in the way.
        import importlib.util
        import sys
        name = '_kinds_leaf_probe'
        spec = importlib.util.spec_from_file_location(
            name, _REPO / 'silk' / 'signals' / 'kinds.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod          # @dataclass resolves annotations here
        try:
            spec.loader.exec_module(mod)
            assert mod.TRANSIENT.produced_by == ('tran',)
        finally:
            del sys.modules[name]
