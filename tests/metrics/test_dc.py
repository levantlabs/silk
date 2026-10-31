"""Tests for silk.metrics.converter and silk.metrics.rf — INL/DNL/monotonicity/missing-codes,
extract_code_transitions helper, and P1dB.
"""
import math
import numpy as np
import pytest

from silk.signals.base import Signal
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.converter import (
    inl,
    dnl,
    is_monotonic,
    missing_codes,
    extract_code_transitions,
    INL,
    DNL,
)
from silk.metrics.rf import (
    p1db,
)


def _power_sweep(p_in_dbm, p_out_dbm):
    return Signal('psweep', 'power', 'power', 'dBm',
                  np.asarray(p_in_dbm, dtype=float),
                  np.asarray(p_out_dbm, dtype=float))


def _code_signal(transitions):
    t = np.asarray(transitions, dtype=float)
    return Signal('codes', 'voltage', 'code', 'V',
                  np.arange(len(t), dtype=float), t)


def _sparse_code_signal(codes, transitions):
    return Signal('codes', 'voltage', 'code', 'V',
                  np.asarray(codes, dtype=float),
                  np.asarray(transitions, dtype=float))


class TestInl:
    def test_perfect(self):
        result = inl(_code_signal(np.linspace(0.0, 16.0, 17)))
        assert isinstance(result, INL)
        assert result.max_abs_lsb < 1e-12

    def test_endpoint(self):
        result = inl(_code_signal([0.0, 1.1, 2.0, 3.0]), method='endpoint')
        assert math.isclose(result.inl_lsb[1], 0.1, abs_tol=1e-12)

    def test_best_fit(self):
        result = inl(_code_signal([0.0, 1.0, 2.0, 3.0]), method='best_fit')
        assert result.method == 'best_fit'
        assert result.max_abs_lsb < 1e-12

    def test_bad_method(self):
        with pytest.raises(ValueError, match="must be 'endpoint' or 'best_fit'"):
            inl(_code_signal([0.0, 1.0, 2.0]), method='foo')

    def test_constant_curve(self):
        with pytest.warns(MetricWarning, match='LSB is zero'):
            result = inl(_code_signal([1.0, 1.0, 1.0]))
        assert math.isnan(result.max_abs_lsb)
        assert np.all(np.isnan(result.inl_lsb))

    def test_rejects_non_code(self):
        sig = Signal('s', 'voltage', 'time', 'V',
                     np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="xlabel='code'"):
            inl(sig)

    def test_endpoint_non_consecutive_codes_is_perfect(self):
        # TASK-315 D1: a perfect converter sampled at codes 0,2,4,6 with
        # voltages 0,2,4,6 must read all-zero INL, not 0/-1/-2/-3 LSB
        # (the old bug used a per-SAMPLE LSB of 2.0 instead of the
        # per-CODE LSB of 1.0 implied by the code span).
        sig = _sparse_code_signal([0, 2, 4, 6], [0.0, 2.0, 4.0, 6.0])
        result = inl(sig, method='endpoint')
        assert np.allclose(result.inl_lsb, 0.0, atol=1e-12)
        assert result.max_abs_lsb < 1e-12

    def test_endpoint_non_uniform_gap_codes(self):
        # Codes 0, 1, 3, 4 (an uneven gap) with a perfectly linear
        # transfer curve must still read all-zero endpoint INL.
        sig = _sparse_code_signal([0, 1, 3, 4], [0.0, 1.0, 3.0, 4.0])
        result = inl(sig, method='endpoint')
        assert np.allclose(result.inl_lsb, 0.0, atol=1e-12)


class TestDnl:
    def test_perfect(self):
        result = dnl(_code_signal(np.linspace(0.0, 1.0, 17)))
        assert isinstance(result, DNL)
        assert result.max_abs_lsb < 1e-12

    def test_known_step_error(self):
        result = dnl(_code_signal([0.0, 1.5, 2.5, 3.5, 4.0]))
        assert math.isclose(result.dnl_lsb[0],  0.5, abs_tol=1e-12)
        assert math.isclose(result.dnl_lsb[-1], -0.5, abs_tol=1e-12)

    def test_rejects_non_consecutive_codes(self):
        # TASK-315: DNL is per code and needs every code present; a gap
        # is a caller error (decision-1), not degenerate converter data.
        sig = _sparse_code_signal([0, 2, 4, 6], [0.0, 2.0, 4.0, 6.0])
        with pytest.raises(ValueError, match='consecutive integers') as excinfo:
            dnl(sig)
        message = str(excinfo.value)
        assert 'code 0' in message and 'code 2' in message
        assert 'inl' in message.lower()

    def test_rejects_non_consecutive_codes_names_first_gap(self):
        sig = _sparse_code_signal([0, 1, 2, 5, 6], [0.0, 1.0, 2.0, 5.0, 6.0])
        with pytest.raises(ValueError, match='gap of 3') as excinfo:
            dnl(sig)
        assert 'code 2' in str(excinfo.value) and 'code 5' in str(excinfo.value)


class TestIsMonotonic:
    def test_strict(self):
        assert is_monotonic(_code_signal([0.0, 1.0, 2.0, 3.0]))

    def test_repeat(self):
        assert not is_monotonic(_code_signal([0.0, 1.0, 1.0, 2.0]))

    def test_decreasing(self):
        assert not is_monotonic(_code_signal([0.0, 2.0, 1.0]))


class TestMissingCodes:
    def test_none(self):
        assert len(missing_codes(_code_signal([0.0, 1.0, 2.0, 3.0]))) == 0

    def test_one_missing(self):
        # T(1) == T(2): code 1's bin has zero width, so code 1 is missing.
        miss = missing_codes(_code_signal([0.0, 1.0, 1.0, 2.0]))
        assert 1.0 in miss

    def test_rejects_non_consecutive_codes(self):
        # missing_codes is a DNL concept (bin width) and calls dnl
        # internally, so it inherits the same consecutive-codes guard.
        sig = _sparse_code_signal([0, 2, 4, 6], [0.0, 2.0, 4.0, 6.0])
        with pytest.raises(ValueError, match='consecutive integers'):
            missing_codes(sig)


class TestExtractCodeTransitions:
    def test_simple_ramp(self):
        n = 400
        t = np.linspace(0, 1.0, n)
        v = t.copy()
        codes = np.floor(v * 4).clip(0, 3).astype(float)
        v_in_sig = Signal('vin',  'voltage', 'time', 'V', t, v)
        code_sig = Signal('code', 'code',    'time', '',  t, codes)
        result = extract_code_transitions(v_in_sig, code_sig)
        assert result.xlabel == 'code'
        # lowest observed code (0) has no transition (IEEE 1241): only
        # codes 1..3 get a row.
        assert len(result.x) == 3
        assert math.isclose(result.y[0], 0.25, abs_tol=0.01)

    def test_mismatched_axes(self):
        v = Signal('v', 'voltage', 'time', 'V',
                   np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        c = Signal('c', 'code', 'time', '',
                   np.array([0.0, 2.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match='same time axis'):
            extract_code_transitions(v, c)


class TestCodeDip:
    """TASK-398 (catalog review D3): a code that goes back down.

    The first-reach table cannot show a 3 -> 2 -> 3 dip (first-reach levels
    are nondecreasing in k by construction), so the extractor records the
    count of falling code steps in ``code_reversals`` beside the table and
    ``is_monotonic`` / ``non_monotonic_steps`` read it.
    """

    @staticmethod
    def _extract(codes):
        codes = np.asarray(codes, dtype=float)
        t = np.arange(len(codes), dtype=float)
        v = t / (len(codes) - 1)
        return extract_code_transitions(
            Signal('vin', 'voltage', 'time', 'V', t, v),
            Signal('code', 'code', 'time', '', t, codes))

    def test_dip_is_not_monotonic(self):
        from silk.metrics.converter import non_monotonic_steps
        tr = self._extract([0, 1, 2, 3, 2, 3, 4, 5])
        # The table itself is today's first-reach table, unchanged.
        np.testing.assert_allclose(tr.y, np.array([1, 2, 3, 6, 7]) / 7)
        assert tr.code_reversals == 1
        assert is_monotonic(tr) is False
        assert non_monotonic_steps(tr) == 1.0
        # A dip is not a missing code.
        assert len(missing_codes(tr)) == 0

    def test_each_falling_step_counts_once(self):
        from silk.metrics.converter import non_monotonic_steps
        tr = self._extract([0, 1, 2, 3, 1, 2, 3, 2, 4, 5])
        assert tr.code_reversals == 2
        assert non_monotonic_steps(tr) == 2.0

    def test_dip_adds_to_table_steps(self):
        from silk.metrics.converter import non_monotonic_steps
        # Code 2 skipped (zero-width step in the table) AND a 3 -> 1 dip.
        tr = self._extract([0, 1, 3, 1, 3, 4])
        assert tr.code_reversals == 1
        assert non_monotonic_steps(tr) == 2.0

    def test_monotonic_staircase_unchanged(self):
        from silk.metrics.converter import non_monotonic_steps
        n = 400
        t = np.linspace(0, 1.0, n)
        codes = np.floor(t * 8).clip(0, 7)
        tr = extract_code_transitions(
            Signal('vin', 'voltage', 'time', 'V', t, t.copy()),
            Signal('code', 'code', 'time', '', t, codes))
        expected = np.array([t[int(np.argmax(codes >= k))] for k in range(1, 8)])
        np.testing.assert_array_equal(tr.x, np.arange(1, 8, dtype=float))
        np.testing.assert_array_equal(tr.y, expected)
        assert tr.code_reversals == 0
        assert is_monotonic(tr) is True
        assert non_monotonic_steps(tr) == 0.0
        assert len(missing_codes(tr)) == 0
        hand = Signal('codes', 'voltage', 'code', 'V', tr.x.copy(), tr.y.copy())
        np.testing.assert_array_equal(inl(tr).inl_lsb, inl(hand).inl_lsb)
        np.testing.assert_array_equal(dnl(tr).dnl_lsb, dnl(hand).dnl_lsb)

    def test_hand_built_table_has_no_reversals(self):
        from silk.metrics.converter import non_monotonic_steps
        sig = _code_signal([0.0, 1.0, 2.0, 3.0])
        assert not hasattr(sig, 'code_reversals')
        assert is_monotonic(sig) is True
        assert non_monotonic_steps(sig) == 0.0

    def test_nan_code_trace_reversals_unknown(self):
        from silk.metrics.converter import non_monotonic_steps
        tr = self._extract([0, 1, float('nan'), 3, 4])
        assert math.isnan(tr.code_reversals)
        assert is_monotonic(tr) is False
        assert math.isnan(non_monotonic_steps(tr))

    def test_reversals_survive_copy_and_pickle(self):
        import copy
        import pickle
        tr = self._extract([0, 1, 2, 3, 2, 3, 4, 5])
        for clone in (copy.copy(tr), copy.deepcopy(tr),
                      pickle.loads(pickle.dumps(tr))):
            assert clone.code_reversals == 1
            assert is_monotonic(clone) is False


class TestP1db:
    """TASK-180: a Signal in, a float out — ``nan``, never ``None``."""

    def test_no_compression_is_nan_not_none(self):
        """decision-1 §1: a sweep that never compresses is DEGENERATE DATA —
        a legitimate simulation result with no compression point in it — so
        the answer is a number. It used to be ``None``, which does not
        compare, does not arithmetic, and breaks a downstream expression
        with a TypeError far from here."""
        p_in = np.array([-40.0, -30.0, -20.0, -10.0])
        got = p1db(_power_sweep(p_in, p_in + 20.0))
        assert isinstance(got, float)
        assert math.isnan(got)

    def test_compression_at_known(self):
        p_in  = np.array([-40.0, -30.0, -20.0, -10.0,  0.0])
        p_out = np.array([-20.0, -10.0,   0.0,   9.0, 15.0])
        result = p1db(_power_sweep(p_in, p_out))
        assert not math.isnan(result)
        assert -20.0 < result <= -10.0

    def test_compressing_sweep_value_is_unchanged_by_the_signal_move(self):
        """The arithmetic is byte-identical to the array-taking form: gain
        G0 up to p_c then falling at s dB/dB puts P1dB exactly at
        ``p_c + 1/s``, interpolated between the bracketing points."""
        p_in = np.linspace(-30, 10, 41)
        G0, pc, s = 20.0, -5.0, 0.25
        gain = G0 - s * np.clip(p_in - pc, 0, None)
        assert p1db(_power_sweep(p_in, p_in + gain)) == pytest.approx(
            pc + 1 / s, rel=1e-12)

    def test_the_reference_is_the_first_points_gain(self):
        """The small-signal reference is ``p_out[0] - p_in[0]``, not a fit:
        a sweep already compressed at its first point measures the drop
        from there, so P1dB lands later than the true compression point.
        (``g_0 - g_0 == 0``, so p1db's ``k == 0`` branch is unreachable —
        an unreachable branch in this case.)"""
        p_in  = np.array([-10.0, 0.0, 10.0])
        p_out = np.array([  0.0, 5.0,  8.0])
        # gains 10, 5, -2 dB: 1 dB below the 10 dB reference falls between
        # the first two points.
        assert p1db(_power_sweep(p_in, p_out)) == pytest.approx(-8.0)

    def test_nan_sample_propagates_instead_of_biasing_the_answer(self):
        """TASK-78: ``g0 - g_k >= 1.0`` is False for a NaN point, so an
        unguarded NaN was SKIPPED and the walk reported the next
        compressing point — a finite number that reads like a measurement
        but is biased high."""
        p_in  = np.array([-40.0, -30.0, -20.0, -10.0,  0.0])
        p_out = np.array([-20.0, -10.0,   0.0,   9.0, 15.0])
        holed = p_out.copy()
        holed[3] = np.nan
        assert math.isnan(p1db(_power_sweep(p_in, holed)))
        with pytest.raises(ValueError, match='p1db:.*axis.*nonfinite.*needs'):
            p1db(_power_sweep([-40.0, np.nan], [-20.0, 0.0]))

    # -- caller error (decision-1 §2) ------------------------------------

    def test_too_few(self):
        with pytest.raises(ValueError, match='at least 2 sweep points'):
            p1db(_power_sweep([1.0], [1.0]))

    def test_wrong_axis_is_refused(self):
        sig = Signal('t', 'voltage', 'time', 'V',
                     np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match=r"p1db: sig must be a power-sweep"):
            p1db(sig)

    def test_empty_sweep_is_refused(self):
        with pytest.raises(ValueError, match='p1db: Signal .* has no samples'):
            p1db(_power_sweep([], []))

    def test_mismatched_lengths_are_refused(self):
        sig = Signal('psweep', 'power', 'power', 'dBm',
                     np.array([-10.0, 0.0, 10.0]), np.array([0.0, 5.0]))
        with pytest.raises(ValueError, match='must have the same length'):
            p1db(sig)

    def test_raw_arrays_are_no_longer_accepted(self):
        """The array-taking form is GONE (TASK-180), not quietly tolerated:
        two positional arrays no longer fit the signature at all."""
        p_in = np.array([-40.0, -30.0, -20.0, -10.0])
        with pytest.raises(TypeError):
            p1db(p_in, p_in + 20.0)
        with pytest.raises(ValueError, match='p1db: sig must be a power-sweep'):
            p1db(Signal('p', 'power', 'dBm', 'dBm', p_in, p_in + 20.0))


class TestP1dbReachesASpec:
    """AC#3 / the F5 measurement: a nan P1dB must FAIL a spec, not pass it.

    Measured through silk's own :meth:`SpecReport.check`, the instrument
    decision-1 used to rule that ``0.0`` passes and ``nan`` fails.
    """

    def _report(self, value):
        from silk.circuit.spec_report import SpecReport
        return SpecReport('p1db_tb', 'pa', {'p1db': value})

    def test_a_measured_compression_point_passes_and_fails_normally(self):
        p_in  = np.array([-40.0, -30.0, -20.0, -10.0,  0.0])
        p_out = np.array([-20.0, -10.0,   0.0,   9.0, 15.0])
        got = p1db(_power_sweep(p_in, p_out))          # about -11 dBm
        assert self._report(got).check('P1dB > -20 dBm', 'p1db', '>', -20.0).passed
        assert not self._report(got).check('P1dB > 0 dBm', 'p1db', '>', 0.0).passed

    def test_nan_fails_a_spec_in_either_direction(self):
        p_in = np.array([-40.0, -30.0, -20.0, -10.0])
        got = p1db(_power_sweep(p_in, p_in + 20.0))
        assert math.isnan(got)
        for op, target in (('>', -20.0), ('>=', -20.0), ('<', 100.0),
                           ('<=', 100.0), ('==', float('nan'))):
            rpt = self._report(got).check(f'P1dB {op} {target}', 'p1db', op, target)
            assert not rpt.passed, (op, target)
            assert rpt.results[0].passed is False

    def test_nan_also_fails_the_first_class_Spec_path(self):
        """The other consumer: ``Spec.evaluate`` behind
        ``Module.reconcile_specs`` — ``'PASS' if passes else 'FAIL'``, so a
        nan measurement reconciles as FAIL, never as PASS."""
        from silk.specs.spec import Measurement, Spec

        p_in = np.array([-40.0, -30.0, -20.0, -10.0])
        got = p1db(_power_sweep(p_in, p_in + 20.0))
        for op, target in (('>', -20.0), ('<', 100.0), ('>=', -20.0),
                           ('<=', 100.0), ('==', 0.0)):
            spec = Spec(name=f'P1dB {op} {target}', module=None,
                        metric='p1db', op=op, target=target, units='dBm')
            assert spec.evaluate(
                Measurement(source='spice', value=got, timestamp=0.0)) is False

    def test_it_is_in_the_schema_and_still_specable(self):
        """TASK-200 (design §9, owner decision D5a), reversing TASK-184.

        TASK-184 held p1db out of the schema because a schema entry
        promised a consumer that a probe could satisfy the input and none
        can: p1db needs ``xlabel='power'``, which no silk analysis
        produces -- the curve is assembled from a parameter sweep across
        N runs. Schema v2 gives that fact somewhere to live, so the metric
        comes back: the slot declares ``kind='power_sweep'``, whose
        ``produced_by`` is empty, and the refusal rule (asserted in
        tests/test_metric_schema.py) keeps it non-offerable. Specs are
        unaffected either way: ``Spec`` names a metric by string and never
        consults the schema (the preceding tests in this class exercise
        that).
        """
        from silk.signals.schema import export_metric_schema, get_metric
        schema = export_metric_schema()
        assert schema['p1db']['signal_inputs'] == [
            {'name': 'sig', 'kind': 'power_sweep'}]
        assert (schema['p1db']['domain'], schema['p1db']['unit']) == (
            'dc', 'dBm')
        # resolvable by name again, which is what type= resolution uses
        assert get_metric('p1db') is p1db
        # TASK-233 adds the four code-transition scalar wrappers.
        assert {k for k, v in schema.items()
                if k != '_meta' and v['domain'] == 'dc'} == {
                    'p1db', 'inl_max_lsb', 'dnl_max_lsb',
                    'missing_code_count', 'non_monotonic_steps',
                    'offset_error_lsb', 'gain_error_lsb',
                    'full_scale_error_lsb'}

    def test_the_power_sweep_kind_it_declares_has_no_producer(self):
        """Why it is safe to export: the kind itself carries the refusal.

        Not a restatement of the export -- this reads the KIND object the
        annotation carries, which is what a consumer's rule indexes into.
        """
        from silk.signals.kinds import SIGNAL_KINDS
        assert SIGNAL_KINDS['power_sweep'].produced_by == ()


class TestImports:
    def test_top_level(self):
        from silk.metrics import inl, dnl, is_monotonic, missing_codes, p1db
        assert callable(inl)


class TestStaticErrors:
    """TASK-399: offset / gain / full-scale error (decision-1: nan vs raise)."""

    def test_nan_transition_returns_nan_with_warning(self):
        from silk.metrics.converter import static_errors
        sig = _code_signal([0.125, float('nan'), 0.375])
        with pytest.warns(MetricWarning):
            r = static_errors(sig, full_scale=1.0, n_bits=3)
        assert math.isnan(r.offset_error_lsb)
        assert math.isnan(r.gain_error_lsb)
        assert math.isnan(r.full_scale_error_lsb)

    def test_single_transition_raises(self):
        from silk.metrics.converter import static_errors
        with pytest.raises(ValueError, match='static_errors'):
            static_errors(_sparse_code_signal([1], [0.25]), full_scale=1.0, n_bits=3)

    @pytest.mark.parametrize('kw', [
        dict(full_scale=0.0, n_bits=3), dict(full_scale=-1.0, n_bits=3),
        dict(full_scale=float('nan'), n_bits=3),
        dict(full_scale=1.0, n_bits=0), dict(full_scale=1.0, n_bits=2.5),
    ])
    def test_caller_errors_raise(self, kw):
        from silk.metrics.converter import static_errors
        with pytest.raises(ValueError, match='static_errors'):
            static_errors(_code_signal([0.125, 0.25]), **kw)

    def test_wrong_axis_raises(self):
        from silk.metrics.converter import static_errors
        sig = Signal('t', 'voltage', 'time', 'V', np.arange(3.0), np.arange(3.0))
        with pytest.raises(ValueError, match='code-transition'):
            static_errors(sig, full_scale=1.0, n_bits=3)

    def test_v_low_shifts_the_ideal(self):
        from silk.metrics.converter import (
            static_errors,
            offset_error_lsb,
            gain_error_lsb,
            full_scale_error_lsb,
        )
        lsb = 0.125
        k = np.arange(1, 8)
        for v_low in (0.7, -0.5):
            sig = _sparse_code_signal(k, v_low + k * lsb)
            kw = dict(full_scale=1.0, n_bits=3, v_low=v_low)
            r = static_errors(sig, **kw)
            assert (r.offset_error_lsb, r.gain_error_lsb, r.full_scale_error_lsb) == pytest.approx((0, 0, 0), abs=1e-9)
            for fn in (offset_error_lsb, gain_error_lsb, full_scale_error_lsb):
                assert fn(sig, **kw) == pytest.approx(0, abs=1e-9)

    def test_half_lsb_first_transition_convention(self):
        from silk.metrics.converter import static_errors
        lsb = 0.125
        k = np.arange(1, 8)
        sig = _sparse_code_signal(k, (k - 0.5) * lsb)
        r = static_errors(sig, full_scale=1.0, n_bits=3, v_low=-0.5 * lsb)
        assert r.offset_error_lsb == pytest.approx(0.0, abs=1e-9)
        assert static_errors(sig, full_scale=1.0, n_bits=3).offset_error_lsb == pytest.approx(-0.5)

    @pytest.mark.parametrize('v_low', [float('nan'), float('inf'), 1j, True, 'x'])
    def test_bad_v_low_raises(self, v_low):
        from silk.metrics.converter import static_errors
        with pytest.raises(ValueError, match='v_low'):
            static_errors(_code_signal([0.125, 0.25]), full_scale=1.0, n_bits=3, v_low=v_low)


class TestCodeOrderAndRange:
    """TASK-441: strictly ascending integer codes; static_errors range."""

    @staticmethod
    def _sig(codes, volts):
        return _sparse_code_signal(codes, volts)

    def test_ascending_pinned_values_unchanged(self):
        from silk.metrics.converter import static_errors
        k = np.arange(1, 8)
        r = static_errors(self._sig(k, k + 0.25 + 0.05 * k), full_scale=8.0, n_bits=3)
        assert (r.offset_error_lsb, r.gain_error_lsb, r.full_scale_error_lsb) == (
            pytest.approx(0.30), pytest.approx(0.30), pytest.approx(0.60))

    @pytest.mark.parametrize('name', ['inl', 'dnl', 'is_monotonic',
                                      'missing_codes', 'non_monotonic_steps',
                                      'static_errors'])
    def test_descending_codes_raise(self, name):
        import silk.metrics.converter as dc
        k = np.arange(7, 0, -1)
        sig = self._sig(k, k + 0.25)
        kw = dict(full_scale=8.0, n_bits=3) if name == 'static_errors' else {}
        with pytest.raises(ValueError, match='axis.*needs.*strictly increasing'):
            getattr(dc, name)(sig, **kw)

    def test_non_integer_codes_raise(self):
        from silk.metrics.converter import inl
        with pytest.raises(ValueError, match='integers'):
            inl(self._sig([1, 1.5, 2], [1, 2, 3]))

    def test_out_of_range_code_raises(self):
        from silk.metrics.converter import static_errors
        k = np.arange(1, 8)
        with pytest.raises(ValueError, match=r'code 4 .*n_bits=2'):
            static_errors(self._sig(k, k + 0.25), full_scale=4.0, n_bits=2)
        with pytest.raises(ValueError, match='code -1'):
            static_errors(self._sig([-1, 0, 1], [0, 1, 2]), full_scale=4.0, n_bits=2)
