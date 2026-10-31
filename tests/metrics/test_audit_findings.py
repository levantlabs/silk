"""REVIEW-METRICS (TASK-34) — repro per FILED finding.

Every test here reproduces one metric that returns a wrong or misleading
number today. Each is ``xfail(strict=True)`` against the task that owns
the fix: the test FAILS (as expected) while the bug is live and turns
into a hard failure the moment the fix lands, so the owner has to flip
the marker — the repro becomes the regression test. (F1 / TASK-52 has
landed: its class is now the plain regression test.)

Severity and task ids are
in each marker's ``reason``. No behavioural change was made inline.
"""
import math
import warnings

import numpy as np
import pytest

from silk.signals.base import Signal
from silk.metrics import (
    sndr, sinad, enob, snr, sfdr, thd, thd_percent, noise_floor_dbfs,
    harmonic, harmonic_freqs, harmonic_levels_db, mix_product,
    extract_code_transitions, inl, dnl, missing_codes,
    rise_time, fall_time,
    passband_ripple_db, stopband_attenuation_db, on_time,
    gain_bw_product, dynamic_energy_per_cycle,
    crossings, average_duty, argmin_x, argmax_x, off_time,
    hysteresis, spectral_power, value_at,
    settling_time, crossing, average, rms, peak_to_peak, stddev, integral,
)
from silk.metrics import (
    bandwidth_3dB, integrated_noise, dc_gain_dB, cmrr_dB,
)
import silk.metrics as M


def _tran(t, y, name='v'):
    return Signal(name, 'voltage', 'time', 'V',
                  np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def _freq(f, y):
    return Signal('h', 'voltage', 'frequency', 'V',
                  np.asarray(f, dtype=float), np.asarray(y))


TCLK = 1e-9
N = 1024


def _coherent(m=11, amp=1.0, n=N):
    t = np.arange(n) * TCLK
    return _tran(t, amp * np.sin(2 * np.pi * m * np.arange(n) / n))


# ---------------------------------------------------------------------------
# F1 — HIGH: spectral metrics on a dead (DC) signal report infinite quality
# ---------------------------------------------------------------------------

class TestF1DeadSignalReportsInfiniteSndr:
    """A flat-lined ADC output has NO fundamental. ``_power_spectrum``
    used to elect bin 1 (argmax of an all-zero spectrum) and every ratio
    metric divided 0 by 0 and reported ``inf`` — a dead converter passed
    every SNDR/ENOB/SFDR spec.

    FIXED (TASK-52, owner-ruled 2026-09-02): the power spectrum carries
    an eps floor (added after squaring), so the dead-signal answer is a
    finite, always-failing floor artifact — ``sndr = -10·log10(N/2 − 1)``
    dB, N-dependent (−21 dB @ 256, −27 dB @ 1024), SFDR 0 dB, THD +6 dB.
    No exception: campaign / sweep flows always get a number.
    """

    def _dead(self, n=N, level=0.9):
        t = np.arange(n) * TCLK
        return _tran(t, np.full(n, level))

    def test_sndr_of_dc_is_the_finite_floor(self):
        s = self._dead()
        floor = -10.0 * math.log10(N / 2 - 1)
        v = sndr(s, tclk=TCLK, n_samples=N)
        assert math.isfinite(v)
        assert v == pytest.approx(floor, abs=1e-9)
        assert v < 0.0                                   # can never pass a spec

    @pytest.mark.parametrize('n', [256, 1024, 2048])
    def test_floor_is_n_dependent_and_always_failing(self, n):
        s = self._dead(n)
        v = sndr(s, tclk=TCLK, n_samples=n)
        assert v == pytest.approx(-10.0 * math.log10(n / 2 - 1), abs=1e-9)
        assert enob(s, tclk=TCLK, n_samples=n) < 0.0

    def test_every_spectral_metric_is_finite_and_failing(self):
        s = self._dead()
        K = dict(tclk=TCLK, n_samples=N)
        floor = -10.0 * math.log10(N / 2 - 1)
        assert sndr(s, **K) == pytest.approx(floor, abs=1e-9)
        assert sinad(s, **K) == pytest.approx(floor, abs=1e-9)
        assert enob(s, **K) == pytest.approx((floor - 1.76) / 6.02, abs=1e-9)
        # snr excludes the 9 harmonic bins (n_harmonics=10) from the noise
        assert snr(s, **K) == pytest.approx(-10.0 * math.log10(N / 2 - 10), abs=1e-9)
        r = sfdr(s, **K)
        assert r.sfdr_db == pytest.approx(0.0, abs=1e-9)
        assert math.isfinite(r.fundamental_hz) and math.isfinite(r.spur_hz)
        # thd: harmonics 2..5 of the elected bin, 4·eps over eps = +6.02 dB
        assert thd(s, **K) == pytest.approx(10.0 * math.log10(4.0), abs=1e-9)
        assert thd_percent(s, **K) == pytest.approx(200.0, abs=1e-9)
        assert noise_floor_dbfs(s, **K) == pytest.approx(0.0, abs=1e-9)
        lv = harmonic_levels_db(s, **K, n_harmonics=4)
        assert np.all(np.isfinite(lv)) and np.allclose(lv, 0.0)
        assert harmonic(s, 1, **K) == 0j
        # the elected bin is still bin 1; its harmonic grid is finite
        hf = harmonic_freqs(s, **K, n=3)
        assert np.all(np.isfinite(hf))
        assert np.allclose(hf, np.arange(1, 4) / (N * TCLK))

    def test_dead_signal_never_raises(self):
        s = self._dead()
        K = dict(tclk=TCLK, n_samples=N)
        for fn in (sndr, sinad, snr, enob, thd, thd_percent,
                   noise_floor_dbfs, harmonic_levels_db,
                   harmonic_freqs):
            fn(s, **K)                                   # must not raise
        sfdr(s, **K)
        harmonic(s, 2, **K)

    def test_clean_tone_reads_finite_precision_floor(self):
        """Residual bins retain their floor when summed without the signal."""
        s = _coherent()
        for metric in (sndr, snr):
            value = metric(s, tclk=TCLK, n_samples=N)
            assert math.isfinite(value)
            assert 150.0 < value < 200.0
        kw = dict(tclk=TCLK, n_samples=N)
        assert sndr(s, **kw) <= sfdr(s, **kw).sfdr_db

    def test_noisy_sine_matches_unfloored_residual(self):
        s = _coherent()
        s.y = s.y + 1e-3 * np.random.default_rng(7).standard_normal(N)
        x = s.y[:N] - np.mean(s.y[:N])
        P = np.abs(np.fft.fft(x)) ** 2                  # full two-sided oracle
        fb = int(np.argmax(P[1:N // 2]) + 1)
        mask = np.ones(N, dtype=bool)
        mask[[0, fb, N - fb]] = False
        ref = 10.0 * math.log10((P[fb] + P[N - fb]) / np.sum(P[mask]))
        assert sndr(s, tclk=TCLK, n_samples=N) == pytest.approx(ref, abs=1e-9)

    def test_small_real_signal_is_not_eaten_by_the_floor(self):
        s = _coherent(amp=1e-6)
        s.y = s.y + 1e-9 * np.random.default_rng(3).standard_normal(N)
        v = sndr(s, tclk=TCLK, n_samples=N)
        assert 50.0 < v < 60.0


# ---------------------------------------------------------------------------
# F2 — HIGH: extract_code_transitions anchors code 0 at the ramp START
# ---------------------------------------------------------------------------

class TestF2FirstCodeTransitionIsRampStart:
    """The t=0 degenerate-extraction class from the 2026-08 artifact audit.

    FIXED (TASK-53, owner-ruled 2026-09-02): ``extract_code_transitions``
    used to take "first sample where code k appears" as the transition
    INTO code k. For the lowest code that was the first sample of the
    capture — the ramp's starting voltage, not a converter transition —
    so a histogram/ramp test (which always over-ranges the ramp, starting
    below the first real transition) anchored the endpoint line on a fake
    point and biased every INL/DNL value. Per IEEE 1241 transition-level
    semantics, the lowest observed code now gets no row at all; each
    remaining code k's row is the first sample where the output has
    REACHED k, so a skipped code shares its successor's voltage (a
    zero-width step) and is caught by :func:`missing_codes`.
    """

    def _ramp_capture(self, v0=-0.2, v1=1.2, n=2801):
        t = np.linspace(0.0, 1.0, n)
        v = v0 + (v1 - v0) * t
        # Ideal 3-bit ADC: transition levels at k/8, k = 1..7; code 0 below.
        code = np.clip(np.floor(v * 8.0), 0, 7)
        return _tran(t, v), Signal('code', 'code', 'time', '', t, code)

    def test_ideal_adc_has_zero_inl_on_overranged_ramp(self):
        v, c = self._ramp_capture()
        tr = extract_code_transitions(v, c)
        # Resolution bound: each transition reads high by at most one
        # ramp step, so INL/DNL can approach but not exceed one ramp
        # step referred to the LSB (code 0 no longer anchors the line).
        resolution_lsb = (1.4 / 2800) / 0.125
        assert inl(tr).max_abs_lsb <= resolution_lsb + 1e-9
        assert dnl(tr).max_abs_lsb <= resolution_lsb + 1e-9

    def test_lowest_code_gets_no_row(self):
        v, c = self._ramp_capture()
        tr = extract_code_transitions(v, c)
        assert 0.0 not in tr.x
        assert list(tr.x) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert tr.y[0] == pytest.approx(0.125, abs=(1.4 / 2800))

    def test_skipped_code_is_reported_missing(self):
        t = np.linspace(0.0, 1.0, 2801)
        v = t.copy()
        code = np.floor(v * 8.0).clip(0, 7)
        code[code == 3] = 4                              # converter skips code 3
        tr = extract_code_transitions(_tran(t, v),
                                      Signal('code', 'code', 'time', '', t, code))
        # code 3 still gets a row — it shares code 4's transition voltage,
        # a zero-width step: DNL hits exactly -1.0 at that row.
        assert 3.0 in tr.x
        i3, i4 = list(tr.x).index(3.0), list(tr.x).index(4.0)
        assert tr.y[i3] == tr.y[i4]
        assert dnl(tr).dnl_lsb[i3] == pytest.approx(-1.0)
        # missing_codes() used to be structurally blind here (code 3 was
        # simply absent from tr.x); now it fires and correctly names the
        # code that's actually never emitted.
        assert list(missing_codes(tr)) == [3.0]
        # code 3 still gets a row — it shares code 4's transition voltage
        assert 3.0 in tr.x
        i3, i4 = list(tr.x).index(3.0), list(tr.x).index(4.0)
        assert tr.y[i3] == tr.y[i4]


# ---------------------------------------------------------------------------
# F4 — MED: rise_time / fall_time miss an edge that fits in one interval
# ---------------------------------------------------------------------------

class TestF4FastEdgeInOneSampleInterval:
    """TASK-55 landed: ``rise_time``/``fall_time`` restarted the second
    crossing search one segment too late (``searchsorted`` on an
    interpolated time lands AFTER the interval it came from), skipping
    the segment that also held the second crossing when both crossings
    fall inside the SAME sample interval. Fix: back the restart index
    up by one segment (``start_idx = max(idx - 1, 0)``) before the
    second ``_first_crossing`` call. These are now plain regression
    tests (see F1/TASK-52 for the precedent of flipping xfail -> assert).
    """

    def test_rise_time_of_one_interval_edge(self):
        # Pre-fix: nan (the 90% segment was skipped by the +1 restart).
        t = np.arange(6) * 1e-9
        y = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        assert rise_time(_tran(t, y)) == pytest.approx(0.8e-9, rel=1e-9, abs=0)

    def test_fall_time_of_one_interval_edge(self):
        # Pre-fix: nan, same off-by-one in fall_time's high_idx restart.
        t = np.arange(6) * 1e-9
        y = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        assert fall_time(_tran(t, y)) == pytest.approx(0.8e-9, rel=1e-9, abs=0)

    def test_two_interval_edge_unaffected(self):
        # Regression guard: the fix backs the restart up by one segment,
        # it must not change an edge that was already found correctly
        # because its crossings span two DIFFERENT segments.
        t = np.arange(6) * 1e-9
        assert rise_time(_tran(t, [0, 0, 0.5, 1, 1, 1])) == pytest.approx(1.6e-9, rel=1e-9, abs=0)


# ---------------------------------------------------------------------------
# F5 — MED: silent zero when the band / window contains no data
# ---------------------------------------------------------------------------

class TestF5EmptyBandRaisesValueError:
    """FIXED (TASK-56, owner ruling 2026-09-03). Seven metrics used to
    answer "0" (a perfect result) when asked about a band or window
    that contains no samples: 0 dB ripple, 0 dB attenuation, 0 V noise,
    0 s jitter, 0 V² spectral power — each read as a
    spec-passing measurement even though no measurement happened. They
    now raise ``ValueError`` naming the metric, the requested band, the
    point count caught, and the range actually available — directly
    actionable for the usual cause, a units slip.

    (``kickback``, one of the seven, was removed by TASK-412; its raise and
    legitimate-zero tests went with it.)

    The trigger is the SAMPLE MASK holding too few points, not the
    computed answer being 0: a genuinely flat passband legitimately
    returns 0 dB ripple and must keep doing so. See the
    ``LegitimateZero`` tests below for the analog of each metric where
    one exists.

    Deliberate boundary (not rediscovered — see
    ``TestF5PointQueryVsBandAggregateBoundary``): a POINT query outside
    the data (``value_at``) returns ``nan``; a BAND
    AGGREGATE over an empty band raises.

    ``on_time`` was originally grouped with these seven by the audit
    ledger; the reviewer removed it from this finding — it has no band
    parameter and its 0 on an empty trace is a real, correct answer, so
    there is nothing here to key a raise off of. Its repro stays xfail,
    untouched by TASK-56 (the empty-trace case belongs to TASK-59).
    """

    F = np.logspace(1, 5, 41)

    # -- raises, with actionable message content -------------------------

    def test_passband_ripple_no_passband_raises(self):
        with pytest.raises(ValueError) as exc:
            passband_ripple_db(_freq(self.F, np.ones(41)), f_pass=1.0)
        msg = str(exc.value)
        assert 'passband_ripple_db' in msg
        assert '0 of 41' in msg
        assert f'{self.F.min():g}' in msg and f'{self.F.max():g}' in msg

    def test_stopband_attenuation_no_stopband_raises(self):
        with pytest.raises(ValueError) as exc:
            stopband_attenuation_db(_freq(self.F, np.ones(41)), f_stop=1e9)
        msg = str(exc.value)
        assert 'stopband_attenuation_db' in msg
        assert '0 of 41' in msg
        assert f'{self.F.min():g}' in msg and f'{self.F.max():g}' in msg

    def test_integrated_noise_band_narrower_than_grid_raises(self):
        # 1 nV/√Hz white; band [15.9, 15.95] Hz holds no grid point (grid: 10, 12.6, 15.8, ...)
        with pytest.raises(ValueError) as exc:
            integrated_noise(_freq(self.F, np.full(41, 1e-9)), 15.9, 15.95)
        msg = str(exc.value)
        assert 'integrated_noise' in msg
        assert '[15.9, 15.95]' in msg
        assert '0 of 41' in msg
        assert f'{self.F.min():g}' in msg and f'{self.F.max():g}' in msg

    def test_spectral_power_band_outside_raises(self):
        t = np.arange(1024) * 1e-9
        sig = _tran(t, np.sin(2 * np.pi * 11 * np.arange(1024) / 1024))
        with pytest.raises(ValueError) as exc:
            spectral_power(sig, 1e12, 2e12, tclk=1e-9, tstart=0.0, n_samples=1024)
        msg = str(exc.value)
        assert 'spectral_power' in msg
        assert '0 of 513' in msg          # 1024-sample rfft -> 513 bins

    # -- on_time was out of TASK-56's scope; TASK-77 landed it -----------

    def test_on_time_empty(self):
        """``on_time`` has no band parameter, so its silent 0.0 was not
        TASK-56's to fix — but it is the same polarity defect one layer
        wider (empty INPUT rather than empty BAND). Retagged TASK-56 ->
        TASK-59 -> TASK-77, and fixed there: an empty trace is a
        malformed request and raises. Kept here as the F5-shaped repro;
        the full empty-input contract lives in
        ``TestF8EmptySignalRaisesTypedValueError``."""
        with pytest.raises(ValueError, match='on_time'):
            on_time(_tran([], []), 0.5)


class TestF5LegitimateZero:
    """The trigger for TASK-56's raise is the sample mask, not the
    computed value — each metric that can legitimately answer exactly 0
    on a non-empty band must keep doing so.
    """

    F = np.logspace(1, 5, 41)

    def test_flat_passband_is_legitimately_zero(self):
        assert passband_ripple_db(_freq(self.F, np.ones(41)), f_pass=1e3) == 0.0

    def test_flat_response_stopband_attenuation_is_legitimately_zero(self):
        assert stopband_attenuation_db(_freq(self.F, np.ones(41)), f_stop=1e3) == 0.0

    def test_zero_psd_integrated_noise_is_legitimately_zero(self):
        # Band [1e2, 1e5] Hz holds most of the 41-point grid — not empty —
        # the density itself is 0.
        assert integrated_noise(_freq(self.F, np.zeros(41)), 1e2, 1e5) == 0.0

    def test_dc_trace_spectral_power_is_legitimately_zero(self):
        # Mean-removed flat (DC) trace has ~0 power at every AC bin.
        t = np.arange(1024) * 1e-9
        sig = _tran(t, np.ones(1024))
        r = spectral_power(sig, 1.0 / (1024e-9), 5e8, tclk=1e-9,
                           tstart=0.0, n_samples=1024)
        assert r == pytest.approx(0.0, abs=1e-12)


class TestF5PointQueryVsBandAggregateBoundary:
    """Deliberate, documented boundary: a POINT query outside the data
    returns ``nan`` (the standard idiom — numpy does the same); a BAND
    AGGREGATE over an empty band is a malformed request and raises.
    """

    def test_point_query_out_of_range_returns_nan(self):
        t = np.linspace(0, 64e-9, 3201)
        sig = _tran(t, np.sin(2 * np.pi * 1e7 * t))
        assert math.isnan(value_at(sig, 1e-6))

    def test_band_aggregate_out_of_range_raises(self):
        F = np.logspace(1, 5, 41)
        sig = _freq(F, np.ones(41))
        with pytest.raises(ValueError):
            passband_ripple_db(sig, f_pass=1.0)
        with pytest.raises(ValueError):
            integrated_noise(sig, 15.9, 15.95)


# ---------------------------------------------------------------------------
# F6 — MED: bandwidth_3dB / gain_bw_product report the sweep top when the
#            response never rolls off inside the sweep
# ---------------------------------------------------------------------------

class TestF6SweepTopReportedAsBandwidth:
    """``gain_margin`` was fixed 2026-08-23 to return NaN when "the sweep
    ended before the answer was knowable". ``bandwidth_3dB`` used to
    still return the last grid point — indistinguishable from a real
    −3 dB point at that frequency — and ``gain_bw_product`` multiplied
    it in.

    FIXED (TASK-63, folded in from TASK-57 2026-09-03): both now return
    NaN when the response is still above −3 dB at the top of the sweep
    (the same "unknowable" convention as ``gain_margin``). Both also
    sort internally by ascending frequency before checking "top of
    sweep" and taking DC gain, so ascending and descending sweeps of
    the same data agree — a positional-only fix (dc_gain at
    ``argmin(x)`` without sorting the rest of the arrays) still leaves
    "last array slot" and "highest swept frequency" out of sync for a
    descending sweep, turning a genuinely-rolled-off descending sweep
    into a false NaN.
    """

    def test_flat_response_bandwidth_is_unknowable(self):
        f = np.logspace(1, 6, 101)
        assert math.isnan(bandwidth_3dB(_freq(f, 10.0 * np.ones(101))))
        assert math.isnan(bandwidth_3dB(_freq(f[::-1], 10.0 * np.ones(101))))

    def test_gain_bw_product_is_order_independent(self):
        f = np.logspace(1, 6, 101)
        h = 100.0 / (1 + 1j * f / 1e3)
        asc = gain_bw_product(_freq(f, h))
        desc = gain_bw_product(_freq(f[::-1], h[::-1]))
        assert asc == pytest.approx(desc)

    def test_bandwidth_3dB_is_order_independent(self):
        f = np.logspace(1, 6, 101)
        h = 100.0 / (1 + 1j * f / 1e3)
        asc = bandwidth_3dB(_freq(f, h))
        desc = bandwidth_3dB(_freq(f[::-1], h[::-1]))
        assert asc == pytest.approx(desc)
        assert not math.isnan(asc)

    def test_flat_response_gain_bw_product_is_also_unknowable(self):
        f = np.logspace(1, 6, 101)
        assert math.isnan(gain_bw_product(_freq(f, 10.0 * np.ones(101))))

    def test_rolled_off_response_unaffected(self):
        """A response that genuinely rolls off inside the sweep is
        byte-for-byte unchanged by the fix."""
        f = np.logspace(1, 6, 101)
        h = 100.0 / (1 + 1j * f / 1e3)
        assert bandwidth_3dB(_freq(f, h)) == pytest.approx(1e3, rel=0.05)
        assert gain_bw_product(_freq(f, h)) == pytest.approx(1e5, rel=0.05)


# ---------------------------------------------------------------------------
# F7 — MED: phase-noise helpers interpolate / integrate in linear f
#
# FIXED (TASK-58, owner-ruled 2026-09-03), then TASK-210 (2026-09-23) removed
# both `phase_noise_at_offset` and `rms_jitter_from_phase_noise` entirely (no
# silk analysis produces the `phase_noise_sideband` Signal either one took),
# so the regression class that pinned F7's fix is retired with them.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# F8 — MED, split 2026-09-03 into TASK-59 (NaN half) / TASK-77 (empty half)
#            — the class below covers the NaN half, the one after it the
#              empty half
# ---------------------------------------------------------------------------

class TestF8NanSampleYieldsFiniteWrongNumber:
    """One NaN sample (an aborted-sim tail, a divide in a derived Signal)
    used to sail past nine metrics and come out the other side as a
    *finite* number that looked like a real answer.

    FIXED (TASK-59, owner-ruled 2026-09-03 via ``decision-1``): a NaN
    sample is DATA-DEPENDENT (a failed convergence point, varies per
    corner), not a caller error — so the fix is NOT to reject it (that
    would abort a whole PVT campaign over one bad timestep in one corner).
    It is to make every affected metric propagate NaN the way the majority
    (``average``, ``rms``, ``peak_to_peak``, ``stddev``, ``integral``)
    already did before this task touched anything.

    Fixture: the reviewer's 101-sample, 50 %-duty, 5-period square wave
    (T = 1 ns). ``_square101`` optionally places one NaN sample at a given
    index; ``HIGH_IDX``/``LOW_IDX`` are a HIGH and a LOW sample on the
    clean trace — the on/off/duty trio's old bug only showed up with the
    NaN on a HIGH sample (the old mechanism was "NaN counts as
    not-above-level"), so those three get both positions to confirm the
    fix doesn't depend on which phase the NaN lands in.
    """

    N = 101
    T = 1e-9
    HIGH_IDX = 20   # y == 1.0 on the clean trace
    LOW_IDX = 30    # y == 0.0 on the clean trace

    def _square101(self, nan_idx=None):
        t = np.linspace(0, 5 * self.T, self.N)
        y = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / self.T + 1e-12))
        if nan_idx is not None:
            y = y.copy()
            y[nan_idx] = np.nan
        return _tran(t, y)

    # -- sanity: the clean fixture isn't already nan on any of these -------

    def test_clean_fixture_is_finite_on_every_affected_metric(self):
        s = self._square101()
        assert not math.isnan(average_duty(s))
        assert not math.isnan(argmin_x(s))
        assert not math.isnan(argmax_x(s))
        assert len(crossings(s)) > 0 and not math.isnan(crossings(s)[0])
        assert not math.isnan(on_time(s, 0.5))
        assert not math.isnan(off_time(s, 0.5))
        assert not math.isnan(settling_time(s, initial=0.0, target=1.0))
        assert not math.isnan(crossing(s, 0.5, edge='any'))

    # -- one NaN sample propagates to nan, not a finite wrong number -------

    def test_duty_cycle_propagates_nan_from_high_sample(self):
        assert math.isnan(average_duty(self._square101(self.HIGH_IDX)))

    def test_duty_cycle_propagates_nan_from_low_sample(self):
        assert math.isnan(average_duty(self._square101(self.LOW_IDX)))

    def test_argmin_x_propagates_nan(self):
        assert math.isnan(argmin_x(self._square101(self.HIGH_IDX)))

    def test_argmax_x_propagates_nan(self):
        assert math.isnan(argmax_x(self._square101(self.HIGH_IDX)))

    def test_on_time_propagates_nan_from_high_sample(self):
        assert math.isnan(on_time(self._square101(self.HIGH_IDX), 0.5))

    def test_on_time_propagates_nan_from_low_sample(self):
        assert math.isnan(on_time(self._square101(self.LOW_IDX), 0.5))

    def test_off_time_propagates_nan_from_high_sample(self):
        assert math.isnan(off_time(self._square101(self.HIGH_IDX), 0.5))

    def test_off_time_propagates_nan_from_low_sample(self):
        assert math.isnan(off_time(self._square101(self.LOW_IDX), 0.5))

    def test_settling_time_propagates_nan(self):
        assert math.isnan(
            settling_time(self._square101(self.HIGH_IDX), initial=0.0, target=1.0))

    def test_crossing_propagates_nan(self):
        assert math.isnan(crossing(self._square101(self.HIGH_IDX), 0.5, edge='any'))

    def test_bandwidth_3dB_propagates_nan(self):
        f = np.logspace(1, 5, 41)
        h = 100.0 / (1 + 1j * f / 1e3)
        h[20] = np.nan
        assert math.isnan(bandwidth_3dB(_freq(f, h)))

    def test_harmonic_freqs_propagates_nan(self):
        t = np.arange(1024) * 1e-9
        y = np.sin(2 * np.pi * 11 * np.arange(1024) / 1024)
        y[5] = np.nan
        r = harmonic_freqs(_tran(t, y), tclk=1e-9, n_samples=1024)
        assert len(r) >= 1
        assert math.isnan(r[0])

    # -- crossings: documented, non-ambiguous NaN answer (not empty) -------

    def test_crossings_returns_single_nan_sentinel_not_empty(self):
        """An empty array already means "genuinely zero crossings" (a
        valid result, e.g. for a flat trace) — reusing it for "a NaN made
        this unknowable" would make the two indistinguishable to a
        caller. ``crossings`` instead returns the one-element
        ``array([nan])`` on NaN input; documented in its docstring.
        """
        r = crossings(self._square101(self.HIGH_IDX))
        assert len(r) == 1
        assert math.isnan(r[0])
        # distinct from the genuine "zero crossings" case
        flat = _tran(np.linspace(0, 5e-9, self.N), np.ones(self.N))
        assert len(crossings(flat)) == 0

    # -- decision-1: NaN never raises, campaigns must be able to continue --

    def test_no_affected_metric_raises_on_nan(self):
        s_high = self._square101(self.HIGH_IDX)
        s_low = self._square101(self.LOW_IDX)
        f = np.logspace(1, 5, 41)
        h = 100.0 / (1 + 1j * f / 1e3)
        h[20] = np.nan
        t = np.arange(1024) * 1e-9
        y = np.sin(2 * np.pi * 11 * np.arange(1024) / 1024)
        y[5] = np.nan
        calls = (
            lambda: average_duty(s_high), lambda: average_duty(s_low),
            lambda: argmin_x(s_high), lambda: argmax_x(s_high),
            lambda: crossings(s_high),
            lambda: on_time(s_high, 0.5), lambda: on_time(s_low, 0.5),
            lambda: off_time(s_high, 0.5), lambda: off_time(s_low, 0.5),
            lambda: settling_time(s_high, initial=0.0, target=1.0),
            lambda: crossing(s_high, 0.5, edge='any'),
            lambda: bandwidth_3dB(_freq(f, h)),
            lambda: harmonic_freqs(_tran(t, y), tclk=1e-9, n_samples=1024),
        )
        for call in calls:
            call()   # must not raise

    # -- the majority (already-correct) metrics are unchanged by this task -

    def test_majority_metrics_unaffected_by_this_task(self):
        s = self._square101(self.HIGH_IDX)
        assert math.isnan(average(s))
        assert math.isnan(rms(s))
        assert math.isnan(peak_to_peak(s))
        assert math.isnan(stddev(s))
        assert math.isnan(integral(s))


class TestF8EmptySignalRaisesTypedValueError:
    """The other half of F8: a ZERO-LENGTH Signal used to produce four
    different behaviours across the library, none of them a contract.

    Reviewer-measured 2026-09-03 across a sample of twelve metrics:
    ``IndexError`` (average, rms, peak_to_peak, argmin_x, stddev,
    integral — a numpy accident), ``ValueError`` (average_duty, rise_time,
    crossings, slew_rate — typed by luck: numpy's untyped "zero-size
    array to reduction operation" happens to BE a ValueError, with no
    mention of the metric), a silent ``0.0`` (on_time — the F5 shape
    exactly: silent 0 PASSES a spec check) and ``nan`` (settling_time).

    FIXED (TASK-77, owner-ruled 2026-09-03 via ``decision-1``): an empty
    Signal is a MALFORMED REQUEST — the caller handed the metric no data
    — which is decision-1 category 2, so it raises ``ValueError`` naming
    the metric and the condition. A run that genuinely produced nothing
    is ``SimulationAborted``'s job, not a metric's.

    NOT folded in: the SINGLE-sample case. One sample is legitimate
    input to several metrics, and the edge matrix pins ``single``
    separately from ``empty``.
    """

    def _empty(self, xlabel='time'):
        return Signal('empty_probe', 'voltage', xlabel, 'V',
                      np.array([], dtype=float), np.array([], dtype=float))

    # -- one typed error across all four of the old behaviours ----------

    def test_the_four_old_behaviours_all_raise_the_same_typed_error(self):
        s = self._empty()
        for name, call in (
                ('average', lambda: average(s)),              # was IndexError
                ('on_time', lambda: on_time(s, 0.5)),         # was silent 0.0
                ('settling_time',                             # was nan
                 lambda: settling_time(s, initial=0.0, target=1.0)),
                ('average_duty', lambda: average_duty(s)),        # was ValueError
        ):
            with pytest.raises(ValueError) as exc:
                call()
            msg = str(exc.value)
            assert msg.startswith(name + ':'), (name, msg)
            assert 'no samples' in msg and 'len(x) == 0' in msg, (name, msg)
            assert "'empty_probe'" in msg, (name, msg)

    def test_on_time_no_longer_returns_a_silently_passing_zero(self):
        """The F5 polarity argument, one layer wider: 0.0 through
        ``SpecReport.check`` is a PASS, so a metric that never measured
        anything used to report success."""
        with pytest.raises(ValueError, match='on_time'):
            on_time(self._empty(), 0.5)

    def test_settling_time_no_longer_returns_nan(self):
        with pytest.raises(ValueError, match='settling_time'):
            settling_time(self._empty(), initial=0.0, target=1.0)

    def test_untyped_index_error_is_gone_from_the_window_primitives(self):
        s = self._empty()
        for call in (lambda: average(s), lambda: rms(s),
                     lambda: peak_to_peak(s), lambda: stddev(s),
                     lambda: integral(s), lambda: argmin_x(s),
                     lambda: argmax_x(s)):
            with pytest.raises(ValueError):
                call()

    def test_every_axis_gets_the_same_contract(self):
        """Transient, frequency and code Signals all route through a
        shared validator, so all three answer the same way."""
        with pytest.raises(ValueError, match='crossings'):
            crossings(self._empty('time'))
        with pytest.raises(ValueError, match='dc_gain_dB'):
            dc_gain_dB(self._empty('frequency'))
        with pytest.raises(ValueError, match='inl'):
            inl(self._empty('code'))

    # -- the two things this fix must NOT have changed ------------------

    def test_single_sample_behaviour_is_unchanged(self):
        """TASK-77's critical scope limit: reject zero samples ONLY.
        These one-sample answers are the pre-fix answers, verbatim."""
        one = _tran([0.0], [1.0])
        assert on_time(one, 0.5) == 0.0
        assert off_time(one, 0.5) == 0.0
        assert settling_time(one, initial=0.0, target=1.0) == 0.0
        assert math.isnan(average_duty(one))
        assert len(crossings(one)) == 0
        # the window primitives still reject a zero-width window on a
        # one-sample trace — their own pre-existing rule, reached AFTER
        # the empty guard passes, not replaced by it
        with pytest.raises(ValueError, match='require end > start'):
            average(one)

    def test_nan_propagation_contract_is_intact(self):
        """TASK-59's ruling stands: the validators gained NO NaN
        rejection, so a NaN-bearing (non-empty) Signal still returns nan
        rather than raising."""
        t = np.arange(101) * 1e-11
        y = np.where((np.arange(101) % 20) < 10, 1.0, 0.0)
        y[20] = np.nan
        s = _tran(t, y)
        assert math.isnan(average_duty(s))
        assert math.isnan(settling_time(s, initial=0.0, target=1.0))
        assert math.isnan(on_time(s, 0.5))
        assert math.isnan(average(s))
        assert np.all(np.isnan(crossings(s)))


# ---------------------------------------------------------------------------
# F9 — LOW: legacy AC helpers + hysteresis accept any x-axis
# ---------------------------------------------------------------------------

class TestF9WrongAxisAccepted:
    """FIXED (TASK-60, 2026-09-02): every ``silk.metrics`` AC/noise
    helper now runs the same ``_check_freq`` contract as ``silk.metrics.ac``
    (``xlabel == 'frequency'`` and >= 2 points) and ``hysteresis`` runs
    ``_check_transient`` on all four inputs. Before: a 100 V constant
    transient read as 40 dB of "DC gain", ``cmrr_dB(tran, tran)`` was 0 dB
    and ``hysteresis`` accepted four frequency sweeps.
    """

    _AC_HELPERS = [
        ('dc_gain_dB',      lambda s: M.dc_gain_dB(s)),
        ('peak_gain_dB',    lambda s: M.peak_gain_dB(s)),
        ('gain_at_freq',    lambda s: M.gain_at_freq(s, 1e3)),
        ('bandwidth_3dB',   lambda s: M.bandwidth_3dB(s)),
        ('unity_gain_freq', lambda s: M.unity_gain_freq(s)),
        ('phase_at_freq',   lambda s: M.phase_at_freq(s, 1e3)),
        ('phase_margin',    lambda s: M.phase_margin(s)),
        ('gain_margins',    lambda s: M.gain_margins(s)),
        ('gain_margin',     lambda s: M.gain_margin(s)),
        ('cmrr_dB',         lambda s: M.cmrr_dB(s, s)),
        ('integrated_noise', lambda s: M.integrated_noise(s, 1e2, 1e5)),
    ]

    @pytest.mark.parametrize('name,call', _AC_HELPERS, ids=[n for n, _ in _AC_HELPERS])
    def test_ac_helper_rejects_time_axis(self, name, call):
        t = np.linspace(0, 1e-6, 101)
        with pytest.raises(ValueError, match="xlabel='frequency'"):
            call(_tran(t, 100.0 * np.ones(101)))

    @pytest.mark.parametrize('name,call', _AC_HELPERS, ids=[n for n, _ in _AC_HELPERS])
    def test_ac_helper_rejects_single_point_sweep(self, name, call):
        with pytest.raises(ValueError, match='at least 2 frequency points'):
            call(_freq([10.0], [1.0]))

    def test_dc_gain_rejects_time_axis(self):
        t = np.linspace(0, 1e-6, 101)
        with pytest.raises(ValueError):
            dc_gain_dB(_tran(t, 100.0 * np.ones(101)))

    def test_cmrr_rejects_either_input_on_the_wrong_axis(self):
        f = np.logspace(1, 5, 101)
        ac = _freq(f, np.ones(101))
        tr = _tran(np.linspace(0, 1e-6, 101), np.ones(101))
        with pytest.raises(ValueError):
            cmrr_dB(ac, tr)
        with pytest.raises(ValueError):
            cmrr_dB(tr, ac)
        assert cmrr_dB(ac, ac) == 0.0                        # right axis still works

    def test_hysteresis_rejects_frequency_axis_on_any_input(self):
        f = np.logspace(1, 5, 101)
        up, down = _freq(f, np.linspace(0.0, 1.0, 101)), _freq(f, np.linspace(1.0, 0.0, 101))
        t = np.linspace(0, 1, 101)
        tup, tdown = _tran(t, np.linspace(0.0, 1.0, 101)), _tran(t, np.linspace(1.0, 0.0, 101))
        with pytest.raises(ValueError, match="xlabel='time'"):
            hysteresis(up, up, down, down)
        for args in [(up, tup, tdown, tdown), (tup, up, tdown, tdown),
                     (tup, tup, down, tdown), (tup, tup, tdown, down)]:
            with pytest.raises(ValueError, match="xlabel='time'"):
                hysteresis(*args)
        assert hysteresis(tup, tup, tdown, tdown) == pytest.approx(0.0)

    def test_right_axis_still_reads_the_same_numbers(self):
        f = np.logspace(1, 5, 101)
        s = _freq(f, 100.0 * np.ones(101))
        assert dc_gain_dB(s) == pytest.approx(40.0)
        assert cmrr_dB(s, s) == 0.0


# ---------------------------------------------------------------------------
# F10 — LOW: dynamic_energy_per_cycle rounds the cycle count
# ---------------------------------------------------------------------------

class TestF10CycleCountRounding:
    """FIXED (TASK-61). Energy used to be integrated over the WHOLE trace
    but divided by ``round(T / period)``: a 2.4-period trace of constant
    power read +20 % per-cycle energy, a 2.6-period one −13 % (and, on a
    pulsed/switching waveform — what this metric is actually for — up to
    +50 %). The fix integrates over exactly ``floor(T / period)`` whole
    periods and discards the partial tail; see the full fractional-trace
    coverage (both waveform shapes, 2.4/2.6/3.4/10.4 periods) in
    ``tests/metrics/test_transient.py::TestDynamicEnergyPerCycle``.
    """

    def test_fractional_trace_length_does_not_bias_energy(self):
        period = 1e-9
        t = np.linspace(0, 2.4 * period, 2401)
        v = _tran(t, np.ones_like(t))
        i = Signal('i', 'current', 'time', 'A', t, 1e-3 * np.ones_like(t))
        assert dynamic_energy_per_cycle(v, i, period) == pytest.approx(1e-3 * period, rel=1e-3, abs=0)


# ---------------------------------------------------------------------------
# F11 — harmonics above Nyquist are folded back in band
# ---------------------------------------------------------------------------

class TestF11AliasedHarmonicsNotFolded:
    """FIXED (TASK-62). In a sampled system a harmonic above fs/2 aliases
    back in band (IEEE 1241 folds it) — the FFT data is already correct,
    only the ``k * fund_bin`` bin lookup used to be wrong. ``thd`` /
    ``harmonic_levels_db`` now fold the lookup so the aliased H3 of a
    200/1024 tone at 1 GS/s (nominal bin 600, folds to bin 424) is
    counted as distortion instead of silently dropped — and ``snr``
    excludes that same folded bin from noise, so the two stay
    consistent (:class:`TestThdSnrMutualConsistency`).
    """

    def _tone_with_aliased_h3(self):
        k = np.arange(N)
        y = np.sin(2 * np.pi * 200 * k / N) + 1e-2 * np.sin(2 * np.pi * 600 * k / N)
        return _tran(k * TCLK, y)            # 600 > 512 ⇒ folds to bin 424

    def test_thd_sees_the_folded_h3(self):
        got = thd(self._tone_with_aliased_h3(), tclk=TCLK, n_samples=N)
        assert got == pytest.approx(-40.0, abs=1e-6)

    def test_fixed_behaviour(self):
        s = self._tone_with_aliased_h3()
        assert thd(s, tclk=TCLK, n_samples=N) == pytest.approx(-40.0, abs=1e-6)
        assert harmonic_levels_db(s, tclk=TCLK, n_samples=N)[1] == pytest.approx(-40.0, abs=1e-6)
        # the folded H3 energy is now excluded from snr's noise sum, so
        # snr jumps far above sndr (which still lumps it into "noise+dist").
        assert snr(s, tclk=TCLK, n_samples=N) > 100.0
        assert sndr(s, tclk=TCLK, n_samples=N) == pytest.approx(40.0, abs=1e-6)   # unchanged (non-goal)


# ---------------------------------------------------------------------------
# TASK-62 — polynomial-nonlinearity sweep (owner-specified test design)
# ---------------------------------------------------------------------------

class TestPolynomialHarmonicFoldSweep:
    """Drive a sine through ``y = x + a2*x^2 + a3*x^3`` and sweep f0 so
    the harmonics fall inside AND outside Nyquist. Trig expansion gives
    ``H2 = a2/2``, ``H3 = a3/4``, fundamental amplitude ``1 + 3*a3/4`` —
    all INDEPENDENT of f0. The analytic THD is therefore a CONSTANT
    across the sweep and correct behaviour is a FLAT LINE; any
    frequency dependence in the measured value is pure defect. Far
    stronger oracle than a single hand-picked aliasing case.

    N=4096, tclk=1e-9, a2=0.05, a3=0.02 -> analytic THD = -32.000 dB at
    every frequency (reviewer-measured). Five frequencies span all
    three regimes: both harmonics in band, one folded (partial miss —
    the regime a fix that only handles total-alias would still get
    wrong), and both folded.
    """

    N = 4096
    TCLK = 1e-9
    A2 = 0.05
    A3 = 0.02
    ANALYTIC_THD_DB = -32.0

    def _signal(self, f0_over_fs):
        fs = 1.0 / self.TCLK
        kf = round(f0_over_fs * self.N)
        assert kf >= 1
        t = np.arange(self.N) * self.TCLK
        x = np.sin(2 * np.pi * (kf * fs / self.N) * t)
        y = x + self.A2 * x ** 2 + self.A3 * x ** 3
        return _tran(t, y), kf

    @pytest.mark.parametrize('f0_over_fs,regime', [
        (0.049, 'both in band'),
        (0.171, 'partial fold (H2 in band, H3 folded)'),
        (0.220, 'partial fold (H2 in band, H3 folded)'),
        (0.342, 'total fold (H2 and H3 both folded)'),
        (0.464, 'total fold (H2 and H3 both folded)'),
    ])
    def test_thd_is_flat_across_the_sweep(self, f0_over_fs, regime):
        sig, kf = self._signal(f0_over_fs)
        got = thd(sig, tclk=self.TCLK, n_samples=self.N, n_harmonics=3)
        assert got == pytest.approx(self.ANALYTIC_THD_DB, abs=1e-2), (
            f'f0/fs={f0_over_fs} (kf={kf}, {regime}): expected a flat '
            f'{self.ANALYTIC_THD_DB} dB, got {got}'
        )

    def test_partial_fold_regime_explicitly(self):
        """H2 stays in band, H3 folds — today (pre-fix) this read
        -32.171 dB (-0.17 dB off): H2 alone, folded H3 silently
        dropped. A fix that only handles total-alias (both harmonics
        folded) would still get this wrong.
        """
        for f0_over_fs in (0.171, 0.220):
            sig, kf = self._signal(f0_over_fs)
            nominal_h2 = 2 * kf
            nominal_h3 = 3 * kf
            assert nominal_h2 <= self.N // 2, 'H2 expected in band'
            assert nominal_h3 > self.N // 2, 'H3 expected above Nyquist'
            got = thd(sig, tclk=self.TCLK, n_samples=self.N, n_harmonics=3)
            assert got == pytest.approx(self.ANALYTIC_THD_DB, abs=1e-2)


# ---------------------------------------------------------------------------
# TASK-62 — collision cases (each must be handled, none summed naively)
# ---------------------------------------------------------------------------

class TestHarmonicFoldCollisions:
    """The four collision cases the fold must handle distinctly, per
    the task's worked-example spec: fold onto DC (skip), fold onto the
    fundamental (skip), fold onto a bin a lower harmonic already
    claimed (count once), fold exactly onto the Nyquist bin (count
    once).
    """

    def test_fold_onto_dc_is_skipped(self):
        # N=8, fund_bin=4: k=2 -> (2*4)%8=0 -> DC. Must not appear in
        # the bin map, and must not contribute (nan/inf) to thd's sum.
        from silk.metrics.spectral import _harmonic_bin_map
        assert 2 not in _harmonic_bin_map(fund_bin=4, n_samples=8, n_harmonics=2)

    def test_fold_onto_fundamental_is_skipped(self):
        # N=9, fund_bin=3: k=4 -> (4*3)%9=3 == fund_bin. Inseparable ->
        # skipped, not double-counted as the fundamental's own power.
        from silk.metrics.spectral import _harmonic_bin_map
        bin_map = _harmonic_bin_map(fund_bin=3, n_samples=9, n_harmonics=4)
        assert 4 not in bin_map

    def test_two_harmonics_colliding_on_one_bin_counted_once(self):
        # fund_bin=2, N=12: k=2 -> (2*2)%12=4 (bin 4, no mirror needed);
        # k=8 -> (8*2)%12=16%12=4 (same bin 4) -- a genuine collision.
        from silk.metrics.spectral import _harmonic_bin_map, _harmonic_power_bins
        bin_map = _harmonic_bin_map(fund_bin=2, n_samples=12, n_harmonics=8)
        assert bin_map[2] == 4
        assert bin_map[8] == 4
        bins = _harmonic_power_bins(fund_bin=2, n_samples=12, n_harmonics=8)
        assert list(bins).count(4) == 1          # a set can't hold a bin twice

        # End-to-end: two harmonics that alias onto the SAME bin must
        # not double the measured power relative to only one of them
        # aliasing there.
        N, TCLK_ = 12, 1e-9
        k = np.arange(N)
        fund_bin_hz = 2
        y = (np.sin(2 * np.pi * fund_bin_hz * k / N)
             + 0.1 * np.sin(2 * np.pi * 4 * k / N))       # power lands once at bin 4
        sig = _tran(k * TCLK_, y)
        got = thd(sig, tclk=TCLK_, n_samples=N, n_harmonics=8)
        # both k=2 and k=8 fold onto bin 4 (which holds ONE tone's power);
        # thd must report that single tone's level, not double it (+3 dB).
        expected = 20.0 * np.log10(0.1)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_fold_exactly_onto_nyquist_bin_counted_once(self):
        # N=8, fund_bin=2: k=2 -> 4%8=4 == N//2 exactly -> Nyquist bin,
        # a legitimate real-valued bin, counted (not skipped).
        from silk.metrics.spectral import _harmonic_bin_map
        bin_map = _harmonic_bin_map(fund_bin=2, n_samples=8, n_harmonics=2)
        assert bin_map[2] == 4
        N, TCLK_ = 8, 1e-9
        k = np.arange(N)
        # a sin() component at exactly the Nyquist bin samples to all
        # zeros at integer points (undersampled) — only cos() is
        # representable there for a real signal, so use cos for the
        # injected "harmonic" tone.
        y = np.sin(2 * np.pi * 2 * k / N) + 0.05 * np.cos(2 * np.pi * 4 * k / N)
        sig = _tran(k * TCLK_, y)
        # Sampled mean squares: Nyquist .05², ordinary fundamental 1/2.
        expected = 10.0 * np.log10(0.05**2 / 0.5)
        lv = harmonic_levels_db(sig, tclk=TCLK_, n_samples=N, n_harmonics=2)
        assert lv[0] == pytest.approx(expected, abs=1e-6)
        # and it must be counted (not skipped like the DC/fundamental
        # collision cases): a real, finite level, not -inf.
        assert np.isfinite(lv[0])


# ---------------------------------------------------------------------------
# TASK-62 — thd/snr mutual consistency (no double count)
# ---------------------------------------------------------------------------

class TestThdSnrMutualConsistency:
    """thd counts folded harmonic energy as distortion; snr must
    EXCLUDE that same energy from its noise sum — otherwise the same
    bin's power is double-booked (as distortion by thd AND as noise by
    snr) and the two metrics disagree about what a fixed pool of
    non-fundamental power is. sndr/sinad (which lump everything into
    one bucket) must be untouched by the fold — TASK-62 is scoped to
    attribution only, not their totals.
    """

    def _aliased_signal(self):
        k = np.arange(N)
        y = np.sin(2 * np.pi * 200 * k / N) + 1e-2 * np.sin(2 * np.pi * 600 * k / N)
        return _tran(k * TCLK, y)

    def test_no_double_count(self):
        s = self._aliased_signal()
        # Reconstruct the bookkeeping directly: sndr's denominator is
        # the sum of every non-fundamental, non-DC bin; thd's
        # numerator and snr's excluded term must be the SAME folded
        # harmonic power, and noise_only + harm_power must reconstruct
        # that denominator (no bin lost, none doubled).
        from silk.metrics.spectral import _sample_at_clock, _power_spectrum, _harmonic_power_bins
        samples = _sample_at_clock(s, metric='test_no_double_count', tclk=TCLK, n_samples=N)
        P, fund_bin = _power_spectrum(samples)
        bins = _harmonic_power_bins(fund_bin, len(samples), n_harmonics=10)
        harm_power = float(np.sum(P[list(bins)]))
        residual = np.ones(len(P), dtype=bool)
        residual[[0, fund_bin]] = False
        total_non_fund = float(np.sum(P[residual]))
        residual[list(bins)] = False
        noise_only = float(np.sum(P[residual]))
        assert noise_only + harm_power == pytest.approx(total_non_fund, abs=1e-12)
        assert noise_only > 0
        # cross-check against the public snr()/sndr() split
        got_snr = snr(s, tclk=TCLK, n_samples=N, n_harmonics=10)
        got_sndr = sndr(s, tclk=TCLK, n_samples=N)
        # Subtraction overstated this tiny residual by tens of dB.
        assert got_snr == pytest.approx(186.72, abs=0.01)
        assert 10.0 * np.log10(P[fund_bin] / noise_only) == pytest.approx(got_snr, abs=1e-9)
        assert 10.0 * np.log10(P[fund_bin] / total_non_fund) == pytest.approx(got_sndr, abs=1e-9)
        # sndr/sinad totals unchanged by the fold (non-goal, TASK-62)
        assert sinad(s, tclk=TCLK, n_samples=N) == got_sndr


# ---------------------------------------------------------------------------
# TASK-79 — harmonic() / harmonic_freqs() agree with harmonic_levels_db
# ---------------------------------------------------------------------------

class TestTask79HarmonicFamilyConsistency:
    """After TASK-62, ``harmonic_levels_db`` reported the aliased H3 of the
    200/1024 fixture at -40 dBc while ``harmonic(sig, 3)`` said nan and
    ``harmonic_freqs`` said H3 did not exist. Ruling (TASK-79):
    ``harmonic`` reads the same folded bin (true phase — the
    ``mix_product`` two-sided convention); ``harmonic_freqs`` reports
    the TRUE ``k·f0`` for every k (``observed=True`` for the folded bin
    centre). One ``_harmonic_bin_map`` feeds all three, so nan / -inf
    appear for exactly the same (unattributable) harmonics.
    """

    def _aliased(self):
        k = np.arange(N)
        y = np.sin(2 * np.pi * 200 * k / N) + 1e-2 * np.sin(2 * np.pi * 600 * k / N)
        return _tran(k * TCLK, y)            # H3 nominal 600 > 512 ⇒ folds to 424

    K = dict(tclk=TCLK, n_samples=N)
    DF = 1.0 / (N * TCLK)

    def test_three_functions_agree_on_the_aliased_h3(self):
        s = self._aliased()
        levels = harmonic_levels_db(s, **self.K, n_harmonics=3)
        h1 = harmonic(s, 1, **self.K)
        h3 = harmonic(s, 3, **self.K)
        assert not math.isnan(h3.real)                             # was nan+0j
        assert 20.0 * math.log10(abs(h3) / abs(h1)) == pytest.approx(levels[1], abs=1e-9)
        assert levels[1] == pytest.approx(-40.0, abs=1e-6)
        true = harmonic_freqs(s, **self.K, n=3)
        obs = harmonic_freqs(s, **self.K, n=3, observed=True)
        assert len(true) == len(obs) == 3                          # was length 2
        assert true[2] == pytest.approx(600 * self.DF)             # what H3 IS
        assert true[2] > 0.5 / TCLK                                # ...and it aliased
        assert obs[2] == pytest.approx(424 * self.DF)              # where it is OBSERVED
        assert np.allclose(obs[:2], true[:2])                      # in-band: identical

    def test_harmonic_phase_is_the_true_phase(self):
        """The mirrored rfft bin holds conj(phasor); harmonic() hands
        back the phasor itself — the convention mix_product already
        uses (two-sided fft, index mod N). rfft of A·sin at its own
        bin is -j·A·N/2."""
        s = self._aliased()
        h3 = harmonic(s, 3, **self.K)
        assert h3 == pytest.approx(-1j * 1e-2 * N / 2, abs=1e-9)
        mp = mix_product(s, 3, 0, f1_hz=200 * self.DF, f2_hz=1.0, **self.K)
        assert h3 == pytest.approx(mp, abs=1e-9)
        # an in-band harmonic is unchanged by TASK-79 (no mirror, no conj)
        k = np.arange(N)
        y = np.sin(2 * np.pi * 100 * k / N) + 1e-2 * np.sin(2 * np.pi * 300 * k / N)
        s_in = _tran(k * TCLK, y)
        assert harmonic(s_in, 3, **self.K) == pytest.approx(-1j * 1e-2 * N / 2, abs=1e-9)

    def test_nan_where_harmonic_levels_are_unattributable(self):
        """Fold onto DC (N=8, fund_bin=4, k=2 -> bin 0) and onto the
        fundamental (N=9, fund_bin=3, k=4 -> bin 3): harmonic_levels_db
        says nan; harmonic and harmonic_freqs(observed=True) also say nan;
        harmonic_freqs (true) still reports k·f0 — the frequency exists
        even when the energy cannot be attributed."""
        for n, fb, k_bad in ((8, 4, 2), (9, 3, 4)):
            t = np.arange(n) * TCLK
            s = _tran(t, np.sin(2 * np.pi * fb * np.arange(n) / n))
            lv = harmonic_levels_db(s, tclk=TCLK, n_samples=n, n_harmonics=k_bad)
            assert math.isnan(lv[k_bad - 2])
            assert math.isnan(harmonic(s, k_bad, tclk=TCLK, n_samples=n).real)
            obs = harmonic_freqs(s, tclk=TCLK, n_samples=n, n=k_bad, observed=True)
            assert math.isnan(obs[k_bad - 1])
            true = harmonic_freqs(s, tclk=TCLK, n_samples=n, n=k_bad)
            assert true[k_bad - 1] == pytest.approx(k_bad * fb / (n * TCLK))
            # every OTHER k: finite in all three, or nan in all three
            for k in range(2, k_bad + 1):
                fin_lv = np.isfinite(lv[k - 2])
                assert fin_lv == (not math.isnan(harmonic(s, k, tclk=TCLK, n_samples=n).real))
                assert fin_lv == (not math.isnan(obs[k - 1]))

    def test_no_measurable_vs_absent_disagreement_across_the_fold_sweep(self):
        """The TASK-62 polynomial sweep: H2 = a2/2 and H3 = a3/4 are
        frequency-independent, so |harmonic(k)| must be a flat line
        across in-band / partial-fold / total-fold — and equal
        harmonic_levels_db at every point."""
        P = TestPolynomialHarmonicFoldSweep()
        for f0_over_fs in (0.049, 0.171, 0.220, 0.342, 0.464):
            sig, kf = P._signal(f0_over_fs)
            K = dict(tclk=P.TCLK, n_samples=P.N)
            lv = harmonic_levels_db(sig, **K, n_harmonics=3)
            h1 = harmonic(sig, 1, **K)
            for k, amp in ((2, P.A2 / 2), (3, P.A3 / 4)):
                hk = harmonic(sig, k, **K)
                assert abs(hk) / abs(h1) == pytest.approx(amp / (1 + 3 * P.A3 / 4), rel=1e-6)
                assert 20 * math.log10(abs(hk) / abs(h1)) == pytest.approx(lv[k - 2], abs=1e-9)
            fr = harmonic_freqs(sig, **K, n=3)
            assert np.allclose(fr, np.arange(1, 4) * kf / (P.N * P.TCLK))
            obs = harmonic_freqs(sig, **K, n=3, observed=True)
            assert np.all(np.isfinite(obs)) and np.all(obs <= 0.5 / P.TCLK)


# ---------------------------------------------------------------------------
# F12 — HIGH: gain_bw_product discards the imaginary part of a complex AC y
# ---------------------------------------------------------------------------

class TestF12GainBwProductComplexCast:
    """``np.abs(np.asarray(sig.y, dtype=float))`` used to cast complex to
    float (ComplexWarning) BEFORE taking the magnitude, so the −3 dB
    search ran on |Re H| = A0/(1+(f/fp)²), whose 3 dB point is at
    0.64·fp. Every ngspice/Xyce AC Signal is complex: GBW read 36 % low
    on a single pole. The pre-fix test used real magnitude data and
    never saw it.

    FIXED (TASK-63): ``np.abs`` is now taken on the complex ``y``
    directly — no cast, no warning, correct magnitude.
    """

    def test_complex_signal_no_longer_warns_or_reads_low(self):
        f = np.logspace(1, 7, 1201)
        h = 100.0 / (1 + 1j * f / 1e4)
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            got = gain_bw_product(_freq(f, h))
        assert got == pytest.approx(1e6, rel=0.012)
        assert got == pytest.approx(gain_bw_product(_freq(f, np.abs(h))), rel=1e-9)


class TestTask78NanInConsumedSamples:
    """TASK-78 (2026-09-04): the 25 ``nan``-column cells of the edge matrix
    that TASK-59 closed without owning, triaged one by one.

    The matrix fixture puts its one NaN at index ``len // 5``. For a POINT
    or WINDOW read that is a sample the metric never looks at, so a finite
    answer there is honest, not a defect — the verdict for those needs the
    NaN placed ON the sample that is read, which is what this class does.
    Three verdicts came out:

    (a) FIXED — a finite or empty wrong answer once the NaN was in the
        consumed samples: ``tie``/``allan_deviation``/``cycle_frequency``
        (empty Signal, indistinguishable from "too few edges"),
        ``extract_code_transitions`` (``rint(nan).astype(int)`` garbage ->
        empty), ``missing_codes`` (``[]`` = "no missing codes", a pass),
        ``sfdr`` (fundamental/spur = bins 1 and 2 of an all-NaN FFT),
        ``eye_diagram`` (height 0 / width 0), ``passband_ripple_db``
        (``band > 0`` silently dropped the NaN), ``unity_gain_freq``
        (crossing skipped -> 0), ``gain_bw_product`` (bandwidth one grid
        point low), ``one_over_f_corner`` (knee one grid point high).
        ``phase_margin`` is a knock-on: with ``unity_gain_freq`` now nan its
        ``argmin(|x - nan|)`` read the lowest sweep point.
    (b) BY DESIGN — point/window reads that propagate nan when the NaN is
        what they read and are unaffected otherwise: ``value_at``,
        ``clip``,
        ``dc_gain_dB``, ``gain_at_freq``, ``phase_at_freq``, ``cmrr_dB``;
        ``is_monotonic`` returns False, the
        always-failing direction (decision-1 §1).
    (c) ALREADY HONEST — ``dnl``, ``inl``, ``evm_constellation`` carry the
        nan in a headline report field; the matrix classifier now sees it.

    Every metric here is also proven never to raise on NaN (decision-1).
    """

    T = 1e-9

    def _clock(self, nan_idx=None):
        t = np.linspace(0, 64 * self.T, 64 * 50 + 1)
        y = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / self.T + 1e-12))
        if nan_idx is not None:
            y[nan_idx] = np.nan
        return _tran(t, y)

    def _ac(self, nan_idx=None):
        f = np.logspace(1, 7, 121)
        h = 100.0 / (1 + 1j * f / 1e4)
        if nan_idx is not None:
            h[nan_idx] = np.nan
        return _freq(f, h)

    def _codes(self, nan_idx=None):
        y = np.arange(8) * 0.125
        if nan_idx is not None:
            y[nan_idx] = np.nan
        return Signal('c', 'voltage', 'code', 'V', np.arange(8, dtype=float), y)

    def _noise(self, nan_idx=None):
        f = np.logspace(1, 7, 121)
        y = np.sqrt(1e-12 * (1 + 1e3 / f))
        if nan_idx is not None:
            y[nan_idx] = np.nan
        return _freq(f, y)

    def _sine(self, nan_idx=None):
        n = 1024
        y = np.sin(2 * np.pi * 11 * np.arange(n) / n)
        if nan_idx is not None:
            y[nan_idx] = np.nan
        return _tran(np.arange(n) * self.T, y)

    AT_5T = 250        # clock sample exactly at t = 5T (read by the point metrics)
    ELSEWHERE = 640    # the matrix fixture's NaN position (len // 5): never read

    # ---------------------------------------------------------- (a) fixed

    def test_edge_signal_metrics_return_one_sample_nan_signal_not_empty(self):
        """An empty Signal already means "fewer than 2 (4) real edges" — the
        flat-trace answer — so it cannot double as "unknown"."""
        s = self._clock(self.ELSEWHERE)
        for fn in (M.tie, M.allan_deviation, M.cycle_frequency):
            r = fn(s)
            assert len(r.y) == 1 and math.isnan(r.y[0]) and math.isnan(r.x[0]), fn.__name__
        flat = _tran(np.linspace(0, 5e-9, 101), np.ones(101))
        for fn in (M.tie, M.allan_deviation, M.cycle_frequency):
            assert len(fn(flat).y) == 0, fn.__name__       # still the honest empty
        assert math.isnan(M.long_term_jitter(s))            # inherits from tie

    def test_extract_code_transitions_keeps_the_observed_axis_and_nans_every_level(self):
        x = np.linspace(0, 1, 401)
        code = np.floor(x * 4).clip(0, 3)
        v = _tran(x, x)
        clean = M.extract_code_transitions(v, _tran(x, code))
        assert list(clean.x) == [1.0, 2.0, 3.0]
        for idx in (80, 400):                                # a low code, the highest code
            c = code.copy(); c[idx] = np.nan
            r = M.extract_code_transitions(v, _tran(x, c))
            assert list(r.x) == [1.0, 2.0, 3.0]
            assert np.all(np.isnan(r.y))
        # all-NaN codes: one-row sentinel, never a "no rows" table
        r = M.extract_code_transitions(v, _tran(x, np.full(401, np.nan)))
        assert len(r.x) == 1 and math.isnan(r.x[0]) and math.isnan(r.y[0])
        # a NaN in v_in is a point read: only the transition that samples it
        vn = x.copy(); vn[100] = np.nan                      # first sample with code >= 1
        r = M.extract_code_transitions(_tran(x, vn), _tran(x, code))
        assert math.isnan(r.y[0]) and not np.isnan(r.y[1:]).any()
        # the downstream chain gets honest nan reports, not a raise
        t = M.extract_code_transitions(v, _tran(x, c))
        assert math.isnan(M.inl(t).max_abs_lsb) and math.isnan(M.dnl(t).max_abs_lsb)
        assert M.is_monotonic(t) is False
        assert math.isnan(M.missing_codes(t)[0])

    def test_missing_codes_returns_nan_sentinel_not_empty(self):
        for idx in (1, 7):
            r = M.missing_codes(self._codes(idx))
            assert len(r) == 1 and math.isnan(r[0])
        assert len(M.missing_codes(self._codes())) == 0     # genuinely none missing

    def test_sfdr_nans_every_headline_field(self):
        r = M.sfdr(self._sine(204), tclk=self.T, n_samples=1024)
        assert math.isnan(r.sfdr_db) and math.isnan(r.fundamental_hz) and math.isnan(r.spur_hz)
        assert r.bin_hz == pytest.approx(1.0 / (1024 * self.T))   # sampling property, kept
        clean = M.sfdr(self._sine(), tclk=self.T, n_samples=1024)
        assert clean.fundamental_hz == pytest.approx(11 / (1024 * self.T))

    # --- BEGIN commented-out (TASK-226.11): eye_diagram NaN-propagation
    # test, superseded by TASK-226.12+. Kept for reference until the
    # replacement is accepted; do not delete. ---
    # def test_eye_diagram_nans_height_width_jitter(self):
    #     for idx in (3 * 50 + 25, 3 * 50 + 1, 3 * 50):        # centre sample, crossing, boundary
    #         r = M.eye_diagram(self._clock(idx), period=self.T)
    #         assert math.isnan(r.height) and math.isnan(r.width) and math.isnan(r.jitter_rms)
    #         assert r.traces.shape == (64, 128)                 # the fold is still returned
    #     clean = M.eye_diagram(self._clock(), period=self.T)
    #     # TASK-226.8: this clock's UNIT INTERVAL is T/2, so folding on T
    #     # puts two transition clusters and two openings in one window and
    #     # the metric returns the widest of them. The anchor is that a
    #     # clean fold still returns a real number where the NaN folds
    #     # above return nan; the number itself moved from T to T/2 with
    #     # the definition.
    #     assert clean.width == pytest.approx(self.T / 2)
    #     assert M.eye_diagram(self._clock(), period=self.T / 2).width == \
    #         pytest.approx(self.T / 2)
    # --- END commented-out (TASK-226.11) ---

    def test_passband_ripple_nan_inside_the_band_propagates_outside_is_ignored(self):
        clean = M.passband_ripple_db(self._ac(), 1e3)
        assert math.isnan(M.passband_ripple_db(self._ac(24), 1e3))    # f = 158 Hz, in band
        assert math.isnan(M.passband_ripple_db(self._ac(0), 1e3))     # the band's own min
        assert M.passband_ripple_db(self._ac(100), 1e3) == clean      # f = 1 MHz, outside

    def test_unity_gain_freq_nan_on_the_crossing_segment(self):
        f = np.logspace(1, 7, 121)
        mag = np.abs(100.0 / (1 + 1j * f / 1e4))
        ci = int(np.argmax(mag < 1.0))
        assert mag[ci - 1] > 1.0 > mag[ci]                    # the bracket
        assert M.unity_gain_freq(self._ac()) == pytest.approx(1e6, rel=1e-3)
        for idx in (ci - 1, ci):
            assert math.isnan(M.unity_gain_freq(self._ac(idx)))   # was 0.0
        assert math.isnan(M.phase_margin(self._ac(ci)))           # was ~180 (knock-on)

    def test_gain_bw_product_nan_on_the_bandwidth_grid_point(self):
        f = np.logspace(1, 7, 121)
        mag = np.abs(100.0 / (1 + 1j * f / 1e4))
        bi = int(np.where(mag >= mag[0] / np.sqrt(2))[0][-1])
        assert M.gain_bw_product(self._ac()) == pytest.approx(1e6, rel=1e-3)
        assert math.isnan(M.gain_bw_product(self._ac(bi)))     # was 11 % low
        assert math.isnan(M.gain_bw_product(self._ac(24)))     # uniform: any NaN

    def test_ac_guards_see_a_nan_in_the_imaginary_part_only(self):
        """AC y is complex; the transient ``_any_nan`` casts to float and
        would drop an imaginary-only NaN (numpy warns). Both guards test
        the magnitude instead, without complex-to-real warnings."""
        from silk.metrics.diagnostics import MetricWarning
        f = np.logspace(1, 7, 121)
        h = 100.0 / (1 + 1j * f / 1e4)
        h[24] = h[24].real + 1j * np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.warns(MetricWarning, match='consumed passband magnitude') as caught:
                assert math.isnan(M.passband_ripple_db(_freq(f, h), 1e3))
            assert len(caught) == 1
            with pytest.warns(MetricWarning, match='consumed magnitude') as caught:
                assert math.isnan(M.gain_bw_product(_freq(f, h)))
            assert len(caught) == 1
            with pytest.warns(MetricWarning, match='consumed magnitude'):
                assert math.isnan(M.unity_gain_freq(_freq(f, h)))

    def test_ac_guards_catch_nan_paired_with_infinity_but_not_a_lone_infinity(self):
        """TASK-178: folding to magnitude before the NaN check (the fix
        above) has its own gap -- ``abs(inf+nanj)`` and ``abs(nan+infj)``
        are both ``inf``, not ``nan``, so a sample with one infinite part
        and one NaN part used to sail through every guard in this module
        unnoticed. Checking the real and imaginary parts separately
        catches both orderings, and a lone infinity (no NaN in either
        part) must still be left alone -- infinity is legitimate AC data
        (a zero at DC reads -inf dB; a pole at DC reads infinite
        magnitude), never bad data on its own (owner ruling)."""
        from silk.metrics.diagnostics import MetricWarning
        f = np.logspace(1, 7, 121)
        h = 100.0 / (1 + 1j * f / 1e4)

        for planted in (complex(np.inf, np.nan), complex(np.nan, np.inf)):
            bad = h.copy()
            bad[24] = planted
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                with pytest.warns(MetricWarning, match='consumed passband magnitude') as caught:
                    assert math.isnan(M.passband_ripple_db(_freq(f, bad), 1e3))
                assert len(caught) == 1
                with pytest.warns(MetricWarning, match='consumed magnitude') as caught:
                    assert math.isnan(M.gain_bw_product(_freq(f, bad)))
                assert len(caught) == 1
                with pytest.warns(MetricWarning, match='consumed magnitude'):
                    assert math.isnan(M.unity_gain_freq(_freq(f, bad)))
                with pytest.warns(MetricWarning, match='consumed magnitude') as caught:
                    assert math.isnan(M.peak_gain_dB(_freq(f, bad)))
                assert len(caught) == 1

        for legit in (complex(np.inf, 0.0), complex(0.0, -np.inf)):
            clean = h.copy()
            clean[24] = legit
            with warnings.catch_warnings():
                warnings.simplefilter('error')       # any warning fails the test
                got = M.gain_bw_product(_freq(f, clean))
            assert got == pytest.approx(M.gain_bw_product(_freq(f, h)))

    def test_one_over_f_corner_nan_on_the_knee(self):
        f = np.logspace(1, 7, 121)
        fc = M.one_over_f_corner(self._noise())
        assert fc == pytest.approx(1e3)
        ki = int(np.argmin(np.abs(f - fc)))
        assert math.isnan(M.one_over_f_corner(self._noise(ki)))   # was one grid point high
        assert math.isnan(M.one_over_f_corner(self._noise(5)))

    def test_inl_best_fit_propagates_nan_without_reaching_polyfit(self):
        r = M.inl(self._codes(3), method='best_fit')
        assert math.isnan(r.max_abs_lsb) and np.all(np.isnan(r.inl_lsb)) and r.method == 'best_fit'
        with pytest.raises(ValueError, match='method must be'):
            M.inl(self._codes(3), method='bogus')             # caller error still first

    # ---------------------------------------------------- (b) by design

    def test_transient_point_reads_propagate_nan_only_from_the_sample_they_read(self):
        on, off = self._clock(self.AT_5T), self._clock(self.ELSEWHERE)
        clean = self._clock()
        reads = (
            lambda s: M.value_at(s, 5 * self.T),
        )
        for read in reads:
            assert math.isnan(read(on))
            assert read(off) == read(clean)
        # value_at between two grid points: either bracketing NaN poisons it
        dt = self.T / 50
        assert math.isnan(M.value_at(self._clock(self.AT_5T), 5 * self.T + 0.5 * dt))
        assert math.isnan(M.value_at(self._clock(self.AT_5T + 1), 5 * self.T + 0.5 * dt))

    def test_clip_keeps_a_nan_inside_the_window_and_never_sees_one_outside(self):
        inside = M.window(self._clock(self.AT_5T), 0, 5 * self.T)
        assert int(np.isnan(inside.y).sum()) == 1
        outside = M.window(self._clock(self.ELSEWHERE), 0, 5 * self.T)
        assert not np.isnan(outside.y).any() and len(outside.y) == 251

    def test_ac_point_reads_propagate_nan_only_from_the_sample_they_read(self):
        f = np.logspace(1, 7, 121)
        near = int(np.argmin(np.abs(f - 1e3)))
        assert math.isnan(M.dc_gain_dB(self._ac(0)))
        assert M.dc_gain_dB(self._ac(24)) == M.dc_gain_dB(self._ac())
        assert math.isnan(M.gain_at_freq(self._ac(near), 1e3))
        assert M.gain_at_freq(self._ac(24), 1e3) == M.gain_at_freq(self._ac(), 1e3)
        assert math.isnan(M.phase_at_freq(self._ac(near), 1e3))
        assert M.phase_at_freq(self._ac(24), 1e3) == M.phase_at_freq(self._ac(), 1e3)
        assert math.isnan(M.cmrr_dB(self._ac(0), self._ac()))
        assert math.isnan(M.cmrr_dB(self._ac(), self._ac(0)))
        assert M.cmrr_dB(self._ac(24), self._ac(24)) == M.cmrr_dB(self._ac(), self._ac())

    def test_is_monotonic_is_false_the_always_failing_direction(self):
        assert M.is_monotonic(self._codes()) is True
        assert M.is_monotonic(self._codes(3)) is False

    # ------------------------------------------------- (c) already honest

    def test_reports_carry_the_nan_in_a_headline_field(self):
        assert math.isnan(M.dnl(self._codes(1)).max_abs_lsb)
        assert math.isnan(M.inl(self._codes(1)).max_abs_lsb)
        i, q = self._sine(204), self._sine()
        r = M.evm_constellation(i, q, symbol_period=10 * self.T, ideal_points=[1 + 1j, -1 - 1j])
        assert math.isnan(r.evm_percent)

    # ------------------------------------------------------- decision-1

    def test_none_of_the_24_raise_on_nan(self):
        c, a, k, n, s = (self._clock(self.AT_5T), self._ac(0), self._codes(1),
                         self._noise(40), self._sine(204))
        x = np.linspace(0, 1, 401)
        code = np.floor(x * 4).clip(0, 3); code[80] = np.nan
        calls = (
            lambda: M.allan_deviation(c), lambda: M.window(c, 0, 5 * self.T),
            lambda: M.cmrr_dB(a, a), lambda: M.dc_gain_dB(a), lambda: M.dnl(k),
            lambda: M.evm_constellation(s, s, symbol_period=10 * self.T, ideal_points=[1 + 1j]),
            lambda: M.extract_code_transitions(_tran(x, x), _tran(x, code)),
            # eye_diagram withdrawn pending replacement (TASK-226.11/.12):
            # lambda: M.eye_diagram(c, period=self.T),
            lambda: M.gain_at_freq(a, 1e3),
            lambda: M.gain_bw_product(a),
            lambda: M.inl(k), lambda: M.inl(k, method='best_fit'),
            lambda: M.cycle_frequency(c), lambda: M.is_monotonic(k),
            lambda: M.missing_codes(k),
            lambda: M.one_over_f_corner(n), lambda: M.passband_ripple_db(a, 1e3),
            lambda: M.phase_at_freq(a, 1e3),
            lambda: M.sfdr(s, tclk=self.T, n_samples=1024),
            lambda: M.tie(c),
            lambda: M.unity_gain_freq(a), lambda: M.value_at(c, 5 * self.T),
            lambda: M.phase_margin(a),
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for call in calls:
                call()   # must not raise


class TestTask86GainMarginNanSample:
    """gain_margins()/gain_margin() (2026-09-04): a NaN sample used to raise
    an untyped-message ``ValueError('cannot convert float NaN to integer')``
    from ``int(np.ceil((lo + 180) / 360))`` once the unwrapped phase went
    NaN from that sample on — a raise on DEGENERATE DATA (decision-1
    category 1: one corner's aborted tail), which would abort a whole PVT
    campaign instead of recording that corner as failing. Ruled (task
    Implementation Notes): ``gain_margins`` returns the one-element
    sentinel ``[(nan, nan)]`` — never confused with ``[]`` ("no
    crossover", a real result for a loop that never inverts phase) — and
    ``gain_margin`` returns ``nan``, not the ``+inf`` its empty-crossings
    branch would otherwise give for the sentinel.
    """

    def _ac(self, nan_idx=None):
        f = np.logspace(1, 7, 121)
        h = 100.0 / (1 + 1j * f / 1e4)          # single pole: phase -> -90 deg, never crosses
        if nan_idx is not None:
            h[nan_idx] = np.nan
        return _freq(f, h)

    def _ac_crossing(self, nan_idx=None):
        f = np.logspace(1, 7, 121)              # 3-pole: has a real phase crossover
        h = 1e6 / ((1 + 1j * f / 1e2) * (1 + 1j * f / 1e5) * (1 + 1j * f / 1e6))
        if nan_idx is not None:
            h[nan_idx] = np.nan
        return _freq(f, h)

    def test_gain_margins_returns_one_element_nan_sentinel_not_raise(self):
        """Pre-fix: ValueError from the crossover-level search. Post-fix:
        a one-element [(nan, nan)] list — never the [] a genuinely
        crossover-free corner returns (the single-pole fixture with no NaN
        legitimately returns [])."""
        r = M.gain_margins(self._ac(24))
        assert len(r) == 1
        assert math.isnan(r[0][0]) and math.isnan(r[0][1])
        assert M.gain_margins(self._ac()) == []          # the real "no crossover" case

    def test_gain_margins_sentinel_independent_of_where_the_crossing_is(self):
        """A NaN sample anywhere makes the unwrapped phase unknown from
        that point on, so a signal that DOES have a real crossover still
        collapses to the sentinel rather than a spuriously-precise number —
        same whole-trace-unknown rule as crossings()/phase_margin()."""
        clean = M.gain_margins(self._ac_crossing())
        assert len(clean) == 1 and math.isfinite(clean[0][0]) and math.isfinite(clean[0][1])
        r = M.gain_margins(self._ac_crossing(24))
        assert len(r) == 1 and math.isnan(r[0][0]) and math.isnan(r[0][1])

    def test_gain_margin_returns_nan_not_the_no_crossover_inf(self):
        """Pre-fix: the same inherited ValueError. A naive fix that treated
        the [(nan, nan)] sentinel as an ordinary empty-crossings result
        would return +inf (gain_margin's unbounded-margin case) instead of
        nan — silently reporting a corner with an unmeasurable margin as
        the SAFEST possible loop."""
        assert math.isnan(M.gain_margin(self._ac(24)))
        assert math.isnan(M.gain_margin(self._ac_crossing(24)))

    def test_gain_margin_still_reports_the_two_real_no_crossover_cases(self):
        """Guard against the fix disturbing gain_margin's existing
        (non-NaN) branches: +inf when the loop rolls off before its phase
        ever inverts, nan only for a NaN sample or a truncated sweep."""
        assert M.gain_margin(self._ac()) == float('inf')

    def test_phase_margin_and_unity_gain_freq_unchanged(self):
        """TASK-86 touches gain_margins/gain_margin only — phase_margin's
        and unity_gain_freq's TASK-78 nan guards must still see the same
        NaN sample the same way."""
        s = self._ac(24)
        assert math.isnan(M.unity_gain_freq(s))
        assert math.isnan(M.phase_margin(s))

    def test_neither_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            M.gain_margins(self._ac(24))
            M.gain_margin(self._ac(24))
            M.gain_margins(self._ac_crossing(24))
            M.gain_margin(self._ac_crossing(24))
