"""Tests for silk.metrics.noise — closed-form noise helpers."""
import math
import warnings
import numpy as np
import pytest

from silk.signals.base import Signal
from silk.metrics.noise import (
    noise_figure,
    noise_figure_from_factor,
    input_referred_noise,
    sum_uncorrelated,
    kT_over_C,
    one_over_f_corner,
    cascade_noise_figure,
)
from silk.metrics.converter import (
    aperture_jitter_snr_db,
)


class TestNoiseFigure:
    def test_known_difference(self):
        assert math.isclose(noise_figure(40.0, 35.0), 5.0, rel_tol=1e-12)

    def test_zero_for_lossless(self):
        assert noise_figure(20.0, 20.0) == 0.0


class TestNoiseFigureFromFactor:
    def test_unity(self):
        assert noise_figure_from_factor(1.0) == 0.0

    def test_factor_2_is_3db(self):
        assert math.isclose(noise_figure_from_factor(2.0), 3.0103, rel_tol=1e-3)

    def test_invalid(self):
        with pytest.raises(ValueError, match='must be > 0'):
            noise_figure_from_factor(0.0)


class TestInputReferredNoise:
    def test_division(self):
        assert math.isclose(input_referred_noise(1e-3, 100.0), 1e-5, rel_tol=1e-12)

    def test_negative_gain(self):
        assert math.isclose(input_referred_noise(1e-3, -100.0), 1e-5, rel_tol=1e-12)

    def test_zero_gain(self):
        with pytest.raises(ValueError, match='non-zero'):
            input_referred_noise(1e-3, 0.0)


class TestSumUncorrelated:
    def test_three_equal(self):
        assert math.isclose(sum_uncorrelated(1e-6, 1e-6, 1e-6),
                            1e-6 * math.sqrt(3.0), rel_tol=1e-12)

    def test_one_dominates(self):
        assert math.isclose(sum_uncorrelated(1e-3, 1e-6), 1e-3, rel_tol=1e-6)

    def test_empty(self):
        assert sum_uncorrelated() == 0.0


class TestKtOverC:
    def test_1pf_at_300k(self):
        result = kT_over_C(1e-12, temp_k=300.0)
        assert 60e-6 < result < 70e-6

    def test_default_temp(self):
        assert kT_over_C(1e-12) == kT_over_C(1e-12, temp_k=300.0)

    def test_smaller_cap_more_noise(self):
        assert kT_over_C(1e-15) > kT_over_C(1e-12)

    def test_invalid_cap(self):
        with pytest.raises(ValueError, match='capacitance must be > 0'):
            kT_over_C(0.0)

    def test_invalid_temp(self):
        with pytest.raises(ValueError, match='temperature must be > 0'):
            kT_over_C(1e-12, temp_k=0)


class TestApertureJitterSnrDb:
    def test_zero_jitter(self):
        assert aperture_jitter_snr_db(1e9, 0.0) == float('inf')

    def test_known(self):
        result = aperture_jitter_snr_db(1e9, 1e-12)
        expected = -20.0 * math.log10(2.0 * math.pi * 1e9 * 1e-12)
        assert math.isclose(result, expected, rel_tol=1e-12)

    def test_invalid_freq(self):
        with pytest.raises(ValueError, match='f_in must be > 0'):
            aperture_jitter_snr_db(0, 1e-12)

    def test_negative_jitter(self):
        with pytest.raises(ValueError, match='sigma_t must be >= 0'):
            aperture_jitter_snr_db(1e9, -1e-12)


class TestImports:
    def test_top_level(self):
        from silk.metrics import (
            noise_figure, kT_over_C, aperture_jitter_snr_db,
        )
        assert callable(noise_figure)


# ---------------------------------------------------------------------------
# Tier 3 — vendor-parity noise additions
# ---------------------------------------------------------------------------

def _L_signal(freqs, L_dBc):
    return Signal('L', 'phase_noise', 'frequency', 'dBc/Hz',
                  np.asarray(freqs, dtype=float),
                  np.asarray(L_dBc, dtype=float))


def _noise_signal(freqs, density):
    return Signal('vn', 'noise', 'frequency', 'V/sqrt(Hz)',
                  np.asarray(freqs, dtype=float),
                  np.asarray(density, dtype=float))


class TestOneOverFCorner:
    def test_known_corner(self):
        # Build noise: 1/sqrt(f) * A + white floor B
        # White floor B = 1e-9 V/sqrt(Hz)
        # 1/f part: A/sqrt(f). Corner at f_c where A/sqrt(f_c) = B → A = B·sqrt(f_c)
        # Pick f_c = 1 kHz, B = 1e-9: A = 1e-9 * sqrt(1000) ≈ 3.16e-8
        f = np.logspace(0, 6, 10000)
        B = 1e-9
        f_c_target = 1e3
        A = B * math.sqrt(f_c_target)
        # Total: sqrt(B² + A²/f)
        y = np.sqrt(B ** 2 + A ** 2 / f)
        sig = _noise_signal(f, y)
        f_c = one_over_f_corner(sig)
        # Should be within 50% of target (algorithm picks first crossing
        # below sqrt(2)*B, finite resolution)
        assert 0.5 * f_c_target < f_c < 2.0 * f_c_target

    def test_pure_white_no_corner(self):
        # All-white noise → never crosses below sqrt(2)*floor
        # floor = median of upper half ≈ 1e-9; threshold = sqrt(2)·1e-9 ≈ 1.414e-9
        # All samples equal 1e-9 < 1.414e-9 → first below is index 0 → returns f[0].
        # Acceptable: corner is at the very low end, indicating no clear knee.
        f = np.logspace(0, 6, 1000)
        y = np.full_like(f, 1e-9)
        result = one_over_f_corner(_noise_signal(f, y))
        # The function's "below threshold" trips at f[0] for pure-white. That's
        # not "nan" but it does indicate the corner is at/below the lowest
        # frequency probed — the user can interpret accordingly. Just check
        # the function returns something finite and small.
        assert result == float(f[0]) or math.isnan(result)

    def test_too_few_samples(self):
        sig = _noise_signal([1.0, 2.0], [1e-9, 1e-9])
        assert math.isnan(one_over_f_corner(sig))


class TestTier3NoiseImports:
    def test_top_level(self):
        from silk.metrics import one_over_f_corner
        assert callable(one_over_f_corner)


# ---------------------------------------------------------------------------
# Beyond-vendor — cascade noise figure (Friis)
# ---------------------------------------------------------------------------

class TestCascadeNoiseFigure:
    def test_single_stage(self):
        # One stage → just that stage's NF
        result = cascade_noise_figure([3.0], [10.0])
        assert math.isclose(result, 3.0, rel_tol=1e-9)

    def test_classic_lna_first(self):
        # LNA with NF=2 dB, gain=20 dB, then mixer NF=10 dB, gain=0 dB.
        # F1 = 10^(0.2) ≈ 1.585; G1 = 10^2 = 100
        # F2 = 10^1 = 10
        # F_tot = 1.585 + (10-1)/100 = 1.585 + 0.09 = 1.675
        # NF_tot = 10·log10(1.675) ≈ 2.24 dB
        result = cascade_noise_figure([2.0, 10.0], [20.0, 0.0])
        assert math.isclose(result, 2.24, abs_tol=0.05)

    def test_no_lna_huge_degradation(self):
        # Mixer first: high NF dominates
        # F1 = 10, G1 = 1; F2 = 1.585, G2 = 100
        # F_tot = 10 + 0.585/1 = 10.585
        # NF_tot = 10·log10(10.585) ≈ 10.25 dB
        result = cascade_noise_figure([10.0, 2.0], [0.0, 20.0])
        assert math.isclose(result, 10.25, abs_tol=0.05)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError, match='same shape'):
            cascade_noise_figure([1.0, 2.0], [10.0])


class TestBeyondVendorNoiseImports:
    def test_top_level(self):
        from silk.metrics import cascade_noise_figure
        assert callable(cascade_noise_figure)


# ---------------------------------------------------------------------------
# TASK-197 (T-E) — the xlabel='frequency' cross-kind guards
# ---------------------------------------------------------------------------

F197 = np.logspace(1, 6, 201)


def _other_real_freq_signal197():
    """Some other real-valued frequency-domain data: -80 at 10 Hz, -20
    dB/dec, floored at -140 — the shape of a published phase-noise table
    (TASK-210 retired the metric that took one), kept here only as a stand-in
    for "real y on the frequency axis that is not a noise density"."""
    return _L_signal(F197, np.maximum(-80.0 - 20.0 * np.log10(F197 / 10.0), -140.0))


def _density197():
    """V/sqrt(Hz): a 1 kHz 1/f corner over a 1e-9 white floor."""
    return _noise_signal(F197, 1e-9 * np.sqrt(1.0 + 1e3 / F197))


def _ac_response197():
    """The complex single pole every simulator .ac sweep loads as."""
    return Signal('vout', 'voltage', 'frequency', 'V',
                  F197, 100.0 / (1.0 + 1j * F197 / 1e4))


class TestTask197CrossKindGuards:
    """Design §3.4's cross-kind table, one test per row.

    Two kinds share the ``xlabel='frequency'`` axis — a ``.noise`` density
    and a frequency response — and until TASK-197 both metrics here took
    whichever they were handed and answered with a confident number. What
    SEPARATES them observably is the dtype of ``y`` and nothing else:
    ``.ac`` is the only producer of complex samples, so a complex ``y``
    RULES OUT the noise kind and those rows now raise. Real-vs-real is
    UNDECIDABLE — a Signal carries no producer-stamped kind (owner ruling
    D6, deferred) — so those rows still return their wrong numbers,
    deliberately, and are pinned here so the D6 follow-on has to come back
    and change them.

    (A third kind, ``phase_noise_sideband``, used to share this axis too;
    TASK-210 retired its one consumer, ``phase_noise_at_offset``, and the
    kind itself, since no silk analysis has ever produced one.)
    """

    # --- CLOSED: a complex y rules the kind out ---------------------------

    def test_integrated_noise_refuses_a_complex_ac_response(self):
        # Was 1.25e+04 "V" -- sqrt(int |H|^2 df) over a transfer function.
        from silk.metrics.noise import integrated_noise
        with pytest.raises(ValueError) as e:
            integrated_noise(_ac_response197(), 10.0, 1e6)
        assert 'integrated_noise' in str(e.value)
        assert 'must be a noise_density Signal' in str(e.value)
        assert 'complex y' in str(e.value)

    def test_one_over_f_corner_refuses_a_complex_ac_response(self):
        # Was 4.73e+04 Hz -- a corner read off Re(H), behind a ComplexWarning.
        with pytest.raises(ValueError) as e:
            one_over_f_corner(_ac_response197())
        assert 'one_over_f_corner' in str(e.value)
        assert 'must be a noise_density Signal' in str(e.value)
        assert 'complex y' in str(e.value)

    def test_the_time_axis_refusal_now_names_the_kind(self):
        t = np.linspace(0.0, 1e-6, 101)
        tran = Signal('v', 'voltage', 'time', 'V', t, np.sin(t))
        with pytest.raises(ValueError) as e:
            one_over_f_corner(tran)
        assert "must be a noise_density Signal" in str(e.value)
        assert "xlabel='time'" in str(e.value)

    # --- OPEN: real vs real is UNDECIDABLE without a stamped kind ---------
    #
    # These assert the WRONG number on purpose. They are the T-E residue
    # handed to the D6 follow-on (TASK-182's remaining half): the day a
    # Signal carries the kind its producer stamped on it, each of these
    # becomes a refusal and this block gets rewritten.

    def test_UNDECIDABLE_integrated_noise_still_takes_other_real_freq_data(self):
        from silk.metrics.noise import integrated_noise
        got = integrated_noise(_other_real_freq_signal197(), 10.0, 1e6)
        assert got == pytest.approx(139918.30084739963)       # "volts", from dBc/Hz-shaped data

    def test_UNDECIDABLE_one_over_f_corner_still_takes_other_real_freq_data(self):
        assert math.isnan(one_over_f_corner(_other_real_freq_signal197()))

    def test_UNDECIDABLE_gain_at_freq_still_takes_a_density(self):
        # The ac half of the family. gain_at_freq declares a
        # frequency_response, and the reason this one cannot be closed is
        # the complex-y question settled in the other direction (T-E
        # investigation): silk's OWN pinned contract requires nine AC
        # metrics to accept a real magnitude Signal and return the same
        # answer as on the complex one (test_rawfile_realism.py's F12
        # cross-check), so frequency_response must NOT require complex y,
        # and a real density is then indistinguishable from a magnitude.
        from silk.metrics.ac import gain_at_freq
        assert gain_at_freq(_density197(), 1e4) == pytest.approx(1.0488088481701516e-09)


class TestNoiseRequestDiagnostics:
    def test_mismatched_array_context(self):
        with pytest.raises(ValueError) as caught:
            cascade_noise_figure([1, 2], [10])
        message = str(caught.value)
        assert 'cascade_noise_figure' in message and 'same shape' in message
        assert str(np.shape([1, 2])) in message and str(np.shape([10])) in message

    @pytest.mark.parametrize('metric,args', [
        (cascade_noise_figure, (1, 10)),
        (cascade_noise_figure, ([[1, 2]], [[10, 20]])),
    ])
    def test_raw_arrays_are_one_dimensional(self, metric, args):
        with pytest.raises(ValueError, match=metric.__name__ + '.*one-dimensional'):
            metric(*args)

    @pytest.mark.parametrize('metric,args,facts', [
        (noise_figure_from_factor, (0,), ('noise factor must be > 0', '0')),
        (input_referred_noise, (1e-3, 0), ('gain_vv must be non-zero', '0.001', '0')),
        (kT_over_C, (0,), ('capacitance must be > 0', '0', 'F')),
        (kT_over_C, (1e-12, 0), ('temperature must be > 0', '0', 'K')),
        (aperture_jitter_snr_db, (0, 1e-12), ('f_in must be > 0', '0', 'Hz')),
        (aperture_jitter_snr_db, (1e6, -1), ('sigma_t must be >= 0', '-1', 's')),
        (cascade_noise_figure, ([], []), ('at least 1 stage', '0')),
    ])
    def test_scalar_and_empty_stage_context(self, metric, args, facts):
        with pytest.raises(ValueError) as caught:
            metric(*args)
        message = str(caught.value)
        assert metric.__name__ in message
        assert all(fact in message for fact in facts)

    @pytest.mark.parametrize('nfs,stage,nf', [
        ([-10, -10], 1, -10), ([-1, 10], 1, -1), ([1, -0.1], 2, -0.1),
    ])
    def test_negative_stage_nf_is_caller_error(self, nfs, stage, nf):
        with pytest.raises(ValueError) as caught:
            cascade_noise_figure(nfs, [0, 0])
        message = str(caught.value)
        assert 'cascade_noise_figure' in message
        assert f'stage {stage}' in message
        assert str(nf) in message and '>= 0' in message and 'dB' in message


class TestNoisePreservedContracts:
    def test_valid_limits_and_attenuators_are_quiet(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            assert cascade_noise_figure([0, 0], [-10, 0]) == 0.0
            expected_nf = 10 * math.log10(91)
            assert cascade_noise_figure([0, 10], [-10, 0]) == pytest.approx(expected_nf)
            assert input_referred_noise(0, -10) == 0.0
            assert sum_uncorrelated() == 0.0
            assert aperture_jitter_snr_db(1e6, 0) == math.inf
        assert caught == []

    @pytest.mark.parametrize('freqs,expected', [
        ([-4, -3, -2, -1], np.nan),
    ])
    def test_deferred_corner_policies_are_unchanged(self, freqs, expected):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value = one_over_f_corner(_noise_signal(freqs, [1e-9] * len(freqs)))
        assert math.isnan(value) if math.isnan(expected) else value == expected
        assert caught == []

    def test_bare_scalar_nan_propagation_remains_quiet(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            assert math.isnan(noise_figure_from_factor(np.nan))
            assert math.isnan(input_referred_noise(np.nan, 1))
            assert math.isnan(kT_over_C(np.nan))
            assert math.isnan(aperture_jitter_snr_db(1e6, np.nan))
            assert math.isnan(cascade_noise_figure([np.nan, 1], [0, 0]))
            assert math.isnan(cascade_noise_figure([1, 1], [np.nan, 0]))
        assert caught == []
