"""TASK-414: noise figure from a noise run — spot, band and per-point NF.

Analytic fixtures only (the real-ngspice comparison against
``cs_amp_ac_noise``'s hand-computed factor is the slow test at the end).
"""
import math
import warnings

import numpy as np
import pytest

from silk.metrics.diagnostics import MetricWarning
from silk.metrics.noise import (
    K_BOLTZMANN, band_noise_figure, integrated_noise, noise_density_at_freq,
    noise_figure_at_freq,
    noise_figure_vs_freq,
)
from silk.signals.base import Signal
from silk.utils._compat import trapezoid

RS = 10e3
T0 = 290.0
T_SIM = 300.15
F = np.logspace(0, 6, 61)


def _density(y, name='noise_in'):
    return Signal(name, 'noise', 'frequency', 'V/sqrt(Hz)', F,
                  np.asarray(y, dtype=float))


def _src(temp):
    """Thermal noise PSD of RS at *temp*."""
    return 4.0 * K_BOLTZMANN * temp * RS


def _run(s_added, gain, temp=T_SIM):
    """noise_in / noise_out of an amplifier with input-referred added PSD
    *s_added* and power gain *gain*, driven through RS at *temp*."""
    s_in = _src(temp) + s_added
    return (_density(np.sqrt(s_in)),
            _density(np.sqrt(s_in * gain), 'noise_out'))


class TestNoiselessAmplifier:
    @pytest.mark.parametrize('temp', [T0, T_SIM, 250.0])
    def test_zero_db_at_any_stated_true_temperature(self, temp):
        nin, nout = _run(np.zeros_like(F), np.full_like(F, 25.0), temp)
        assert noise_figure_at_freq(nin, 1e3, RS, temp) == pytest.approx(0.0, abs=1e-12)
        assert band_noise_figure(nin, nout, 1.0, 1e6, RS, temp) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(noise_figure_vs_freq(nin, RS, temp).y, 0.0, atol=1e-12)

    def test_misstated_temperature_offset(self):
        # A 300.15 K run read as if it were at T0: the documented offset.
        nin, nout = _run(np.zeros_like(F), np.full_like(F, 25.0), T_SIM)
        expected = 10 * math.log10(1 + (T_SIM - T0) / T0)
        assert expected == pytest.approx(0.1490, abs=1e-3)
        assert noise_figure_at_freq(nin, 1e3, RS, T0) == pytest.approx(expected, rel=1e-12)
        assert band_noise_figure(nin, nout, 1.0, 1e6, RS, T0) == pytest.approx(expected, rel=1e-12)

    def test_temp_ref_equal_to_sim_matches_naive_ratio(self):
        nin, _ = _run(np.zeros_like(F), np.ones_like(F), T_SIM)
        naive = 10 * math.log10(float(nin.y[0]) ** 2 / _src(T_SIM))
        assert noise_figure_at_freq(nin, F[0], RS, T_SIM, temp_ref_k=T_SIM) == pytest.approx(naive, abs=1e-12)


class TestKnownAddedNoise:
    def test_spot_closed_form(self):
        added = 3.0 * _src(T0)              # F = 1 + 3 = 4 -> 6.02 dB
        nin, _ = _run(np.full_like(F, added), np.ones_like(F))
        assert noise_figure_at_freq(nin, 1234.5, RS, T_SIM) == pytest.approx(10 * math.log10(4.0), rel=1e-12)

    def test_spot_interpolates_like_noise_density_at_freq(self):
        # Off-grid, the spot NF reads the density exactly as
        # noise_density_at_freq does (log-log between neighbours, TASK-413).
        added = _src(T0) * 1e3 / F
        nin, _ = _run(added, np.ones_like(F))
        fq = 3.3e3
        for interp in (True, False):
            d = noise_density_at_freq(nin, fq, interpolate=interp)
            expected = 10 * math.log10(1 + (d ** 2 - _src(T_SIM)) / _src(T0))
            assert noise_figure_at_freq(nin, fq, RS, T_SIM, interpolate=interp) \
                == pytest.approx(expected, rel=1e-12)
        exact = 10 * math.log10(1 + 1e3 / fq)
        assert noise_figure_at_freq(nin, fq, RS, T_SIM) == pytest.approx(exact, abs=0.01)
        assert noise_figure_at_freq(nin, fq, RS, T_SIM, interpolate=False) \
            != pytest.approx(exact, abs=0.01)

    def test_band_is_gain_weighted_not_unweighted_mean(self):
        # Added noise only where the gain is small: the gain-weighted band
        # factor must discount it; the unweighted mean of F would not.
        added = np.where(F > 1e4, 99.0 * _src(T0), 0.0)
        gain = np.where(F > 1e4, 1e-4, 1.0)
        nin, nout = _run(added, gain)
        x = F
        fs = 1 + added / _src(T0)
        expected = trapezoid(fs * gain, x) / trapezoid(gain, x)
        nf = band_noise_figure(nin, nout, 1.0, 1e6, RS, T_SIM)
        assert nf == pytest.approx(10 * math.log10(expected), rel=1e-9)
        unweighted = 10 * math.log10(np.mean(fs))
        assert unweighted - nf > 10.0

    def test_band_equals_output_power_ratio_when_tsim_is_t0(self):
        # The example's route: (v_out / v_rs_out)^2 from integrated_noise.
        added = _src(T0) * (1 + 1e2 / F)
        gain = 25.0 / (1 + (F / 1e5) ** 2)
        nin, nout = _run(added, gain, T0)
        v_out = integrated_noise(nout, 1.0, 1e6)
        v_rs = integrated_noise(_density(np.sqrt(_src(T0) * gain)), 1.0, 1e6)
        nf = band_noise_figure(nin, nout, 1.0, 1e6, RS, T0)
        assert nf == pytest.approx(10 * math.log10((v_out / v_rs) ** 2), rel=1e-9)


class TestFailures:
    def _pair(self):
        return _run(np.zeros_like(F), np.full_like(F, 4.0))

    @pytest.mark.parametrize('kw', [{'rs': 0.0}, {'rs': -1.0}, {'temp_sim_k': 0.0},
                                    {'temp_sim_k': -5.0}, {'temp_ref_k': 0.0},
                                    {'rs': float('nan')}, {'temp_sim_k': float('inf')}])
    def test_nonpositive_conditions_raise(self, kw):
        nin, nout = self._pair()
        args = {'rs': RS, 'temp_sim_k': T_SIM, **kw}
        ref = args.pop('temp_ref_k', T0)
        with pytest.raises(ValueError, match=r'must be finite and > 0'):
            noise_figure_at_freq(nin, 1e3, args['rs'], args['temp_sim_k'], temp_ref_k=ref)
        with pytest.raises(ValueError, match=r'must be finite and > 0'):
            band_noise_figure(nin, nout, 1.0, 1e6, args['rs'], args['temp_sim_k'], temp_ref_k=ref)
        with pytest.raises(ValueError, match=r'must be finite and > 0'):
            noise_figure_vs_freq(nin, args['rs'], args['temp_sim_k'], temp_ref_k=ref)

    @pytest.mark.parametrize('fq', [0.5, 2e6])
    def test_spot_outside_sweep_is_nan_with_warning(self, fq):
        nin, _ = self._pair()
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(noise_figure_at_freq(nin, fq, RS, T_SIM))
        assert caught[0].message.code == 'frequency_outside_sweep'

    def test_spot_nonfinite_freq_raises(self):
        nin, _ = self._pair()
        with pytest.raises(ValueError, match='freq must be finite'):
            noise_figure_at_freq(nin, float('nan'), RS, T_SIM)

    @pytest.mark.parametrize('band', [(2e6, 3e6), (100.0, 110.0), (1e6, 1.0)])
    def test_band_with_fewer_than_two_points_raises(self, band):
        # decision-1 points 2 and 4: a band aggregate over an empty band (or a
        # band outside the sweep) is a malformed request, as integrated_noise.
        nin, nout = self._pair()
        with pytest.raises(ValueError, match=r'band_noise_figure: band .* contains [01] of 61'):
            band_noise_figure(nin, nout, *band, RS, T_SIM)

    def test_band_dead_output_is_nan_with_warning(self):
        nin, nout = _run(np.zeros_like(F), np.zeros_like(F))
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(band_noise_figure(nin, nout, 1.0, 1e6, RS, T_SIM))
        assert caught[0].message.code == 'zero_gain'

    def test_band_nan_density_is_nan_with_warning(self):
        nin, nout = self._pair()
        nout.y[10] = np.nan
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(band_noise_figure(nin, nout, 1.0, 1e6, RS, T_SIM))
        assert caught[0].message.code == 'nan_density'
        assert caught[0].message.details['signal'] == 'noise_out'

    def test_band_zero_input_density_is_nan_with_warning(self):
        nin, nout = self._pair()
        nin.y[5] = 0.0
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(band_noise_figure(nin, nout, 1.0, 1e6, RS, T_SIM))
        assert caught[0].message.code == 'zero_input_density'

    def test_band_grid_mismatch_raises(self):
        nin, _ = self._pair()
        other = Signal('noise_out', 'noise', 'frequency', 'V/sqrt(Hz)', F[:-1], np.ones(60))
        with pytest.raises(ValueError, match='share one frequency grid'):
            band_noise_figure(nin, other, 1.0, 1e6, RS, T_SIM)

    def test_density_below_source_noise_is_nan_with_warning(self):
        # Rs overstated a hundredfold: the density is far below 4kT·Rs.
        nin, nout = self._pair()
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(noise_figure_at_freq(nin, 1e3, 100 * RS, T_SIM))
        assert caught[0].message.code == 'nonpositive_noise_factor'
        with pytest.warns(MetricWarning):
            assert math.isnan(band_noise_figure(nin, nout, 1.0, 1e6, 100 * RS, T_SIM))
        with pytest.warns(MetricWarning) as caught:
            assert np.isnan(noise_figure_vs_freq(nin, 100 * RS, T_SIM).y).all()
        assert caught[0].message.details['unavailable_count'] == 61

    def test_complex_input_refused(self):
        z = Signal('h', 'ac', 'frequency', 'V/V', F, np.ones(61, dtype=complex))
        with pytest.raises(ValueError):
            noise_figure_at_freq(z, 1e3, RS, T_SIM)

    def test_clean_input_emits_nothing(self):
        nin, nout = self._pair()
        with warnings.catch_warnings():
            warnings.simplefilter('error', MetricWarning)
            noise_figure_at_freq(nin, 1e3, RS, T_SIM)
            band_noise_figure(nin, nout, 1.0, 1e6, RS, T_SIM)
            noise_figure_vs_freq(nin, RS, T_SIM)


class TestNfSignalTypeAndUnits443:
    """TASK-443: the NF curve is not a density; V^2/Hz inputs are diagnosed."""

    def test_nf_curve_keeps_name_and_is_db(self):
        nin, _ = _run(np.zeros_like(F), np.full_like(F, 25.0))
        nf = noise_figure_vs_freq(nin, RS, T_SIM)
        assert nf.name == nin.name
        assert nf.ylabel == 'dB'
        assert nf.type == 'noise-figure'

    def test_density_guards_refuse_the_nf_curve(self):
        nin, _ = _run(np.zeros_like(F), np.full_like(F, 25.0))
        nf = noise_figure_vs_freq(nin, RS, T_SIM)
        with pytest.raises(ValueError, match='noise_density'):
            integrated_noise(nf, 1.0, 1e6)
        with pytest.raises(ValueError, match='noise_density'):
            noise_density_at_freq(nf, 1e3)

    def test_power_density_input_refused_with_units_message(self):
        psd = Signal('noise_in', 'noise', 'frequency', 'V^2/Hz', F,
                     np.full_like(F, _src(T_SIM)))
        for call in (lambda: noise_figure_at_freq(psd, 1e3, RS, T_SIM),
                     lambda: noise_figure_vs_freq(psd, RS, T_SIM),
                     lambda: band_noise_figure(psd, psd, 1.0, 1e6, RS, T_SIM)):
            with pytest.raises(ValueError, match='V/sqrt'):
                call()
