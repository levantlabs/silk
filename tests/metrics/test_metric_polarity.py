"""TASK-234: public spectral/AC inventory with direction-aware edge checks.

Quality scalars go through real SpecReport comparisons. Raw spectra, phase
queries and harmonic lists have no universal passing direction: their numeric
contracts are checked explicitly instead. Infinities are legitimate for exact
rejection with a nonzero reference and unbounded gain margin; a blanket finite
assertion would reject those and miss finite false passes. Existing loud all-zero passband rejection is pinned separately.
"""
import inspect
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.circuit.spec_report import SpecReport
from silk.signals.base import Signal


# Each exported function belongs to exactly one exercised group. Adding a new
# export requires a deliberate polarity/raw-value contract, not an exemption.
QUALITY = {
    'sndr': ('>', 60), 'sinad': ('>', 60), 'snr': ('>', 60),
    'enob': ('>', 10), 'sfdr': ('>', 60), 'sfdr_db': ('>', 60), 'thd': ('<', -60),
    'thd_percent': ('<', 1), 'noise_floor_dbfs': ('<', -60),
}
RAW_SPECTRAL = {
    'fft', 'psd', 'welch_psd', 'spectral_power',
    'harmonic', 'harmonic_levels_db', 'harmonic_freqs', 'mix_product',
    'coherence', 'evm_constellation', 'harmonic_level_dbc', 'evm_percent',
}
# Builds a reference grid from a name; takes no Signal and measures nothing.
CONSTELLATION = {'modulation_grid'}
AC = {
    'passband_ripple_db', 'stopband_attenuation_db', 'q_from_peak',
    'group_delay', 'gain_bw_product', 'psrr_db', 'dc_gain_dB',
    'peak_gain_dB', 'gain_at_freq', 'bandwidth_3dB', 'unity_gain_freq',
    'phase_at_freq', 'phase_margin', 'gain_margins', 'gain_margin', 'cmrr_dB',
}
WAVES = ('dead', 'near_perfect', 'nyquist', 'digital_toggle', 'fs_over_3')


def _signal(y, *, axis='time', x=None):
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y), dtype=float)
    return Signal('polarity', 'voltage', axis, 'V', np.asarray(x), y)


def _wave(kind):
    n = 252 if kind == 'fs_over_3' else 256
    k = np.arange(n)
    if kind == 'dead':
        y = np.zeros(n)
    elif kind == 'near_perfect':
        y = np.cos(2 * np.pi * 11 * k / n)
    elif kind == 'nyquist':
        y = (-1.0) ** k
    elif kind == 'digital_toggle':
        y = np.where(k % 2 == 0, 1.2, 0.0)
    else:
        y = np.cos(2 * np.pi * k / 3)
    return _signal(y)


def _kwargs(s):
    return dict(tclk=1.0, n_samples=len(s.x))


def _passes(name, value, op, threshold):
    return SpecReport('edge', 'dut', {name: float(value)}).check(
        f'{name} {op} {threshold}', name, op, threshold).passed


def test_inventory_covers_every_exported_spectral_and_ac_function():
    public = {
        name for name in M.__all__
        if inspect.isfunction(getattr(M, name))
        and name not in {'p1db', 'aperture_jitter_snr_db',
                         'extract_code_transitions', 'inl', 'inl_max_lsb',
                         'dnl', 'dnl_max_lsb', 'missing_codes', 'missing_code_count',
                         'is_monotonic', 'non_monotonic_steps', 'static_errors',
                         'offset_error_lsb', 'gain_error_lsb', 'full_scale_error_lsb'}
        and getattr(M, name).__module__ in (
            M.spectral.__name__, M.ac.__name__, M.rf.__name__, M.converter.__name__,
        )
    }
    groups = [set(QUALITY), RAW_SPECTRAL, {'imd3_db', 'oip3_dbm', 'iip3_dbm'}, AC, CONSTELLATION]
    assert sum(map(len, groups)) == len(set().union(*groups))
    assert public == set().union(*groups)


@pytest.mark.parametrize('kind', WAVES)
@pytest.mark.parametrize('name', QUALITY)
def test_spectral_quality_polarity(name, kind):
    s = _wave(kind)
    value = getattr(M, name)(s, **_kwargs(s))
    if name == 'sfdr':
        assert value.bin_hz == pytest.approx(1 / len(s.x))
        value = value.sfdr_db
    unavailable = name in {'thd', 'thd_percent'} and kind in {
        'nyquist', 'digital_toggle', 'fs_over_3'}
    op, threshold = QUALITY[name]
    if unavailable:
        assert math.isnan(value)
        assert not _passes(name, value, op, threshold)
    elif kind == 'dead':
        # Explicit directions, not merely finite: -inf THD and +inf SNDR
        # would both fail this assertion, as would invented finite passes.
        assert not _passes(name, value, op, threshold)
        if name in {'sndr', 'sinad', 'snr', 'enob'}:
            assert value < 0
        elif name in {'sfdr', 'sfdr_db', 'noise_floor_dbfs'}:
            assert value == 0
        else:
            assert value > 0
    else:
        # A coherent sampled tone really is clean when independent bins exist.
        # Nyquist's sampled-record SNR is measurable even though its THD isn't.
        assert _passes(name, value, op, threshold)
        assert math.isfinite(value)  # numerical floor ceiling is documented


def test_snr_empty_residual_cannot_pass():
    s = _signal([1.0, 0.0, -1.0, 0.0])
    value = M.snr(s, **_kwargs(s))
    assert math.isnan(value)
    assert not _passes('snr', value, '>', 60)


@pytest.mark.parametrize('kind', WAVES)
@pytest.mark.parametrize('name', sorted(RAW_SPECTRAL))
def test_spectral_raw_contracts(name, kind):
    s = _wave(kind)
    n = len(s.x)
    kw = _kwargs(s)
    centered = s.y - np.mean(s.y)
    X = np.fft.rfft(centered)
    fund = int(np.argmax(np.abs(X[1:]))) + 1
    if name == 'fft':
        result = M.fft(s, **kw)
        np.testing.assert_allclose(result.y, abs(X) ** 2, atol=1e-24)
    elif name in {'psd', 'welch_psd'}:
        extra = dict(segments=2, overlap=0) if name == 'welch_psd' else {}
        result = getattr(M, name)(s, window=None, **extra, **kw)
        # Parseval provides an independent time-domain oracle, including the
        # unpaired Nyquist component and the dead trace's legitimate zero PSD.
        if name == 'welch_psd':
            expected = np.mean([np.var(part) for part in np.split(s.y, 2)])
        else:
            expected = np.var(s.y)
        power = np.sum(result.y) * (result.x[1] - result.x[0])
        assert power == pytest.approx(expected, rel=1e-12, abs=1e-24)
        assert np.all(result.y >= 0)
    elif name == 'spectral_power':
        # Full-band trapezoidal integration half-weights the edge samples.
        expected = np.var(s.y) - abs(X[-1]) ** 2 / (2 * n * n)
        value = M.spectral_power(s, 0, 0.5, window=None, **kw)
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-24)
    elif name in {'harmonic', 'harmonic_levels_db', 'harmonic_freqs', 'harmonic_level_dbc'}:
        orders = np.arange(1, 6)
        folded = (orders * fund) % n
        folded = np.minimum(folded, n - folded)
        observable = (folded != 0) & ((orders == 1) | (folded != fund))
        if name == 'harmonic':
            value = M.harmonic(s, 2, **kw)
            if observable[1]:
                expected = X[folded[1]]
                if (2 * fund) % n > n / 2:
                    expected = expected.conjugate()
                assert value == pytest.approx(expected, abs=1e-20)
            else:
                assert np.isnan(value)
        elif name == 'harmonic_level_dbc':
            value = M.harmonic_level_dbc(s, n=2, **kw)
            if not observable[1]:
                assert np.isnan(value)
            elif kind == 'dead':
                assert value == 0.0
                assert not _passes(name, value, '<', -60)
            else:
                assert value < -100
        elif name == 'harmonic_freqs':
            result = M.harmonic_freqs(s, n=5, observed=True, **kw)
            np.testing.assert_array_equal(np.isnan(result), ~observable)
            np.testing.assert_allclose(result[observable], folded[observable] / n)
        else:
            result = M.harmonic_levels_db(s, n_harmonics=5, **kw)
            np.testing.assert_array_equal(np.isnan(result), ~observable[1:])
            if kind == 'dead':
                np.testing.assert_array_equal(result, np.zeros(4))
            else:
                assert np.all(result[observable[1:]] < -100)
    elif name == 'mix_product':
        value = M.mix_product(s, 2, -1, f1_hz=11/n, f2_hz=13/n, **kw)
        assert value == pytest.approx(np.fft.fft(centered)[9], abs=1e-20)
    elif name == 'coherence':
        result = M.coherence(s, s, segments=2, overlap=0, window=None, **kw)
        if kind == 'dead':
            assert np.all(np.isnan(result.y))
        else:
            # Identical records correlate perfectly where auto-power exists;
            # exactly empty bins legitimately carry NaN, not a false zero.
            valid = ~np.isnan(result.y)
            assert np.any(valid)
            np.testing.assert_allclose(result.y[valid], 1)
    elif name == 'evm_percent':
        # TASK-369: the fitted-gain path. Symbols that sample to zero (the
        # dead trace; the Nyquist toggle, whose symbol centres interpolate to
        # 0) have no gain to fit and must not read as a perfect 0 %.
        q = _signal(np.zeros(n))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            value = M.evm_percent(s, q, symbol_period=1, modulation='bpsk')
            result = M.evm_constellation(s, q, symbol_period=1,
                                         modulation='bpsk')
        if not np.any(result.received):
            assert kind in ('dead', 'nyquist')
            assert np.isnan(value)
            assert not _passes(name, value, '<', 5)
            assert any(getattr(w.message, 'code', None) == 'zero_gain'
                       for w in caught)
        else:
            assert value == result.evm_percent
            expected = 100 * np.sqrt(np.mean(np.abs(result.errors) ** 2))
            assert value == pytest.approx(expected / abs(result.fitted_gain))
    else:
        assert name == 'evm_constellation'
        result = M.evm_constellation(
            s, _signal(np.zeros(n)), symbol_period=1, ideal_points=[-1, 1])
        expected = 100 * np.sqrt(np.mean(np.abs(result.errors) ** 2))
        assert result.evm_percent == pytest.approx(expected)
        np.testing.assert_allclose(result.received - result.assigned_ideals,
                                   result.errors)
        if kind == 'dead':
            assert result.evm_percent == 100
            assert not _passes(name, result.evm_percent, '<', 5)


@pytest.mark.parametrize('kind', WAVES)
def test_imd3_reference_polarity(kind):
    s = _wave(kind)
    n = len(s.x)
    # For Nyquist/toggle the requested ordinary tone bins are exactly empty.
    # For fs/3 put the reference on its real tone and the intermods in distinct
    # measurable bins. IMD3 uses the larger of the requested references.
    f1, f2 = ((84/n, 82/n) if kind == 'fs_over_3' else (11/n, 13/n))
    value = M.imd3_db(s, f1_hz=f1, f2_hz=f2, **_kwargs(s))
    if kind in {'dead', 'nyquist', 'digital_toggle'}:
        assert math.isnan(value)
        assert not _passes('imd3_db', value, '<', -60)
    else:
        assert value < -100
        assert _passes('imd3_db', value, '<', -60)


def test_imd3_missing_reference_fails_but_real_exact_rejection_is_allowed():
    dead = _wave('dead')
    value = M.imd3_db(dead, f1_hz=11/256, f2_hz=13/256, **_kwargs(dead))
    assert math.isnan(value)
    assert not _passes('imd3_db', value, '<', -60)
    # Two coherent tones with exactly representable samples and empty-power
    # (but independently present) IMD bins at fs/12 and fs/3. Eight periods
    # keep these bins bit-exact zero on both numpy 1.26 and numpy 2.x.
    s = _signal(np.tile([2., .5, -1.5, -1., .5, .5,
                        0., .5, .5, -1., -1.5, .5], 8))
    spectrum = abs(np.fft.rfft(s.y - np.mean(s.y))) ** 2
    assert np.all(spectrum[[16, 24]] > 0)  # both reference tones exist
    np.testing.assert_array_equal(spectrum[[8, 32]], [0, 0])
    value = M.imd3_db(s, f1_hz=1/6, f2_hz=1/4, **_kwargs(s))
    assert value == -math.inf
    assert _passes('imd3_db', value, '<', -60)


def _ac(kind):
    x = np.array([1., 2., 4., 8.])
    y = {'dead': [0, 0, 0, 0], 'zero_stopband': [2, 2, 0, 0],
         'zero_common_mode': [2, 2, 2, 2]}[kind]
    return _signal(np.asarray(y, dtype=complex), axis='frequency', x=x)


@pytest.mark.parametrize('kind', ('dead', 'zero_stopband', 'zero_common_mode'))
@pytest.mark.parametrize('name', sorted(AC))
def test_ac_polarity_and_explicit_raw_contracts(name, kind):
    s = _ac(kind)
    dead = kind == 'dead'
    stopped = kind == 'zero_stopband'
    zero = _ac('dead')
    if name == 'passband_ripple_db':
        if dead:
            value = M.passband_ripple_db(s, 2)
            assert math.isnan(value)
            assert not _passes(name, value, '<', 1)
        else:
            assert M.passband_ripple_db(s, 2) == 0  # actually flat passband
    elif name in {'stopband_attenuation_db', 'cmrr_dB'}:
        value = (M.stopband_attenuation_db(s, 4) if name.startswith('stop')
                 else M.cmrr_dB(s, zero))
        if dead:
            assert math.isnan(value)
            assert not _passes(name, value, '>', 60)
        elif name == 'cmrr_dB' or stopped:
            assert value == math.inf  # nonzero reference, exact rejection
            assert _passes(name, value, '>', 60)
        else:
            assert value == 0
            assert not _passes(name, value, '>', 60)
    elif name in {'dc_gain_dB', 'peak_gain_dB', 'gain_at_freq'}:
        with np.errstate(divide='ignore'):
            value = (getattr(M, name)(s, 1) if name == 'gain_at_freq'
                     else getattr(M, name)(s))
        expected = ((0 if dead else 2) if name == 'gain_at_freq'
                    else (-math.inf if dead else 20 * np.log10(2)))
        assert value == pytest.approx(expected)
        assert not _passes(name, value, '>', 20)
    elif name in {'bandwidth_3dB', 'gain_bw_product', 'q_from_peak'}:
        value = getattr(M, name)(s)
        if dead or not stopped or name == 'q_from_peak':
            assert math.isnan(value)
            assert not _passes(name, value, '>', 10)
        else:
            expected = {'bandwidth_3dB': 2, 'gain_bw_product': 4}[name]
            assert value == pytest.approx(expected)
    elif name == 'unity_gain_freq':
        value = M.unity_gain_freq(s)
        if stopped:
            assert value == pytest.approx(np.sqrt(8))
        else:
            assert math.isnan(value)
        assert not _passes(name, value, '>', 10)
    elif name == 'phase_margin':
        value = M.phase_margin(s)
        if stopped:
            assert value == -180
        else:
            assert math.isnan(value)
        assert not _passes(name, value, '>', 45)
    elif name == 'gain_margins':
        assert M.gain_margins(s) == []  # no negative-real-axis crossover
    elif name == 'gain_margin':
        value = M.gain_margin(s)
        if dead or stopped:
            assert value == math.inf  # loop gain < 1, no phase crossover
            assert _passes(name, value, '>', 10)
        else:
            assert math.isnan(value)  # sweep ends with gain still >= 1
            assert not _passes(name, value, '>', 10)
    elif name in {'phase_at_freq', 'group_delay'}:
        # Raw angle(0)=0 convention, not a claim of measured phase margin.
        if name == 'group_delay':
            # TASK-450: zero response has no defined phase -> nan, never a delay
            has_zero = bool((np.abs(s.y) == 0).any())
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                value = M.group_delay(s).y
            assert any(issubclass(x.category, MetricWarning) for x in w) == has_zero
            assert np.array_equal(np.isnan(value), np.abs(s.y) == 0)
        else:
            value = M.phase_at_freq(s, 1)
            np.testing.assert_array_equal(value, np.zeros_like(value))
    else:
        assert name == 'psrr_db'
        result = M.psrr_db(s, zero)
        if dead:
            assert np.all(np.isnan(result.y))
            assert not _passes(name, result.y[0], '>', 60)
        else:
            # Documented rejection ceiling; both-zero points remain unknown.
            nonzero = abs(s.y) > 0
            np.testing.assert_allclose(result.y[nonzero], -20 * np.log10(np.finfo(float).eps))
            assert np.all(np.isnan(result.y[~nonzero]))
