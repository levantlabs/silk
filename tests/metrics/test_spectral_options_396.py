"""TASK-396 — fundamental_hz=, guard_bins=, window= and full_scale= on the
one tone-spectrum core shared by every metric that names a fundamental."""
import math

import numpy as np
import pytest

from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema
import silk.metrics.spectral as S
from silk.metrics import converter

TCLK = 1e-6
N = 4096
DF = 1.0 / (N * TCLK)


def _sig(y):
    return Signal('v', 'voltage', 'time', 'V', np.arange(len(y)) * TCLK, y)


def _tone(cycles, amp=0.5, n=N, phase=0.3):
    return amp * np.sin(2 * np.pi * cycles * np.arange(n) / n + phase)


def _noise(sigma, n=N, seed=7):
    return sigma * np.random.default_rng(seed).standard_normal(n)


def _weighted_power(y, window=None):
    x = y - np.mean(y)
    if window is not None:
        x = x * window
    P = np.abs(np.fft.rfft(x)) ** 2
    P[1:-1 if len(y) % 2 == 0 else None] *= 2
    return P + np.finfo(float).eps


# Every metric in the inventory, with the call that reaches it.
INVENTORY = {
    'sndr': lambda s, **kw: S.sndr(s, tclk=TCLK, **kw),
    'sinad': lambda s, **kw: S.sinad(s, tclk=TCLK, **kw),
    'enob': lambda s, **kw: converter.enob(s, tclk=TCLK, **kw),
    'snr': lambda s, **kw: S.snr(s, tclk=TCLK, **kw),
    'sfdr': lambda s, **kw: S.sfdr(s, tclk=TCLK, **kw),
    'sfdr_db': lambda s, **kw: S.sfdr_db(s, tclk=TCLK, **kw),
    'thd': lambda s, **kw: S.thd(s, tclk=TCLK, **kw),
    'thd_percent': lambda s, **kw: S.thd_percent(s, tclk=TCLK, **kw),
    'noise_floor_dbfs': lambda s, **kw: S.noise_floor_dbfs(s, tclk=TCLK, **kw),
    'harmonic_levels_db': lambda s, **kw: S.harmonic_levels_db(s, tclk=TCLK, **kw),
    'harmonic_level_dbc': lambda s, **kw: S.harmonic_level_dbc(s, n=3, tclk=TCLK, **kw),
    'harmonic': lambda s, **kw: S.harmonic(s, 2, tclk=TCLK, **kw),
    'harmonic_freqs': lambda s, **kw: S.harmonic_freqs(s, tclk=TCLK, **kw),
}
RATIO = ['sndr', 'sinad', 'enob', 'snr', 'sfdr', 'sfdr_db', 'thd',
         'thd_percent', 'noise_floor_dbfs']
FULL_SCALE = ['sndr', 'sinad', 'enob', 'snr', 'noise_floor_dbfs']


def _same(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    return repr(a) == repr(b)


@pytest.fixture
def distorted():
    x = _tone(101)
    return _sig(x + 0.01 * x ** 2 + 0.003 * x ** 3 + _noise(1e-4))


# --- AC#1 fundamental_hz ----------------------------------------------------

@pytest.mark.parametrize('name', sorted(INVENTORY))
def test_fundamental_hz_none_and_elected_bin_give_todays_result(name, distorted):
    call = INVENTORY[name]
    default = call(distorted)
    assert _same(call(distorted, fundamental_hz=None), default)
    # Naming the bin the election picks reproduces it exactly, and a value
    # a fraction of a bin off rounds to the same bin.
    assert _same(call(distorted, fundamental_hz=101 * DF), default)
    assert _same(call(distorted, fundamental_hz=101.4 * DF), default)


@pytest.mark.parametrize('name', sorted(INVENTORY))
@pytest.mark.parametrize('bad', [0.0, 0.4 * DF, -5 * DF, (N // 2 + 1) * DF,
                                 1.0 / TCLK, math.nan, math.inf])
def test_fundamental_hz_outside_the_spectrum_raises(name, bad, distorted):
    with pytest.raises(ValueError, match='fundamental_hz'):
        INVENTORY[name](distorted, fundamental_hz=bad)


def test_fundamental_hz_selects_a_weaker_tone():
    y = _tone(101, amp=0.5) + _tone(300, amp=0.05, phase=1.0)
    sig = _sig(y)
    P = _weighted_power(y)
    res = S.sfdr(sig, tclk=TCLK, fundamental_hz=300 * DF)
    assert res.fundamental_hz == pytest.approx(300 * DF)
    assert res.spur_hz == pytest.approx(101 * DF)
    assert res.sfdr_db == pytest.approx(10 * np.log10(P[300] / P[101]))
    assert res.sfdr_db < 0          # the named tone is the weaker one
    mask = np.ones(len(P), bool)
    mask[[0, 300]] = False
    assert S.sndr(sig, tclk=TCLK, fundamental_hz=300 * DF) == pytest.approx(
        10 * np.log10(P[300] / P[mask].sum()))
    assert S.harmonic_freqs(sig, tclk=TCLK, n=2,
                            fundamental_hz=300 * DF)[1] == pytest.approx(600 * DF)


# --- AC#2 guard_bins --------------------------------------------------------

def test_guard_bins_default_is_todays_result(distorted):
    for name in RATIO:
        call = INVENTORY[name]
        assert _same(call(distorted, guard_bins=0), call(distorted))


@pytest.mark.parametrize('k', [1, 3, 8])
def test_guard_bins_on_a_leaking_tone_match_the_band_reference(k):
    """A non-coherent tone leaks into its neighbours. guard_bins=k moves
    k bins on each side into the fundamental (and each harmonic) and out
    of the noise and spur sums. DC owns bins 0..k too, and the noise sums
    are referred back to the default selection's bin count (TASK-436)."""
    x = _tone(100.37)
    y = x + 0.02 * x ** 2 + _noise(1e-4)
    sig = _sig(y)
    P = _weighted_power(y)
    f = int(np.argmax(P[1:]) + 1)
    fund_band = np.arange(f - k, f + k + 1)
    h_band = np.arange(2 * f - k, 2 * f + k + 1)
    p_fund = P[fund_band].sum()

    noise = np.ones(len(P), bool)
    noise[:k + 1] = False
    noise[fund_band] = False
    base = len(P) - 2                # default: all but DC and the fundamental
    assert S.sndr(sig, tclk=TCLK, guard_bins=k) == pytest.approx(
        10 * np.log10(p_fund / (P[noise].sum() * base / noise.sum())),
        abs=1e-9)

    thd = S.thd(sig, tclk=TCLK, guard_bins=k, n_harmonics=2)
    assert thd == pytest.approx(10 * np.log10(P[h_band].sum() / p_fund),
                                abs=1e-9)

    noise_only = noise.copy()
    noise_only[h_band] = False
    base_only = len(P) - 3           # ... and the 2nd harmonic's centre bin
    assert S.snr(sig, tclk=TCLK, guard_bins=k, n_harmonics=2) == pytest.approx(
        10 * np.log10(p_fund / (P[noise_only].sum() * base_only
                                / noise_only.sum())), abs=1e-9)

    spur_pool = np.where(noise, P, 0.0)
    s = int(np.argmax(spur_pool))
    res = S.sfdr(sig, tclk=TCLK, guard_bins=k)
    assert res.spur_hz == pytest.approx(s * DF)
    assert res.sfdr_db == pytest.approx(
        10 * np.log10(p_fund / spur_pool[s - k:s + k + 1].sum()), abs=1e-9)

    assert S.noise_floor_dbfs(sig, tclk=TCLK, guard_bins=k) == pytest.approx(
        10 * np.log10(P[noise].mean() / p_fund), abs=1e-9)


def test_guard_bins_recover_a_leaking_tones_snr():
    """Rectangular leakage reads as noise without guard bins."""
    sig = _sig(_tone(100.37) + _noise(1e-4))
    readings = [S.sndr(sig, tclk=TCLK, guard_bins=k) for k in (0, 2, 8, 32)]
    assert readings == sorted(readings)
    assert readings[0] < 20 < readings[-1]


@pytest.mark.parametrize('bad', [-1, 1.5, True, '2', None])
@pytest.mark.parametrize('name', RATIO)
def test_guard_bins_must_be_a_nonnegative_integer(name, bad, distorted):
    with pytest.raises(ValueError, match='guard_bins'):
        INVENTORY[name](distorted, guard_bins=bad)


# --- AC#3 window ------------------------------------------------------------

def test_window_default_is_todays_result(distorted):
    for name in RATIO:
        call = INVENTORY[name]
        assert _same(call(distorted, window='rectangular'), call(distorted))
        assert _same(call(distorted, window=None), call(distorted))


@pytest.mark.parametrize('window', ['rectangular', 'hann', 'hamming',
                                    'blackman'])
@pytest.mark.parametrize('name', ['snr', 'sndr', 'sinad'])
def test_windowed_coherent_tone_reads_the_analytic_snr(name, window):
    amp, sigma = 0.5, 1e-3
    sig = _sig(_tone(101, amp=amp) + _noise(sigma))
    analytic = 10 * np.log10((amp ** 2 / 2) / sigma ** 2)       # 50.97 dB
    got = INVENTORY[name](sig, window=window)
    assert abs(got - analytic) < 0.5, (name, window, got, analytic)


def test_windowed_tone_without_its_lobe_would_read_as_noise():
    """The window's main lobe belongs to the tone: the same record read
    with the lobe counted as noise is tens of dB worse."""
    sig = _sig(_tone(101) + _noise(1e-3))
    P = _weighted_power(sig.y, np.hanning(N))
    mask = np.ones(len(P), bool)
    mask[[0, 101]] = False
    naive = 10 * np.log10(P[101] / P[mask].sum())
    assert S.snr(sig, tclk=TCLK, window='hann') - naive > 20


def test_window_name_is_validated(distorted):
    with pytest.raises(ValueError, match='window'):
        S.sndr(distorted, tclk=TCLK, window='kaiser')


# --- AC#4 full_scale --------------------------------------------------------

@pytest.mark.parametrize('window', ['rectangular', 'hann', 'blackman'])
def test_full_scale_refers_snr_and_enob_to_full_scale(window):
    amp, fs = 0.25, 1.0
    sig = _sig(_tone(101, amp=amp) + _noise(1e-4))
    backoff = 20 * np.log10(fs / amp)                           # 12.04 dB
    for name in ('snr', 'sndr', 'sinad'):
        call = INVENTORY[name]
        assert (call(sig, window=window, full_scale=fs)
                - call(sig, window=window)) == pytest.approx(backoff, abs=0.05)
    assert (converter.enob(sig, tclk=TCLK, window=window, full_scale=fs)
            - converter.enob(sig, tclk=TCLK, window=window)) == pytest.approx(
                backoff / 6.02, abs=0.01)


def test_full_scale_sets_the_noise_floor_reference():
    amp = 0.25
    sig = _sig(_tone(101, amp=amp) + _noise(1e-4))
    today = S.noise_floor_dbfs(sig, tclk=TCLK)
    # full_scale equal to the tone's own amplitude is today's reference.
    assert S.noise_floor_dbfs(sig, tclk=TCLK, full_scale=amp) == pytest.approx(
        today, abs=0.01)
    assert (S.noise_floor_dbfs(sig, tclk=TCLK, full_scale=4 * amp)
            == pytest.approx(today - 20 * np.log10(4), abs=0.01))


@pytest.mark.parametrize('name', FULL_SCALE)
def test_full_scale_none_is_todays_result(name, distorted):
    call = INVENTORY[name]
    assert _same(call(distorted, full_scale=None), call(distorted))


@pytest.mark.parametrize('bad', [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize('name', FULL_SCALE)
def test_full_scale_must_be_a_positive_amplitude(name, bad, distorted):
    with pytest.raises(ValueError, match='full_scale'):
        INVENTORY[name](distorted, full_scale=bad)


# --- shape and schema -------------------------------------------------------

def test_no_option_switches_the_return_type(distorted):
    for kw in ({}, {'guard_bins': 2}, {'window': 'hann'},
               {'fundamental_hz': 202 * DF}):
        assert isinstance(S.sfdr(distorted, tclk=TCLK, **kw), S.SFDR)
        assert isinstance(S.sfdr_db(distorted, tclk=TCLK, **kw), float)
        assert isinstance(S.sndr(distorted, tclk=TCLK, **kw), float)


def test_schema_declares_the_options():
    schema = export_metric_schema()
    for name in ('sndr', 'sinad', 'enob', 'snr', 'sfdr_db', 'thd',
                 'thd_percent', 'noise_floor_dbfs', 'harmonic_level_dbc'):
        params = {p['name']: p for p in schema[name]['parameters']}
        assert params['fundamental_hz']['unit'] == 'Hz'
        assert params['fundamental_hz']['default'] is None
        if name == 'harmonic_level_dbc':
            continue
        assert params['guard_bins']['default'] == 0
        assert params['window']['default'] == 'rectangular'
        assert params['window']['choices'] == ['hann', 'hamming', 'blackman',
                                               'rectangular']
        assert ('full_scale' in params) == (name in FULL_SCALE)
        if name in FULL_SCALE:
            assert params['full_scale']['unit'] == '[y]'
