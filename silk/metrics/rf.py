"""RF compression, intercept, intermodulation, mixing and modulation metrics.

Compression uses a power sweep, with input and output in dBm. Two-tone and
mixing measurements use clock-sampled time waveforms; EVM fits complex I/Q
constellations. Each function documents its accepted inputs and sampling options.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional

import math
import typing

import numpy as np

from silk.signals.base import Signal
from silk.signals.schema import metric, Transient, ComplexPoints, PowerSweep
from silk.metrics._guards import (
    _nan_safe_transform,
    _axis,
    _SampleSpan,
    _check_nonempty,
    _any_nan,
)
from silk.metrics.diagnostics import emit
from silk.metrics.spectral import (
    _fold_raw_bin_mirrored,
    _sample_at_clock,
    _sample_details,
    _warn_spectrum,
)
from silk.metrics.waveform import _check_transient, _observed_range


class _TwoTone(NamedTuple):
    """Folded two-tone spectrum read by :func:`_two_tone_core`."""
    P: np.ndarray        # |rfft(x - mean)|^2, one value per bin
    f1_bin: int
    f2_bin: int
    p_fund: float        # larger fundamental bin power
    p_imd: float         # larger folded IM3 product bin power
    n_samples: int


def _two_tone_core(sig, *, metric: str, tclk: float, f1_hz: float,
                   f2_hz: float, tstart: float,
                   n_samples: Optional[int]) -> Optional[_TwoTone]:
    """Shared two-tone read behind :func:`imd3_db`, :func:`oip3_dbm` and
    :func:`iip3_dbm`: validation, binning folding, the collision,
    NaN and zero-reference diagnostics. Returns ``None`` where a diagnostic
    was emitted and the caller returns ``nan``."""
    if f1_hz <= 0 or f2_hz <= 0 or f1_hz == f2_hz:
        raise ValueError(
            f'{metric}: tone frequencies must be > 0 and distinct, got '
            f'{f1_hz}, {f2_hz}'
        )
    samples = _sample_at_clock(sig, metric=metric, tclk=tclk, tstart=tstart,
                               n_samples=n_samples)
    P = np.abs(np.fft.rfft(samples - np.mean(samples))) ** 2
    n = len(P)
    fs = 1.0 / tclk
    df = fs / len(samples)
    f1_bin = int(round(f1_hz / df))
    f2_bin = int(round(f2_hz / df))
    grid = (f'available FFT grid bins 1..{n - 1}, {df}..{(n - 1) * df} Hz '
            f'(df={df} Hz, n={n}, n_samples={len(samples)})')
    for name, frequency, b in (('f1_hz', f1_hz, f1_bin),
                               ('f2_hz', f2_hz, f2_bin)):
        if not 0 < b < n or frequency > (n - 1) * df:
            raise ValueError(
                f'{metric}: {name}={frequency} Hz (bin={b}) is outside '
                f'the supported tone range; {grid}'
            )
    if f1_bin == f2_bin:
        raise ValueError(
            f'{metric}: tone bins after binning are degenerate: '
            f'f1_hz={f1_hz} Hz (bin={f1_bin}), '
            f'f2_hz={f2_hz} Hz (bin={f2_bin}); {grid}'
        )
    imd_bins_raw = (2 * f1_bin - f2_bin, 2 * f2_bin - f1_bin)
    imd_bins = tuple(_fold_raw_bin_mirrored(b, len(samples))[0]
                     for b in imd_bins_raw)
    colliding = [(raw, folded) for raw, folded in zip(imd_bins_raw, imd_bins)
                if folded == 0 or folded == f1_bin or folded == f2_bin]
    if colliding:
        details = _sample_details(sig, samples, tclk, tstart)
        details.update(f1_bin=f1_bin, f2_bin=f2_bin, bin_hz=df,
                       lower_product_bin_raw=imd_bins_raw[0],
                       upper_product_bin_raw=imd_bins_raw[1],
                       lower_product_bin_folded=imd_bins[0],
                       upper_product_bin_folded=imd_bins[1])
        emit(
            metric=metric, code='unattributable_imd_product',
            message='IMD3 product unavailable because its folded bin '
                    'coincides with DC or a requested tone bin.',
            details=details,
            suggestion='Try a sampling rate or coherent tones that separate '
                       'the IMD3 products from DC and the tones.',
        )
        return None
    consumed_bins = [f1_bin, f2_bin] + list(imd_bins)
    nan_count = int(np.count_nonzero(np.isnan(P[consumed_bins])))
    if nan_count:
        emit(
            metric=metric, code='nan_consumed_bins',
            message='Measurement unavailable because the consumed FFT data '
                    'contains NaNs.',
            details={'signal_name': sig.name, 'nan_bin_count': nan_count,
                     'consumed_bin_count': len(consumed_bins),
                     'f1_bin': f1_bin, 'f2_bin': f2_bin,
                     'lower_product_bin': imd_bins[0],
                     'upper_product_bin': imd_bins[1],
                     'bin_hz': df, 'fft_bin_count': n,
                     'sample_count': len(samples)},
            suggestion='Check the source result for the consumed samples.',
        )
        return None
    p_imd = max(float(P[imd_bins[0]]), float(P[imd_bins[1]]))
    p_fund = max(float(P[f1_bin]), float(P[f2_bin]))
    if p_fund <= 0:
        details = _sample_details(sig, samples, tclk, tstart)
        details.update(f1_bin=f1_bin, f2_bin=f2_bin, bin_hz=df,
                       f1_hz=f1_bin * df, f2_hz=f2_bin * df,
                       reference_power=0.0, power_unit=f'({sig.ylabel})^2')
        emit(
            metric=metric, code='zero_reference_power',
            message='Neither requested fundamental bin carries power; '
                    'the IMD reference is unavailable.',
            details=details,
            suggestion='Check the source result and requested tone frequencies.',
        )
        return None
    if p_imd > 0 and np.isinf(p_imd) and np.isinf(p_fund):
        _warn_spectrum(metric, sig, samples, tclk, tstart, P[consumed_bins])
    return _TwoTone(P, f1_bin, f2_bin, p_fund, p_imd, len(samples))


@metric(domain='tran', unit='dB',
        param_units={'tclk': 's', 'f1_hz': 'Hz', 'f2_hz': 'Hz',
                     'tstart': 's'})
def imd3_db(sig: Transient, *, tclk: float, f1_hz: float, f2_hz: float,
            tstart: float = 0.0,
            n_samples: Optional[int] = None) -> float:
    """3rd-order intermodulation distortion in dB, relative to the
    larger fundamental.

    For tones at f1, f2 the 3rd-order intermods fall at ``2f1 − f2``
    and ``2f2 − f1``. Both products are folded back in-band
    the same way :func:`thd`/:func:`harmonic` fold harmonics above
    Nyquist (see ``_fold_raw_bin_mirrored``, the module docstring's aliasing section) — a product below
    DC or above Nyquist aliases into the sampled record and is read at
    its folded bin, never dropped. Returns the larger of the two folded
    products, in dB below the larger fundamental. If a folded product
    lands on DC (bin 0) or on either requested tone bin, its power is
    inseparable from DC/the tone: returns ``nan`` with a diagnostic naming
    the collision (before either fundamental is checked). If neither
    requested tone carries power, returns ``nan``: the reference is
    unavailable on this data. With a nonzero reference,
    exactly zero consumed (folded) product power retains ``-inf``. NaNs
    in any consumed fundamental or folded-product bin return ``nan`` with
    a diagnostic before selecting either maximum.

    Resolution: ``f1_hz``/``f2_hz`` are rounded to the nearest FFT bin
    (``1/(n_samples·tclk)``); an off-grid tone leaks into neighbouring
    bins and the product is under-read. Keep both tones coherent. Tones above
    FFT coverage or with rounded bins outside the non-DC grid raise
    ``ValueError`` with the available grid. In-range nearest-bin selection
    is unchanged.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    f1_hz : float
        First tone frequency, Hz; rounded to a non-DC FFT bin.
    f2_hz : float
        Second distinct tone frequency, Hz; rounded to a different non-DC
        FFT bin. Both requested tones must fit the available spectrum.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.

    Returns
    -------
    float
        Larger folded IM3 power relative to the larger tone, dB (normally
        negative). Colliding products, NaN consumed bins, or zero reference
        power give nan. Exactly zero product power with a nonzero reference
        gives -inf without warning; inf/inf also yields nan.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Nonpositive, identical, out-of-spectrum, or identically binned tones.

    OverflowError
        Infinite tone frequencies reach integer bin conversion without a
        dedicated validation guard.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Products colliding with DC or tones, NaN consumed bins, zero
        reference power, or infinite product and reference powers together.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    r = _two_tone_core(sig, metric='imd3_db', tclk=tclk, f1_hz=f1_hz,
                       f2_hz=f2_hz, tstart=tstart, n_samples=n_samples)
    if r is None:
        return float('nan')
    if r.p_imd <= 0:
        return float('-inf')
    return float(10.0 * np.log10(r.p_imd / r.p_fund))


def _tone_dbm(p_bin: float, n: int, r_ref: float) -> float:
    """Power in dBm of the sinusoid whose rectangular-window FFT bin power
    is ``p_bin`` (volts in, ``r_ref`` ohms): ``P = 2*p_bin/(n^2*r_ref)`` W."""
    return float(10.0 * np.log10(2.0 * p_bin / (n * n * r_ref) / 1e-3))


def _check_r_ref(metric: str, r_ref: float) -> None:
    if not r_ref > 0:
        raise ValueError(f'{metric}: r_ref must be > 0 ohm, got {r_ref}')


@metric(domain='tran', unit='dBm',
        param_units={'tclk': 's', 'f1_hz': 'Hz', 'f2_hz': 'Hz',
                     'r_ref': 'ohm', 'tstart': 's'})
def oip3_dbm(sig: Transient, *, tclk: float, f1_hz: float, f2_hz: float,
             r_ref: float = 50.0, tstart: float = 0.0,
             n_samples: Optional[int] = None) -> float:
    """Output-referred 3rd-order intercept point in dBm from one two-tone
    spectrum.

    ``OIP3 = P_fund + (P_fund - P_IM3) / 2`` with ``P_fund`` the larger
    fundamental and ``P_IM3`` the larger folded IM3 product, both read by
    the same core as :func:`imd3_db` (so its folding, collision and NaN
    handling, bin resolution and ``ValueError`` cases apply unchanged).
    ``sig`` is a voltage in volts; powers are ``Vrms^2 / r_ref`` in dBm
    (``r_ref`` defaults to 50 ohm and must be > 0). Equivalent to
    ``P_fund_dbm - imd3_db / 2``. Degenerate data follows :func:`imd3_db`:
    ``nan`` where it returns ``nan``, ``+inf`` where it returns ``-inf``.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    f1_hz : float
        First tone frequency, Hz; rounded to a non-DC FFT bin.
    f2_hz : float
        Second distinct tone frequency, Hz; rounded to a different non-DC
        FFT bin. Both requested tones must fit the available spectrum.
    r_ref : float, default 50.0
        Positive reference resistance, ohms, for converting voltage power
        to dBm. Signal values must represent volts.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.

    Returns
    -------
    float
        Output intercept in dBm. Colliding products, NaN consumed bins, or
        zero reference give nan; exactly zero IM3 with a nonzero reference
        gives +inf without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Nonpositive, identical, out-of-spectrum, or identically binned tones.
        Nonpositive or NaN r_ref.

    OverflowError
        Infinite tone frequencies reach integer bin conversion without a
        dedicated validation guard.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Products colliding with DC or tones, NaN consumed bins, zero
        reference power, or infinite product and reference powers together.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    _check_r_ref('oip3_dbm', r_ref)
    r = _two_tone_core(sig, metric='oip3_dbm', tclk=tclk, f1_hz=f1_hz,
                       f2_hz=f2_hz, tstart=tstart, n_samples=n_samples)
    if r is None:
        return float('nan')
    if r.p_imd <= 0:
        return float('inf')
    p_fund_dbm = _tone_dbm(r.p_fund, r.n_samples, r_ref)
    return p_fund_dbm + 0.5 * float(10.0 * np.log10(r.p_fund / r.p_imd))


@metric(domain='tran', unit='dBm',
        param_units={'tclk': 's', 'f1_hz': 'Hz', 'f2_hz': 'Hz',
                     'r_ref': 'ohm', 'tstart': 's'})
def iip3_dbm(out_sig: Transient, in_sig: Transient, *, tclk: float,
             f1_hz: float, f2_hz: float, r_ref: float = 50.0,
             tstart: float = 0.0, n_samples: Optional[int] = None) -> float:
    """Input-referred 3rd-order intercept point in dBm: ``OIP3 - gain``.

    ``out_sig`` is the output-node voltage (as for :func:`oip3_dbm`);
    ``in_sig`` is the input-node voltage on the same time grid, so the
    input tone power includes the real source loading. Gain is the dB ratio
    of the two signals' power at the stronger output tone (both read at the
    same bin; ``r_ref`` cancels in it), so
    ``IIP3 = P_in + (P_fund - P_IM3) / 2`` and agrees with
    :func:`silk.metrics.fom.iip3_extrapolation <silk.metrics.iip3_extrapolation>` on the same powers.
    ``nan`` if either signal's two-tone read is unavailable (a diagnostic
    names the metric and signal), or ``in_sig`` carries no power at that
    tone; ``+inf`` where :func:`oip3_dbm` is ``+inf``.

    Parameters
    ----------
    out_sig : Signal
        Output voltage transient in volts, with one-dimensional time and
        value arrays; not mutated. See fft for sampling requirements.
    in_sig : Signal
        Input voltage transient in volts; not mutated. The sampled length
        must match out_sig; the original time grids are not compared.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    f1_hz : float
        First tone frequency, Hz; rounded to a non-DC FFT bin.
    f2_hz : float
        Second distinct tone frequency, Hz; rounded to a different non-DC
        FFT bin. Both requested tones must fit the available spectrum.
    r_ref : float, default 50.0
        Positive reference resistance, ohms, for converting voltage power
        to dBm. Signal values must represent volts.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.

    Returns
    -------
    float
        Input intercept in dBm. Unavailable two-tone reads give nan. Zero
        input power at the stronger output tone also gives nan, without a
        warning on that particular branch. Zero output IM3 with a usable
        reference gives +inf without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Nonpositive, identical, out-of-spectrum, or identically binned tones.
        Nonpositive or NaN r_ref.
        Different sampled input and output lengths.

    OverflowError
        Infinite tone frequencies reach integer bin conversion without a
        dedicated validation guard.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Products colliding with DC or tones, NaN consumed bins, zero
        reference power, or infinite product and reference powers together.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    _check_r_ref('iip3_dbm', r_ref)
    kw = dict(metric='iip3_dbm', tclk=tclk, f1_hz=f1_hz, f2_hz=f2_hz,
              tstart=tstart, n_samples=n_samples)
    ro = _two_tone_core(out_sig, **kw)
    ri = _two_tone_core(in_sig, **kw)
    if ro is None or ri is None:
        return float('nan')
    if ro.n_samples != ri.n_samples:
        raise ValueError(
            f'iip3_dbm: out_sig and in_sig sampled to different lengths '
            f'({ro.n_samples} vs {ri.n_samples}); put both on the same time '
            f'grid or pass n_samples')
    k = ro.f1_bin if ro.P[ro.f1_bin] >= ro.P[ro.f2_bin] else ro.f2_bin
    if ri.P[k] <= 0:
        return float('nan')
    if ro.p_imd <= 0:
        return float('inf')
    p_in_dbm = _tone_dbm(float(ri.P[k]), ri.n_samples, r_ref)
    return p_in_dbm + 0.5 * float(10.0 * np.log10(ro.p_fund / ro.p_imd))


def mix_product(sig: Signal, m: int, n: int, *,
                f1_hz: float, f2_hz: float,
                tclk: float, tstart: float = 0.0,
                n_samples: Optional[int] = None) -> complex:
    """Complex value of the FFT bin at frequency ``m·f1 + n·f2``.

    Useful for fishing out specific intermodulation products (e.g.
    ``mix_product(sig, 2, -1, f1_hz=f1, f2_hz=f2, ...)`` returns the
    lower IMD3 product at ``2f1 − f2``).

    The two-sided FFT is used so negative frequencies wrap naturally.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    m : int
        Signed multiplier of f1_hz, dimensionless.
    n : int
        Signed multiplier of f2_hz, dimensionless.
    f1_hz : float
        Tone frequency, Hz. Only the combined target is rounded to a bin;
        negative and above-Nyquist targets wrap modulo the sample rate.
    f2_hz : float
        Tone frequency, Hz. Only the combined target is rounded to a bin;
        negative and above-Nyquist targets wrap modulo the sample rate.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.

    Returns
    -------
    complex
        Raw, unnormalized DFT coefficient in input amplitude units at the
        nearest bin modulo n_samples. Constant data gives 0j; nonfinite
        selected coefficients propagate, including nan for consumed NaNs.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.

    OverflowError
        An infinite target frequency reaches integer bin conversion without
        a dedicated validation guard.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite selected DFT coefficient.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    """
    samples = _sample_at_clock(sig, metric='mix_product',
                               tclk=tclk, tstart=tstart,
                               n_samples=n_samples)
    selected_samples = samples
    samples = samples - np.mean(samples)
    N = len(samples)
    X = np.fft.fft(samples)
    fs = 1.0 / tclk
    df = fs / N
    target_hz = m * f1_hz + n * f2_hz
    bin_idx = int(round(target_hz / df)) % N
    if not np.isfinite(X[bin_idx]):
        _warn_spectrum('mix_product', sig, selected_samples, tclk, tstart,
                       X[bin_idx], quantity='FFT coefficient')
    return complex(X[bin_idx])


#: Modulations :func:`evm_constellation` and :func:`evm_percent` can build a
#: reference grid. The ``Literal`` is the single source: the
#: schema reflects its values into ``choices`` and :data:`MODULATIONS` is read
#: from it, so the menu and the body guard cannot drift apart.
Modulation = Literal['bpsk', 'qpsk', '8psk', '16qam', '64qam', '256qam',
                     'pam4', 'pam8']


MODULATIONS: tuple[str, ...] = typing.get_args(Modulation)


def modulation_grid(modulation: str) -> np.ndarray:
    """Unit-RMS ideal constellation for a named modulation.

    ``bpsk`` is ``±1``; ``qpsk`` is ``(±1 ± 1j)/√2``; ``8psk`` is the eight
    unit phasors ``exp(j·2πk/8)``; ``16qam``/``64qam``/``256qam`` are the
    square grids of odd integer levels on both axes; ``pam4``/``pam8`` are
    the odd integer levels on the real axis. Every grid is scaled so the RMS
    over its points is exactly 1, and none contains the origin.


    Parameters
    ----------
    modulation : str
        One of bpsk, qpsk, 8psk, 16qam, 64qam, 256qam, pam4, or pam8.

    Returns
    -------
    numpy.ndarray
        New one-dimensional complex array with one point per symbol (2, 4,
        8, 16, 64, or 256 points), dimensionless and normalized to unit RMS.
        No input is mutated; unsupported names raise rather than return nan.

    Raises
    ------
    ValueError
        Unsupported modulation name.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.metrics import modulation_grid
    >>> grid = modulation_grid('qpsk')
    >>> grid.shape, bool(np.isclose(np.mean(np.abs(grid) ** 2), 1.0))
    ((4,), True)
    """
    if modulation not in MODULATIONS:
        raise ValueError(
            f'unknown modulation {modulation!r}; supported: '
            f'{", ".join(MODULATIONS)}')
    if modulation == 'bpsk':
        grid = np.array([-1.0, 1.0], dtype=complex)
    elif modulation == 'qpsk':
        grid = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    elif modulation == '8psk':
        grid = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    elif modulation.endswith('qam'):
        side = math.isqrt(int(modulation[:-3]))
        levels = np.arange(-(side - 1), side, 2, dtype=float)
        grid = (levels[:, None] + 1j * levels[None, :]).ravel()
    else:  # pamN
        order = int(modulation[3:])
        grid = np.arange(-(order - 1), order, 2, dtype=float).astype(complex)
    return grid / np.sqrt(np.mean(np.abs(grid) ** 2))


def _grid_symmetry(modulation: str) -> int:
    """Rotational symmetry order of :func:`modulation_grid` — the power that
    maps every point onto one phase, used to seed the rotation estimate."""
    if modulation == 'bpsk' or modulation.startswith('pam'):
        return 2
    if modulation == '8psk':
        return 8
    return 4                    # qpsk and the square QAM grids


def _fit_complex_gain(received: np.ndarray, grid: np.ndarray,
                      symmetry: int) -> complex:
    """Least-squares complex gain ``g`` minimising ``Σ|r_k − g·d_k|²``.

    ``d_k`` is the grid point nearest ``r_k/g``, so the fit and the decisions
    depend on each other. Two starting estimates are refined: received RMS
    with rotation from the ``symmetry``-th power, and the largest received
    symbol aligned with the largest grid point. For each, decisions and
    the closed-form fit ``g = Σ conj(d_k)·r_k / Σ|d_k|²`` alternate until
    decisions stop changing or 100 iterations are reached. The lower squared
    nearest-point residual wins; ties retain the RMS-start candidate.
    These local fits do not guarantee a global optimum. The fitted rotation
    is only defined modulo the grid's own symmetry, which does not change
    the error vectors. Returns ``0`` for an all-zero symbol set.
    """
    power = float(np.mean(np.abs(received) ** 2))
    if not power > 0:
        return 0j
    lifted = np.mean(received ** symmetry)
    theta = 0.0
    if abs(lifted) > 0:
        theta = (np.angle(lifted) - np.angle(np.mean(grid ** symmetry))) / symmetry
    rms_seed = math.sqrt(power) * complex(np.exp(1j * theta))
    peak_seed = complex(received[np.argmax(np.abs(received))] /
                        grid[np.argmax(np.abs(grid))])
    best_gain = rms_seed
    best_error = float('inf')
    for index, gain in enumerate((rms_seed, peak_seed)):
        previous = None
        for _ in range(100):
            decided = np.argmin(np.abs(received[:, None] - gain * grid[None, :]),
                                axis=1)
            if previous is not None and np.array_equal(decided, previous):
                break
            ideal = grid[decided]
            gain = complex(np.vdot(ideal, received) / np.vdot(ideal, ideal))
            previous = decided
        distances = np.abs(received[:, None] - gain * grid[None, :])
        error = float(np.sum(np.min(distances, axis=1) ** 2))
        if index == 0 or error < best_error:
            best_gain, best_error = gain, error
    return best_gain


@dataclass(frozen=True)
class EVMResult:
    """Result of an :func:`evm_constellation` measurement.


    Parameters
    ----------
    evm_percent : float
        RMS error vector magnitude as a percentage of RMS reference
        amplitude. 0% means zero error against the chosen reference.
    received : numpy.ndarray
        Complex sampled symbols, shape (n_symbols,), in input amplitude units.
    errors : numpy.ndarray
        Per-symbol error vectors, shape (n_symbols,), in input amplitude
        units: ``received − assigned_ideal``.
    assigned_ideals : numpy.ndarray
        Complex array of shape (n_symbols,), in input amplitude units.
        The ideal constellation point each received symbol was scored
        against — the nearest ``ideal_points`` value, or, when
        ``reference`` was given, the ``ideal_points`` entry each
        ``reference`` symbol matched. With ``modulation``, the nearest
        point of the FITTED grid (``fitted_gain`` times the unit grid).
    fitted_gain : complex or None, default None
        With ``modulation``: the least-squares complex gain (scale and
        rotation) mapping the unit-RMS grid onto the received symbols, in input
        amplitude units.
        ``None`` with ``ideal_points``, which are used as given.

    Returns
    -------
    EVMResult
        Frozen result container; constructor values are stored without
        validation or copying. Array fields remain mutable and may alias
        caller arrays. nan fields are permitted; construction emits no warning.

    Attributes
    ----------
    evm_percent : float
        RMS error vector magnitude as a percentage of RMS reference
        amplitude. 0% means zero error against the chosen reference.
    received : numpy.ndarray
        Complex sampled symbols, shape (n_symbols,), in input amplitude units.
    errors : numpy.ndarray
        Per-symbol error vectors, shape (n_symbols,), in input amplitude
        units: ``received − assigned_ideal``.
    assigned_ideals : numpy.ndarray
        Complex array of shape (n_symbols,), in input amplitude units.
        The ideal constellation point each received symbol was scored
        against — the nearest ``ideal_points`` value, or, when
        ``reference`` was given, the ``ideal_points`` entry each
        ``reference`` symbol matched. With ``modulation``, the nearest
        point of the FITTED grid (``fitted_gain`` times the unit grid).
    fitted_gain : complex or None, default None
        With ``modulation``: the least-squares complex gain (scale and
        rotation) mapping the unit-RMS grid onto the received symbols, in input
        amplitude units.
        ``None`` with ``ideal_points``, which are used as given.
    """
    evm_percent: float
    received: np.ndarray
    errors: np.ndarray
    assigned_ideals: np.ndarray
    fitted_gain: Optional[complex] = None


@metric(domain='tran', outputs={'evm_percent': '%'},
        param_units={'symbol_period': 's', 'tstart': 's'})
def evm_constellation(i_sig: Transient, q_sig: Transient, *,
                      symbol_period: float,
                      modulation: Optional[Modulation] = None,
                      ideal_points: Optional[ComplexPoints] = None,
                      tstart: Optional[float] = None,
                      n_symbols: Optional[int] = None,
                      reference: Optional[ComplexPoints] = None) -> EVMResult:
    """Error Vector Magnitude from baseband I/Q transients.

    Samples ``i_sig`` and ``q_sig`` at the centre of each symbol period,
    builds a complex constellation, and measures the RMS error to a
    reference point normalised to the RMS reference amplitude.

    The reference comes from exactly one of two inputs:

    * ``modulation`` — a name from ``MODULATIONS``. silk builds the
      unit-RMS grid (:func:`modulation_grid`) and fits ONE complex gain
      (scale and rotation) to the received symbols by least squares, with
      the decisions and the fit alternated from two starting estimates:
      received RMS and the largest symbol. The lower squared residual wins.
      EVM is then measured against the fitted grid and normalised to its
      RMS, ``|fitted_gain|``. A constellation that is only scaled and
      rotated reads 0 %; amplitude and phase offsets common to every
      symbol are not counted as error, which is the usual EVM convention.
      The fit sees only the symbols present: they should visit the whole
      constellation. A stream that sits on one point (a constant I/Q) is a
      valid one-symbol sequence and reads 0 %; one that only visits the
      inner points of a QAM grid can be fitted to the wrong scale. Incomplete
      coverage emits ``incomplete_constellation`` while keeping the result.
      Each fitted point needs a sample within half the minimum grid spacing,
      inclusive. Full coverage alone does not prove statistical precision
      or that the local fit found the global optimum.
    * ``ideal_points`` — the expert override: the exact complex points, at
      the amplitude and rotation the caller expects. Nothing is fitted, so
      a gain or phase error counts in full.

    Giving both, or neither, raises ``ValueError``. ``reference`` requires
    ``ideal_points``.

    By default, and always when ``reference`` is not given, each received
    symbol is scored against its NEAREST ``ideal_points`` value — the
    textbook definition, and the only one available to a receiver that
    does not know what was transmitted. That nearest-point assignment is
    also a *decision*: once a received point has drifted past the decision
    boundary between two ideal points (inter-symbol interference, a
    closing eye, a runaway phase), it is scored against the WRONG point
    and reads as **closer to ideal than it is** — a wrong symbol can never
    make the error vector bigger than the distance to whichever point it
    got assigned to, but a correctly-attributed error can exceed that
    distance by a lot. So near a decision threshold and beyond, nearest-
    point ``evm_percent`` is a lower bound on the true error, not an
    estimate of it, and it degrades silently: the returned number is
    always finite and plausible-looking, with nothing in ``EVMResult`` to
    say whether any decisions were actually wrong.

    Passing ``reference`` — the known transmitted symbols, when the
    caller has them (e.g. a testbench driving a known pattern) — removes
    this bias: each received symbol is then scored against the
    ``ideal_points`` entry its OWN transmitted symbol names, not the
    nearest one, so a wrong decision costs the full distance to the
    correct point instead of being hidden by re-scoring to the wrong one.

    I/Q time coordinates must agree within 1e-3 of the smallest spacing
    across their grids, in acquisition order. Mismatches raise ``ValueError``;
    singleton grids must agree exactly.


    Parameters
    ----------
    i_sig : Signal
        Baseband I transient, with one-dimensional time in seconds and
        real values in the same amplitude units as q_sig; not mutated.
    q_sig : Signal
        Baseband Q transient; not mutated. Time coordinates must agree
        with i_sig within 1e-3 of the smallest spacing across the grids.
    symbol_period : float
        Positive symbol duration, seconds; values are linearly interpolated
        at tstart + (k + 0.5) * symbol_period.
    modulation : str or None, optional
        One of bpsk, qpsk, 8psk, 16qam, 64qam, 256qam, pam4, or pam8.
        Exactly one of modulation and ideal_points is required.
    ideal_points : array-like of complex, optional
        One-dimensional reference grid in input amplitude units. Must be
        nonempty with no zero point. Exactly one of ideal_points and
        modulation is required; these points are used without fitting.
    tstart : float or None, optional
        Symbol anchor, seconds; None uses i_sig.x[0]. The first sample
        is half a symbol period later and must be covered by both traces.
    n_symbols : int or None, optional
        Number of symbol centres, at least 1; None uses all that fit.
    reference : array-like of complex, optional
        Known transmitted values, shape (n_symbols,), in ideal_points
        units; each must match an ideal point by numpy.isclose. Requires
        ideal_points and scores against matched rather than nearest points.

    Returns
    -------
    EVMResult
        Frozen result with evm_percent in percent and complex received,
        errors, and assigned_ideals arrays of shape (n_symbols,) in the
        input amplitude units. The arrays remain mutable. fitted_gain is
        complex in input units for modulation, or None for explicit points.
        Sampled NaNs give nan EVM; with modulation they also give nan error,
        assignment, and gain values. Zero fitted gain gives nan EVM.
        Explicit-point mode normalizes by RMS of all ideal_points, not
        only the assigned points. Inputs are not mutated.

    Raises
    ------
    ValueError
        Non-transient, empty, invalid, or mismatched time axes; nonpositive
        symbol period; count below 1 or uncovered symbol centres; unknown
        modulation; missing or conflicting reference mode; empty or zero
        ideal points; reference length or membership mismatch where accepted.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        NaNs in interpolated symbols, zero fitted gain, or incomplete
        constellation coverage with modulation. Coverage warnings suggest
        a longer or more varied capture and retain the calculated result.
        Only consumed interpolation support is inspected.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Without reference, n_symbols=2.5 evaluates three symbol centres.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals.base import Signal
    >>> from silk.metrics import evm_constellation
    >>> t = np.arange(5, dtype=float)
    >>> i = Signal('i', 'voltage', 'time', 'V', t, np.ones(5))
    >>> q = Signal('q', 'voltage', 'time', 'V', t, np.zeros(5))
    >>> result = evm_constellation(i, q, symbol_period=1.0,
    ...                            ideal_points=[-1, 1])
    >>> result.evm_percent, result.received.shape
    (0.0, (4,))
    """
    if i_sig.xlabel != 'time':
        raise ValueError(
            f"evm_constellation: i_sig must be a transient (xlabel='time'), "
            f"got {i_sig.xlabel!r}"
        )
    if q_sig.xlabel != 'time':
        raise ValueError(
            f"evm_constellation: q_sig must be a transient (xlabel='time'), "
            f"got {q_sig.xlabel!r}"
        )
    _check_nonempty(i_sig, 'evm_constellation')
    _check_nonempty(q_sig, 'evm_constellation')
    i_axis = _axis(i_sig, 'evm_constellation', order='increasing')
    q_axis = _axis(q_sig, 'evm_constellation', order='increasing')
    if not i_axis.same_grid(q_axis):
        raise ValueError(
            f'evm_constellation: i_sig and q_sig must share the same time axis; '
            f'i_sig has {len(i_sig.x)} points spanning '
            f'{i_sig.x[0]}..{i_sig.x[-1]} s, q_sig has {len(q_sig.x)} points '
            f'spanning {q_sig.x[0]}..{q_sig.x[-1]} s'
        )
    coverage = (f'I trace spans {i_axis.lower}..{i_axis.upper} s '
                f'({len(i_sig.x)} points); Q trace spans '
                f'{q_axis.lower}..{q_axis.upper} s ({len(q_sig.x)} points)')
    if symbol_period <= 0:
        raise ValueError(
            f'evm_constellation: symbol_period must be > 0, '
            f'got {symbol_period}; {coverage}'
        )
    if (modulation is None) == (ideal_points is None):
        raise ValueError(
            'evm_constellation: give exactly one of modulation (a name from '
            f'{", ".join(MODULATIONS)}) or ideal_points (explicit complex '
            'points); got '
            + ('both' if modulation is not None else 'neither'))
    if modulation is not None:
        if modulation not in MODULATIONS:
            raise ValueError(
                f'evm_constellation: unknown modulation {modulation!r}; '
                f'supported: {", ".join(MODULATIONS)}')
        if reference is not None:
            raise ValueError(
                'evm_constellation: reference requires ideal_points; with '
                'modulation the grid is fitted to the received symbols, so '
                'there are no fixed points for reference to name')
        ideals = modulation_grid(modulation)
    else:
        ideals = np.asarray(ideal_points, dtype=complex)
    if ideals.size == 0:
        raise ValueError('evm_constellation: ideal_points must not be empty; '
                         'got 0 reference points')
    zero_indices = np.flatnonzero(ideals == 0)
    if zero_indices.size:
        raise ValueError(
            f'evm_constellation: ideal_points contains a zero-magnitude point '
            f'at index {int(zero_indices[0])} of {ideals.size} reference points; '
            'each ideal point must have nonzero magnitude'
        )
    t_start = float(i_sig.x[0]) if tstart is None else float(tstart)
    first_sample = t_start + 0.5 * symbol_period
    for name, sig in (('i_sig', i_sig), ('q_sig', q_sig)):
        if first_sample < float(sig.x[0]):
            raise ValueError(
                f'evm_constellation: first symbol sample {first_sample} s is '
                f'before {name} for tstart={t_start} s, '
                f'symbol_period={symbol_period} s; trace spans '
                f'{float(sig.x[0])}..{float(sig.x[-1])} s ({len(sig.x)} points)'
            )
    # n_max derived from the sample-centre constraint
    # (n - 0.5)·sp ≤ avail  ⇔  n ≤ avail/sp + 0.5.
    span = _SampleSpan((i_axis, q_axis), t_start, symbol_period, offset=.5)
    n_max = span.max_count
    if n_max < 1:
        raise ValueError(
            f'evm_constellation: window after tstart={t_start} fits 0 symbols of '
            f'period={symbol_period} s; {coverage}'
        )
    if n_symbols is None:
        n_symbols = n_max
    if n_symbols < 1:
        raise ValueError(
            f'evm_constellation: n_symbols must be >= 1, got {n_symbols}; '
            f'{n_max} symbols available; {coverage}'
        )
    if n_symbols > n_max:
        raise ValueError(
            f'evm_constellation: requested {n_symbols} symbols, only {n_max} '
            f'fit in window after tstart={t_start} s with '
            f'symbol_period={symbol_period} s; {coverage}'
        )
    matched_reference = None
    if reference is not None:
        ref_arr = np.asarray(reference, dtype=complex)
        if ref_arr.shape != (n_symbols,):
            raise ValueError(
                f'evm_constellation: reference has {ref_arr.size} symbols '
                f'(shape {ref_arr.shape}), expected {n_symbols} — one per '
                f'evaluated symbol; {coverage}'
            )
        ref_diffs = np.abs(ref_arr[:, None] - ideals[None, :])
        ref_nearest_idx = np.argmin(ref_diffs, axis=1)
        matched_reference = ideals[ref_nearest_idx]
        mismatched = ~np.isclose(ref_arr, matched_reference)
        if np.any(mismatched):
            bad = int(np.flatnonzero(mismatched)[0])
            raise ValueError(
                f'evm_constellation: reference[{bad}]={ref_arr[bad]!r} does '
                f'not match any of the {ideals.size} ideal_points (nearest '
                f'is {matched_reference[bad]!r}); every reference entry must '
                f'be one of ideal_points'
            )
    sample_times = span.times(n_symbols, count_name='n_symbols')
    i_vals = i_axis.interp(sample_times, i_sig.y)
    q_vals = q_axis.interp(sample_times, q_sig.y)
    if _any_nan(i_vals) or _any_nan(q_vals):
        details = {}
        for name, sig, values in (('i', i_sig, i_vals), ('q', q_sig, q_vals)):
            details.update({f'{name}_{key}': value for key, value in
                            _sample_details(sig, values, symbol_period,
                                            sample_times[0]).items()})
        emit(
            metric='evm_constellation', code='nan_symbols',
            message='Measurement unavailable because the interpolated symbol '
                    'data contains NaNs.',
            details=details,
            suggestion='Check the source result at the requested symbol centres '
                       'and their interpolation support.',
        )
    received = i_vals + 1j * q_vals
    if modulation is not None:
        return _evm_against_fitted_grid(received, ideals, modulation,
                                        symbol_period, sample_times[0])
    if matched_reference is not None:
        assigned = matched_reference
    else:
        diffs = received[:, None] - ideals[None, :]
        nearest_idx = np.argmin(np.abs(diffs), axis=1)
        assigned = ideals[nearest_idx]
    errors = received - assigned
    rms_err = float(np.sqrt(np.mean(_nan_safe_transform(errors, np.abs) ** 2)))
    rms_ref = float(np.sqrt(np.mean(np.abs(ideals) ** 2)))
    if rms_ref == 0:
        evm = float('inf')
    else:
        evm = 100.0 * rms_err / rms_ref
    return EVMResult(evm, received, errors, assigned)


def _evm_against_fitted_grid(received: np.ndarray, grid: np.ndarray,
                             modulation: str, symbol_period: float,
                             first_sample: float) -> EVMResult:
    """The ``modulation`` branch of :func:`evm_constellation`."""
    nan = float('nan')
    if _any_nan(received):
        # nan_symbols was already emitted; no gain can be fitted.
        unknown = np.full(received.shape, complex(nan, nan))
        return EVMResult(nan, received, unknown, unknown,
                         complex(nan, nan))
    gain = _fit_complex_gain(received, grid, _grid_symmetry(modulation))
    if gain == 0:
        emit(
            metric='evm_constellation', code='zero_gain',
            message='Measurement unavailable because the fitted constellation '
                    'gain is zero: the sampled symbols carry no signal.',
            details={'modulation': modulation, 'n_symbols': received.size,
                     'symbol_period': symbol_period,
                     'first_sample': float(first_sample)},
            suggestion='Check that I and Q carry the modulated signal at the '
                       'symbol centres.',
        )
        zeros = np.zeros(received.shape, dtype=complex)
        return EVMResult(nan, received, received - zeros, zeros, gain)
    fitted = gain * grid
    distances = np.abs(received[:, None] - fitted[None, :])
    assigned = fitted[np.argmin(distances, axis=1)]
    if np.isfinite(gain):
        spacing = np.abs(grid[:, None] - grid[None, :])
        radius = .5 * abs(gain) * float(np.min(spacing[spacing > 0]))
        covered = int(np.count_nonzero(np.min(distances, axis=0) <= radius))
        if covered < grid.size:
            emit(
                metric='evm_constellation', code='incomplete_constellation',
                message='Some fitted constellation points have no nearby '
                        'samples; limited coverage can make the fit ambiguous.',
                details={'modulation': modulation, 'n_symbols': received.size,
                         'covered_points': covered, 'total_points': grid.size,
                         'proximity_fraction': .5},
                suggestion='Use a longer or more varied capture to visit all '
                           'constellation points.',
            )
    errors = received - assigned
    rms_err = float(np.sqrt(np.mean(_nan_safe_transform(errors, np.abs) ** 2)))
    return EVMResult(100.0 * rms_err / abs(gain), received, errors, assigned,
                     gain)


@metric(domain='tran', unit='%',
        param_units={'symbol_period': 's', 'tstart': 's'})
def evm_percent(i_sig: Transient, q_sig: Transient, *,
                symbol_period: float, modulation: Modulation,
                tstart: Optional[float] = None,
                n_symbols: Optional[int] = None) -> float:
    """RMS error vector magnitude in percent, against a named modulation.

    Scalar view of :func:`evm_constellation` with ``modulation=``: silk
    builds the unit-RMS grid for ``modulation``, fits one complex gain
    (scale and rotation) to the received symbols by least squares, and
    reports the RMS error against the fitted grid as a percentage of its
    RMS. Inputs are scalars and one name, so a schema consumer can offer
    it. For explicit points or a known transmitted ``reference``, call
    :func:`evm_constellation` with ``ideal_points=`` instead. Inherits that
    function's validation, sampling and NaN diagnostics.

    Parameters
    ----------
    i_sig : Signal
        Baseband I transient, with one-dimensional time in seconds and
        real values in the same amplitude units as q_sig; not mutated.
    q_sig : Signal
        Baseband Q transient; not mutated. Time coordinates must agree
        with i_sig within 1e-3 of the smallest spacing across the grids.
    symbol_period : float
        Positive symbol duration, seconds; values are linearly interpolated
        at tstart + (k + 0.5) * symbol_period.
    modulation : str
        One of bpsk, qpsk, 8psk, 16qam, 64qam, 256qam, pam4, or pam8.
    tstart : float or None, optional
        Symbol anchor, seconds; None uses i_sig.x[0]. The first sample
        is half a symbol period later and must be covered by both traces.
    n_symbols : int or None, optional
        Number of symbol centres, at least 1; None uses all that fit.

    Returns
    -------
    float
        RMS error, percent of the fitted-grid RMS amplitude. Sampled NaNs
        or zero fitted gain give nan; constant nonzero I/Q can fit one point
        and give zero error. Uses evm_constellation sampling and fit.

    Raises
    ------
    ValueError
        Non-transient, empty, invalid, or mismatched time axes; nonpositive
        symbol period; count below 1 or uncovered symbol centres; unknown
        modulation.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        NaNs in sampled symbols, zero fitted gain, or incomplete constellation
        coverage, emitted once by evm_constellation. Incomplete coverage
        keeps the numeric result and suggests a longer or more varied capture.
        Warnings follow the active Python warning filter.
    """
    _check_transient(i_sig, 'evm_percent')
    _check_transient(q_sig, 'evm_percent')
    return evm_constellation(
        i_sig, q_sig, symbol_period=symbol_period, modulation=modulation,
        tstart=tstart, n_symbols=n_symbols).evm_percent


def _check_power_sweep(sig: Signal, metric: str) -> None:
    """Reject anything that is not a usable power sweep.

    Every failure here is CALLER ERROR — the request is malformed
    whatever the circuit did — so each raises ``ValueError`` naming the
    metric, what was asked for and what was found.
    """
    if sig.xlabel != 'power':
        raise ValueError(
            f"{metric}: sig must be a power-sweep Signal (xlabel='power', "
            f"x = p_in in dBm, y = p_out in dBm), got xlabel={sig.xlabel!r}"
        )
    _check_nonempty(sig, metric)
    if len(sig.y) != len(sig.x):
        raise ValueError(
            f"{metric}: sig.x (p_in) and sig.y (p_out) must have the same "
            f"length, got {len(sig.x)} and {len(sig.y)}"
        )
    if len(sig.x) < 2:
        raise ValueError(
            f'{metric}: need at least 2 sweep points, got {len(sig.x)}'
        )
    _axis(sig, metric, order='nondecreasing')


def _power_sweep_details(sig: Signal, p_in: np.ndarray, p_out: np.ndarray) -> dict:
    """Facts about the full sweep consumed by p1db's existing NaN guard."""
    in_min, in_max = _observed_range(p_in)
    out_min, out_max = _observed_range(p_out)
    return {
        'signal_name': sig.name, 'sample_count': len(p_in),
        'input_power_min_dbm': in_min, 'input_power_max_dbm': in_max,
        'output_power_min_dbm': out_min, 'output_power_max_dbm': out_max,
        'nan_input_power_count': int(np.count_nonzero(np.isnan(p_in))),
        'nan_output_power_count': int(np.count_nonzero(np.isnan(p_out))),
    }


# p1db takes an output-vs-input power curve (xlabel='power', both in dBm),
# assembled from one run per input power with silk.signals.power_sweep.
# No single analysis produces this curve, so its schema kind 'power_sweep'
# lists no producing analysis. Schema-reading tools do not offer p1db as a
# probe-wired metric; Python callers can still call it.
@metric(domain='dc', unit='dBm')
def p1db(sig: PowerSweep) -> float:
    """Input 1-dB compression point, in dBm.

    Walks through a power-sweep curve and returns the first ``p_in`` at
    which the small-signal gain has dropped by 1 dB from its initial
    value.

    Parameters
    ----------
    sig : Signal
        Power sweep (schema ``kind='power_sweep'``): ``xlabel='power'``,
        ``x`` = input power in dBm (nondecreasing; repeated powers allowed), ``y`` = output
        power in dBm. Strict — a Signal on any other axis is rejected,
        the same way the transfer-curve metrics here demand
        ``xlabel='code'``. Arrays are one-dimensional, equal-length, shape
        (N,), N >= 2; finite x coordinates are required. Input is not modified.

    Returns
    -------
    float
        The input power (dBm) at which gain is 1 dB below its
        small-signal value; ``nan`` when no 1 dB compression is observed
        in the measured range or output powers contain NaNs.

    Notes
    -----
    Resolution: linear interpolation between the two sweep points that
    bracket the 1 dB drop (exact for a piecewise-linear gain curve);
    the small-signal reference is the FIRST point's gain, so a sweep
    that is already compressed at its first point cannot show it — the
    reference compresses with the sweep and the drop is measured from
    there.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For NaN output powers or no observed 1 dB compression, returning
        nan. Warning filters apply normally, including warnings-as-errors.

    Raises
    ------
    ValueError
        For a wrong x-axis, fewer than two samples, unequal or non-1-D
        arrays, or non-finite or descending input-power coordinates.
    """
    _check_power_sweep(sig, 'p1db')
    p_in  = np.asarray(sig.x, dtype=float)
    p_out = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    if _any_nan(p_in, p_out):
        emit(
            metric='p1db', code='nan_power_sweep',
            message='Compression measurement unavailable: the consumed '
                    'input/output power sweep (dBm) contains NaNs.',
            details=_power_sweep_details(sig, p_in, p_out),
            suggestion='Check the source input and output power results '
                       'for the affected sweep samples.',
        )
        return float('nan')
    g0 = p_out[0] - p_in[0]
    for k in range(len(p_in)):
        g_k = p_out[k] - p_in[k]
        if g0 - g_k >= 1.0:
            if k == 0:
                return float(p_in[0])
            g_prev = p_out[k - 1] - p_in[k - 1]
            num = (g0 - 1.0) - g_prev
            den = g_k - g_prev
            if den == 0:
                return float(p_in[k])
            frac = num / den
            return float(p_in[k - 1] + frac * (p_in[k] - p_in[k - 1]))
    emit(
        metric='p1db', code='no_compression_observed',
        message='No 1 dB compression relative to the first-point gain was '
                'observed in the measured input-power range (dBm).',
        details={**_power_sweep_details(sig, p_in, p_out),
                 'reference_gain_db': float(g0), 'compression_threshold_db': 1.0},
        suggestion='Check the first-point gain reference and sweep coverage; '
                   'try extending the input-power range. Compression outside '
                   'this range is not established.',
    )
    return float('nan')
