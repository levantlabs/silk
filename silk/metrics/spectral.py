"""FFT and power-spectrum metrics from clock-sampled time waveforms.

**Resolution**

* **Frequency**: one FFT bin, ``df = 1 / (n_samples · tclk)``. Every
  reported frequency (``SFDR.fundamental_hz``, ``spur_hz``,
  ``harmonic_freqs``) is a bin centre; ``SFDR.bin_hz`` carries ``df``.
* **Sample timing**: values at ``tstart + k·tclk`` use linear interpolation
  between recorded points, retaining exact matches. A ``MetricWarning``
  summarizes off-grid counts and the maximum absolute distance to the nearest
  recorded time. This distance is not a voltage-error bound; significant
  distances may warrant finer simulation time resolution. Clock period and
  phase are unchanged. Endpoint-roundoff allowance remains unchanged.
* **Ratio metrics** (``sndr``, ``sinad``, ``snr``, ``enob``,
  ``noise_floor_dbfs``): when the floor is random noise its estimate is
  a mean over ``n_samples/2`` bins, so the statistical resolution is
  ≈ 4.3 / √(n_samples/2) dB — 0.54 dB (0.09 bit) at 128 samples,
  0.19 dB (0.03 bit) at 1024. Clean tones reach the double-precision
  power floor, not infinity: every residual bin carries at least
  ``eps = np.finfo(float).eps``. The ceiling depends on signal power
  and sample count (SNDR ≈ 172.69 dB / ENOB ≈ 28.39 bits for a 0.4-peak,
  256-sample coherent sine). SNDR sums the residual bins directly;
  SFDR uses their maximum, so ``sndr <= sfdr.sfdr_db``.
* **Harmonics above Nyquist** are folded back in band (``thd``,
  ``thd_percent``, ``harmonic_levels_db``, ``snr``), matching
  the aliasing sampling already applied to the data: ``b_raw =
  (k·fund_bin) mod n_samples``, mirrored to ``n_samples − b_raw`` when
  that exceeds the Nyquist bin. A harmonic folding onto DC or onto the
  fundamental bin is unattributable and excluded; two harmonics folding
  onto the same bin are counted once, never summed twice, and the same
  bin set is excluded from ``snr``'s noise sum so the two metrics stay
  mutually consistent. An empty THD harmonic selection or SNR residual
  selection returns ``nan``; no bins means unavailable data, not zero
  distortion/noise. ``harmonic`` reads the same folded bin (a raw DFT
  coefficient, while ``harmonic_levels_db`` compares weighted powers;
  its phase is conjugated back when the fold mirrored — the
  ``mix_product`` two-sided convention), and ``harmonic_freqs`` reports
  the *true* ``k·f0`` for every harmonic (no Nyquist truncation) with
  ``observed=True`` giving the folded bin centre instead. A
  harmonic that is unattributable (folds onto DC / the fundamental) is
  ``nan`` with a diagnostic in ``harmonic_levels_db`` and ``nan`` in
  ``harmonic`` /
  ``harmonic_freqs(observed=True)`` — the same skip set, one map. See
  ``_harmonic_bin_map``. ``imd3_db``'s two 3rd-order products fold
  in-band the same way (generalised as
  ``_fold_raw_bin_mirrored`` since a product bin is not a multiple
  of a fundamental); a product folding onto DC or onto either requested
  tone is unattributable and returns ``nan``, the IMD3 analogue of the
  DC/fundamental exclusion above.
* **Sampled power**: ratio metrics use one-sided power, doubling each
  conjugate pair but not DC or even-length Nyquist. The last bin of an
  odd-length record is paired. This measures the sampled record; it cannot
  reconstruct analog amplitude lost when a Nyquist sine samples to zero.
  ``fft`` and ``harmonic`` retain raw DFT power/coefficients.
* **No fundamental** (dead / DC / bit-exact-constant input): the power
  spectrum is eps-floored, so ratios with nonempty required selections
  with the default reference return a finite floor value instead of
  ``inf`` — ``sndr`` =
  ``−10·log10(n_samples/2 − 1)`` dB, SFDR 0 dB, THD +6 dB, harmonic
  levels 0 dBc. An empty harmonic/noise selection instead returns ``nan``,
  even on dead data (e.g. SNR at four samples). Explicit ``full_scale``
  bypasses the measured reference: dead data can then report high SNDR
  or ENOB against the eps floor without warning. See :func:`silk.metrics.sndr`.

Spectrum options
---------------------------
Every metric that names a fundamental takes its spectrum from one core
(``_tone_spectrum``), so the options mean the same thing everywhere:

* ``fundamental_hz=None`` — ``None`` elects the strongest non-DC bin;
  a frequency selects the nearest bin, and one that rounds to DC or
  past the last bin raises ``ValueError``. All of ``sndr``, ``sinad``,
  ``enob``, ``snr``, ``sfdr``, ``sfdr_db``, ``thd``, ``thd_percent``,
  ``noise_floor_dbfs``, ``harmonic_levels_db``, ``harmonic_level_dbc``,
  ``harmonic`` and ``harmonic_freqs`` take it.
* ``window='rectangular'`` and ``guard_bins=0`` (the power-ratio metrics:
  ``sndr``, ``sinad``, ``enob``, ``snr``, ``sfdr``, ``sfdr_db``,
  ``thd``, ``thd_percent``, ``noise_floor_dbfs``). The window
  (``'hann'``, ``'hamming'``, ``'blackman'``) multiplies the
  mean-removed samples. Each tone owns a band of ``lobe + guard_bins``
  bins on each side of its centre, where ``lobe`` is the window's main
  lobe (0 rectangular, 1 hann and hamming, 2 blackman): the fundamental's
  band is its signal power, each harmonic's band its distortion power,
  the spur's band its power, and every band is left out of the noise
  sums. DC owns the band ``0..lobe + guard_bins`` the same way, and the
  fundamental is elected above DC's window lobe: removing the
  sample mean before a window leaves a DC-centred window lobe on a
  non-coherent record, which is neither noise nor a spur. The noise sums
  of ``sndr``/``sinad``/``enob``/``snr`` are then multiplied by
  ``M / (M − excluded)``, ``M`` being the residual count of the default
  single-bin selection and ``excluded`` the bins the lobe and guard
  bands take on top of it, so wider bands do not read high. With the
  defaults each band is the single centre bin and the factor is 1, so
  the results are unchanged. A ``guard_bins`` that leaves no noise bins,
  or puts the strongest tone inside DC's band, raises ``ValueError``.
* ``full_scale=None`` (``sndr``, ``sinad``, ``enob``, ``snr``,
  ``noise_floor_dbfs``) — the peak amplitude of a full-scale sine, in the
  signal's unit (e.g. ``2**(bits-1)`` for a two's-complement code). The
  ratio is then referred to that sine's power instead of the measured
  fundamental's: full-scale SNR/SNDR in dBFS, ENOB credited with the
  input's back-off. ``None`` keeps the measured fundamental.

Options that do not fit raise ``ValueError``;
data that cannot be measured still returns a number as above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math

import numpy as np

from silk.utils._compat import trapezoid
from silk.signals.base import Signal
from silk.signals.schema import metric, Transient
from silk.metrics._guards import (
    _nan_safe_transform, _axis, _SampleSpan, _check_nonempty, _any_nan,
)
from silk.metrics.diagnostics import emit
from silk.metrics.waveform import _check_transient


@dataclass(frozen=True)
class SFDR:
    """Result of an SFDR measurement.


    Parameters
    ----------
    sfdr_db : float
        Spurious-free dynamic range, dB.
    fundamental_hz : float
        Frequency of the fundamental, Hz.
    spur_hz : float
        Frequency of the largest non-fundamental, non-DC spur, Hz.
    bin_hz : float, default nan
        FFT bin width in Hz, ``1 / (n_samples · tclk)`` — the resolution of both
        frequencies above (each is a bin centre, true value within
        ±``bin_hz``/2).

    Returns
    -------
    SFDR
        Frozen result. Frequencies and bin width are in Hz, sfdr_db in dB.
        Constructor values are stored as supplied without validation; nan
        fields are permitted. No measurement or warning occurs here.

    Attributes
    ----------
    sfdr_db : float
        Spurious-free dynamic range, dB.
    fundamental_hz : float
        Frequency of the fundamental, Hz.
    spur_hz : float
        Frequency of the largest non-fundamental, non-DC spur, Hz.
    bin_hz : float, default nan
        FFT bin width in Hz, ``1 / (n_samples · tclk)`` — the resolution of both
        frequencies above (each is a bin centre, true value within
        ±``bin_hz``/2).
    """
    sfdr_db: float
    fundamental_hz: float
    spur_hz: float
    bin_hz: float = float('nan')


def _sample_at_clock(sig: Signal, *, metric: str, tclk: float,
                     tstart: float = 0.0,
                     n_samples: Optional[int] = None) -> np.ndarray:
    """Extract one sample per clock period from a transient Signal.

    Sample times are ``tstart + k·tclk`` for ``k = 0..n_samples-1``.
    Use the shared linear interpolator on uniform or irregular time grids;
    exact stored-time matches retain their recorded values. Off-grid requests
    emit one summary with counts and absolute nearest-recorded-time distance.
    The existing endpoint-roundoff allowance clamps only the final boundary.
    """
    _check_transient(sig, metric)
    axis = _axis(sig, metric, order='increasing')
    t = axis.x
    coverage = f'trace spans {float(t[0])} .. {float(t[-1])} s ({len(t)} points)'
    if tclk <= 0:
        raise ValueError(f'{metric}: tclk must be > 0, got {tclk}; {coverage}')
    if tstart < float(t[0]):
        raise ValueError(
            f'{metric}: tstart {tstart} is before the start of the trace; '
            f'{coverage}'
        )
    avail = float(t[-1]) - tstart
    if avail < tclk:
        raise ValueError(
            f'{metric}: trace too short after tstart={tstart}: need at least one '
            f'period ({tclk} s), have {avail} s; {coverage}'
        )
    span = _SampleSpan((axis,), tstart, tclk)
    max_n = span.max_count
    if n_samples is None:
        n_samples = 1 << int(math.floor(math.log2(max_n)))
    if n_samples < 4:
        raise ValueError(
            f'{metric}: n_samples must be >= 4 for a meaningful spectrum, '
            f'got {n_samples}; {max_n} samples available at tclk={tclk} s '
            f'after tstart={tstart} s; {coverage}'
        )
    sample_times = span.times(n_samples)
    # Preserve the requested instants for diagnostics even when _SampleSpan
    # admits and clamps an endpoint rounded just beyond the recorded trace.
    requested = tstart + np.arange(n_samples) * tclk
    right = np.clip(axis.search(requested), 0, len(t) - 1)
    left = np.maximum(right - 1, 0)
    distances = np.minimum(np.abs(requested - t[left]),
                           np.abs(requested - t[right]))
    effective_right = np.clip(axis.search(sample_times), 0, len(t) - 1)
    interpolated = int(np.count_nonzero(t[effective_right] != sample_times))
    clamped = int(np.count_nonzero(requested != sample_times))
    if interpolated or clamped:
        emit(
            metric=metric, code='interpolated_samples',
            message=(f'Linearly interpolated {interpolated} of {n_samples} '
                     f'requested samples. Maximum absolute distance to the '
                     f'nearest recorded time: {float(distances.max()):.6g} s.'
                     + (f' {clamped} endpoint sample(s) used the recorded '
                        'boundary under the endpoint-roundoff allowance.'
                        if clamped else '')),
            details={'signal_name': sig.name, 'sample_count': int(n_samples),
                     'interpolated_sample_count': interpolated,
                     'max_nearest_time_distance_s': float(distances.max()),
                     'endpoint_clamped_sample_count': clamped},
            suggestion=('Significant distances may warrant finer simulation '
                        'time resolution around the requested instants; '
                        'roundoff-sized distances generally need no action. '
                        'Keep the intended clock period and phase unchanged.'),
        )
    # Protect NaNs in either complex component before projecting to real:
    # only interpolation support is consumed, so unused NaNs remain harmless.
    values = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    return axis.interp(sample_times, values)


def _rfft_spectrum(samples: np.ndarray, window: Optional[str] = None, *,
                   metric: str = 'spectrum',
                   ) -> tuple[np.ndarray, np.ndarray, int]:
    """DC-removed rfft; returns ``(X, P, fundamental_bin)``.

    ``window`` (a :func:`_make_window` name) multiplies the mean-removed
    samples before the FFT; ``None``/``'rectangular'`` is today's
    unwindowed spectrum, computed exactly as before.

    ``P = w·|X|² + eps``: conjugate pairs have weight 2; DC and the
    even-length Nyquist bin have weight 1. For odd lengths the last bin
    is paired too. Before the floor, ``sum(P)/N²`` is the mean-square
    value of the centered samples, not inferred pre-sampling analog power.
    ``X`` retains the raw DFT coefficients. ``eps = np.finfo(float).eps``
    is added *after* weighting so every empty bin has the same floor.
    The floor is the no-fundamental guard: on a
    bit-exact-constant input every bin is 0, the elected "fundamental"
    is 0 and every ratio metric was 0/0 → ``inf``. With the floor the
    ratio becomes ``eps / ((n_bins − 1)·eps)`` — a finite, always-failing
    number that is independent of the eps magnitude. On a clean tone,
    residual bins retain this double-precision floor when summed directly,
    giving a finite upper limit set by the fundamental power and the
    number of residual bins. Summing them with the fundamental first
    would absorb the floor and lose the residual to rounding.
    The fundamental election itself is the strongest non-DC bin, so the
    single election path is shared by every metric that names one.
    """
    x = samples - np.mean(samples)
    if window is not None and window != 'rectangular':
        x = x * _make_window(window, len(samples), metric=metric)
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2
    P[1:-1 if len(samples) % 2 == 0 else None] *= 2
    P += np.finfo(float).eps
    fund_bin = int(np.argmax(P[1:]) + 1)
    return X, P, fund_bin


def _power_spectrum(samples: np.ndarray) -> tuple[np.ndarray, int]:
    """DC-removed, eps-floored power spectrum; returns (P, fundamental_bin).
    See :func:`_rfft_spectrum` for the floor."""
    _, P, fund_bin = _rfft_spectrum(samples)
    return P, fund_bin


#: Windows the tone-spectrum metrics accept, as a ``Literal`` so the schema
#: reflects them into ``choices`` (``None`` also means rectangular).
SpectrumWindow = Literal['hann', 'hamming', 'blackman', 'rectangular']


#: Bins on each side of a coherent tone's centre that the window's main lobe
#: spreads its power into (the lobe's first null is one bin further out).
#: These bins belong to the tone, so a windowed tone is measured whole.
_WINDOW_LOBE_BINS = {None: 0, 'rectangular': 0, 'hann': 1, 'hamming': 1,
                     'blackman': 2}


@dataclass(frozen=True)
class _ToneSpectrum:
    """One sampled record's power spectrum with its fundamental chosen.

    ``half`` is the half-width, in bins, of the band each tone owns: the
    window's main lobe (``lobe``) plus ``guard_bins``. The fundamental's
    band is its signal power, a harmonic's band its distortion power, and
    both bands are left out of every noise and spur sum. DC owns the band
    ``0..half`` the same way: mean removal before a window
    leaves a DC-centred window lobe on a non-coherent record, which is
    not noise and not a spur. With the defaults (rectangular,
    ``guard_bins=0``) each band is the single centre bin, which is
    today's result exactly.
    """
    samples: np.ndarray
    X: np.ndarray
    P: np.ndarray
    fund_bin: int
    half: int
    full_scale_power: Optional[float]
    lobe: int = 0
    guard_bins: int = 0

    def band(self, center: int) -> np.ndarray:
        """Bins ``center ± half``, clipped to the non-DC spectrum."""
        lo = max(1, center - self.half)
        hi = min(len(self.P) - 1, center + self.half)
        return np.arange(lo, hi + 1)

    @property
    def fund_power(self):
        if self.half == 0:
            return self.P[self.fund_bin]
        return float(np.sum(self.P[self.band(self.fund_bin)]))

    def reference_power(self):
        """The ratio reference: a full-scale sine's power when the caller
        named ``full_scale``, else the measured fundamental's."""
        if self.full_scale_power is not None:
            return self.full_scale_power
        return self.fund_power

    def _mask(self, half: int, n_harmonics: Optional[int]) -> np.ndarray:
        exclude = (None if n_harmonics is None else _harmonic_power_bins(
            self.fund_bin, len(self.samples), n_harmonics, half))
        return _residual_bins(self.P, self.fund_bin, exclude, half=half)

    def residual(self, metric: str,
                 n_harmonics: Optional[int] = None) -> tuple[np.ndarray, float]:
        """``(mask, scale)``: the noise bins and the factor that refers
        their sum to the default selection.

        The mask leaves out DC's band, the fundamental's and, with
        ``n_harmonics``, every harmonic's. ``scale = M / (M − excluded)``,
        where ``M`` is the residual count of the default single-bin
        selection (``half = 0``) and ``excluded`` the bins the window lobe
        and ``guard_bins`` take on top of it: the residual sum times
        ``scale`` estimates the noise in the same ``M`` bins whatever the
        options, so excluding lobe or guard bins does not bias SNR/SNDR
        high. With the defaults ``scale`` is exactly 1.

        A ``guard_bins`` that leaves no residual when the window lobe
        alone would leave some is a caller error: ``ValueError`` naming
        ``guard_bins`` and the bin budget. An empty residual
        without ``guard_bins`` is degenerate data; the caller returns
        ``nan`` with the ``no_residual_bins`` diagnostic.
        """
        mask = self._mask(self.half, n_harmonics)
        count = int(np.count_nonzero(mask))
        if count == 0:
            if self.guard_bins and np.any(self._mask(self.lobe, n_harmonics)):
                self._raise_guard(metric, 'leaves no residual (noise) bins')
            return mask, float('nan')
        if self.half == 0:
            return mask, 1.0
        base = int(np.count_nonzero(self._mask(0, n_harmonics)))
        return mask, base / count

    def _raise_guard(self, metric: str, what: str):
        n_bins = len(self.P) - 1
        raise ValueError(
            f'{metric}: guard_bins={self.guard_bins} {what}: each tone and DC '
            f'own {self.half} bins on each side (window lobe {self.lobe} + '
            f'guard_bins {self.guard_bins}), fundamental at bin '
            f'{self.fund_bin}, non-DC bins 1..{n_bins} '
            f'(n_samples={len(self.samples)}); use a smaller guard_bins or '
            f'a longer record')


def _tone_spectrum(sig: Signal, *, metric: str, tclk: float,
                   tstart: float = 0.0, n_samples: Optional[int] = None,
                   fundamental_hz: Optional[float] = None,
                   window: Optional[str] = 'rectangular',
                   guard_bins: int = 0,
                   full_scale: Optional[float] = None) -> _ToneSpectrum:
    """Sample, window and transform a record, then choose its fundamental.

    The single core behind every metric that names a fundamental.
    ``fundamental_hz=None`` elects the strongest non-DC bin (today's
    rule); a value selects the nearest bin and raises ``ValueError`` when
    that bin is DC or beyond the spectrum. ``full_scale`` is the peak
    amplitude of a full-scale sine in the signal's unit; its power,
    ``full_scale²/2 · N · Σw²`` in this spectrum's units, is what a
    coherent sine of that amplitude sums to over its band.
    """
    if (isinstance(guard_bins, (bool, np.bool_))
            or not isinstance(guard_bins, (int, np.integer)) or guard_bins < 0):
        raise ValueError(
            f'{metric}: guard_bins must be an integer >= 0, got {guard_bins!r}')
    if full_scale is not None and not (np.isfinite(full_scale)
                                       and full_scale > 0):
        raise ValueError(
            f'{metric}: full_scale must be a finite amplitude > 0 '
            f'(peak of a full-scale sine, in the signal unit), '
            f'got {full_scale!r}')
    if fundamental_hz is not None and not np.isfinite(fundamental_hz):
        raise ValueError(
            f'{metric}: fundamental_hz must be finite, got {fundamental_hz!r}')
    samples = _sample_at_clock(sig, metric=metric, tclk=tclk, tstart=tstart,
                               n_samples=n_samples)
    n = len(samples)
    w = _make_window(window, n, metric=metric)
    X, P, fund_bin = _rfft_spectrum(samples, window, metric=metric)
    lobe = _WINDOW_LOBE_BINS[window]
    half = lobe + int(guard_bins)
    if fundamental_hz is None and lobe and lobe < len(P) - 1:
        # DC's window lobe is not a tone (TASK-436): elect above it.
        fund_bin = int(np.argmax(P[lobe + 1:]) + lobe + 1)
    if fundamental_hz is not None:
        df = 1.0 / (n * tclk)
        b = int(round(fundamental_hz / df))
        if not 0 < b < len(P) or fundamental_hz > (len(P) - 1) * df:
            raise ValueError(
                f'{metric}: fundamental_hz={fundamental_hz} Hz (bin {b}) is '
                f'outside the spectrum: non-DC bins 1..{len(P) - 1}, '
                f'{df}..{(len(P) - 1) * df} Hz (df={df} Hz, '
                f'n_samples={n})')
        fund_bin = b
    fs_power = None
    if full_scale is not None:
        fs_power = float(full_scale) ** 2 / 2.0 * n * float(np.sum(w * w))
    t = _ToneSpectrum(samples, X, P, fund_bin, half, fs_power,
                      lobe, int(guard_bins))
    if fundamental_hz is None and guard_bins and fund_bin <= half:
        t._raise_guard(metric, f'puts the strongest tone (bin {fund_bin}) '
                               f'inside the DC band 0..{half}')
    return t


def _sample_details(sig, samples, tclk, tstart):
    """Describe only selected samples, with nominal sampling times in seconds."""
    non_nan = samples[~np.isnan(samples)]
    return {
        'signal_name': sig.name, 'sample_count': len(samples),
        'sample_start_s': float(tstart),
        'sample_end_s': float(tstart + (len(samples) - 1) * tclk),
        'sample_interval_s': float(tclk),
        'sample_min': float(np.min(non_nan)) if non_nan.size else None,
        'sample_max': float(np.max(non_nan)) if non_nan.size else None,
        'sample_unit': sig.ylabel or 'unspecified',
        'nan_sample_count': int(np.count_nonzero(np.isnan(samples))),
        'inf_sample_count': int(np.count_nonzero(np.isinf(samples))),
    }


def _warn_spectrum(metric, sig, samples, tclk, tstart, spectrum, *,
                   quantity='FFT power'):
    """Explain a non-finite spectral operand without changing its consumer."""
    details = _sample_details(sig, samples, tclk, tstart)
    details.update(
        spectral_value_count=int(np.size(spectrum)),
        nonfinite_spectral_value_count=int(np.count_nonzero(
            ~np.isfinite(spectrum))),
    )
    if details['nan_sample_count']:
        code = 'nan_samples'
        message = 'Measurement unavailable because the consumed data contains NaNs.'
        suggestion = 'Check the source result for the consumed samples.'
    else:
        code = 'nonfinite_spectrum'
        message = f'{quantity} is non-finite for the consumed samples.'
        suggestion = 'Check the source result and input scaling for overflow.'
        if not details['inf_sample_count']:
            message += ' The consumed samples are finite.'
    emit(metric=metric, code=code, message=message, details=details,
         suggestion=suggestion)


def _undefined_power_ratio(numerator, denominator):
    """NaN powers and inf/inf propagate NaN; finite/inf and inf/finite do not."""
    return (np.any(np.isnan(numerator)) or np.isnan(denominator)
            or (np.isinf(denominator) and np.any(np.isinf(numerator))))


def _warn_empty_bins(metric, sig, samples, tclk, tstart, fund_bin,
                     n_harmonics, *, residual=False):
    details = _sample_details(sig, samples, tclk, tstart)
    details.update(fundamental_bin=fund_bin, n_harmonics=n_harmonics,
                   selected_bin_count=0, bin_hz=1.0 / (len(samples) * tclk))
    emit(
        metric=metric,
        code='no_residual_bins' if residual else 'no_harmonic_bins',
        message=('No noise bins remain after excluding DC, the fundamental '
                 'and requested harmonics.' if residual else
                 'No requested harmonic has a folded bin separate from DC '
                 'and the fundamental.'),
        details=details,
        suggestion='Try a longer record, a different coherent tone or sampling '
                   'rate, or fewer requested harmonics.',
    )


def _warn_harmonic_location(metric, sig, samples, tclk, tstart, fund_bin, k):
    details = _sample_details(sig, samples, tclk, tstart)
    details.update(harmonic=k, folded_bin=_fold_bin(k, fund_bin, len(samples)),
                   fundamental_bin=fund_bin,
                   bin_hz=1.0 / (len(samples) * tclk))
    emit(
        metric=metric, code='unattributable_harmonic',
        message='Requested harmonic unavailable because its folded bin '
                'coincides with DC or the fundamental.',
        details=details,
        suggestion='Try a sampling rate or coherent tone that separates '
                   'the requested harmonic from DC and the fundamental.',
    )


def _fold_raw_bin_mirrored(b_signed: int, n_samples: int) -> tuple[int, bool]:
    """``(bin, mirrored)`` — the rfft bin that a raw two-sided-DFT index
    ``b_signed`` (any integer, mod ``n_samples``) aliases to, folded
    in-band, and whether the fold *mirrored* it (``b_raw > n_samples //
    2``). A mirrored bin holds the conjugate of that index's phasor
    (``X[N − b] = conj(X_full[b])`` for a real signal — true for ANY
    index, not just a harmonic of a fundamental). This is the ONE place
    the fold arithmetic lives; :func:`_fold_bin_mirrored` (harmonics)
    and :func:`imd3_db` (intermod products) both specialize it.
    """
    b_raw = b_signed % n_samples
    if b_raw > n_samples // 2:
        return n_samples - b_raw, True
    return b_raw, False


def _fold_bin_mirrored(k: int, fund_bin: int,
                       n_samples: int) -> tuple[int, bool]:
    """``(bin, mirrored)`` — the rfft bin harmonic ``k`` of ``fund_bin``
    aliases to, folded in-band, and whether the fold *mirrored* it. See
    ``_fold_raw_bin_mirrored``, which this specializes to the raw
    index ``k * fund_bin``.
    """
    return _fold_raw_bin_mirrored(k * fund_bin, n_samples)


def _fold_bin(k: int, fund_bin: int, n_samples: int) -> int:
    """rfft bin harmonic ``k`` of ``fund_bin`` aliases to, folded in-band."""
    return _fold_bin_mirrored(k, fund_bin, n_samples)[0]


def _harmonic_bin_map(fund_bin: int, n_samples: int,
                      n_harmonics: int, half: int = 0) -> dict:
    """``{k: bin}`` for harmonics ``2..n_harmonics``, folded in-band.

    A harmonic that folds onto DC (bin 0 — its energy was removed by the
    mean subtraction, unmeasurable; with ``half`` DC's band ``0..half``) or onto the fundamental bin (its
    power is inseparable from the signal's) is omitted: neither can be
    attributed to that harmonic, so it is skipped rather than reported
    (or silently summed into the wrong bucket). A bin that a *lower*
    harmonic already claims still appears here — :func:`harmonic_levels_db`
    reports its (shared, correct) level for every k that lands there —
    but callers that SUM power over harmonics must dedupe by bin (see
    :func:`_harmonic_power_bins`) so two harmonics aliased onto the same
    bin are counted once, not twice. ``half`` widens the
    fundamental to its band ``fund_bin ± half``: a harmonic centred inside
    it is just as inseparable. ``half=0`` is the single-bin rule above.
    """
    out = {}
    for k in range(2, n_harmonics + 1):
        b = _fold_bin(k, fund_bin, n_samples)
        if b <= half or abs(b - fund_bin) <= half:
            continue
        out[k] = b
    return out


def _harmonic_power_bins(fund_bin: int, n_samples: int,
                         n_harmonics: int, half: int = 0) -> set:
    """De-duplicated, collision-filtered bin set for power summation
    (:func:`thd`, :func:`thd_percent`, :func:`snr`) — the single fold
    shared by every harmonic-consuming metric.

    With ``half > 0`` each harmonic owns its band
    ``b ± half``, clipped to above DC's band ``0..half``, less any bin of
    the fundamental's band; overlapping bands are counted once."""
    centres = _harmonic_bin_map(fund_bin, n_samples, n_harmonics,
                                half).values()
    if half == 0:
        return set(centres)
    last = n_samples // 2
    out = set()
    for b in centres:
        out.update(c for c in range(max(half + 1, b - half),
                                    min(last, b + half) + 1)
                   if abs(c - fund_bin) > half)
    return out


def _residual_bins(P: np.ndarray, fund_bin: int,
                   exclude: Optional[set] = None,
                   half: int = 0) -> np.ndarray:
    """Mask non-DC, non-fundamental bins, optionally excluding harmonics.

    Counterpart to _harmonic_power_bins: select the residual before
    summing, so tiny powers are never rounded away beside the signal.
    Shared with noise_floor_dbfs, which averages the same residual.
    DC owns the band ``0..half`` like every tone.
    """
    mask = np.ones(len(P), dtype=bool)
    mask[:half + 1] = False
    mask[max(1, fund_bin - half):fund_bin + half + 1] = False
    if exclude:
        mask[list(exclude)] = False
    return mask


def fft(sig: Signal, *, tclk: float, tstart: float = 0.0,
        n_samples: Optional[int] = None,
        window: Optional[str] = 'rectangular',
        scaling: Optional[str] = None) -> Signal:
    """Spectrum of a transient Signal, returned as a Signal.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.
    scaling : str or None, default None
        None returns raw squared DFT magnitude. 'amplitude' returns
        one-sided peak amplitude in input units; 'power' returns
        mean-square power in squared input units. Both correct window
        coherent gain, leaving DC and even-length Nyquist unpaired.

    Returns
    -------
    Signal
        New spectrum with x in Hz and y of shape (n_samples // 2 + 1,).
        Raw power has squared input units but ylabel="power"; amplitude and
        normalized power use the input label and its square respectively.
        A constant trace returns zeros. NaN samples propagate NaN bins;
        overflow can leave nonfinite bins.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Unknown window.
        Unknown scaling.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite computed spectrum, including consumed NaNs or overflow.
        Unused samples are not inspected.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    The shared sampler accepts n_samples=4.5 and requests five sample times.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals.base import Signal
    >>> from silk.metrics import fft
    >>> t = np.arange(64, dtype=float)
    >>> sig = Signal('tone', 'voltage', 'time', 'V', t,
    ...              np.cos(2 * np.pi * 8 * t / 64))
    >>> spectrum = fft(sig, tclk=1.0, n_samples=64, scaling='amplitude')
    >>> float(spectrum.x[np.argmax(spectrum.y)])
    0.125
    >>> bool(np.isclose(spectrum.y.max(), 1.0))
    True
    """
    if scaling not in (None, 'power', 'amplitude'):
        raise ValueError(
            f"fft: scaling must be None|'power'|'amplitude', got {scaling!r}"
        )
    samples = _sample_at_clock(sig, metric='fft', tclk=tclk, tstart=tstart,
                               n_samples=n_samples)
    n = len(samples)
    w = _make_window(window, n, metric='fft')
    X, _, _ = _rfft_spectrum(samples, window, metric='fft')
    if scaling is None:
        P = np.abs(X) ** 2
    else:
        a = np.abs(X) / float(np.sum(w))
        paired = np.ones(len(a), dtype=bool)     # conjugate-paired bins
        paired[0] = False
        if n % 2 == 0:
            paired[-1] = False
        if scaling == 'amplitude':
            P = np.where(paired, 2.0 * a, a)
        else:
            P = np.where(paired, 2.0 * a ** 2, a ** 2)
    if not np.all(np.isfinite(P)):
        _warn_spectrum('fft', sig, samples, tclk, tstart, P)
    freqs = np.fft.rfftfreq(n, d=tclk)
    if scaling == 'amplitude':
        return Signal(f'fft({sig.name})', 'amplitude', 'frequency', sig.ylabel,
                      freqs, P)
    if scaling == 'power':
        return Signal(f'fft({sig.name})', 'power', 'frequency',
                      f'{sig.ylabel}^2', freqs, P)
    return Signal(
        f'fft({sig.name})', 'power', 'frequency', 'power', freqs, P,
    )


@metric(domain='tran', unit='dB',
        param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz',
                     'full_scale': '[y]'})
def sndr(sig: Transient, *, tclk: float, tstart: float = 0.0,
         n_samples: Optional[int] = None,
         fundamental_hz: Optional[float] = None,
         guard_bins: int = 0,
         window: Optional[SpectrumWindow] = 'rectangular',
         full_scale: Optional[float] = None) -> float:
    """Signal-to-noise-and-distortion ratio (dB).

    ``fundamental_hz``, ``guard_bins``, ``window`` and ``full_scale``
    are the shared spectrum options (module docstring, *Spectrum
    options*); with ``full_scale`` the result is referred to full scale
    (dBFS) instead of the measured fundamental.

    ``10·log10(P_fund / P_(noise+distortion))``, summing non-DC,
    non-fundamental bins directly. Even an exact tone has a finite ceiling:
    each residual bin carries the double-precision power floor ``eps``.
    With ``m`` residual bins the ceiling is ``10·log10(P_fund/(m·eps))``
    dB: about 172.69 dB (28.39 ENOB) for a 0.4-peak coherent sine at
    256 samples, or 186.65 dB for a 1-peak sine at 1024 samples.
    These are numerical limits, dependent on amplitude and sample count,
    not measured converter performance. An unrepresentable fundamental
    power (NaN or FFT power overflow) returns ``nan``.

    Estimator: with a window or ``guard_bins`` the residual excludes the
    tone and DC bands; its sum is multiplied by ``M / (M − excluded)``
    (``M`` = the default selection's residual count) so the noise is
    estimated over the same ``M`` bins and the ratio is not biased high. An empty residual returns ``nan`` with a
    ``no_residual_bins`` diagnostic, or raises ``ValueError`` when
    ``guard_bins`` emptied it.

    Resolution: the fundamental is the strongest non-DC bin (one bin,
    ``1/(n_samples·tclk)``); with a random noise floor the result is a
    mean over ``n_samples/2`` bins, ≈ 4.3/√(n_samples/2) dB statistical
    resolution (0.54 dB at 128, 0.19 dB at 1024 samples).

    No fundamental (dead / DC / bit-exact-constant input — a stuck
    comparator, a flat-lined ADC code, an aborted-sim tail): the power
    spectrum carries an eps floor (see ``_rfft_spectrum``), so the
    elected "fundamental" and every other bin are equal and the result
    is the floor value ``−10·log10(n_samples/2 − 1)`` dB — **−21 dB at
    256 samples, −27 dB at 1024, −30 dB at 2048** (ENOB ≈ −3.8 / −4.8 /
    −5.3 bits). With the default reference this is an N-dependent numerical
    floor. No exception is raised for a constant input with a nonempty
    residual selection. A dead analog
    node that carries real noise does not hit the floor — it reports its
    (also failing) noise-over-noise ratio, ≈ −13 dB.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.
    full_scale : float or None, optional
        Finite positive peak amplitude of a full-scale sine, in input y units.
        None uses measured fundamental power as the ratio reference.

    Returns
    -------
    float
        SNDR in dB, or dBFS with full_scale. NaN fundamental power or no
        residual bins returns nan. Without full_scale, a constant record
        returns the finite numerical floor described above, without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, full_scale, guard_bins.
        Guard bins that eliminate otherwise available residuals also raise.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite fundamental power or an empty residual selection.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    With explicit full_scale, a constant trace uses the fixed reference
    and eps-floored residuals, without MetricWarning. For example, 64 zero
    samples with full_scale=1 and a rectangular window give the numerical
    ceiling sndr=174.735 dBFS and its corresponding enob=28.733 bits.
    """
    t = _tone_spectrum(sig, metric='sndr', tclk=tclk, tstart=tstart,
                       n_samples=n_samples, fundamental_hz=fundamental_hz,
                       guard_bins=guard_bins, window=window,
                       full_scale=full_scale)
    P, fund_bin = t.P, t.fund_bin
    sig_power = t.fund_power
    if not np.isfinite(sig_power):
        _warn_spectrum('sndr', sig, t.samples, tclk, tstart, P)
        return float('nan')
    mask, scale = t.residual('sndr')
    if not np.any(mask):
        _warn_empty_bins('sndr', sig, t.samples, tclk, tstart, fund_bin, 0,
                         residual=True)
        return float('nan')
    noise_dist = float(np.sum(P[mask])) * scale
    # Log powers separately: even finite powers can overflow on division.
    return float(10.0 * (np.log10(t.reference_power())
                         - np.log10(noise_dist)))


@metric(domain='tran', unit='dB',
        param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz',
                     'full_scale': '[y]'})
def sinad(sig: Transient, *, tclk: float, tstart: float = 0.0,
          n_samples: Optional[int] = None,
          fundamental_hz: Optional[float] = None,
          guard_bins: int = 0,
          window: Optional[SpectrumWindow] = 'rectangular',
          full_scale: Optional[float] = None) -> float:
    """SINAD (dB). Identical to SNDR for coherently-sampled single-tone
    tests, with the same spectrum options.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.
    full_scale : float or None, optional
        Finite positive peak amplitude of a full-scale sine, in input y units.
        None uses measured fundamental power as the ratio reference.

    Returns
    -------
    float
        SNDR in dB (dBFS with full_scale), exactly as sndr: nan for
        unavailable power or no residual bins, and the finite negative
        numerical-floor value for constant data without full_scale.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, full_scale, guard_bins.
        Guard bins that eliminate otherwise available residuals also raise.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        The same unavailable spectrum or residual selection as sndr;
        the diagnostic identifies sndr.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    With explicit full_scale, a constant trace uses the fixed reference
    and eps-floored residuals, without MetricWarning. For example, 64 zero
    samples with full_scale=1 and a rectangular window give the numerical
    ceiling sndr=174.735 dBFS and its corresponding enob=28.733 bits.
    """
    _check_nonempty(sig, 'sinad')
    _check_transient(sig, 'sinad')
    try:
        return sndr(sig, tclk=tclk, tstart=tstart, n_samples=n_samples,
                    fundamental_hz=fundamental_hz, guard_bins=guard_bins,
                    window=window, full_scale=full_scale)
    except ValueError as exc:
        raise ValueError(f'sinad: {exc}') from exc


@metric(domain='tran', outputs={'sfdr_db': 'dB', 'fundamental_hz': 'Hz',
                               'spur_hz': 'Hz', 'bin_hz': 'Hz'},
        param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz'})
def sfdr(sig: Transient, *, tclk: float, tstart: float = 0.0,
         n_samples: Optional[int] = None,
         fundamental_hz: Optional[float] = None,
         guard_bins: int = 0,
         window: Optional[SpectrumWindow] = 'rectangular') -> SFDR:
    """Spurious-free dynamic range.

    ``fundamental_hz``, ``guard_bins`` and ``window`` are the shared
    spectrum options (module docstring). The spur is the strongest bin
    outside DC's band and the fundamental's; its power is summed over its
    own band, so a windowed spur is measured as whole as the fundamental.
    A ``guard_bins`` that leaves no spur candidate raises ``ValueError``;
    a record too short for the window's bands returns ``nan`` fields.

    The largest non-DC, non-fundamental bin carries at least the
    double-precision power floor ``eps``. An exact tone therefore reads
    at most ``10·log10(P_fund/eps)`` dB, not infinity (about 193.73 dB
    for a 0.4-peak coherent sine at 256 samples). This numerical ceiling
    depends on amplitude and sample count. A dead signal reads 0 dB.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    SFDR
        Frozen result containing sfdr_db in dB and fundamental_hz, spur_hz,
        bin_hz in Hz. Nonfinite fundamental power or no spur bins makes the
        first three fields nan; bin_hz remains 1 / (n_samples * tclk).
        A constant rectangular-window record gives a 0 dB floor value
        and elected bin-centre frequencies, without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, guard_bins.
        Guard bins that eliminate otherwise available residuals also raise.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite fundamental power or no spur candidates.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    t = _tone_spectrum(sig, metric='sfdr', tclk=tclk, tstart=tstart,
                       n_samples=n_samples, fundamental_hz=fundamental_hz,
                       guard_bins=guard_bins, window=window)
    samples, P, fund_bin = t.samples, t.P, t.fund_bin
    fund_power = t.fund_power
    if not np.isfinite(fund_power):
        _warn_spectrum('sfdr', sig, samples, tclk, tstart, P)
        nan = float('nan')
        return SFDR(nan, nan, nan, 1.0 / (len(samples) * tclk))
    # Same exclusions as _residual_bins; keep original indices for argmax.
    pool, _ = t.residual('sfdr')
    if not np.any(pool):
        _warn_empty_bins('sfdr', sig, samples, tclk, tstart, fund_bin, 0,
                         residual=True)
        nan = float('nan')
        return SFDR(nan, nan, nan, 1.0 / (len(samples) * tclk))
    P_no_fund = np.where(pool, P, 0.0)
    spur_bin = int(np.argmax(P_no_fund))
    spur_power = (P_no_fund[spur_bin] if t.half == 0
                  else float(np.sum(P_no_fund[t.band(spur_bin)])))
    freqs = np.fft.rfftfreq(len(samples), d=tclk)
    df = 1.0 / (len(samples) * tclk)
    sfdr_db = float(10.0 * (np.log10(fund_power) - np.log10(spur_power)))
    return SFDR(sfdr_db, float(freqs[fund_bin]), float(freqs[spur_bin]), df)


@metric(domain='tran', unit='dB', param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz'})
def thd(sig: Transient, *, tclk: float, tstart: float = 0.0,
        n_samples: Optional[int] = None,
        n_harmonics: int = 5,
        fundamental_hz: Optional[float] = None,
        guard_bins: int = 0,
        window: Optional[SpectrumWindow] = 'rectangular') -> float:
    """Total harmonic distortion in dB. Sum runs over harmonics
    2..n_harmonics.

    ``fundamental_hz``, ``guard_bins`` and ``window`` are the shared
    spectrum options (module docstring): each harmonic's power is the sum
    over its band, the fundamental's over its own.

    Harmonics above Nyquist are folded back in band, matching
    the physical aliasing already present in the sampled data — see the
    module docstring and ``_harmonic_bin_map``. A harmonic that
    folds onto DC or onto the fundamental bin is unattributable and
    excluded; two harmonics that fold onto the same bin are counted
    once (their power is inseparable in that bin), never summed twice —
    the same de-duplicated bin set :func:`snr` excludes from its noise
    sum, so the two metrics stay mutually consistent.

    If that selection is empty, returns ``nan``: a valid waveform/request
    has no independently measurable requested harmonics. Coherent records
    at ``f0 = fs/2`` or ``fs/3`` (``fs = 1/tclk``) are examples. This is
    unavailable data under the metric failure convention, so sweeps continue; malformed
    requests (e.g. ``n_harmonics < 2`` or an invalid sample window) still
    raise ``ValueError``. A nonempty selection retains its measured power
    and numerical floor, including distinct aliases and Nyquist harmonics.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    n_harmonics : int, default 5
        Highest harmonic order included, at least 2; includes orders 2
        through n_harmonics, folding aliases into the sampled spectrum.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    float
        10 * log10(harmonic power / fundamental power), dB. Empty harmonic
        selection or undefined power ratios give nan. A constant record
        retains the eps-floor value (10 * log10 of the selected bin count
        with the rectangular window), without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, guard_bins.
        Guard bins placing the automatically elected fundamental in the
        DC band also raise.
        n_harmonics below 2.

    TypeError
        Noninteger n_harmonics can reach integer-only bin iteration; no
        explicit integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Empty harmonic selection or an undefined consumed power ratio.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    if n_harmonics < 2:
        raise ValueError(
            f'thd: n_harmonics must be >= 2, got {n_harmonics}'
        )
    t = _tone_spectrum(sig, metric='thd', tclk=tclk, tstart=tstart,
                       n_samples=n_samples, fundamental_hz=fundamental_hz,
                       guard_bins=guard_bins, window=window)
    samples, P, fund_bin = t.samples, t.P, t.fund_bin
    fund_power = t.fund_power
    bins = _harmonic_power_bins(fund_bin, len(samples), n_harmonics, t.half)
    if not bins:
        _warn_empty_bins('thd', sig, samples, tclk, tstart,
                         fund_bin, n_harmonics)
        return float('nan')
    harm_power = float(np.sum(P[list(bins)]))
    if _undefined_power_ratio(harm_power, fund_power):
        _warn_spectrum('thd', sig, samples, tclk, tstart,
                       np.array([harm_power, fund_power]))
    if harm_power <= 0:
        return float('-inf')
    return float(10.0 * np.log10(harm_power / fund_power))


@metric(domain='tran', unit='%', param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz'})
def thd_percent(sig: Transient, *, tclk: float, tstart: float = 0.0,
                n_samples: Optional[int] = None,
                n_harmonics: int = 5,
                fundamental_hz: Optional[float] = None,
                guard_bins: int = 0,
                window: Optional[SpectrumWindow] = 'rectangular') -> float:
    """THD as a percentage of the fundamental amplitude, with
    :func:`thd`'s spectrum options.

    Same folded, de-duplicated, DC/fundamental-excluding harmonic set as
    :func:`thd`. Returns ``nan`` when that selection is empty: a valid
    waveform/request has no independently measurable requested harmonics,
    for example a coherent fundamental at ``fs/2`` or ``fs/3`` with
    ``fs = 1/tclk``. This is unavailable data under the metric failure convention, not an
    exception; malformed requests still raise ``ValueError``. Nonempty
    selections (including distinct aliases and Nyquist harmonics) retain
    their measured power and numerical floor.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    n_harmonics : int, default 5
        Highest harmonic order included, at least 2; includes orders 2
        through n_harmonics, folding aliases into the sampled spectrum.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    float
        100 * sqrt(harmonic power / fundamental power), percent. Empty
        selection or undefined ratios give nan. Constant data retains an
        eps-floor value (100 * sqrt(selected bin count) for the
        rectangular window), without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, guard_bins.
        Guard bins placing the automatically elected fundamental in the
        DC band also raise.
        n_harmonics below 2.

    TypeError
        Noninteger n_harmonics can reach integer-only bin iteration; no
        explicit integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Empty harmonic selection or an undefined consumed power ratio.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    if n_harmonics < 2:
        raise ValueError(
            f'thd_percent: n_harmonics must be >= 2, got {n_harmonics}'
        )
    t = _tone_spectrum(sig, metric='thd_percent', tclk=tclk, tstart=tstart,
                       n_samples=n_samples, fundamental_hz=fundamental_hz,
                       guard_bins=guard_bins, window=window)
    samples, P, fund_bin = t.samples, t.P, t.fund_bin
    fund_power = t.fund_power
    bins = _harmonic_power_bins(fund_bin, len(samples), n_harmonics, t.half)
    if not bins:
        _warn_empty_bins('thd_percent', sig, samples, tclk, tstart,
                         fund_bin, n_harmonics)
        return float('nan')
    harm_power = float(np.sum(P[list(bins)]))
    if _undefined_power_ratio(harm_power, fund_power):
        _warn_spectrum('thd_percent', sig, samples, tclk, tstart,
                       np.array([harm_power, fund_power]))
    if harm_power <= 0:
        return 0.0
    return float(100.0 * np.sqrt(harm_power / fund_power))


@metric(domain='tran', unit='dBFS',
        param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz',
                     'full_scale': '[y]'})
def noise_floor_dbfs(sig: Transient, *, tclk: float, tstart: float = 0.0,
                     n_samples: Optional[int] = None,
                     fundamental_hz: Optional[float] = None,
                     guard_bins: int = 0,
                     window: Optional[SpectrumWindow] = 'rectangular',
                     full_scale: Optional[float] = None) -> float:
    """Average non-fundamental, non-DC bin power in dBFS.

    Full scale is the measured fundamental's power by default, or, with
    ``full_scale`` (the peak amplitude of a full-scale sine, in the
    signal's unit), the power such a sine would have. ``fundamental_hz``,
    ``guard_bins`` and ``window`` are the shared spectrum options (module
    docstring); the fundamental's and DC's bands are left out of the
    average (a mean needs no rescaling). An empty residual returns
    ``nan`` with a ``no_residual_bins`` diagnostic, or raises
    ``ValueError`` when ``guard_bins`` emptied it.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.
    full_scale : float or None, optional
        Finite positive peak amplitude of a full-scale sine, in input y units.
        None uses measured fundamental power as the ratio reference.

    Returns
    -------
    float
        Mean residual-bin power relative to reference power, dBFS. Empty
        residuals or undefined ratios give nan; constant data without
        full_scale gives a 0 dB floor value, without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, full_scale, guard_bins.
        Guard bins that eliminate otherwise available residuals also raise.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Empty residual selection or an undefined consumed power ratio.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    With explicit full_scale, a constant trace uses the fixed reference
    and eps-floored residuals, without MetricWarning. For example, 64 zero
    samples with full_scale=1 and a rectangular window give the numerical
    ceiling sndr=174.735 dBFS and its corresponding enob=28.733 bits.
    """
    t = _tone_spectrum(sig, metric='noise_floor_dbfs', tclk=tclk,
                       tstart=tstart, n_samples=n_samples,
                       fundamental_hz=fundamental_hz, guard_bins=guard_bins,
                       window=window, full_scale=full_scale)
    samples, P, fund_bin = t.samples, t.P, t.fund_bin
    ref_power = t.reference_power()
    mask, _ = t.residual('noise_floor_dbfs')
    if not np.any(mask):
        _warn_empty_bins('noise_floor_dbfs', sig, samples, tclk, tstart,
                         fund_bin, 0, residual=True)
        return float('nan')
    avg_noise = float(np.mean(P[mask]))
    if avg_noise <= 0:
        return float('-inf')
    if _undefined_power_ratio(avg_noise, ref_power):
        _warn_spectrum('noise_floor_dbfs', sig, samples, tclk, tstart,
                       np.array([avg_noise, ref_power]))
    return float(10.0 * np.log10(avg_noise / ref_power))


def harmonic_levels_db(sig: Signal, *, tclk: float, tstart: float = 0.0,
                       n_samples: Optional[int] = None,
                       n_harmonics: int = 10,
                       fundamental_hz: Optional[float] = None) -> np.ndarray:
    """Per-harmonic level relative to the fundamental, dB.

    ``fundamental_hz`` names the fundamental instead of electing the
    strongest bin (module docstring, *Spectrum options*).

    Levels compare sampled-record power (conjugate pairs weighted by 2,
    even-length Nyquist by 1), not raw complex DFT amplitudes.
    Returns an array of length ``n_harmonics − 1`` covering harmonics
    2..n_harmonics. Harmonics above Nyquist are folded back in band using the same map as :func:`thd`; a harmonic that folds
    onto DC or onto the fundamental bin — unattributable, see
    ``_harmonic_bin_map`` — is reported as ``nan`` with a diagnostic.
    Two harmonics that fold onto the same bin both report that bin's
    (shared) level —
    :func:`thd`/:func:`snr` dedupe the bin when summing power, but this
    per-harmonic view intentionally does not hide the collision.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    n_harmonics : int, default 10
        Highest harmonic order included, at least 2; includes orders 2
        through n_harmonics, folding aliases into the sampled spectrum.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.

    Returns
    -------
    numpy.ndarray
        New float array of shape (n_harmonics - 1,), in dBc. Entry k - 2
        corresponds to harmonic k. Unattributable aliases and undefined
        power ratios are nan. Constant data gives the 0 dBc floor value
        for attributable bins, without a warning for the constant data.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz.
        n_harmonics below 2.

    TypeError
        Noninteger n_harmonics can reach integer-only bin iteration; no
        explicit integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Each unattributable alias, or undefined consumed power ratios.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    if n_harmonics < 2:
        raise ValueError(
            f'harmonic_levels_db: n_harmonics must be >= 2, got {n_harmonics}'
        )
    t = _tone_spectrum(sig, metric='harmonic_levels_db', tclk=tclk,
                       tstart=tstart, n_samples=n_samples,
                       fundamental_hz=fundamental_hz)
    samples, P, fund_bin = t.samples, t.P, t.fund_bin
    bin_map = _harmonic_bin_map(fund_bin, len(samples), n_harmonics)
    out = np.full(n_harmonics - 1, -np.inf, dtype=float)
    for k in range(2, n_harmonics + 1):
        if k not in bin_map:
            b = _fold_bin(k, fund_bin, len(samples))
            emit(
                metric='harmonic_levels_db', code='unattributable_harmonic',
                message='Harmonic level unavailable because its folded bin '
                        'coincides with DC or the fundamental.',
                details={'harmonic': k, 'folded_bin': b,
                         'fundamental_bin': fund_bin,
                         'sample_count': len(samples),
                         'bin_hz': 1.0 / (len(samples) * tclk)},
                suggestion='Try a sampling rate or coherent tone that separates '
                           'the requested harmonic from DC and the fundamental.',
            )
            out[k - 2] = np.nan
    powers = P[list(bin_map.values())]
    if bin_map and _undefined_power_ratio(powers, P[fund_bin]):
        _warn_spectrum('harmonic_levels_db', sig, samples, tclk, tstart,
                       np.append(powers, P[fund_bin]))
    for k, b in bin_map.items():
        if P[b] <= 0:
            continue
        out[k - 2] = float(10.0 * np.log10(P[b] / P[fund_bin]))
    return out


def _make_window(name: Optional[str], n: int, *, metric: str) -> np.ndarray:
    """Return a periodic window of length n. Supported: 'hann', 'hamming',
    'blackman', 'rectangular' / None.
    """
    if name is None or name == 'rectangular':
        return np.ones(n)
    fn = {'hann': np.hanning, 'hamming': np.hamming,
          'blackman': np.blackman}.get(name)
    if fn is not None:
        # Periodic (DFT-even): the symmetric n+1 window minus its last
        # sample, so the window repeats cleanly over the record and a
        # coherent tone does not leak.
        return np.ones(n) if n <= 1 else fn(n + 1)[:-1]
    raise ValueError(
        f"{metric}: window must be 'hann'|'hamming'|'blackman'|'rectangular'|None, "
        f"got {name!r} for {n} samples"
    )


def psd(sig: Signal, *, tclk: float, tstart: float = 0.0,
        n_samples: Optional[int] = None,
        window: Optional[str] = 'hann') -> Signal:
    """Single-sided power spectral density, returned as a Signal.

    Standard normalisation: ``PSD[k] = 2·|X[k]|² / (fs · Σw²)`` for
    interior bins, ``|X[k]|² / (fs · Σw²)`` for the DC and Nyquist bins.
    Units: V²/Hz when input is volts.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    window : str or None, default 'hann'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    Signal
        New frequency Signal with arrays of shape (n_samples // 2 + 1,).
        x is Hz and y is squared input units per Hz; ylabel is always
        "V^2/Hz", so that label assumes voltage input. Constant data gives
        zeros; NaN samples propagate nan and overflow can be nonfinite.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Unknown window.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite computed spectrum, including consumed NaNs or overflow.
        Unused samples are not inspected.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    """
    samples = _sample_at_clock(sig, metric='psd', tclk=tclk, tstart=tstart,
                               n_samples=n_samples)
    selected_samples = samples
    samples = samples - np.mean(samples)
    n = len(samples)
    w = _make_window(window, n, metric='psd')
    fs = 1.0 / tclk
    X = np.fft.rfft(samples * w)
    norm = fs * float(np.sum(w * w))
    P = 2.0 * np.abs(X) ** 2 / norm
    P[0] /= 2.0
    if n % 2 == 0:
        P[-1] /= 2.0
    if not np.all(np.isfinite(P)):
        _warn_spectrum('psd', sig, selected_samples, tclk, tstart, P)
    freqs = np.fft.rfftfreq(n, d=tclk)
    return Signal(
        f'psd({sig.name})', 'psd', 'frequency', 'V^2/Hz', freqs, P,
    )


@metric(domain='tran', unit='[y]^2',
        param_units={'f1_hz': 'Hz', 'f2_hz': 'Hz', 'tclk': 's',
                     'tstart': 's'})
def spectral_power(sig: Transient, f1_hz: float, f2_hz: float, *,
                   tclk: float, tstart: float = 0.0,
                   n_samples: Optional[int] = None,
                   window: Optional[
                       Literal['hann', 'hamming', 'blackman', 'rectangular']
                   ] = 'hann') -> float:
    """Total power in the band ``[f1_hz, f2_hz]``, V² (assuming volts).

    Trapezoidal integration of the single-sided PSD over the band.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    f1_hz : float
        Inclusive band boundary, Hz. f2_hz must exceed f1_hz, and the
        selected band must contain at least two FFT bin centres.
    f2_hz : float
        Inclusive band boundary, Hz. f2_hz must exceed f1_hz, and the
        selected band must contain at least two FFT bin centres.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    window : str or None, default 'hann'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    float
        Trapezoidal integral over selected PSD bin centres, in squared
        input units (V squared for volts). Constant data gives zero power;
        NaNs in the computed PSD yield nan, and overflow may yield inf.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Unknown window.
        Reversed band or fewer than two selected FFT bins.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite PSD bins, emitted by psd before band-bin validation.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    """
    _check_nonempty(sig, 'spectral_power')
    if f2_hz <= f1_hz:
        raise ValueError(
            f'spectral_power: require f2_hz > f1_hz, got {f1_hz}, {f2_hz} Hz'
        )
    try:
        spec = psd(sig, tclk=tclk, tstart=tstart, n_samples=n_samples,
                   window=window)
    except ValueError as exc:
        raise ValueError(f'spectral_power: {exc}') from exc
    mask = (spec.x >= f1_hz) & (spec.x <= f2_hz)
    n = int(np.sum(mask))
    if n < 2:
        lo, hi = float(np.min(spec.x)), float(np.max(spec.x))
        raise ValueError(
            f'spectral_power: band [{f1_hz:g}, {f2_hz:g}] Hz contains {n} '
            f'of {len(spec.x)} FFT bins (spectrum spans {lo:g} Hz .. '
            f'{hi:g} Hz)'
        )
    return float(trapezoid(spec.y[mask], spec.x[mask]))


def harmonic(sig: Signal, k: int, *, tclk: float, tstart: float = 0.0,
             n_samples: Optional[int] = None,
             fundamental_hz: Optional[float] = None) -> complex:
    """Complex value of the k-th harmonic FFT bin.

    ``fundamental_hz`` names the fundamental instead of electing the
    strongest bin (module docstring, *Spectrum options*).

    ``k=1`` is the fundamental, ``k=2`` is the second harmonic, etc.
    Returns the *complex* DFT bin so both magnitude and phase are
    available — distinguishes this from :func:`harmonic_levels_db`
    which returns dB only.

    A harmonic above Nyquist is read at the bin sampling folded it to
    (same map as :func:`harmonic_levels_db` — see
    ``_harmonic_bin_map``). Away from the numerical floor,
    ``20·log10(|harmonic(k)| / |harmonic(1)|)`` equals
    ``harmonic_levels_db(...)[k − 2]`` when both bins have the same power
    weight. A paired bin has weight 2 and even-length Nyquist has weight 1;
    otherwise add ``10·log10(w_k/w_1)`` to compare with that power level.
    The value returned is the two-sided-DFT coefficient at the
    harmonic's nominal index ``(k · fund_bin) mod N``: when the fold
    mirrored the bin (``b_raw > N/2``) the rfft holds the *conjugate* of
    the harmonic's phasor, which is conjugated back here — the same
    convention as :func:`mix_product`, so ``harmonic(sig, k, ...) ==
    mix_product(sig, k, 0, f1_hz=f0, ...)``.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    k : int
        Harmonic order, at least 1; 1 selects the fundamental.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.

    Returns
    -------
    complex
        Raw, unnormalized two-sided DFT coefficient in input amplitude
        units, proportional to sample count. An unattributable folded alias
        returns complex nan. Nonfinite selected coefficients propagate; a
        constant trace returns 0j at attributable bins without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz.
        Harmonic count/order below its minimum; harmonic_level_dbc also
        rejects noninteger orders.

    TypeError
        Noninteger harmonic orders/counts can reach integer-only iteration
        or allocation; no explicit integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Unattributable aliases or nonfinite selected DFT coefficients.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    """
    if k < 1:
        raise ValueError(f'harmonic: k must be >= 1, got {k}')
    t = _tone_spectrum(sig, metric='harmonic', tclk=tclk, tstart=tstart,
                       n_samples=n_samples, fundamental_hz=fundamental_hz)
    samples, X, fund_bin = t.samples, t.X, t.fund_bin
    if k == 1:
        if not np.isfinite(X[fund_bin]):
            _warn_spectrum('harmonic', sig, samples, tclk, tstart,
                           X[fund_bin], quantity='FFT coefficient')
        return complex(X[fund_bin])
    n = len(samples)
    b = _harmonic_bin_map(fund_bin, n, k).get(k)
    if b is None:
        _warn_harmonic_location('harmonic', sig, samples, tclk, tstart,
                               fund_bin, k)
        return complex('nan')
    if not np.isfinite(X[b]):
        _warn_spectrum('harmonic', sig, samples, tclk, tstart,
                       X[b], quantity='FFT coefficient')
    value = complex(X[b])
    if _fold_bin_mirrored(k, fund_bin, n)[1]:
        value = value.conjugate()
    return value


@metric(domain='tran', unit='dB',
        param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz',
                     'full_scale': '[y]'})
def snr(sig: Transient, *, tclk: float, tstart: float = 0.0,
        n_samples: Optional[int] = None,
        n_harmonics: int = 10,
        fundamental_hz: Optional[float] = None,
        guard_bins: int = 0,
        window: Optional[SpectrumWindow] = 'rectangular',
        full_scale: Optional[float] = None) -> float:
    """Signal-to-noise ratio in dB, *excluding* harmonic distortion.

    ``fundamental_hz``, ``guard_bins``, ``window`` and ``full_scale``
    are the shared spectrum options (module docstring): the bands of the
    fundamental and of each harmonic are left out of the noise sum, and
    with ``full_scale`` the result is full-scale SNR (dBFS).

    Differs from :func:`sndr` which lumps harmonics into the
    "noise+distortion" denominator. Here, the first ``n_harmonics``
    harmonic bins are removed before noise is summed:

        SNR = 10·log10(P_fund / P_noise_only)

    where ``P_noise_only`` is the direct sum of the non-DC,
    non-fundamental, non-harmonic bins. Each bin carries the same
    double-precision power floor as :func:`sndr`. If harmonic exclusion
    leaves no noise bins (e.g. four samples with the fundamental at bin 1),
    returns ``nan``: noise is unavailable on this valid data, not zero or
    one invented ``eps``. Malformed requests still raise ``ValueError``.
    With a nonempty selection the finite ceiling depends on amplitude,
    sample count and excluded bins; it is a numerical limit, not measured
    noise performance. With a window or ``guard_bins`` the noise sum is
    multiplied by ``M / (M − excluded)`` as in :func:`sndr` (``M`` = the
    residual count with single-bin bands, harmonic centres excluded), and
    a ``guard_bins`` that empties the residual raises ``ValueError``.
    An unrepresentable fundamental power returns ``nan`` as in :func:`sndr`.

    Datasheet "dynamic range" for a precision/high-speed converter is
    conventionally full-scale SNR — this metric, not a separate one.
    Audio "dynamic range" (AES17) is a different measurement: a
    −60 dBFS notched-tone capture, not derivable from a full-scale FFT.

    Harmonics above Nyquist are folded back in band using the
    same de-duplicated bin set :func:`thd` sums as distortion, so the
    same energy is never counted as noise AND distortion at once — see
    ``_harmonic_bin_map``.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    n_harmonics : int, default 10
        Highest harmonic order included, at least 2; includes orders 2
        through n_harmonics, folding aliases into the sampled spectrum.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.
    full_scale : float or None, optional
        Finite positive peak amplitude of a full-scale sine, in input y units.
        None uses measured fundamental power as the ratio reference.

    Returns
    -------
    float
        SNR in dB, or dBFS with full_scale. Nonfinite fundamental power or
        no residual noise bins gives nan. Constant data without full_scale
        returns the ratio determined by the eps-floored bins.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, full_scale, guard_bins.
        Guard bins that eliminate otherwise available residuals also raise.
        n_harmonics below 2.

    TypeError
        Noninteger n_harmonics can reach integer-only bin iteration; no
        explicit integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite fundamental power or an empty residual selection.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    With explicit full_scale, a constant trace uses the fixed reference
    and eps-floored residuals, without MetricWarning. For example, 64 zero
    samples with full_scale=1 and a rectangular window give the numerical
    ceiling sndr=174.735 dBFS and its corresponding enob=28.733 bits.
    """
    if n_harmonics < 2:
        raise ValueError(
            f'snr: n_harmonics must be >= 2, got {n_harmonics}'
        )
    t = _tone_spectrum(sig, metric='snr', tclk=tclk, tstart=tstart,
                       n_samples=n_samples, fundamental_hz=fundamental_hz,
                       guard_bins=guard_bins, window=window,
                       full_scale=full_scale)
    samples, P, fund_bin = t.samples, t.P, t.fund_bin
    sig_power = t.fund_power
    if not np.isfinite(sig_power):
        _warn_spectrum('snr', sig, samples, tclk, tstart, P)
        return float('nan')
    residual, scale = t.residual('snr', n_harmonics)
    if not np.any(residual):
        _warn_empty_bins('snr', sig, samples, tclk, tstart,
                         fund_bin, n_harmonics, residual=True)
        return float('nan')
    noise_only = float(np.sum(P[residual])) * scale
    return float(10.0 * (np.log10(t.reference_power())
                         - np.log10(noise_only)))


def harmonic_freqs(sig: Signal, *, tclk: float, tstart: float = 0.0,
                   n_samples: Optional[int] = None,
                   n: int = 10, observed: bool = False,
                   fundamental_hz: Optional[float] = None) -> np.ndarray:
    """Frequencies of the first ``n`` harmonics of the detected
    fundamental, in Hz.

    ``fundamental_hz`` names the fundamental instead of electing the
    strongest bin; it is rounded to the nearest bin like every reported
    frequency (module docstring, *Spectrum options*).

    Returns ``[f_fund, 2·f_fund, …, n·f_fund]`` — the *true* frequency
    of every harmonic, always length ``n`` and index-aligned with
    :func:`harmonic_levels_db` (``freqs[k − 1]`` ↔ ``levels[k − 2]``).
    A harmonic above Nyquist is
    still reported at ``k·f_fund``: that is what it *is*; the
    sampled spectrum merely *observes* it folded in band (see
    ``_harmonic_bin_map``), so ``freqs > 0.5 / tclk`` tells you
    which harmonics aliased. Pass ``observed=True`` for where each
    harmonic sits in the sampled spectrum instead.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    n : int, default 10
        Number of harmonic frequencies, at least 1, including the fundamental.
    observed : bool, default False
        Return true harmonic frequencies when False (possibly above Nyquist),
        or folded bin centres when True; unattributable aliases become nan.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.

    Returns
    -------
    numpy.ndarray
        New float array of shape (n,), frequencies in Hz. Sampled NaNs
        return the special shape-(1,) array [nan]. With observed=True,
        unattributable harmonics are nan. Constant or overflowing data can
        return the frequencies of the elected bins without warning.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz.
        Harmonic count/order below its minimum; harmonic_level_dbc also
        rejects noninteger orders.

    TypeError
        Noninteger harmonic orders/counts can reach integer-only iteration
        or allocation; no explicit integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Sampled NaNs, or each unattributable harmonic with observed=True.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; the returned array is newly allocated.
    With observed=False, n=2.5 produces three entries; with observed=True
    it raises TypeError.
    """
    if n < 1:
        raise ValueError(f'harmonic_freqs: n must be >= 1, got {n}')
    t = _tone_spectrum(sig, metric='harmonic_freqs', tclk=tclk,
                       tstart=tstart, n_samples=n_samples,
                       fundamental_hz=fundamental_hz)
    samples, fund_bin = t.samples, t.fund_bin
    if _any_nan(samples):
        _warn_spectrum('harmonic_freqs', sig, samples, tclk, tstart, samples,
                       quantity='Sampled data')
        return np.array([float('nan')])
    fs = 1.0 / tclk
    df = fs / len(samples)
    if not observed:
        return np.arange(1, n + 1, dtype=float) * (fund_bin * df)
    out = np.full(n, np.nan, dtype=float)
    out[0] = fund_bin * df
    bin_map = _harmonic_bin_map(fund_bin, len(samples), n)
    for k in range(2, n + 1):
        if k not in bin_map:
            _warn_harmonic_location('harmonic_freqs', sig, samples, tclk,
                                   tstart, fund_bin, k)
    for k, b in bin_map.items():
        out[k - 1] = b * df
    return out


def _segment_layout(n_total: int, segments: int,
                    overlap: float, *, metric: str) -> tuple[int, int]:
    """Return ``(seg_len, step)`` for splitting ``n_total`` samples into
    ``segments`` overlapping segments with ``overlap`` ratio in [0, 1).
    """
    if segments < 1:
        raise ValueError(
            f'{metric}: segments must be >= 1, got {segments}; '
            f'{n_total} samples available'
        )
    if not 0.0 <= overlap < 1.0:
        raise ValueError(
            f'{metric}: overlap must be in [0, 1), got {overlap}; '
            f'{n_total} samples available for {segments} segments'
        )
    seg_len = int(n_total / (1.0 + (segments - 1) * (1.0 - overlap)))
    if seg_len < 4:
        raise ValueError(
            f'{metric}: not enough samples for {segments} segments with '
            f'overlap={overlap}: derived seg_len={seg_len}, need >= 4; '
            f'{n_total} samples available'
        )
    step = max(1, int(seg_len * (1.0 - overlap)))
    return seg_len, step


def welch_psd(sig: Signal, *, tclk: float, tstart: float = 0.0,
              n_samples: Optional[int] = None,
              segments: int = 8, overlap: float = 0.5,
              window: Optional[str] = 'hann') -> Signal:
    """Welch's segment-averaged single-sided PSD.

    Lower-variance estimate than the single-FFT :func:`psd` for noisy
    or non-coherent data. Splits the sampled trace into ``segments``
    overlapping windows, computes a windowed FFT of each, and averages
    the magnitudes.


    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    segments : int, default 8
        Requested number of overlapping segments. Each derived segment
        must hold at least 4 samples; unused trailing samples are ignored.
    overlap : float, default 0.5
        Fractional overlap in [0, 1); sample offsets are rounded down.
    window : str or None, default 'hann'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    Signal
        New frequency Signal of shape (segment_length // 2 + 1,). x is Hz;
        y is squared input units per Hz, labelled "V^2/Hz" regardless of
        input label. Constant data gives zeros. NaNs in consumed segments
        propagate nan; overflow can be nonfinite.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Unknown window.
        Segment count below 1, overlap outside [0, 1), or derived segments
        shorter than 4 samples.

    TypeError
        Noninteger segments can reach integer-only iteration; no explicit
        integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Nonfinite computed spectrum, including consumed NaNs or overflow.
        Unused samples are not inspected.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    """
    samples = _sample_at_clock(sig, metric='welch_psd',
                               tclk=tclk, tstart=tstart,
                               n_samples=n_samples)
    n = len(samples)
    seg_len, step = _segment_layout(n, segments, overlap, metric='welch_psd')
    w = _make_window(window, seg_len, metric='welch_psd')
    fs = 1.0 / tclk
    norm = fs * float(np.sum(w * w))
    P_sum = np.zeros(seg_len // 2 + 1, dtype=float)
    n_segs_used = 0
    for k in range(segments):
        start = k * step
        end = start + seg_len
        if end > n:
            break
        seg = samples[start:end]
        seg = seg - np.mean(seg)
        X = np.fft.rfft(seg * w)
        P_sum += np.abs(X) ** 2
        n_segs_used += 1
    P_avg = (2.0 * P_sum / n_segs_used) / norm
    P_avg[0] /= 2.0
    if seg_len % 2 == 0:
        P_avg[-1] /= 2.0
    if not np.all(np.isfinite(P_avg)):
        # The segment layout can leave a tail that no FFT consumes.
        used_end = (n_segs_used - 1) * step + seg_len
        _warn_spectrum('welch_psd', sig, samples[:used_end], tclk, tstart,
                       P_avg)
    freqs = np.fft.rfftfreq(seg_len, d=tclk)
    return Signal(
        f'welch_psd({sig.name})', 'psd', 'frequency', 'V^2/Hz',
        freqs, P_avg,
    )


def coherence(a: Signal, b: Signal, *, tclk: float,
              tstart: float = 0.0,
              n_samples: Optional[int] = None,
              segments: int = 8, overlap: float = 0.5,
              window: Optional[str] = 'hann') -> Signal:
    """Magnitude-squared coherence ``γ²(f) = |S_xy|² / (S_xx · S_yy)``.

    Welch-style segment averaging — segments are required because the
    single-segment cross-spectrum collapses to γ² ≡ 1 by construction.

    γ² ≈ 1 means the two signals are linearly related at frequency f
    (one is a deterministic function of the other plus minimal noise);
    γ² ≈ 0 means they are uncorrelated.

    A zero-variance segment pair (both ``a`` and ``b`` constant over every
    averaged segment, so each is mean-removed to all-zero before its FFT)
    has no defined coherence at any frequency: ``S_xx = S_yy = S_xy = 0``,
    so γ² is ``0/0``. This is degenerate DATA, not a malformed request — the two constant segments are not "uncorrelated",
    they carry no variation to correlate — so this returns ``nan``, not
    ``0.0`` as it did before 2026-09-05: the ``np.finfo(float).tiny``
    denominator floor that produced the old ``0.0`` read as a real
    "uncorrelated" measurement, and has been dropped (``S_xy`` is
    ``0`` in exactly the cases ``S_xx · S_yy`` is, by Cauchy-Schwarz, so
    the floor never guarded a nonzero numerator).


    Parameters
    ----------
    a : Signal
        First real transient; one-dimensional time (seconds) and value
        arrays, not mutated.
    b : Signal
        Second real transient, not mutated. Its time grid must agree with
        a within 1e-3 of the smallest spacing across both grids.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    segments : int, default 8
        Requested number of overlapping segments. Each derived segment
        must hold at least 4 samples; unused trailing samples are ignored.
    overlap : float, default 0.5
        Fractional overlap in [0, 1); sample offsets are rounded down.
    window : str or None, default 'hann'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    Signal
        New frequency Signal of shape (segment_length // 2 + 1,); x in Hz,
        y dimensionless, nominally in [0, 1]. Zero auto-power product gives
        nan at that bin; constant inputs give all nan. Consumed NaNs and
        overflow propagate nonfinite values.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Unknown window.
        Segment count below 1, overlap outside [0, 1), or derived segments
        shorter than 4 samples.
        Mismatched time grids or fewer than two averaged segments.

    TypeError
        Noninteger segments can reach integer-only iteration; no explicit
        integer validation is performed.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Consumed NaNs, nonfinite segment powers, or zero auto-power bins.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.

    Notes
    -----
    Inputs are not mutated; returned arrays and Signals are newly allocated.
    """
    _check_transient(a, 'coherence')
    _check_transient(b, 'coherence')
    if not _axis(a, 'coherence').same_grid(_axis(b, 'coherence')):
        raise ValueError(
            f'coherence: a and b must share the same time axis; '
            f'a has {len(a.x)} points spanning {a.x[0]}..{a.x[-1]} s, '
            f'b has {len(b.x)} points spanning {b.x[0]}..{b.x[-1]} s'
        )
    a_s = _sample_at_clock(a, metric='coherence', tclk=tclk,
                           tstart=tstart, n_samples=n_samples)
    b_s = _sample_at_clock(b, metric='coherence', tclk=tclk,
                           tstart=tstart, n_samples=n_samples)
    n = len(a_s)
    seg_len, step = _segment_layout(n, segments, overlap, metric='coherence')
    w = _make_window(window, seg_len, metric='coherence')
    Sxx_sum = np.zeros(seg_len // 2 + 1, dtype=float)
    Syy_sum = np.zeros(seg_len // 2 + 1, dtype=float)
    Sxy_sum = np.zeros(seg_len // 2 + 1, dtype=complex)
    n_segs_used = 0
    for k in range(segments):
        start = k * step
        end = start + seg_len
        if end > n:
            break
        a_seg = (a_s[start:end] - np.mean(a_s[start:end])) * w
        b_seg = (b_s[start:end] - np.mean(b_s[start:end])) * w
        A = np.fft.rfft(a_seg)
        B = np.fft.rfft(b_seg)
        Sxx_sum += np.abs(A) ** 2
        Syy_sum += np.abs(B) ** 2
        Sxy_sum += A * np.conj(B)
        n_segs_used += 1
    if n_segs_used < 2:
        raise ValueError(
            f'coherence: requires >= 2 averaged segments, got {n_segs_used}; '
            f'requested {segments} segments with overlap={overlap}, '
            f'seg_len={seg_len} from {n} samples'
        )
    denom = Sxx_sum * Syy_sum
    # denom == 0 implies Sxy == 0 too (Cauchy-Schwarz), so this is an
    # intentional 0/0 -> nan (see docstring); do not let numpy's default
    # warning obscure it.
    with np.errstate(divide='ignore', invalid='ignore'):
        coh = (np.abs(Sxy_sum) ** 2) / denom
    used_end = (n_segs_used - 1) * step + seg_len
    a_used, b_used = a_s[:used_end], b_s[:used_end]
    nan_samples = _any_nan(a_used) or _any_nan(b_used)
    nonfinite_power = (not np.all(np.isfinite(Sxx_sum))
                       or not np.all(np.isfinite(Syy_sum))
                       or not np.all(np.isfinite(Sxy_sum))
                       or not np.all(np.isfinite(denom)))
    zero_bins = int(np.count_nonzero(denom == 0))
    if nan_samples or nonfinite_power or zero_bins:
        details = {'segment_count': n_segs_used,
                   'segment_sample_count': seg_len,
                   'fft_bin_count': len(denom), 'zero_power_bin_count': zero_bins,
                   'bin_hz': 1.0 / (seg_len * tclk)}
        for name, sig, values in (('a', a, a_used), ('b', b, b_used)):
            details.update({f'{name}_{key}': value for key, value in
                            _sample_details(sig, values, tclk, tstart).items()})
        if nan_samples:
            code = 'nan_samples'
            message = ('Measurement unavailable because the consumed segment '
                       'data contains NaNs.')
            suggestion = 'Check the source result for the consumed samples.'
        elif nonfinite_power:
            code = 'nonfinite_spectrum'
            message = 'Segment FFT powers or their normalization are non-finite.'
            suggestion = 'Check the source result and input scaling for overflow.'
        else:
            code = 'zero_segment_power'
            message = ('Coherence is unavailable at bins with zero segment '
                       'auto-power product.')
            suggestion = 'Check signal variation and the segment length or window.'
        emit(metric='coherence', code=code, message=message, details=details,
             suggestion=suggestion)
    freqs = np.fft.rfftfreq(seg_len, d=tclk)
    return Signal(
        f'coherence({a.name},{b.name})', 'coherence', 'frequency', '',
        freqs, coh,
    )


@metric(domain='tran', unit='dB', param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz'})
def sfdr_db(sig: Transient, *, tclk: float, tstart: float = 0.0,
            n_samples: Optional[int] = None,
            fundamental_hz: Optional[float] = None,
            guard_bins: int = 0,
            window: Optional[SpectrumWindow] = 'rectangular') -> float:
    """Scalar SFDR in dB, from :func:`sfdr`, with its spectrum options.

    Inherits its finite numerical ceiling, 0 dB dead-signal floor, NaN
    propagation and caller-error validation; this performs no new FFT method.

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.
    guard_bins : int, default 0
        Nonnegative number of extra bins on either side of each tone and DC
        to exclude from residuals, in addition to the window main lobe.
    window : str or None, default 'rectangular'
        Window: 'hann', 'hamming', 'blackman', 'rectangular', or None
        (rectangular). Applied after subtracting the sample mean.

    Returns
    -------
    float
        SFDR in dB; nan for unavailable power or no spur bins. A constant
        rectangular-window record returns the 0 dB numerical-floor value.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz, window, guard_bins.
        Guard bins that eliminate otherwise available residuals also raise.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Unavailable power or no spur candidates, emitted by sfdr.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    _check_transient(sig, 'sfdr_db')
    return sfdr(sig, tclk=tclk, tstart=tstart, n_samples=n_samples,
                fundamental_hz=fundamental_hz, guard_bins=guard_bins,
                window=window).sfdr_db


@metric(domain='tran', unit='dBc', param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz'})
def harmonic_level_dbc(sig: Transient, *, n: int, tclk: float,
                       tstart: float = 0.0,
                       n_samples: Optional[int] = None,
                       fundamental_hz: Optional[float] = None) -> float:
    """Harmonic ``n >= 2`` level relative to the fundamental, in dBc.

    Selects entry ``n - 2`` from :func:`harmonic_levels_db`. Inherits its
    sampled-power convention and NaN diagnostics for unattributable harmonics
    or invalid data. Dead data reads the underlying 0 dBc floor value.
    A noninteger harmonic index or one below 2 raises ``ValueError``.
    ``fundamental_hz`` is passed through (module docstring, *Spectrum
    options*).

    Parameters
    ----------
    sig : Signal
        Transient with one-dimensional x (seconds) and real y arrays of equal
        length; x must be finite and strictly increasing. Input is not mutated.
    n : int
        Harmonic order, at least 2; booleans are rejected.
    tclk : float
        Positive sampling interval, seconds. Values at requested times use
        linear interpolation between recorded points; exact matches are kept.
    tstart : float, default 0.0
        First sampling time, seconds; must lie within the trace.
    n_samples : int or None, optional
        Sample count, at least 4 and fitting within the trace. None chooses
        the largest power of two fitting from tstart through the last point.
    fundamental_hz : float or None, optional
        Fundamental frequency, Hz, rounded to the nearest non-DC FFT bin.
        None elects the strongest eligible bin; explicit values must fit
        the available spectrum.

    Returns
    -------
    float
        Harmonic n level in dBc; nan for an unattributable alias or undefined
        power ratio. Constant data gives the underlying 0 dBc floor value.

    Raises
    ------
    ValueError
        Non-transient, empty, or invalid time axes; nonpositive tclk; a
        start or sample window outside trace coverage; fewer than 4 samples.
        Invalid fundamental_hz.
        Harmonic count/order below its minimum; harmonic_level_dbc also
        rejects noninteger orders.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Unattributable aliases or undefined ratios in harmonic_levels_db;
        all harmonics through n are evaluated, so other orders may warn.
        Off-grid sampling also reports interpolation counts and absolute
        nearest-recorded-time distance (seconds), separately for each input.
        Warnings follow the active Python warning filter.
    """
    _check_transient(sig, 'harmonic_level_dbc')
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f'harmonic_level_dbc: n must be an integer >= 2, got {n!r}')
    return float(harmonic_levels_db(
        sig, tclk=tclk, tstart=tstart, n_samples=n_samples, n_harmonics=n,
        fundamental_hz=fundamental_hz)[n - 2])
