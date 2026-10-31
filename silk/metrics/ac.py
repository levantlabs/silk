"""silk.metrics.ac — metrics from a .ac (small-signal) sweep.

All functions take a :class:`silk.signals.Signal` with ``xlabel='frequency'``
(``x = freqs`` in Hz, ``y`` = the complex response ``H(jω)`` exactly as
``from_rawfile`` loads a ``.ac`` sweep; the magnitude metrics also accept
a real ``|H|``) and return a ``float``, except for the per-frequency Signals returned by
:func:`silk.metrics.group_delay` and :func:`silk.metrics.psrr_db`, and the
list of frequency/margin pairs returned by :func:`silk.metrics.gain_margins`.

For real-valued input, phase_margin, phase_at_freq, gain_margin and
gain_margins emit one MetricWarning about missing phase per call and
preserve their real-axis calculation (0 or 180 degrees). Pass the complex
AC response H(jw), rather than its magnitude, to retain phase information.
Other diagnostics may still describe unavailable measurements.

This module also owns the nine schema-decorated AC
metrics (``dc_gain_dB``, ``peak_gain_dB``, ``gain_at_freq``,
``bandwidth_3dB``, ``unity_gain_freq``, ``phase_at_freq``,
``phase_margin``, ``gain_margin``, ``cmrr_dB``) and the undecorated
``gain_margins`` they build on.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from silk.signals.base import Signal
from silk.signals.schema import metric, FrequencyResponse
from silk.metrics._guards import (  # noqa: F401
    _check_nonempty, _check_freq, _any_nan, _isnan_complex_aware, _nan_safe_transform,
    _bracket_freq,
)
from silk.metrics._guards import _axis, _Axis
from silk.metrics.diagnostics import emit


def _diagnostic_details(sig: Signal, selection=None) -> dict:
    """Summarize the consumed AC data only when an explanation is needed."""
    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y)
    if selection is not None:
        x, y = x[selection], y[selection]
    nan = _isnan_complex_aware(y)
    mag = _nan_safe_transform(y, np.abs)
    # complex-aware (TASK-178): matches the guard these details explain,
    # so a NaN-bearing sample (e.g. inf+nanj) is never reported as clean
    observed = mag[~nan]
    details = {
        'signal_name': sig.name,
        'frequency_min_hz': float(np.min(x)),
        'frequency_max_hz': float(np.max(x)),
        'magnitude_min': float(np.min(observed)) if observed.size else None,
        'magnitude_max': float(np.max(observed)) if observed.size else None,
        'magnitude_unit': sig.ylabel or 'unspecified',
        'sample_count': int(mag.size),
        'nan_sample_count': int(np.count_nonzero(nan)),
    }
    if np.any(nan):
        details['nan_frequency_min_hz'] = float(np.min(x[nan]))
        details['nan_frequency_max_hz'] = float(np.max(x[nan]))
    return details


def _warn_real_phase(sig: Signal, metric_name: str) -> None:
    """Explain the phase information absent from a real response."""
    if not np.iscomplexobj(sig.y):
        emit(
            metric=metric_name, code='real_response_no_phase',
            message='Real-valued input carries no phase information from the '
                    'complex AC response; using real-axis phase (0 or 180 degrees).',
            details={'signal_name': sig.name},
            suggestion='Pass the complex AC response H(jw), rather than its magnitude.',
        )


def _check_reference(reference: str, metric_name: str) -> None:
    """Validate ``bandwidth_3dB``'s ``reference`` choice."""
    if reference not in ('dc', 'peak'):
        raise ValueError(
            f"{metric_name}: reference must be 'dc'|'peak', got {reference!r}"
        )


@metric(domain='ac', unit='dB', param_units={'f_pass': 'Hz'})
def passband_ripple_db(sig: FrequencyResponse, f_pass: float) -> float:
    """Passband peak-to-valley magnitude ratio in dB.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.
    f_pass : float
        Passband edge (Hz).

    Notes
    -----
    Returns ``nan`` if any sweep point INSIDE the band has NaN magnitude.
    A NaN outside the band is not consumed and does not affect the result.
    Zero in-band magnitudes also return ``nan`` with a ``MetricWarning``:
    a partially zero band has unbounded ripple, and an all-zero band has
    no measurable reference. Fewer than two in-band samples still raises
    ``ValueError`` for a malformed request. Warning filters apply normally.

    Returns
    -------
    float
        Peak-to-valley ripple in dB: 20*log10(max(abs(y))/min(abs(y)))
        over samples with x <= f_pass; nan for unavailable in-band data.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'passband_ripple_db')
    mask = sig.x <= f_pass
    n = int(np.sum(mask))
    lo, hi = float(np.min(sig.x)), float(np.max(sig.x))
    if n < 2:
        raise ValueError(
            f'passband_ripple_db: band [0, {f_pass:g}] Hz contains {n} of '
            f'{len(sig.x)} sweep points (sweep spans {lo:g} Hz .. {hi:g} Hz)'
        )
    # guard directly on the complex samples (TASK-178): _any_nan checks
    # real/imag separately, so it does not need a magnitude fold first
    if _any_nan(np.asarray(sig.y)[mask]):
        emit(
            metric='passband_ripple_db', code='nan_magnitude',
            message='Measurement unavailable: consumed passband magnitude '
                    'data contains NaNs.',
            details={**_diagnostic_details(sig, mask),
                     'band_max_hz': float(f_pass)},
            suggestion='Check the source result for the affected in-band samples.',
        )
        return float('nan')
    band = np.abs(sig.y[mask])
    zero_count = int(np.count_nonzero(band == 0))
    if zero_count:
        no_reference = zero_count == n
        emit(
            metric='passband_ripple_db',
            code=('no_passband_reference' if no_reference
                  else 'zero_passband_magnitude'),
            message=('No measurable passband reference: all in-band '
                     'magnitudes are zero.' if no_reference else
                     'Passband ripple is unbounded: the band contains '
                     'both zero and nonzero magnitudes.'),
            details={
                'signal_name': sig.name, 'band_max_hz': float(f_pass),
                'frequency_min_hz': float(np.min(sig.x[mask])),
                'frequency_max_hz': float(np.max(sig.x[mask])),
                'sample_count': n, 'zero_sample_count': zero_count,
                'magnitude_min': float(np.min(band)),
                'magnitude_max': float(np.max(band)),
                'magnitude_unit': sig.ylabel or 'unspecified',
            },
            suggestion='Check the in-band response, transfer normalization, '
                       'and operating point.',
        )
        return float('nan')
    return float(20.0 * np.log10(np.max(band) / np.min(band)))


@metric(domain='ac', unit='dB', param_units={'f_stop': 'Hz'})
def stopband_attenuation_db(sig: FrequencyResponse, f_stop: float) -> float:
    """Stopband attenuation = passband peak / stopband peak, in dB.

    Zero peaks are properties of valid data, not malformed
    frequency requests. With a nonzero passband peak and exactly zero
    stopband peak, returns ``+inf``: the supplied response has infinite
    rejection. This is a mathematical limit, not evidence of a physical
    measurement's dynamic range. A zero passband peak returns ``nan``:
    no passband reference exists, so attenuation is unavailable. NaN in
    either consumed band also returns ``nan``, even with a zero reference.
    These unavailable results emit ``MetricWarning`` with the observed
    cause; warning filters apply normally.
    Empty passband/stopband selections still raise ``ValueError``.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.
    f_stop : float
        Stopband lower edge in Hz; passband uses x < f_stop and stopband
        uses x >= f_stop. Both selections must contain samples.

    Returns
    -------
    float
        20*log10(passband_peak/stopband_peak) in dB; nan without a
        measurable reference, or +inf for zero stopband and nonzero passband.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'stopband_attenuation_db')
    pass_mask = sig.x < f_stop
    stop_mask = sig.x >= f_stop
    lo, hi = float(np.min(sig.x)), float(np.max(sig.x))
    if not np.any(pass_mask):
        raise ValueError(
            f'stopband_attenuation_db: passband f < {f_stop:g} Hz contains 0 '
            f'of {len(sig.x)} sweep points (sweep spans {lo:g} Hz .. {hi:g} Hz)'
        )
    if not np.any(stop_mask):
        raise ValueError(
            f'stopband_attenuation_db: stopband f >= {f_stop:g} Hz contains 0 '
            f'of {len(sig.x)} sweep points (sweep spans {lo:g} Hz .. {hi:g} Hz)'
        )
    pass_peak = float(np.max(_nan_safe_transform(sig.y[pass_mask], np.abs)))
    stop_peak = float(np.max(_nan_safe_transform(sig.y[stop_mask], np.abs)))
    # complex-aware (TASK-178): np.isnan(pass_peak/stop_peak) on the
    # already-folded peak would miss e.g. inf+nanj — see passband_ripple_db
    if _any_nan(sig.y[pass_mask]) or _any_nan(sig.y[stop_mask]):
        emit(
            metric='stopband_attenuation_db', code='nan_magnitude',
            message='Measurement unavailable: consumed passband or stopband '
                    'magnitude data contains NaNs.',
            details={**_diagnostic_details(sig),
                     'stopband_min_hz': float(f_stop),
                     'passband_peak_magnitude': pass_peak,
                     'stopband_peak_magnitude': stop_peak},
            suggestion='Check the source result for the affected samples.',
        )
        return float('nan')
    if pass_peak <= 0:
        emit(
            metric='stopband_attenuation_db', code='no_passband_reference',
            message='No measurable passband reference: all passband '
                    'magnitudes are zero.',
            details={**_diagnostic_details(sig),
                     'stopband_min_hz': float(f_stop),
                     'passband_peak_magnitude': pass_peak,
                     'stopband_peak_magnitude': stop_peak},
            suggestion='Check the passband response, transfer normalization, '
                       'and operating point.',
        )
        return float('nan')
    if stop_peak <= 0:
        return float('inf')
    return float(20.0 * np.log10(pass_peak / stop_peak))


@metric(domain='ac', unit='')
def q_from_peak(sig: FrequencyResponse) -> float:
    """Q-factor of a resonant peak: f_peak / Δf_3dB.

    Resolution: both −3 dB edges are the first/last GRID points at or
    above the half-power level (no interpolation), so Δf is under-read
    by up to two grid steps and Q over-read by up to ``2·(r−1)·Q``
    relative, ``r`` = grid ratio (``10^(1/points_per_decade)``): 4.6 %
    for Q = 10 at 1000 pts/decade, 46 % at 100. Intended for a single
    resonant peak. Returns ``nan`` with a ``MetricWarning`` if the
    half-power band reaches either sweep edge: both boundaries were not
    measured. Warning filters apply normally. Existing ``nan`` results
    for consumed NaNs, fewer than three samples, no positive peak, fewer than two
    half-power samples, or nonpositive bandwidth are retained.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.

    Returns
    -------
    float
        Dimensionless Q. Returns nan with MetricWarning for consumed NaNs,
        no positive peak, fewer than three points, an unresolved half-power
        span, or a span reaching either sweep edge.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'q_from_peak')
    axis = _axis(sig, 'q_from_peak')
    x, y = axis.increasing, axis.orient(sig.y)
    if _any_nan(y):
        emit(
            metric='q_from_peak', code='nan_magnitude',
            message='Measurement unavailable: consumed magnitude data contains NaNs; '
                    'the peak and half-power threshold are NaN.',
            details={**_diagnostic_details(sig), 'peak_magnitude': float('nan'),
                     'threshold_magnitude': float('nan'), 'half_power_sample_count': 0},
            suggestion='Check the source result for the affected samples.',
        )
        return float('nan')
    mag = np.abs(y)
    if len(mag) < 3:
        emit(
            metric='q_from_peak', code='too_few_samples',
            message='No half-power bandwidth could be established: the sweep '
                    f'has {len(mag)} sample(s) and a bounded peak needs at '
                    'least three.',
            details={**_diagnostic_details(sig), 'sample_count': int(len(mag))},
            suggestion='Sweep at least three frequencies across the peak.',
        )
        return float('nan')
    peak_idx = int(np.argmax(mag))
    peak = mag[peak_idx]
    if peak <= 0:
        emit(
            metric='q_from_peak', code='no_positive_peak',
            message='No positive peak measured: all sweep magnitudes are zero.',
            details={**_diagnostic_details(sig), 'peak_magnitude': float(peak)},
            suggestion='Check the response, transfer normalization, and operating point.',
        )
        return float('nan')
    half_pwr = peak / np.sqrt(2.0)
    above = np.where(mag >= half_pwr)[0]
    if above.size < 2:
        emit(
            metric='q_from_peak',
            code='half_power_span_unresolved',
            message='Half-power span unresolved: fewer than two samples reach '
                    'the peak-relative threshold.',
            details={**_diagnostic_details(sig), 'peak_magnitude': float(peak),
                     'threshold_magnitude': float(half_pwr),
                     'half_power_sample_count': int(above.size)},
            suggestion='Try a denser frequency sweep around the peak.',
        )
        return float('nan')
    selected_axis = _Axis(x[above], 'q_from_peak')
    f_lo, f_hi = selected_axis.lower, selected_axis.upper
    bw = f_hi - f_lo
    if bw <= 0:
        emit(
            metric='q_from_peak', code='nonpositive_bandwidth',
            message='No half-power bandwidth could be established: the '
                    'half-power edges coincide.',
            details={**_diagnostic_details(sig), 'threshold_magnitude': float(half_pwr)},
            suggestion='Check the frequency grid for duplicated points.',
        )
        return float('nan')
    if above[0] == 0 or above[-1] == len(mag) - 1:
        emit(
            metric='q_from_peak', code='half_power_band_unbounded',
            message='A bounded half-power band was not measured: '
                    'the response remains at/above threshold at a sweep edge.',
            details={**_diagnostic_details(sig),
                     'threshold_magnitude': float(half_pwr),
                     'lower_edge_above_threshold': bool(above[0] == 0),
                     'upper_edge_above_threshold': bool(above[-1] == len(mag) - 1)},
            suggestion='Try extending the sweep at the affected edge; '
                       'a bounded resonance outside this range is not established.',
        )
        return float('nan')
    return float(x[peak_idx]) / bw


def group_delay(sig: Signal) -> Signal:
    """Group delay τ_g(ω) = −dφ/dω of a complex AC response, per frequency.

    The phase is ``np.unwrap(np.angle(sig.y))`` and the derivative is
    ``numpy.gradient`` over ``ω = 2π·f`` on the sweep grid — no
    resampling, no fit. Samples whose response magnitude is exactly zero
    have no defined phase: their delay is ``nan`` (one ``MetricWarning``
    for the whole trace), and the remaining samples are unwrapped and
    differentiated on their own grid, so a zero point neither reports an
    invented delay nor corrupts its neighbours.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values must be complex
        linear gain (V/V); real magnitude or phase arrays are rejected.
        The input is not modified.

    Returns
    -------
    Signal
        Shape (N,), ``xlabel='frequency'`` on the input's own grid and order, ``y`` = τ_g in
        seconds (``ylabel='s'``). Query one frequency with
        :func:`silk.metrics.value_at`. Every sample is ``nan`` when
        ``sig.y`` holds a NaN (a failed convergence
        point is data, not a caller error, and ``np.unwrap`` would
        otherwise smear one NaN over every later sample while the
        gradient left the earlier ones looking valid).

    Notes
    -----

    **Resolution**

    Finite differences on the sweep grid, ``r = 10^(1/points_per_decade)``
    the grid ratio. Interior points are second-order central differences
    on the non-uniform ω grid: relative error ≤ ``(r − 1)²/r`` — 0.05 %
    at ``.ac dec 100``, 1.3 % at ``dec 20``, 5.3 % at ``dec 10`` — the
    worst case being far above a pole, where τ falls as 1/ω². The two
    END points are first-order one-sided differences and can read
    ``r − 1`` high or low in that same region: 2.3 % at ``dec 100``,
    12 % at ``dec 20``, 26 % at ``dec 10`` (the same one-grid-step number
    :func:`bandwidth_3dB` quotes). Where τ is flat
    (well below every pole) all three are exact to round-off. The
    unwrap needs the phase to move less than π between adjacent grid
    points — ``τ_g · Δf < 1/2`` — so a long pure delay on a coarse
    sweep aliases (a 1 µs delay needs ``Δf < 500 kHz`` at the top of the
    sweep); sweep more points per decade, not a narrower band.

    Previously, this function cast ``y`` to float (``ComplexWarning``)
    and differentiated Re ``H`` instead of the phase: on the
    3-pole fixture it read 32.3 µs at the sweep start (true 16.0 µs) and
    7.99 ms at the 10 kHz pole (true 8.0 µs — the number scaled with the
    60 dB gain, it was not even a time).

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For any NaN response (all delays become nan), or zero response
        magnitude (those samples become nan). Fewer than two nonzero
        responses also leaves all delays nan; a two-point nonzero response
        uses the same first difference at both endpoints.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. Real-valued response arrays are also rejected. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'group_delay')
    axis = _axis(sig, 'group_delay')
    y = axis.orient(sig.y)
    if not np.iscomplexobj(y):
        raise ValueError(
            f"group_delay: Signal {sig.name!r} has real-valued y "
            f"({y.dtype}); group delay needs the complex AC response "
            f"H(jw) as from_rawfile loads it — a magnitude or phase array "
            f"carries no recoverable phase (rebuild H = exp(1j*phase) if "
            f"you only hold the phase)"
        )
    x = axis.increasing
    if _any_nan(y):                               # guard on the complex samples: see passband_ripple_db
        emit(
            metric='group_delay', code='nan_magnitude',
            message='Group delay unavailable: consumed complex-response magnitude '
                    'contains NaNs; phase unwrapping and differentiation cannot '
                    'measure the delay.',
            details=_diagnostic_details(sig),
            suggestion='Check the source result for the affected samples.',
        )
        tau = np.full(len(x), np.nan)
    else:
        tau = np.full(len(x), np.nan)
        live = np.abs(y) > 0
        if not live.all():
            emit(
                metric='group_delay', code='zero_response',
                message='Group delay undefined where the response is zero: '
                        f'{int((~live).sum())} of {len(x)} samples have zero '
                        'magnitude, so their phase is undefined.',
                details={**_diagnostic_details(sig),
                         'zero_sample_count': int((~live).sum())},
                suggestion='Check the source response at the affected frequencies.',
            )
        if live.sum() >= 2:
            omega = 2.0 * np.pi * x[live]
            tau[live] = -np.gradient(np.unwrap(np.angle(y[live])), omega)
    return Signal(f'group_delay({sig.name})', 'group_delay',
                  'frequency', 's', axis.x, axis.orient(tau))


# ---------------------------------------------------------------------------
# Gain-bandwidth product
# ---------------------------------------------------------------------------

@metric(domain='ac', unit='Hz*V/V')
def gain_bw_product(sig: FrequencyResponse) -> float:
    """Gain × −3 dB bandwidth.

    DC gain is taken at the LOWEST frequency in the sweep (order-
    independent, like :func:`dc_gain_dB` — an
    ascending and a descending sweep of the same data agree); bandwidth
    is the highest frequency where ``|sig.y|`` is still within 3 dB of
    DC. Multiplied gives the gain-bandwidth product. ``sig.y`` may be
    real or complex — ``np.abs`` takes the magnitude either way.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.

    Returns
    -------
    float
        GBW in Hz·V/V (or whatever the gain unit is). ``nan`` if the
        gain never drops to the −3 dB point, or if it is still within
        3 dB of DC at the top of the sweep — the true −3 dB point, if
        any, is above the swept range and unknowable from this data —
        or if ``sig.y`` contains a NaN sample (the same guard
        :func:`bandwidth_3dB` has): ``nan >=
        threshold`` is ``False``, so a NaN sitting on the −3 dB grid
        point was skipped and the bandwidth read one grid point low —
        11 % at 20 pts/decade — a finite wrong number.

    Notes
    -----
    Resolution: the bandwidth is the last GRID point at or above −3 dB
    (no interpolation) — up to one grid step low: 1.2 % at 200
    pts/decade, 12 % at 20.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For consumed NaNs, zero lowest-frequency gain (returns nan), or
        an upper roll-off not observed within the sweep (returns nan).

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'gain_bw_product')
    if _any_nan(np.asarray(sig.y)):               # guard on the complex samples: see passband_ripple_db
        emit(
            metric='gain_bw_product', code='nan_magnitude',
            message='Measurement unavailable: consumed magnitude data contains NaNs.',
            details=_diagnostic_details(sig),
            suggestion='Check the source result for the affected samples.',
        )
        return float('nan')
    order = _axis(sig, 'gain_bw_product').indices
    x = np.asarray(sig.x, dtype=float)[order]
    mag = np.abs(np.asarray(sig.y))[order]
    dc_gain = float(mag[0])
    if dc_gain <= 0:
        emit(
            metric='gain_bw_product', code='no_dc_reference',
            message='No measurable DC reference: the lowest-frequency magnitude is zero.',
            details={**_diagnostic_details(sig), 'dc_magnitude': dc_gain,
                     'dc_frequency_hz': float(x[0])},
            suggestion='Check the low-frequency response, transfer normalization, '
                       'and operating point.',
        )
        return float('nan')
    threshold = dc_gain / np.sqrt(2.0)
    above = np.where(mag >= threshold)[0]
    if len(above) == 0:
        return float('nan')
    if above[-1] == len(mag) - 1:
        emit(
            metric='gain_bw_product', code='upper_rolloff_not_observed',
            message='Upper roll-off boundary not measured within the sweep: '
                    'the highest-frequency magnitude remains at/above the '
                    'DC-relative threshold.',
            details={**_diagnostic_details(sig), 'dc_magnitude': dc_gain,
                     'threshold_magnitude': float(threshold),
                     'endpoint_magnitude': float(mag[-1])},
            suggestion='Try extending the upper sweep limit; a roll-off '
                       'outside this range is not established.',
        )
        return float('nan')
    bw_3db = float(x[above[-1]])
    return dc_gain * bw_3db


# ---------------------------------------------------------------------------
# Supply rejection
# ---------------------------------------------------------------------------

def psrr_db(v_supply: Signal, v_out: Signal) -> Signal:
    """Power Supply Rejection Ratio in dB at each frequency.

    PSRR(f) = 20·log10(``|V_supply(f) / V_out(f)|``)

    A larger PSRR means the supply noise is more strongly rejected at
    the output. Both inputs must be AC-sweep Signals on the same
    frequency grid (typical setup: small-signal-source on the supply,
    measure both supply node and output). Coordinates must agree within
    1e-3 of the smallest spacing across the two grids; mismatches raise
    ``ValueError``. Acquisition order must also agree.

    A perfectly rejected point (``|V_out| == 0`` at some frequency) is
    degenerate DATA, not a malformed request, so this
    still returns a number rather than raising. ``|V_out|`` is floored
    at ``np.finfo(float).eps`` *relative to* ``|V_supply|`` at that
    frequency (i.e. ``max(|V_out|, eps · |V_supply|)``) rather than
    left unguarded (a bare divide-by-zero) or floored at
    ``np.finfo(float).tiny`` as before 2026-09-05 — the ``tiny`` floor
    made an exactly-zero output read as ``20·log10(1/tiny) ≈ +6153``
    dB, a floor artifact indistinguishable from a real measurement.
    The ``eps`` floor is scale-independent and caps the reportable
    value at ``20·log10(1/eps) ≈ +313`` dB, which is still finite and
    unambiguously a rejection no real circuit achieves, but is small
    enough to not be mistaken for a plausible PSRR by inspection. If
    ``V_supply`` is *also* zero at that frequency the floor is zero and
    the ratio is ``0/0`` → ``nan`` (unknowable, the degenerate-data convention: no
    supply ripple was injected, so no PSRR can be measured there).

    Parameters
    ----------
    v_supply : Signal
        Supply ripple in V, real or complex, with ``xlabel='frequency'``.
        Equal-length one-dimensional x/y arrays of shape (N,), N >= 2;
        finite Hz coordinates must be strictly increasing or decreasing.
    v_out : Signal
        Output response in V on the same ordered frequency grid, within
        1e-3 of the smallest spacing. Opposite grid orders are rejected.
        Neither input is modified.

    Returns
    -------
    Signal
        ``xlabel='frequency'``, ``y`` in dB, capped at ≈313 dB (see
        above). Use :func:`silk.metrics.value_at` to query a single
        frequency.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For NaN pairs or zero/zero pairs, whose output samples are nan.
        Other samples remain measurable. A zero supply with nonzero output
        returns -inf without this warning. The returned Signal has shape
        (N,) and retains the supply grid and its order.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. Different frequency grids are also rejected. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(v_supply, 'psrr_db')
    _check_freq(v_out, 'psrr_db')
    if not _axis(v_supply, 'psrr_db').same_grid(_axis(v_out, 'psrr_db')):
        raise ValueError(
            'psrr_db: v_supply and v_out must share the same frequency axis; '
            f'found differing grids: v_supply has {len(v_supply.x)} points '
            f'({np.min(v_supply.x):g} Hz .. {np.max(v_supply.x):g} Hz), '
            f'v_out has {len(v_out.x)} points '
            f'({np.min(v_out.x):g} Hz .. {np.max(v_out.x):g} Hz)'
        )
    out_y = np.asarray(v_out.y)
    sup_y = np.asarray(v_supply.y)
    # complex-aware (TASK-178): np.isnan(sup_mag)|np.isnan(out_mag) would
    # miss e.g. inf+nanj, whose magnitude folds to inf — see passband_ripple_db
    contaminated = _isnan_complex_aware(sup_y) | _isnan_complex_aware(out_y)
    out_mag = _nan_safe_transform(out_y, np.abs)
    sup_mag = _nan_safe_transform(sup_y, np.abs)
    zero_pair = (sup_mag == 0) & (out_mag == 0)
    if np.any(contaminated | zero_pair):
        details = {
            **{f'supply_{key}': value
               for key, value in _diagnostic_details(v_supply).items()},
            **{f'output_{key}': value
               for key, value in _diagnostic_details(v_out).items()},
            'nan_pair_count': int(np.count_nonzero(contaminated)),
            'zero_pair_count': int(np.count_nonzero(zero_pair)),
        }
        if np.any(zero_pair):
            details.update(
                zero_pair_frequency_min_hz=float(np.min(v_supply.x[zero_pair])),
                zero_pair_frequency_max_hz=float(np.max(v_supply.x[zero_pair])),
            )
        message = (
            'Some PSRR samples are unavailable: consumed supply/output '
            'magnitudes contain NaNs.' if np.any(contaminated) else
            'Some PSRR samples are unavailable: no supply reference was measured.'
        )
        if np.any(zero_pair):
            message += ' Both supply and output magnitudes are zero at the counted points (0/0).'
        emit(
            metric='psrr_db',
            code=('nan_magnitude' if np.any(contaminated) else 'no_supply_reference'),
            message=message,
            details=details,
            suggestion='Check the source result at NaN samples and the injected '
                       'supply ripple at zero/zero samples.',
        )
    # Floor |V_out| relative to |V_supply| rather than an absolute tiny
    # value: caps the reportable PSRR at 20*log10(1/eps) ~= 313 dB
    # regardless of signal scale (see docstring).
    floor = np.finfo(float).eps * sup_mag
    out_mag = np.maximum(out_mag, floor)
    # sup_mag == out_mag == 0 gives an intentional 0/0 -> nan (see
    # docstring); do not let numpy's default warning obscure it.
    with np.errstate(divide='ignore', invalid='ignore'):
        psrr = 20.0 * np.log10(sup_mag / out_mag)
    # TASK-178: an inf-paired-with-nan sample folds to an infinite, not
    # NaN, magnitude, so the ratio above would no longer propagate a NaN
    # on its own at a `contaminated` sample -- preserve NaN explicitly.
    psrr = np.where(contaminated, np.nan, psrr)
    return Signal(
        f'psrr({v_supply.name},{v_out.name})', 'psrr',
        'frequency', 'dB',
        np.asarray(v_supply.x, dtype=float), psrr,
    )


# ===========================================================================
# Loop / gain-response metrics
# (TASK-172). These nine carry the @metric schema decorator; gain_margins
# is the undecorated list-valued primitive gain_margin is built on.
# ===========================================================================


@metric(domain='ac', unit='dB')
def dc_gain_dB(sig: FrequencyResponse) -> float:
    """Gain in dB at the LOWEST frequency of the sweep — the DC gain, to the
    extent the sweep reaches down to it.

    History (2026-08-23): this function used to return the PEAK magnitude
    anywhere in the sweep. For a monotonically rolling-off response the two
    coincide — every existing example is such a case — but for a peaked
    response (underdamped filter, gain peaking) the peak overstates the DC
    gain by up to the full peaking. The peak now lives in
    :func:`peak_gain_dB`, whose name says what it computes.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.

    Returns
    -------
    float
        20 * log10(``|y|``) at min(x), in dB.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For a NaN in the selected lowest-frequency sample, returning nan.
    RuntimeWarning
        NumPy emits a logarithm warning for zero magnitude, returning -inf dB.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals import Signal
    >>> from silk.metrics import dc_gain_dB, gain_at_freq
    >>> response = Signal('gain', 'voltage', 'frequency', 'V/V',
    ...                   np.array([1., 10., 100.]), np.array([10., 1., .1]))
    >>> dc_gain_dB(response)
    20.0
    >>> round(gain_at_freq(response, 10.), 6)
    1.0
    """
    _check_freq(sig, 'dc_gain_dB')
    idx = int(np.argmin(sig.x))
    if _any_nan(sig.y[idx]):                   # guard on the complex sample: see passband_ripple_db
        emit(
            metric='dc_gain_dB', code='nan_magnitude',
            message='Measurement unavailable: the selected lowest-frequency '
                    'magnitude is NaN.',
            details=_diagnostic_details(sig, [idx]),
            suggestion='Check the source result at the selected frequency.',
        )
        # TASK-178: an inf-paired-with-nan sample folds to an infinite,
        # not NaN, magnitude, so np.log10 would no longer propagate a NaN
        # on its own -- return NaN explicitly.
        return float('nan')
    return float(20.0 * np.log10(np.abs(sig.y[idx])))


@metric(domain='ac', unit='dB')
def peak_gain_dB(sig: FrequencyResponse) -> float:
    """Peak magnitude in dB anywhere in an AC frequency-response signal.

    (The computation that ``dc_gain_dB`` performed before 2026-08-23, under
    the name that matches it.)

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.

    Returns
    -------
    float
        20 * log10(max(``|y|``)), in dB.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For a NaN in the sweep, returning nan.
    RuntimeWarning
        NumPy emits a logarithm warning for zero magnitude, returning -inf dB.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'peak_gain_dB')
    if _any_nan(sig.y):
        emit(
            metric='peak_gain_dB', code='nan_magnitude',
            message='Measurement unavailable: consumed magnitude data contains NaNs.',
            details=_diagnostic_details(sig),
            suggestion='Check the source result for the affected samples.',
        )
        # TASK-178: an inf-paired-with-nan sample folds to an infinite,
        # not NaN, magnitude, so max(|y|)/log10 would no longer propagate
        # a NaN on its own -- return NaN explicitly.
        return float('nan')
    return float(20.0 * np.log10(np.max(np.abs(sig.y))))


@metric(domain='ac', unit='V/V', param_units={'freq': 'Hz'})
def gain_at_freq(sig: FrequencyResponse, freq: float,
                 *, interpolate: bool = True) -> float:
    """Linear gain magnitude at *freq*, interpolated between sweep points.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.
    freq : float
        Target frequency in Hz.
    interpolate : bool
        ``True`` (default): ``|y|`` is interpolated log-log (dB
        against log frequency) between the two neighbouring sweep
        points, linear in ``|y|`` if an endpoint is exactly zero. Log-log is
        exact on a power-law roll-off and measured 0.028 dB worst case on
        a one-pole at 10 points/decade, against 0.058 dB for linear
        magnitude (the rule :func:`unity_gain_freq` uses, which is right
        there because it solves a linear crossing, not a point value).
        The nearest sample can be ~12 % off in ``|H|`` at that density.
        ``False`` returns the nearest
        sample, as before. A request exactly on a grid point returns
        that sample in both modes.

    Returns
    -------
    float
        Linear magnitude in the input gain unit (normally V/V) at *freq* (nearest sample if ``interpolate=False``). Strictly outside the supplied frequency
        range, returns ``nan`` with a ``MetricWarning`` naming the requested
        frequency and available range. Warning filters apply normally.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
        A non-finite requested frequency is also rejected.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For an out-of-range point or a NaN in a selected sample or bracket,
        returning nan. Unselected NaNs do not affect the result.
    """
    _check_freq(sig, 'gain_at_freq')
    if not np.isfinite(freq):
        raise ValueError(f'gain_at_freq: freq must be finite, got {freq!r} Hz')
    axis = _axis(sig, 'gain_at_freq')
    lo, hi = axis.lower, axis.upper
    if not axis.contains(freq):
        emit(
            metric='gain_at_freq', code='frequency_outside_sweep',
            message='Requested frequency is outside the supplied sweep.',
            details={'signal_name': sig.name,
                     'requested_frequency_hz': float(freq),
                     'frequency_min_hz': lo, 'frequency_max_hz': hi,
                     'sample_count': len(sig.x)},
            suggestion='Choose a frequency within the sweep or extend '
                       'the sweep to cover the requested point.',
        )
        return float('nan')
    if interpolate:
        i0, i1, t = _bracket_freq(sig.x, freq)
        idx = i0 if t < 0.5 else i1            # nearer neighbour (non-interpolating return)
        pair = [i0] if i0 == i1 else [i0, i1]
    else:
        idx = int(np.argmin(np.abs(sig.x - freq)))
        pair = [idx]
    bad = [j for j in pair if _any_nan(sig.y[j])]  # guard on the complex sample: see passband_ripple_db
    if bad:
        emit(
            metric='gain_at_freq', code='nan_magnitude',
            message='Measurement unavailable: a selected neighbouring '
                    'sample magnitude is NaN.',
            details={**_diagnostic_details(sig, bad),
                     'requested_frequency_hz': float(freq)},
            suggestion='Check the source result at the selected frequency.',
        )
        # TASK-178: an inf-paired-with-nan sample folds to an infinite,
        # not NaN, magnitude -- return NaN explicitly.
        return float('nan')
    if interpolate and i0 != i1:
        m0, m1 = float(np.abs(sig.y[i0])), float(np.abs(sig.y[i1]))
        if m0 > 0 and m1 > 0:
            return float(np.exp(np.log(m0) + t * (np.log(m1) - np.log(m0))))
        return float(m0 + t * (m1 - m0))        # a zero endpoint: log undefined
    return float(np.abs(sig.y[idx]))


@metric(domain='ac', unit='Hz')
def bandwidth_3dB(sig: FrequencyResponse,
                   *, reference: Literal['dc', 'peak'] = 'dc') -> float:
    """3 dB bandwidth: highest frequency where ``|gain|`` >= reference / sqrt(2).

    ``reference='dc'`` (the default) anchors the threshold to
    the DC gain — the same lowest-frequency magnitude :func:`dc_gain_dB`
    reports, reused here rather than re-derived — matching what a
    designer means by "the -3 dB bandwidth". ``reference='peak'``
    anchors it to the highest magnitude anywhere in the sweep instead;
    the two agree for a monotonically rolling-off response (DC IS the
    peak there), but ``reference='peak'`` reads LOW on a peaking
    response, since gain rises above DC before it rolls off — 25 % low
    on a zeta = 0.2 second-order low-pass (see
    tests/fixtures/task326_zeta02_lowpass.py). ``reference='peak'`` is
    still the right choice for a band-pass or high-pass response, which
    has no DC gain worth referencing.

    Viewed internally in increasing frequency order, so an
    ascending and a descending sweep of the same data agree.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear gain (V/V), or real linear magnitude where phase is not needed.
        The input is not modified.
    reference : {'dc', 'peak'}
        Which magnitude the -3 dB threshold is taken relative to.

    Returns
    -------
    float
        Frequency in Hz (same units as sig.x). Returns 0 if nothing in
        the sweep reaches the threshold, ``nan`` if the response is
        still at/above threshold at the top of the sweep — the upper
        roll-off boundary was not measured (under ``reference='dc'``
        this is also what a band-pass/high-pass response with little or
        no DC gain looks like: the near-zero threshold is essentially
        never crossed) — ``nan`` if ``reference='dc'`` and the DC
        magnitude is exactly zero (no measurable DC reference at all),
        or ``nan`` if ``sig.y`` contains a NaN sample: an unguarded NaN magnitude makes the reference (and
        so ``threshold``) NaN, every ``mag >= threshold`` comparison
        False, and the function would fall into the same branch as
        "nothing reaches the threshold" and silently return 0 — a
        finite value indistinguishable from a real, very-low bandwidth.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
        A reference other than ``'dc'`` or ``'peak'`` is also rejected.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Explains either NaN result with observed ranges and a suggested
        next step. An explicit warnings-as-errors filter raises it.

    Notes
    -----

    **Resolution**

    The result is the last SWEEP GRID POINT still at/above -3 dB of the
    reference — no interpolation — so it reads low by up to one grid
    step: a relative resolution of ``10^(1/points_per_decade) − 1``,
    i.e. 1.2 % at ``.ac dec 200``, 12 % at ``dec 20``, 26 % at ``dec
    10``. A before/after bandwidth comparison is only meaningful when
    the difference exceeds that (the design-loop walkthroughs cite it
    next to their bandwidth-drop diff).

    Dip and recover: if ``|gain|`` dips below the threshold and later climbs
    back at/above it, the LAST such point anywhere in the sweep is what
    is reported, not the first crossing — a resonance or notch after the
    "true" first -3 dB corner can push the reported bandwidth higher
    than that corner. This is unchanged by ``reference`` — both values
    run the identical scan against a different threshold.
    """
    _check_freq(sig, 'bandwidth_3dB')
    _check_reference(reference, 'bandwidth_3dB')
    if _any_nan(np.asarray(sig.y)):           # guard on the complex samples: see passband_ripple_db
        emit(
            metric='bandwidth_3dB', code='nan_magnitude',
            message='Measurement unavailable: consumed magnitude data '
                    'contains NaNs.',
            details=_diagnostic_details(sig),
            suggestion='Check the source result for the affected samples.',
        )
        return float('nan')
    order = _axis(sig, 'bandwidth_3dB').indices
    x = np.asarray(sig.x, dtype=float)[order]
    mag = np.abs(np.asarray(sig.y))[order]
    if reference == 'dc':
        dc_mag = 10.0 ** (dc_gain_dB(sig) / 20.0)
        if not dc_mag > 0:
            emit(
                metric='bandwidth_3dB', code='no_dc_reference',
                message='No measurable DC reference: the lowest-frequency '
                        'magnitude is zero.',
                details={**_diagnostic_details(sig),
                         'dc_magnitude': float(dc_mag),
                         'dc_frequency_hz': float(x[0])},
                suggestion="This response has no DC gain to reference; "
                          "try reference='peak'.",
            )
            return float('nan')
        ref_mag = dc_mag
    else:
        ref_mag = np.max(mag)
    threshold = ref_mag / np.sqrt(2.0)
    above = np.where(mag >= threshold)[0]
    if len(above) == 0:
        return 0.0
    if above[-1] == len(mag) - 1:
        suggestion = ('Try extending the upper sweep limit; if this is a '
                      "band-pass or high-pass response with little or no "
                      "DC gain, try reference='peak' instead."
                      if reference == 'dc' else
                      'Try extending the upper sweep limit.')
        emit(
            metric='bandwidth_3dB', code='upper_rolloff_not_observed',
            message='Upper roll-off boundary not measured within the sweep: '
                    'the highest-frequency magnitude remains at/above the '
                    f'{reference}-relative threshold.',
            details={**_diagnostic_details(sig),
                     'threshold_magnitude': float(threshold),
                     'endpoint_magnitude': float(mag[-1])},
            suggestion=suggestion,
        )
        return float('nan')
    return float(x[above[-1]])


@metric(domain='ac', unit='Hz')
def unity_gain_freq(sig: FrequencyResponse) -> float:
    """Frequency where ``|gain|`` crosses 1 (0 dB), interpolated.

    Rising and falling sweeps are viewed in increasing frequency order
    before searching. Non-monotonic or repeated coordinates raise ValueError.

    With more than one 0 dB crossing (peaking or a notch in the loop
    gain), the LOWEST-frequency crossing is returned — the usual
    convention — and a ``silk.metrics.diagnostics.MetricWarning``
    lists every crossing frequency and the phase margin at each, so a
    worse margin at a higher-frequency crossing cannot be silently
    hidden by the one chosen. A single-crossing sweep emits no such
    warning.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear loop gain (V/V). Real arrays are accepted but numpy.angle
        then uses only their real-axis phase (0 or 180 degrees); real
        magnitudes cannot recover the phase of a complex response.
        The input is not modified.

    Returns
    -------
    float
        Frequency in Hz. Returns ``nan`` if no unity crossing is observed
        in the supplied sweep (magnitude above or below one throughout),
        or if the consumed magnitude contains a NaN. Exact sampled
        crossings retain their measured frequency.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        Explains an unavailable measurement with observed ranges and a
        suggested next step, or (with more than one crossing) lists every
        crossing and its phase margin. An explicit warnings-as-errors
        filter raises it.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.


    Notes
    -----
    Resolution: interpolated (linear in magnitude, log in frequency)
    between the two grid points bracketing the crossing — well inside a
    grid step for any sane sweep density.
    """
    _check_freq(sig, 'unity_gain_freq')
    if _any_nan(np.asarray(sig.y)):           # guard on the complex samples: see passband_ripple_db
        emit(
            metric='unity_gain_freq', code='nan_magnitude',
            message='Measurement unavailable: consumed magnitude data '
                    'contains NaNs.',
            details=_diagnostic_details(sig),
            suggestion='Check the source result for the affected samples.',
        )
        return float('nan')
    order = _axis(sig, 'unity_gain_freq').indices
    x = np.asarray(sig.x, dtype=float)[order]
    y = np.asarray(sig.y)[order]
    mag = np.abs(y)
    ph_unwrapped = np.degrees(np.unwrap(np.angle(y)))

    def _pm_near(fc: float) -> float:
        idx = int(np.argmin(np.abs(x - fc)))
        return float(np.remainder(ph_unwrapped[idx], 360.0) - 180.0)

    crossings = []
    in_touch_run = False   # a flat run sitting exactly on 0 dB is ONE
    for i in range(len(mag) - 1):           # touch, not one crossing per step
        if (mag[i] >= 1.0 >= mag[i + 1]) or (mag[i] <= 1.0 <= mag[i + 1]):
            is_touch_pair = (mag[i] == 1.0 and mag[i + 1] == 1.0)
            if is_touch_pair and in_touch_run:
                continue
            in_touch_run = is_touch_pair
            f0, f1 = float(x[i]), float(x[i + 1])
            m0, m1 = float(mag[i]), float(mag[i + 1])
            if m0 == 1.0:
                fc = f0
            elif m1 == 1.0:
                fc = f1
            else:
                t = (1.0 - m0) / (m1 - m0)
                # interpolate in log-frequency space
                fc = float(np.exp(np.log(f0) + t * (np.log(f1) - np.log(f0))))
            if crossings and fc == crossings[-1]:
                continue
            crossings.append(fc)
        else:
            in_touch_run = False
    if crossings:
        if len(crossings) > 1:
            emit(
                metric='unity_gain_freq', code='multiple_unity_gain_crossings',
                message=f'{len(crossings)} unity-gain (0 dB) crossings observed '
                        'in the supplied sweep; returning the lowest-frequency '
                        'crossing.',
                details={**_diagnostic_details(sig),
                          'crossing_count': len(crossings),
                          'crossing_frequencies_hz': crossings,
                          'phase_margins_deg': [_pm_near(fc) for fc in crossings]},
                suggestion='Peaking or a notch can hide a worse margin at another '
                           'crossing; inspect every listed frequency and margin.',
            )
        return crossings[0]
    if np.all(mag > 1.0):
        emit(
            metric='unity_gain_freq', code='no_unity_crossing_gain_above_one',
            message='No unity-gain crossing observed in the supplied sweep: '
                    'magnitude remains above 1 throughout.',
            details=_diagnostic_details(sig),
            suggestion='Try extending the upper sweep limit; a crossing '
                       'outside this range is not established.',
        )
    else:
        emit(
            metric='unity_gain_freq', code='no_unity_crossing_gain_below_one',
            message='No unity-gain crossing observed in the supplied sweep: '
                    'magnitude remains below 1 throughout.',
            details=_diagnostic_details(sig),
            suggestion='Check sweep coverage, transfer normalization, and '
                       'operating point.',
        )
    return float('nan')


@metric(domain='ac', unit='deg', param_units={'freq': 'Hz'})
def phase_at_freq(sig: FrequencyResponse, freq: float,
                  *, interpolate: bool = True) -> float:
    """Phase in degrees at *freq*, interpolated between sweep points.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear loop gain (V/V). Real arrays are accepted but numpy.angle
        then uses only their real-axis phase (0 or 180 degrees); real
        magnitudes cannot recover the phase of a complex response.
        Real-valued input emits one MetricWarning about missing phase
        per call; the real-axis calculation is preserved. Pass the
        complex AC response H(jw) to retain phase information.
        The input is not modified.
    freq : float
        Target frequency in Hz.
    interpolate : bool
        ``True`` (default): phase is interpolated linearly in
        degrees against log frequency between the two neighbouring sweep
        points, along the SHORT arc: the neighbours' phase difference is
        folded into (-180, 180] first, so a ±180° wrap between them is
        never interpolated across. The result is returned on the
        principal (-180, 180] branch. ``False`` returns the nearest
        sample's phase, as before. A request exactly on a grid point
        returns that sample in both modes.

    Returns
    -------
    float
        Phase in degrees at *freq* (nearest sample if
        ``interpolate=False``). Strictly outside the supplied
        frequency range, returns ``nan`` with a ``MetricWarning`` naming the
        requested frequency and available range. Warning filters apply normally.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
        A non-finite requested frequency is also rejected.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For an out-of-range point or a NaN in a selected sample or bracket,
        returning nan. Unselected NaNs do not affect the result.

    Notes
    -----
    Zero response is passed to numpy.angle and gives zero degrees, without
    a zero-response warning. Real values emit the missing-phase warning
    and are accepted with their real-axis phase.
    Exact-grid and nearest-sample reads use numpy.angle directly, which can
    return -180 degrees; interpolated results use (-180, 180].
    """
    _check_freq(sig, 'phase_at_freq')
    _warn_real_phase(sig, 'phase_at_freq')
    if not np.isfinite(freq):
        raise ValueError(f'phase_at_freq: freq must be finite, got {freq!r} Hz')
    axis = _axis(sig, 'phase_at_freq')
    lo, hi = axis.lower, axis.upper
    if not axis.contains(freq):
        emit(
            metric='phase_at_freq', code='frequency_outside_sweep',
            message='Requested frequency is outside the supplied sweep.',
            details={'signal_name': sig.name,
                     'requested_frequency_hz': float(freq),
                     'frequency_min_hz': lo, 'frequency_max_hz': hi,
                     'sample_count': len(sig.x)},
            suggestion='Choose a frequency within the sweep or extend '
                       'the sweep to cover the requested point.',
        )
        return float('nan')
    if interpolate:
        i0, i1, t = _bracket_freq(sig.x, freq)
        idx = i0 if t < 0.5 else i1
        pair = [i0] if i0 == i1 else [i0, i1]
    else:
        idx = int(np.argmin(np.abs(sig.x - freq)))
        pair = [idx]
    bad = [j for j in pair
           if np.isnan(np.real(sig.y[j])) or np.isnan(np.imag(sig.y[j]))]
    if bad:
        emit(
            metric='phase_at_freq', code='nan_response',
            message='Phase unavailable: a selected neighbouring response '
                    'sample has a NaN real or imaginary component.',
            details={**_diagnostic_details(sig, bad),
                     'requested_frequency_hz': float(freq),
                     'response_real': float(np.real(sig.y[bad[0]])),
                     'response_imag': float(np.imag(sig.y[bad[0]]))},
            suggestion='Check the source result at the selected frequency.',
        )
    if interpolate and i0 != i1:
        p0 = float(np.degrees(np.angle(sig.y[i0])))
        p1 = float(np.degrees(np.angle(sig.y[i1])))
        d = (p1 - p0 + 180.0) % 360.0 - 180.0        # short arc, no wrap crossing
        if d == -180.0:
            d = 180.0
        ph = (p0 + t * d + 180.0) % 360.0 - 180.0
        return float(180.0 if ph == -180.0 else ph)
    return float(np.degrees(np.angle(sig.y[idx])))


@metric(domain='ac', unit='deg')
def phase_margin(sig: FrequencyResponse) -> float:
    """Phase margin in degrees: 180 + phase(f_unity_gain).

    Delegates to :func:`unity_gain_freq` for the crossing frequency, so
    it inherits the same rising/falling-axis support. Search and phase
    unwrapping use increasing frequency order; non-monotonic or repeated
    coordinates raise ValueError. With more than one 0 dB crossing, the margin is
    reported at the lowest-frequency crossing, and
    :func:`unity_gain_freq` emits the multiple-crossing
    ``silk.metrics.diagnostics.MetricWarning`` (naming every
    crossing and its margin) once, since this function calls it.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear loop gain (V/V). Real arrays are accepted but numpy.angle
        then uses only their real-axis phase (0 or 180 degrees); real
        magnitudes cannot recover the phase of a complex response.
        Real-valued input emits one MetricWarning about missing phase
        per call; the real-axis calculation is preserved. Pass the
        complex AC response H(jw) to retain phase information.
        The input is not modified.

    Returns
    -------
    float
        Phase margin in degrees. Returns NaN if unity-gain frequency is 0,
        or if it is ``nan`` — no observed unity crossing or a NaN magnitude.
        Crossing diagnostics originate from :func:`unity_gain_freq`, once. An
        unguarded ``argmin(|x - nan|)`` is 0, and the phase at the lowest
        sweep point read as a plausible ~180° margin.

    Notes
    -----
    Resolution: the phase is read at the grid sample NEAREST the
    (interpolated) unity-gain frequency, so the error is
    ``|dφ/d ln f| · ln(r)/2`` with ``r`` the grid ratio — for a
    dominant-pole loop ``|dφ/d ln f| ≈ 57° · f_p/f_u`` — negligible at
    200 pts/decade, up to a few degrees on a fast-rolling phase at 10.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'phase_margin')
    _warn_real_phase(sig, 'phase_margin')
    f_ugf = unity_gain_freq(sig)
    if f_ugf == 0.0 or np.isnan(f_ugf):
        return float('nan')
    # UNWRAP, exactly like gain_margin (2026-08-23): phase_at_freq reads the
    # wrapped principal value, so any loop whose true phase at unity gain is
    # past -180° — an UNSTABLE amplifier — came back shifted +360°: a true
    # PM of -62° reported as +298°, silently PASSING a `phase_margin >= 45`
    # spec. Verified live on a 3-pole loop. The remainder form maps the
    # result onto the standard (-180°, +180°] branch (same formula as
    # python-control's stability_margins).
    order = _axis(sig, 'phase_margin').indices    # TASK-309: unwrap
    x = np.asarray(sig.x, dtype=float)[order]              # must follow the
    y = np.asarray(sig.y)[order]                           # same order as
    idx = int(np.argmin(np.abs(x - f_ugf)))                # unity_gain_freq
    ph = float(np.degrees(np.unwrap(np.angle(y)))[idx])
    return float(np.remainder(ph, 360.0) - 180.0)


def gain_margins(sig: Signal):
    """Every gain margin of the loop, one per phase crossover.

    A phase crossover is a crossing of the NEGATIVE REAL AXIS: the unwrapped
    phase passing −180° + k·360° for any integer k, in either direction —
    an upward re-crossing (phase recovering back through −180°) is the
    signature of a conditionally stable loop and counts fully. The crossing
    point is interpolated (linear in phase, log-linear in frequency and
    magnitude) rather than snapped to a sample: at ``.ac dec 10`` density
    the sample-before-the-crossing value is off by several dB.

    Returns
    -------
    list of tuple
        ``(freq_Hz, gm_dB)`` pairs in ascending frequency.
        ``gm_dB = -20*log10(|L|)`` at the crossing: positive = stability
        lost by RAISING the gain that much, negative =
        lost by LOWERING it (conditionally stable loops have both kinds).
        If a bracketing magnitude is zero, magnitude interpolation is linear
        rather than logarithmic. A resulting zero crossing magnitude gives
        ``gm_dB = +inf`` at that crossing; other crossings are unaffected.
        Empty list when the unwrapped phase never touches a crossover level
        inside the sweep. ``[(nan, nan)]`` — a one-element sentinel list,
        never confused with ``[]`` — when ``sig.y`` contains a NaN sample: unwrapping a NaN phase makes it NaN from that sample
        on, and the crossover-level search (``int(np.ceil(...))`` on that
        NaN) would otherwise raise instead of reporting a corner that
        simply couldn't be measured. The list analogue of ``crossings``'
        ``array([nan])`` and of ``phase_margin``'s scalar ``nan``.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear loop gain (V/V). Real arrays are accepted but numpy.angle
        then uses only their real-axis phase (0 or 180 degrees); real
        magnitudes cannot recover the phase of a complex response.
        Real-valued input emits one MetricWarning about missing phase
        per call; the real-axis calculation is preserved. Pass the
        complex AC response H(jw) to retain phase information.
        The input is not modified.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For a consumed NaN response, returning [(nan, nan)]. An empty
        crossing list emits no crossing warning.

    Notes
    -----
    Phase unwrapping starts on the principal branch at the lowest
    frequency. Begin the sweep below phase crossover; crossings below
    the measured range cannot be recovered. A sweep beginning exactly
    on a crossover level does not count that initial point.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'gain_margins')
    _warn_real_phase(sig, 'gain_margins')
    return _gain_margins(sig)


def _gain_margins(sig: Signal):
    """Calculate crossings after public input checks and phase diagnostics."""
    order = _axis(sig, 'gain_margins').indices
    f = np.asarray(sig.x, dtype=float)[order]
    y = np.asarray(sig.y)[order]
    if _any_nan(y):                            # guard on the complex samples: see passband_ripple_db
        emit(
            metric='gain_margins', code='nan_magnitude',
            message='Gain margins unavailable: consumed magnitude data contains '
                    'NaNs; phase crossings cannot be enumerated.',
            details=_diagnostic_details(sig),
            suggestion='Check the source result for the affected samples.',
        )
        return [(float('nan'), float('nan'))]
    mag = np.abs(y)
    ph = np.degrees(np.unwrap(np.angle(y)))

    # Anchor assumption, stated because it is UNDETECTABLE from the data:
    # absolute phase from complex y is only known modulo 360°, and unwrap
    # anchors on the principal value at the lowest frequency — always in
    # (-180°, +180°]. A sweep that starts with the loop's true phase already
    # past -180° is indistinguishable from one that hasn't got there yet, so
    # crossings below the sweep are silently invisible. Start loop-gain
    # sweeps well below crossover (the universal practice) and this holds.

    crossings = []
    for i in range(len(ph) - 1):
        a, b = ph[i], ph[i + 1]
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        # Levels -180 + 360k inside [lo, hi]; a level equal to the LEFT
        # endpoint is excluded (it was this crossing's right endpoint one
        # pair earlier — and a sweep STARTING exactly on a level is not a
        # crossing, per the warning above).
        k_min = int(np.ceil((lo + 180.0) / 360.0))
        k_max = int(np.floor((hi + 180.0) / 360.0))
        for k in range(k_min, k_max + 1):
            level = -180.0 + 360.0 * k
            if level == a:
                continue
            t = (level - a) / (b - a)
            if f[i] > 0 and f[i + 1] > 0:
                fc = float(f[i] * (f[i + 1] / f[i]) ** t)
            else:
                fc = float(f[i] + t * (f[i + 1] - f[i]))
            if mag[i] > 0 and mag[i + 1] > 0:
                mc = float(mag[i] ** (1.0 - t) * mag[i + 1] ** t)
            else:
                mc = float(mag[i] + t * (mag[i + 1] - mag[i]))
            gm = float('inf') if mc == 0.0 else float(-20.0 * np.log10(mc))
            crossings.append((fc, gm))
    crossings.sort(key=lambda c: c[0])
    return crossings


@metric(domain='ac', unit='dB')
def gain_margin(sig: FrequencyResponse) -> float:
    """Gain margin in dB — the crossover margin CLOSEST TO 0 dB.

    "The smallest multiplier of the loop gain that makes the system
    unstable" (Åström & Murray §10.3): of every negative-real-axis crossing
    (see :func:`gain_margins`), the binding one is the one with the smallest
    ``|margin|`` in dB — the convention python-control's
    ``stability_margins`` implements. NOT the first
    crossing (2026-08-23: on a conditionally stable amplifier the first
    crossing reported −102 dB where the real constraints were −17.6/+40.4),
    and not the crossing nearest the unity-gain frequency (a high-Q
    parasitic resonance near 0 dB far above the UGF is exactly the binding
    case that rule misses). Ties break toward the lowest frequency. For a
    conditionally stable loop the single number is inherently lossy — use
    :func:`gain_margins` for both limits.

    Parameters
    ----------
    sig : Signal
        Frequency response with equal-length one-dimensional ``x`` and ``y``
        arrays of shape (N,), N >= 2, and ``xlabel='frequency'``. Coordinates
        are finite Hz, strictly increasing or decreasing. Values are complex
        linear loop gain (V/V). Real arrays are accepted but numpy.angle
        then uses only their real-axis phase (0 or 180 degrees); real
        magnitudes cannot recover the phase of a complex response.
        Real-valued input emits one MetricWarning about missing phase
        per call; the real-axis calculation is preserved. Pass the
        complex AC response H(jw) to retain phase information.
        The input is not modified.

    Returns
    -------
    float
        Gain margin in dB. With no crossover in the sweep: ``+inf`` when
        ``|gain|`` has fallen below 1 by the top of the sweep (the loop rolled
        off before its phase could invert — genuinely unbounded margin;
        NaN here made ``gain_margin(sig) >= 10`` FAIL the safest possible
        loop), ``NaN`` when ``|gain|`` is still >= 1 at the top (the sweep
        ended before the answer was knowable — extend it). Also ``NaN``
        when ``sig.y`` contains a NaN sample: :func:`gain_margins`
        then returns its ``[(nan, nan)]`` sentinel, which is NOT the
        no-crossover case above — mistaking it for one would silently
        report the unbounded-margin ``+inf`` for a corner whose margin is
        simply unknown.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For consumed NaNs (originating in gain_margins) or no phase
        crossover while the highest-frequency gain remains >= 1.
        Both return nan. The +inf limit emits no crossover warning.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
    """
    _check_freq(sig, 'gain_margin')
    _warn_real_phase(sig, 'gain_margin')
    crossings = _gain_margins(sig)
    if len(crossings) == 1 and _any_nan(crossings[0]):
        return float('nan')
    if not crossings:
        order = _axis(sig, 'gain_margin').indices
        mag_top = float(np.abs(np.asarray(sig.y)[order][-1]))
        if mag_top >= 1.0:
            emit(
                metric='gain_margin', code='no_phase_crossing_gain_at_least_one',
                message='No phase crossover observed in the supplied sweep, '
                        'and the highest-frequency magnitude remains at/above 1.',
                details={**_diagnostic_details(sig), 'endpoint_magnitude': mag_top},
                suggestion='Try extending the upper sweep limit; a phase '
                           'crossover outside this range is not established.',
            )
        return float('inf') if mag_top < 1.0 else float('nan')
    finite = [c for c in crossings if np.isfinite(c[1])]
    if not finite:
        return float('inf')
    best = min(finite, key=lambda c: (abs(c[1]), c[0]))
    return best[1]


@metric(domain='ac', unit='dB', param_units={'freq': 'Hz'})
def cmrr_dB(vdiff: FrequencyResponse, vcm: FrequencyResponse, freq: Optional[float] = None) -> float:
    """CMRR in dB: 20*log10(``|diff_gain|`` / ``|cm_gain|``) at one frequency.

    *freq* None (the default) measures at each sweep's LOWEST frequency —
    the low-frequency CMRR. Before 2026-08-23 this indexed sample 0 of each
    array, which is the same thing whenever the sweep is in ascending order
    (the normal case) but was an accident of ordering, not a choice; the
    frequency is now explicit and order-independent. Magnitudes only —
    relative phase between the two gains is deliberately ignored.

    Parameters
    ----------
    vdiff : Signal
        Differential linear gain (V/V), real or complex, with
        ``xlabel='frequency'`` and equal-length one-dimensional x/y arrays
        of shape (N,), N >= 2. Finite Hz coordinates are strictly monotonic.
    vcm : Signal
        Common-mode linear gain (V/V) with the same shape/axis requirements.
        Its grid may differ from vdiff. Neither input is modified.
    freq : float, optional
        Finite query frequency in Hz. Selects the nearest sample independently
        in each sweep; None selects each sweep's lowest frequency.

    Returns
    -------
    float
        CMRR in dB. A nonzero differential gain with exactly zero
        common-mode gain returns ``+inf``: infinite rejection in the
        supplied response, not a claim of measured physical dynamic range.
        If both gains are zero, returns ``nan`` (unavailable data under
        the metric failure convention: no differential reference), allowing sweeps to continue.
        An explicit frequency strictly outside either sweep also returns
        ``nan``, with a ``MetricWarning`` naming the request and both ranges.
        In-range nearest-sample selection and the default lowest-frequency
        selection are unchanged. Warning filters apply normally.

    Raises
    ------
    ValueError
        For an empty or fewer-than-two-point sweep, unequal array lengths,
        non-finite or non-monotonic frequency coordinates. A wrong axis label
        raises ``SignalKindError``, a subclass of ``ValueError``.
        A non-finite explicitly requested frequency is also rejected.

    Warns
    -----
    silk.metrics.diagnostics.MetricWarning
        For a NaN in either selected sample, two zero gains, or an explicit
        frequency outside either sweep, returning nan. Zero differential
        gain with nonzero common-mode gain returns -inf with a NumPy
        RuntimeWarning from log10.
    """
    _check_freq(vdiff, 'cmrr_dB')
    _check_freq(vcm, 'cmrr_dB')
    if freq is None:
        di = int(np.argmin(vdiff.x))
        ci = int(np.argmin(vcm.x))
    else:
        if not np.isfinite(freq):
            raise ValueError(f'cmrr_dB: freq must be finite, got {freq!r} Hz')
        diff_lo, diff_hi = float(np.min(vdiff.x)), float(np.max(vdiff.x))
        cm_lo, cm_hi = float(np.min(vcm.x)), float(np.max(vcm.x))
        if freq < diff_lo or freq > diff_hi or freq < cm_lo or freq > cm_hi:
            emit(
                metric='cmrr_dB', code='frequency_outside_sweep',
                message='Requested frequency is outside one or both supplied sweeps.',
                details={'requested_frequency_hz': float(freq),
                         'vdiff_frequency_min_hz': diff_lo,
                         'vdiff_frequency_max_hz': diff_hi,
                         'vdiff_sample_count': len(vdiff.x),
                         'vcm_frequency_min_hz': cm_lo,
                         'vcm_frequency_max_hz': cm_hi,
                         'vcm_sample_count': len(vcm.x)},
                suggestion='Choose a frequency covered by both sweeps or extend '
                           'the affected sweep to cover the requested point.',
            )
            return float('nan')
        di = int(np.argmin(np.abs(vdiff.x - freq)))
        ci = int(np.argmin(np.abs(vcm.x - freq)))
    # complex-aware (TASK-178): checking the magnitudes would miss e.g.
    # inf+nanj, whose magnitude folds to inf — see passband_ripple_db
    contaminated = _any_nan(vdiff.y[di]) or _any_nan(vcm.y[ci])
    diff_mag = float(np.abs(vdiff.y[di]))
    cm_mag   = float(np.abs(vcm.y[ci]))
    if contaminated or (diff_mag == 0.0 and cm_mag == 0.0):
        emit(
            metric='cmrr_dB',
            code=('nan_magnitude' if contaminated else 'no_differential_reference'),
            message=('CMRR unavailable: a selected differential/common-mode '
                     'magnitude is NaN.' if contaminated else
                     'CMRR unavailable: both selected differential and common-mode '
                     'magnitudes are zero; there is no differential reference.'),
            details={
                **{f'differential_{key}': value
                   for key, value in _diagnostic_details(vdiff, [di]).items()},
                **{f'common_mode_{key}': value
                   for key, value in _diagnostic_details(vcm, [ci]).items()},
                'requested_frequency_hz': freq,
                'differential_magnitude': diff_mag, 'common_mode_magnitude': cm_mag,
            },
            suggestion=('Check the source result at the selected frequencies.'
                        if contaminated else
                        'Check the differential excitation and operating point.'),
        )
    if contaminated:
        # TASK-178: an inf-paired-with-nan sample folds to an infinite,
        # not NaN, magnitude, so diff_mag/cm_mag would no longer propagate
        # a NaN on their own -- return NaN explicitly.
        return float('nan')
    if cm_mag == 0.0:
        return float('inf') if diff_mag > 0 else float('nan')
    return float(20.0 * np.log10(diff_mag / cm_mag))
