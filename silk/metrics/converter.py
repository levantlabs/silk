"""Data-converter static linearity, effective resolution and jitter limits.

Static metrics measure code-transition curves (xlabel='code'), not simulator
DC sweeps. Transition extraction builds that curve from a ramp capture. ENOB
uses a clock-sampled time waveform and the spectral SNDR calculation; the
aperture-jitter SNR limit takes scalar frequency and jitter values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math

import numpy as np

from silk.signals.base import Signal
from silk.signals.schema import metric, Transient, CodeTransitions
from silk.metrics._guards import _nan_safe_transform, _axis, _check_nonempty, _any_nan
from silk.metrics.diagnostics import emit
from silk.metrics.spectral import SpectrumWindow, sndr
from silk.metrics.waveform import _check_transient, _observed_range


@metric(domain='tran', unit='bits',
        param_units={'tclk': 's', 'tstart': 's', 'fundamental_hz': 'Hz',
                     'full_scale': '[y]'})
def enob(sig: Transient, *, tclk: float, tstart: float = 0.0,
         n_samples: Optional[int] = None,
         fundamental_hz: Optional[float] = None,
         guard_bins: int = 0,
         window: Optional[SpectrumWindow] = 'rectangular',
         full_scale: Optional[float] = None) -> float:
    """Effective number of bits = (SNDR − 1.76) / 6.02.

    Takes :func:`sndr`'s spectrum options. With ``full_scale`` the SNDR
    is referred to full scale, so a backed-off input is credited with
    the headroom it left: ``(SINAD + 20·log10(full_scale/A) − 1.76) /
    6.02`` for a tone of peak ``A``.

    Resolution: inherits :func:`sndr`'s — with a random noise floor
    ≈ 0.72/√(n_samples/2) bits (0.09 bit at 128 samples, 0.03 bit at
    1024). Quote ENOB to no more decimals than that.

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
        Effective bits, (sndr - 1.76) / 6.02. Unavailable SNDR gives nan;
        constant data without full_scale retains the negative floor value
        (e.g. about -3.8 bits for 256 samples).

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
    _check_nonempty(sig, 'enob')
    _check_transient(sig, 'enob')
    try:
        sndr_db = sndr(sig, tclk=tclk, tstart=tstart, n_samples=n_samples,
                       fundamental_hz=fundamental_hz, guard_bins=guard_bins,
                       window=window, full_scale=full_scale)
    except ValueError as exc:
        raise ValueError(f'enob: {exc}') from exc
    return (sndr_db - 1.76) / 6.02


@dataclass(frozen=True)
class INL:
    """Result of an INL (integral non-linearity) measurement.

    Attributes
    ----------
    inl_lsb : numpy.ndarray
        Per-code INL in LSBs, shape (N,), one entry per transition.
    max_abs_lsb : float
        Maximum absolute INL in LSBs; nan if any entry is unavailable.
    method : str
        ``'endpoint'`` or ``'best_fit'``.

    Parameters
    ----------
    inl_lsb : numpy.ndarray
        Per-code INL in LSBs, shape (N,), one entry per transition.
    max_abs_lsb : float
        Maximum absolute INL in LSBs; nan if any entry is unavailable.
    method : str
        ``'endpoint'`` or ``'best_fit'``.

    Returns
    -------
    INL
        Frozen result container. Construction stores supplied fields without
        validation or copying; nan fields represent unavailable measurements.

    Notes
    -----
    Field rebinding is prohibited by the frozen dataclass. Array fields,
    where present, remain mutable and are shared with constructor arguments.
    The constructor emits no MetricWarning; the measuring function owns
    validation and diagnostics.
    """
    inl_lsb: np.ndarray
    max_abs_lsb: float
    method: str


@dataclass(frozen=True)
class DNL:
    """Result of a DNL (differential non-linearity) measurement.

    Attributes
    ----------
    dnl_lsb : numpy.ndarray
        Per-step DNL in LSBs, shape (N-1,) for N transition voltages.
    max_abs_lsb : float
        Maximum absolute DNL in LSBs; nan if any entry is unavailable.

    Parameters
    ----------
    dnl_lsb : numpy.ndarray
        Per-step DNL in LSBs, shape (N-1,) for N transition voltages.
    max_abs_lsb : float
        Maximum absolute DNL in LSBs; nan if any entry is unavailable.

    Returns
    -------
    DNL
        Frozen result container. Construction stores supplied fields without
        validation or copying; nan fields represent unavailable measurements.

    Notes
    -----
    Field rebinding is prohibited by the frozen dataclass. Array fields,
    where present, remain mutable and are shared with constructor arguments.
    The constructor emits no MetricWarning; the measuring function owns
    validation and diagnostics.
    """
    dnl_lsb: np.ndarray
    max_abs_lsb: float


@dataclass(frozen=True)
class StaticErrors:
    """Result of an offset / gain / full-scale error measurement.

    Attributes
    ----------
    offset_error_lsb : float
        First measured transition minus its ideal, in ideal LSBs.
    gain_error_lsb : float
        Measured span (last minus first transition) minus the ideal span,
        in ideal LSBs.
    full_scale_error_lsb : float
        Last measured transition minus its ideal, in ideal LSBs. Equals
        ``offset_error_lsb + gain_error_lsb``.

    Parameters
    ----------
    offset_error_lsb : float
        First measured transition minus its ideal, in ideal LSBs.
    gain_error_lsb : float
        Measured span (last minus first transition) minus the ideal span,
        in ideal LSBs.
    full_scale_error_lsb : float
        Last measured transition minus its ideal, in ideal LSBs. Equals
        ``offset_error_lsb + gain_error_lsb``.

    Returns
    -------
    StaticErrors
        Frozen result container. Construction stores supplied fields without
        validation or copying; nan fields represent unavailable measurements.

    Notes
    -----
    Field rebinding is prohibited by the frozen dataclass. The constructor
    emits no MetricWarning; static_errors owns validation and diagnostics.
    """
    offset_error_lsb: float
    gain_error_lsb: float
    full_scale_error_lsb: float


def _check_code_signal(sig: Signal, metric: str) -> None:
    if sig.xlabel != 'code':
        raise ValueError(
            f"{metric}: sig must be a code-transition Signal (xlabel='code'), "
            f"got xlabel={sig.xlabel!r}"
        )
    _check_nonempty(sig, metric)
    if len(sig.x) != len(sig.y):
        raise ValueError(
            f'{metric}: codes and transition voltages must have the same length, '
            f'got {len(sig.x)} codes and {len(sig.y)} voltages'
        )
    _axis(sig, metric, order='increasing')
    if len(sig.x) < 2:
        raise ValueError(f'{metric}: need at least 2 codes, got {len(sig.x)}')
    codes = np.asarray(sig.x, dtype=float)
    frac = np.flatnonzero(~np.isfinite(codes) | (codes != np.floor(codes)))
    if frac.size:
        raise ValueError(
            f'{metric}: codes must be integers; found {codes[frac[0]]!r} '
            f'at index {int(frac[0])}')


def _ideal_lsb(codes: np.ndarray, transitions: np.ndarray) -> float:
    """Endpoint-line LSB: the ideal step size implied by the code span.

    ``(V[last] - V[first]) / (code[last] - code[first])`` — an LSB per
    CODE, not per sample. The two coincide only when ``codes`` are
    consecutive integers (sample count - 1 == code span); for sparse or
    unevenly spaced codes they differ, so callers must not substitute
    ``len(transitions) - 1`` for the code span.
    """
    n = len(transitions)
    if n < 2:
        raise ValueError(f'need at least 2 codes, got {n}')
    return (transitions[-1] - transitions[0]) / (codes[-1] - codes[0])


def _check_consecutive_codes(codes: np.ndarray, metric: str) -> None:
    """Raise if ``codes`` are not consecutive integers (caller error).

    DNL is defined per code and needs every code present between the
    endpoints; a gap means some code was never sampled, which is a
    malformed request for this metric, not degenerate converter data. :func:`inl` has no such requirement and is the
    suggested metric for sparse/non-consecutive codes.
    """
    gaps = np.diff(codes)
    bad = np.flatnonzero(gaps != 1)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f'{metric}: codes must be consecutive integers (DNL is defined '
            f'per code and needs every code present); found a gap of '
            f'{gaps[i]:g} between code {codes[i]:g} and code {codes[i + 1]:g} '
            f'at index {i}. Use inl for sparse or non-consecutive codes.'
        )


def _emit_zero_lsb(metric: str, transitions: np.ndarray) -> None:
    """Explain zero normalization without assuming equal endpoints are flat."""
    details = {'transition_count': len(transitions), 'lsb_v': 0.0}
    if np.all(transitions == transitions[0]):
        voltage = float(transitions[0])
        details['transition_voltage_v'] = voltage
        code = 'constant_transfer_curve'
        message = (f'The transfer curve is constant: {len(transitions)} '
                   f'transitions at {voltage:g} V; the derived LSB is zero. '
                   'Linearity is unavailable.')
    else:
        code = 'zero_derived_lsb'
        details['first_transition_voltage_v'] = float(transitions[0])
        details['last_transition_voltage_v'] = float(transitions[-1])
        message = 'The derived LSB is zero; linearity cannot be normalized.'
    emit(metric=metric, code=code, message=message, details=details,
         suggestion='Check the source transition voltages and ramp coverage.')


def _emit_nan_transitions(sig: Signal, metric: str) -> None:
    """Identify each unavailable transition without changing reference policy."""
    voltages = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    nan_indices = np.flatnonzero(np.isnan(voltages))
    if not nan_indices.size:
        return
    codes = np.asarray(sig.x, dtype=float)
    voltage_min, voltage_max = _observed_range(voltages)
    for index in nan_indices:
        code = float(sig.x[index])
        details = {
            'code': code, 'transition_index': int(index),
            'transition_count': len(voltages),
            'nan_transition_count': int(nan_indices.size),
            'code_min': float(np.min(codes)), 'code_max': float(np.max(codes)),
            'transition_voltage_unit': 'V',
            'transition_voltage_min_v': voltage_min,
            'transition_voltage_max_v': voltage_max,
        }
        effect = 'Linearity measurements depending on it are unavailable.'
        if metric in ('dnl', 'missing_codes'):
            widths = []
            for side, start in (('left', index - 1), ('right', index)):
                if 0 <= start < len(codes) - 1:
                    lo, hi = float(codes[start]), float(codes[start + 1])
                    details[f'{side}_width_from_code'] = lo
                    details[f'{side}_width_to_code'] = hi
                    widths.append(f'code {lo:g} to code {hi:g}')
            details['affected_width_count'] = len(widths)
            effect = ('Adjacent transition widths (V) are unavailable: '
                      + '; '.join(widths) + '.')
            if index == 0 or index == len(codes) - 1:
                effect += (' This endpoint also makes the shared LSB '
                           'unavailable, so every DNL interval is unavailable.')
            if metric == 'missing_codes':
                effect += ' Missing-code enumeration is unavailable.'
        emit(
            metric=metric, code='nan_transition_voltage',
            message=(f'The transition voltage (V) at code {code:g} is NaN. '
                     + effect),
            details=details,
            suggestion='Check the source result at this code transition.',
        )


def extract_code_transitions(v_in: Signal, code: Signal) -> Signal:
    """Build a code→voltage transition Signal from a slow-ramp capture.

    Per IEEE Std 1241-2010 transition-level semantics, the transition
    level T[k] is the input voltage at which the converter switches
    FROM code k−1 INTO code k. Transition levels exist only for
    k = lowest_observed_code+1 .. max_observed_code — the lowest
    observed code has no transition (there is no lower code to
    transition from) and never gets a row. For each such k, this
    returns the ``v_in`` voltage at the first sample where the code
    has REACHED k (``code_int >= k``). Returns a Signal with
    ``xlabel='code'``, ``x = lowest+1 .. max`` (every integer in the
    span, one row per code), ``y = transition_voltages`` — ready for
    :func:`inl`, :func:`dnl`, :func:`is_monotonic`, :func:`missing_codes`.
    The endpoint LSB (see :func:`inl`/:func:`dnl`) is then the code
    span, ``(T[max]-T[lowest+1]) / (max-lowest-1)``.

    A code the converter skips over still gets a row: the first
    sample reaching k+1 also reaches k, so it shares its successor's
    voltage. That reads as a zero-width step, so ``DNL[k] == -1.0``
    exactly and :func:`missing_codes` reports it. (IEEE 1241 uses a
    −0.9 LSB threshold there for noise robustness against a
    histogram-estimated transition; :func:`missing_codes` keeps
    ≤ −1.0 since these transitions are read directly rather than
    estimated.)

    Resolution: one ramp step — each transition is the first SAMPLE at
    or after the true level, so it reads high by up to
    ``slope × Δt``; INL/DNL inherit that divided by the LSB.

    Time grids must agree within 1e-3 of the smallest sample spacing;
    singleton grids must agree exactly. A mismatch raises ``ValueError``.

    Raises ``ValueError`` if either input Signal is empty —
    an empty capture is a malformed request, not a converter with no
    transitions. A one-sample capture still returns an empty (no-row)
    transition Signal: one observed code is the lowest, and IEEE 1241
    transition levels start at lowest+1.

    NaN handling: a NaN in ``code.y`` cannot be rounded to a
    code (``rint(nan).astype(int)`` is platform garbage, and used to make
    this return an EMPTY Signal — "no transitions", a passing-looking
    answer). Instead the code axis is built from the finite codes that
    were observed and EVERY transition voltage is ``nan``: the
    first-sample-reaching-k search is unreliable with a NaN anywhere in
    the trace, so no level is knowable. If no finite code exists at all,
    or the finite codes span a single code, a one-row ``nan`` Signal is
    returned (x = y = nan) so the result is never mistaken for a real
    "no rows" table. A NaN in ``v_in.y`` is a point read: it propagates
    into each transition that samples it and nowhere else. Such a selected
    voltage emits ``MetricWarning`` naming the code and sample time (s);
    voltages never selected as transitions and empty enumerations stay quiet.

    A code that goes back down. The first-reach table cannot
    show a dip: "first sample with code >= k" is nondecreasing in k by
    construction, so output codes 0,1,2,3,2,3,4,5 on a rising ramp give
    the same table as a clean staircase. The returned Signal therefore
    also carries ``code_reversals``: the number of adjacent sample pairs
    at which the rounded output code FELL (``code[i+1] < code[i]``),
    counted once per falling step whatever its size. A non-monotonic
    converter is data, so this is recorded, never raised.
    :func:`is_monotonic` and :func:`non_monotonic_steps` add it to the
    table's own count; :func:`inl`, :func:`dnl` and :func:`missing_codes`
    read the table only, so a dip moves none of them. A NaN code trace
    sets ``code_reversals = nan`` (unknown). The extractor assumes a
    rising ramp: comparator noise that flickers the code at a transition
    also counts as a reversal, which it is in the captured trace.
    ``code_reversals`` lives on the Signal this returns; a Signal derived
    from it (arithmetic, a rebuilt Signal) does not carry it and is judged
    on its voltages alone, so pass this output to :func:`is_monotonic` and
    :func:`non_monotonic_steps` directly.

    Parameters
    ----------
    v_in : Signal
        Ramp voltage in V with ``xlabel='time'`` and equal-length
        one-dimensional x/y arrays of shape (N,), N >= 1. Coordinates are
        finite, strictly increasing seconds.
    code : Signal
        Dimensionless output code on the same time grid. Values are rounded
        with numpy.rint (ties to even). Neither input is modified.

    Returns
    -------
    Signal
        New transition table: shape (M,) code coordinates and voltages in V,
        with type voltage, xlabel code, ylabel V and code_reversals metadata.
        M = maximum rounded code - minimum rounded code, including skipped
        codes. A constant capture gives M = 0; NaN-code sentinels are
        described above. NaN code traces emit no MetricWarning.

    Raises
    ------
    ValueError
        For wrong axis labels, empty captures, unequal x/y lengths,
        non-finite or non-increasing time axes, or different time grids.
    """
    if v_in.xlabel != 'time':
        raise ValueError(
            f"extract_code_transitions: v_in must be a transient "
            f"(xlabel='time'), got {v_in.xlabel!r}"
        )
    if code.xlabel != 'time':
        raise ValueError(
            f"extract_code_transitions: code must be a transient "
            f"(xlabel='time'), got {code.xlabel!r}"
        )
    _check_nonempty(v_in, 'extract_code_transitions')
    _check_nonempty(code, 'extract_code_transitions')
    input_axis = _axis(v_in, 'extract_code_transitions', order='increasing')
    code_axis = _axis(code, 'extract_code_transitions', order='increasing')
    if not input_axis.same_grid(code_axis):
        raise ValueError(
            'extract_code_transitions: v_in and code must share the same time axis; '
            f'v_in has {len(v_in.x)} points spanning {v_in.x[0]:g} .. '
            f'{v_in.x[-1]:g} s; code has {len(code.x)} points spanning '
            f'{code.x[0]:g} .. {code.x[-1]:g} s; coordinates differ'
        )
    code_y = _nan_safe_transform(code.y, lambda y: y.astype(float))
    if _any_nan(code_y):
        finite = code_y[np.isfinite(code_y)]
        levels = np.array([], dtype=float)
        if finite.size:
            lo, hi = int(np.rint(finite.min())), int(np.rint(finite.max()))
            levels = np.arange(lo + 1, hi + 1).astype(float)
        if levels.size == 0:
            levels = np.array([float('nan')])
        result = Signal(
            f'transitions({v_in.name})', 'voltage', 'code', 'V',
            levels, np.full(len(levels), float('nan')),
        )
        result.code_reversals = float('nan')
        return result
    code_int = np.rint(code_y).astype(int)
    lowest, highest = int(code_int.min()), int(code_int.max())
    levels_int = np.arange(lowest + 1, highest + 1)
    transitions = np.empty(len(levels_int), dtype=float)
    for i, k in enumerate(levels_int):
        idx = int(np.argmax(code_int >= k))
        transitions[i] = float(_nan_safe_transform(v_in.y[idx], lambda y: y.astype(float)))
        if np.isnan(transitions[i]):
            emit(
                metric='extract_code_transitions', code='nan_transition_voltage',
                message=(f'The input voltage (V) selected at sample {idx} '
                         f'for the transition into code {k} is NaN; '
                         'this transition voltage is unavailable.'),
                details={
                    'code': float(k), 'transition_index': i,
                    'transition_count': len(levels_int),
                    'transition_voltage_unit': 'V',
                    'sample_index': idx, 'sample_time_s': float(v_in.x[idx]),
                    'sample_count': len(v_in.x),
                    'time_min_s': float(v_in.x[0]),
                    'time_max_s': float(v_in.x[-1]),
                },
                suggestion='Check the source input voltage at this code transition.',
            )
    levels = levels_int.astype(float)
    result = Signal(
        f'transitions({v_in.name})', 'voltage', 'code', 'V',
        levels, transitions,
    )
    result.code_reversals = int(np.count_nonzero(np.diff(code_int) < 0))
    return result


@metric(domain='dc', outputs={'max_abs_lsb': 'LSB'})
def inl(sig: CodeTransitions, method: Literal['endpoint', 'best_fit'] = 'endpoint') -> INL:
    """Integral non-linearity per code, in LSBs.

    ``endpoint`` draws the ideal line through the first and last
    measured code, with slope ``(V[last]-V[first]) / (code[last]-code[first])``
    — an LSB per code, so non-consecutive or unevenly spaced codes are
    fine; ``best_fit`` uses a least-squares fit against the codes
    (IEEE 1241 §4.4).

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
    method : {'endpoint', 'best_fit'}, default 'endpoint'
        Reference-line construction: endpoint line or least-squares line.

    Notes
    -----
    A NaN transition voltage gives ``inl_lsb`` all-``nan`` and
    ``max_abs_lsb = nan`` for either method. The endpoint line
    propagated it arithmetically already; the guard exists so the
    ``best_fit`` path never reaches ``numpy.polyfit`` with a NaN, which
    raises ``LinAlgError`` on some LAPACK builds and returns ``nan`` on
    others — a NaN is data-dependent and must never raise.

    A constant transfer curve or zero derived LSB also returns all-NaN
    numeric fields, with a ``silk.metrics.diagnostics.MetricWarning``.
    Missing transition voltages warn with the affected code. NaN code
    coordinates, invalid axes and unknown methods raise ``ValueError``.

    Returns
    -------
    INL
        Per-code INL array of shape (N,) and maximum absolute INL in LSBs,
        plus the requested method. NaN voltages or zero LSB give all-nan
        numeric fields with MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        An unknown method is also rejected.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals import Signal
    >>> from silk.metrics import inl, dnl
    >>> table = Signal('transitions', 'voltage', 'code', 'V',
    ...                np.array([1., 2., 3.]), np.array([.25, .5, .75]))
    >>> inl(table).max_abs_lsb
    0.0
    >>> dnl(table).dnl_lsb.tolist()
    [0.0, 0.0]
    """
    _check_code_signal(sig, 'inl')
    if method not in ('endpoint', 'best_fit'):
        raise ValueError(
            f"inl: method must be 'endpoint' or 'best_fit', got {method!r}"
        )
    v = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    codes = np.asarray(sig.x, dtype=float)
    if _any_nan(v):
        _emit_nan_transitions(sig, 'inl')
        return INL(np.full(len(v), float('nan')), float('nan'), method)
    # A nonzero constant can acquire a tiny nonzero slope from polyfit.
    if np.isfinite(v[0]) and np.all(v == v[0]):
        _emit_zero_lsb('inl', v)
        return INL(np.full(len(v), float('nan')), float('nan'), method)
    if method == 'endpoint':
        lsb = _ideal_lsb(codes, v)
        ideal = v[0] + (codes - codes[0]) * lsb
    else:
        slope, intercept = np.polyfit(codes, v, 1)
        ideal = intercept + slope * codes
        lsb = float(slope)
    if lsb == 0:
        _emit_zero_lsb('inl', v)
        return INL(np.full(len(v), float('nan')), float('nan'), method)
    inl_arr = (v - ideal) / lsb
    return INL(inl_arr, float(np.max(np.abs(inl_arr))), method)


@metric(domain='dc', outputs={'offset_error_lsb': 'LSB', 'gain_error_lsb': 'LSB',
                              'full_scale_error_lsb': 'LSB'})
def static_errors(sig: CodeTransitions, *, full_scale: float, n_bits: int, v_low: float = 0.0) -> StaticErrors:
    """Offset, gain and full-scale error from code transitions, in LSBs.

    Ideal-transition convention: the ideal LSB is ``full_scale / 2**n_bits``
    and the ideal transition into code ``k`` is at ``v_low + k * LSB``
    volts, where ``v_low`` (default 0 V) is the input voltage at the bottom
    of the range, so a range not starting at 0 V or a bipolar range is
    expressed by ``v_low``. The half-LSB first-transition convention is
    ``v_low`` shifted down by half an LSB (``v_low = v_bottom - LSB / 2``).
    Errors are
    measured transition minus ideal transition, divided by the ideal LSB:

    - offset error: first transition against its ideal;
    - gain error: measured span (last minus first transition) against the
      ideal span ``(code[last] - code[first]) * LSB``;
    - full-scale error: last transition against its ideal
      (``offset + gain``).

    Codes must be strictly ascending integers (every ``dc`` code-table metric
    enforces this) and, here, within ``[0, 2**n_bits - 1]``; otherwise
    ``ValueError`` names the offending code.

    The ideal is set by ``full_scale`` and ``n_bits``, not by the measured
    curve, so a table covering only part of the code range is fine.

    A NaN transition voltage returns all-``nan`` fields with a
    ``silk.metrics.diagnostics.MetricWarning``. A
    non-finite or non-positive ``full_scale``, a non-integer or
    non-positive ``n_bits``, a non-finite or non-real ``v_low`` and a
    malformed table raise ``ValueError``. Infinite ``n_bits`` instead
    raises ``OverflowError`` during integer conversion.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
    full_scale : float
        Finite positive input-voltage span in V; ideal LSB = full_scale / 2**n_bits.
    n_bits : int
        Positive converter resolution in bits. Codes must lie in
        [0, 2**n_bits - 1].
    v_low : float, default 0.0
        Finite real input voltage in V at the bottom of the ideal range.

    Returns
    -------
    StaticErrors
        Offset, gain and full-scale errors in ideal LSBs. A NaN transition
        gives three nan fields with MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Invalid full_scale, n_bits or v_low, or codes outside the specified
        converter range, are also rejected. NaN voltages return before
        the code-range check.
    OverflowError
        Infinite n_bits fails during integer conversion.
    """
    _check_code_signal(sig, 'static_errors')
    if not (np.isfinite(full_scale) and full_scale > 0):
        raise ValueError(
            f'static_errors: full_scale must be a finite positive voltage span, '
            f'got {full_scale!r}')
    if isinstance(n_bits, bool) or int(n_bits) != n_bits or n_bits < 1:
        raise ValueError(
            f'static_errors: n_bits must be a positive integer, got {n_bits!r}')
    if (isinstance(v_low, (bool, complex)) or not isinstance(v_low, (int, float, np.integer, np.floating))
            or not np.isfinite(v_low)):
        raise ValueError(
            f'static_errors: v_low must be a finite real voltage, got {v_low!r}')
    v = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    codes = np.asarray(sig.x, dtype=float)
    nan = float('nan')
    if _any_nan(v):
        _emit_nan_transitions(sig, 'static_errors')
        return StaticErrors(nan, nan, nan)
    top = 2 ** int(n_bits) - 1
    out = np.flatnonzero((codes < 0) | (codes > top))
    if out.size:
        raise ValueError(
            f'static_errors: code {int(codes[out[0]])} is outside '
            f'[0, {top}] for n_bits={int(n_bits)}')
    lsb = full_scale / 2.0 ** int(n_bits)
    offset = (v[0] - v_low - codes[0] * lsb) / lsb
    full = (v[-1] - v_low - codes[-1] * lsb) / lsb
    gain = ((v[-1] - v[0]) - (codes[-1] - codes[0]) * lsb) / lsb
    return StaticErrors(float(offset), float(gain), float(full))


@metric(domain='dc', outputs={'max_abs_lsb': 'LSB'})
def dnl(sig: CodeTransitions) -> DNL:
    """Differential non-linearity per step, in LSBs.

    DNL[k] = (V[k+1] − V[k]) / LSB − 1, where LSB is the endpoint-line
    LSB.

    DNL is defined per code, so ``sig.x`` must be consecutive integers
    covering every code between its endpoints; a gap raises
    ``ValueError`` naming the first missing code (caller error,
    the metric failure convention) and points to :func:`inl` for sparse or non-consecutive
    codes.

    A zero LSB returns all-NaN numeric fields with a
    ``silk.metrics.diagnostics.MetricWarning``. A NaN transition
    voltage warns with its code and propagates through adjacent widths;
    a NaN endpoint makes the shared LSB and all widths unavailable.
    NaN code coordinates and malformed axes raise ``ValueError``.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
        DNL-based measurements additionally require consecutive codes.

    Returns
    -------
    DNL
        DNL array of shape (N-1,) and maximum absolute DNL in LSBs. Zero
        LSB gives all-nan numeric fields. A NaN voltage affects adjacent
        widths (all widths for a NaN endpoint) and makes the maximum nan;
        these cases emit MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Nonconsecutive codes are rejected by dnl; missing_codes and
        missing_code_count return their NaN result before that check when
        transition voltages contain NaNs.
    """
    _check_code_signal(sig, 'dnl')
    codes = np.asarray(sig.x, dtype=float)
    _check_consecutive_codes(codes, 'dnl')
    v = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    _emit_nan_transitions(sig, 'dnl')
    lsb = _ideal_lsb(codes, v)
    if lsb == 0:
        _emit_zero_lsb('dnl', v)
        return DNL(np.full(len(v) - 1, float('nan')), float('nan'))
    steps = np.diff(v) / lsb - 1.0
    return DNL(steps, float(np.max(np.abs(steps))))


def is_monotonic(sig: Signal) -> bool:
    """True if the transfer curve is strictly increasing.

    A NaN transition voltage cannot be ordered, so the curve is not
    provably increasing: returns ``False``. A bool has no ``nan``, and
    ``False`` is the always-failing direction — a
    monotonicity spec cannot pass on a NaN-bearing curve.

    A table from :func:`extract_code_transitions` also carries
    ``code_reversals``, the falling code steps the table cannot show; any reversal makes the curve non-monotonic, and an
    unknown (``nan``) count returns ``False``. A hand-built table without
    the attribute is judged on its transition voltages alone.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.

    Returns
    -------
    bool
        True only for strictly increasing voltages and no recorded reversals.
        NaN voltages or unknown reversals return False without MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
    """
    _check_code_signal(sig, 'is_monotonic')
    return bool(_count_non_monotonic_steps(sig) == 0)


def missing_codes(sig: Signal) -> np.ndarray:
    """Codes (in ``sig.x``) whose bin has zero width — DNL ≤ −1.

    ``dnl_lsb[i]`` is the width of code ``sig.x[i]``'s bin,
    ``(V[i+1] − V[i]) / LSB − 1`` (IEEE 1241): the transition into
    ``sig.x[i]`` and the transition into ``sig.x[i+1]`` coincide, so
    ``sig.x[i]`` is never actually output — the converter skips it.

    Returns ``array([nan])`` — a one-element NaN sentinel — if ``sig.y``
    contains a NaN transition voltage: ``nan <= -1.0`` is
    ``False``, so an unguarded comparison silently dropped every code
    whose bin width was unknown and returned ``[]``, which already
    means "no missing codes" — a passing answer for a curve that was
    never measured. Same convention as :func:`crossings`.

    A zero derived LSB also returns this sentinel, preserving :func:`dnl`'s
    originating warning. A real zero-width bin remains a missing code.
    NaN code coordinates raise ``ValueError``; missing transition voltages
    warn with the affected code. Bin width is a DNL concept, so this
    inherits :func:`dnl`'s requirement that ``sig.x`` be consecutive
    integers, via the internal call to :func:`dnl`: non-consecutive codes
    raise the same ``ValueError`` (unless ``sig.y`` already contains a NaN
    transition voltage, which returns the NaN sentinel above first).

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
        DNL-based measurements additionally require consecutive codes.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array of missing code numbers, shape (K,),
        possibly empty. NaN transition voltages or zero LSB return the
        shape-(1,) sentinel [nan], with the originating MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Nonconsecutive codes are rejected by dnl; missing_codes and
        missing_code_count return their NaN result before that check when
        transition voltages contain NaNs.
    """
    _check_code_signal(sig, 'missing_codes')
    if _any_nan(sig.y):
        _emit_nan_transitions(sig, 'missing_codes')
        return np.array([float('nan')])
    d = dnl(sig).dnl_lsb
    codes = np.asarray(sig.x, dtype=float)
    if _ideal_lsb(codes, _nan_safe_transform(sig.y, lambda y: y.astype(float))) == 0:
        # dnl explains a zero LSB; do not turn its NaNs into "no missing codes".
        return np.array([float('nan')])
    miss_idx = np.where(d <= -1.0)[0]
    return np.asarray(sig.x, dtype=float)[miss_idx]


@metric(domain='dc', unit='LSB')
def inl_max_lsb(sig: CodeTransitions, method: Literal['endpoint', 'best_fit'] = 'endpoint') -> float:
    """Maximum absolute INL in LSBs from :func:`inl`.

    Inherits NaN for missing transitions/zero LSB and caller-error validation.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
    method : {'endpoint', 'best_fit'}, default 'endpoint'
        Reference-line construction: endpoint line or least-squares line.

    Returns
    -------
    float
        Maximum absolute INL in LSBs. NaN voltages or zero LSB return nan
        with the originating MetricWarning from inl.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        An unknown method is also rejected.
    """
    _check_code_signal(sig, 'inl_max_lsb')
    return inl(sig, method=method).max_abs_lsb


@metric(domain='dc', unit='LSB')
def dnl_max_lsb(sig: CodeTransitions) -> float:
    """Maximum absolute DNL in LSBs from :func:`dnl`.

    Inherits NaN for missing transitions/zero LSB and caller-error validation.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
        DNL-based measurements additionally require consecutive codes.

    Returns
    -------
    float
        Maximum absolute DNL in LSBs. NaN voltages or zero LSB return nan
        with the originating MetricWarning from dnl.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Nonconsecutive codes are rejected by dnl; missing_codes and
        missing_code_count return their NaN result before that check when
        transition voltages contain NaNs.
    """
    _check_code_signal(sig, 'dnl_max_lsb')
    return dnl(sig).max_abs_lsb


@metric(domain='dc', unit='LSB')
def offset_error_lsb(sig: CodeTransitions, *, full_scale: float, n_bits: int, v_low: float = 0.0) -> float:
    """Offset error in LSBs from :func:`static_errors`.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
    full_scale : float
        Finite positive input-voltage span in V; ideal LSB = full_scale / 2**n_bits.
    n_bits : int
        Positive converter resolution in bits. Codes must lie in
        [0, 2**n_bits - 1].
    v_low : float, default 0.0
        Finite real input voltage in V at the bottom of the ideal range.

    Returns
    -------
    float
        The offset_error_lsb field of static_errors, in ideal LSBs.
        NaN transition voltages return nan with its originating MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Invalid full_scale, n_bits or v_low, or codes outside the specified
        converter range, are also rejected. NaN voltages return before
        the code-range check.
    OverflowError
        Infinite n_bits fails during integer conversion.
    """
    _check_code_signal(sig, 'offset_error_lsb')
    return static_errors(sig, full_scale=full_scale, n_bits=n_bits,
                         v_low=v_low).offset_error_lsb


@metric(domain='dc', unit='LSB')
def gain_error_lsb(sig: CodeTransitions, *, full_scale: float, n_bits: int, v_low: float = 0.0) -> float:
    """Gain error in LSBs from :func:`static_errors`.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
    full_scale : float
        Finite positive input-voltage span in V; ideal LSB = full_scale / 2**n_bits.
    n_bits : int
        Positive converter resolution in bits. Codes must lie in
        [0, 2**n_bits - 1].
    v_low : float, default 0.0
        Finite real input voltage in V at the bottom of the ideal range.

    Returns
    -------
    float
        The gain_error_lsb field of static_errors, in ideal LSBs.
        NaN transition voltages return nan with its originating MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Invalid full_scale, n_bits or v_low, or codes outside the specified
        converter range, are also rejected. NaN voltages return before
        the code-range check.
    OverflowError
        Infinite n_bits fails during integer conversion.
    """
    _check_code_signal(sig, 'gain_error_lsb')
    return static_errors(sig, full_scale=full_scale, n_bits=n_bits,
                         v_low=v_low).gain_error_lsb


@metric(domain='dc', unit='LSB')
def full_scale_error_lsb(sig: CodeTransitions, *, full_scale: float, n_bits: int, v_low: float = 0.0) -> float:
    """Full-scale error in LSBs from :func:`static_errors`.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
    full_scale : float
        Finite positive input-voltage span in V; ideal LSB = full_scale / 2**n_bits.
    n_bits : int
        Positive converter resolution in bits. Codes must lie in
        [0, 2**n_bits - 1].
    v_low : float, default 0.0
        Finite real input voltage in V at the bottom of the ideal range.

    Returns
    -------
    float
        The full_scale_error_lsb field of static_errors, in ideal LSBs.
        NaN transition voltages return nan with its originating MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Invalid full_scale, n_bits or v_low, or codes outside the specified
        converter range, are also rejected. NaN voltages return before
        the code-range check.
    OverflowError
        Infinite n_bits fails during integer conversion.
    """
    _check_code_signal(sig, 'full_scale_error_lsb')
    return static_errors(sig, full_scale=full_scale, n_bits=n_bits,
                         v_low=v_low).full_scale_error_lsb


@metric(domain='dc', unit='')
def missing_code_count(sig: CodeTransitions) -> float:
    """Number of missing codes from :func:`missing_codes`, as a float.

    An unknown count (the parent's NaN sentinel) stays NaN, never a
    plausible finite count. Inherits transition-table input validation.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.
        DNL-based measurements additionally require consecutive codes.

    Returns
    -------
    float
        Dimensionless missing-code count, zero when no bins are missing.
        NaN transition voltages or zero LSB return nan with the originating
        MetricWarning from missing_codes or dnl.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
        Nonconsecutive codes are rejected by dnl; missing_codes and
        missing_code_count return their NaN result before that check when
        transition voltages contain NaNs.
    """
    _check_code_signal(sig, 'missing_code_count')
    codes = missing_codes(sig)
    return float('nan') if _any_nan(codes) else float(len(codes))


def _count_non_monotonic_steps(sig):
    """Shared strict-increase predicate/count; NaN is not a measured step.

    Table steps that are not strictly increasing, plus the extractor's
    ``code_reversals`` (0 when absent, i.e. a hand-built table).
    """
    v = _nan_safe_transform(sig.y, lambda y: y.astype(float))
    reversals = float(getattr(sig, 'code_reversals', 0))
    if _any_nan(v) or math.isnan(reversals):
        return float('nan')
    return float(np.count_nonzero(~(np.diff(v) > 0))) + reversals


@metric(domain='dc', unit='')
def non_monotonic_steps(sig: CodeTransitions) -> float:
    """Count adjacent steps that are not strictly increasing.

    Shares :func:`is_monotonic`'s predicate, including equal transitions,
    and adds the ``code_reversals`` that :func:`extract_code_transitions`
    records for a code that goes back down, one per falling
    code step. NaN transition voltages or an unknown reversal count return
    NaN rather than a passing zero count; invalid code-transition tables
    raise ``ValueError``.

    Parameters
    ----------
    sig : Signal
        Code-transition table with ``xlabel='code'``, equal-length
        one-dimensional arrays of shape (N,), N >= 2. ``x`` contains finite,
        strictly increasing integer codes; ``y`` contains real transition
        voltages in V. The input is not modified.

    Returns
    -------
    float
        Dimensionless count of non-increasing adjacent voltage steps plus
        recorded code reversals. NaN voltages or unknown reversals return
        nan without MetricWarning.

    Raises
    ------
    ValueError
        For a wrong axis label, fewer than two codes, unequal array lengths,
        or non-finite, non-integer, repeated or descending code coordinates.
    """
    _check_code_signal(sig, 'non_monotonic_steps')
    return _count_non_monotonic_steps(sig)


def aperture_jitter_snr_db(f_in: float, sigma_t: float) -> float:
    """SNR limit from sampling-clock aperture jitter.

    Parameters
    ----------
    f_in : float
        Input sinusoid frequency in Hz; must be positive.
    sigma_t : float
        RMS sampling-time jitter in s; must be nonnegative.

    Returns
    -------
    float
        ``-20*log10(2*pi*f_in*sigma_t)`` in dB. Zero jitter returns ``inf``.
        Otherwise NaN propagates and an infinite product gives ``-inf``.
        No ``MetricWarning`` is emitted; NumPy may warn if a positive product
        underflows to zero.

    Raises
    ------
    ValueError
        If ``f_in <= 0`` or ``sigma_t < 0``, naming the value and unit.
        The frequency check runs first.
    """
    if f_in <= 0:
        raise ValueError(f'aperture_jitter_snr_db: f_in must be > 0, got {f_in} Hz')
    if sigma_t < 0:
        raise ValueError(f'aperture_jitter_snr_db: sigma_t must be >= 0, got {sigma_t} s')
    if sigma_t == 0:
        return float('inf')
    return float(-20.0 * np.log10(2.0 * np.pi * f_in * sigma_t))
