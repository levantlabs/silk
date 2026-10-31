"""General waveform tools that work on any x-axis (time, a DC sweep, frequency).

Crossings, point values, extrema, calculus, sampling, shifts, reshaping and
waveform combinations operate on the coordinates supplied by the Signal.
Crossings are linearly interpolated between samples; integrals use trapezoids.
Malformed requests raise; unavailable data return NaNs as each function documents.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import math

import numpy as np

from silk.utils._compat import trapezoid
from silk.signals.base import Signal, _mirror_x
from silk.signals.schema import metric, AnySignal
from silk.metrics._guards import (
    _check_nonempty,
    _any_nan,
    _axis,
    _Axis,
    _nan_safe_transform,
)
from silk.metrics.diagnostics import emit


def _axis_context(sig: Signal) -> str:
    """Describe available coordinates without inspecting waveform values."""
    span = (f'trace spans [{sig.x[0]}, {sig.x[-1]}]'
            if len(sig.x) else 'trace is empty')
    return f'xlabel={sig.xlabel!r}, sample_count={len(sig.x)}; {span}'


def _check_edge(edge: str, metric: str) -> None:
    """Validate the existing edge choices under the invoking metric's name."""
    if edge not in ('rising', 'falling', 'any'):
        raise ValueError(
            f"{metric}: edge must be 'rising'|'falling'|'any', got {edge!r}"
        )


def _check_transient(sig: Signal, metric: str) -> None:
    if sig.xlabel != 'time':
        raise ValueError(
            f"{metric}: sig must be a transient Signal (xlabel='time'), got "
            f"{_axis_context(sig)}"
        )
    _check_nonempty(sig, metric)
    _axis(sig, metric, order='increasing')


def _timing_details(sig: Signal) -> dict:
    """Describe the supplied time record without inspecting waveform values."""
    return {
        'signal_name': sig.name, 'sample_count': len(sig.x),
        'time_start_s': float(sig.x[0]), 'time_stop_s': float(sig.x[-1]),
    }


def _all_crossings(t: np.ndarray, y: np.ndarray, level: float,
                   edge: str) -> List[float]:
    out: List[float] = []
    for i in range(len(y) - 1):
        a, b = float(y[i]), float(y[i + 1])
        is_rising = a < level <= b
        is_falling = a >= level > b
        if edge == 'rising' and not is_rising:
            continue
        if edge == 'falling' and not is_falling:
            continue
        if edge == 'any' and not (is_rising or is_falling):
            continue
        if b == a:
            out.append(float(t[i]))
            continue
        frac = (level - a) / (b - a)
        out.append(float(t[i]) + frac * float(t[i + 1] - t[i]))
    return out


def _first_crossing(t: np.ndarray, y: np.ndarray, level: float,
                    edge: str, start_idx: int = 0) -> Optional[float]:
    for i in range(max(start_idx, 0), len(y) - 1):
        a, b = float(y[i]), float(y[i + 1])
        is_rising = a < level <= b
        is_falling = a >= level > b
        if edge == 'rising' and not is_rising:
            continue
        if edge == 'falling' and not is_falling:
            continue
        if edge == 'any' and not (is_rising or is_falling):
            continue
        if b == a:
            return float(t[i])
        frac = (level - a) / (b - a)
        return float(t[i]) + frac * float(t[i + 1] - t[i])
    return None


def _crossing_data_has_nan(sig: Signal, *, metric: str, signal_role: str,
                           purpose: str, start_idx: int = 0,
                           stop_idx: Optional[int] = None) -> bool:
    """Diagnose only the samples consumed by this crossing measurement."""
    y = sig.y[start_idx:stop_idx]
    mask = np.isnan(y)
    if not np.any(mask):
        return False
    t = sig.x[start_idx:stop_idx]
    if sig.xlabel == 'time':
        coordinates = {
            'time_start_s': float(t[0]), 'time_stop_s': float(t[-1]),
            'nan_time_start_s': float(t[mask][0]),
            'nan_time_stop_s': float(t[mask][-1]),
        }
    else:
        coordinates = {
            'x_label': sig.xlabel,
            'x_unit': {'frequency': 'Hz', 'voltage': 'V', 'current': 'A'}.get(
                sig.xlabel, sig.xlabel),
            'x_start': float(t[0]), 'x_stop': float(t[-1]),
            'nan_x_start': float(t[mask][0]), 'nan_x_stop': float(t[mask][-1]),
        }
    emit(
        metric=metric, code='nan_waveform',
        message='Measurement unavailable because the consumed data contains NaNs.',
        details={
            'signal_name': sig.name, 'signal_role': signal_role,
            'purpose': purpose, 'sample_count': len(y),
            'nan_sample_count': int(np.count_nonzero(mask)),
            **coordinates,
        },
        suggestion='Check the source result for the affected waveform samples.',
    )
    return True


def _measured_crossing(sig: Signal, threshold: Optional[float], edge: str, *,
                       metric: str, signal_role: str,
                       start_idx: int = 0) -> Optional[float]:
    """Find the first edge without searching past unknown earlier intervals."""
    if threshold is None:
        if _crossing_data_has_nan(
            sig, metric=metric, signal_role=signal_role, purpose='threshold',
        ):
            return float('nan')
        threshold = 0.5 * (float(np.min(sig.y)) + float(np.max(sig.y)))
    start_idx = max(start_idx, 0)
    unknown = np.flatnonzero(np.isnan(sig.y[start_idx:]))
    stop_idx = start_idx + int(unknown[0]) if len(unknown) else len(sig.y)
    crossing = _first_crossing(
        sig.x[:stop_idx], sig.y[:stop_idx], float(threshold), edge,
        start_idx=start_idx,
    )
    if crossing is None and len(unknown):
        _crossing_data_has_nan(
            sig, metric=metric, signal_role=signal_role, purpose='crossing_search',
            start_idx=start_idx, stop_idx=stop_idx + 1,
        )
        return float('nan')
    return crossing


def _check_occurrence(n, metric: str, *, name: str = 'n',
                      plural: str = 'crossings',
                      noun: str = 'crossing') -> None:
    """Validate an occurrence: a nonzero integer (``None`` raises)."""
    if (isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or n == 0):
        raise ValueError(
            f'{metric}: {name} must be a nonzero integer (k for the k-th '
            f'{noun}, -k for the k-th from the end), got {n!r}; use '
            f'{plural}() for every {noun}'
        )


def _crossing_window(sig: Signal, start: Optional[float],
                     end: Optional[float], metric: str) -> tuple[int, int]:
    """Validate ``[start, end]`` and return the consumed sample indices.

    Returns ``(lo, hi)``: samples ``lo..hi`` inclusive are every sample of
    a segment that touches the window, so a window edge that lies between
    two samples still sees the segment it cuts. A window that does not
    overlap the trace at all is a malformed request and
    raises, naming the span that is available.
    """
    x = sig.x
    for name, value in (('start', start), ('end', end)):
        if value is not None and not (isinstance(value, (int, float, np.integer,
                                                         np.floating))
                                      and math.isfinite(value)):
            raise ValueError(
                f'{metric}: {name} must be a finite number or None, got '
                f'{value!r}; {_axis_context(sig)}'
            )
    if start is not None and end is not None and end < start:
        raise ValueError(
            f'{metric}: require end >= start, got start={start}, end={end}; '
            f'{_axis_context(sig)}'
        )
    return _axis(sig, metric).segment_window(start, end)


def _window_threshold(x: np.ndarray, y: np.ndarray,
                      start: Optional[float], end: Optional[float]) -> float:
    """Midpoint of the swing INSIDE the window (the whole trace without one).

    The window bounds count as observations (linearly interpolated), so
    a window narrower than one sample interval still has a swing.
    """
    axis = _Axis(x, 'crossing', y)
    a, b = axis.span(start, end)
    inside = y[(x >= a) & (x <= b)]
    edges = axis.interp([a, b], y)
    values = np.concatenate([inside, edges])
    return 0.5 * (float(np.min(values)) + float(np.max(values)))


def _select_crossings(sig: Signal, y: np.ndarray, threshold: Optional[float],
                      edge: str, n: Optional[int], start: Optional[float],
                      end: Optional[float], *, metric: str, lo: int, hi: int,
                      details: dict):
    """The one crossing contract behind :func:`crossings` and :func:`crossing`.

    The caller has validated the request and checked the consumed samples
    ``lo..hi`` for NaN. Returns the array of every crossing inside
    ``[start, end]`` when ``n`` is ``None``, else the selected crossing
    as a float — ``nan`` with a ``MetricWarning`` when fewer than ``|n|``
    crossings were observed.
    """
    x = np.asarray(sig.x, dtype=float)
    xs, ys = x[lo:hi + 1], y[lo:hi + 1]
    if threshold is None:
        threshold = _window_threshold(xs, ys, start, end)
    found = np.array(_all_crossings(xs, ys, float(threshold), edge),
                     dtype=float)
    if start is not None:
        found = found[found >= start]
    if end is not None:
        found = found[found <= end]
    if n is None:
        return found
    if abs(n) > len(found):
        emit(
            metric=metric, code='insufficient_crossings',
            message='The requested crossing was not observed in the supplied record.',
            details={
                **details, 'threshold': float(threshold), 'edge': edge,
                'window_start': start, 'window_end': end,
                'crossing_count': len(found), 'requested_crossing': int(n),
            },
            suggestion=('Extend the record or review the threshold, direction, '
                        'crossing number and window.'),
        )
        return float('nan')
    return float(found[n - 1] if n > 0 else found[n])


def crossings(sig: Signal, *,
              threshold: Optional[float] = None,
              edge: str = 'rising',
              start: Optional[float] = None,
              end: Optional[float] = None) -> np.ndarray:
    """Times at which ``sig`` crosses ``threshold`` in ``edge`` direction.

    The public edge-extraction primitive. Period jitter, settling time,
    and propagation-delay metrics all build on this so the crossing
    convention stays consistent. :func:`crossing` is its single-value
    form — the same implementation, returning the n-th crossing as one
    float. The return type never depends on an argument:
    ``crossings`` always returns an array. Descending sweeps retain their
    acquisition order; start/end bound coordinates with start <= end.

    Parameters
    ----------
    sig : Signal
        Matching one-dimensional x/y arrays on a finite strictly rising or
        falling axis, in the input x unit. Events follow acquisition
        order; rising/falling describes y along that order.
    threshold : float, optional
        Level in the input y unit. If ``None``, uses the midpoint of the signal's
        swing inside the window (the whole trace when there is no
        window) — the natural choice for digital and clock waveforms.
    edge : str, default ``'rising'``
        ``'rising'``, ``'falling'``, or ``'any'``.
    start, end : float, optional
        Bounds in the input x unit: ``start <= t <= end``; either, both or
        neither. A bound may fall between samples — the crossing is
        interpolated as usual and kept iff it lies inside. A window that
        does not overlap the trace, or ``end < start``, raises.

    Returns
    -------
    numpy.ndarray
        One-dimensional crossing coordinates in input x units (seconds
        for time), length = number of crossings (may be 0).

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    Resolution: linear interpolation inside the sample interval; error
    ≤ Δt²·``|y''|``/(8·``|y'|``) per crossing (exact for piecewise-linear data).

    NaN handling: a NaN sample among the consumed samples (the
    whole trace, or the segments touching the window) makes every
    crossing near it unsafe to trust (an auto threshold computed from
    ``min``/``max`` would itself become NaN, and a crossing search would
    silently skip the affected segment either way), so the whole result
    is treated as unknown. The unknown case returns ``array([nan])`` — a
    single NaN sentinel — rather than an empty array: an empty array already means "found zero real crossings"
    for a valid trace, and reusing it for "couldn't tell" would make the
    two indistinguishable to a caller. The unknown case emits a
    ``MetricWarning``; a valid zero-crossing enumeration stays quiet.

    Empty input, mismatched or nonfinite coordinate arrays, a nonmonotonic
    axis, an invalid edge or a malformed/non-overlapping window raises
    ValueError. Use real-valued y for crossing comparisons.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals import Signal
    >>> from silk.metrics import crossings
    >>> sig = Signal('edge', 'voltage', 'time', 'V',
    ...              np.array([0., 1., 2.]), np.array([0., 1., 0.]))
    >>> crossings(sig, threshold=0.5, edge='any').tolist()
    [0.5, 1.5]
    """
    _check_nonempty(sig, 'crossings')
    _axis(sig, 'crossings')
    _check_edge(edge, 'crossings')
    lo, hi = _crossing_window(sig, start, end, 'crossings')
    if _crossing_data_has_nan(
        sig, metric='crossings', signal_role='sig',
        purpose='crossing_times' if sig.xlabel == 'time' else 'crossing_coordinates',
        start_idx=lo, stop_idx=hi + 1,
    ):
        return np.array([float('nan')])
    return _select_crossings(
        sig, _magnitude_samples(sig), threshold, edge, None, start, end,
        metric='crossings', lo=lo, hi=hi, details=_timing_details(sig),
    )


def _magnitude_samples(sig: Signal) -> np.ndarray:
    """Legacy sample reading for time-only metrics."""
    y = np.asarray(sig.y)
    return _nan_safe_transform(y, np.abs) if np.iscomplexobj(y) else y.astype(float)


def _samples(sig: Signal, metric: str) -> np.ndarray:
    """Read samples according to the metric: complex, magnitude, or ordered real."""
    y = np.asarray(sig.y)
    if np.iscomplexobj(y):
        if metric == 'rms':
            return _nan_safe_transform(y, np.abs)
        if metric in ('average', 'integral', 'value_at', 'final_value',
                      'stddev', 'variance'):
            return y
        reason = 'Ordering requires a chosen real-valued projection.'
        raise ValueError(
            f'{metric}: complex samples require an explicit projection; '
            f'use .real(), .mag(), or .imag() (or a phase transform). {reason}'
        )
    return y.astype(float)


def _window(sig: Signal, start: Optional[float], end: Optional[float],
            metric: str, interpolate: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` arrays restricted to ``[start, end]``.

    ``None`` defaults to the coordinate minimum/maximum. Descending sources
    are paired and returned in increasing coordinate order. With ``interpolate`` (the
    default) a window edge that falls strictly between two
    samples (and inside the trace) adds a linearly interpolated end point
    there, so no partial segment is dropped; ``False`` keeps only the
    samples inside the window. Raises ``ValueError`` if the input Signal is
    empty, or the window is empty or backwards. ``y`` is read
    through :func:`_samples`; RMS uses magnitudes, ordering metrics require
    a real projection, and the remaining metrics preserve complex samples.
    Complex window edges interpolate real and imaginary parts separately.
    """
    _check_nonempty(sig, metric)
    axis = _axis(sig, metric)
    xw, yw = axis.window(_samples(sig, metric), start, end,
                         interpolate=interpolate)
    if xw.size < 2:
        raise ValueError(
            f'{metric}: requested x window [{start}, {end}] on {sig.xlabel} axis '
            f'contains {int(xw.size)} samples; {_axis_context(sig)}; '
            f'need at least 2 samples'
        )
    return xw, yw


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration; same length as input, starts
    at 0. Replacement for ``numpy.cumulative_trapezoid`` (numpy 2.0+) /
    ``scipy.integrate.cumtrapz`` so we don't take a scipy dependency.
    """
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1]))
    return out


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def peak(sig: AnySignal, *, start: Optional[float] = None,
         end: Optional[float] = None,
         interpolate: bool = True) -> float:
    """Largest-magnitude excursion: ``max(|y|)`` in the window.

    This is the magnitude peak, which for a symmetric signal (most analog
    waveforms) is the signal's amplitude. For the y of the largest local maximum (signed), use
    :func:`maximum`.
    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly.

    Ordering requires a chosen real-valued projection; this refusal is permanent.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.

    Returns
    -------
    float
        Reduction in input y units. Consumed NaNs propagate to nan with
        MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'peak', interpolate)
    if _any_nan(y):
        emit(
            metric='peak', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return float(np.max(np.abs(y)))


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def peak_to_peak(sig: AnySignal, *, start: Optional[float] = None,
                 end: Optional[float] = None,
                 interpolate: bool = True) -> float:
    """Peak-to-peak excursion: ``max(y) − min(y)`` in the window.

    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly.

    Ordering requires a chosen real-valued projection; this refusal is permanent.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.

    Returns
    -------
    float
        Reduction in input y units. Consumed NaNs propagate to nan with
        MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'peak_to_peak', interpolate)
    if _any_nan(y):
        emit(
            metric='peak_to_peak', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return float(np.max(y) - np.min(y))


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def minimum(sig: AnySignal, *, start: Optional[float] = None,
            end: Optional[float] = None,
            interpolate: bool = True) -> float:
    """Minimum y value in the window.

    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly.

    Ordering requires a chosen real-valued projection; this refusal is permanent.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.

    Returns
    -------
    float
        Reduction in input y units. Consumed NaNs propagate to nan with
        MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'minimum', interpolate)
    if _any_nan(y):
        emit(
            metric='minimum', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return float(np.min(y))


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def maximum(sig: AnySignal, *, start: Optional[float] = None,
            end: Optional[float] = None,
            interpolate: bool = True) -> float:
    """Maximum y value in the window.

    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly.

    Ordering requires a chosen real-valued projection; this refusal is permanent.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.

    Returns
    -------
    float
        Reduction in input y units. Consumed NaNs propagate to nan with
        MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'maximum', interpolate)
    if _any_nan(y):
        emit(
            metric='maximum', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return float(np.max(y))


def _extremum_x(x: np.ndarray, y: np.ndarray, i: int, refine: bool) -> float:
    """X of the extreme sample ``i``; with ``refine``, the vertex of the
    parabola through samples ``i-1, i, i+1`` clamped to ``[x[i-1], x[i+1]]``.
    At a window edge, or on a degenerate (collinear or repeated-x) triple,
    the sample x is returned."""
    if not refine or i == 0 or i == len(y) - 1:
        return float(x[i])
    x0, x1, x2 = (float(v) for v in x[i - 1:i + 2])
    y0, y1, y2 = (float(v) for v in y[i - 1:i + 2])
    d01, d12 = x1 - x0, x2 - x1
    if d01 <= 0 or d12 <= 0:
        return x1
    s01, s12 = (y1 - y0) / d01, (y2 - y1) / d12
    curv = (s12 - s01) / (x2 - x0)
    if curv == 0:
        return x1
    # Newton form: y' = s01 + 2*curv*(x - (x0+x1)/2) = 0
    xv = 0.5 * (x0 + x1) - s01 / (2.0 * curv)
    return float(min(max(xv, x0), x2))


def _refined_extremum_x(sig: Signal, xw: np.ndarray, i: int, metric: str) -> float:
    """Refined x of the extreme windowed sample ``i``.

    The parabola is fitted through the ORIGINAL samples around the
    extremum, never through a synthetic window-edge point (which lies on a
    chord, not on the curve), then clamped to the window. A synthetic edge
    point that is itself the extreme returns its own x, and so does a
    neighbour that is not finite.
    """
    axis = _axis(sig, metric)
    x = axis.increasing
    j = int(axis.search(xw[i]))
    if j >= len(x) or x[j] != xw[i]:
        return float(xw[i])
    y = axis.orient(_samples(sig, metric))
    if 0 < j < len(y) - 1 and not np.all(np.isfinite(y[j - 1:j + 2])):
        return float(x[j])
    xv = _extremum_x(x, y, j, True)
    return float(min(max(xv, xw[0]), xw[-1]))


@metric(domain='tran', unit='[x]',
        param_units={'start': '[x]', 'end': '[x]'})
def argmin_x(sig: AnySignal, *, start: Optional[float] = None,
             end: Optional[float] = None,
             interpolate: bool = True, refine: bool = False) -> float:
    """X value at which y is minimum in the window.

    Returns ``nan`` if the window contains a NaN sample:
    ``numpy.argmin`` treats NaN as its own minimum, so it would silently
    return the NaN sample's own x position rather than the true minimum.
    ``refine=True`` returns the vertex of the parabola through the extreme
    sample and its two neighbours in the original trace (clamped to them and
    to the window; never a synthetic window-edge point), falling back to the
    sample x at a trace end; the default ``False`` returns the sample x.
    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly.

    Ordering requires a chosen real-valued projection; this refusal is permanent.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.
    refine : bool
        Fit a parabola to original neighbours of the extreme when True; False
        returns the selected sample coordinate.

    Returns
    -------
    float
        Reduction in input x units (seconds for time, Hz for frequency). Consumed
        NaNs propagate to nan with MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'argmin_x', interpolate)
    if _any_nan(y):
        emit(
            metric='argmin_x', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
        return float('nan')
    i = int(np.argmin(y))
    return _refined_extremum_x(sig, x, i, 'argmin_x') if refine else float(x[i])


@metric(domain='tran', unit='[x]',
        param_units={'start': '[x]', 'end': '[x]'})
def argmax_x(sig: AnySignal, *, start: Optional[float] = None,
             end: Optional[float] = None,
             interpolate: bool = True, refine: bool = False) -> float:
    """X value at which y is maximum in the window.

    Returns ``nan`` if the window contains a NaN sample:
    ``numpy.argmax`` treats NaN as its own maximum, so it would silently
    return the NaN sample's own x position rather than the true maximum.
    ``refine=True`` returns the vertex of the parabola through the extreme
    sample and its two neighbours in the original trace (clamped to them and
    to the window; never a synthetic window-edge point), falling back to the
    sample x at a trace end; the default ``False`` returns the sample x.
    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly.

    Ordering requires a chosen real-valued projection; this refusal is permanent.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.
    refine : bool
        Fit a parabola to original neighbours of the extreme when True; False
        returns the selected sample coordinate.

    Returns
    -------
    float
        Reduction in input x units (seconds for time, Hz for frequency). Consumed
        NaNs propagate to nan with MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'argmax_x', interpolate)
    if _any_nan(y):
        emit(
            metric='argmax_x', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
        return float('nan')
    i = int(np.argmax(y))
    return _refined_extremum_x(sig, x, i, 'argmax_x') if refine else float(x[i])


@metric(domain='tran', unit='[y]*[x]',
        param_units={'start': '[x]', 'end': '[x]'})
def integral(sig: AnySignal, *, start: Optional[float] = None,
             end: Optional[float] = None,
             interpolate: bool = True) -> float | complex:
    """Trapezoidal definite integral of y over x in the window.

    Resolution: error
    ≤ ``Δx²·(b−a)·max|y''|/12`` — exactly Δx²/6 for ``y = x²`` on [0, 1].
    Complex input returns a complex value, preserving both real and imaginary
    parts; interpolation is Cartesian (each part separately) on the x-axis.
    Real input returns a float. Spec limits require a real projection: use
    ``.real`` or ``abs`` on the result, or project the input Signal.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    interpolate : bool
        Insert linearly interpolated window endpoints when True (default); otherwise
        retain only existing samples.

    Returns
    -------
    float or complex
        Reduction in input y units times input x units. Consumed NaNs propagate to
        nan with MetricWarning. Complex input preserves both components.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    x, y = _window(sig, start, end, 'integral', interpolate)
    if _any_nan(y):
        emit(
            metric='integral', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return (complex if np.iscomplexobj(y) else float)(trapezoid(y, x))


def running_integral(sig: Signal) -> Signal:
    """Cumulative trapezoidal integral as a Signal.

    Returns a Signal with the same x-axis
    and ``y[0] = 0``. Real and complex values are integrated without
    projection; NaNs propagate through the affected cumulative intervals.
    Raises ``ValueError`` on an empty Signal rather than returning an empty one.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).

    Returns
    -------
    Signal
        Same-length x/y arrays, preserving acquisition order, with cumulative
        integral in input y units times x units. The first value is zero, including
        singleton input. Metadata ylabel is inherited unchanged; consumed NaNs emit
        MetricWarning for two or more samples and contaminate subsequent integrals.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned x may share storage with sig.x when already float64;
    editing its coordinates can then change the input.
    Invalid coordinate arrays raise ValueError. Coordinates must be finite,
    one-dimensional and match y. Repeated or reversing coordinates are accepted: the integral follows
    the acquisition path.
    """
    _check_nonempty(sig, 'running_integral')
    _axis(sig, 'running_integral', order='path')
    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y, dtype=complex if np.iscomplexobj(sig.y) else float)
    if len(y) > 1 and np.any(np.isnan(y)):
        emit(
            metric='running_integral', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return Signal(
        f'iinteg({sig.name})', sig.type, sig.xlabel, sig.ylabel,
        x, np.concatenate(([0], np.cumsum(
            0.5 * (y[1:] + y[:-1]) * np.diff(x),
        ))),
    )


def moving_average(sig: Signal, *, window: float) -> Signal:
    """Sliding-window time average as a Signal.

    Coordinates must be strictly increasing; other orders raise ValueError.

    ``y(t) = (I(t) - I(t - window)) / window`` with ``I`` the
    :func:`running_integral` of ``sig``; ``I(t - window)`` includes the
    trapezoidal integral over the cut piecewise-linear segment. The result
    is on ``sig``'s x grid. Where the window
    reaches before the first sample (``t < x[0] + window``) the average is
    undefined and the value is ``nan`` rather than a biased partial
    window. ``window`` must be finite and > 0, else ``ValueError``.
    Resolution: the trapezoid error of the integral; exact for
    piecewise-linear ``y`` (``I(t - window)`` adds the trapezoid over the
    cut segment), so ``y[k]`` equals ``average(sig, start=x[k]-window,
    end=x[k])``. A NaN sample makes the running integral NaN, so every
    later output is NaN.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    window : float
        Positive finite averaging width in the input x unit (seconds for time).

    Returns
    -------
    Signal
        Same-length x/y arrays with averages in input y units. Leading unsupported
        windows are nan without warning. Real and complex samples are preserved;
        waveform NaNs propagate through the cumulative integral with its
        MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned x may share storage with sig.x when already float64;
    editing its coordinates can then change the input.
    Invalid coordinate arrays raise ValueError. Coordinates must be finite,
    one-dimensional and match y. Coordinates must be strictly increasing.
    """
    if not (np.isfinite(window) and window > 0):
        raise ValueError(
            f'moving_average: window must be finite and > 0, got {window!r}'
        )
    axis = _axis(sig, 'moving_average', order='increasing')
    integ = running_integral(sig)
    x = np.asarray(integ.x, dtype=float)
    ii = np.asarray(integ.y)
    start = x - window
    # tolerance so a window ending exactly on the first sample is defined
    valid = start >= x[0] - 1e-12 * max(abs(window), abs(x[-1] - x[0]))
    # I(t-w) = I(x_j) + trapezoid over the cut segment, so the cut segment
    # is piecewise-linear like every other primitive (TASK-442)
    yy = np.asarray(sig.y)
    j = np.clip(axis.search(start, side='right') - 1, 0, len(x) - 1)
    jn = np.minimum(j + 1, len(x) - 1)
    dx = np.where(jn > j, x[jn] - x[j], 1.0)
    frac = np.clip((start - x[j]) / dx, 0.0, 1.0)
    frac = np.where(jn > j, frac, 0.0)
    y_cut = yy[j] + frac * (yy[jn] - yy[j])
    lagged = ii[j] + 0.5 * (yy[j] + y_cut) * (start - x[j])
    out = (ii - lagged) / window
    out[~valid] = np.nan
    return Signal(
        f'mavg({sig.name})', sig.type, sig.xlabel, sig.ylabel, x, out,
    )


def derivative(sig: Signal) -> Signal:
    """Numerical derivative dy/dx as a Signal, central differences.

    Endpoints use
    one-sided differences (per ``numpy.gradient``). Raises ``ValueError``
    on an empty Signal or when fewer than two points are
    available to form a difference.

    Complex-aware: ``np.gradient`` of a complex ``y`` is well
    defined (it differentiates the real and imaginary parts together),
    so this is also the right way to take d/dw of an AC transfer function
    H(jw) — a real Signal comes back real, a complex one comes back
    complex, and neither raises ``ComplexWarning``. Previously, this
    function did ``np.asarray(sig.y, dtype=float)`` unconditionally, so a
    complex Signal handed in by mistake was silently real-cast and
    differentiated on Re(H) alone.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).

    Returns
    -------
    Signal
        Same-length x/y arrays with derivative in input y units divided by x units.
        The inherited ylabel is not rewritten. Real and complex samples are
        supported; consumed NaNs emit MetricWarning and contaminate affected
        differences.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned x may share storage with sig.x when already float64;
    editing its coordinates can then change the input.
    Invalid coordinate arrays raise ValueError. Coordinates must be finite,
    one-dimensional and match y. Either strictly monotonic orientation is accepted.
    """
    _check_nonempty(sig, 'derivative')
    _axis(sig, 'derivative')
    sample_count = len(sig.x)
    if sample_count < 2:
        raise ValueError(
            f"derivative: Signal {sig.name!r} has {sample_count} sample; "
            f"a derivative needs at least 2 points"
        )
    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y)
    if not np.iscomplexobj(y):
        y = y.astype(float)
    if np.any(np.isnan(y)):
        emit(
            metric='derivative', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(y))),
                'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return Signal(
        f'd/dx({sig.name})', sig.type, sig.xlabel, sig.ylabel,
        x, np.gradient(y, x),
    )


@metric(domain='tran', unit='[y]', param_units={'x': '[x]'})
def value_at(sig: AnySignal, x: float) -> float | complex:
    """Linearly interpolated y at the given x.

    Supports strictly rising or falling coordinates.
    Returns ``nan`` if x is outside the trace; raises ``ValueError``
    if the trace itself is empty (a point query needs
    somewhere to look). Complex input returns a complex value, interpolating
    real and imaginary parts separately on the x-axis; real input returns
    a float. Spec limits require a real projection: use ``.real`` or ``abs``
    on the result, or project the input Signal.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    x : float
        Query coordinate in the input x unit (seconds for time, Hz for frequency).

    Returns
    -------
    float or complex
        Interpolated amplitude in input y units; complex input retains both
        components. Outside coverage returns float nan with MetricWarning, even for
        complex input. Consumed NaNs warn and propagate. A NaN query is
        passed to the interpolator without validation: a one-point signal
        returns its sole sample, while a signal with two or more points
        returns nan. Neither NaN-query case emits MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    _check_nonempty(sig, 'value_at')
    axis = _axis(sig, 'value_at')
    y = axis.orient(_samples(sig, 'value_at'))
    sx = axis.increasing
    if not np.isnan(x) and not axis.contains(x):
        emit(
            metric='value_at', code='point_outside_trace',
            message='Requested point lies outside the available trace.',
            details={
                'requested_x': float(x),
                'signal_name': sig.name, 'sample_count': 0,
                'available_sample_count': len(sx),
                'available_x_min': float(sx[0]),
                'available_x_max': float(sx[-1]),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Extend the trace or choose points inside its measured coverage.',
        )
        return float('nan')
    consumed_indices = set()
    for point in (x,):
        if np.isnan(point):
            continue
        i = int(np.clip(axis.search(point, side='right') - 1,
                        0, len(sx) - 1))
        consumed_indices.add(i)
        if sx[i] != point and i + 1 < len(sx):
            consumed_indices.add(i + 1)
    indices = sorted(consumed_indices)
    consumed_x = np.asarray(sx)[indices]
    consumed_y = y[indices]
    if _any_nan(consumed_y):
        emit(
            metric='value_at', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'requested_x': float(x),
                'signal_name': sig.name, 'sample_count': len(consumed_y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(consumed_y))),
                'x_min': float(np.min(consumed_x)),
                'x_max': float(np.max(consumed_x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    return (complex if np.iscomplexobj(y) else float)(
        axis.interp(x, _samples(sig, 'value_at')))


@metric(domain='tran', unit='[y]')
def final_value(sig: AnySignal) -> float | complex:
    """The trace's value at its last x — the settled end point.

    The endpoint is last in acquisition order, not the greatest x.
    Works on a transient, the end of a dc sweep, or a one-sample op
    point. Implemented as ``value_at(sig, sig.x[-1])``, so it inherits
    that metric's empty-trace ``ValueError``, its NaN diagnostics (a NaN
    last sample returns ``nan`` with a ``MetricWarning``).
    Complex input returns a complex value, preserving both real and imaginary
    parts; interpolation is Cartesian (each part separately) on the x-axis.
    Real input returns a float. Spec limits require a real projection: use
    ``.real`` or ``abs`` on the result, or project the input Signal.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).

    Returns
    -------
    float or complex
        Last acquired amplitude in input y units; complex input retains both
        components. A NaN last sample propagates with MetricWarning from value_at.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    _check_nonempty(sig, 'final_value')
    _axis(sig, 'final_value')
    _samples(sig, 'final_value')
    return value_at(sig, float(np.asarray(sig.x, dtype=float)[-1]))


@metric(domain='tran', unit='[x]',
        param_units={'threshold': '[y]', 'start': '[x]', 'end': '[x]'})
def crossing(sig: AnySignal, threshold: Optional[float] = None, *,
             edge: Literal['rising', 'falling', 'any'] = 'rising',
             n: int = 1,
             start: Optional[float] = None,
             end: Optional[float] = None) -> float:
    """X value at the n-th crossing of ``threshold`` in ``edge`` direction.

    The single-value form of :func:`crossings`: one implementation, so
    ``threshold``/``edge``/``start``/
    ``end`` mean exactly what they mean there, and ``n=k`` picks the
    k-th crossing (1-based), ``n=-k`` the k-th from the end;
    ``threshold=None`` is the midpoint of the swing inside the window.
    It always returns ONE float; ``n=0``, ``n=None`` or a non-integer
    raises — :func:`crossings` returns every crossing. Both forms work on any 1-D Signal, including descending DC sweeps.
    Events and n follow acquisition order; edge direction follows y in that
    order. start/end bound a coordinate interval in either orientation.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    n : int
        Nonzero occurrence index: 1 selects the first, -1 the last.
    start : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.
    end : float, optional
        Inclusive coordinate bound in input x units; None uses the trace extent.

    Returns
    -------
    float
        Coordinate in input x units (seconds for time, Hz for frequency).
        ``nan`` if fewer than ``|n|``
        crossings lie in the window, or if a consumed sample is NaN: the crossing search would otherwise silently skip the
        segment straddling the NaN, which can shift which crossing is
        "n-th" without any indication anything was skipped.

    Raises
    ------
    ValueError
        If ``sig`` is empty — no crossing search is possible
        with no samples, and ``nan`` there would be indistinguishable
        from an honest "fewer than n crossings" — or for an invalid
        ``edge``, ``n`` (``0``, ``None`` or non-integer) or window
        (``end < start``, or one that misses the trace).

    Notes
    -----
    The input Signal objects and their arrays are not modified.


    Complex samples raise ``ValueError``; choose ``.real()``, ``.mag()``
    or ``.imag()`` explicitly. Both unavailable cases emit
    a ``MetricWarning`` with counts and the consumed axis range and units.

    Ordering requires a chosen real-valued projection; this refusal is permanent.
    """
    _check_nonempty(sig, 'crossing')
    _check_edge(edge, 'crossing')
    _check_occurrence(n, 'crossing')
    lo, hi = _crossing_window(sig, start, end, 'crossing')
    y = _samples(sig, 'crossing')
    x_unit = {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel, sig.xlabel)
    consumed = y[lo:hi + 1]
    if _any_nan(consumed):
        mask = np.isnan(consumed)
        cx = sig.x[lo:hi + 1]
        emit(
            metric='crossing', code='nan_waveform',
            message='NaN samples prevent identifying the requested crossing.',
            details={
                'signal_name': sig.name, 'sample_count': len(consumed),
                'nan_sample_count': int(np.count_nonzero(mask)),
                'x_start': float(cx[0]), 'x_stop': float(cx[-1]),
                'x_unit': x_unit,
                'nan_x_start': float(cx[mask][0]),
                'nan_x_stop': float(cx[mask][-1]),
                'threshold': threshold, 'value_unit': sig.ylabel, 'edge': edge,
                'window_start': start, 'window_end': end,
                'requested_crossing': n,
            },
            suggestion='Check the source result for the affected waveform samples.',
        )
        return float('nan')
    return _select_crossings(
        sig, y, threshold, edge, n, start, end, metric='crossing', lo=lo, hi=hi,
        details={
            'signal_name': sig.name, 'sample_count': len(y),
            'x_start': float(sig.x[0]), 'x_stop': float(sig.x[-1]),
            'x_unit': x_unit, 'value_unit': sig.ylabel,
        },
    )


def _window_span(x: np.ndarray, start: Optional[float],
                 end: Optional[float]) -> tuple[float, float]:
    """``[start, end]`` clipped to the trace; ``None`` is the trace end."""
    a, b = _Axis(x, 'time window').span(start, end)
    return a, b


def _interpolated_window(x: np.ndarray, y: np.ndarray, lo: int, hi: int,
                         start: Optional[float],
                         end: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    """Samples of ``[start, end]`` with each bound sampled by linear
    interpolation; the trace's own samples when there is no window.
    ``lo``/``hi`` come from :func:`_crossing_window`."""
    if start is None and end is None:
        return x, y
    a, b = _window_span(x, start, end)
    xs, ys = x[lo:hi + 1], y[lo:hi + 1]
    xw = np.concatenate([[a], xs[(xs > a) & (xs < b)], [b]])
    return xw, _Axis(xs, 'time window', ys).interp(xw, ys)


def convolve(a: Signal, b: Signal) -> Signal:
    """Convolution of two Signals on a uniform shared dx grid.

    Coordinates must be strictly increasing; other orders raise ValueError.

    Returns a Signal of length ``len(a) + len(b) − 1`` with the
    continuous-time normalisation (``y = np.convolve(...) · dx``).
    Real and complex values are convolved without projection; NaNs propagate
    through the affected output samples. Both inputs must be on uniform
    x-grids of the same step; mixed dx is
    rejected with ValueError to avoid silent grid-resampling surprises.
    Use :func:`sample` explicitly to prepare uniform grids with the same
    dx before calling this function; inputs are not resampled automatically.

    Parameters
    ----------
    a, b : Signal
        One-dimensional real or complex waveforms, each with at least two
        samples, on finite increasing uniform grids with the same spacing
        and compatible x units. Values carry each input's y unit.

    Returns
    -------
    Signal
        One-dimensional arrays of length ``len(a.x) + len(b.x) - 1``.
        Coordinates start at ``a.x[0] + b.x[0]`` in input x units; y has
        the product of both input y units times the x unit. The xlabel
        and ylabel metadata are inherited from ``a`` unchanged.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Each input needs at least two samples; an empty input, invalid axis or
    grid mismatch raises ValueError. Consumed NaNs emit MetricWarning.
    """
    _check_nonempty(a, 'convolve')
    _check_nonempty(b, 'convolve')
    _axis(a, 'convolve', order='increasing')
    _axis(b, 'convolve', order='increasing')
    a_x = np.asarray(a.x, dtype=float)
    a_y = np.asarray(a.y, dtype=complex if np.iscomplexobj(a.y) else float)
    b_x = np.asarray(b.x, dtype=float)
    b_y = np.asarray(b.y, dtype=complex if np.iscomplexobj(b.y) else float)
    if len(a_x) < 2 or len(b_x) < 2:
        raise ValueError(
            f'convolve: signals must have at least 2 samples each; '
            f'a: {_axis_context(a)}; b: {_axis_context(b)}'
        )
    da = float(a_x[1] - a_x[0])
    db = float(b_x[1] - b_x[0])
    if not np.isclose(da, db, rtol=1e-9, atol=0.0):
        raise ValueError(
            f'convolve: a and b must have the same dx; got da={da}, db={db}. '
            'Resample explicitly with sample() so both inputs have uniform '
            'spacing and the same dx before calling convolve.'
        )
    if not np.allclose(np.diff(a_x), da, rtol=1e-6, atol=0.0):
        spacing = np.diff(a_x)
        raise ValueError(
            "convolve: input 'a' must have uniform x-spacing. "
            f'Expected dx approximately {da}; observed dx ranges from '
            f'{spacing.min()} to {spacing.max()}. '
            'Resample explicitly with sample() before calling convolve; '
            'both inputs must have uniform spacing and the same dx.'
        )
    if not np.allclose(np.diff(b_x), db, rtol=1e-6, atol=0.0):
        spacing = np.diff(b_x)
        raise ValueError(
            "convolve: input 'b' must have uniform x-spacing. "
            f'Expected dx approximately {db}; observed dx ranges from '
            f'{spacing.min()} to {spacing.max()}. '
            'Resample explicitly with sample() before calling convolve; '
            'both inputs must have uniform spacing and the same dx.'
        )
    if np.any(np.isnan(a_y)) or np.any(np.isnan(b_y)):
        emit(
            metric='convolve', code='nan_waveform',
            message='Consumed waveform operands contain NaNs.',
            details={
                'a_name': a.name, 'b_name': b.name,
                'a_sample_count': len(a_y), 'b_sample_count': len(b_y),
                'a_nan_sample_count': int(np.count_nonzero(np.isnan(a_y))),
                'b_nan_sample_count': int(np.count_nonzero(np.isnan(b_y))),
                'a_x_min': float(np.min(a_x)), 'a_x_max': float(np.max(a_x)),
                'b_x_min': float(np.min(b_x)), 'b_x_max': float(np.max(b_x)),
                'a_x_label': a.xlabel, 'b_x_label': b.xlabel,
                'a_x_unit': {'time': 's', 'frequency': 'Hz'}.get(a.xlabel),
                'b_x_unit': {'time': 's', 'frequency': 'Hz'}.get(b.xlabel),
                'a_y_unit': a.ylabel, 'b_y_unit': b.ylabel,
            },
            suggestion='Check the source results for NaNs in each waveform operand.',
        )
    y = np.convolve(a_y, b_y, mode='full') * da
    n = len(y)
    x = float(a_x[0]) + float(b_x[0]) + np.arange(n) * da
    return Signal(
        f'{a.name}*{b.name}', a.type, a.xlabel, a.ylabel, x, y,
    )


def lshift(sig: Signal, dx: float) -> Signal:
    """Time-shift the signal left by ``dx``: ``x_new = x − dx``.

    Events that occurred at time ``t`` now appear at ``t − dx`` (i.e.
    ``dx`` earlier). Negative ``dx`` shifts right. Real or complex y values, including NaNs, are preserved.
    Raises ``ValueError`` on an empty Signal.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    dx : float
        Signed shift in the input x unit (seconds for time).

    Returns
    -------
    Signal
        Same-length arrays with shifted x and unchanged y, in the original units.
        NaNs in y are preserved without MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned y may share storage with sig.y when no dtype conversion is
    needed; editing the returned samples can then change the input.
    These transforms do not validate coordinate ordering.
    """
    _check_nonempty(sig, 'lshift')
    x = np.asarray(sig.x, dtype=float) - float(dx)
    y = np.asarray(sig.y, dtype=complex if np.iscomplexobj(sig.y) else float)
    return Signal(
        f'lshift({sig.name},{dx})', sig.type, sig.xlabel, sig.ylabel, x, y,
    )


def rshift(sig: Signal, dx: float) -> Signal:
    """Time-shift the signal right by ``dx``: ``x_new = x + dx``.

    The companion of :func:`lshift`; exactly ``lshift(sig, -dx)``, so
    events at time ``t`` now appear ``dx`` later. Raises ``ValueError``
    on an empty Signal.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    dx : float
        Signed shift in the input x unit (seconds for time).

    Returns
    -------
    Signal
        Same-length arrays with shifted x and unchanged y, in the original units.
        NaNs in y are preserved without MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned y may share storage with sig.y when no dtype conversion is
    needed; editing the returned samples can then change the input.
    These transforms do not validate coordinate ordering.
    """
    _check_nonempty(sig, 'rshift')
    out = lshift(sig, -float(dx))
    out.name = f'rshift({sig.name},{dx})'
    return out


def window(sig: Signal, start: float, end: float) -> Signal:
    """Restrict a signal to the x-window ``[start, end]``.

    Returns a new Signal containing only the (x, y) pairs whose x falls
    inside the window. Real or complex y values, including NaNs, are
    preserved within that window.

    Same samples as :meth:`Signal.window <silk.signals.Signal.window>`. This is the x-range cut;
    :func:`clip` limits y values instead. Differs from the private ``_window`` helper in this module: that one
    returns raw arrays for internal use; this one returns a Signal so
    the result can be passed onward to other metrics or plotted.

    An empty INPUT raises ``ValueError``. An empty WINDOW
    over a non-empty input still returns an empty Signal — and the two are different conditions: no data
    was supplied vs. the supplied data misses the window.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    start : float
        Inclusive coordinate bound in input x units; require end > start.
    end : float
        Inclusive coordinate bound in input x units; require end > start.

    Returns
    -------
    Signal
        One-dimensional retained samples in acquisition order, in original units;
        possibly empty. NaNs are retained without MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    _check_nonempty(sig, 'window')
    _axis(sig, 'window', order='path')
    if end <= start:
        raise ValueError(
            f'window: require end > start, got start={start}, end={end}; '
            f'{_axis_context(sig)}'
        )
    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y, dtype=complex if np.iscomplexobj(sig.y) else float)
    mask = (x >= start) & (x <= end)
    return Signal(
        f'window({sig.name},{start},{end})', sig.type, sig.xlabel, sig.ylabel,
        x[mask], y[mask],
    )


def clip(sig: Signal, *, low: Optional[float] = None,
         high: Optional[float] = None) -> Signal:
    """Limit the y values of a signal to ``[low, high]`` (numpy semantics).

    x is unchanged; every y below ``low`` becomes ``low`` and every y
    above ``high`` becomes ``high``. At least one bound is required and
    ``low <= high``. ``low``/``high`` are keyword-only so a legacy
    positional ``clip(sig, x1, x2)`` (the old x-range cut, now
    :func:`window`) raises ``TypeError`` instead of silently changing
    meaning. NaNs pass through. A complex Signal raises ``ValueError``
    (no ordering). An empty INPUT raises ``ValueError``.

    Parameters
    ----------
    sig : Signal
        One-dimensional real waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).
    low : float, optional
        Lower amplitude bound in input y units; None leaves it unbounded.
    high : float, optional
        Upper amplitude bound in input y units; None leaves it unbounded.

    Returns
    -------
    Signal
        Same-length arrays in original units, with bounded y. NaNs pass through
        without MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned x may share storage with sig.x when already float64;
    editing its coordinates can then change the input.
    These transforms do not validate coordinate ordering.
    """
    _check_nonempty(sig, 'clip')
    if low is None and high is None:
        raise ValueError(
            f'clip: give at least one of low= / high=; {_axis_context(sig)}'
        )
    if low is not None and high is not None and low > high:
        raise ValueError(
            f'clip: require low <= high, got low={low}, high={high}; '
            f'{_axis_context(sig)}'
        )
    if np.iscomplexobj(sig.y):
        raise ValueError(
            f'clip: y limiting is undefined for a complex Signal; '
            f'{_axis_context(sig)}'
        )
    y = np.clip(np.asarray(sig.y, dtype=float), low, high)
    return Signal(
        f'clip({sig.name},{low},{high})', sig.type, sig.xlabel, sig.ylabel,
        np.asarray(sig.x, dtype=float), y,
    )


def flip(sig: Signal) -> Signal:
    """Time-reverse a signal: ``y[i] = y_orig[N-1-i]``; x mirrored to match.

    ``x_new = x[0] + x[-1] - x[::-1]`` — the same span, each sample
    relocated to its mirrored position. On a uniform grid this is x
    unchanged; on a non-uniform grid (e.g. SPICE's adaptive transient
    steps) a plain y-reversal would misattribute each sample to the
    wrong x. Useful for matched-filter / correlation
    constructions and for symmetry tests. Raises ``ValueError`` on an
    empty Signal. Real or complex y values, including NaNs,
    are preserved without conjugation.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform with matching x/y arrays. Coordinates are in the
        axis unit (seconds for time, Hz for frequency).

    Returns
    -------
    Signal
        Same-length arrays in original units, with mirrored x and reversed y.
        Complex values are not conjugated; NaNs are preserved without MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The reversed y may be a view of sig.y when no dtype conversion is needed;
    editing returned samples can then change the input.
    These transforms do not validate coordinate ordering.
    """
    _check_nonempty(sig, 'flip')
    return Signal(
        f'flip({sig.name})', sig.type, sig.xlabel, sig.ylabel,
        _mirror_x(sig.x),
        np.asarray(sig.y, dtype=complex if np.iscomplexobj(sig.y) else float)[::-1],
    )


def sample(sig: Signal, *,
           x_new: Optional[np.ndarray] = None,
           n: Optional[int] = None) -> Signal:
    """Resample a signal onto a new x-grid via linear interpolation.

    Provide either ``x_new`` (an explicit array of x values) or ``n``
    (number of uniformly-spaced points spanning the signal's full x
    range). Exactly one is required. Complex values are interpolated in
    both components. NaNs propagate from the samples used for interpolation;
    target points outside the input range retain the nearest endpoint value.

    Parameters
    ----------
    sig : Signal
        One-dimensional real or complex waveform on a finite strictly
        monotonic x axis, with matching x/y lengths.
    x_new : array-like, optional
        One-dimensional target coordinates in the input x unit.
    n : int, optional
        Number of uniformly-spaced samples between ``sig.x[0]`` and
        ``sig.x[-1]`` (inclusive).

    Returns
    -------
    Signal
        One-dimensional target x and interpolated y arrays; length is n
        or len(x_new). Original units and metadata are preserved. NaNs
        propagate from consumed samples without MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified during the call.
    The returned x may share storage with a float64 x_new array; editing
    its coordinates can then change that caller-supplied array.
    Invalid coordinate arrays raise ValueError. Coordinates must be finite,
    one-dimensional and match y. Either strictly monotonic orientation is accepted.

    Exactly one of x_new or n must be supplied; an empty target grid or
    n < 2 raises ValueError. A non-integer n is rejected by NumPy with
    TypeError. Target coordinates are not checked for finiteness or ordering.
    """
    _check_nonempty(sig, 'sample')
    axis = _axis(sig, 'sample')
    if (x_new is None) == (n is None):
        raise ValueError(
            f"sample: provide exactly one of 'x_new' or 'n'; got n={n!r}, "
            f"x_new={'not provided' if x_new is None else 'provided'}; "
            f"{_axis_context(sig)}"
        )
    if n is not None:
        if n < 2:
            raise ValueError(
                f'sample: n must be >= 2, got {n}; {_axis_context(sig)}'
            )
        x_target = np.linspace(float(sig.x[0]), float(sig.x[-1]), n)
    else:
        x_target = np.asarray(x_new, dtype=float)
        if len(x_target) < 1:
            raise ValueError(
                f'sample: x_new must contain at least 1 point, got '
                f'{len(x_target)}; {_axis_context(sig)}'
            )
    y = np.asarray(sig.y, dtype=complex if np.iscomplexobj(sig.y) else float)
    y_target = axis.interp(x_target, y)
    return Signal(
        f'sample({sig.name})', sig.type, sig.xlabel, sig.ylabel,
        x_target, y_target,
    )


def cross_correlation(a: Signal, b: Signal) -> Signal:
    """Discrete cross-correlation of two Signals.

    Coordinates must be strictly increasing; other orders raise ValueError.

    Returns ``c[k] = Σ a[n+k]·conj(b[n])`` for all valid sample lags
    ``k``, with the output x-axis converted to absolute time:
    ``lag = k·da + (a.x[0] - b.x[0])``. The ``a.x[0] - b.x[0]`` term
    carries each input's own start time into the lag axis, the same way
    :func:`convolve` folds both start times into its output x-axis; two
    inputs that start at the same time reduce to plain ``k·da``.

    Sign convention: at the peak, ``lag`` equals the absolute time of
    the aligned sample of ``a`` minus the absolute time of the aligned
    sample of ``b``. If ``b``'s feature occurs *after* ``a``'s (``b`` is
    delayed relative to ``a``), the peak lag is **negative**; if
    ``b``'s feature occurs *before* ``a``'s (``b`` is advanced relative
    to ``a``, or equivalently ``a`` is delayed relative to ``b``), the
    peak lag is **positive**.

    Real and complex values are supported; NaNs propagate through the
    affected output samples.

    Both inputs must be on uniform x-grids of the same step; otherwise
    ValueError is raised. Use :func:`sample` explicitly to prepare uniform
    grids with the same dx; inputs are not resampled automatically.

    Parameters
    ----------
    a, b : Signal
        One-dimensional real or complex waveforms, each with at least two
        samples, on finite increasing uniform grids with the same spacing
        and compatible x units. Values carry each input's y unit.

    Returns
    -------
    Signal
        ``xlabel='lag'``; one-dimensional arrays of length
        ``len(a.x) + len(b.x) - 1``. The x values are lags in the input
        x unit (seconds for time); y has the product of the input y units.
        No dx factor is applied; the output ylabel is empty.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Each input needs at least two samples; an empty input, invalid axis or
    grid mismatch raises ValueError. Consumed NaNs emit MetricWarning.
    """
    _check_nonempty(a, 'cross_correlation')
    _check_nonempty(b, 'cross_correlation')
    _axis(a, 'cross_correlation', order='increasing')
    _axis(b, 'cross_correlation', order='increasing')
    a_x = np.asarray(a.x, dtype=float)
    a_y = np.asarray(a.y, dtype=complex if np.iscomplexobj(a.y) else float)
    b_x = np.asarray(b.x, dtype=float)
    b_y = np.asarray(b.y, dtype=complex if np.iscomplexobj(b.y) else float)
    if len(a_x) < 2 or len(b_x) < 2:
        raise ValueError(
            f'cross_correlation: signals must have at least 2 samples each; '
            f'a: {_axis_context(a)}; b: {_axis_context(b)}'
        )
    da = float(a_x[1] - a_x[0])
    db = float(b_x[1] - b_x[0])
    if not np.isclose(da, db, rtol=1e-9, atol=0.0):
        raise ValueError(
            f'cross_correlation: a and b must have the same dx; got da={da}, db={db}. '
            'Resample explicitly with sample() so both inputs have uniform '
            'spacing and the same dx before calling cross_correlation.'
        )
    if not np.allclose(np.diff(a_x), da, rtol=1e-6, atol=0.0):
        spacing = np.diff(a_x)
        raise ValueError(
            "cross_correlation: input 'a' must have uniform x-spacing. "
            f'Expected dx approximately {da}; observed dx ranges from '
            f'{spacing.min()} to {spacing.max()}. '
            'Resample explicitly with sample() before calling cross_correlation; '
            'both inputs must have uniform spacing and the same dx.'
        )
    if not np.allclose(np.diff(b_x), db, rtol=1e-6, atol=0.0):
        spacing = np.diff(b_x)
        raise ValueError(
            "cross_correlation: input 'b' must have uniform x-spacing. "
            f'Expected dx approximately {db}; observed dx ranges from '
            f'{spacing.min()} to {spacing.max()}. '
            'Resample explicitly with sample() before calling cross_correlation; '
            'both inputs must have uniform spacing and the same dx.'
        )
    if np.any(np.isnan(a_y)) or np.any(np.isnan(b_y)):
        emit(
            metric='cross_correlation', code='nan_waveform',
            message='Consumed waveform operands contain NaNs.',
            details={
                'a_name': a.name, 'b_name': b.name,
                'a_sample_count': len(a_y), 'b_sample_count': len(b_y),
                'a_nan_sample_count': int(np.count_nonzero(np.isnan(a_y))),
                'b_nan_sample_count': int(np.count_nonzero(np.isnan(b_y))),
                'a_x_min': float(np.min(a_x)), 'a_x_max': float(np.max(a_x)),
                'b_x_min': float(np.min(b_x)), 'b_x_max': float(np.max(b_x)),
                'a_x_label': a.xlabel, 'b_x_label': b.xlabel,
                'a_x_unit': {'time': 's', 'frequency': 'Hz'}.get(a.xlabel),
                'b_x_unit': {'time': 's', 'frequency': 'Hz'}.get(b.xlabel),
                'a_y_unit': a.ylabel, 'b_y_unit': b.ylabel,
            },
            suggestion='Check the source results for NaNs in each waveform operand.',
        )
    c = np.correlate(a_y, b_y, mode='full')
    n_a = len(a_y)
    n_b = len(b_y)
    lags = np.arange(-(n_b - 1), n_a) * da + (a_x[0] - b_x[0])
    return Signal(
        f'xcorr({a.name},{b.name})', 'correlation', 'lag', '', lags, c,
    )


def _observed_range(values: np.ndarray) -> tuple[float | None, float | None]:
    """Diagnostic bounds excluding NaNs, without filtering measurement data."""
    observed = values[~np.isnan(values)]
    if not observed.size:
        return None, None
    return float(np.min(observed)), float(np.max(observed))
