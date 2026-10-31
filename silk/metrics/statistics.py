"""Waveform statistics and comparisons against a target waveform.

Average, RMS, standard deviation, variance and error norms work on any x-axis.
Histogram is time-weighted and accepts only time waveforms. Reductions use
trapezoidal integration; each function documents its window and NaN handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from silk.utils._compat import trapezoid
from silk.signals.base import Signal
from silk.signals.schema import metric, AnySignal, ReferenceWaveform, Transient
from silk.metrics._guards import _any_nan, _axis
from silk.metrics.diagnostics import emit
from silk.metrics.waveform import (
    _axis_context,
    _check_transient,
    _crossing_data_has_nan,
    _crossing_window,
    _interpolated_window,
    _magnitude_samples,
    _window,
)


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def average(sig: AnySignal, *, start: Optional[float] = None,
            end: Optional[float] = None,
            interpolate: bool = True) -> float | complex:
    """Time-weighted (or x-weighted) mean: ``∫ y dx / (b - a)``.

    Uses trapezoidal
    integration so non-uniform sample grids are handled correctly.
    The trapezoid half-weights the two end samples, so on a uniform grid
    this is *not* ``np.mean(sig.y)`` (which weights every sample equally):
    over ``N`` samples it is ``np.mean(y) + (np.mean(y) - (y[0] + y[-1])/2)
    / (N - 1)``, so the two agree exactly when the two end samples average
    to ``np.mean(y)`` (a constant signal, for example; a signal flat at both
    ends at another level does not). Resolution:
    trapezoid error ≤ ``Δx²·max|y''|/12`` (exact for piecewise-linear y);
    with ``interpolate=True`` (default) the window edges are linearly
    interpolated; ``False`` snaps to the samples inside ``[start, end]``.
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
        Reduction in input y units. Consumed NaNs propagate to nan with
        MetricWarning. Complex input preserves both components.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals import Signal
    >>> from silk.metrics import average
    >>> sig = Signal('response', 'voltage', 'frequency', 'V',
    ...              np.array([1., 2., 3.]), np.array([1.+2j, 3.+4j, 5.+6j]))
    >>> average(sig)
    (3+4j)
    """
    x, y = _window(sig, start, end, 'average', interpolate)
    if _any_nan(y):
        emit(
            metric='average', code='nan_waveform',
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
    return (complex if np.iscomplexobj(y) else float)(
        trapezoid(y, x) / (x[-1] - x[0]))


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def rms(sig: AnySignal, *, start: Optional[float] = None,
        end: Optional[float] = None,
        interpolate: bool = True) -> float:
    """Time-weighted RMS: ``√(∫ |y|² dx / (b - a))``.

    Uses trapezoidal
    integration so non-uniform grids are handled correctly.
    Complex samples remain supported: mean squared magnitude is the
    definition of complex RMS, not an implicit choice of input projection.
    It returns a real nonnegative scalar; real-valued samples retain their
    signed interpolation before squaring. For complex data, magnitudes are
    interpolated at window edges, preserving the existing quadrature rule.
    A NaN in either raw complex component propagates through that magnitude
    fold and emits the same warning as a consumed real NaN.

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
    x, y = _window(sig, start, end, 'rms', interpolate)
    if _any_nan(y):
        emit(
            metric='rms', code='nan_waveform',
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
    return float(np.sqrt(trapezoid(y * y, x) / (x[-1] - x[0])))


def _variance(sig: AnySignal, start: Optional[float], end: Optional[float],
              interpolate: bool, name: str) -> float:
    """Shared core of :func:`stddev` and :func:`variance`."""
    x, y = _window(sig, start, end, name, interpolate)
    if _any_nan(y):
        emit(
            metric=name, code='nan_waveform',
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
    mu = trapezoid(y, x) / (x[-1] - x[0])
    squared = np.abs(y - mu) ** 2 if np.iscomplexobj(y) else (y - mu) ** 2
    return float(trapezoid(squared, x) / (x[-1] - x[0]))


@metric(domain='tran', unit='[y]',
        param_units={'start': '[x]', 'end': '[x]'})
def stddev(sig: AnySignal, *, start: Optional[float] = None,
           end: Optional[float] = None,
           interpolate: bool = True) -> float:
    """Time-weighted standard deviation: ``√(∫ |y − μ|² dx / (b - a))``.

    For real samples, or complex windows without inserted edges,
    ``stddev² = rms² − |mean|²``: stddev excludes the mean level.
    At inserted complex edges, RMS interpolates magnitudes instead of
    Cartesian values, so that identity need not hold.
    Complex samples use the complex x-weighted mean and squared magnitudes
    of deviations, not spread of magnitudes. Returns a real float.
    Window edges interpolate real and imaginary parts separately.

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
    return float(np.sqrt(_variance(sig, start, end, interpolate, 'stddev')))


@metric(domain='tran', unit='[y]^2',
        param_units={'start': '[x]', 'end': '[x]'})
def variance(sig: AnySignal, *, start: Optional[float] = None,
             end: Optional[float] = None,
             interpolate: bool = True) -> float:
    """Time-weighted variance: ``∫ |y − μ|² dx / (b - a)``, i.e. ``stddev²``.

    Unit is ``[y]^2``. Complex samples use the complex x-weighted mean and
    squared magnitudes of deviations, not spread of magnitudes. Returns
    a real float. Window edges interpolate real and imaginary parts separately.

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
    float
        Reduction in squared input y units. Consumed NaNs propagate to nan with
        MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Empty, nonfinite or nonmonotonic coordinate arrays, a backwards or
    non-overlapping window, or fewer than two selected/interpolated points
    raise ValueError. Descending axes are integrated in increasing order.
    """
    return _variance(sig, start, end, interpolate, 'variance')


@dataclass(frozen=True)
class HistogramResult:
    """Result of :func:`histogram`: a time-weighted distribution of values.

    Parameters
    ----------
    edges : numpy.ndarray
        Bin edges in the signal's y unit, ascending, ``len(fractions) + 1``
        long. Bin ``k`` is ``[edges[k], edges[k+1])``; the last bin also
        holds ``edges[-1]`` (the ``numpy.histogram`` convention).
    fractions : numpy.ndarray
        Fraction of the window's time spent in each bin. Sums to 1 when the
        edges span every value in the window (always, for an integer
        ``bins``); time spent outside explicit edges is not counted.

    Returns
    -------
    HistogramResult
        Frozen result container. Field assignment is prohibited, but contained
        lists, arrays and Signals remain mutable. Construction stores the supplied
        fields without validation or copying and emits no MetricWarning.

    Attributes
    ----------
    edges : numpy.ndarray
        Bin edges in the signal's y unit, ascending, ``len(fractions) + 1``
        long. Bin ``k`` is ``[edges[k], edges[k+1])``; the last bin also
        holds ``edges[-1]`` (the ``numpy.histogram`` convention).
    fractions : numpy.ndarray
        Fraction of the window's time spent in each bin. Sums to 1 when the
        edges span every value in the window (always, for an integer
        ``bins``); time spent outside explicit edges is not counted.
    """
    edges: np.ndarray
    fractions: np.ndarray


def _histogram_edges(bins, yw: np.ndarray) -> np.ndarray:
    """``bins`` as an edge array: a positive int spans ``yw``'s range
    (``numpy.histogram_bin_edges``); a sequence must be finite and
    strictly increasing. Anything else is a caller error."""
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError(
                f'histogram: bins must be a positive integer or a sequence '
                f'of edges, got {bins!r}')
        if np.any(np.isnan(yw)):
            return np.full(int(bins) + 1, np.nan)
        return np.histogram_bin_edges(yw, bins=int(bins))
    if isinstance(bins, (str, bool)) or np.ndim(bins) != 1:
        raise ValueError(
            f'histogram: bins must be a positive integer or a 1-D sequence '
            f'of edges, got {bins!r}')
    edges = np.asarray(bins, dtype=float)
    if (len(edges) < 2 or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges) <= 0)):
        raise ValueError(
            f'histogram: bin edges must be at least two finite, strictly '
            f'increasing values, got {bins!r}')
    return edges


def histogram(sig: Transient, *, bins, start: Optional[float] = None,
              end: Optional[float] = None) -> HistogramResult:
    """Fraction of time ``sig`` spends in each y-value bin over
    ``[start, end]``.

    Time-weighted, not a sample count: SPICE crowds samples onto fast
    edges, and counting them would over-weight the edges. The waveform is
    linear between samples, so each sample interval spreads its duration
    uniformly over the y range it sweeps — an interval crossing several
    bins splits its time among them in proportion to the y span in each.
    An interval with a constant value puts all its time in the bin holding
    that value.

    Parameters
    ----------
    sig : Signal
        Matching one-dimensional x/y arrays, with finite strictly increasing
        time coordinates in seconds (``xlabel='time'``). A complex ``y`` reads as its
        magnitude.
    bins : int or sequence of float
        A positive integer gives that many equal-width bins spanning the
        window's minimum to maximum value (``numpy.histogram_bin_edges``;
        a constant window gets the range ``value ± 0.5``). A sequence is
        the bin edges themselves, finite and strictly increasing.
    start, end : float, optional
        Bounds in seconds, following :func:`crossings`: a bound
        between samples is sampled by linear interpolation, and a window
        that does not overlap the trace raises ``ValueError``.

    Returns
    -------
    HistogramResult
        One-dimensional edges in input y units, of length B + 1, and
        dimensionless fractions of length B, where B is the bin count.
        It is not a time waveform: its axis is amplitude.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    Resolution: exact for the piecewise-linear waveform through the
    samples; the true waveform's curvature between samples is not seen.
    A NaN among the consumed samples gives all-``nan`` fractions (and
    ``nan`` edges for an integer ``bins``) with a ``nan_waveform``
    ``MetricWarning``. Not a registered metric.

    Invalid bins, empty/wrong-axis input, malformed coordinate arrays, a
    backwards/non-overlapping window or zero duration raises ValueError.

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals import Signal
    >>> from silk.metrics import histogram
    >>> sig = Signal('ramp', 'voltage', 'time', 'V',
    ...              np.array([0., 2.]), np.array([0., 2.]))
    >>> result = histogram(sig, bins=[0., 1., 2.])
    >>> result.fractions.tolist()
    [0.5, 0.5]
    """
    _check_transient(sig, 'histogram')
    lo, hi = _crossing_window(sig, start, end, 'histogram')
    x = np.asarray(sig.x, dtype=float)
    if _crossing_data_has_nan(
        sig, metric='histogram', signal_role='sig', purpose='histogram',
        start_idx=lo, stop_idx=hi + 1,
    ):
        edges = _histogram_edges(bins, np.array([np.nan]))
        return HistogramResult(edges, np.full(len(edges) - 1, np.nan))
    y = _magnitude_samples(sig)
    xw, yw = _interpolated_window(x, y, lo, hi, start, end)
    edges = _histogram_edges(bins, yw)
    duration = float(xw[-1] - xw[0])
    if duration <= 0:
        raise ValueError(
            f'histogram: the window has zero duration; {_axis_context(sig)}')
    dt = np.diff(xw)
    y_lo = np.minimum(yw[:-1], yw[1:])
    y_hi = np.maximum(yw[:-1], yw[1:])
    sloped = y_hi > y_lo
    # Sloped intervals: time spent below each edge, differenced into bins.
    dt_s, lo_s, span_s = dt[sloped], y_lo[sloped], (y_hi - y_lo)[sloped]
    below = np.array([np.sum(dt_s * np.clip((e - lo_s) / span_s, 0.0, 1.0))
                      for e in edges])
    time = np.diff(below)
    # Constant intervals: the bin holding the value (last bin closed).
    flat_y, flat_dt = y_lo[~sloped], dt[~sloped]
    idx = np.searchsorted(edges, flat_y, side='right') - 1
    idx[flat_y == edges[-1]] = len(edges) - 2
    inside = (idx >= 0) & (idx < len(edges) - 1)
    time += np.bincount(idx[inside], weights=flat_dt[inside],
                        minlength=len(edges) - 1)
    return HistogramResult(edges, time / duration)


def _error_series(sig, goal, kind, metric_name):
    """Shared front half of :func:`error_norm` / :func:`relative_error_norm`.

    Validates, resamples ``goal`` onto ``sig.x`` and returns
    ``(sig_x, |sig - goal|, |goal resampled|)``, or ``(None, None, None)``
    after a ``MetricWarning`` when the consumed data contains NaN.
    """
    # error_norm keeps its OWN empty guard rather than routing through
    # _check_nonempty (TASK-77): it takes two Signals and its message
    # says WHICH of them was empty, which the shared one-Signal message
    # cannot. Same contract — typed ValueError naming the metric.
    sig_x = np.asarray(sig.x, dtype=float)
    sig_y = np.asarray(sig.y)
    goal_x = np.asarray(goal.x, dtype=float)
    goal_y = np.asarray(goal.y)
    if sig_x.size == 0 or goal_x.size == 0:
        raise ValueError(
            f'{metric_name}: sig and goal must each contain at least one '
            f'sample (got len(sig)={sig_x.size}, len(goal)={goal_x.size})'
        )
    sig_axis = _axis(sig, metric_name)
    goal_axis = _axis(goal, metric_name)
    sig_x, sig_y = sig_axis.increasing, sig_axis.orient(sig_y)
    if kind not in ('max', 'rms', 'mean_abs'):
        raise ValueError(
            f"{metric_name}: kind must be 'max'|'rms'|'mean_abs', got {kind!r}"
        )
    sig_nan = np.isnan(sig_y)
    # Interpolate contamination on the same support as the reference values.
    # Checking before taking magnitudes also catches complex(inf, nan).
    goal_nan = np.isnan(goal_axis.interp(
        sig_x, np.where(np.isnan(goal_y), np.nan, 0.0),
    ))
    nan_count = int(np.count_nonzero(sig_nan | goal_nan))
    if nan_count:
        emit(
            metric=metric_name, code='nan_waveform',
            message='Measurement unavailable because the consumed data contains NaNs.',
            details={
                'kind': kind, 'sig_name': sig.name, 'goal_name': goal.name,
                'sample_count': len(sig_y), 'nan_sample_count': nan_count,
                'sig_nan_sample_count': int(np.count_nonzero(sig_nan)),
                'goal_nan_sample_count': int(np.count_nonzero(goal_nan)),
                'x_min': float(np.min(sig_x)), 'x_max': float(np.max(sig_x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'sig_y_unit': sig.ylabel, 'goal_y_unit': goal.ylabel,
            },
            suggestion='Check the source result for the affected waveform samples.',
        )
        return None, None, None
    sig_y = np.asarray(sig_y, dtype=complex if np.iscomplexobj(sig_y) else float)
    goal_y = np.asarray(goal_y, dtype=complex if np.iscomplexobj(goal_y) else float)
    goal_resampled = goal_axis.interp(sig_x, goal_y)
    err = sig_y - goal_resampled
    err_mag = np.abs(err)
    return sig_x, err_mag, np.abs(goal_resampled)


def _norm_of(x: np.ndarray, e: np.ndarray, kind: str) -> float:
    """``kind`` norm of the non-negative error samples ``e`` over ``x``."""
    if kind == 'max':
        return float(np.max(e))
    span = x[-1] - x[0]
    if kind == 'mean_abs':
        return float(trapezoid(e, x) / span)
    return float(np.sqrt(trapezoid(e * e, x) / span))


@metric(domain='tran', unit='[y]', param_units={'start': '[x]', 'end': '[x]'})
def error_norm(sig: AnySignal, goal: ReferenceWaveform, *,
               kind: Literal['max', 'rms', 'mean_abs'] = 'rms',
               start: Optional[float] = None,
               end: Optional[float] = None) -> float:
    """Norm of (sig − goal).

    ``goal`` is interpolated onto ``sig.x`` so the two need not share a
    grid. Common use: optimisation-objective evaluation against a
    reference waveform. For complex inputs, the real and imaginary parts
    of ``goal`` are interpolated separately, then ``err = sig_y -
    goal_resampled`` is formed. Every ``kind`` measures ``abs(err)``.
    ``rms`` and ``mean_abs`` are time-weighted (trapezoid over the
    window, like :func:`rms`), so they are unbiased on SPICE's uneven
    grid; a one-sample ``sig`` with no window returns ``|err|``. Magnitudes are never taken before interpolation or
    separately compared between the two waveforms. This operation order is
    deliberate: the complex goal is interpolated, then compared by complex
    difference. The magnitude measures the size of the complex error,
    not a projection of either input.

    Parameters
    ----------
    sig : Signal
        Measured waveform with matching one-dimensional x/y arrays on a
        finite strictly monotonic axis; real and complex y are supported.
    goal : Signal
        Target waveform with matching one-dimensional x/y arrays on a
        finite strictly monotonic axis. Supply the same x/y units as sig;
        labels are not checked or converted. Outside its coverage,
        interpolation holds the nearest endpoint value.
    kind : str, default ``'rms'``
        ``'max'``  — maximum absolute error.
        ``'rms'``  — time-weighted RMS error, ``√(∫ e² dx / (b - a))``.
        ``'mean_abs'`` — time-weighted mean absolute error (L1),
        ``∫ |e| dx / (b - a)``.
    start, end : float, optional
        Bounds in input x units; restrict to samples with ``start <= x <= end``
        (default: the whole trace). NaN checking covers the whole trace,
        not just the window.

    Returns
    -------
    float
        Error norm in the same units as ``sig.y``. A NaN at any compared
        sample in either waveform returns ``nan`` with a ``MetricWarning``
        reporting the contaminated comparison count. Unused reference
        samples do not invalidate the norm.

    Raises
    ------
    ValueError
        If ``sig`` or ``goal`` has no samples. silk checks this itself
        rather than leaning on ``np.interp``: numpy < 2.5 raised on an
        empty sample-point array, numpy >= 2.5 returns an empty array and
        the norm silently became ``nan``.
        If ``kind`` is not one of ``'max'``, ``'rms'`` or ``'mean_abs'``, or
        the window is backwards or holds fewer than two samples, for every
        kind, unless a prior waveform-NaN exit has already returned. A one-sample ``sig`` with no explicit window returns the
        absolute error.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    A NaN in the compared full record returns before explicit window
    validation. Window bounds therefore do not exclude NaNs from checking.
    """
    sig_x, err_mag, _ = _error_series(sig, goal, kind, 'error_norm')
    if sig_x is None:
        return float('nan')
    if start is None and end is None and sig_x.size == 1:
        return float(err_mag[0])
    # The shared window core (same contract as rms()): backwards or
    # too-narrow windows raise ValueError naming error_norm.
    x, e = _window(Signal(sig.name, sig.type, sig.xlabel, sig.ylabel,
                          sig_x, err_mag), start, end, 'error_norm', interpolate=False)
    return _norm_of(x, e, kind)


@metric(domain='tran', unit='', param_units={'floor': '[y]', 'start': '[x]',
                                              'end': '[x]'})
def relative_error_norm(sig: AnySignal, goal: ReferenceWaveform, *,
                        kind: Literal['max', 'rms', 'mean_abs'],
                        floor: float,
                        start: Optional[float] = None,
                        end: Optional[float] = None) -> float:
    """Dimensionless norm of ``|sig - goal| / |goal|``.

    The relative companion of :func:`error_norm` (which cannot express a
    dimensionless result: its unit is the input's y unit). It shares
    ``error_norm``'s interpolation of ``goal`` onto ``sig.x``, window and
    norm definitions. Complex inputs remain supported: the magnitude measures
    the complex difference, not an implicit projection of the inputs. Points where ``|goal| < floor`` are excluded, so a
    goal crossing zero does not divide by (nearly) zero; the norm is taken
    over the remaining points of the window, the trapezoid bridging the
    excluded gaps. A single remaining point returns its own relative error.

    Parameters
    ----------
    sig, goal : Signal
        Matching one-dimensional x/y arrays on finite strictly monotonic
        axes, in compatible x/y units; real and complex y are supported.
        Goal values are interpolated onto sig coordinates, holding the
        endpoint outside goal coverage.
    kind : {'max', 'rms', 'mean_abs'}
        As for :func:`error_norm`, applied to the relative error.
    floor : float
        Exclusion threshold in ``goal``'s y unit; points with
        ``|goal| < floor`` are ignored. Required, ``>= 0``. Points where the goal is exactly
        zero are always excluded, so ``0`` excludes only those.
    start, end : float, optional
        Bounds in input x units, as for :func:`error_norm`.

    Returns
    -------
    float
        Dimensionless relative error. ``nan`` with a ``MetricWarning``
        naming ``floor`` when every windowed point is excluded, or when the
        consumed data contains NaN.

    Raises
    ------
    ValueError
        As for :func:`error_norm`; also if ``floor`` is negative or NaN.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    A NaN in the compared full record returns before explicit window
    validation. Window bounds therefore do not exclude NaNs from checking.
    """
    if not floor >= 0:
        raise ValueError(
            f'relative_error_norm: floor must be >= 0, got {floor!r}'
        )
    sig_x, err_mag, goal_mag = _error_series(sig, goal, kind,
                                             'relative_error_norm')
    if sig_x is None:
        return float('nan')
    if start is None and end is None and sig_x.size == 1:
        x, e, g = sig_x, err_mag, goal_mag
    else:
        x, e = _window(Signal(sig.name, sig.type, sig.xlabel, sig.ylabel,
                              sig_x, err_mag), start, end,
                       'relative_error_norm', interpolate=False)
        g = goal_mag[(sig_x >= x[0]) & (sig_x <= x[-1])]
    keep = (g >= floor) & (g > 0)   # a zero goal has no relative error
    if not keep.any():
        emit(
            metric='relative_error_norm', code='all_points_excluded',
            message=f'Every point has |goal| < floor ({floor:g}); no relative '
                    'error could be formed.',
            details={'kind': kind, 'floor': float(floor),
                     'sample_count': int(len(x)),
                     'goal_abs_max': float(np.max(g)),
                     'goal_y_unit': goal.ylabel},
            suggestion='Lower floor, or widen the window to include larger '
                       '|goal| samples.',
        )
        return float('nan')
    x, rel = x[keep], e[keep] / g[keep]
    if x.size == 1:
        return float(rel[0])
    return _norm_of(x, rel, kind)
