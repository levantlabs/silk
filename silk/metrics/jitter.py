"""Clock jitter and stability measurements on time waveforms.

Period and cycle-to-cycle jitter, crossing jitter, time-interval error (TIE),
long-term jitter and Allan deviation share linearly interpolated crossings.
Each function documents its reference, units and data-resolution limits.
"""
from __future__ import annotations

from typing import Literal, Optional

import math

import numpy as np

from silk.signals.base import Signal
from silk.signals.schema import metric, Transient
from silk.metrics._guards import _check_nonempty, _any_nan
from silk.metrics.diagnostics import emit
from silk.metrics.timing import (
    _check_timing_axis,
    _cycle_edges,
    _cycle_periods,
    _nan_signal,
)
from silk.metrics.waveform import (
    _axis_context,
    _check_edge,
    _check_transient,
    _crossing_data_has_nan,
    _crossing_window,
    _magnitude_samples,
    _select_crossings,
    _timing_details,
    crossings,
)


#: The per-cycle period reference for the period-jitter family.
_JITTER_REFERENCES = ('mean', 'moving_average')


#: ``param_units`` shared by the four period-jitter scalars.
_JITTER_PARAM_UNITS = {'threshold': '[y]', 'start': 's', 'end': 's'}


def _check_jitter_reference(reference, cycles, metric: str) -> None:
    """Raise ``ValueError`` for a malformed ``reference`` / ``cycles`` pair.

    ``'mean'`` takes no ``cycles``; ``'moving_average'`` needs an integer
    ``cycles >= 2``. Both are caller errors.
    """
    if reference not in _JITTER_REFERENCES:
        raise ValueError(
            f"{metric}: reference must be 'mean' or 'moving_average', "
            f"got {reference!r}"
        )
    if reference == 'mean':
        if cycles is not None:
            raise ValueError(
                f"{metric}: cycles applies only to "
                f"reference='moving_average'; got cycles={cycles!r} with "
                f"reference='mean'"
            )
        return
    if (isinstance(cycles, (bool, np.bool_))
            or not isinstance(cycles, (int, np.integer)) or cycles < 2):
        raise ValueError(
            f"{metric}: reference='moving_average' needs cycles, an integer "
            f">= 2 (the number of periods averaged), got {cycles!r}"
        )


def _moving_average_kernel(cycles: int) -> np.ndarray:
    """The centred moving average over ``cycles`` periods, as weights.

    Odd ``cycles``: ``cycles`` equal weights centred on the period. Even
    ``cycles``: ``cycles + 1`` taps with half weight at both ends, the
    classical centred average of even order. Either way the kernel is
    symmetric about its centre, so a linear drift of the period passes
    through unchanged and its deviation from the reference is exactly 0.
    """
    if cycles % 2:
        return np.full(cycles, 1.0 / cycles)
    kernel = np.full(cycles + 1, 1.0 / cycles)
    kernel[0] = kernel[-1] = 0.5 / cycles
    return kernel


def _jitter_core(clock: Signal, threshold, edge, start, end, reference,
                 cycles, *, metric: str, form: str, min_points: int):
    """The one per-cycle core of the period-jitter family.

    Periods come from :func:`_cycle_periods` over :func:`_cycle_edges` (the crossings contract and full cycle). Each period
    ``P_k`` has a deviation ``d_k = P_k - R_k`` from its reference: the
    mean of every period in the window (``'mean'``), or the centred
    moving average over ``cycles`` periods (``'moving_average'``, defined
    only where the full average fits, so the first and last
    ``cycles // 2`` periods have no deviation).

    ``form='period'`` returns the deviations, stamped at the midpoint of
    each period's two edges like :func:`cycle_period`. ``form='c2c'``
    returns the difference between each cycle's deviation and the next
    cycle's — the cycle that starts on the edge this one ends on (for
    ``edge='any'`` two crossings on) — stamped at that shared edge.

    Returns ``(times, values)``, or ``None`` after the ``nan_waveform``
    warning when a consumed sample is NaN. Fewer than ``min_points``
    values emits ``insufficient_edges`` with the edge count the request
    needs; the arrays are then returned as they are (possibly empty).
    """
    _check_nonempty(clock, metric)
    _check_jitter_reference(reference, cycles, metric)
    _check_transient(clock, metric)
    _check_edge(edge, metric)
    _crossing_window(clock, start, end, metric)
    if start is not None and end is not None and end <= start:
        # A zero-width window holds at most one edge, so no jitter request
        # can be met by it: a caller error, stricter than crossings().
        raise ValueError(
            f'{metric}: require end > start, got start={start}, end={end}; '
            f'{_axis_context(clock)}'
        )
    edges = _cycle_edges(clock, threshold, edge, start, end, metric=metric)
    if edges is None:
        return None
    step = 2 if edge == 'any' else 1
    mid, periods = _cycle_periods(edges, edge)
    index = np.arange(len(periods))
    if reference == 'mean':
        taps = 1
        if len(periods) >= 2:
            dev = periods - np.mean(periods)
        else:
            index, dev = index[:0], periods[:0]
    else:
        kernel = _moving_average_kernel(int(cycles))
        taps = len(kernel)
        half = taps // 2
        if len(periods) >= taps:
            index = index[half:len(periods) - half]
            dev = periods[index] - np.convolve(periods, kernel, mode='valid')
        else:
            index, dev = index[:0], periods[:0]
    if form == 'period':
        times, values = mid[index], dev
        needed_dev = min_points
    else:
        times = edges[index[:-step] + step] if len(index) > step else edges[:0]
        values = dev[step:] - dev[:-step]
        needed_dev = min_points + step
    if len(values) < min_points:
        needed_periods = (max(needed_dev, 2) if reference == 'mean'
                          else needed_dev + taps - 1)
        emit(
            metric=metric, code='insufficient_edges',
            message=('Too few edges were observed for the requested '
                     'jitter reference.'),
            details={**_timing_details(clock), 'threshold': threshold,
                     'value_unit': clock.ylabel, 'edge': edge,
                     'window_start': start, 'window_end': end,
                     'reference': reference, 'cycles': cycles,
                     'edge_count': len(edges),
                     'required_edge_count': needed_periods + step},
            suggestion=('Inspect the clock and threshold, widen the window or '
                        'capture a longer record'
                        + (', or average fewer cycles.'
                           if reference == 'moving_average' else '.')),
        )
    return times, values


def _jitter_scalar(clock, threshold, edge, start, end, reference, cycles,
                   *, metric: str, form: str, statistic: str) -> float:
    found = _jitter_core(clock, threshold, edge, start, end, reference,
                         cycles, metric=metric, form=form, min_points=2)
    if found is None:
        return float('nan')
    _, values = found
    if len(values) < 2:
        return float('nan')
    if statistic == 'rms':
        return float(np.sqrt(np.mean(np.square(values))))
    return float(np.max(values) - np.min(values))


def _jitter_waveform(clock, threshold, edge, start, end, reference, cycles,
                     *, metric: str, form: str) -> Signal:
    name = f'{metric}({clock.name})'
    type_ = 'period_deviation' if form == 'period' else 'period_difference'
    found = _jitter_core(clock, threshold, edge, start, end, reference,
                         cycles, metric=metric, form=form, min_points=1)
    if found is None:
        return _nan_signal(name, type_, 'time', 's')
    times, values = found
    return Signal(name, type_, 'time', 's', times, values)


@metric(domain='tran', unit='s', param_units=_JITTER_PARAM_UNITS)
def period_jitter(clock: Transient, *,
                  threshold: Optional[float] = None,
                  edge: Literal['rising', 'falling', 'any'] = 'rising',
                  start: Optional[float] = None,
                  end: Optional[float] = None,
                  reference: Literal['mean', 'moving_average'] = 'mean',
                  cycles: Optional[int] = None) -> float:
    """RMS period jitter in seconds: the RMS of :func:`period_deviation`.

    Each full cycle's period minus its reference, ``P_k - R_k``, squared,
    averaged and rooted: ``sqrt(mean((P_k - R_k)**2))``. With
    ``reference='mean'`` that is the population standard deviation of
    the periods (``ddof=0``) and counts slow frequency drift as jitter;
    ``reference='moving_average'`` with ``cycles=N`` measures each period
    against the centred average of the ``N`` periods around it, which
    removes drift slower than about ``N`` cycles (a linear drift exactly).

    Parameters
    ----------
    clock : Signal
        Transient clock: matching one-dimensional x/y arrays, finite
        strictly increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in clock y units. ``None`` is the clock's mid-swing,
        ``(min + max) / 2`` of the samples inside the window — the
        :func:`crossings` contract.
    edge : str, default ``'rising'``
        ``'rising'``, ``'falling'`` or ``'any'``. A period is one full
        cycle — rising to rising, falling to falling, and for ``'any'``
        each crossing to the next crossing of the same direction — as
        for :func:`cycle_period`.
    start, end : float, optional
        Bounds in seconds. A period counts only when both of its edges lie in
        ``[start, end]``. Every edge time is already interpolated between
        samples; the window only chooses which edges count.
    reference : str, default ``'mean'``
        ``'mean'`` or ``'moving_average'``.
    cycles : int, optional
        The moving-average length in periods, an integer ``>= 2``;
        required with ``reference='moving_average'`` and refused with
        ``'mean'``. Odd ``N`` averages ``N`` periods centred on the one
        measured; even ``N`` uses ``N + 1`` periods with half weight on
        the two end ones. Only periods with a full average around them
        are measured, so the first and last ``N // 2`` periods drop out.

    Returns
    -------
    float
        Jitter in seconds; nan with MetricWarning for insufficient deviations or
        consumed NaNs. The statistic and reference are defined above.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    Invalid reference/cycles combinations raise ValueError before measuring.


    Returns ``nan`` with an ``insufficient_edges`` ``MetricWarning`` when
    fewer than two deviations exist (three edges for the defaults), and
    with ``nan_waveform`` when a consumed sample is NaN. An invalid
    ``edge``, window, ``reference`` or ``cycles`` raises ``ValueError``.
    Resolution: the interpolated-crossing floor of :func:`crossings`
    (Δt²·``|y''|``/(8·``|y'|``) per edge) — jitter below it is numerical.
    """
    return _jitter_scalar(clock, threshold, edge, start, end, reference,
                          cycles, metric='period_jitter', form='period',
                          statistic='rms')


@metric(domain='tran', unit='s', param_units=_JITTER_PARAM_UNITS)
def period_jitter_pp(clock: Transient, *,
                     threshold: Optional[float] = None,
                     edge: Literal['rising', 'falling', 'any'] = 'rising',
                     start: Optional[float] = None,
                     end: Optional[float] = None,
                     reference: Literal['mean', 'moving_average'] = 'mean',
                     cycles: Optional[int] = None) -> float:
    """Peak-to-peak period jitter in seconds: ``max - min`` of
    :func:`period_deviation`.

    Same periods, reference and arguments as :func:`period_jitter`, and
    the same ``nan`` cases and ``ValueError`` s. With
    ``reference='mean'`` it is the spread of the periods,
    ``max(P) - min(P)``.

    Parameters
    ----------
    clock : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    reference : {'mean', 'moving_average'}
        Use a global mean period or a centred local period average.
    cycles : int, optional
        Moving reference length in periods, >= 2; required for moving_average and
        refused for mean. Odd lengths use that many periods; even lengths use one
        extra with half-weighted endpoints.

    Returns
    -------
    float
        Jitter in seconds; nan with MetricWarning for insufficient deviations or
        consumed NaNs. The statistic and reference are defined above.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    Invalid reference/cycles combinations raise ValueError before measuring.
    """
    return _jitter_scalar(clock, threshold, edge, start, end, reference,
                          cycles, metric='period_jitter_pp', form='period',
                          statistic='pp')


@metric(domain='tran', unit='s', param_units=_JITTER_PARAM_UNITS)
def cycle_to_cycle_jitter(clock: Transient, *,
                          threshold: Optional[float] = None,
                          edge: Literal['rising', 'falling', 'any'] = 'rising',
                          start: Optional[float] = None,
                          end: Optional[float] = None,
                          reference: Literal['mean', 'moving_average'] = 'mean',
                          cycles: Optional[int] = None) -> float:
    """RMS cycle-to-cycle jitter in seconds: the RMS of
    :func:`cycle_to_cycle_difference`.

    Each cycle's period deviation subtracted from the next cycle's,
    ``c_k = d_{k+1} - d_k`` with ``d_k = P_k - R_k`` as in
    :func:`period_jitter`, then ``sqrt(mean(c_k**2))``. It is a true RMS,
    not a standard deviation: with ``reference='mean'`` (where
    ``c_k = P_{k+1} - P_k``) a steady frequency drift of ``s`` seconds
    per cycle reads ``|s|``. ``reference='moving_average'`` takes the
    drift out of both periods first, so the same clock reads 0.

    Same arguments, edges and windows as :func:`period_jitter`; for
    ``edge='any'`` the next cycle is two crossings on (the one starting
    where this one ends). Returns ``nan`` with an ``insufficient_edges``
    ``MetricWarning`` when fewer than two differences exist (four edges
    for the defaults), and with ``nan_waveform`` when a consumed sample
    is NaN. An invalid ``edge``, window, ``reference`` or ``cycles``
    raises ``ValueError``.

    Parameters
    ----------
    clock : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    reference : {'mean', 'moving_average'}
        Use a global mean period or a centred local period average.
    cycles : int, optional
        Moving reference length in periods, >= 2; required for moving_average and
        refused for mean. Odd lengths use that many periods; even lengths use one
        extra with half-weighted endpoints.

    Returns
    -------
    float
        Jitter in seconds; nan with MetricWarning for insufficient deviations or
        consumed NaNs. The statistic and reference are defined above.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    Invalid reference/cycles combinations raise ValueError before measuring.
    """
    return _jitter_scalar(clock, threshold, edge, start, end, reference,
                          cycles, metric='cycle_to_cycle_jitter', form='c2c',
                          statistic='rms')


@metric(domain='tran', unit='s', param_units=_JITTER_PARAM_UNITS)
def cycle_to_cycle_jitter_pp(clock: Transient, *,
                             threshold: Optional[float] = None,
                             edge: Literal['rising', 'falling', 'any'] = 'rising',
                             start: Optional[float] = None,
                             end: Optional[float] = None,
                             reference: Literal['mean', 'moving_average'] = 'mean',
                             cycles: Optional[int] = None) -> float:
    """Peak-to-peak cycle-to-cycle jitter in seconds: ``max - min`` of
    :func:`cycle_to_cycle_difference`.

    Same differences, arguments, ``nan`` cases and ``ValueError`` s as
    :func:`cycle_to_cycle_jitter`. For the largest single cycle-to-cycle
    step, read ``max(abs(cycle_to_cycle_difference(...).y))``.

    Parameters
    ----------
    clock : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    reference : {'mean', 'moving_average'}
        Use a global mean period or a centred local period average.
    cycles : int, optional
        Moving reference length in periods, >= 2; required for moving_average and
        refused for mean. Odd lengths use that many periods; even lengths use one
        extra with half-weighted endpoints.

    Returns
    -------
    float
        Jitter in seconds; nan with MetricWarning for insufficient deviations or
        consumed NaNs. The statistic and reference are defined above.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    Invalid reference/cycles combinations raise ValueError before measuring.
    """
    return _jitter_scalar(clock, threshold, edge, start, end, reference,
                          cycles, metric='cycle_to_cycle_jitter_pp',
                          form='c2c', statistic='pp')


def period_deviation(clock: Signal, *,
                     threshold: Optional[float] = None,
                     edge: Literal['rising', 'falling', 'any'] = 'rising',
                     start: Optional[float] = None,
                     end: Optional[float] = None,
                     reference: Literal['mean', 'moving_average'] = 'mean',
                     cycles: Optional[int] = None) -> Signal:
    """Per-cycle period deviation as a waveform: seconds versus time.

    One point per measured period, ``P_k - R_k`` (see
    :func:`period_jitter` for the periods, the reference and every
    argument), at the midpoint time of the period's two edges — the same
    times as :func:`cycle_period`. :func:`period_jitter` is its RMS and
    :func:`period_jitter_pp` its peak-to-peak.

    Parameters
    ----------
    clock : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    reference : {'mean', 'moving_average'}
        Use a global mean period or a centred local period average.
    cycles : int, optional
        Moving reference length in periods, >= 2; required for moving_average and
        refused for mean. Odd lengths use that many periods; even lengths use one
        extra with half-weighted endpoints.

    Returns
    -------
    Signal
        ``xlabel='time'``, ``y`` in seconds. Empty, with an
        ``insufficient_edges`` ``MetricWarning``, when no deviation can be
        formed (fewer than two periods for ``'mean'``, fewer than the
        moving-average length); a one-sample ``nan`` Signal with
        ``nan_waveform`` when a consumed sample is NaN.

    Notes
    -----
    Returned x and y arrays are one-dimensional with matching lengths;
    time coordinates are in seconds.
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    Invalid reference/cycles combinations raise ValueError before measuring.
    """
    return _jitter_waveform(clock, threshold, edge, start, end, reference,
                            cycles, metric='period_deviation', form='period')


def cycle_to_cycle_difference(clock: Signal, *,
                              threshold: Optional[float] = None,
                              edge: Literal['rising', 'falling', 'any'] = 'rising',
                              start: Optional[float] = None,
                              end: Optional[float] = None,
                              reference: Literal['mean', 'moving_average'] = 'mean',
                              cycles: Optional[int] = None) -> Signal:
    """Per-cycle cycle-to-cycle difference as a waveform: seconds versus
    edge time.

    One point per pair of adjacent cycles, ``d_{k+1} - d_k`` (see
    :func:`cycle_to_cycle_jitter`), stamped at the edge the two cycles
    share. :func:`cycle_to_cycle_jitter` is its RMS and
    :func:`cycle_to_cycle_jitter_pp` its peak-to-peak.

    Parameters
    ----------
    clock : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    reference : {'mean', 'moving_average'}
        Use a global mean period or a centred local period average.
    cycles : int, optional
        Moving reference length in periods, >= 2; required for moving_average and
        refused for mean. Odd lengths use that many periods; even lengths use one
        extra with half-weighted endpoints.

    Returns
    -------
    Signal
        ``xlabel='time'``, ``y`` in seconds. Empty, with an
        ``insufficient_edges`` ``MetricWarning``, when no pair of measured
        cycles exists; a one-sample ``nan`` Signal with ``nan_waveform``
        when a consumed sample is NaN.

    Notes
    -----
    Returned x and y arrays are one-dimensional with matching lengths;
    time coordinates are in seconds.
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    Invalid reference/cycles combinations raise ValueError before measuring.
    """
    return _jitter_waveform(clock, threshold, edge, start, end, reference,
                            cycles, metric='cycle_to_cycle_difference',
                            form='c2c')


@metric(domain='tran', unit='s',
        param_units={'period': 's', 'threshold': '[y]',
                     'start': 's', 'end': 's'})
def crossing_jitter_rms(sig: Transient, *,
                        period: float,
                        threshold: Optional[float] = None,
                        edge: Literal['rising', 'falling', 'any'] = 'rising',
                        start: Optional[float] = None,
                        end: Optional[float] = None) -> float:
    """RMS deviation of every crossing from an ideal crossing grid.

    The data-signal jitter metric. Each crossing
    ``t_k = crossings(sig, threshold=, edge=, start=, end=)`` is folded
    onto the grid ``mu + n·period`` with no edge index: its phase
    ``t_k mod period`` is all that is used, so runs of equal bits —
    missing crossings — do not inflate the result. ``mu`` is the
    circular mean of those phases; each deviation is wrapped to
    ``[-period/2, period/2)`` around it (so a cluster near phase 0 is
    not split) and the result is ``std(deviations, ddof=1)``.

    Contrast: :func:`period_jitter` and :func:`cycle_to_cycle_jitter`
    difference CONSECUTIVE edges, and :func:`long_term_jitter` places
    edge k at ``edge0 + k·T`` — all three assume one edge per period, which
    a data pattern breaks.

    Parameters
    ----------
    sig : Signal
        Transient data: matching one-dimensional x/y arrays, finite
        strictly increasing time in seconds and xlabel="time".
    period : float
        The unit interval in seconds — required; no period is fitted. A
        frequency offset between ``sig`` and ``period`` reads as phase
        drift and so counts as jitter; once drift exceeds ``period/2``
        the wrap aliases it.
    threshold : float, optional
        Crossing level in input y units; ``None`` is the midpoint of the swing inside the
        window, as for :func:`crossings`.
    edge : str, default ``'rising'``
        ``'rising'``, ``'falling'`` or ``'any'``. ``'any'`` pools both
        directions on ONE grid, so a constant rising/falling offset
        (duty-cycle distortion, asymmetric edges, an off-centre
        threshold) counts as jitter; with one direction it does not.
    start, end : float, optional
        Bounds in seconds, exactly as for :func:`crossings` —
        use them to exclude a start-up transient.

    Returns
    -------
    float
        Sample standard deviation of wrapped crossing deviations in seconds; nan
        with MetricWarning for fewer than two crossings or consumed NaNs.

    Notes
    -----
    The input Signal objects and their arrays are not modified.


    Returns ``nan`` with a ``MetricWarning`` if fewer than 2 crossings
    are selected; contaminated data retains the single warning from
    :func:`crossings`. An invalid ``edge``, window or ``period`` (not
    finite, or ``<= 0``) raises ``ValueError``.
    Resolution: the interpolated-crossing floor of :func:`crossings`.
    """
    _check_nonempty(sig, 'crossing_jitter_rms')
    _check_transient(sig, 'crossing_jitter_rms')
    _check_edge(edge, 'crossing_jitter_rms')
    if (isinstance(period, (bool, np.bool_))
            or not isinstance(period, (int, float, np.integer, np.floating))
            or not math.isfinite(period) or period <= 0):
        raise ValueError(
            f'crossing_jitter_rms: period must be a finite number > 0 s, '
            f'got {period!r}; available {_timing_details(sig)}'
        )
    edges = crossings(sig, threshold=threshold, edge=edge,
                      start=start, end=end)
    if len(edges) < 2:
        if not _any_nan(edges):
            emit(
                metric='crossing_jitter_rms', code='insufficient_edges',
                message='Too few crossings were observed to measure crossing jitter.',
                details={**_timing_details(sig), 'edge': edge,
                         'window_start': start, 'window_end': end,
                         'edge_count': len(edges), 'required_edge_count': 2},
                suggestion=('Inspect the signal, threshold and window or '
                            'capture a longer record.'),
            )
        return float('nan')
    period = float(period)
    angles = 2.0 * np.pi * np.mod(edges, period) / period
    mu = period * np.angle(np.mean(np.exp(1j * angles))) / (2.0 * np.pi)
    dev = np.mod(edges - mu + 0.5 * period, period) - 0.5 * period
    return float(np.std(dev, ddof=1))


def _timing_edges(clock: Signal, metric: str, required: int) -> np.ndarray:
    """Diagnose unavailable clock observations once at the consuming metric."""
    if _crossing_data_has_nan(
        clock, metric=metric, signal_role='clock', purpose='clock_edges',
    ):
        return np.array([float('nan')])
    edges = crossings(clock, edge='rising')
    if len(edges) < required:
        emit(
            metric=metric, code='insufficient_edges',
            message='Too few rising edges were observed for this measurement.',
            details={**_timing_details(clock), 'edge_count': len(edges),
                     'required_edge_count': required},
            suggestion='Inspect the clock or capture a longer record.',
        )
    return edges


#: The TIE output units: seconds, or a fraction of the reference
#: period expressed as unit intervals, radians or degrees.
_TIE_UNITS = {'s': None, 'ui': 1.0, 'rad': 2.0 * math.pi, 'deg': 360.0}


_TIE_REFERENCES = ('fit', 'endpoints')


def _check_tie_request(clock: Signal, units, reference, ideal_period, phase,
                       edge: str, metric: str) -> None:
    """Raise for a malformed TIE request."""
    _check_timing_axis(clock, metric)
    _check_edge(edge, metric)
    if units not in _TIE_UNITS:
        raise ValueError(
            f'{metric}: units must be one of {list(_TIE_UNITS)}, got '
            f'{units!r}'
        )
    if reference not in _TIE_REFERENCES:
        raise ValueError(
            f'{metric}: reference must be one of {list(_TIE_REFERENCES)}, '
            f'got {reference!r}'
        )
    for name, value in (('ideal_period', ideal_period), ('phase', phase)):
        if value is None:
            continue
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or math.isinf(value)):
            # nan passes through to a nan result (an upstream metric that
            # could not measure the period must not abort a sweep).
            raise ValueError(
                f'{metric}: {name} must be a finite number in s or None; '
                f'requested {name}={value!r}; available '
                f'{_timing_details(clock)}'
            )
    if ideal_period is not None and ideal_period <= 0:
        raise ValueError(
            f'{metric}: ideal_period must be > 0 s; requested '
            f'ideal_period={ideal_period!r} s; available {_timing_details(clock)}'
        )


def _tie_edges(clock: Signal, threshold, edge: str, start, end, *,
               metric: str) -> Optional[np.ndarray]:
    """The selected edges, ``None`` when a consumed sample is NaN.

    The :func:`crossings` contract: ``threshold=None`` is the
    midpoint of the swing inside the window and an edge counts iff it lies
    in ``[start, end]``. Fewer than three edges emit
    ``insufficient_edges``: two edges always lie on a fitted line, so their
    TIE carries no information.
    """
    lo, hi = _crossing_window(clock, start, end, metric)
    if threshold is not None:
        threshold = float(threshold)
    if _crossing_data_has_nan(
        clock, metric=metric, signal_role='clock', purpose='clock_edges',
        start_idx=lo, stop_idx=hi + 1,
    ):
        return None
    edges = _select_crossings(
        clock, _magnitude_samples(clock), threshold, edge, None, start, end,
        metric=metric, lo=lo, hi=hi, details=_timing_details(clock),
    )
    if len(edges) < 3:
        emit(
            metric=metric, code='insufficient_edges',
            message='Too few edges were observed to measure time interval error.',
            details={**_timing_details(clock), 'threshold': threshold,
                     'edge': edge, 'window_start': start, 'window_end': end,
                     'edge_count': len(edges), 'required_edge_count': 3},
            suggestion=('Inspect the clock, threshold and window or capture '
                        'a longer record.'),
        )
    return np.asarray(edges, dtype=float)


def _tie_reference(edges: np.ndarray, reference: str, spacing: Optional[float],
                   phase: Optional[float]) -> tuple[float, float]:
    """``(phase, spacing)`` of the ideal line ``phase + k·spacing``.

    ``spacing`` is the ideal interval between consecutive selected edges
    (the period, or half of it for ``edge='any'``). An argument that is
    given is used as is; the rest is derived from the edges — by least
    squares over every edge (``'fit'``), or from the first and last edge
    (``'endpoints'``, the legacy reference, kept bit for bit).
    """
    k = np.arange(len(edges), dtype=float)
    if reference == 'endpoints':
        if spacing is None:
            spacing = float(np.mean(np.diff(edges)))
        return (float(edges[0]) if phase is None else float(phase)), spacing
    # Least squares, relative to the first edge for precision.
    rel = edges - edges[0]
    if spacing is None and phase is None:
        kc = k - np.mean(k)
        spacing = float(np.sum(kc * (rel - np.mean(rel))) / np.sum(kc * kc))
        return float(edges[0]) + float(np.mean(rel)) - spacing * float(np.mean(k)), spacing
    if spacing is None:
        r = rel - (float(phase) - float(edges[0]))
        return float(phase), float(np.sum(k * r) / np.sum(k * k))
    if phase is None:
        return float(edges[0]) + float(np.mean(rel - k * spacing)), spacing
    return float(phase), spacing


def _tie(clock: Signal, *, units: str, reference: str,
         ideal_period: Optional[float], phase: Optional[float],
         threshold: Optional[float], edge: str, start: Optional[float],
         end: Optional[float], metric: str) -> Signal:
    """Share TIE extraction while retaining the invoking metric in diagnostics."""
    _check_tie_request(clock, units, reference, ideal_period, phase, edge,
                       metric)
    name = f'tie({clock.name})'
    edges = _tie_edges(clock, threshold, edge, start, end, metric=metric)
    if edges is None:
        return _nan_signal(name, 'time_error', 'time', units)
    if len(edges) < 3:
        empty = np.array([], dtype=float)
        return Signal(name, 'time_error', 'time', units, empty, empty.copy())
    # With edge='any' consecutive edges are half a period apart.
    per_edge = 0.5 if edge == 'any' else 1.0
    spacing = None if ideal_period is None else float(ideal_period) * per_edge
    phase0, spacing = _tie_reference(edges, reference, spacing, phase)
    indices = np.arange(len(edges), dtype=float)
    tie_vals = edges - (phase0 + indices * spacing)
    scale = _TIE_UNITS[units]
    if scale is not None:
        tie_vals = tie_vals * (scale / (spacing / per_edge))
    return Signal(name, 'time_error', 'time', units, edges, tie_vals)


def tie(clock: Signal, *,
        units: Literal['s', 'ui', 'rad', 'deg'] = 's',
        reference: Literal['fit', 'endpoints'] = 'fit',
        ideal_period: Optional[float] = None,
        phase: Optional[float] = None,
        threshold: Optional[float] = None,
        edge: Literal['rising', 'falling', 'any'] = 'rising',
        start: Optional[float] = None,
        end: Optional[float] = None) -> Signal:
    """Time Interval Error: deviation of each clock edge from its ideal
    time.

    The edges are ``crossings(clock, threshold=, edge=, start=, end=)``.
    Edge ``k`` (``k = 0`` for the first edge in the window) is compared
    with the ideal line ``phase + k·ideal_period`` and the TIE is
    ``edge[k] − (phase + k·ideal_period)``.

    Parameters
    ----------
    clock : Signal
        Transient clock: matching one-dimensional x/y arrays, finite
        strictly increasing time in seconds and xlabel="time".
    units : str, default ``'s'``
        ``'s'`` (seconds), ``'ui'`` (TIE / period), ``'rad'``
        (2π·TIE / period) or ``'deg'`` (360·TIE / period). The period is
        the reference's: ``ideal_period`` when given, else the fitted one.
    reference : str, default ``'fit'``
        How the line's unknowns are derived from the edges. ``'fit'`` is
        the least-squares line through every edge in the window: a
        constant frequency offset and phase are removed and no single
        edge anchors the result. ``'endpoints'`` is the line through the
        first and last edge (slope = mean period), so the TIE is exactly
        0 at both ends and an error on either end edge spreads over the
        whole record. It reproduces silk's legacy results bit
        for bit: the old ``tie(clock)`` is
        ``tie(clock, reference='endpoints')``.
    ideal_period : float, optional
        Overrides the slope (s). The intercept is still derived — the
        least-squares one for ``'fit'``, the first edge for
        ``'endpoints'``.
    phase : float, optional
        Overrides the intercept: the ideal time (s) of the first edge in
        the window. The slope is still derived (for ``'fit'``, the
        least-squares slope through ``phase``). With both overrides the
        reference is fully given and ``reference`` has no effect.
    threshold : float, optional
        Crossing level in clock y units; None uses the window mid-swing.
    edge : {'rising', 'falling', 'any'}
        Select the edges exactly as for :func:`crossings`. With
        ``edge='any'`` consecutive ideal edges are half a period apart, so
        duty-cycle distortion counts as TIE; ``ideal_period`` is still a
        full period.
    start, end : float, optional
        Inclusive edge-selection bounds in seconds; None uses the trace extent.

    Returns
    -------
    Signal
        ``xlabel='time'``, x = edge times (s), y = TIE in ``units``
        (``ylabel`` = ``units``). With a ``MetricWarning``:
        an EMPTY Signal when fewer than 3 edges are selected — two edges
        always lie on the fitted line, so there is nothing to measure —
        and a one-sample ``nan`` Signal when a consumed sample is NaN
        ("unknown" stays distinct from "too few edges"). A
        ``nan`` ``ideal_period`` or ``phase`` gives a ``nan`` TIE without
        a warning. An unknown ``units``/``reference``/``edge``, a
        nonpositive or infinite ``ideal_period``, an infinite ``phase``,
        a malformed window or a non-increasing time axis raises
        ``ValueError``.

    Notes
    -----
    Returned x and y arrays are one-dimensional with matching lengths;
    time coordinates are in seconds.
    The input Signal objects and their arrays are not modified.
    """
    return _tie(clock, units=units, reference=reference,
                ideal_period=ideal_period, phase=phase, threshold=threshold,
                edge=edge, start=start, end=end, metric='tie')


@metric(domain='tran', unit='s',
        param_units={'ideal_period': 's', 'phase': 's', 'threshold': '[y]',
                     'start': 's', 'end': 's'})
def long_term_jitter(clock: Transient, *,
                     units: Literal['s', 'ui', 'rad', 'deg'] = 's',
                     reference: Literal['fit', 'endpoints'] = 'fit',
                     ideal_period: Optional[float] = None,
                     phase: Optional[float] = None,
                     threshold: Optional[float] = None,
                     edge: Literal['rising', 'falling', 'any'] = 'rising',
                     start: Optional[float] = None,
                     end: Optional[float] = None) -> float:
    """RMS of TIE (long-term / accumulated jitter): ``std(tie(...).y, ddof=1)``.

    Captures the total accumulated phase error of the clock over the
    measurement window — *includes* slow drift, unlike
    :func:`period_jitter` (cycle-by-cycle σ_T) and
    :func:`cycle_to_cycle_jitter` (adjacent-cycle σ_ΔT). Every argument
    means what it means for :func:`tie`; the result is in ``units``
    (the catalog unit is the default, ``'s'``). The legacy result
    is ``long_term_jitter(clock, reference='endpoints')``.

    Returns ``nan`` with a ``MetricWarning`` if fewer than 3 edges are
    selected or consumed waveform NaNs prevent extraction. Invalid
    requests raise ``ValueError`` naming this function, as for
    :func:`tie`.

    Parameters
    ----------
    clock : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    units : str
        TIE unit: s, ui, rad or deg; default s.
    reference : {'fit', 'endpoints'}
        Derive the ideal edge line by least squares or from its endpoints; explicit
        period and phase override slope and intercept.
    ideal_period : float, optional
        Full reference period in seconds; None fits it from the selected edges.
    phase : float, optional
        Ideal first-edge time in seconds; None derives the intercept from the
        reference.
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Sample standard deviation of TIE (ddof=1), in the requested units. Fewer
        than three edges or consumed NaNs return nan with MetricWarning; NaN phase
        or period propagates quietly.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    An unknown units/reference/edge, nonpositive or infinite ideal_period,
    infinite phase, or malformed window raises ValueError. NaN period/phase
    are allowed and propagate. The clock must be nonempty with a finite
    strictly increasing time axis.
    """
    tie_sig = _tie(clock, units=units, reference=reference,
                   ideal_period=ideal_period, phase=phase,
                   threshold=threshold, edge=edge, start=start, end=end,
                   metric='long_term_jitter')
    if len(tie_sig.y) < 3:
        return float('nan')
    return float(np.std(tie_sig.y, ddof=1))


def allan_deviation(clock: Signal, *,
                    taus: Optional[np.ndarray] = None,
                    threshold: Optional[float] = None,
                    edge: Literal['rising', 'falling', 'any'] = 'rising',
                    start: Optional[float] = None,
                    end: Optional[float] = None) -> Signal:
    """Non-overlapping Allan deviation σ_y(τ) for a clock waveform.

    Standard clock-stability metric, especially for crystal /
    atomic-reference work. Per-cycle fractional frequency deviations
    ``y_i = (T_i − ⟨T⟩) / ⟨T⟩`` are formed from the clock edges, then
    grouped into bins of size ``m = round(τ / ⟨T⟩)`` and the Allan
    variance is computed as

        σ_y²(τ) = ½ ⟨ (y̅_{i+1} − y̅_i)² ⟩

    averaging over consecutive non-overlapping bin pairs.

    Parameters
    ----------
    clock : Signal
        Transient clock: matching one-dimensional x/y arrays, finite
        strictly increasing time in seconds and xlabel="time".
    taus : array-like, optional
        Requested averaging times τ in seconds. If ``None``, defaults to
        powers of 2 of the mean period, up to a quarter of the periods.
    threshold : float, optional
        Crossing amplitude in clock y units.
    edge : {'rising', 'falling', 'any'}
        The :func:`crossings` contract: ``threshold=None`` is
        the midpoint of the swing inside the window, and only crossings
        in ``[start, end]`` are used. ``edge='any'`` measures each period
        between crossings of the first crossing's direction, so the
        periods stay consecutive and non-overlapping.
    start, end : float, optional
        Inclusive edge-selection bounds in seconds; None uses the trace extent.

    Returns
    -------
    Signal
        ``xlabel='tau'``, ``x`` = the τ actually used (s): a whole number
        of mean periods, ``m·⟨T⟩``, not the requested value. When a
        requested τ differs from the one used (sub-period requests use one
        period; others round to the nearest whole number of periods) one
        ``MetricWarning`` names every requested and used value. ``y`` =
        σ_y(τ) (dimensionless fractional). A τ with fewer than two
        complete averaging groups gets ``nan`` with a ``MetricWarning``.
        If ``m·⟨T⟩`` is not finite, the unavailable point retains
        the requested finite τ coordinate; otherwise it uses ``m·⟨T⟩``.
        Empty Signal if fewer than 4 edges; a one-sample ``nan`` Signal if
        a consumed sample is NaN, each with a warning.

    Raises
    ------
    ValueError
        Explicit taus are not a one-dimensional array of finite positive
        times, an invalid ``edge`` or window, or the time axis is not
        finite and strictly increasing. Requests are checked before
        waveform-dependent exits.

    Notes
    -----
    Returned x and y arrays are one-dimensional with matching lengths;
    averaging-time coordinates are in seconds and deviations are dimensionless.
    The input Signal objects and their arrays are not modified.
    """
    _check_timing_axis(clock, 'allan_deviation')
    _check_edge(edge, 'allan_deviation')
    _crossing_window(clock, start, end, 'allan_deviation')
    if taus is not None:
        taus = np.asarray(taus, dtype=float)
        if taus.ndim != 1 or np.any(~np.isfinite(taus) | (taus <= 0)):
            raise ValueError(
                'allan_deviation: requested taus must be a one-dimensional '
                f'array of finite values > 0 s; requested taus={taus!r}; '
                f'available {_timing_details(clock)}'
            )
    edges = _cycle_edges(clock, threshold, edge, start, end,
                         metric='allan_deviation')
    if edges is None:
        return _nan_signal(f'allan({clock.name})', 'allan_dev', 'tau', '')
    edges = np.asarray(edges, dtype=float)
    if edge == 'any':
        edges = edges[::2]
    if len(edges) < 4:
        emit(
            metric='allan_deviation', code='insufficient_edges',
            message='Too few edges were observed for this measurement.',
            details={**_timing_details(clock), 'edge': edge,
                     'window_start': start, 'window_end': end,
                     'edge_count': len(edges), 'required_edge_count': 4},
            suggestion='Inspect the clock or capture a longer record.',
        )
        return Signal(
            f'allan({clock.name})', 'allan_dev', 'tau', '',
            np.array([], dtype=float), np.array([], dtype=float),
        )
    periods = np.diff(edges)
    T_avg = float(np.mean(periods))
    y_i = (periods - T_avg) / T_avg
    n = len(y_i)
    if taus is None:
        max_m = max(1, n // 4)
        if max_m < 2:
            ms = np.array([1])
        else:
            ms = np.array([1 << k for k in range(int(np.log2(max_m)) + 1)])
        taus = ms.astype(float) * T_avg
    taus = np.asarray(taus, dtype=float)
    with np.errstate(over='ignore'):
        ms = np.maximum(1, np.rint(taus / T_avg))
        used = ms * T_avg
    used = np.where(np.isfinite(used), used, taus)
    sigmas = np.empty_like(taus)
    for i, tau in enumerate(taus):
        m = ms[i]
        n_groups = int(n // m)
        if m > n // 2 or n_groups < 2:
            emit(
                metric='allan_deviation', code='insufficient_tau_support',
                message='Requested tau has fewer than two complete averaging groups.',
                details={
                    **_timing_details(clock), 'tau_index': i, 'tau_s': float(tau),
                    'mean_period_s': T_avg, 'period_count': n,
                    'periods_per_group': int(m) if np.isfinite(m) else float('inf'),
                    'group_count': n_groups,
                    'required_group_count': 2,
                    'observed_duration_s': float(edges[-1] - edges[0]),
                },
                suggestion='Capture more clock periods or request a shorter tau.',
            )
            sigmas[i] = float('nan')
            continue
        m = int(m)
        y_groups = y_i[:n_groups * m].reshape(n_groups, m).mean(axis=1)
        diffs = np.diff(y_groups)
        sigma_sq = 0.5 * float(np.mean(diffs ** 2))
        sigmas[i] = float(np.sqrt(max(sigma_sq, 0.0)))
    changed = np.flatnonzero(
        ~np.isclose(taus, used, rtol=1e-9, atol=0) & np.isfinite(sigmas))
    if len(changed):
        emit(
            metric='allan_deviation', code='tau_rounded',
            message=('Requested tau is not a whole number of clock periods; '
                     'the output is labelled with the tau actually used.'),
            details={
                **_timing_details(clock), 'mean_period_s': T_avg,
                'tau_indices': [int(i) for i in changed],
                'requested_tau_s': [float(taus[i]) for i in changed],
                'used_tau_s': [float(used[i]) for i in changed],
                'periods_per_group': [int(ms[i]) for i in changed],
            },
            suggestion='Request taus that are whole multiples of the mean period.',
        )
    return Signal(
        f'allan({clock.name})', 'allan_dev', 'tau', '', used, sigmas,
    )
