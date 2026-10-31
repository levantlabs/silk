"""Edge timing and response measurements on time waveforms.

Rise, fall, slew, settling, delay, pulse, duty, period, frequency, overshoot
and hysteresis measurements validate time axes. Crossing times are linearly
interpolated; settling instants use the recorded grid. Swing-relative thresholds
use the observed range unless the caller states levels explicitly.
"""
from __future__ import annotations

from typing import List, Literal, Optional

import math

import numpy as np

from silk.utils._compat import trapezoid
from silk.signals.base import Signal
from silk.signals.schema import metric, Transient
from silk.metrics._guards import _check_nonempty, _axis, _Axis, _nan_safe_transform
from silk.metrics.diagnostics import emit
from silk.metrics.waveform import (
    _all_crossings,
    _axis_context,
    _check_edge,
    _check_occurrence,
    _check_transient,
    _crossing_data_has_nan,
    _crossing_window,
    _interpolated_window,
    _magnitude_samples,
    _select_crossings,
    _timing_details,
    _window_span,
    _window_threshold,
)


def _check_swing_fractions(low: float, high: float, metric: str) -> None:
    """Validate the existing swing-fraction bounds for edge-shape metrics."""
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            f'{metric}: require 0 <= low < high <= 1, got low={low}, high={high}'
        )


def _check_two_transients(a: Signal, b: Signal, metric: str) -> None:
    _check_transient(a, metric)
    _check_transient(b, metric)


def _check_timing_axis(sig: Signal, metric: str) -> None:
    """Validate timing requests before inspecting observed events."""
    if sig.xlabel != 'time':
        raise ValueError(
            f"{metric}: requested a transient Signal (xlabel='time'); "
            f"available {_axis_context(sig)}"
        )
    _check_nonempty(sig, metric)
    _axis(sig, metric, order='increasing')


def _nan_signal(name: str, type_: str, xlabel: str, ylabel: str) -> Signal:
    """The Signal-valued NaN sentinel: one sample, ``x = y = nan``.

    The Signal analogue of ``crossings``' ``array([nan])``. The edge-derived
    Signal metrics (``tie``, ``allan_deviation``, ``inst_frequency`` — now
    ``cycle_frequency``) used
    to come back EMPTY on NaN input because the NaN sentinel from
    ``crossings`` counted as "one edge" — and an empty Signal already means
    "fewer than two real edges", a legitimate result for a flat trace, so
    the two were indistinguishable to a caller. One nan sample is not.
    """
    nan = np.array([float('nan')])
    return Signal(name, type_, xlabel, ylabel, nan, nan.copy())


_EDGE_LEVEL_UNITS = {'initial': '[y]', 'final': '[y]',
                     'start': '[x]', 'end': '[x]'}


def _check_level(value, name: str, metric: str) -> None:
    """A stated level is a finite real number, or ``None`` (observed)."""
    if value is None:
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value)):
        raise ValueError(
            f'{metric}: {name} must be a finite number or None (None = the '
            f'observed level), got {value!r}'
        )


def _check_stated_levels(initial, final, edge: str, metric: str) -> None:
    """Both levels stated: they must give a swing in the edge's direction."""
    _check_level(initial, 'initial', metric)
    _check_level(final, 'final', metric)
    if edge == 'any' and (initial is None) != (final is None):
        raise ValueError(
            f"{metric}: edge='any' needs both initial and final stated, or "
            f"neither; got initial={initial!r}, final={final!r}"
        )
    if initial is None or final is None:
        return
    if initial == final:
        raise ValueError(
            f'{metric}: initial and final are both {final!r}; a zero-swing '
            f'edge has no thresholds to cross'
        )
    if edge == 'rising' and final < initial:
        raise ValueError(
            f'{metric}: a rising edge needs initial < final, got '
            f'initial={initial!r}, final={final!r}'
        )
    if edge == 'falling' and final > initial:
        raise ValueError(
            f'{metric}: a falling edge needs initial > final, got '
            f'initial={initial!r}, final={final!r}'
        )


def _segment_crossings(x: np.ndarray, y: np.ndarray, level: float, edge: str,
                       lo: int, hi: int, start: Optional[float],
                       end: Optional[float]) -> List[tuple]:
    """``(segment, time)`` of every crossing in samples ``lo..hi``.

    The :func:`crossings` convention (``a < level <= b``, linear
    interpolation) and window rule (kept iff ``start <= t <= end``), with
    the segment index retained so edges pair by sample order, not by
    interpolated time.
    """
    out = []
    for i in range(lo, hi):
        a, b = float(y[i]), float(y[i + 1])
        if edge == 'rising' and not a < level <= b:
            continue
        if edge == 'falling' and not a >= level > b:
            continue
        t = float(x[i]) + (level - a) / (b - a) * float(x[i + 1] - x[i])
        if (start is None or t >= start) and (end is None or t <= end):
            out.append((i, t))
    return out


def _pair_edges(firsts: List[tuple], seconds: List[tuple]):
    """Pair first-threshold crossings with the second threshold's.

    An edge is a first-threshold crossing followed by the first
    second-threshold crossing in the same or a later segment; the next
    edge starts at a first-threshold crossing after that segment, so
    ringing about one threshold never splits an edge. Returns the edges
    as ``(t_first, t_second, segment)`` plus the first crossing still
    waiting for its second, if any.
    """
    edges, after, j = [], -1, 0
    for seg, t0 in firsts:
        if seg <= after:
            continue
        while j < len(seconds) and seconds[j][0] < seg:
            j += 1
        if j == len(seconds):
            return edges, (seg, t0)
        edges.append((t0, seconds[j][1], seg))
        after = seconds[j][0]
    return edges, None


def _edge_core(sig: Signal, low: float, high: float, *, edge: str,
               initial: Optional[float], final: Optional[float],
               n: int, start: Optional[float], end: Optional[float],
               metric: str, every: bool = False):
    """Find the selected edge for rise_time, fall_time and slew_rate.

    ``every=True`` ignores ``n`` and returns the list of
    ``(duration, swing, direction)`` tuples of all edges in time order (the
    plurals). Otherwise returns one tuple — ``duration`` is the time
    from the first threshold the edge crosses to the second (low then
    high rising, high then low falling), ``swing`` the positive
    distance between the resolved levels — or ``None`` after emitting
    the ``MetricWarning`` that explains why no edge was measured.
    """
    _check_edge(edge, metric)
    if not every:
        _check_occurrence(n, metric)
    _check_stated_levels(initial, final, edge, metric)
    lo, hi = _crossing_window(sig, start, end, metric)
    observed = initial is None or final is None
    if _crossing_data_has_nan(
        sig, metric=metric, signal_role='sig',
        purpose='threshold' if observed else 'crossing_search',
        start_idx=lo, stop_idx=hi + 1,
    ):
        return None
    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y, dtype=float)
    if start is None and end is None:
        y_min, y_max = float(np.min(y)), float(np.max(y))
    else:
        axis = _Axis(x, metric, y)
        a, b = axis.span(start, end)
        values = np.concatenate([y[(x >= a) & (x <= b)],
                                 axis.interp([a, b], y)])
        y_min, y_max = float(np.min(values)), float(np.max(values))
    details = {**_timing_details(sig), 'value_unit': sig.ylabel}
    if start is not None or end is not None:
        details.update(window_start=start, window_end=end,
                       sample_count=hi + 1 - lo)
    directions = ('rising', 'falling') if edge == 'any' else (edge,)
    found, missing = [], None
    for direction in directions:
        if direction == 'rising':
            v0 = y_min if initial is None else float(initial)
            v1 = y_max if final is None else float(final)
        else:
            v0 = y_max if initial is None else float(initial)
            v1 = y_min if final is None else float(final)
        if edge == 'any' and initial is not None and final is not None:
            v0, v1 = (min(v0, v1), max(v0, v1)) if direction == 'rising' \
                else (max(v0, v1), min(v0, v1))
        base, swing = min(v0, v1), abs(v1 - v0)
        level_details = {
            'initial': v0, 'final': v1,
            'initial_source': 'observed' if initial is None else 'stated',
            'final_source': 'observed' if final is None else 'stated',
        }
        if (v1 - v0 if direction == 'rising' else v0 - v1) <= 0:
            # Both levels stated were validated above, so at least one
            # level here is observed: this is data, not a caller error.
            kind = 'an' if edge == 'any' else f'a {direction}'
            if initial is None and final is None:
                message = (f'The observed swing is zero; no '
                           f'{"" if edge == "any" else direction + " "}'
                           f'edge can be measured.')
                extra = {}
            else:
                message = (f'The stated and observed levels give no '
                           f'{direction} swing; no {direction} edge can be '
                           f'measured.')
                extra = level_details
            emit(
                metric=metric, code='zero_swing', message=message,
                details={**details, 'value_min': y_min, 'value_max': y_max,
                         **extra},
                suggestion=(f'Check the selected node and record {kind} '
                            f'complete edge, or review the stated levels.'),
            )
            return None
        v_low, v_high = base + low * swing, base + high * swing
        roles = (('low', v_low, low), ('high', v_high, high))
        if direction == 'falling':
            roles = roles[::-1]
        (r0, t0v, f0), (r1, t1v, f1) = roles
        firsts = _segment_crossings(x, y, t0v, direction, lo, hi, start, end)
        seconds = _segment_crossings(x, y, t1v, direction, lo, hi, start, end)
        pairs, pending = _pair_edges(firsts, seconds)
        found += [(t0, t1, seg, direction, swing) for t0, t1, seg in pairs]
        if missing is None and not pairs:
            if pending is None:
                missing = dict(role=r0, threshold=t0v, fraction=f0,
                               direction=direction, extra={})
            else:
                seg, t_first = pending
                missing = dict(
                    role=r1, threshold=t1v, fraction=f1, direction=direction,
                    after=r0, extra={
                        f'{r0}_crossing_s': t_first,
                        'search_start_s': float(x[seg]),
                        'search_sample_count': hi + 1 - seg,
                    })
            missing['levels'] = level_details
    found.sort(key=lambda e: (e[2], e[0]))   # sample order, then time
    if not found:
        m = missing
        where = f'{m["direction"]} crossing of the {m["role"]} threshold'
        message = (f'No {where} was found.' if 'after' not in m else
                   f'No {where} was found after the {m["after"]} crossing.')
        if edge == 'any':
            message += ' No complete rising or falling edge was found.'
        emit(
            metric=metric, code='missing_crossing', message=message,
            details={**details, 'threshold': m['threshold'],
                     'threshold_fraction': m['fraction'],
                     'edge': m['direction'] if edge != 'any' else 'any',
                     'threshold_role': m['role'], **m['levels'], **m['extra']},
            suggestion=(f'Record the complete {m["direction"]} edge or review the '
                        f'{m["role"]} fraction and stated levels.'),
        )
        return None
    if every:
        return [(t1 - t0, swing, direction)
                for t0, t1, _, direction, swing in found]
    if abs(n) > len(found):
        emit(
            metric=metric, code='insufficient_edges',
            message='The requested edge was not observed in the supplied record.',
            details={**details, 'edge': edge, 'edge_count': len(found),
                     'requested_edge': int(n), 'low': low, 'high': high},
            suggestion=('Extend the record or review the edge number, window, '
                        'fractions and stated levels.'),
        )
        return None
    t0, t1, _, direction, swing = found[n - 1 if n > 0 else n]
    return t1 - t0, swing, direction


@metric(domain='tran', unit='s', param_units=_EDGE_LEVEL_UNITS)
def rise_time(sig: Transient, low: float = 0.1, high: float = 0.9, *,
              initial: Optional[float] = None, final: Optional[float] = None,
              n: int = 1, start: Optional[float] = None,
              end: Optional[float] = None) -> float:
    """Rise time from the ``low`` to the ``high`` threshold of a rising edge.

    Shares one edge core with :func:`fall_time` and :func:`slew_rate`.
    Resolution as :func:`crossings` (two interpolated crossings), plus a
    sample-rate FLOOR: the metric cannot resolve an edge faster than one
    sample interval. When both crossings land in the SAME segment the
    result is not a measurement of the edge; it IS the sample-rate
    floor, roughly ``(high - low) * dt`` scaled by how much of that
    segment's voltage jump lies between the thresholds — a 10 ps edge
    sampled at 1 ns reads about 0.8 ns, 80x slow.

    Parameters
    ----------
    sig : Signal
        Transient waveform: matching one-dimensional x/y arrays, finite
        strictly increasing time in seconds and xlabel="time".
    low, high : float, default 0.1 and 0.9
        Threshold fractions of the swing, measured up from the LOWER of
        the two levels (so ``high=0.9`` is the 90 % point for either
        direction); ``0 <= low < high <= 1``.
    initial, final : float, optional
        The settled level the edge starts from and ends at, in the
        trace's units (a falling edge starts high: ``initial > final``).
        ``None`` uses the observed extreme of the consumed samples — the
        minimum for the low level, the maximum for the high one — which
        is the whole trace without a window and the legacy result. State them when the edge rings or overshoots, so the
        10/90 points sit on the settled values instead of on the peak.
    n : int, default 1
        Which complete edge: ``k`` is the k-th, ``-k`` the k-th from the
        end. An edge is a crossing of its first threshold followed by
        the next crossing of its second one; the next edge starts after
        that, so ringing about one threshold never splits an edge.
    start, end : float, optional
        Bounds in seconds. Keep only edges with both crossings in ``start <= t <= end``
        (the :func:`crossings` window rule). With a window, observed
        levels are the extremes inside it.

    Returns
    -------
    float
        Selected rising-edge duration in seconds; nan with MetricWarning when
        unavailable.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    An occurrence n that is not a nonzero integer raises ValueError.

    Unavailable cases
    return ``nan`` with a ``MetricWarning``: NaN among the consumed
    samples (``nan_waveform``), zero observed swing or a stated level
    that leaves no swing (``zero_swing``), no complete edge — including
    a stated level the edge never reaches (``missing_crossing``, naming
    the threshold, its role and the levels used) — and fewer than
    ``|n|`` edges (``insufficient_edges``).

    Raises ``ValueError`` (caller error) for a non-transient or empty
    ``sig``, bad fractions, an ``n`` that is not a nonzero integer, a
    non-finite level, both levels stated equal or in the wrong order
    for the direction, or a window that misses the trace.
    """
    _check_transient(sig, 'rise_time')
    _check_swing_fractions(low, high, 'rise_time')
    found = _edge_core(sig, low, high, edge='rising', initial=initial,
                       final=final, n=n, start=start, end=end,
                       metric='rise_time')
    return float('nan') if found is None else found[0]


@metric(domain='tran', unit='s', param_units=_EDGE_LEVEL_UNITS)
def fall_time(sig: Transient, low: float = 0.1, high: float = 0.9, *,
              initial: Optional[float] = None, final: Optional[float] = None,
              n: int = 1, start: Optional[float] = None,
              end: Optional[float] = None) -> float:
    """Fall time from the ``high`` to the ``low`` threshold of a falling edge.

    The mirror of :func:`rise_time` on the same edge core, with the same
    interpolated-crossing resolution and sample-rate FLOOR.
    Parameters, warnings and errors are :func:`rise_time`'s; ``initial``
    is the HIGH settled level here.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    low : float
        Lower fraction of the settled swing; require 0 <= low < high <= 1.
    high : float
        Upper fraction of the settled swing; require 0 <= low < high <= 1.
    initial : float, optional
        High initial amplitude in input y units; None uses the observed maximum.
    final : float, optional
        Low final amplitude in input y units; None uses the observed minimum.
    n : int
        Nonzero occurrence index: 1 selects the first, -1 the last.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Selected falling-edge duration in seconds; nan with MetricWarning when
        unavailable.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Invalid fractions, levels or window raise ValueError. The time axis must be finite and strictly
    increasing and the transient input nonempty; no swing or no complete
    edge emits MetricWarning.
    An occurrence n that is not a nonzero integer raises ValueError.
    """
    _check_transient(sig, 'fall_time')
    _check_swing_fractions(low, high, 'fall_time')
    found = _edge_core(sig, low, high, edge='falling', initial=initial,
                       final=final, n=n, start=start, end=end,
                       metric='fall_time')
    return float('nan') if found is None else found[0]


def _edge_durations(sig, low, high, *, edge, initial, final, start, end,
                    metric):
    """Every edge of the core as ``(durations, swings, directions)`` arrays.

    Empty arrays when the core measured nothing (it has then emitted its
    ``MetricWarning``) or the window holds no edge.
    """
    found = _edge_core(sig, low, high, edge=edge, initial=initial,
                       final=final, n=1, start=start, end=end,
                       metric=metric, every=True)
    if not found:
        return np.empty(0), np.empty(0), []
    duration, swing, direction = zip(*found)
    return np.array(duration, float), np.array(swing, float), list(direction)


def rise_times(sig: Transient, low: float = 0.1, high: float = 0.9, *,
               initial: Optional[float] = None, final: Optional[float] = None,
               start: Optional[float] = None,
               end: Optional[float] = None) -> np.ndarray:
    """Every rise time: one per complete rising edge, in time order.

    The plural of :func:`rise_time` on the same edge core with the same
    arguments, less ``n``. Element ``k`` equals ``rise_time(sig, ...,
    n=k + 1)``. Always returns an ``np.ndarray`` (unregistered, like
    :func:`crossings`: a registered metric returns one number).

    When the singular would return ``nan`` with a ``MetricWarning`` because
    no edge was measured (``nan_waveform``, ``zero_swing``, or
    ``missing_crossing`` — including a stated level the edge never
    reaches) the plural emits the same warning and returns an EMPTY array.
    A window or record holding no edge returns an empty array with the
    ``missing_crossing`` warning, as the singular's ``nan`` does. Caller
    errors raise ``ValueError`` as in :func:`rise_time`.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    low : float
        Lower fraction of the settled swing; require 0 <= low < high <= 1.
    high : float
        Upper fraction of the settled swing; require 0 <= low < high <= 1.
    initial : float, optional
        Initial settled amplitude in the input y unit; None uses the observed
        extreme.
    final : float, optional
        Final settled amplitude in the input y unit; None uses the observed extreme.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of rising-edge durations in seconds; empty with
        MetricWarning when no measurement is available.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Invalid fractions, levels or window raise ValueError. The time axis must be finite and strictly
    increasing and the transient input nonempty; no swing or no complete
    edge emits MetricWarning.
    """
    _check_transient(sig, 'rise_times')
    _check_swing_fractions(low, high, 'rise_times')
    return _edge_durations(sig, low, high, edge='rising', initial=initial,
                           final=final, start=start, end=end,
                           metric='rise_times')[0]


def fall_times(sig: Transient, low: float = 0.1, high: float = 0.9, *,
               initial: Optional[float] = None, final: Optional[float] = None,
               start: Optional[float] = None,
               end: Optional[float] = None) -> np.ndarray:
    """Every fall time: one per complete falling edge, in time order.

    The plural of :func:`fall_time`; see :func:`rise_times` for the
    contract. ``initial`` is the HIGH settled level here.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    low : float
        Lower fraction of the settled swing; require 0 <= low < high <= 1.
    high : float
        Upper fraction of the settled swing; require 0 <= low < high <= 1.
    initial : float, optional
        High initial amplitude in input y units; None uses the observed maximum.
    final : float, optional
        Low final amplitude in input y units; None uses the observed minimum.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of falling-edge durations in seconds; empty with
        MetricWarning when no measurement is available.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Invalid fractions, levels or window raise ValueError. The time axis must be finite and strictly
    increasing and the transient input nonempty; no swing or no complete
    edge emits MetricWarning.
    """
    _check_transient(sig, 'fall_times')
    _check_swing_fractions(low, high, 'fall_times')
    return _edge_durations(sig, low, high, edge='falling', initial=initial,
                           final=final, start=start, end=end,
                           metric='fall_times')[0]


@metric(domain='tran', unit='[y]/s', param_units=_EDGE_LEVEL_UNITS)
def slew_rate(sig: Transient, low: float = 0.1, high: float = 0.9, *,
              edge: Literal['rising', 'falling', 'any'] = 'rising',
              initial: Optional[float] = None, final: Optional[float] = None,
              n: int = 1, start: Optional[float] = None,
              end: Optional[float] = None) -> float:
    """Average slew rate between the ``low`` and ``high`` thresholds.

    ``(high - low) * swing / duration`` of the edge that :func:`rise_time`
    (``edge='rising'``) or :func:`fall_time` (``edge='falling'``) would
    measure with the same arguments — one edge core, a different output.
    ``edge='any'`` counts rising and falling edges together in time
    order; it takes both levels stated (as the two settled levels, in
    either order) or neither — one alone raises ``ValueError``. The result is a positive MAGNITUDE for a
    falling edge too. Inherits the sample-rate floor: an edge faster
    than one sample interval reads as the floor slew, not the true slew.
    A non-increasing time axis raises ``ValueError``. A computed
    nonpositive edge duration returns
    ``nan`` with a ``nonpositive_rise_time`` / ``nonpositive_fall_time``
    ``MetricWarning``. Every other parameter, warning and error is
    :func:`rise_time`'s.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    low : float
        Lower fraction of the settled swing; require 0 <= low < high <= 1.
    high : float
        Upper fraction of the settled swing; require 0 <= low < high <= 1.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    initial : float, optional
        Initial settled amplitude in the input y unit; None uses the observed
        extreme.
    final : float, optional
        Final settled amplitude in the input y unit; None uses the observed extreme.
    n : int
        Nonzero occurrence index: 1 selects the first, -1 the last.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Positive slew magnitude in input y units per second; nan with MetricWarning
        when unavailable.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Invalid fractions, levels or window raise ValueError. The time axis must be finite and strictly
    increasing and the transient input nonempty; no swing or no complete
    edge emits MetricWarning.
    An invalid edge direction raises ValueError.
    An occurrence n that is not a nonzero integer raises ValueError.
    """
    _check_nonempty(sig, 'slew_rate')
    _check_transient(sig, 'slew_rate')
    _check_swing_fractions(low, high, 'slew_rate')
    found = _edge_core(sig, low, high, edge=edge, initial=initial,
                       final=final, n=n, start=start, end=end,
                       metric='slew_rate')
    if found is None:
        return float('nan')
    duration, swing, direction = found
    if duration <= 0:
        kind = 'rise' if direction == 'rising' else 'fall'
        emit(
            metric='slew_rate',
            code=f'nonpositive_{kind}_time',
            message=f'Slew rate unavailable because the {kind} time is not positive.',
            details={**_timing_details(sig), f'{kind}_time_s': float(duration),
                     'low': low, 'high': high, 'edge': direction},
            suggestion=f'Inspect the {direction} edge and its time samples.',
        )
        return float('nan')
    return (high - low) * swing / duration


def slew_rates(sig: Transient, low: float = 0.1, high: float = 0.9, *,
               edge: Literal['rising', 'falling', 'any'] = 'rising',
               initial: Optional[float] = None, final: Optional[float] = None,
               start: Optional[float] = None,
               end: Optional[float] = None) -> np.ndarray:
    """Every slew rate: one positive magnitude per complete edge, in time order.

    The plural of :func:`slew_rate` on the same edge core with the same
    arguments, less ``n``; ``edge='any'`` interleaves rising and falling
    edges in time order. Element ``k`` equals ``slew_rate(sig, ...,
    n=k + 1)``. An edge with a nonpositive duration reads ``nan`` in its
    slot (one ``nonpositive_*_time`` ``MetricWarning`` for the first such
    edge). Otherwise empty-array and warning behaviour is :func:`rise_times`'.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    low : float
        Lower fraction of the settled swing; require 0 <= low < high <= 1.
    high : float
        Upper fraction of the settled swing; require 0 <= low < high <= 1.
    edge : {'rising', 'falling', 'any'}
        Crossing direction along acquisition order.
    initial : float, optional
        Initial settled amplitude in input y units. For edge="any", supply
        both initial and final (either order) or neither; one alone raises
        ValueError. None uses the observed extreme.
    final : float, optional
        Final settled amplitude in the input y unit; None uses the observed extreme.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    numpy.ndarray
        One-dimensional slew magnitudes in input y units per second, one per
        complete edge; empty if none can be measured, with MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Invalid fractions, levels or window raise ValueError. The time axis must be finite and strictly
    increasing and the transient input nonempty; no swing or no complete
    edge emits MetricWarning.
    An invalid edge direction raises ValueError.
    """
    _check_nonempty(sig, 'slew_rates')
    _check_transient(sig, 'slew_rates')
    _check_swing_fractions(low, high, 'slew_rates')
    duration, swing, direction = _edge_durations(
        sig, low, high, edge=edge, initial=initial, final=final,
        start=start, end=end, metric='slew_rates')
    out = np.full(duration.shape, np.nan)
    ok = duration > 0
    out[ok] = (high - low) * swing[ok] / duration[ok]
    bad = np.flatnonzero(~ok)
    if bad.size:
        kind = 'rise' if direction[bad[0]] == 'rising' else 'fall'
        emit(
            metric='slew_rates',
            code=f'nonpositive_{kind}_time',
            message=f'Slew rate unavailable because the {kind} time is not positive.',
            details={**_timing_details(sig),
                     f'{kind}_time_s': float(duration[bad[0]]),
                     'low': low, 'high': high, 'edge': direction[bad[0]],
                     'unavailable_edge_count': int(bad.size)},
            suggestion=f'Inspect the {direction[bad[0]]} edge and its time samples.',
        )
    return out


@metric(domain='tran', unit='s', param_units={'initial': 'V', 'target': 'V', 'start': 's'})
def settling_time(sig: Transient, initial: float, target: Optional[float] = None,
                  tol: float = 0.01, start: float = 0.0) -> float:
    """Elapsed time from ``start`` to the earliest sample such that
    ``|y − target| ≤ tol·|target − initial|`` holds for the rest of the
    window — the standard "last-out-of-band" convention, with the band a
    fraction of the step's SWING rather than of the target level.

    ``initial`` is the step's starting level and is required. ``target``
    is the step's final level; when omitted it defaults to the trace's own
    final value (``final_value(sig)``) — settle to where the output ends
    up. An explicit ``target`` behaves exactly as before. There is no form
    that reads ``initial`` off the waveform at a given time — a caller that
    wants that samples the trace itself and passes the number in.

    Resolution: grid-snapped — the result is the first SAMPLE at/after
    the true settling instant, so it reads high by up to one Δt (no
    interpolation). Returns ``nan`` with a ``MetricWarning`` if the trace
    ends out of band or a consumed sample at/after ``start`` is NaN.
    Samples before ``start`` do not contribute to the result.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    initial : float
        Required initial step amplitude in input y units (normally volts).
    target : float, optional
        Final amplitude in the input y unit; None uses the last sample.
    tol : float
        Dimensionless settling tolerance; its absolute value multiplies the step
        swing.
    start : float, default 0.0
        Reference start time in seconds; defaults to 0. Only samples at or after it
        are consumed.

    Returns
    -------
    float
        Elapsed seconds from start to the first settled sample, not its absolute
        time; nan with MetricWarning for an unsettled or contaminated record, or a
        measured final level equal to initial.

    Raises
    ------
    ValueError
        If ``initial == target`` — a zero-swing step has no band to settle
        within, which is caller error under the metric failure convention, not degenerate
        data. Also raised for an empty requested window or a non-increasing
        time axis, before inspecting waveform values.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    The magnitude of tol is used; negative tol is accepted. No finite-value
    check is applied to initial, target or tol.
    """
    _check_timing_axis(sig, 'settling_time')
    if target is not None and initial == target:
        raise ValueError(
            f'settling_time: initial and target are both {target!r} V; '
            f'a zero-swing step has no band to settle within'
        )
    mask = sig.x >= start
    if not np.any(mask):
        raise ValueError(
            f'settling_time: requested window start={start!r} s contains 0 '
            f'of {len(sig.x)} available samples; time span '
            f'[{sig.x[0]}, {sig.x[-1]}] s'
        )
    first_in = int(np.flatnonzero(mask)[0])
    if _crossing_data_has_nan(
        sig, metric='settling_time', signal_role='sig',
        purpose='settling_window', start_idx=first_in,
    ):
        return float('nan')
    if target is None:
        target = float(sig.y[-1])
        if initial == target:
            # The target was measured, not requested: a trace that ends where
            # it started is degenerate data, not a caller error.
            emit(
                metric='settling_time', code='zero_swing',
                message='The trace ends at the initial level; there is no step to settle.',
                details={**_timing_details(sig), 'initial': initial,
                         'final_value': target, 'value_unit': sig.ylabel},
                suggestion='Check the selected node and the initial level, or pass target.',
            )
            return float('nan')
    band = abs(tol) * abs(target - initial)
    in_band = np.abs(sig.y[first_in:] - target) <= band
    last_out = -1
    for i, within in enumerate(in_band, start=first_in):
        if not within:
            last_out = i
    if last_out == -1:
        return float(sig.x[first_in] - start)
    if last_out == len(sig.x) - 1:
        emit(
            metric='settling_time', code='not_settled',
            message='The last consumed sample remains outside the settling band.',
            details={
                'signal_name': sig.name, 'sample_count': len(in_band),
                'time_start_s': float(sig.x[first_in]),
                'time_stop_s': float(sig.x[-1]), 'requested_start_s': start,
                'target': target, 'band': band,
                'final_value': float(sig.y[-1]), 'value_unit': sig.ylabel,
            },
            suggestion='Inspect the response or extend the observed settling window.',
        )
        return float('nan')
    return float(sig.x[last_out + 1] - start)


_DELAY_EDGES = ('rising', 'falling', 'any')


def _delay_threshold(sig: Signal, threshold: Optional[float], lo: int,
                     hi: int, start: Optional[float], end: Optional[float],
                     *, metric: str, role: str) -> Optional[float]:
    """One side's threshold: the explicit level, else the window's mid-swing.

    ``None`` when the automatic threshold consumed a NaN sample (the
    ``MetricWarning`` is already emitted).
    """
    if threshold is not None:
        return threshold
    if _crossing_data_has_nan(sig, metric=metric, signal_role=role,
                              purpose='threshold', start_idx=lo,
                              stop_idx=hi + 1):
        return None
    x = np.asarray(sig.x, dtype=float)
    return _window_threshold(x[lo:hi + 1], _magnitude_samples(sig)[lo:hi + 1],
                             start, end)


def _crossings_from(sig: Signal, threshold: float, edge: str,
                    after: Optional[float], lo: int, hi: int,
                    start: Optional[float], end: Optional[float], *,
                    upto: Optional[int], metric: str,
                    role: str) -> Optional[np.ndarray]:
    """Crossings at ``t >= after`` inside ``[start, end]``, scanned forward.

    The scan begins at the segment that contains ``after`` (the window
    start without one), so samples before it are never consumed. With
    ``upto=k`` it consumes samples only until ``k`` crossings are found:
    a NaN later than that does not invalidate them. With ``upto=None``
    every sample to the window end is consumed. Returns ``None`` when a
    consumed NaN sample makes the answer unknown (warning emitted).
    """
    x = np.asarray(sig.x, dtype=float)
    y = _magnitude_samples(sig)
    first = lo
    if after is not None:
        first = min(max(lo, int(_axis(sig, 'crossing').search(after, side='left')) - 1),
                    hi)
    bad = np.flatnonzero(np.isnan(y[first:hi + 1]))
    stop = hi + 1 if not len(bad) else first + int(bad[0])
    found = np.array(_all_crossings(x[first:stop], y[first:stop],
                                    float(threshold), edge), dtype=float)
    keep = np.ones(len(found), dtype=bool)
    if start is not None:
        keep &= found >= start
    if end is not None:
        keep &= found <= end
    if after is not None:
        keep &= found >= after
    found = found[keep]
    if len(bad) and (upto is None or len(found) < upto):
        _crossing_data_has_nan(sig, metric=metric, signal_role=role,
                               purpose='crossing_search', start_idx=first,
                               stop_idx=stop + 1)
        return None
    return found


def _pick(found: np.ndarray, n: int) -> float:
    return float(found[n - 1] if n > 0 else found[n])


def _propagation_delay_core(sig_in: Signal, sig_out: Signal,
                            in_threshold, out_threshold, in_edge: str,
                            out_edge: Optional[str], in_n: Optional[int],
                            out_n: int, start: Optional[float],
                            end: Optional[float], *, metric: str,
                            plural: bool):
    """The one pairing rule behind :func:`propagation_delay` (the
    ``in_n``-th trigger, returns a float) and :func:`propagation_delays`
    (``plural=True``: every trigger edge, ``in_n`` unused, returns an
    array).

    The trigger is the selected input crossing; the target is the
    ``out_n``-th output crossing at or after THAT trigger, so a delay is
    never negative. Validation of every argument precedes any data-
    dependent diagnostic.
    """
    _check_two_transients(sig_in, sig_out, metric)
    for name, edge in (('in_edge', in_edge), ('out_edge', out_edge)):
        if (edge is not None or name == 'in_edge') and edge not in _DELAY_EDGES:
            raise ValueError(
                f"{metric}: {name} must be 'rising'|'falling'|'any', "
                f"got {edge!r}"
            )
    out_edge = in_edge if out_edge is None else out_edge
    if not plural:
        _check_occurrence(in_n, metric, name='in_n',
                          plural='propagation_delays')
    _check_occurrence(out_n, metric, name='out_n',
                      plural='propagation_delays')
    if in_threshold is not None:
        in_threshold = float(in_threshold)
    if out_threshold is not None:
        out_threshold = float(out_threshold)
    lo_i, hi_i = _crossing_window(sig_in, start, end, metric)
    lo_o, hi_o = _crossing_window(sig_out, start, end, metric)

    unknown = np.array([float('nan')]) if plural else float('nan')

    thr_in = _delay_threshold(sig_in, in_threshold, lo_i, hi_i, start, end,
                              metric=metric, role='sig_in')
    if thr_in is None:
        return unknown
    triggers = _crossings_from(
        sig_in, thr_in, in_edge, None, lo_i, hi_i, start, end,
        upto=in_n if not plural and in_n > 0 else None,
        metric=metric, role='sig_in',
    )
    if triggers is None:
        return unknown
    window = {'window_start': start, 'window_end': end}
    if plural:
        if not len(triggers):
            return np.array([], dtype=float)
    elif len(triggers) < abs(in_n):
        emit(
            metric=metric, code='missing_crossing',
            message='The requested input crossing was not found.',
            details={**_timing_details(sig_in), 'signal_role': 'sig_in',
                     'threshold': thr_in, 'value_unit': sig_in.ylabel,
                     'edge': in_edge, 'requested_crossing': int(in_n),
                     'crossing_count': len(triggers), **window},
            suggestion=('Review the input threshold, edge, in_n and window, '
                        'or record the complete requested transition.'),
        )
        return float('nan')
    else:
        triggers = np.array([_pick(triggers, in_n)])

    thr_out = _delay_threshold(sig_out, out_threshold, lo_o, hi_o, start, end,
                               metric=metric, role='sig_out')
    if thr_out is None:
        return unknown
    targets = _crossings_from(
        sig_out, thr_out, out_edge, float(triggers[0]), lo_o, hi_o, start, end,
        upto=out_n if not plural and out_n > 0 else None,
        metric=metric, role='sig_out',
    )
    if targets is None:
        return unknown
    delays = np.full(len(triggers), float('nan'))
    for k, t_in in enumerate(triggers):
        after = targets[targets >= t_in]
        if len(after) >= abs(out_n):
            delays[k] = _pick(after, out_n) - float(t_in)
    unpaired = np.flatnonzero(np.isnan(delays))
    if len(unpaired):
        t_first = float(triggers[unpaired[0]])
        emit(
            metric=metric, code='missing_crossing',
            message=('No output crossing follows the input crossing; a '
                     'crossing before the trigger is never paired with it.'),
            details={**_timing_details(sig_out), 'signal_role': 'sig_out',
                     'threshold': thr_out, 'value_unit': sig_out.ylabel,
                     'edge': out_edge, 'requested_crossing': int(out_n),
                     'trigger_time_s': t_first,
                     'trigger_count': len(triggers),
                     'unpaired_trigger_count': len(unpaired), **window},
            suggestion=('Review the output threshold, out_edge and out_n, or '
                        'extend the record past the input crossing.'),
        )
    return delays if plural else float(delays[0])


@metric(domain='tran', unit='s',
        param_units={'in_threshold': 'V', 'out_threshold': 'V',
                     'start': 's', 'end': 's'})
def propagation_delay(sig_in: Transient, sig_out: Transient, *,
                      in_threshold: Optional[float] = None,
                      out_threshold: Optional[float] = None,
                      in_edge: Literal['rising', 'falling', 'any'] = 'rising',
                      out_edge: Literal['rising', 'falling', 'any'] | None = None,
                      in_n: int = 1,
                      out_n: int = 1,
                      start: Optional[float] = None,
                      end: Optional[float] = None) -> float:
    """Delay from one input crossing (the trigger) to the output crossing
    that follows it (the target).

    **The pairing rule.** The trigger is the ``in_n``-th
    ``in_edge`` crossing of ``in_threshold`` on ``sig_in`` (``in_n=-k`` is
    the k-th from the end). The target is the ``out_n``-th ``out_edge``
    crossing of ``out_threshold`` on ``sig_out`` counted from that
    trigger — at or after it, never from ``t=0`` — so the result is never
    negative: an output glitch or start-up transient before the input
    edge cannot be paired with it. ``out_n=-k`` is the k-th from the end
    of the output crossings after the trigger. ``start``/``end`` bound
    both searches. Every side follows the :func:`crossings` contract:
    linear interpolation, ``a < level <= b``.

    ``out_edge=None`` means the same direction as ``in_edge``. An
    INVERTING stage needs it named: ``in_edge='rising',
    out_edge='falling'`` measures ``t_pHL`` and ``in_edge='falling',
    out_edge='rising'`` measures ``t_pLH``; with the same edge on both
    sides an inverter returns about half a stimulus period plus the
    delay. ``in_edge='any'`` (with ``out_edge`` left ``None``) pairs the
    first input edge of either direction with the next output edge of
    either direction.

    A ``None`` threshold is that side's own mid-swing inside the window —
    input and output may sit on different rails.

    For every trigger edge at once use :func:`propagation_delays` (same
    arguments without ``in_n``; returns an array, unregistered).

    Returns ``nan`` with a ``MetricWarning`` when the requested input
    crossing is missing, when no ``out_n``-th output crossing follows the
    trigger (``code='missing_crossing'``, naming ``sig_out`` and the
    trigger time), or when a NaN sample is consumed — by an automatic
    threshold, or by a search before it found the crossing it needs
    (``code='nan_waveform'``). A NaN after the crossings it needs does not
    invalidate the delay.

    Parameters
    ----------
    sig_in : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    sig_out : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    in_threshold : float, optional
        Input crossing amplitude in sig_in y units; None uses its window mid-swing.
    out_threshold : float, optional
        Output crossing amplitude in output y units; None uses the window mid-swing.
    in_edge : {'rising', 'falling', 'any'}
        Direction of the input trigger.
    out_edge : {'rising', 'falling', 'any'} or None
        Direction of the output target; None follows in_edge.
    in_n : int
        Nonzero input crossing index, counted from the end when negative.
    out_n : int
        Nonzero output crossing index, counted only at or after each trigger;
        negative counts from the end.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Nonnegative output-minus-trigger delay in seconds; nan with MetricWarning
        when unavailable.

    Raises
    ------
    ValueError
        For an edge outside ``'rising'|'falling'|'any'``, an ``in_n`` or
        ``out_n`` that is not a nonzero integer, a non-numeric threshold,
        or a window that is backwards or misses either trace — caller
        errors under the metric failure convention, raised before any data is inspected.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    return _propagation_delay_core(
        sig_in, sig_out, in_threshold, out_threshold, in_edge, out_edge,
        in_n, out_n, start, end, metric='propagation_delay', plural=False,
    )


def propagation_delays(sig_in: Signal, sig_out: Signal, *,
                       in_threshold: Optional[float] = None,
                       out_threshold: Optional[float] = None,
                       in_edge: str = 'rising',
                       out_edge: Optional[str] = None,
                       out_n: int = 1,
                       start: Optional[float] = None,
                       end: Optional[float] = None) -> np.ndarray:
    """Every delay: one per ``in_edge`` crossing of ``sig_in`` in the window.

    The plural of :func:`propagation_delay` on the same core and with the
    same arguments, less ``in_n`` (every trigger edge is used, as
    :func:`crossings` has no ``n``). Element ``k`` belongs to the k-th
    trigger, i.e. to ``crossings(sig_in, threshold=in_threshold,
    edge=in_edge, start=start, end=end)[k]``. Always returns an array;
    unregistered, like :func:`crossings`, because a registered metric
    returns one number.

    * No trigger edge: an empty array, quietly (as :func:`crossings`).
    * A trigger with no ``out_n``-th output crossing after it: ``nan`` in
      its slot, and ONE ``MetricWarning`` (``missing_crossing``) giving
      the first such trigger and how many there are.
    * A NaN sample in the consumed data (every input sample in the window;
      every output sample from the first trigger to the window end):
      ``array([nan])`` with a ``MetricWarning``, the :func:`crossings`
      sentinel for "unknown".

    Parameters
    ----------
    sig_in : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    sig_out : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    in_threshold : float, optional
        Input crossing amplitude in sig_in y units; None uses its window mid-swing.
    out_threshold : float, optional
        Output crossing amplitude in output y units; None uses the window mid-swing.
    in_edge : {'rising', 'falling', 'any'}
        Direction of the input trigger.
    out_edge : {'rising', 'falling', 'any'} or None
        Direction of the output target; None follows in_edge.
    out_n : int
        Nonzero output crossing index, counted only at or after each trigger;
        negative counts from the end.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    numpy.ndarray
        One-dimensional delays in seconds, one per trigger. Missing targets give nan
        slots; no triggers gives an empty array; consumed NaNs give a one-element
        nan sentinel. See warnings above.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    return _propagation_delay_core(
        sig_in, sig_out, in_threshold, out_threshold, in_edge, out_edge,
        None, out_n, start, end, metric='propagation_delays', plural=True,
    )


@metric(domain='tran', unit='', param_units={'initial': 'V', 'target': 'V'})
def overshoot(sig: Transient, initial: float, target: float) -> float:
    """Peak excursion beyond ``target``, as a fraction of the step's swing
    ``|target − initial|`` — never of the target level itself, which is
    wrong for any step that does not start at 0 V (a 1.4 V → 1.5 V step
    peaking at 1.519 V is a 19 % overshoot of the 100 mV swing, not 1.3 %
    of the 1.5 V level).

    The direction comes from the step: a rising step (``initial < target``)
    measures the excursion ABOVE target; a falling step
    (``initial > target``) measures the excursion BELOW target. ``initial``
    and ``target`` are both required voltages — there is no form that reads
    either off the waveform at a given time.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    initial : float
        Required initial step amplitude in input y units (normally volts).
    target : float
        Required final step amplitude in input y units (normally volts).

    Returns
    -------
    float
        Nonnegative dimensionless excursion fraction; zero if no excursion beyond
        target occurs. Consumed waveform NaNs propagate with MetricWarning.

    Raises
    ------
    ValueError
        If ``initial == target`` — a zero-swing step has no fraction to
        report, which is caller error under the metric failure convention, not degenerate
        data.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    Empty/wrong-axis input or invalid time coordinates also raises
    ValueError. Initial and target NaNs are not validated separately.
    """
    _check_transient(sig, 'overshoot')
    if initial == target:
        raise ValueError(
            f'overshoot: initial and target are both {target!r} V; '
            f'a zero-swing step has no fraction to report'
        )
    if np.any(np.isnan(sig.y)):
        emit(
            metric='overshoot', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(sig.y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(sig.y))),
                'x_min': float(np.min(sig.x)), 'x_max': float(np.max(sig.x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    swing = abs(target - initial)
    if target > initial:
        peak = float(_nan_safe_transform(np.max(sig.y), lambda y: y.astype(float)))
        if peak <= target:
            return 0.0
        return (peak - target) / swing
    trough = float(_nan_safe_transform(np.min(sig.y), lambda y: y.astype(float)))
    if trough >= target:
        return 0.0
    return (target - trough) / swing


@metric(domain='tran', unit='', param_units={'initial': 'V', 'target': 'V'})
def undershoot(sig: Transient, initial: float, target: float) -> float:
    """Post-transition undershoot, as a fraction of the step's swing
    ``|target − initial|`` (IEEE Std 181-2003, *Transitions, Pulses and
    Related Waveforms* — the standard "undershoot" definition, not a mirror
    of :func:`overshoot`).

    The excursion is measured only AFTER the trace first reaches ``target``:
    for a rising step (``initial < target``) this is
    ``max(0, target − min(y from the first sample with y ≥ target to the
    end)) / swing``; for a falling step (``initial > target``) it mirrors,
    searching for the first ``y ≤ target`` and measuring the excursion
    ABOVE target afterwards. A dip that happens BEFORE the trace reaches
    ``target`` is *preshoot* in the standard, not undershoot, and is never
    counted here. The searched region runs to the end of the trace — a
    caller wanting a narrower window (e.g. one ring cycle) windows the
    signal itself, the same way callers already window for other metrics.

    ``initial`` and ``target`` are both required voltages — there is no
    form that reads either off the waveform at a given time.

    Returns ``nan`` with a ``MetricWarning`` if the trace never reaches
    ``target`` — the post-transition window is then undefined, which is
    degenerate data under the metric failure convention, not caller error.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    initial : float
        Required initial step amplitude in input y units (normally volts).
    target : float
        Required final step amplitude in input y units (normally volts).

    Returns
    -------
    float
        Nonnegative dimensionless post-transition excursion fraction; nan with
        MetricWarning if target is not reached. NaNs in the post-transition
        reduction propagate; any waveform NaN emits MetricWarning, including earlier
        NaNs that do not affect that reduction.

    Raises
    ------
    ValueError
        If ``initial == target`` — a zero-swing step has no fraction to
        report, which is caller error under the metric failure convention, not degenerate
        data.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    Empty/wrong-axis input or invalid time coordinates also raises
    ValueError. A NaN initial or target returns nan without an additional warning.
    """
    _check_transient(sig, 'undershoot')
    if initial == target:
        raise ValueError(
            f'undershoot: initial and target are both {target!r} V; '
            f'a zero-swing step has no fraction to report'
        )
    if np.any(np.isnan(sig.y)):
        emit(
            metric='undershoot', code='nan_waveform',
            message='Consumed waveform samples contain NaNs.',
            details={
                'signal_name': sig.name, 'sample_count': len(sig.y),
                'nan_sample_count': int(np.count_nonzero(np.isnan(sig.y))),
                'x_min': float(np.min(sig.x)), 'x_max': float(np.max(sig.x)),
                'x_label': sig.xlabel,
                'x_unit': {'time': 's', 'frequency': 'Hz'}.get(sig.xlabel),
                'y_unit': sig.ylabel,
            },
            suggestion='Check the source result for NaNs in the consumed samples.',
        )
    if math.isnan(initial) or math.isnan(target):
        # a NaN parameter (not waveform data) makes every comparison below
        # false regardless of the trace, which is deferred provenance, not
        # a "never reaches target" finding about the DATA.
        return float('nan')
    swing = abs(target - initial)
    rising = target > initial
    reached = np.flatnonzero(sig.y >= target) if rising else np.flatnonzero(sig.y <= target)
    if reached.size == 0:
        emit(
            metric='undershoot', code='target_not_reached',
            message='The trace never reaches target; the post-transition window is undefined.',
            details={
                **_timing_details(sig), 'target': target, 'initial': initial,
                'edge': 'rising' if rising else 'falling', 'value_unit': sig.ylabel,
                'value_min': float(np.min(sig.y)), 'value_max': float(np.max(sig.y)),
            },
            suggestion='Extend the trace or review the target and initial levels.',
        )
        return float('nan')
    idx = int(reached[0])
    if rising:
        trough = float(_nan_safe_transform(np.min(sig.y[idx:]), lambda y: y.astype(float)))
        excursion = target - trough
    else:
        peak = float(_nan_safe_transform(np.max(sig.y[idx:]), lambda y: y.astype(float)))
        excursion = peak - target
    if excursion <= 0:
        return 0.0
    return excursion / swing


def _cycle_edges(sig: Signal, threshold: Optional[float], edge: str,
                 start: Optional[float], end: Optional[float], *,
                 metric: str) -> Optional[np.ndarray]:
    """Validated crossings for the period family, or ``None`` on NaN data.

    The crossings contract: ``threshold=None`` is the midpoint of
    the swing inside the window, and a crossing counts iff it lies in
    ``[start, end]``. A NaN among the consumed samples (the segments that
    touch the window) emits ``nan_waveform`` and returns ``None``.
    """
    _check_transient(sig, metric)
    _check_edge(edge, metric)
    lo, hi = _crossing_window(sig, start, end, metric)
    if threshold is not None:
        threshold = float(threshold)
    if _crossing_data_has_nan(
        sig, metric=metric, signal_role='sig', purpose='period_search',
        start_idx=lo, stop_idx=hi + 1,
    ):
        return None
    return _select_crossings(
        sig, _magnitude_samples(sig), threshold, edge, None, start, end,
        metric=metric, lo=lo, hi=hi, details=_timing_details(sig),
    )


def _cycle_periods(edges: np.ndarray,
                   edge: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-cycle periods from crossing times: ``(midpoint_times, periods)``.

    A period is one FULL cycle: rising to rising, falling to falling. For
    ``edge='any'`` the crossings alternate in direction, so each period
    runs from a crossing to the next crossing of the SAME direction
    (``edges[k+2] - edges[k]``): every crossing starts a period and the
    result is still a period, never a half-period. Each period is stamped
    at the midpoint of its two edges.
    """
    step = 2 if edge == 'any' else 1
    first, last = edges[:-step], edges[step:]
    return 0.5 * (first + last), last - first


def _insufficient_cycles(sig: Signal, metric: str, threshold, edge: str,
                         start, end, edge_count: int) -> None:
    required = 3 if edge == 'any' else 2
    emit(
        metric=metric, code='insufficient_crossings',
        message='Too few crossings were observed to complete one period.',
        details={**_timing_details(sig), 'threshold': threshold,
                 'value_unit': sig.ylabel, 'edge': edge,
                 'window_start': start, 'window_end': end,
                 'crossing_count': edge_count,
                 'required_crossing_count': required},
        suggestion=('Extend the record or window to include a full cycle, or '
                    'review the threshold.'),
    )


def _average_period(sig: Signal, threshold, edge, start, end, *,
                    metric: str) -> float:
    edges = _cycle_edges(sig, threshold, edge, start, end, metric=metric)
    if edges is None:
        return float('nan')
    _, periods = _cycle_periods(edges, edge)
    if len(periods) == 0:
        _insufficient_cycles(sig, metric, threshold, edge, start, end,
                             len(edges))
        return float('nan')
    return float(np.mean(periods))


def _cycle_period(sig: Signal, threshold, edge, start, end, *,
                  metric: str) -> tuple[Optional[np.ndarray], np.ndarray]:
    edges = _cycle_edges(sig, threshold, edge, start, end, metric=metric)
    if edges is None:
        return None, np.array([], dtype=float)
    times, periods = _cycle_periods(edges, edge)
    if len(periods) == 0:
        _insufficient_cycles(sig, metric, threshold, edge, start, end,
                             len(edges))
    return times, periods


@metric(domain='tran', unit='s',
        param_units={'threshold': '[y]', 'start': 's', 'end': 's'})
def average_period(sig: Transient, threshold: Optional[float] = None, *,
                   edge: Literal['rising', 'falling', 'any'] = 'rising',
                   start: Optional[float] = None,
                   end: Optional[float] = None) -> float:
    """Mean period in seconds over ``[start, end]``.

    The mean of :func:`cycle_period`: every full cycle whose two edges
    lie inside the window. ``threshold``, ``edge``, ``start`` and ``end``
    follow the :func:`crossings` contract: ``threshold=None``
    is the midpoint of the swing inside the window. A period is a full
    cycle — rising to rising, falling to falling, and for ``edge='any'``
    each crossing to the next crossing of the same direction.

    Returns ``nan`` with a ``MetricWarning`` when fewer than one complete
    period lies in the window (two same-direction crossings) or when a
    consumed sample is NaN. An invalid ``edge`` or window raises
    ``ValueError``.

    Parameters
    ----------
    sig : Signal
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

    Returns
    -------
    float
        Mean full-cycle period in seconds; nan with MetricWarning for missing cycles
        or consumed NaNs.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    return _average_period(sig, threshold, edge, start, end,
                           metric='average_period')


@metric(domain='tran', unit='Hz',
        param_units={'threshold': '[y]', 'start': 's', 'end': 's'})
def average_frequency(sig: Transient, threshold: Optional[float] = None, *,
                      edge: Literal['rising', 'falling', 'any'] = 'rising',
                      start: Optional[float] = None,
                      end: Optional[float] = None) -> float:
    """``1 / average_period`` in hertz — same core, same arguments.

    It is the reciprocal of the MEAN period, not the mean of the
    per-cycle frequencies. ``nan`` (with a ``MetricWarning``) exactly
    when :func:`average_period` is.

    Parameters
    ----------
    sig : Signal
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

    Returns
    -------
    float
        Reciprocal mean period in Hz; nan with MetricWarning for missing cycles or
        consumed NaNs.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    """
    return 1.0 / _average_period(sig, threshold, edge, start, end,
                                 metric='average_frequency')


def cycle_period(sig: Signal, threshold: Optional[float] = None, *,
                 edge: Literal['rising', 'falling', 'any'] = 'rising',
                 start: Optional[float] = None,
                 end: Optional[float] = None) -> Signal:
    """Per-cycle period as a waveform: seconds versus time.

    One point per full cycle inside ``[start, end]`` (see
    :func:`average_period` for the cycle and argument definitions), at
    the midpoint time of the cycle's two edges.

    Parameters
    ----------
    sig : Signal
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

    Returns
    -------
    Signal
        ``xlabel='time'``, ``y`` in seconds. Empty when no full cycle lies
        in the window; a one-sample ``nan`` Signal when a consumed sample
        is NaN (an empty one would read as "no cycles"). Both emit a
        ``MetricWarning``.

    Notes
    -----
    Returned x and y arrays are one-dimensional with matching lengths;
    time coordinates are in seconds.
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    """
    name = f'cycle_period({sig.name})'
    times, periods = _cycle_period(sig, threshold, edge, start, end,
                                   metric='cycle_period')
    if times is None:
        return _nan_signal(name, 'period', 'time', 's')
    return Signal(name, 'period', 'time', 's', times, periods)


def cycle_frequency(sig: Signal, threshold: Optional[float] = None, *,
                    edge: Literal['rising', 'falling', 'any'] = 'rising',
                    start: Optional[float] = None,
                    end: Optional[float] = None) -> Signal:
    """``1 / cycle_period`` as a waveform: hertz versus time.

    Same core, arguments and empty / ``nan`` cases as
    :func:`cycle_period`. Useful for VCO/PLL characterisation (frequency
    versus time).

    Parameters
    ----------
    sig : Signal
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

    Returns
    -------
    Signal
        One-dimensional cycle-midpoint times in seconds and reciprocal periods in
        Hz. Empty for missing cycles, or one nan sample for consumed NaNs; either
        emits MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    """
    name = f'cycle_frequency({sig.name})'
    times, periods = _cycle_period(sig, threshold, edge, start, end,
                                   metric='cycle_frequency')
    if times is None:
        return _nan_signal(name, 'frequency', 'time', 'Hz')
    return Signal(name, 'frequency', 'time', 'Hz', times, 1.0 / periods)


def _level_crossings(sig: Signal, threshold: Optional[float],
                     start: Optional[float], end: Optional[float], *,
                     metric: str):
    """Every crossing of one level inside ``[start, end]``, with direction.

    Returns ``(level, times, rising)`` — ``times`` sorted, ``rising`` a
    boolean array (``True`` for a rising crossing) — or ``None`` after a
    ``nan_waveform`` warning when a consumed sample is NaN. The
    :func:`crossings` contract: ``threshold=None`` is the
    midpoint of the swing inside the window, resolved ONCE so that the
    rising and falling crossings share it. With the ``a < level <= b`` /
    ``a >= level > b`` convention a rising crossing takes the trace from
    below the level to at-or-above it and a falling one back, so the two
    directions strictly alternate: between two consecutive crossings the
    trace is wholly high (after a rising one) or wholly low.
    """
    _check_transient(sig, metric)
    lo, hi = _crossing_window(sig, start, end, metric)
    if _crossing_data_has_nan(
        sig, metric=metric, signal_role='sig', purpose='crossing_search',
        start_idx=lo, stop_idx=hi + 1,
    ):
        return None
    x = np.asarray(sig.x, dtype=float)
    y = _magnitude_samples(sig)
    if threshold is None:
        level = _window_threshold(x[lo:hi + 1], y[lo:hi + 1], start, end)
    else:
        level = float(threshold)
    found = {
        edge: _select_crossings(
            sig, y, level, edge, None, start, end, metric=metric, lo=lo,
            hi=hi, details=_timing_details(sig),
        )
        for edge in ('rising', 'falling')
    }
    times = np.concatenate([found['rising'], found['falling']])
    rising = np.concatenate([np.ones(len(found['rising']), dtype=bool),
                             np.zeros(len(found['falling']), dtype=bool)])
    order = np.argsort(times, kind='stable')
    return level, times[order], rising[order]


def _cycle_duty(sig: Signal, threshold, edge, start, end, *, metric: str):
    """The duty core: ``(midpoint_times, high_times, periods)`` per cycle.

    The cycle calculation uses :func:`_cycle_periods` over the crossings
    of ``edge`` (a full cycle; for ``'any'`` each crossing to the next of
    the same direction). Each cycle spans exactly two consecutive
    crossings intervals, one high and one low; its high time is the
    length of the interval that starts on a rising crossing. Returns
    ``None`` on NaN data; warns ``insufficient_crossings`` when no full
    cycle lies in the window (the arrays are then empty).
    """
    _check_edge(edge, metric)
    found = _level_crossings(sig, threshold, start, end, metric=metric)
    if found is None:
        return None
    _, times, rising = found
    if edge == 'any':
        starts = np.arange(len(times))
    else:
        starts = np.flatnonzero(rising if edge == 'rising' else ~rising)
    mid, periods = _cycle_periods(times[starts], edge)
    if len(periods) == 0:
        _insufficient_cycles(sig, metric, threshold, edge, start, end,
                             len(starts))
    starts = starts[:len(periods)]
    high = np.where(rising[:-1], np.diff(times), 0.0)
    return mid, high[starts] + high[starts + 1], periods


@metric(domain='tran', unit='',
        param_units={'threshold': '[y]', 'start': 's', 'end': 's'})
def average_duty(sig: Transient, threshold: Optional[float] = None, *,
                 edge: Literal['rising', 'falling', 'any'] = 'rising',
                 start: Optional[float] = None,
                 end: Optional[float] = None) -> float:
    """Fraction of time above ``threshold`` over the full cycles in
    ``[start, end]``.

    The cycles are :func:`average_period`'s: every full cycle
    whose two edges lie inside the window — rising to rising by default,
    falling to falling, or each crossing to the next of the same
    direction for ``edge='any'``. The result is the total high time
    divided by the total cycle time; it equals the mean of
    :func:`cycle_duty` when the cycles are equally long, and weights each
    cycle by its period otherwise. ``threshold``, ``edge``, ``start`` and
    ``end`` follow the :func:`crossings` contract:
    ``threshold=None`` is the midpoint of the swing inside the window.

    Returns ``nan`` with an ``insufficient_crossings`` ``MetricWarning``
    when no full cycle lies in the window — use :func:`on_time` for the
    time above a level on a non-periodic or partial record — and with a
    ``nan_waveform`` warning when a consumed sample is NaN. An invalid
    ``edge`` or window raises ``ValueError``.

    Parameters
    ----------
    sig : Signal
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

    Returns
    -------
    float
        Dimensionless time-weighted fraction high over complete cycles; nan with
        MetricWarning when no cycle is measurable or consumed samples contain NaN.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    found = _cycle_duty(sig, threshold, edge, start, end,
                        metric='average_duty')
    if found is None:
        return float('nan')
    _, high_times, periods = found
    if len(periods) == 0:
        return float('nan')
    return float(np.sum(high_times) / np.sum(periods))


def cycle_duty(sig: Signal, threshold: Optional[float] = None, *,
               edge: Literal['rising', 'falling', 'any'] = 'rising',
               start: Optional[float] = None,
               end: Optional[float] = None) -> Signal:
    """Per-cycle duty as a waveform: fraction high versus time.

    One point per full cycle inside ``[start, end]``, stamped at the
    midpoint of the cycle's two edges — the same cycles and times as
    :func:`cycle_period`, with the same arguments (see
    :func:`average_duty`).

    Parameters
    ----------
    sig : Signal
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

    Returns
    -------
    Signal
        ``xlabel='time'``, ``y`` a fraction in ``[0, 1]``. Empty when no
        full cycle lies in the window; a one-sample ``nan`` Signal when a
        consumed sample is NaN. Both emit a ``MetricWarning``.

    Notes
    -----
    Returned x and y arrays are one-dimensional with matching lengths;
    time coordinates are in seconds.
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    edge direction or malformed windows raise ValueError.
    """
    name = f'cycle_duty({sig.name})'
    found = _cycle_duty(sig, threshold, edge, start, end,
                        metric='cycle_duty')
    if found is None:
        return _nan_signal(name, 'duty', 'time', '')
    times, high_times, periods = found
    return Signal(name, 'duty', 'time', '', times, high_times / periods)


def _check_polarity(polarity, metric: str) -> None:
    if polarity not in ('positive', 'negative'):
        raise ValueError(
            f"{metric}: polarity must be 'positive' (a high pulse) or "
            f"'negative' (a low pulse), got {polarity!r}"
        )


def _pulse_widths(sig: Signal, threshold, polarity, start, end, *,
                  metric: str) -> Optional[np.ndarray]:
    """Widths of every complete pulse in the window, or ``None`` on NaN.

    A positive pulse is a rising crossing and the falling crossing after
    it; a negative pulse a falling crossing and the rising one after it.
    Both crossings must lie in ``[start, end]``.
    """
    _check_polarity(polarity, metric)
    found = _level_crossings(sig, threshold, start, end, metric=metric)
    if found is None:
        return None
    _, times, rising = found
    opens = rising[:-1] if polarity == 'positive' else ~rising[:-1]
    return np.diff(times)[opens]


@metric(domain='tran', unit='s',
        param_units={'threshold': '[y]', 'start': 's', 'end': 's'})
def pulse_width(sig: Transient, threshold: Optional[float] = None, *,
                polarity: Literal['positive', 'negative'] = 'positive',
                n: int = 1,
                start: Optional[float] = None,
                end: Optional[float] = None) -> float:
    """Width in seconds of the ``n``-th complete pulse in ``[start, end]``.

    ``polarity='positive'`` measures a high pulse (a rising crossing of
    ``threshold`` to the next falling crossing), ``'negative'`` a low
    pulse (falling to the next rising). A pulse counts only when both of
    its crossings lie inside the window, so a pulse the record or window
    cuts is never measured. ``n`` is 1-based; ``-1`` is the last complete
    pulse. ``threshold``, ``start`` and ``end`` follow the
    :func:`crossings` contract: ``threshold=None`` is the
    midpoint of the swing inside the window. :func:`pulse_widths` returns
    every pulse.

    Returns ``nan`` with an ``insufficient_crossings`` ``MetricWarning``
    when fewer than ``|n|`` complete pulses lie in the window, and with a
    ``nan_waveform`` warning when a consumed sample is NaN. An invalid
    ``polarity``, ``n`` or window raises ``ValueError``.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    polarity : {'positive', 'negative'}
        Select complete high or low pulses, respectively.
    n : int
        Nonzero occurrence index: 1 selects the first, -1 the last.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Selected complete pulse duration in seconds; nan with MetricWarning for a
        missing pulse or consumed NaNs.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    _check_occurrence(n, 'pulse_width', noun='pulse', plural='pulse_widths')
    widths = _pulse_widths(sig, threshold, polarity, start, end,
                           metric='pulse_width')
    if widths is None:
        return float('nan')
    if abs(n) > len(widths):
        emit(
            metric='pulse_width', code='insufficient_crossings',
            message='The requested pulse was not observed in the supplied record.',
            details={**_timing_details(sig), 'threshold': threshold,
                     'value_unit': sig.ylabel, 'polarity': polarity,
                     'window_start': start, 'window_end': end,
                     'pulse_count': len(widths), 'requested_pulse': int(n)},
            suggestion=('Extend the record or window to include the complete '
                        'pulse, or review the threshold, polarity and pulse '
                        'number.'),
        )
        return float('nan')
    return float(widths[n - 1] if n > 0 else widths[n])


def pulse_widths(sig: Signal, threshold: Optional[float] = None, *,
                 polarity: Literal['positive', 'negative'] = 'positive',
                 start: Optional[float] = None,
                 end: Optional[float] = None) -> np.ndarray:
    """Widths in seconds of every complete pulse in ``[start, end]``.

    The plural of :func:`pulse_width`: the same pulses and arguments
    without ``n``, in time order. Always an array: empty when no complete
    pulse lies in the window (no warning — zero pulses is an answer), and
    ``array([nan])`` with a ``nan_waveform`` ``MetricWarning`` when a
    consumed sample is NaN, as :func:`crossings` does.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    polarity : {'positive', 'negative'}
        Select complete high or low pulses, respectively.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    numpy.ndarray
        One-dimensional complete pulse widths in seconds, in time order. Empty
        without warning if no pulse exists; one nan element with MetricWarning for
        consumed NaNs.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    Wrong or empty time axes, non-increasing coordinates, an invalid
    pulse polarity or malformed windows raise ValueError.
    """
    widths = _pulse_widths(sig, threshold, polarity, start, end,
                           metric='pulse_widths')
    if widths is None:
        return np.array([float('nan')])
    return widths


@metric(domain='tran', unit='V',
        param_units={'out_threshold': 'V', 'start': 's', 'end': 's'})
def hysteresis(sig_in_rising: Transient, sig_out_rising: Transient,
               sig_in_falling: Transient, sig_out_falling: Transient, *,
               out_threshold: Optional[float] = None,
               n_rising: int = 1,
               n_falling: int = 1,
               start: Optional[float] = None,
               end: Optional[float] = None) -> float:
    """Hysteresis = v_trip_rising − v_trip_falling.

    Paired input grids agree within 1e-3 of their smallest spacing;
    singleton coordinates must match exactly. Mismatches raise ``ValueError``.

    **The pairing rule.** Each trip instant is an OUTPUT
    crossing of ``out_threshold`` — rising on the rising sweep, falling on
    the falling sweep — and the trip voltage is the input interpolated at
    that instant. The search is confined to ``[start, end]`` (both
    sweeps; the whole trace without them), so a start-up transient before
    ``start`` is never read as the decision. The crossing used is the
    ``n_rising``-th / ``n_falling``-th of the window (default ``1``, the
    first; ``-k`` counts from the end, ``-1`` being the decision the
    output finally stays in). When the output crosses more than once in
    the searched direction inside the window — a glitch or earlier pulse
    before the real decision, or chatter — the first crossing may not be
    the decision: one ``MetricWarning`` per side
    (``code='multiple_crossings'``) names the side, the count and the
    instant used; pass ``start=`` past the transient or ``n_rising`` /
    ``n_falling=-1`` to choose. Only finite crossings are counted. The
    default threshold is the mid-swing of the rising output inside the
    window.

    Each pair must share its own x-axis (rising/rising, falling/falling).
    NaNs consumed by the threshold or the crossing search return ``nan``
    with a ``MetricWarning`` (a negative ``n`` consumes the whole window;
    a positive ``n`` stops at the first NaN once it has its crossing, so a
    NaN tail after the trip does not invalidate it). Input interpolation
    consumes only the samples at each trip point and warns on NaNs in
    those samples. A missing crossing (fewer than ``|n|`` in the window)
    also returns ``nan`` with a ``MetricWarning``.

    Parameters
    ----------
    sig_in_rising : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    sig_out_rising : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    sig_in_falling : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    sig_out_falling : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    out_threshold : float, optional
        Output crossing level in output y units; None uses the rising-output window
        mid-swing for both sweeps.
    n_rising : int
        Nonzero rising-output crossing index; negative counts from the end.
    n_falling : int
        Nonzero falling-output crossing index; negative counts from the end.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Rising trip input minus falling trip input, in input y units (normally
        volts); nan with MetricWarning if either trip is unavailable.

    Raises
    ------
    ValueError
        For inputs that are not transients or do not share an x-axis, an
        ``n_rising`` / ``n_falling`` that is not a nonzero integer, or a
        window that is backwards or misses a trace.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    for sig in (sig_in_rising, sig_out_rising, sig_in_falling, sig_out_falling):
        _check_transient(sig, 'hysteresis')
    if not _axis(sig_in_rising, 'hysteresis').same_grid(
            _axis(sig_out_rising, 'hysteresis')):
        raise ValueError(
            'hysteresis: rising-sweep input and output must share x-axis; '
            f'input: {_axis_context(sig_in_rising)}; '
            f'output: {_axis_context(sig_out_rising)}'
        )
    if not _axis(sig_in_falling, 'hysteresis').same_grid(
            _axis(sig_out_falling, 'hysteresis')):
        raise ValueError(
            'hysteresis: falling-sweep input and output must share x-axis; '
            f'input: {_axis_context(sig_in_falling)}; '
            f'output: {_axis_context(sig_out_falling)}'
        )
    _check_occurrence(n_rising, 'hysteresis', name='n_rising')
    _check_occurrence(n_falling, 'hysteresis', name='n_falling')
    lo_r, hi_r = _crossing_window(sig_out_rising, start, end, 'hysteresis')
    lo_f, hi_f = _crossing_window(sig_out_falling, start, end, 'hysteresis')
    if out_threshold is None:
        out_threshold = _delay_threshold(
            sig_out_rising, None, lo_r, hi_r, start, end,
            metric='hysteresis', role='sig_out_rising',
        )
        if out_threshold is None:
            return float('nan')
    out_threshold = float(out_threshold)
    trips = []
    for sig_out, edge, n, lo, hi, role in (
        (sig_out_rising, 'rising', n_rising, lo_r, hi_r, 'sig_out_rising'),
        (sig_out_falling, 'falling', n_falling, lo_f, hi_f, 'sig_out_falling'),
    ):
        found = _crossings_from(
            sig_out, out_threshold, edge, None, lo, hi, start, end,
            upto=n if n > 0 else None, metric='hysteresis', role=role,
        )
        if found is None:
            return float('nan')
        if len(found) < abs(n):
            emit(
                metric='hysteresis', code='missing_crossing',
                message='The required threshold crossing was not found.',
                details={**_timing_details(sig_out), 'signal_role': role,
                         'threshold': out_threshold,
                         'value_unit': sig_out.ylabel, 'edge': edge,
                         'requested_crossing': int(n),
                         'crossing_count': len(found),
                         'window_start': start, 'window_end': end},
                suggestion='Review the threshold, n and window, and record '
                           'the complete requested transition.',
            )
            return float('nan')
        if n == 1 and len(found) > 1:
            emit(
                metric='hysteresis', code='multiple_crossings',
                message=(f'The output crosses the threshold {len(found)} '
                         f'times ({edge}) inside the window; the first, at '
                         f'{float(found[0])!r}, was used.'),
                details={**_timing_details(sig_out), 'signal_role': role,
                         'threshold': out_threshold, 'edge': edge,
                         'crossing_count': len(found),
                         'used_crossing_time': float(found[0]),
                         'window_start': start, 'window_end': end},
                suggestion='Pass start= past the early transient, or '
                           'n_rising / n_falling=-1 for the last crossing.',
            )
        trips.append(_pick(found, n))
    t_dec_r, t_dec_f = trips
    contaminated = False
    for sig_in, t_dec, role in (
        (sig_in_rising, t_dec_r, 'sig_in_rising'),
        (sig_in_falling, t_dec_f, 'sig_in_falling'),
    ):
        right = int(_axis(sig_in, 'hysteresis').search(t_dec, side='right'))
        left = max(right - 1, 0)
        stop = min(right + 1, len(sig_in.x))
        if t_dec == sig_in.x[left]:
            stop = left + 1
        contaminated |= _crossing_data_has_nan(
            sig_in, metric='hysteresis', signal_role=role,
            purpose='decision_interpolation', start_idx=left, stop_idx=stop,
        )
    if contaminated:
        return float('nan')
    trip_r = float(_axis(sig_in_rising, 'hysteresis').interp(t_dec_r, sig_in_rising.y))
    trip_f = float(_axis(sig_in_falling, 'hysteresis').interp(t_dec_f, sig_in_falling.y))
    return trip_r - trip_f


def _time_above(sig: Signal, threshold: Optional[float],
                start: Optional[float], end: Optional[float], *,
                metric: str) -> tuple[float, float]:
    """``(time above threshold, window duration)`` over ``[start, end]``.

    Shared by :func:`on_time` and :func:`off_time`. The window is clipped
    to the trace and its bounds are sampled by linear interpolation;
    without a window the samples are the trace's own, so the result is
    the original whole-trace integral. ``threshold=None`` is the midpoint
    of the swing inside the window (the :func:`crossings` contract).
    Returns ``(nan, duration)`` after a ``nan_waveform`` warning when a
    consumed sample is NaN: ``NaN > threshold`` is ``False``, so an
    unguarded comparison would silently count the sample as low.
    """
    _check_transient(sig, metric)
    lo, hi = _crossing_window(sig, start, end, metric)
    x = np.asarray(sig.x, dtype=float)
    a, b = _window_span(x, start, end)
    if _crossing_data_has_nan(
        sig, metric=metric, signal_role='sig', purpose='time_above_level',
        start_idx=lo, stop_idx=hi + 1,
    ):
        return float('nan'), b - a
    y = _magnitude_samples(sig)
    if threshold is None:
        threshold = _window_threshold(x[lo:hi + 1], y[lo:hi + 1], start, end)
    xw, yw = _interpolated_window(x, y, lo, hi, start, end)
    above = (yw > float(threshold)).astype(float)
    return float(trapezoid(above, xw)), b - a


@metric(domain='tran', unit='s',
        param_units={'threshold': '[y]', 'start': 's', 'end': 's'})
def on_time(sig: Transient, threshold: Optional[float] = None, *,
            start: Optional[float] = None,
            end: Optional[float] = None) -> float:
    """Total time the signal is above ``threshold`` over ``[start, end]``.

    Trapezoidal integration of ``(y > threshold)`` over time. No cycles
    or edge pairing — it works on a bursty, non-periodic or partial
    record and on a trace that starts or ends mid-pulse; for a periodic
    signal, :func:`average_duty` gives the fraction over full cycles.
    ``threshold``, ``start`` and ``end`` follow the :func:`crossings`
    contract: ``threshold=None`` is the midpoint of the swing
    inside the window; a window bound between samples is sampled by
    linear interpolation, and a window that does not overlap the trace
    raises ``ValueError``.

    Resolution: each edge is smeared over the sample interval it falls
    in, ±Δt per edge (the answer is "samples above threshold" × Δt for a
    uniform grid).

    Returns ``nan`` if a consumed sample is NaN: ``NaN >
    threshold`` is ``False``, so an unguarded comparison would silently
    count the NaN sample as LOW instead of unknown. A ``MetricWarning``
    explains the consumed NaNs; :func:`off_time` propagates this
    explanation without a second warning.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Integrated time strictly above threshold in seconds; consumed NaNs return
        nan with MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    return _time_above(sig, threshold, start, end, metric='on_time')[0]


@metric(domain='tran', unit='s',
        param_units={'threshold': '[y]', 'start': 's', 'end': 's'})
def off_time(sig: Transient, threshold: Optional[float] = None, *,
             start: Optional[float] = None,
             end: Optional[float] = None) -> float:
    """Total time the signal is at or below ``threshold`` over
    ``[start, end]``: the window duration − :func:`on_time`.

    Same arguments and NaN handling as :func:`on_time`; a NaN warns once,
    under ``on_time``.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    threshold : float, optional
        Crossing level in input y units; None uses the observed mid-swing inside the
        window.
    start : float, optional
        Inclusive time bound in seconds; None uses the trace extent.
    end : float, optional
        Inclusive time bound in seconds; None uses the trace extent.

    Returns
    -------
    float
        Integrated time at or below threshold in seconds; consumed NaNs return nan
        with one MetricWarning attributed to on_time.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    """
    _check_transient(sig, 'off_time')
    _crossing_window(sig, start, end, 'off_time')
    above, duration = _time_above(sig, threshold, start, end,
                                  metric='on_time')
    return duration - above
