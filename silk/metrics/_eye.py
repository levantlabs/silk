"""Private alignment and opening geometry for the original-sample eye fold."""

from typing import Optional

import numpy as np

from silk.metrics._guards import _Axis


# Advisory heuristics, not a calibrated confidence score or refusal threshold.
_CENTERING_THRESHOLD_FRACTION = .05
_CENTERING_PHASE_LIMIT_UI = .10


def _measure_eye(x, y, period, origin, count, threshold):
    """Return singleton threshold/height/width tuples for the full windows.

    Called only with eligible centering and at least two windows. Interpolate
    the original PL waveform, including segments spanning window boundaries;
    the returned fold itself is never resampled or extended.
    """
    # Like alignment, analyse in float precision; np.interp rejects longdouble
    # arrays. Keep the original sample dtype/bytes in the separate output fold.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Measure around a local zero, retaining small gaps on a common DC offset.
    offset = float(y[0])
    y = y - offset
    level = threshold - offset
    centers = origin + (np.arange(count) + .5) * period
    axis = _Axis(x, 'eye_diagram', y, order='increasing')
    values = axis.interp(centers, y)
    below, above = values[values <= level], values[values >= level]
    if not len(below) or not len(above):
        return (threshold,), (float('nan'),), (float('nan'),)
    height = float(np.min(above) - np.max(below))
    if height == 0:
        return (threshold,), (0.,), (0.,)

    # Clipped endpoints are analysis-only: they retain the actual waveform
    # segments at the limits without inventing a last-to-first connection.
    end = origin + count * period
    inside = (x > origin) & (x < end)
    t = np.r_[0., (x[inside] - origin) / period, float(count)]
    v = np.r_[axis.interp(origin, y), y[inside], axis.interp(end, y)]
    left, right = _opening_bounds(t, v, level)
    return (threshold,), (height,), (float((right - left) * period),)


def _opening_bounds(t, y, level):
    """Nearest contacts around phase .5 on the UI circle; t is in UI units."""
    relative = y - level
    strict = (((relative[:-1] < 0) & (relative[1:] > 0))
              | ((relative[:-1] > 0) & (relative[1:] < 0)))
    indices = np.flatnonzero(strict)
    fraction = -relative[indices] / (relative[indices + 1] - relative[indices])
    crossings = t[indices] + fraction * (t[indices + 1] - t[indices])
    # Unlike alignment events, touches and plateau endpoints bound an opening.
    phases = np.r_[crossings, t[relative == 0]] % 1.
    if not len(phases):
        return float('nan'), float('nan')
    left = phases[phases <= .5]
    right = phases[phases >= .5]
    return (float(np.max(left) if len(left) else np.max(phases) - 1.),
            float(np.min(right) if len(right) else np.min(phases) + 1.))


def _analyse_eye(x, y, period) -> Optional[tuple[float, Optional[dict[str, object]]]]:
    """Return (relative centre, optional warning facts), or None if undefined.

    Integrate the full piecewise-linear record, then give every genuine
    crossing of its time average equal weight in a circular phase mean.
    This measures display alignment only, not the existence of an open eye.
    Caller validation and the untouched sample fold belong to eye_diagram.
    """
    if not np.all(np.isfinite(y)):
        return None
    # Removing a common offset keeps small, resolvable changes on a large DC
    # level from disappearing when the segment averages are accumulated.
    offset = float(y[0])
    y = np.asarray(y, dtype=float) - offset
    t = np.asarray(x, dtype=float) - x[0]
    average = np.sum(np.diff(t) * (y[:-1] / 2 + y[1:] / 2)) / t[-1]
    center = _crossing_center(t, y - average, period)
    if center is None:
        return None

    delta = float(np.ptp(y)) * _CENTERING_THRESHOLD_FRACTION
    probes = [_crossing_center(t, y - threshold, period)
              for threshold in (average - delta, average + delta)]
    # Compare each probe with nominal: the probes can agree with each other
    # while disagreeing with nominal by half a UI (e.g. a threshold plateau).
    shifts = [float(abs((probe / period - center / period + .5) % 1. - .5))
              for probe in probes if probe is not None]
    maximum = max(shifts, default=None)
    undefined = sum(probe is None for probe in probes)
    diagnostic = None
    if undefined or (maximum is not None and maximum > _CENTERING_PHASE_LIMIT_UI):
        diagnostic = {
            'period_s': float(period),
            'threshold': float(average + offset),
            'threshold_offset': delta,
            'threshold_offset_fraction': _CENTERING_THRESHOLD_FRACTION,
            'phase_shift_limit_ui': _CENTERING_PHASE_LIMIT_UI,
            'nominal_center_s': center,
            'lower_center_s': probes[0],
            'upper_center_s': probes[1],
            'max_phase_shift_ui': maximum,
            'undefined_probe_count': undefined,
        }
    return center, diagnostic


def _crossing_center(t, relative, period) -> Optional[float]:
    """Use identical event and circular-mean rules at every threshold."""
    # Adjacent non-threshold samples provide both sides of an event. Skipping
    # exact zeros counts a vertex once and excludes touches and endpoint runs.
    nonzero = np.flatnonzero(relative != 0)
    left, right = nonzero[:-1], nonzero[1:]
    changes = np.signbit(relative[left]) != np.signbit(relative[right])
    left, right = left[changes], right[changes]
    if not len(left):
        return None

    events = np.empty(len(left))
    direct = right == left + 1
    lo, hi = left[direct], right[direct]
    fraction = -relative[lo] / (relative[hi] - relative[lo])
    events[direct] = t[lo] + fraction * (t[hi] - t[lo])
    lo, hi = left[~direct] + 1, right[~direct] - 1
    events[~direct] = t[lo] + (t[hi] - t[lo]) / 2

    phases = (events / period) % 1.
    mean = np.mean(np.exp(2j * np.pi * phases))
    if abs(mean) <= 1e-12:
        return None
    return float(period * ((np.angle(mean) / (2 * np.pi) + .5) % 1.))
