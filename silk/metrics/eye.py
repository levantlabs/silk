"""Serial-data eye diagrams, eye height and eye width from time waveforms.

The eye fold uses recorded samples without resampling; geometric measurements
use the folded segments. Each function documents eligibility and unavailable eyes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from silk.signals.base import Signal
from silk.signals.schema import metric, Transient
from silk.metrics.diagnostics import emit
from silk.metrics.waveform import _axis_context, _check_transient, _timing_details


@dataclass(frozen=True)
class EyeResult:
    """Result of an :func:`eye_diagram` fold.

    ``traces`` contains original samples folded into chronological windows.
    With ``auto_center=True``, ``sampling_instant_s`` reports the selected
    display centre relative to the original input origin; the returned windows
    place that centre at half a period. Undeterminable centering returns the
    original fold and a ``nan`` phase.

    Height and width describe the opening at an explicit amplitude threshold,
    not a recovered symbol alphabet or a BER measurement. No threshold, or
    disabled, unavailable or sensitive centering, leaves measurements ``nan``
    and their tuples empty. See :func:`eye_diagram` for limitations.

    Parameters
    ----------
    height : float
        Distance between the nearest centre observations bounding the supplied
        threshold, in the input amplitude unit. A trace at the threshold gives
        zero; absent bounds give ``nan``. A closed eye can read a small positive
        gap: use a meaningful spec margin, not merely ``height > 0``.
    width : float
        Clear interval at the supplied threshold, in seconds. Zero for a
        centre contact; ``nan`` when measurements are unavailable.
    traces : list of Signal
        The overlaid windows, one Signal per fold window, in window
        order. FILLED. Each carries its OWN x (time within the period)
        and y (the original sample values, bit-identical): there is no
        shared time axis, because a shared axis is exactly what forced
        the old implementation to resample. Plot the eye by
        drawing every element onto one axis — see :func:`eye_diagram`.
    sampling_instant_s : float
        Selected crossing-based display centre in ``[0, period)``,
        relative to the ORIGINAL ``sig.x[0]``, when ``auto_center=True``.
        Absolute sampling instants are ``sig.x[0] + sampling_instant_s +
        k*period``. The centred display places that lattice at ``period/2``;
        this field reports the original phase, not the display coordinate.
        ``nan`` when centering is disabled or the average-voltage crossing
        phase is undefined. This display alignment does not establish an open
        eye or an optimal receiver sampling point.
    levels : tuple of float
        Singleton containing the requested measurement threshold when eligible
        for measurement; otherwise empty. NOT inferred symbol levels.
    heights : tuple of float
        Singleton containing ``height`` when eligible for measurement;
        otherwise empty. Units follow ``sig.ylabel``.
    widths : tuple of float
        Clear horizontal intervals containing the display centre at the
        corresponding ``levels``, bounded by nearest waveform contacts on
        either side on the UI circle, in seconds. Includes touches/plateaus.
        Singleton containing ``width`` when eligible, otherwise empty. All
        three tuples are aligned; none identifies transmitted symbols.

    Returns
    -------
    EyeResult
        Frozen result container. Field assignment is prohibited, but contained
        lists, arrays and Signals remain mutable. Construction stores the supplied
        fields without validation or copying and emits no MetricWarning.

    Attributes
    ----------
    height : float
        Distance between the nearest centre observations bounding the supplied
        threshold, in the input amplitude unit. A trace at the threshold gives
        zero; absent bounds give ``nan``. A closed eye can read a small positive
        gap: use a meaningful spec margin, not merely ``height > 0``.
    width : float
        Clear interval at the supplied threshold, in seconds. Zero for a
        centre contact; ``nan`` when measurements are unavailable.
    traces : list of Signal
        The overlaid windows, one Signal per fold window, in window
        order. FILLED. Each carries its OWN x (time within the period)
        and y (the original sample values, bit-identical): there is no
        shared time axis, because a shared axis is exactly what forced
        the old implementation to resample. Plot the eye by
        drawing every element onto one axis — see :func:`eye_diagram`.
    sampling_instant_s : float
        Selected crossing-based display centre in ``[0, period)``,
        relative to the ORIGINAL ``sig.x[0]``, when ``auto_center=True``.
        Absolute sampling instants are ``sig.x[0] + sampling_instant_s +
        k*period``. The centred display places that lattice at ``period/2``;
        this field reports the original phase, not the display coordinate.
        ``nan`` when centering is disabled or the average-voltage crossing
        phase is undefined. This display alignment does not establish an open
        eye or an optimal receiver sampling point.
    levels : tuple of float
        Singleton containing the requested measurement threshold when eligible
        for measurement; otherwise empty. NOT inferred symbol levels.
    heights : tuple of float
        Singleton containing ``height`` when eligible for measurement;
        otherwise empty. Units follow ``sig.ylabel``.
    widths : tuple of float
        Clear horizontal intervals containing the display centre at the
        corresponding ``levels``, bounded by nearest waveform contacts on
        either side on the UI circle, in seconds. Includes touches/plateaus.
        Singleton containing ``width`` when eligible, otherwise empty. All
        three tuples are aligned; none identifies transmitted symbols.
    """
    height: float
    width: float
    traces: List[Signal]
    sampling_instant_s: float = float('nan')
    levels: tuple[float, ...] = ()
    heights: tuple[float, ...] = ()
    widths: tuple[float, ...] = ()


def _check_eye_threshold(threshold, metric_name):
    """Validate a single explicit measurement line, not an inferred level."""
    try:
        if np.ndim(threshold) != 0 or np.iscomplexobj(threshold):
            raise ValueError
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{metric_name}: threshold must be a finite real scalar') from None
    return threshold


@metric(domain='tran', outputs={'height': '[y]', 'width': 's',
                               'sampling_instant_s': 's'},
        param_units={'period': 's', 'threshold': '[y]'})
def eye_diagram(sig: Transient, *, period: float,
                n_traces: Optional[int] = None,
                auto_center: bool = False,
                threshold: Optional[float] = None) -> EyeResult:
    """Fold original transient samples, optionally aligning their crossing phase.

    Each returned Signal has its own original sample positions and bit-identical
    values. No shared output grid, resampling or artificial seam connection is
    introduced. Supply a measurement threshold and enable auto-centering to
    fill geometric height and width when centering is eligible.

    Parameters
    ----------
    sig : Signal
        Matching one-dimensional x/y arrays, with finite strictly increasing
        time coordinates in seconds (``xlabel='time'``).
    period : float
        Supplied unit interval in seconds. No clock recovery is performed.
    n_traces : int, optional
        Exact number of full windows to return. Defaults to all full windows
        after the chosen origin. Centering can leave one fewer full window;
        an explicit count that no longer fits raises ``ValueError``. The
        centering analysis uses the input independently of this output limit;
        height/width measurements use only the returned full-window span.
    auto_center : bool, default False
        Find all crossings of the full record's time-weighted average voltage.
        Select the equal-weight circular mean crossing phase plus half a UI,
        then shift the windows so this centre plots at ``period/2``. The
        default preserves the original fold and leaves measurements unset.
    threshold : float, optional
        Explicit amplitude at which to measure one eye, in ``sig.ylabel``
        units. There is no inferred default: omitting it requests only folding
        and optional centering. For PAM, call once per desired eye threshold
        and take the minimum of the resulting scalar measurements if needed.
        This does not change the average-voltage threshold used for centering.

    Returns
    -------
    EyeResult
        ``traces`` is a chronological list of Signals. ``sampling_instant_s``
        is the selected phase ``c`` relative to the ORIGINAL ``sig.x[0]``.
        It is ``nan`` when centering is disabled or undeterminable.
        For eligible measurements, ``levels``, ``heights`` and ``widths`` are
        singletons containing the supplied threshold, ``height`` and ``width``.
        Otherwise they are empty and scalar measurements are ``nan``.

    Raises
    ------
    ValueError
        For an empty/non-transient signal, nonpositive/NaN period, input shorter
        than two periods, or an invalid/full-window count that does not fit.
        Centering additionally requires real samples, a finite period and a
        finite, strictly increasing time axis with matching 1-D arrays.
        A supplied threshold must be a finite real scalar.

    Notes
    -----
    The input Signal objects and their arrays are not modified.

    Without centering, ``origin = sig.x[0]``. On success::

        shift = (c - period/2) % period
        origin = sig.x[0] + shift
        phase = sample_time - (origin + k*period)

    Window ``k`` contains exactly the original samples in the half-open span
    ``[origin + k*period, origin + (k+1)*period)``. A boundary sample belongs
    only to the later window. The leading partial span before the shifted
    origin and the trailing partial window are omitted; every covered sample
    appears once. Empty windows remain in the list. No endpoint is synthesized.

    The CENTERING threshold is the piecewise-linear time average of the input:
    ``sum(diff(x) * (y[:-1] + y[1:]) / 2) / (x[-1] - x[0])``. Additional
    vertices on a straight segment do not reweight time. Every rising and
    falling crossing contributes equally, including ringing. Crossing times
    are interpolated between original vertices for analysis only. An exact
    threshold vertex counts once; a threshold plateau counts at its midpoint
    only when the non-threshold samples on its sides have opposite signs.
    Same-side touches and threshold endpoints without both sides do not count.

    For crossing times ``tc``, let ``p = ((tc - x[0]) / period) % 1`` and
    ``z = mean(exp(2j*pi*p))``. The centre is
    ``c = period * ((angle(z)/(2*pi) + 0.5) % 1)``. A single crossing suffices.
    Nonfinite voltage, no crossings (including constant data), or
    ``abs(z) <= 1e-12`` return the original uncentred fold and a ``nan`` phase.
    This guard only handles numerical zero; it is not a confidence cutoff.
    Requested but unavailable centering emits exactly one ``MetricWarning``
    with code ``centering_unavailable`` and available input context.
    Disabled centering and intentionally unset result fields stay quiet.

    An available centre is also checked at thresholds offset by +/-5% of the
    observed peak-to-peak range. If either probe moves more than 0.10 UI from
    the original centre (shortest circular distance), or becomes undefined,
    one advisory ``MetricWarning`` with code ``centering_sensitive`` is emitted.
    The original centre and traces are unchanged; height/width remain ``nan``
    and their per-opening tuples empty even if warnings are ignored. These fixed
    heuristic limits are not a confidence score: no warning does not certify
    correct alignment. No symbols or voltage levels are inferred. Diagnostic
    amplitudes use the input's ``ylabel`` unit; phase details are relative to
    the original input start. Normal Python warning filters apply.

    This is display alignment, not clock recovery, opening optimisation or
    proof of an open eye. Uneven level populations, ringing and distributed
    crossing times can make the alignment less satisfactory. A closed opening
    can still have a definable crossing phase. No symbol count is inferred.
    Clipping a record can change its time average and crossing population.

    With an explicit threshold, an available, non-sensitive centre and at
    least two output windows, interpolate the ORIGINAL piecewise-linear
    waveform at each window's centre. Height is the nearest observation above
    the threshold minus the nearest below. An observation exactly at the
    threshold gives zero height and width. Missing either boundary gives
    ``nan`` and ``eye_threshold_unbracketed``. Width is the clear interval
    containing the centre at that SAME supplied threshold, between nearest
    waveform contacts on either side on the UI circle. Contacts include
    same-side touches, plateau endpoints and original segments straddling
    window boundaries. One distinct contact phase gives a full-UI width.
    No output samples change.

    Fewer than two centre observations emit ``insufficient_eye_samples`` and
    leave measurements unset while retaining the phase and fold. This point
    geometry cannot identify overlapping symbol populations: a closed noisy
    eye can report a small positive spacing between neighbouring traces.
    Specs need a meaningful positive margin, not merely ``height > 0``. No
    measurement window, level count, symbol labels or BER are inferred.

    Examples
    --------
    The returned traces show the centering directly::

        result = eye_diagram(vrx, period=ui, auto_center=True)
        fig, ax = plt.subplots()
        for window in result.traces:
            window.plot(ax=ax, color='steelblue', lw=0.5, alpha=0.3)

    Examples
    --------
    >>> import numpy as np
    >>> from silk.signals import Signal
    >>> from silk.metrics import eye_diagram
    >>> sig = Signal('data', 'voltage', 'time', 'V',
    ...              np.arange(7.) * 0.5, np.array([0., 1., 0., 1., 0., 1., 0.]))
    >>> result = eye_diagram(sig, period=1.0)
    >>> len(result.traces)
    3
    >>> result.traces[0].x.tolist()
    [0.0, 0.5]
    """
    _check_transient(sig, 'eye_diagram')
    if threshold is not None:
        threshold = _check_eye_threshold(threshold, 'eye_diagram')
    period = float(period)
    if not period > 0.0:                      # also catches nan
        raise ValueError(f'eye_diagram: period must be > 0, got {period}')

    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y)
    if auto_center:
        if np.iscomplexobj(y):
            raise ValueError('eye_diagram: auto_center requires real voltage samples')
        if not np.isfinite(period):
            raise ValueError(f'eye_diagram: period must be finite, got {period}')
    t0 = float(x[0])
    t_total = float(x[-1] - x[0])
    max_traces = int(t_total / period)
    if max_traces < 2:
        raise ValueError(
            f'eye_diagram: trace duration {t_total} fits fewer than 2 periods '
            f'of {period}; {max_traces} complete periods available; '
            f'{_axis_context(sig)}'
        )
    explicit_count = n_traces is not None
    if n_traces is None:
        n_traces = max_traces
    else:
        n_traces = int(n_traces)
        if n_traces < 1:
            raise ValueError(
                f'eye_diagram: n_traces must be >= 1, got {n_traces}; '
                f'{max_traces} traces available at period={period}; '
                f'{_axis_context(sig)}'
            )
        if n_traces > max_traces:
            raise ValueError(
                f'eye_diagram: requested {n_traces} traces, only {max_traces} fit '
                f'at period={period}; {_axis_context(sig)}'
            )

    sampling_instant = float('nan')
    measure_openings = False
    if auto_center:
        # Keep optional analysis separate from the lossless sample fold.
        from silk.metrics._eye import _analyse_eye
        analysis = _analyse_eye(x, y, period)
        if analysis is not None:
            sampling_instant, sensitivity = analysis
            shift = (sampling_instant - period / 2) % period
            t0 += shift
            shifted_count = int((x[-1] - t0) / period)
            if explicit_count and n_traces > shifted_count:
                raise ValueError(
                    f'eye_diagram: centering shifts the fold origin by {shift:g} s; '
                    f'only {shifted_count} full traces fit, requested {n_traces}. '
                    'Reduce n_traces, supply more input, or use auto_center=False.'
                )
            if not explicit_count:
                n_traces = shifted_count
            measure_openings = sensitivity is None
            if sensitivity is not None:
                message = (
                    'Centering may be unreliable: a nearby crossing threshold '
                    'has no defined phase.'
                    if sensitivity['undefined_probe_count'] else
                    'Centering may be unreliable: changing the crossing '
                    'threshold shifts the inferred center substantially.'
                )
                emit(
                    metric='eye_diagram', code='centering_sensitive',
                    message=message,
                    details={**_timing_details(sig), **sensitivity,
                             'magnitude_unit': str(sig.ylabel)},
                    suggestion='Inspect the eye alignment before using '
                               'center-dependent measurements. The original '
                               'center and traces are unchanged.',
                )
        else:
            emit(
                metric='eye_diagram', code='centering_unavailable',
                message='Centering could not be determined from the supplied data.',
                details={
                    **_timing_details(sig), 'period_s': period,
                    'full_period_count': max_traces,
                    'nonfinite_sample_count': int(np.count_nonzero(~np.isfinite(y))),
                },
                suggestion='Inspect the waveform and supplied period; provide more '
                           'usable input or use auto_center=False to request only the fold.',
            )

    # Which window each ORIGINAL sample belongs to. Grouping by this index
    # (rather than slicing the array by time) is what keeps the partition
    # exact: every sample is assigned once, by the same expression the
    # phase is then measured against.
    k = np.floor((x - t0) / period).astype(np.int64)
    # The division can round ACROSS a window edge: a sample a fraction of
    # an ulp below ``t0 + k*period`` can still divide to exactly ``k``, and
    # it would then carry a (tiny) NEGATIVE phase in window k. Settle the
    # assignment against the very expression the phase is measured from, so
    # each sample sits in the window whose origin it is actually at or
    # after. One step in each direction is enough — the floor is never more
    # than an ulp out — and the partition stays exact either way.
    k -= (x < t0 + k * period)
    k += (x >= t0 + (k + 1) * period)
    order = np.argsort(k, kind='stable')       # stable => original order
    ks = k[order]
    idx = np.arange(n_traces, dtype=np.int64)
    lo = np.searchsorted(ks, idx, side='left')
    hi = np.searchsorted(ks, idx, side='right')

    traces: List[Signal] = []
    for w in range(n_traces):
        take = order[lo[w]:hi[w]]
        # The window's own origin, subtracted from the sample's own time:
        # phase + start reproduces the original time exactly, so the fold
        # can be undone sample by sample.
        start = t0 + w * period
        traces.append(Signal(
            f'eye({sig.name})[{w}]', sig.type, sig.xlabel, sig.ylabel,
            x[take] - start, y[take],
        ))

    nan = float('nan')
    height, width = nan, nan
    levels, heights, widths = (), (), ()
    if measure_openings and threshold is not None:
        if n_traces < 2:
            emit(
                metric='eye_diagram', code='insufficient_eye_samples',
                message='Eye height and width need at least two center observations.',
                details={**_timing_details(sig), 'period_s': period,
                         'center_sample_count': n_traces},
                suggestion='Increase n_traces or supply more full periods. '
                           'The selected center and original traces are retained.',
            )
        else:
            # Local import keeps optional geometry separate from the sample fold.
            from silk.metrics._eye import _measure_eye
            levels, heights, widths = _measure_eye(x, y, period, t0, n_traces, threshold)
            height, width = heights[0], widths[0]
            if not np.isfinite(height):
                emit(
                    metric='eye_diagram', code='eye_threshold_unbracketed',
                    message='Center observations do not bound the supplied threshold.',
                    details={**_timing_details(sig), 'period_s': period,
                             'threshold': threshold, 'magnitude_unit': str(sig.ylabel),
                             'center_sample_count': n_traces},
                    suggestion='Inspect the centered traces and choose a threshold '
                               'inside the desired eye, or include more full periods.',
                )
            elif not np.isfinite(width):
                emit(
                    metric='eye_diagram', code='eye_width_unavailable',
                    message='No waveform contacts bound the requested opening.',
                    details={**_timing_details(sig), 'period_s': period,
                             'threshold': threshold},
                    suggestion='Inspect the waveform at the returned measurement levels.',
                )
    return EyeResult(height, width, traces, sampling_instant_s=sampling_instant,
                     levels=levels, heights=heights, widths=widths)


@metric(domain='tran', unit='[y]', param_units={'period': 's', 'threshold': '[y]'})
def eye_height(sig: Transient, *, period: float, threshold: float,
               n_traces: Optional[int] = None, auto_center: bool = True) -> float:
    """Geometric opening height at an explicit threshold from :func:`eye_diagram`.

    Centering is enabled by default; disabled, unavailable or sensitive
    centering returns NaN. For PAM, call once per eye and take the minimum.
    A closed eye can read a small positive nearest-trace gap: specs need a
    meaningful margin, not merely > 0. No symbol populations are inferred.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    period : float
        Supplied positive period in seconds.
    threshold : float
        Required finite real amplitude in input y units defining the opening.
    n_traces : int, optional
        Number of full fold windows; None uses all that fit after centering.
    auto_center : bool
        Align crossing phases before measurement; defaults to True for scalar eye
        metrics.

    Returns
    -------
    float
        Opening height in input y units; nan when measurement is unavailable.
        Disabled centering stays quiet; unavailable/sensitive centering,
        insufficient centre observations or an unbracketed threshold emit
        MetricWarning through eye_diagram.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    ValueError is raised for a malformed transient, nonpositive/nonfinite
    period, nonfinite or complex threshold, insufficient period coverage or
    a count that does not fit. Centering requires real waveform samples.
    """
    _check_transient(sig, 'eye_height')
    threshold = _check_eye_threshold(threshold, 'eye_height')
    return eye_diagram(sig, period=period, n_traces=n_traces,
                       auto_center=auto_center, threshold=threshold).height


@metric(domain='tran', unit='s', param_units={'period': 's', 'threshold': '[y]'})
def eye_width(sig: Transient, *, period: float, threshold: float,
              n_traces: Optional[int] = None, auto_center: bool = True) -> float:
    """Opening width in seconds at an explicit threshold from :func:`eye_diagram`.

    Centering is enabled by default; disabled, unavailable or sensitive
    centering returns NaN. Inherits the parent's point-geometry limitations;
    it does not determine whether transmitted symbol populations overlap.

    Parameters
    ----------
    sig : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time".
    period : float
        Supplied positive period in seconds.
    threshold : float
        Required finite real amplitude in input y units defining the opening.
    n_traces : int, optional
        Number of full fold windows; None uses all that fit after centering.
    auto_center : bool
        Align crossing phases before measurement; defaults to True for scalar eye
        metrics.

    Returns
    -------
    float
        Opening width in seconds; nan when measurement is unavailable. Disabled
        centering stays quiet; unavailable/sensitive centering, insufficient
        observations, unbracketed threshold or absent contacts emit MetricWarning
        through eye_diagram.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    ValueError is raised for a malformed transient, nonpositive/nonfinite
    period, nonfinite or complex threshold, insufficient period coverage or
    a count that does not fit. Centering requires real waveform samples.
    """
    _check_transient(sig, 'eye_width')
    threshold = _check_eye_threshold(threshold, 'eye_width')
    return eye_diagram(sig, period=period, n_traces=n_traces,
                       auto_center=auto_center, threshold=threshold).width
