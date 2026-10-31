"""silk.metrics._guards — the shared input guards every metric calls.

A LEAF module on purpose: it imports numpy and nothing else from silk.
``from __future__ import annotations`` keeps the ``Signal`` annotations
unevaluated so this file never has to import :mod:`silk.signals`.

The metric files depend on ``silk.signals.base``. In ``silk.signals.schema``,
``get_metric()`` and ``export_metric_schema()`` load the ``silk.metrics`` facade
locally when called. Import shared guards directly from this leaf to keep
``ac`` and ``noise`` independent of the other metric implementation files
(waveform, statistics, timing, jitter, eye, power, spectral, rf and converter)
and preserve that lazy dependency relationship. The function-local
exception import below runs only when validation needs it.
"""
from __future__ import annotations

import numpy as np


def _check_nonempty(sig: Signal, metric: str) -> None:  # noqa: F821
    """Reject a zero-length Signal.

    An empty Signal is a MALFORMED REQUEST — the caller handed the
    metric no data at all — so it raises, per the caller-error convention.
    A simulation that genuinely produced nothing is
    :class:`~silk.simulate.SimulationAborted`'s job, not a metric's.

    Zero samples ONLY. A one-sample Signal is legitimate input to
    several metrics (``overshoot`` reads the lone sample as the peak,
    ``on_time`` reports a zero-duration trace) and is deliberately left
    alone; a metric that genuinely needs >= 2 points says so itself.

    A metric that DELEGATES to another one (``slew_rate`` -> ``rise_time``,
    the jitter family -> ``crossings``, ``sinad``/``enob`` -> ``sndr``)
    calls this itself as well, redundantly but deliberately: without it
    the message would name the callee, and the caller is what the user
    actually invoked.
    """
    if len(sig.x) == 0:
        raise ValueError(
            f"{metric}: Signal {sig.name!r} has no samples (len(x) == 0); "
            f"a metric cannot measure an empty trace"
        )


def _check_freq(sig: Signal, metric: str) -> None:  # noqa: F821
    """Reject a Signal that is not a frequency sweep, or one
    with no samples at all.

    Every AC and noise-sweep metric reads ``sig.x`` as Hz, so a transient
    Signal on the wrong axis used to return a confident number (40 dB
    "DC gain" of a 100 V constant transient) instead of raising.

    Shared by ``silk.metrics.ac`` and ``silk.metrics.noise``, so both modules
    apply the same frequency-axis validation.

    The wrong-axis refusal raises
    :class:`~silk.signals.kinds.SignalKindError`. That is a
    ``ValueError`` SUBCLASS, so existing ``except ValueError`` handlers
    still catch it. The class identifies a wrong signal KIND, which
    :meth:`silk.circuit.module_test.ModuleTestbench.run` promotes to
    :class:`~silk.exceptions.MetricBindingError` so an AC metric handed a
    transient trace raises instead of landing in the report as an error
    string because a malformed request is caller error. Each refusal names the invoking
    metric using the caller-supplied ``metric`` argument.

    Only the axis test is a statement about kind. The emptiness and
    ``< 2 points`` refusals below are about the DATA, so they keep the
    plain ``ValueError`` they have always raised.

    The import is function-local so this module keeps the leaf import
    property its header describes; ``silk.signals.kinds`` is itself a
    leaf (it imports nothing from ``silk``), so no cycle exists either
    way — the same reasoning used when ``silk.metrics.noise``
    began importing it.
    """
    if sig.xlabel != 'frequency':
        from silk.signals.kinds import SignalKindError
        raise SignalKindError(
            f"{metric}: sig must have xlabel='frequency', got {sig.xlabel!r}"
        )
    _check_nonempty(sig, metric)
    _axis(sig, metric)
    if len(sig.x) < 2:
        raise ValueError(
            f'{metric}: need at least 2 frequency points, got {len(sig.x)}'
        )


# ---------------------------------------------------------------------------
# NaN propagation guard
# ---------------------------------------------------------------------------
#
# A NaN sample can come from a failed convergence point and vary per corner.
# NaN data in gives NaN out: never raise and abort a whole corner or Monte
# Carlo run over one bad sample, and never return a plausible finite measurement.
# Most metrics get this right because their arithmetic (sum/mean/trapezoid)
# propagates NaN automatically. The functions that don't — they search
# for a threshold crossing or an argmin/argmax index, operations where a
# NaN sample is silently skipped or silently "wins" instead of poisoning
# the result — call this guard explicitly on the samples they are about
# to consume, before doing anything else with them.

def _any_nan(*arrays) -> bool:
    """True if any of the given arrays contains a NaN.

    A COMPLEX array is checked on its real and imaginary parts
    separately — a caller hands one in directly, ``H(jw)``
    as-is, rather than pre-folding it with ``np.abs``. Folding to
    magnitude first used to be this guard's documented convention
    (``_any_nan(np.abs(y))``), on the claim that ``np.abs`` of a
    complex sample with a NaN in either part is itself NaN. That is
    only true when the OTHER part is finite: ``abs(1+nanj)`` is ``nan``
    (caught), but ``abs(inf+nanj)`` is ``inf`` and ``abs(nan+infj)`` is
    ``inf`` too — both silently passed the guard. Checking
    ``arr.real``/``arr.imag`` directly catches a NaN in either part
    regardless of what the other part holds, and
    still leaves a plain infinity alone: infinity is legitimate AC data
    (a zero at DC reads ``-inf`` dB, a pole at DC reads an infinite
    magnitude), not a sample this guard should refuse.

    A non-complex array is cast to float, so it also works on plain
    lists and other object-free numeric input.
    """
    return any(np.any(_isnan_complex_aware(a)) for a in arrays)


def _isnan_complex_aware(a):
    """Elementwise ``isnan``, correct on a complex array.

    Casting a complex array to float (what a plain ``.astype(float)``
    guard does) silently DROPS the imaginary part under a
    ``ComplexWarning`` rather than raising, so a guard built that way
    was always fed a pre-folded magnitude instead (``np.abs(y)``). That
    fold is itself the bug this module fixes: ``np.abs`` of a complex
    sample with an infinite real (or imaginary) part is ``inf``, which
    swallows a NaN sitting in the other part (see :func:`_any_nan`).
    Checking ``a.real``/``a.imag`` separately avoids the fold entirely.
    Shared by :func:`_any_nan` (any-of-array) and by callers that need
    the per-sample mask itself, such as ``psrr_db``'s NaN/zero
    diagnostics.
    """
    arr = np.asarray(a)
    if np.iscomplexobj(arr):
        return np.isnan(arr.real) | np.isnan(arr.imag)
    return np.isnan(arr.astype(float))


def _nan_safe_transform(values, transform):
    """Transform samples without losing NaN in either raw complex component.

    The transform preserves shape (magnitude or the existing real projection).
    Return fresh storage, including when the transform returns its input. Do
    not warn here: each metric owns its consumed support and diagnostics.
    """
    raw = np.asarray(values)
    contaminated = _isnan_complex_aware(raw)
    result = np.array(transform(raw), copy=True)
    if np.any(contaminated):
        result[contaminated] = np.nan
    return result


def _bracket_freq(x, freq):
    """Bracket *freq* on a sweep grid: ``(i0, i1, t)`` into ``x``.

    ``t`` is the fraction of the way from ``x[i0]`` to ``x[i1]`` in LOG
    frequency. A request exactly on a grid point returns that sample
    twice with ``t == 0``, so interpolating modes return the sample
    itself. Works on strictly rising or falling coordinates; *freq* must
    lie inside the sweep and be > 0 for ``t`` to be meaningful (non-positive grids fall back to
    linear frequency).
    """
    x = np.asarray(x, dtype=float)
    hit = np.flatnonzero(x == freq)
    if hit.size:
        return int(hit[0]), int(hit[0]), 0.0
    axis = _Axis(x, 'frequency interpolation')
    order = axis.indices
    xs = axis.increasing
    k = int(axis.search(freq))
    k = min(max(k, 1), len(xs) - 1)
    i0, i1 = int(order[k - 1]), int(order[k])
    a, b = x[i0], x[i1]
    if a > 0 and b > 0 and freq > 0:
        t = float(np.log(freq / a) / np.log(b / a))
    else:
        t = float((freq - a) / (b - a))
    return i0, i1, t


class _Axis:
    """Validated coordinates, with separate acquisition and increasing views.

    ``path`` accepts fold-backs for masks and path integrals. ``nondecreasing``
    retains repeated power-table entries; neither permits interpolation.
    All numerical searches use the increasing view, never an arbitrary sort.
    """

    def __init__(self, x, metric, y=None, *, order='either'):
        self.x = np.asarray(x, dtype=float)
        self.metric = metric
        self.policy = order
        need = {'either': 'an axis that only rises or only falls',
                'increasing': 'a strictly increasing axis that only rises',
                'nondecreasing': 'an ascending axis that rises or repeats',
                'path': 'an axis in acquisition order'}[order]

        def refuse(found):
            raise ValueError(
                f'{metric}: the x-axis {found}; {metric} needs finite, matching 1-D '
                f'coordinates and samples on {need}.')

        if self.x.ndim != 1:
            refuse(f'has shape {self.x.shape}')
        if y is not None and (np.ndim(y) != 1 or len(y) != len(self.x)):
            refuse(f'has {len(self.x)} coordinates but samples have shape {np.shape(y)}')
        if not len(self.x):
            refuse('has no samples')
        invalid = np.flatnonzero(~np.isfinite(self.x))
        if len(invalid):
            i = int(invalid[0])
            refuse(f'has a nonfinite coordinate at index {i} (x={self.x[i]:g})')
        self.descending = False
        if order != 'path' and len(self.x) > 1:
            delta = np.diff(self.x)
            up, down, equal = delta > 0, delta < 0, delta == 0
            if up.any() and down.any():
                first_up = bool(delta[np.flatnonzero(delta)[0]] > 0)
                i = int(np.flatnonzero(down if first_up else up)[0])
                turn = 'up then down' if first_up else 'down then up'
                refuse(f'goes {turn} at index {i} '
                       f'(x={self.x[i]:g} -> {self.x[i + 1]:g})')
            if equal.any() and order != 'nondecreasing':
                i = int(np.flatnonzero(equal)[0])
                refuse(f'repeats a coordinate at index {i} '
                       f'(x={self.x[i]:g} -> {self.x[i + 1]:g})')
            self.descending = bool(down.any())
            if self.descending and order in ('increasing', 'nondecreasing'):
                i = int(np.flatnonzero(down)[0])
                refuse(f'goes down at index {i} '
                       f'(x={self.x[i]:g} -> {self.x[i + 1]:g})')
        self.lower = float(np.min(self.x))
        self.upper = float(np.max(self.x))

    @property
    def increasing(self):
        return self.x[::-1] if self.descending else self.x

    @property
    def indices(self):
        indices = np.arange(len(self.x))
        return indices[::-1] if self.descending else indices

    def orient(self, values):
        """Pair samples with increasing x, or restore their acquisition order."""
        values = np.asarray(values)
        return values[::-1] if self.descending else values

    def same_grid(self, other):
        """Match acquisition coordinates within 1e-3 of the smallest spacing.

        Tolerance depends on spacing, never units or the coordinate origin.
        Singleton grids must match exactly because they have no spacing.
        Unequal grids are conservatively refused if a spacing overflows;
        exact equal finite axes still match, including extreme coordinates.
        Both axes have already passed their caller's validation policy.
        band_noise_figure deliberately uses stricter exact same-capture equality.
        """
        if self.x.shape != other.x.shape:
            return False
        if np.array_equal(self.x, other.x):
            return True
        # Finite coordinates can still overflow on subtraction. Never allow
        # an infinite spacing to produce an unbounded matching tolerance.
        with np.errstate(over='ignore'):
            steps = np.concatenate((np.abs(np.diff(self.x)),
                                    np.abs(np.diff(other.x))))
            if not np.all(np.isfinite(steps)):
                return False
            steps = steps[steps > 0]
            if not steps.size:
                return False
            return bool(np.all(np.abs(self.x - other.x) <= 1e-3 * steps.min()))

    def contains(self, point):
        return (np.asarray(point) >= self.lower) & (np.asarray(point) <= self.upper)

    def search(self, points, *, side='left'):
        if self.policy in ('path', 'nondecreasing'):
            raise ValueError(f'{self.metric}: searching needs a strictly monotonic axis')
        return np.searchsorted(self.increasing, points, side=side)

    def interp(self, points, values):
        if self.policy in ('path', 'nondecreasing'):
            raise ValueError(f'{self.metric}: interpolation needs a strictly monotonic axis')
        return np.interp(points, self.increasing, self.orient(values))

    def span(self, start=None, end=None):
        return (self.lower if start is None else max(float(start), self.lower),
                self.upper if end is None else min(float(end), self.upper))

    def window(self, values, start=None, end=None, *, interpolate=True):
        """Coordinate-ordered samples, inserting only in-span interval edges."""
        x = self.increasing
        y = self.orient(values)
        a = self.lower if start is None else float(start)
        b = self.upper if end is None else float(end)
        if b <= a:
            raise ValueError(
                f'{self.metric}: require end > start, got start={a}, end={b}; '
                f'trace spans [{self.lower}, {self.upper}]')
        mask = (x >= a) & (x <= b)
        xw, yw = x[mask], y[mask]
        if interpolate:
            lo = [a] if self.lower < a < self.upper and not (xw.size and xw[0] == a) else []
            hi = [b] if self.lower < b < self.upper and not (xw.size and xw[-1] == b) else []
            if lo or hi:
                xw = np.concatenate([lo, xw, hi])
                yw = np.concatenate([self.interp(lo, values), yw,
                                     self.interp(hi, values)])
        return xw, yw

    def segment_window(self, start=None, end=None):
        """Acquisition indices of segments touching a coordinate interval."""
        if ((start is not None and start > self.upper)
                or (end is not None and end < self.lower)):
            raise ValueError(
                f'{self.metric}: requested window [{start}, {end}] does not '
                f'overlap the trace; available trace spans [{self.lower}, {self.upper}]')
        lo = 0 if start is None else max(int(self.search(start, side='right')) - 1, 0)
        hi = len(self.x)-1 if end is None else min(int(self.search(end)), len(self.x)-1)
        if self.descending:
            return len(self.x)-1-hi, len(self.x)-1-lo
        return lo, hi


def _axis(sig, metric, *, order='either'):
    return _Axis(sig.x, metric, sig.y, order=order)


class _SampleSpan:
    """One count and endpoint contract for regularly sampled traces.

    Tolerance is measured in periods, not absolute seconds. It only permits
    the endpoint rounding already admitted by the available-count calculation.
    """

    _PERIOD_TOLERANCE = 1e-9

    def __init__(self, axes, origin, period, *, offset=0.):
        self.metric = axes[0].metric
        self.lower = max(axis.lower for axis in axes)
        self.upper = min(axis.upper for axis in axes)
        self.origin, self.period = origin, period
        self.offset = offset
        self.tolerance = self._PERIOD_TOLERANCE * period
        self.max_count = int(np.floor((self.upper-origin)/period-offset
                                      + self._PERIOD_TOLERANCE)) + 1

    def times(self, count, *, count_name='n_samples'):
        times = self.origin + (np.arange(count) + self.offset)*self.period
        if (times[0] < self.lower
                or times[-1] > self.upper + self.tolerance):
            raise ValueError(
                f'{self.metric}: sample window [{times[0]}, {times[-1]}] '
                f'exceeds trace end; trace spans {self.lower}..{self.upper}; '
                f'reduce {count_name}={count} or tstart={self.origin}')
        return np.minimum(times, self.upper)
