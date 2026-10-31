"""silk.signals.view.encode — numeric encoding for the waveform explorer.

Pure numpy: base64-encoded typed arrays, the decimation pyramid, and the
nutmeg-type -> unit table. No knowledge of plots, payload shape, or the
rawfile reader — :mod:`silk.signals.view.emit` is the only caller.

Encoding constants are shared with the viewer frontend; hand-authored sample
payloads must match this module's output.
"""
import base64
import math

import numpy as np

__all__ = [
    'DECIMATE_MIN_POINTS', 'DECIMATE_BASE', 'DECIMATE_MIN_BUCKET',
    'DECIMATE_MAX_LEVELS',
    'UNIT_TABLE', 'unit_for_type',
    'b64_f4', 'b64_f8', 'decide_y_dtype', 'count_nonfinite',
    'decimation_strides', 'minmax_level',
    'build_x_signal', 'build_data_signal',
]

# ---------------------------------------------------------------------------
# Decimation pyramid tuning (spec §4.8) — module-level defaults, all
# overridable per call via emit_view() keywords.
# ---------------------------------------------------------------------------
DECIMATE_MIN_POINTS = 8000
DECIMATE_BASE = 8
DECIMATE_MIN_BUCKET = 1000
DECIMATE_MAX_LEVELS = 8

# Float32 range escape hatch (spec §4.2): beyond this magnitude float32
# overflows to inf; below this (and nonzero) it flushes to zero.
_F32_MAX_SAFE = 3.0e38
_F32_MIN_SAFE = 1e-37

# ---------------------------------------------------------------------------
# Units from nutmeg types (spec §4.7)
# ---------------------------------------------------------------------------
UNIT_TABLE = {
    'voltage': 'V',
    'current': 'A',
    'power': 'W',
    'charge': 'C',
    'capacitance': 'F',
    'notype': '',
    '': '',
    'time': 's',
    'frequency': 'Hz',
    'temp-sweep': '°C',
    'res-sweep': 'Ω',
    'impedance': 'Ω',
    'admittance': 'S',
}


def unit_for_type(type_str):
    """Map a nutmeg type string (lower-cased) to its unit, '' if unknown."""
    return UNIT_TABLE.get((type_str or '').lower(), '')


# ---------------------------------------------------------------------------
# Base64 encoding
# ---------------------------------------------------------------------------

def b64_f4(arr):
    """Base64-encode *arr* as little-endian float32."""
    return base64.b64encode(np.asarray(arr).astype('<f4').tobytes()).decode('ascii')


def b64_f8(arr):
    """Base64-encode *arr* as little-endian float64."""
    return base64.b64encode(np.asarray(arr).astype('<f8').tobytes()).decode('ascii')


def decide_y_dtype(arr):
    """Return ``'f32'`` or ``'f64'`` for a Y array per the range escape hatch.

    ``'f64'`` iff some finite value has ``|v| > 3.0e38`` or some nonzero
    finite value has ``|v| < 1e-37`` (float32 would overflow or flush to
    zero). NaN/Inf entries never trigger promotion — they survive the
    float32 round trip unchanged.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return 'f32'
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 'f32'
    mags = np.abs(finite)
    if np.any(mags > _F32_MAX_SAFE):
        return 'f64'
    nonzero = mags[mags != 0]
    if nonzero.size and np.any(nonzero < _F32_MIN_SAFE):
        return 'f64'
    return 'f32'


def count_nonfinite(arr):
    """Count of NaN/+-Inf entries in *arr*."""
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    return int(np.count_nonzero(~np.isfinite(arr)))


def _encode_y(arr, dt):
    return b64_f4(arr) if dt == 'f32' else b64_f8(arr)


# ---------------------------------------------------------------------------
# The decimation pyramid (spec §4.8)
# ---------------------------------------------------------------------------

def decimation_strides(n, *, min_points=DECIMATE_MIN_POINTS, base=DECIMATE_BASE,
                       min_bucket=DECIMATE_MIN_BUCKET, max_levels=DECIMATE_MAX_LEVELS):
    """Return the list of strides, always starting with ``1`` (level 0).

    Worked examples with the default arguments:

    ``N=8000`` ->
    ``[1]``; ``N=8001`` -> ``[1, 8]``; ``N=16000`` -> ``[1, 8]``;
    ``N=1_000_000`` -> ``[1, 8, 64, 512]``.
    """
    strides = [1]
    if n > min_points:
        stride = 1
        while len(strides) < max_levels:
            stride *= base
            nbuckets = math.ceil(n / stride)
            if nbuckets < min_bucket:
                break
            strides.append(stride)
    return strides


def minmax_level(x, y, stride, n):
    """One min/max-decimated level at *stride* over the first *n* samples.

    Returns ``(x_out, y_out)`` — original sample values at the argmin/argmax
    of each bucket, emitted in original sample order (so the level's X stays
    monotonically non-decreasing, which uPlot requires). A bucket that is
    entirely NaN emits a single ``(x[lo], NaN)`` pair instead of two.

    Vectorized (numpy, no per-bucket Python loop). Equivalent reference
    semantics (also pinned as ``_reference_minmax_level`` in the test
    module)::

        for b in range(nbuckets):
            lo, hi = b*stride, min(b*stride + stride, n)
            seg = y[lo:hi]
            if every value in seg is NaN: emit ONE pair: (x[lo], NaN); continue
            imin = lo + nanargmin(seg); imax = lo + nanargmax(seg)
            if imin <= imax: emit (x[imin],y[imin]), (x[imax],y[imax])
            else:            emit (x[imax],y[imax]), (x[imin],y[imin])

    Implementation: reshape the (NaN-padded, so the ragged tail bucket never
    contaminates a real bucket) first *n* samples into one row per bucket,
    then run ``nanargmin``/``nanargmax`` along the row axis in bulk. The pad
    value is NaN, never +-Inf — +-Inf is legal data, and an Inf
    sentinel could tie with a genuine +-Inf sample and mis-point the argmin/
    argmax at a padded slot. NaN padding cannot cause that: nanargmin/
    nanargmax already skip NaN by definition, real or padded alike, so a
    bucket is "all NaN" (the single-pair case) iff its real samples are all
    NaN, exactly matching the reference. All-NaN rows would make
    ``nanargmin``/``nanargmax`` raise/warn, so those rows are dummy-filled
    before the argmin/argmax call and the result is discarded and replaced
    by the single-pair case afterward.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    y_dtype = y.dtype
    x_n = x[:n]
    y_n = y[:n]

    if n <= 0:
        return np.asarray([], dtype='<f8'), np.asarray([], dtype=y_dtype)

    nbuckets = math.ceil(n / stride)
    pad = nbuckets * stride - n
    y_work = np.concatenate([y_n, np.full(pad, np.nan)]) if pad else y_n
    y2d = y_work.reshape(nbuckets, stride)

    all_nan = np.all(np.isnan(y2d), axis=1)
    # Dummy-fill all-NaN rows so nanargmin/nanargmax neither warn nor raise;
    # their (meaningless) results are overwritten below.
    y2d_safe = np.where(all_nan[:, None], 0.0, y2d) if np.any(all_nan) else y2d

    imin = np.nanargmin(y2d_safe, axis=1)
    imax = np.nanargmax(y2d_safe, axis=1)

    lo = np.arange(nbuckets) * stride
    imin_g = lo + imin
    imax_g = lo + imax

    first_is_min = imin <= imax
    first_idx = np.where(first_is_min, imin_g, imax_g)
    second_idx = np.where(first_is_min, imax_g, imin_g)

    first_x, first_y = x_n[first_idx], y_n[first_idx]
    second_x, second_y = x_n[second_idx], y_n[second_idx]

    if np.any(all_nan):
        first_x = np.where(all_nan, x_n[lo], first_x)
        first_y = np.where(all_nan, np.nan, first_y)

    xs = np.empty(2 * nbuckets, dtype='<f8')
    ys = np.empty(2 * nbuckets, dtype=y_dtype)
    xs[0::2], xs[1::2] = first_x, second_x
    ys[0::2], ys[1::2] = first_y, second_y

    keep = np.ones(2 * nbuckets, dtype=bool)
    keep[1::2] = ~all_nan  # all-NaN buckets emit only the first slot
    return xs[keep], ys[keep]


# ---------------------------------------------------------------------------
# Signal assembly
# ---------------------------------------------------------------------------

def build_x_signal(x_arr, *, type_=''):
    """Build the payload dict for ``Plot.x`` — the shared independent axis.

    A Signal whose ``levels`` has exactly one entry (stride 1) carrying the
    X array under the ``x`` key (no ``y`` key; the "signal" *is* the axis).
    """
    x_arr = np.asarray(x_arr, dtype=np.float64)
    n = len(x_arr)
    return {
        'name': '',           # filled in by emit.py from the vector's own name
        'type': type_,
        'unit': unit_for_type(type_),
        'part': None,
        'group': None,
        'nonfinite': count_nonfinite(x_arr),
        'levels': [
            {'stride': 1, 'n': n, 'dt': 'f64', 'x': b64_f8(x_arr)},
        ],
    }


def build_data_signal(name, type_, y_arr, x_arr, *, part=None, group=None,
                      min_points=DECIMATE_MIN_POINTS, base=DECIMATE_BASE,
                      min_bucket=DECIMATE_MIN_BUCKET, max_levels=DECIMATE_MAX_LEVELS):
    """Build the payload dict for one data Signal (a plotted vector).

    *y_arr* and *x_arr* are full-precision numpy arrays (float64); *x_arr* is
    the plot's shared independent axis, used only to build ``levels[k>=1]``
    (level 0 shares ``plot.x.levels[0].x`` and carries no ``x`` of its own).
    """
    y_arr = np.asarray(y_arr, dtype=np.float64)
    x_arr = np.asarray(x_arr, dtype=np.float64)
    n = len(y_arr)
    dt = decide_y_dtype(y_arr)
    nonfinite = count_nonfinite(y_arr)

    levels = [{'stride': 1, 'n': n, 'dt': dt, 'y': _encode_y(y_arr, dt)}]
    strides = decimation_strides(
        n, min_points=min_points, base=base,
        min_bucket=min_bucket, max_levels=max_levels,
    )
    for stride in strides[1:]:
        xs, ys = minmax_level(x_arr, y_arr, stride, n)
        levels.append({
            'stride': stride,
            'n': len(xs),
            'dt': dt,
            'x': b64_f8(xs),
            'y': _encode_y(ys, dt),
        })

    return {
        'name': name,
        'type': type_,
        # Spec §4.6: magnitude keeps the vector's unit, phase is degrees.
        'unit': 'deg' if part == 'phase' else unit_for_type(type_),
        'part': part,
        'group': group,
        'nonfinite': nonfinite,
        'levels': levels,
    }
