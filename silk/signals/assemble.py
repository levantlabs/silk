"""silk.signals.assemble — build the Signals no analysis hands back.

Three of the seven kinds in the ``SIGNAL_KINDS`` vocabulary (in
``silk/signals/kinds.py``) have an empty ``produced_by``:
``code_transitions``, ``power_sweep`` and ``reference_waveform``. (A
fourth, ``phase_noise_sideband``, was genuinely unproducible and was
retired along with its one consumer, ``phase_noise_at_offset``.) These three are reachable — a user assembles the Signal by
hand from data a simulation already gave them — and **this module is the
supported way to do the two assemblies that are more than a constructor
call**.

Why provide helpers
-------------------------------------------------------------
Because the snippet was already written twice, privately, and
differently. ``silk/examples/flash_adc_example.py`` reconstructs a
converter code with ``np.sum(therm > 0.9, axis=0)`` and then calls none
of silk's five static-linearity metrics;
``silk/examples/sar_adc_example.py`` goes further and reimplements DNL
and INL from scratch. Two ADC examples, zero uses of the ADC metrics —
the cost of leaving a three-line step undefined. A helper is the only
form of "written down" that a third example can be required to use.

What is NOT here, deliberately
------------------------------
``reference_waveform`` gets no helper. A goal waveform is
``Signal(name, type, xlabel, ylabel, x, y)`` with the author's own
arrays — there is no assembly step to share, only a constructor call,
and a wrapper around it would add a name to learn and nothing else. It
is documented instead, in the "Signal kinds no analysis produces"
section of ``docs/metrics-library.md``.

A LEAF-ish module
-----------------
stdlib + numpy + :class:`silk.signals.base.Signal`. Nothing from
:mod:`silk.metrics`, which imports :mod:`silk.signals.base` — see the
same rule stated at length in ``silk/signals/kinds.py``. The
``code_transitions`` Signal itself is built by
:func:`silk.metrics.converter.extract_code_transitions`, which lives on the
metrics side and stays there; what this module supplies is the
*transient code trace* that function takes as its second argument.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from silk.signals.base import Signal

__all__ = ['codes_from_bits', 'power_sweep']


_WEIGHTINGS = ('binary', 'thermometer')


def _bit_signals(bits) -> tuple:
    """Normalise *bits* to a tuple of scalar per-bit Signals.

    Accepts a bus handle (``Signal`` with aggregate metadata, as
    ``SubcircuitInstance.__getattr__`` returns for a multi-bit port) or
    any iterable of scalar Signals.
    """
    if isinstance(bits, Signal):
        if bits._underlying_signals is None:
            raise TypeError(
                "codes_from_bits: a single scalar Signal is not a bit "
                f"vector; {bits.name!r} has width 1. Pass a bus handle "
                "(inst.dout) or a sequence of per-bit Signals "
                "([b0, b1, b2])."
            )
        return tuple(bits)
    try:
        seq = tuple(bits)
    except TypeError:
        raise TypeError(
            "codes_from_bits: bits must be a bus Signal or a sequence of "
            f"per-bit Signals, got {type(bits).__name__}"
        ) from None
    if not seq:
        raise ValueError('codes_from_bits: bits is empty; need at least one bit')
    for i, s in enumerate(seq):
        if not isinstance(s, Signal):
            raise TypeError(
                f"codes_from_bits: bits[{i}] is {type(s).__name__}, not a "
                "Signal. silk metrics take Signals, not raw arrays — load "
                "each bit with load_signal()/probe() first."
            )
    return seq


def codes_from_bits(bits, *, weighting: str = 'binary',
                    msb_first: Optional[bool] = None,
                    threshold: float = 0.5,
                    name: Optional[str] = None) -> Signal:
    """Decode N bit traces into one decimal-code transient.

    The step between a converter's simulation output — N separate
    digital nets — and the input
    :func:`silk.metrics.converter.extract_code_transitions <silk.metrics.extract_code_transitions>` takes: a single
    ``xlabel='time'`` Signal whose ``y`` is the decimal output code on
    the same time axis. That Signal is what unlocks ``inl``, ``dnl``,
    ``is_monotonic`` and ``missing_codes``; nothing silk simulates
    produces it, because a converter's output is bits.

    Parameters
    ----------
    bits : Signal or sequence of Signal
        The per-bit traces, all on one shared time axis. Either a bus
        handle (including a width-one bus) or a sequence such as ``[b0, b1, b2]``.
    weighting : {'binary', 'thermometer'}, optional
        How the bits add up to a code. ``'binary'`` (default) packs them
        as a binary word, the SAR / successive-approximation case.
        ``'thermometer'`` COUNTS the bits above *threshold*, the flash
        case — a 3-bit flash has seven comparators, not three bits, and
        no binary packing of them is meaningful.
    msb_first : bool, optional
        Which end of *bits* carries the highest-order bit.
        **Required for ``weighting='binary'`` and refused for
        ``'thermometer'``.** There is no defensible default: a list of
        bit Signals carries no ordering convention, and the wrong guess
        returns a plausible wrong code rather than an error, which is
        exactly the failure this helper exists to prevent. (The same
        reasoning made ``Pulse``'s ``tr``/``tf`` mandatory.)
        :meth:`silk.signals.base.Signal.as_int <silk.signals.Signal.as_int>` keeps its
        ``msb_first=False`` default because a bus handle's bit order
        comes from the bus declaration; passing that same handle here
        still requires the choice to be stated.
    threshold : float, optional
        A bit is 1 where ``y > threshold``. Default 0.5 — the same
        default and the same strict ``>`` as
        :meth:`as_int <silk.signals.Signal.as_int>`. Raise it (0.9 of a
        supply, say) when the digital levels are not rail-to-rail or the
        trace has ringing.
    name : str, optional
        Name for the returned Signal. Defaults to ``'code'``.

    Returns
    -------
    Signal
        ``xlabel`` = the bits' own x-axis label (``'time'`` for a
        transient capture), ``ylabel='code'``, ``type='count'``,
        ``x`` = the shared axis, ``y`` = the decimal code as floats.
        A NaN in any contributing bit makes that sample's code NaN rather
        than logical zero. Contaminated decoding emits one structured
        ``MetricWarning`` with the affected sample count (from ``as_int``
        for delegated binary decoding).
        Binary words are packed as exact integers, then converted once to
        float64. Beyond its 53 significant binary bits, loss of low-order
        integer bits depends on the code value. Delegated binary decoding
        emits at most one ``binary_code_precision_loss`` ``MetricWarning``
        naming ``as_int`` when available codes are rounded, with the bit width
        and rounded sample count. Exact conversions do not warn. Keep the bit
        Signals to compare whole words exactly; the returned codes are floats.
        Schema kind ``transient`` — **not** ``code_transitions``: the
        ``code_transitions`` Signal is what
        :func:`silk.metrics.converter.extract_code_transitions <silk.metrics.extract_code_transitions>` returns when
        given this Signal and the ramp that drove it.

    Raises
    ------
    OverflowError
        An available exact binary code cannot convert to finite float64.
        Delegated binary errors name ``as_int`` and the zero-based sample
        index and x coordinate. A very wide word with a small code is accepted.
    TypeError
        *bits* is not a bus Signal or a sequence of Signals, or
        *msb_first* was omitted for ``'binary'`` / supplied for
        ``'thermometer'``.
    ValueError
        *bits* is empty, an unknown *weighting* was asked for, or the
        bit traces do not have exactly equal coordinate shapes and values
        (``np.array_equal``). Equal copied axes are accepted, without requiring
        matching dtypes. No interpolation or snapping is performed. Use matching
        recorded coordinates or, when resampling is appropriate,
        ``silk.metrics.sample(other, x_new=reference.x)`` before decoding.

    Examples
    --------
    A 3-bit SAR, bits written out most-significant first::

        code = codes_from_bits([tb.probe('d2'), tb.probe('d1'),
                                tb.probe('d0')], msb_first=True)
        trans = extract_code_transitions(tb.probe('vin'), code)
        print(dnl(trans).max_abs_lsb)

    A 3-bit flash, seven comparator outputs, levels that are not
    rail-to-rail::

        therm = [tb.probe(f'vcmp{k}') for k in range(1, 8)]
        code = codes_from_bits(therm, weighting='thermometer',
                               threshold=0.9)
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(
            f"codes_from_bits: weighting must be one of {list(_WEIGHTINGS)}, "
            f"got {weighting!r}"
        )
    seq = _bit_signals(bits)

    if weighting == 'binary':
        if msb_first is None:
            raise TypeError(
                "codes_from_bits: msb_first is required for "
                "weighting='binary' — a sequence of bit Signals carries no "
                "ordering convention, and the wrong order returns a "
                "plausible wrong code rather than an error. Pass "
                "msb_first=True if bits[0] is the highest-order bit, "
                "msb_first=False if bits[0] is bit 0."
            )
    elif msb_first is not None:
        raise TypeError(
            "codes_from_bits: msb_first is meaningless for "
            "weighting='thermometer' — a thermometer code is the COUNT of "
            "asserted bits, which does not depend on their order. Drop the "
            "argument, or pass weighting='binary' if the bits are "
            "binary-weighted."
        )

    x = np.asarray(seq[0].x)
    for i, s in enumerate(seq[1:], start=1):
        xi = np.asarray(s.x)
        context = (
            f'codes_from_bits: every bit must share one x-axis; '
            f'bits[{i}] ({s.name!r}, {xi.size} samples, shape {xi.shape}) and '
            f'bits[0] ({seq[0].name!r}, {x.size} samples, shape {x.shape}); '
        )
        remedy = (
            'Use matching recorded coordinates or, when resampling is appropriate, '
            'silk.metrics.sample(other, x_new=reference.x).'
        )
        if xi.shape != x.shape:
            raise ValueError(context + 'shape/length mismatch. ' + remedy)
        if not np.array_equal(xi, x):
            index = int(np.flatnonzero(xi != x)[0])
            raise ValueError(
                context + f'different grid at first differing index {index}: '
                f'{xi.flat[index].item()!r} vs {x.flat[index].item()!r}. ' + remedy
            )

    out_name = name if name is not None else 'code'

    if weighting == 'binary' and len(seq) > 1:
        # Delegate to Signal.as_int so the binary pack has exactly ONE
        # implementation in silk. The bus handle is the argument shape
        # as_int takes; building one here is cheaper than duplicating
        # the shift-and-add and it keeps the threshold semantics
        # identical by construction. A ONE-bit "bus" cannot go through it
        # (the sum below handles a single bit directly), and a 1-bit binary word is the same count as a 1-bit
        # thermometer word, so it falls through to the sum below.
        handle = type(seq[0])._make_bus_handle(name=out_name, underlying=seq)
        decoded = handle.as_int(threshold=threshold, msb_first=msb_first)
        decoded.name = out_name
        return decoded

    counts = np.zeros(len(x), dtype=float)
    unavailable = np.zeros(len(x), dtype=bool)
    for s in seq:
        samples = np.asarray(s.y, dtype=float)
        unavailable |= np.isnan(samples)
        counts += (samples > threshold).astype(float)
    counts[unavailable] = np.nan
    if np.any(unavailable):
        from silk.metrics.diagnostics import emit
        emit(metric='codes_from_bits', code='nan_bits',
             message='NaN contributing bits make decoded samples unavailable.',
             details={'affected_sample_count': int(np.count_nonzero(unavailable))})
    return type(seq[0])(
        name=out_name,
        type='count',
        xlabel=seq[0].xlabel,
        ylabel='code',
        xvalues=np.asarray(x, dtype=float),
        yvalues=counts,
    )


def power_sweep(p_in_dbm: Iterable[float], p_out_dbm: Iterable[float], *,
                name: str = 'power_sweep') -> Signal:
    """Assemble the ``power_sweep`` Signal :func:`silk.metrics.rf.p1db <silk.metrics.p1db>` takes.

    A power sweep is a curve across RUNS, not a waveform from one run:
    the caller simulates the same stage at N input amplitudes, converts
    each run's input and output power to dBm, and hands both arrays
    here. That is why ``power_sweep``'s ``produced_by`` is empty and why
    ``p1db`` is in the catalog but never offerable to a consumer that
    can only author analyses.

    Parameters
    ----------
    p_in_dbm : array-like
        Input power per sweep point, in dBm, **sorted ascending**.
        ``p1db`` walks the curve from its first point and takes that
        point's gain as the small-signal reference, so an unsorted sweep
        measures compression from the wrong place.
    p_out_dbm : array-like
        Output power per sweep point, in dBm, same length and same
        ordering as *p_in_dbm*.
    name : str, optional
        Name for the returned Signal.

    Returns
    -------
    Signal
        ``xlabel='power'``, ``ylabel='p_out'``, ``type='power'``,
        ``x`` = *p_in_dbm*, ``y`` = *p_out_dbm* — schema kind
        ``power_sweep``.

    Raises
    ------
    ValueError
        The two arrays differ in length, either is not 1-D, or the sweep
        has fewer than two points. These are the same conditions
        ``p1db`` refuses on; refusing here names the assembly step
        instead of the metric, which is where the mistake was made.
        A sweep whose *p_in_dbm* is not ascending is refused too — it is
        the one malformed input ``p1db`` cannot detect, because a
        descending sweep produces a confident wrong compression point.

    Notes
    -----
    Converting a measured amplitude to dBm is the caller's job and
    deliberately not done here: it needs the reference impedance, and
    whether the measured quantity is an amplitude, an RMS value or a
    single FFT bin. For a sinusoid of amplitude ``A`` volts into
    ``R`` ohms, ``P_dBm = 10*log10(A**2 / (2*R) / 1e-3)``.
    :meth:`silk.signals.base.Signal.dbm <silk.signals.Signal.dbm>` does the same conversion
    sample-by-sample on a waveform.

    Examples
    --------
    ::

        p_in, p_out = [], []
        for a in amplitudes:
            res = run_one(a)
            p_in.append(dbm_of(a))
            p_out.append(dbm_of(amplitude_at(res, f0)))
        print(p1db(power_sweep(p_in, p_out)))
    """
    x = np.asarray(p_in_dbm, dtype=float)
    y = np.asarray(p_out_dbm, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f'power_sweep: p_in_dbm and p_out_dbm must be 1-D, got '
            f'{x.ndim}-D and {y.ndim}-D'
        )
    if len(x) != len(y):
        raise ValueError(
            f'power_sweep: p_in_dbm and p_out_dbm must have the same '
            f'length, got {len(x)} and {len(y)}'
        )
    if len(x) < 2:
        raise ValueError(
            f'power_sweep: need at least 2 sweep points, got {len(x)}'
        )
    finite = np.isfinite(x)
    if np.any(np.diff(x[finite]) <= 0):
        raise ValueError(
            'power_sweep: p_in_dbm must be sorted ascending (p1db reads its '
            "small-signal gain reference from the FIRST point and walks "
            'forward, so an out-of-order sweep reports a confident wrong '
            'compression point)'
        )
    return Signal(name, 'power', 'power', 'p_out', x, y)
