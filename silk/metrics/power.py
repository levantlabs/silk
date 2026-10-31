"""Power and energy from paired voltage and current time waveforms.

Average power integrates their product. Dynamic energy per cycle integrates
complete periods and divides by the retained count. Trapezoidal integration
includes an interpolated endpoint when a retained period ends between samples.
"""
from __future__ import annotations

from typing import Optional

import math

import numpy as np

from silk.utils._compat import trapezoid
from silk.signals.base import Signal
from silk.signals.schema import metric, Transient
from silk.metrics._guards import _axis
from silk.metrics.diagnostics import emit
from silk.metrics.waveform import _axis_context, _check_transient


#: ``Signal.type`` spellings that mean volts.
_VOLTAGE_TYPES = frozenset({'voltage', 'volt', 'volts', 'v'})


#: ``Signal.type`` spellings that mean amperes.
_CURRENT_TYPES = frozenset({'current', 'amp', 'amps', 'i', 'a'})


def _recognised_quantity(sig: Signal) -> Optional[str]:
    """``'voltage'``, ``'current'``, or ``None`` when the Signal's type is
    not one this vocabulary knows.

    Membership is EXACT (after ``strip().lower()``), never a prefix or a
    substring test, and that half is deliberate. A prefix test would read
    ``'voltage-density'`` — what a ``.noise`` Signal carries, in V/√Hz —
    and ``'voltage*current'`` — what ``v * i`` composes, in watts — as
    voltages, which neither is. Neither legitimately reaches a power
    metric; neither is refused here either. They fall outside both tables,
    so they are undecidable and pass, exactly like any other type this
    vocabulary has never seen.
    """
    t = getattr(sig, 'type', None)
    if not isinstance(t, str):
        return None
    t = t.strip().lower()
    if t in _VOLTAGE_TYPES:
        return 'voltage'
    if t in _CURRENT_TYPES:
        return 'current'
    return None


def _check_v_i(sig_v: Signal, sig_i: Signal, metric: str) -> None:
    """Both inputs transient, on one time axis, and not the wrong quantity.

    Grids agree within 1e-3 of the smallest spacing; singleton axes agree exactly.

    The last of those is ``average_power(vout, vin)`` used to
    return V² labelled W in silence, because nothing here read
    ``Signal.type``. A RECOGNISED wrong type is a malformed request, not
    degenerate data — the caller-error convention — so it raises
    ``ValueError`` naming the metric, the slot, the type found and the
    type wanted. An UNRECOGNISED type passes: see
    :func:`_recognised_quantity`.
    """
    _check_transient(sig_v, metric)
    _check_transient(sig_i, metric)
    if not _axis(sig_v, metric).same_grid(_axis(sig_i, metric)):
        raise ValueError(
            f'{metric}: sig_v and sig_i must share the same time axis; '
            f'sig_v: {_axis_context(sig_v)}; sig_i: {_axis_context(sig_i)}'
        )
    for sig, slot, want in ((sig_v, 'sig_v', 'voltage'),
                            (sig_i, 'sig_i', 'current')):
        found = _recognised_quantity(sig)
        if found is not None and found != want:
            raise ValueError(
                f"{metric}: {slot} must be a {want} Signal, but Signal "
                f"{sig.name!r} has type={sig.type!r}, a recognised {found} "
                f"type; {metric} takes the voltage as sig_v and the current "
                f"as sig_i"
            )


@metric(domain='tran', unit='W')
def average_power(sig_v: Transient, sig_i: Transient) -> float:
    """Time-averaged instantaneous power, ⟨v(t)·i(t)⟩ over the trace.

    Paired input grids agree within 1e-3 of their smallest spacing;
    singleton coordinates must match exactly. Mismatches raise ``ValueError``.

    Parameters
    ----------
    sig_v : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time". Values represent volts; a
        recognised current type is refused.
    sig_i : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time". Values represent amperes; a
        recognised voltage type is refused.

    Returns
    -------
    float
        Signed average power in watts for volt and ampere inputs. NaN products
        propagate to nan with MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    ValueError is raised for non-transient, empty or single-sample inputs,
    non-increasing axes, mismatched grids, or a recognised quantity in the
    wrong voltage/current slot. Unrecognised Signal types are accepted.
    """
    _check_v_i(sig_v, sig_i, 'average_power')
    t = sig_v.x
    if len(t) < 2:
        raise ValueError(
            f'average_power: need at least 2 samples, got {len(t)}; '
            f'{_axis_context(sig_v)}'
        )
    T = float(t[-1] - t[0])
    if T <= 0:
        raise ValueError(
            f'average_power: time array must be strictly increasing; '
            f'got duration={T:g}; {_axis_context(sig_v)}'
        )
    power = sig_v.y * sig_i.y
    consumed_t = t
    if len(power) > 1 and np.any(np.isnan(power)):
        emit(
            metric='average_power', code='nan_power',
            message='Integrated voltage-current products contain NaNs.',
            details={
                'voltage_name': sig_v.name, 'current_name': sig_i.name,
                'sample_count': len(power),
                'nan_sample_count': int(np.count_nonzero(np.isnan(power))),
                'time_start_s': float(consumed_t[0]),
                'time_stop_s': float(consumed_t[-1]),
                'voltage_unit': sig_v.ylabel, 'current_unit': sig_i.ylabel,
                'power_unit': 'W',
            },
            suggestion='Check voltage and current source results in the integrated interval.',
        )
    return float(trapezoid(power, t)) / T


@metric(domain='tran', unit='J', param_units={'period': 's'})
def dynamic_energy_per_cycle(sig_v: Transient, sig_i: Transient,
                             period: float) -> float:
    """Energy per clock cycle = (∫v·i dt over whole cycles) / cycle_count.

    Paired input grids agree within 1e-3 of their smallest spacing;
    singleton coordinates must match exactly. Mismatches raise ``ValueError``.

    Selects ``floor(T / period)`` complete periods starting at ``t[0]``
    for integration and discards whatever partial-cycle tail is
    left over: a trace of 2.4 periods measures its 2 complete cycles,
    not 2.4 of them (the previous ``round()``-based count
    mismatched the integration window and the divisor, which read up to
    +50 % high on a switching waveform and, via ``max(1.0, ...)``,
    silently under-reported energy on a sub-period trace). If the
    full-cycle endpoint lies between samples, the integral includes a
    linearly interpolated power (v·i) sample at that endpoint. On an exact whole number of periods it
    equals ``average_power(sig_v, sig_i) * period`` to within
    floating-point rounding (a few ULP, not bit-for-bit: this divides the
    energy by the cycle count, ``average_power`` by the measured duration
    ``t[-1] - t[0]``, which is only approximately ``n * period``).

    Raises ``ValueError`` if the trace spans less than one full period:
    a per-cycle quantity cannot be measured from less than a cycle.

    Parameters
    ----------
    sig_v : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time". Values represent volts; a
        recognised current type is refused.
    sig_i : Signal
        Transient waveform with matching one-dimensional x/y arrays, finite strictly
        increasing time in seconds and xlabel="time". Values represent amperes; a
        recognised voltage type is refused.
    period : float
        Supplied positive period in seconds.

    Returns
    -------
    float
        Signed energy per complete period in joules for volt and ampere inputs.
        Consumed NaN products, including those needed to interpolate the
        endpoint power, propagate with MetricWarning.

    Notes
    -----
    The input Signal objects and their arrays are not modified.
    ValueError is raised for non-transient, empty or single-sample inputs,
    non-increasing axes, mismatched grids, or a recognised quantity in the
    wrong voltage/current slot. Unrecognised Signal types are accepted.

    Examples
    --------
    Even a sparse grid integrates constant power to the exact cycle end:

    >>> from silk.signals import Signal
    >>> v = Signal('v', 'voltage', 'time', 'V', [0., 2.], [1., 1.])
    >>> i = Signal('i', 'current', 'time', 'A', [0., 2.], [1., 1.])
    >>> dynamic_energy_per_cycle(v, i, period=1.5)
    1.5
    """
    if period <= 0:
        raise ValueError(
            f'dynamic_energy_per_cycle: period must be > 0, got {period}'
        )
    _check_v_i(sig_v, sig_i, 'dynamic_energy_per_cycle')
    t = sig_v.x
    if len(t) < 2:
        raise ValueError(
            f'dynamic_energy_per_cycle: need at least 2 samples, got {len(t)}; '
            f'{_axis_context(sig_v)}'
        )
    T = float(t[-1] - t[0])
    eps = 1e-9  # guards float error on an exact multiple of period
    n_cycles = math.floor(T / period + eps)
    if n_cycles < 1:
        raise ValueError(
            f'dynamic_energy_per_cycle: trace spans {T:.6g} s, shorter '
            f'than one period ({period:.6g} s) — {T / period:.3g} cycles '
            f'found, need >= 1'
        )
    t_end = t[0] + n_cycles * period
    mask = t <= t_end
    all_power = sig_v.y * sig_i.y
    power = all_power[mask]
    consumed_t = t[mask]
    if consumed_t[-1] < t_end < t[-1]:
        power = np.append(power, np.interp(t_end, t, all_power))
        consumed_t = np.append(consumed_t, t_end)
    if len(power) > 1 and np.any(np.isnan(power)):
        emit(
            metric='dynamic_energy_per_cycle', code='nan_power',
            message='Integrated voltage-current products contain NaNs.',
            details={
                'voltage_name': sig_v.name, 'current_name': sig_i.name,
                'sample_count': len(power),
                'nan_sample_count': int(np.count_nonzero(np.isnan(power))),
                'time_start_s': float(consumed_t[0]),
                'time_stop_s': float(consumed_t[-1]),
                'voltage_unit': sig_v.ylabel, 'current_unit': sig_i.ylabel,
                'power_unit': 'W', 'period_s': float(period),
                'complete_cycle_count': n_cycles,
            },
            suggestion='Check voltage and current source results in the integrated interval.',
        )
    energy = float(trapezoid(power, consumed_t))
    return energy / n_cycles
