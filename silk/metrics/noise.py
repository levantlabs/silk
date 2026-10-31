"""Noise figure, integrated and input-referred noise, density and noise formulas.

Noise-density signals provide frequency queries, integrated noise and the 1/f
corner. Closed-form helpers cover noise factors, cascades, uncorrelated sources
and kT/C. Each function documents its accepted units and failure conditions.
"""
from __future__ import annotations

from math import sqrt

import numpy as np

from silk.utils._compat import trapezoid
from silk.signals.base import Signal
from silk.signals.schema import metric, NoiseDensity
from silk.signals.kinds import NOISE_DENSITY, _check_kind
from silk.metrics._guards import (  # noqa: F401
    _check_nonempty, _check_freq, _any_nan, _bracket_freq,
)
from silk.metrics._guards import _axis
from silk.metrics.diagnostics import emit


# Boltzmann's constant in J/K.
K_BOLTZMANN = 1.380649e-23


def _density_details(f, y, unit):
    """Summarize consumed densities without discarding their sign or units."""
    missing = np.isnan(y)
    known = y[~missing]
    return {
        'frequency_min_hz': float(np.min(f)),
        'frequency_max_hz': float(np.max(f)),
        'sample_count': int(y.size),
        'nan_sample_count': int(np.count_nonzero(missing)),
        'nan_frequency_min_hz': float(np.min(f[missing])) if missing.any() else None,
        'nan_frequency_max_hz': float(np.max(f[missing])) if missing.any() else None,
        'density_min': float(np.min(known)) if known.size else None,
        'density_max': float(np.max(known)) if known.size else None,
        'density_unit': unit,
    }


# ---------------------------------------------------------------------------
# Noise figure
# ---------------------------------------------------------------------------

def noise_figure(snr_in_db: float, snr_out_db: float) -> float:
    """Noise figure from input minus output signal-to-noise ratio.

    Parameters
    ----------
    snr_in_db, snr_out_db : float
        Input and output SNR, respectively, in dB.

    Returns
    -------
    float
        Noise figure in dB. Equal SNRs give zero. NaN propagates and
        infinities follow subtraction; no validation or ``MetricWarning``
        is emitted.

    Examples
    --------
    >>> from silk.metrics import noise_figure
    >>> noise_figure(40.0, 35.0)
    5.0
    """
    return float(snr_in_db - snr_out_db)


def noise_figure_from_factor(f: float) -> float:
    """Convert linear noise factor to noise figure, ``10*log10(f)``.

    Parameters
    ----------
    f : float
        Dimensionless noise factor; must be positive.

    Returns
    -------
    float
        Noise figure in dB. Unity factor gives zero; NaN gives ``nan`` and
        positive infinity gives ``inf``, without a ``MetricWarning``.

    Raises
    ------
    ValueError
        If ``f <= 0``, naming the invalid noise factor.
    """
    if f <= 0:
        raise ValueError(f'noise_figure_from_factor: noise factor must be > 0, got {f}')
    return float(10.0 * np.log10(f))


def input_referred_noise(output_noise_v: float, gain_vv: float) -> float:
    """Refer an output RMS noise total to the input using one flat gain.

    Parameters
    ----------
    output_noise_v : float
        Integrated RMS output noise in V.
    gain_vv : float
        Voltage gain in V/V; its absolute value is used and it must be nonzero.

    Returns
    -------
    float
        RMS input noise in V, ``output_noise_v / abs(gain_vv)``. Zero noise
        returns zero; negative noise is not rejected. NaN operands propagate
        ``nan`` and infinite operands follow arithmetic. No ``MetricWarning``
        is emitted.

    Raises
    ------
    ValueError
        If ``gain_vv == 0``, naming the gain and output noise.

    Notes
    -----
    This is exact only for flat gain across the integrated band. For varying
    gain, pass the input-referred amplitude density from
    ``load_signal('noise_in')`` to :func:`integrated_noise` instead.
    """
    if gain_vv == 0:
        raise ValueError(
            f'input_referred_noise: gain_vv must be non-zero, got {gain_vv} V/V '
            f'for output_noise_v={output_noise_v} V'
        )
    return float(output_noise_v / abs(gain_vv))


def sum_uncorrelated(*sigmas: float) -> float:
    """Combine independent RMS noise contributions by root sum of squares.

    Parameters
    ----------
    *sigmas : float
        RMS contributions, all in the same unit (for example V or A).
        Signs are discarded by squaring.

    Returns
    -------
    float
        Combined RMS noise in the input unit. No contributions returns
        ``0.0``. NaN propagates; an infinite contribution gives ``inf``
        unless another is NaN. No ``MetricWarning`` or physical validation
        is emitted.

    Raises
    ------
    OverflowError
        If squaring a large finite Python float overflows.

    Examples
    --------
    >>> from silk.metrics import sum_uncorrelated
    >>> sum_uncorrelated(3.0, 4.0)
    5.0
    """
    if len(sigmas) == 0:
        return 0.0
    return float(sqrt(sum(float(s) ** 2 for s in sigmas)))


def kT_over_C(c_farads: float, temp_k: float = 300.0) -> float:
    """RMS sampled capacitor noise, ``sqrt(k*T/C)``.

    Parameters
    ----------
    c_farads : float
        Capacitance in F; must be positive.
    temp_k : float, optional
        Absolute temperature in K; default 300, must be positive.

    Returns
    -------
    float
        RMS noise in V (about 64 microvolts for 1 pF at 300 K). NaN inputs
        propagate ``nan``; infinite capacitance with finite temperature
        gives zero. No ``MetricWarning`` is emitted.

    Raises
    ------
    ValueError
        If capacitance or temperature is nonpositive, naming its value and
        unit. Capacitance is checked first.
    """
    if c_farads <= 0:
        raise ValueError(f'kT_over_C: capacitance must be > 0, got {c_farads} F')
    if temp_k <= 0:
        raise ValueError(f'kT_over_C: temperature must be > 0 K, got {temp_k} K')
    return float(sqrt(K_BOLTZMANN * temp_k / c_farads))




# ---------------------------------------------------------------------------
# Flicker-noise corner
# ---------------------------------------------------------------------------


@metric(domain='noise', unit='Hz')
def one_over_f_corner(noise_sig: NoiseDensity) -> float:
    """Frequency at which 1/f noise equals the white-noise floor.

    Standard EE convention: at the corner, the 1/f and white
    contributions are equal so the total noise sits ``√2`` above the
    white floor (3 dB above in V²/Hz terms). The white floor is
    estimated as the median of the upper half of the spectrum on a
    log-frequency scale.

    Parameters
    ----------
    noise_sig : Signal
        Noise spectrum: ``xlabel='frequency'``, ``y`` = noise spectral
        density in V/√Hz (the standard convention), with matching 1-D
        arrays; no samples are modified. For V²/Hz inputs,
        take the square root first (``np.sqrt(psd)``): the √2-above-floor
        threshold applies to amplitude density, not power density.

    Returns
    -------
    float
        Corner frequency in Hz. ``nan`` if the white floor can't be
        identified, or if y never crosses below ``√2 · floor`` (e.g.
        the data is dominated by 1/f throughout). An EMPTY spectrum
        raises instead: there is no floor to fail to find.
        Also ``nan`` if ``y`` contains a NaN sample: ``nan <
        threshold`` is ``False``, so a NaN sitting on the knee was
        skipped and the corner read one grid point high — a finite
        wrong number — while a NaN in the upper half made the median
        floor NaN and happened to give ``nan`` by accident. Now uniform.
        Consumed NaNs and a nonpositive estimated white floor emit a
        ``MetricWarning`` with density units and the affected range.

    Raises
    ------
    ValueError
        If ``noise_sig``'s observables RULE OUT a noise density: anything off the frequency axis, and — new — a
        COMPLEX ``y``, which is an ``.ac`` response and nothing else. A
        complex Signal used to be cast to real behind a numpy
        ``ComplexWarning`` and answered a confident corner frequency
        read off Re ``H`` (measured 47.3 kHz on a single-pole sweep).
        A REAL ``y`` on the frequency axis is accepted whatever it
        really is — a phase-noise sideband is observationally identical
        to a density (the contract in docs/known_issues.md), so ``one_over_f_corner(sideband)``
        still returns ``nan`` rather than refusing; closing that needs
        a producer-stamped kind.
        Also if ``noise_sig`` has no samples, mismatched/non-1-D arrays,
        or a nonfinite or nonmonotonic frequency axis. Either strictly
        increasing or strictly decreasing frequency order is accepted.

    Notes
    -----
    Resolution: the first GRID point below the threshold (one grid step,
    upward-snapped), and the floor is the median of the upper half of
    the log grid — which still carries some 1/f, so the corner reads
    LOW by ``f_c / (2(1+ε)² − 1)`` with ``ε = √(1 + f_c/f_med) − 1``
    (4.5 % for a 1 kHz corner on a 1 Hz–1 MHz sweep).

    Lower-edge rule: a corner is reported only if the lowest swept
    frequency is still ABOVE ``√2 · floor`` (a 1/f region was observed)
    and a later sample falls below it. A corner just above the lowest
    point therefore returns a finite frequency (the first grid point
    below the threshold, never the lowest sweep frequency itself). If the
    lowest sample is already below the threshold — a white-only
    spectrum, or a corner at or below the sweep — no 1/f region was
    observed: ``nan`` with a ``MetricWarning``. Fewer than four samples
    also returns ``nan`` with a ``MetricWarning``.
    """
    # UNDECIDABLE is a deliberate pass, not a verified input (D6) — see
    # the Raises section.
    _check_kind(noise_sig, NOISE_DENSITY, 'one_over_f_corner', 'noise_sig')
    _check_nonempty(noise_sig, 'one_over_f_corner')
    axis = _axis(noise_sig, 'one_over_f_corner')
    f = axis.increasing
    y = axis.orient(np.asarray(noise_sig.y, dtype=float))
    if _any_nan(y):
        emit(
            metric='one_over_f_corner', code='nan_density',
            message='Consumed noise density contains NaN; the 1/f corner '
                    'is unavailable.',
            details=_density_details(f, y, noise_sig.ylabel),
            suggestion='Inspect the noise result at the affected frequencies '
                       'and obtain finite density samples before finding a corner.',
        )
        return float('nan')
    if len(f) < 4:
        emit(
            metric='one_over_f_corner', code='too_few_samples',
            message='The 1/f corner is unavailable: fewer than four samples '
                    f'({len(f)}) cannot separate a 1/f region from the white floor.',
            details=_density_details(f, y, noise_sig.ylabel),
            suggestion='Sweep more frequencies across the expected corner.',
        )
        return float('nan')
    if float(f[0]) <= 0:
        f_lo_pos = f[f > 0]
        if len(f_lo_pos) == 0:
            return float('nan')
        f_min_log = float(f_lo_pos[0])
    else:
        f_min_log = float(f[0])
    f_geom_mid = np.sqrt(f_min_log * float(f[-1]))
    upper_mask = f >= f_geom_mid
    if np.sum(upper_mask) < 2:
        upper_mask = f >= f[len(f) // 2]
    white_floor = float(np.median(y[upper_mask]))
    if white_floor <= 0:
        emit(
            metric='one_over_f_corner', code='nonpositive_white_floor',
            message='Estimated median white-noise floor is nonpositive; '
                    'the 1/f corner is unavailable.',
            details={
                **_density_details(f[upper_mask], y[upper_mask], noise_sig.ylabel),
                'available_sample_count': int(f.size),
                'white_floor': white_floor,
            },
            suggestion='Check the density scale and units and measure a '
                       'positive white-noise floor in the upper sweep region.',
        )
        return float('nan')
    threshold = np.sqrt(2.0) * white_floor
    below = np.where(y < threshold)[0]
    if len(below) == 0:
        return float('nan')
    if below[0] == 0:
        emit(
            metric='one_over_f_corner', code='no_one_over_f_region',
            message='No 1/f region was observed within the sweep: the density '
                    'at the lowest swept frequency is already within sqrt(2) of '
                    'the white floor.',
            details={**_density_details(f, y, noise_sig.ylabel),
                     'white_floor': white_floor},
            suggestion='Extend the sweep to lower frequencies, or the block has '
                       'no 1/f noise in this range.',
        )
        return float('nan')
    return float(f[below[0]])


# ---------------------------------------------------------------------------
# Cascaded noise figure
# ---------------------------------------------------------------------------

def cascade_noise_figure(stage_nfs_db, stage_gains_db) -> float:
    """Total noise figure of a cascade via the Friis formula.

    F_total = F_1 + (F_2 − 1)/G_1 + (F_3 − 1)/(G_1·G_2) + …

    where F_i = 10^(NF_i/10) and G_i = 10^(Gain_i/10) (power gain).

    Standard receiver-chain analysis: a low-noise first stage with
    high gain dominates the cascade NF; later stages contribute
    proportionally less.

    Parameters
    ----------
    stage_nfs_db : array-like
        Per-stage noise figures, dB; each stage must have NF >= 0 dB.
    stage_gains_db : array-like
        Per-stage power gains, dB. Same length as ``stage_nfs_db``.
        Negative gains (attenuators) and zero dB are allowed. The last
        stage's gain is ignored (only used for downstream contribution).

    Returns
    -------
    float
        Cascaded NF in dB. NaNs in consumed noise figures or gains propagate
        ``nan`` without a ``MetricWarning``; the last gain is not consumed.
        Extreme gains/factors may overflow or underflow with NumPy runtime
        warnings, yielding ``inf`` or ``nan``. Input arrays are not modified.

    Raises
    ------
    ValueError
        If the arrays are empty, not one-dimensional or differ in shape,
        or any stage has a negative noise figure. Stage numbers in error
        messages are one-based. The computed cascade factor is not used
        to classify caller errors.
    """
    nfs = np.asarray(stage_nfs_db, dtype=float)
    gains = np.asarray(stage_gains_db, dtype=float)
    if nfs.shape != gains.shape:
        raise ValueError(
            f'cascade_noise_figure: stage_nfs_db and stage_gains_db '
            f'must have the same shape, '
            f'got {nfs.shape} and {gains.shape}'
        )
    if nfs.ndim != 1:
        raise ValueError(
            f'cascade_noise_figure: stage_nfs_db and stage_gains_db must '
            f'be one-dimensional, got shape {nfs.shape}'
        )
    if len(nfs) < 1:
        raise ValueError('cascade_noise_figure: need at least 1 stage, got 0')
    bad_nfs = np.flatnonzero(nfs < 0)
    if bad_nfs.size:
        i0 = int(bad_nfs[0])
        raise ValueError(
            f'cascade_noise_figure: stage {i0 + 1} of {len(nfs)} requires '
            f'NF >= 0 dB, got {nfs[i0]:g} dB '
            f'(stage gain {gains[i0]:g} dB)'
        )
    fs = 10.0 ** (nfs / 10.0)
    gs = 10.0 ** (gains / 10.0)
    f_total = float(fs[0])
    cum_g = 1.0
    for i in range(1, len(fs)):
        cum_g *= gs[i - 1]
        f_total += (fs[i] - 1.0) / cum_g
    return float(10.0 * np.log10(f_total))


# ---------------------------------------------------------------------------
# Band-integrated noise
# ---------------------------------------------------------------------------

@metric(domain='noise', unit='[y]*sqrt([x])', param_units={'f1': 'Hz', 'f2': 'Hz'})
def integrated_noise(sig: NoiseDensity, f1: float, f2: float) -> float:
    """RMS integrated noise between f1 and f2 Hz.

    Parameters
    ----------
    sig : Signal
        Noise amplitude density, with matching 1-D arrays: finite, strictly
        monotonic frequency ``x`` in Hz and real ``y`` in V/sqrt(Hz) or
        A/sqrt(Hz). At least two samples are required; either sweep direction
        is accepted. Input samples are not modified.
    f1, f2 : float
        Integration band in Hz.

    Returns
    -------
    float
        RMS integrated noise over the band, in the input density's own
        unit times sqrt(Hz) — V for a voltage-noise probe, A for a
        current-noise one (the exported unit is ``'[y]*sqrt([x])'``,
        not a hard-coded ``'V'``, for the same reason Silk fixed
        ``error_norm`` and declared ``noise_density_at_freq``'s
        unit relative to the input).
        NaN density samples inside the selected band emit a ``MetricWarning``
        and propagate ``nan``. Samples outside the band are not inspected;
        valid zero integrated noise remains zero and quiet. Diagnostic
        density units come from ``sig.ylabel`` (also for current noise).

    Raises
    ------
    ValueError
        If ``sig``'s observables RULE OUT a noise density:
        anything off the frequency axis, and — new — a COMPLEX ``y``,
        which is an ``.ac`` response and nothing else. ``sqrt(int |H|^2
        df)`` over a transfer function is not an RMS voltage; it used
        to come back as a confident number in V (measured 1.25e+04 on a
        100 V/V single pole — the contract in docs/known_issues.md). A REAL ``y`` on the
        frequency axis is accepted whatever it really is: a phase-noise
        sideband in dBc/Hz is observationally identical to a density
        (the contract in docs/known_issues.md), so ``integrated_noise(sideband, 10, 1e6)`` still
        returns its nonsense volts (measured 1.4e+05); closing that
        needs a producer-stamped kind.

        Also if ``[f1, f2]`` holds fewer than 2 sweep points — a band
        aggregate over an empty (or near-empty) band is a malformed
        request, not a measurement. This is deliberately
        different from a point query (e.g. ``value_at``) outside the
        data, which returns ``nan``. Mismatched/non-1-D arrays and
        nonfinite/nonmonotonic frequency axes also raise.

    Notes
    -----
    Resolution: the band edges snap INWARD to the grid points inside
    ``[f1, f2]`` (no interpolation to the exact edges — up to one grid
    step lost at each end); the trapezoid on a log grid over-reads a
    1/f PSD by ``½(r − 1/r)/ln r − 1`` in power (0.9 % at 10 pts/decade,
    0.45 % in V_rms) and is exact for a white PSD.

    Examples
    --------
    >>> import numpy as np
    >>> from silk import Signal
    >>> from silk.metrics import integrated_noise
    >>> density = Signal('n', 'noise', 'frequency', 'V/sqrt(Hz)',
    ...                  np.array([1.0, 2.0, 3.0]), np.full(3, 2e-9))
    >>> bool(np.isclose(integrated_noise(density, 1.0, 3.0), 2e-9*np.sqrt(2)))
    True
    """
    # UNDECIDABLE is a deliberate pass, not a verified input (D6) — see
    # the Raises section. _check_freq still runs after it: the kind check
    # subsumes its axis test but not its emptiness / >=2-points tests, and
    # re-running the axis comparison is cheaper than splitting the guard.
    _check_kind(sig, NOISE_DENSITY, 'integrated_noise', 'sig')
    _check_freq(sig, 'integrated_noise')
    mask = (sig.x >= f1) & (sig.x <= f2)
    axis = _axis(sig, 'integrated_noise')
    x = axis.orient(sig.x[mask])
    y = axis.orient(sig.y[mask])
    if len(x) < 2:
        lo, hi = float(np.min(sig.x)), float(np.max(sig.x))
        raise ValueError(
            f'integrated_noise: band [{f1:g}, {f2:g}] Hz contains {len(x)} '
            f'of {len(sig.x)} sweep points (sweep spans {lo:g} Hz .. '
            f'{hi:g} Hz)'
        )
    if _any_nan(y):
        emit(
            metric='integrated_noise', code='nan_density',
            message='Consumed integration-band density contains NaN; '
                    'RMS integrated noise is unavailable.',
            details={
                **_density_details(x, y, sig.ylabel),
                'band_min_hz': float(f1),
                'band_max_hz': float(f2),
            },
            suggestion='Inspect the noise result at the affected frequencies '
                       'and obtain finite density samples in the requested band.',
        )
        return float('nan')
    psd = np.abs(y) ** 2   # power spectral density
    return float(np.sqrt(trapezoid(psd, x)))


# ---------------------------------------------------------------------------
# Noise density at a frequency (TASK-170)
# ---------------------------------------------------------------------------

@metric(domain='noise', unit='[y]', param_units={'freq': 'Hz'})
def noise_density_at_freq(sig: NoiseDensity, freq: float,
                          *, interpolate: bool = True) -> float:
    """Noise spectral density at *freq*, interpolated between sweep points.

    Parameters
    ----------
    sig : Signal
        Noise amplitude density with matching 1-D arrays: a finite,
        strictly monotonic frequency axis in Hz and real ``y`` in V/sqrt(Hz)
        or A/sqrt(Hz). At least two samples are required, in either sweep
        direction. The input is not modified. Returned density keeps the
        input's y unit, including when interpolated between samples.
    freq : float
        Target frequency in Hz.
    interpolate : bool
        ``True`` (default): interpolate log-log between the two
        neighbouring sweep points, which recovers a power law
        (``S ∝ f^k``, including 1/f and white) exactly. Falls back to
        linear interpolation in density if either neighbour is
        non-positive. ``False`` returns the nearest sample, as before. A
        request exactly on a grid point returns that sample in both modes.

    Returns
    -------
    float
        The density at *freq* (the nearest sample if
        ``interpolate=False``). Strictly outside the supplied
        frequency range, returns ``nan`` with a ``MetricWarning`` naming
        the requested frequency and available range. A NaN in a sample the result is read from
        (the nearest one, or either neighbour when interpolating) also
        emits and returns ``nan``. Warning filters apply normally.

    Raises
    ------
    ValueError
        If ``sig``'s observables RULE OUT a noise density:
        anything off the frequency axis, and a COMPLEX ``y``, which is an
        ``.ac`` response and nothing else. A REAL ``y`` on the frequency
        axis is accepted whatever it really is: a phase-noise sideband in
        dBc/Hz is observationally identical to a density (the contract in docs/known_issues.md);
        closing that needs a producer-stamped kind (not currently available; see docs/known_issues.md).

        Also if ``freq`` is not finite (a NaN or infinite
        request is malformed, not a measurement outside coverage) or if
        ``sig`` has fewer than two samples, mismatched/non-1-D arrays,
        or a nonfinite/nonmonotonic frequency axis.

    Notes
    -----
    Resolution: with ``interpolate=True`` the error is that of a log-log
    straight line between grid points (zero on a power law).
    ``interpolate=False`` selects the sweep point with the smallest absolute
    distance from the requested frequency in Hz.
    """
    # UNDECIDABLE is a deliberate pass, not a verified input (D6) — see
    # the Raises section.
    _check_kind(sig, NOISE_DENSITY, 'noise_density_at_freq', 'sig')
    _check_freq(sig, 'noise_density_at_freq')
    if not np.isfinite(freq):
        raise ValueError(
            f'noise_density_at_freq: freq must be finite, got {freq!r} Hz'
        )
    axis = _axis(sig, 'noise_density_at_freq')
    lo, hi = axis.lower, axis.upper
    if not axis.contains(freq):
        emit(
            metric='noise_density_at_freq', code='frequency_outside_sweep',
            message='Requested frequency is outside the supplied sweep; '
                    'noise density is unavailable.',
            details={
                'requested_frequency_hz': float(freq),
                'frequency_min_hz': lo, 'frequency_max_hz': hi,
                'sample_count': len(sig.x),
                'density_unit': sig.ylabel,
            },
            suggestion='Choose a frequency within the sweep or extend '
                       'the sweep to cover the requested point.',
        )
        return float('nan')
    if interpolate:
        i0, i1, t = _bracket_freq(sig.x, freq)
        idx = i0 if t < 0.5 else i1
        pair = [i0] if i0 == i1 else [i0, i1]
    else:
        idx = int(np.argmin(np.abs(sig.x - freq)))
        pair = [idx]
    bad = [j for j in pair if np.isnan(sig.y[j])]
    if bad:
        emit(
            metric='noise_density_at_freq', code='nan_density',
            message='Measurement unavailable: a selected neighbouring '
                    'sample density is NaN.',
            details={
                **_density_details(sig.x[bad], sig.y[bad], sig.ylabel),
                'requested_frequency_hz': float(freq),
            },
            suggestion='Check the noise result at the selected frequency.',
        )
    if interpolate and i0 != i1:
        y0, y1 = float(sig.y[i0]), float(sig.y[i1])
        if y0 > 0 and y1 > 0:
            return float(np.exp(np.log(y0) + t * (np.log(y1) - np.log(y0))))
        return float(y0 + t * (y1 - y0))
    return float(sig.y[idx])


# ---------------------------------------------------------------------------
# Noise figure from a noise run (TASK-414)
# ---------------------------------------------------------------------------

#: The IEEE standard reference temperature for the noise factor, in K.
T0_KELVIN = 290.0


def _check_nf_conditions(metric_name, rs, temp_sim_k, temp_ref_k):
    """Rs and both temperatures must be finite and > 0 (caller error)."""
    for label, value, unit in (('rs', rs, 'ohm'), ('temp_sim_k', temp_sim_k, 'K'),
                               ('temp_ref_k', temp_ref_k, 'K')):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f'{metric_name}: {label} must be finite and > 0, '
                f'got {value!r} {unit}'
            )


def _check_nf_units(sig, metric_name, slot):
    """Refuse a density labelled in V^2/Hz (or A^2/Hz): the NF trio takes
    amplitude density (V/sqrt(Hz)) and squares it itself."""
    label = str(getattr(sig, 'ylabel', '') or '').replace(' ', '').lower()
    if '^2' in label or '**2' in label or '\u00b2' in label or 'v2/hz' in label:
        raise ValueError(
            f'{metric_name}: {slot} is labelled {sig.ylabel!r}, a power '
            f'spectral density; this metric takes an amplitude density in '
            f'V/sqrt(Hz) and squares it itself. Pass sqrt of the V^2/Hz '
            f'values with ylabel V/sqrt(Hz).'
        )


def _factor_from_density(s_in, rs, temp_sim_k, temp_ref_k):
    """Spot noise factor from an input-referred PSD (V^2/Hz)."""
    added = s_in - 4.0 * K_BOLTZMANN * temp_sim_k * rs
    return 1.0 + added / (4.0 * K_BOLTZMANN * temp_ref_k * rs)


def _emit_nonpositive_factor(metric_name, factor, rs, temp_sim_k, temp_ref_k,
                             extra):
    emit(
        metric=metric_name, code='nonpositive_noise_factor',
        message='Computed noise factor is not positive; the noise figure '
                'is unavailable.',
        details={'noise_factor': float(factor), 'rs_ohm': float(rs),
                 'temp_sim_k': float(temp_sim_k),
                 'temp_ref_k': float(temp_ref_k), **extra},
        suggestion='The input-referred density is below the thermal noise '
                   'of rs at temp_sim_k: check that rs is the source '
                   'resistance in series with the noise source and that '
                   'temp_sim_k is the temperature the run used.',
    )


@metric(domain='noise', unit='dB',
        param_units={'freq': 'Hz', 'rs': 'ohm', 'temp_sim_k': 'K',
                     'temp_ref_k': 'K'})
def noise_figure_at_freq(noise_in: NoiseDensity, freq: float, rs: float,
                         temp_sim_k: float, *, temp_ref_k: float = T0_KELVIN,
                         interpolate: bool = True) -> float:
    """Spot noise figure (dB) at *freq* from the input-referred noise density.

    The noise factor references the source resistance at the standard
    temperature ``T0`` (``temp_ref_k``, 290 K). The input-referred
    density of a noise run (``load_signal('noise_in')``, the output noise
    divided by the gain from the named input source) already contains the
    thermal noise of the source resistance, but at the SIMULATION
    temperature. So::

        S_in(f) = noise_in(f)^2                       [V^2/Hz]
        F(f)    = 1 + (S_in(f) - 4 k T_sim Rs) / (4 k T0 Rs)
        NF(f)   = 10 log10 F(f)

    which reduces to ``S_in / (4 k T0 Rs)`` only when ``T_sim = T0``.
    ngspice simulates at 27 C (300.15 K) by default; pass the temperature
    the run actually used — there is deliberately no default for it.

    Parameters
    ----------
    noise_in : Signal
        Input-referred noise density in V/sqrt(Hz) from a ``.noise`` run
        whose input source drives the circuit through ``rs``. Matching
        1-D arrays of at least two samples are required; the frequency axis
        in Hz must be finite and strictly increasing or decreasing. The
        input is not modified.
    freq : float
        Spot frequency in Hz.
    rs : float
        Source resistance in ohm: the resistance in series with the noise
        analysis's input source, whose thermal noise is the reference.
    temp_sim_k : float
        Temperature the simulation ran at, in K (no default).
    temp_ref_k : float
        Reference temperature ``T0`` in K. Default 290 K, the IEEE
        standard; pass ``temp_sim_k`` to get the factor referred to the
        simulation's own source noise.
    interpolate : bool
        Frequency handling identical to :func:`noise_density_at_freq`:
        ``True`` (default) interpolates the density log-log between the
        two neighbouring sweep points; ``False`` reads the nearest sample.

    Returns
    -------
    float
        Spot NF in dB. A noiseless amplifier returns 0 dB at ANY
        ``temp_sim_k``, provided ``temp_sim_k`` is the run's real
        temperature. Stating the wrong one shifts the answer: a run at
        ``T_run`` read with ``temp_sim_k = T_stated`` reads
        ``10 log10(1 + (T_run - T_stated)/T0)`` dB too high (+0.15 dB for a
        300.15 K run stated as 290 K). Outside the sweep, or where the
        density read is NaN, returns ``nan`` with a ``MetricWarning``. A non-positive factor (the density lies
        below the stated source noise by more than ``4 k T0 Rs``) also
        returns ``nan`` with a ``MetricWarning``.

    Raises
    ------
    ValueError
        If ``noise_in`` is not a noise density (off the frequency axis,
        complex ``y``), has fewer than 2 samples, ``freq`` is not finite,
        has mismatched/non-1-D arrays or an invalid frequency axis, is
        labelled as power density, or ``rs``, ``temp_sim_k`` or
        ``temp_ref_k`` is not finite and > 0.
    """
    name = 'noise_figure_at_freq'
    _check_kind(noise_in, NOISE_DENSITY, name, 'noise_in')
    _check_nf_units(noise_in, name, 'noise_in')
    _check_freq(noise_in, name)
    if not np.isfinite(freq):
        raise ValueError(f'{name}: freq must be finite, got {freq!r} Hz')
    _check_nf_conditions(name, rs, temp_sim_k, temp_ref_k)
    x = np.asarray(noise_in.x, dtype=float)
    y = np.asarray(noise_in.y, dtype=float)
    axis = _axis(noise_in, name)
    lo, hi = axis.lower, axis.upper
    if not axis.contains(freq):
        emit(
            metric=name, code='frequency_outside_sweep',
            message='Requested frequency is outside the supplied sweep; '
                    'the noise figure is unavailable.',
            details={'requested_frequency_hz': float(freq),
                     'frequency_min_hz': lo, 'frequency_max_hz': hi,
                     'sample_count': len(x),
                     'density_unit': noise_in.ylabel},
            suggestion='Choose a frequency within the sweep or extend '
                       'the sweep to cover the requested point.',
        )
        return float('nan')
    if interpolate:
        i0, i1, t = _bracket_freq(x, freq)
        pair = [i0] if i0 == i1 else [i0, i1]
    else:
        i0 = i1 = int(np.argmin(np.abs(x - freq)))
        t = 0.0
        pair = [i0]
    bad = [j for j in pair if np.isnan(y[j])]
    if bad:
        emit(
            metric=name, code='nan_density',
            message='Measurement unavailable: a selected neighbouring '
                    'sample density is NaN.',
            details={**_density_details(x[bad], y[bad], noise_in.ylabel),
                     'requested_frequency_hz': float(freq)},
            suggestion='Check the noise result at the selected frequency.',
        )
        return float('nan')
    y0, y1 = abs(float(y[i0])), abs(float(y[i1]))
    if i0 == i1:
        density = y0
    elif y0 > 0 and y1 > 0:
        density = float(np.exp(np.log(y0) + t * (np.log(y1) - np.log(y0))))
    else:
        density = y0 + t * (y1 - y0)
    factor = _factor_from_density(density ** 2, rs, temp_sim_k, temp_ref_k)
    if factor <= 0:
        _emit_nonpositive_factor(name, factor, rs, temp_sim_k, temp_ref_k,
                                 {'requested_frequency_hz': float(freq),
                                  'density': density,
                                  'density_unit': noise_in.ylabel})
        return float('nan')
    return float(10.0 * np.log10(factor))


@metric(domain='noise', unit='dB',
        param_units={'f1': 'Hz', 'f2': 'Hz', 'rs': 'ohm', 'temp_sim_k': 'K',
                     'temp_ref_k': 'K'})
def band_noise_figure(noise_in: NoiseDensity, noise_out: NoiseDensity,
                      f1: float, f2: float, rs: float, temp_sim_k: float,
                      *, temp_ref_k: float = T0_KELVIN) -> float:
    """Average (band) noise figure (dB) over ``[f1, f2]``, gain-weighted.

    The standard average noise factor weights the spot factor by the
    power gain ``G(f)``::

        F_band = integral F(f) G(f) df / integral G(f) df

    which is the ratio of the total output noise power to the part of it
    the source resistance would produce at ``T0``. Both spectra come from
    the same ``.noise`` run: ``G(f) = S_out(f) / S_in(f)`` is the run's
    own gain from the named input source (``rs`` in series with that
    source sees the same gain), so with ``S = density^2``::

        F_band = 1 + (integral S_out df - 4 k T_sim Rs integral G df)
                     / (4 k T0 Rs integral G df)
        NF     = 10 log10 F_band

    When ``T_sim = T0`` this is exactly ``(v_out / v_rs_out)^2``, the
    integrated total output noise over the source resistor's integrated
    output contribution. It is NOT the unweighted mean of the spot factor
    over the band: frequencies where the gain is low count for less.

    Parameters
    ----------
    noise_in : Signal
        Input-referred amplitude density in V/sqrt(Hz)
        (``load_signal('noise_in')``), with matching 1-D arrays of at least
        two samples and a finite, strictly monotonic frequency axis in Hz.
        Either sweep direction is accepted; neither signal is modified.
    noise_out : Signal
        Output amplitude density in V/sqrt(Hz) from the same run
        (``load_signal('noise_out')``),
        on exactly the same frequency grid. This same-capture requirement
        is intentionally stricter than the spacing-scaled tolerance used by
        other paired metrics; even a one-step floating-point difference raises
        ``ValueError``.
    f1, f2 : float
        Band in Hz. Edges snap inward to the sweep points inside
        ``[f1, f2]``, and the integrals are the trapezoid on those points,
        exactly as :func:`integrated_noise`.
    rs : float
        Source resistance in ohm (see :func:`noise_figure_at_freq`).
    temp_sim_k : float
        Temperature the simulation ran at, in K (no default).
    temp_ref_k : float
        Reference temperature ``T0`` in K; default 290 K (IEEE).

    Returns
    -------
    float
        Band NF in dB. ``nan`` with a ``MetricWarning`` when the data
        cannot carry a figure: a NaN density
        in the band, a zero input density (the gain is undefined), zero
        gain across the whole band (a dead output), or a non-positive
        band factor.

    Raises
    ------
    ValueError
        If either Signal is not a noise density, the two grids differ,
        ``[f1, f2]`` holds fewer than 2 sweep points (a band aggregate over
        an empty band, or a band outside the sweep, is a malformed request:
        the empty-band convention, as :func:`integrated_noise`), or ``rs``,
        ``temp_sim_k`` or ``temp_ref_k`` is not finite and > 0. Invalid
        array shapes, nonfinite/nonmonotonic frequency axes, and power
        density labels also raise.
    """
    name = 'band_noise_figure'
    _check_kind(noise_in, NOISE_DENSITY, name, 'noise_in')
    _check_kind(noise_out, NOISE_DENSITY, name, 'noise_out')
    _check_nf_units(noise_in, name, 'noise_in')
    _check_nf_units(noise_out, name, 'noise_out')
    _check_freq(noise_in, name)
    _check_freq(noise_out, name)
    _check_nf_conditions(name, rs, temp_sim_k, temp_ref_k)
    x = np.asarray(noise_in.x, dtype=float)
    xo = np.asarray(noise_out.x, dtype=float)
    # This exact same-capture contract is intentionally stricter than
    # _Axis.same_grid: paired noise densities must come from one .noise run.
    if x.shape != xo.shape or not np.array_equal(x, xo):
        raise ValueError(
            f'{name}: noise_in and noise_out must share one frequency grid '
            f'(the same .noise run), got {x.size} and {xo.size} points '
            f'spanning {np.min(x):g}..{np.max(x):g} Hz and '
            f'{np.min(xo):g}..{np.max(xo):g} Hz'
        )
    mask = (x >= f1) & (x <= f2)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f'{name}: band [{f1:g}, {f2:g}] Hz contains '
            f'{np.count_nonzero(mask)} of {x.size} sweep points (sweep spans '
            f'{np.min(x):g} Hz .. {np.max(x):g} Hz)'
        )
    axis = _axis(noise_in, name)
    xb = axis.orient(x[mask])
    yin = axis.orient(np.asarray(noise_in.y, dtype=float)[mask])
    yout = axis.orient(np.asarray(noise_out.y, dtype=float)[mask])
    band = {'band_min_hz': float(f1), 'band_max_hz': float(f2)}
    if _any_nan(yin, yout):
        nan_in = np.isnan(yin)
        which = yin if nan_in.any() else yout
        unit = noise_in.ylabel if nan_in.any() else noise_out.ylabel
        emit(
            metric=name, code='nan_density',
            message='Consumed band density contains NaN; the band noise '
                    'figure is unavailable.',
            details={**_density_details(xb, which, unit), **band,
                     'signal': 'noise_in' if nan_in.any() else 'noise_out'},
            suggestion='Inspect the noise result at the affected frequencies '
                       'and obtain finite density samples in the band.',
        )
        return float('nan')
    s_in = yin ** 2
    s_out = yout ** 2
    zero_in = s_in == 0
    if zero_in.any():
        emit(
            metric=name, code='zero_input_density',
            message='Input-referred density is zero inside the band, so the '
                    'gain S_out/S_in is undefined; the band noise figure is '
                    'unavailable.',
            details={**band, 'zero_sample_count': int(np.count_nonzero(zero_in)),
                     'zero_frequency_min_hz': float(np.min(xb[zero_in])),
                     'zero_frequency_max_hz': float(np.max(xb[zero_in])),
                     'sample_count': int(xb.size)},
            suggestion='A source resistance contributes noise at every '
                       'frequency; check that noise_in comes from a run '
                       'whose input source drives the circuit through rs.',
        )
        return float('nan')
    gain = s_out / s_in
    gain_int = float(trapezoid(gain, xb))
    if gain_int == 0:
        emit(
            metric=name, code='zero_gain',
            message='The power gain is zero across the band (a dead output); '
                    'the band noise figure is unavailable.',
            details={**band, 'sample_count': int(xb.size),
                     'density_unit': noise_out.ylabel},
            suggestion='Check that the noise output node is driven by the '
                       'input in this band.',
        )
        return float('nan')
    out_int = float(trapezoid(s_out, xb))
    factor = 1.0 + ((out_int - 4.0 * K_BOLTZMANN * temp_sim_k * rs * gain_int)
                    / (4.0 * K_BOLTZMANN * temp_ref_k * rs * gain_int))
    if factor <= 0:
        _emit_nonpositive_factor(name, factor, rs, temp_sim_k, temp_ref_k, band)
        return float('nan')
    return float(10.0 * np.log10(factor))


def noise_figure_vs_freq(noise_in: Signal, rs: float, temp_sim_k: float,
                         *, temp_ref_k: float = T0_KELVIN) -> Signal:
    """Spot noise figure in dB at every input frequency.

    Uses :func:`noise_figure_at_freq`'s source-resistance and temperature
    correction at each sample, without interpolation.

    Parameters
    ----------
    noise_in : Signal
        Input-referred amplitude density in V/sqrt(Hz), with matching 1-D
        ``x`` and real ``y`` arrays of at least two samples. The frequency
        axis in Hz must be finite and strictly increasing or decreasing.
    rs : float
        Source resistance in ohm, finite and positive.
    temp_sim_k : float
        Actual simulation temperature in K, finite and positive.
    temp_ref_k : float, optional
        Reference temperature in K, finite and positive; default 290.

    Returns
    -------
    Signal
        New signal with a 1-D dB array matching the input shape/order, its
        name and frequency axis, and ``type='noise-figure'``. Input data is
        not modified; the axis may share storage. Nonpositive or NaN factors
        become ``nan``; one ``MetricWarning`` reports all unavailable points.
        Positive infinite factors give ``inf``.

    Raises
    ------
    ValueError
        If the input is not a real noise density on a valid frequency axis,
        has fewer than two samples, is labelled as power density (V^2/Hz),
        or resistance/temperatures are not finite and positive.

    Notes
    -----
    The result is not a noise density, so density metrics refuse it.
    """
    name = 'noise_figure_vs_freq'
    _check_kind(noise_in, NOISE_DENSITY, name, 'noise_in')
    _check_nf_units(noise_in, name, 'noise_in')
    _check_freq(noise_in, name)
    _check_nf_conditions(name, rs, temp_sim_k, temp_ref_k)
    x = np.asarray(noise_in.x, dtype=float)
    y = np.asarray(noise_in.y, dtype=float)
    factor = _factor_from_density(y ** 2, rs, temp_sim_k, temp_ref_k)
    bad = ~(factor > 0)
    nf = np.full(factor.shape, np.nan)
    nf[~bad] = 10.0 * np.log10(factor[~bad])
    if bad.any():
        emit(
            metric=name, code='unavailable_points',
            message='Some sweep points have a NaN density or a non-positive '
                    'noise factor; the noise figure is nan there.',
            details={'unavailable_count': int(np.count_nonzero(bad)),
                     'sample_count': int(x.size),
                     'frequency_min_hz': float(np.min(x[bad])),
                     'frequency_max_hz': float(np.max(x[bad])),
                     'rs_ohm': float(rs), 'temp_sim_k': float(temp_sim_k),
                     'temp_ref_k': float(temp_ref_k)},
            suggestion='Check rs, temp_sim_k and the noise result at the '
                       'affected frequencies.',
        )
    return Signal(noise_in.name, 'noise-figure', noise_in.xlabel, 'dB', x, nf)
