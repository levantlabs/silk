"""silk.metrics.fom — figures of merit (closed-form derived scalars).

ADC efficiency / quality FoMs and the standard one-point IIP3
extrapolation. All inputs are scalars from upstream metric calls; no
Signal involved.
"""
from __future__ import annotations

import numpy as np


def walden_fom(power_w: float, fs_hz: float, enob_bits: float) -> float:
    """Walden figure of merit, ``P / (2**ENOB * fs)``.

    Lower values indicate less energy per conversion step.

    Parameters
    ----------
    power_w : float
        Converter power in W. Zero and negative values are not rejected.
    fs_hz : float
        Sampling rate in Hz; must be positive.
    enob_bits : float
        Effective resolution in bits, typically from :func:`enob`.

    Returns
    -------
    float
        Energy in J per conversion step. NaN operands propagate ``nan``
        without a ``MetricWarning``; zero power returns zero. Infinite
        operands follow the arithmetic (for example infinite rate gives zero
        for finite power).

    Raises
    ------
    ValueError
        If ``fs_hz <= 0``; the message includes the requested rate and unit.
    OverflowError
        If a finite ``enob_bits`` overflows Python's exponentiation.
    ZeroDivisionError
        If ``2**enob_bits * fs_hz`` underflows to zero.

    Examples
    --------
    >>> from silk.metrics import walden_fom
    >>> walden_fom(1e-3, 1e6, 10) == 1e-3 / (1024 * 1e6)
    True
    """
    if fs_hz <= 0:
        raise ValueError(
            f'walden_fom: fs_hz must be > 0 Hz, got {fs_hz} Hz')
    return power_w / ((2.0 ** enob_bits) * fs_hz)


def schreier_fom(sndr_db: float, bw_hz: float, power_w: float) -> float:
    """Schreier figure of merit, ``SNDR + 10*log10(BW/P)``, in dB.

    Higher values indicate better resolution and bandwidth per unit power.

    Parameters
    ----------
    sndr_db : float
        Signal-to-noise-and-distortion ratio in dB.
    bw_hz : float
        Signal bandwidth in Hz; must be positive.
    power_w : float
        Converter power in W; must be positive.

    Returns
    -------
    float
        Figure of merit in dB. NaN operands propagate ``nan`` quietly;
        infinite operands follow NumPy arithmetic. No ``MetricWarning`` is
        emitted; a zero ratio from underflow or infinite power gives ``-inf``
        and may emit a NumPy ``RuntimeWarning``.

    Raises
    ------
    ValueError
        If ``bw_hz <= 0`` or ``power_w <= 0``. The bandwidth is checked first;
        the message includes the offending value and unit.

    Examples
    --------
    >>> from silk.metrics import schreier_fom
    >>> schreier_fom(70.0, 10e6, 1e-3)
    170.0
    """
    if bw_hz <= 0:
        raise ValueError(
            f'schreier_fom: bw_hz must be > 0 Hz, got {bw_hz} Hz')
    if power_w <= 0:
        raise ValueError(
            f'schreier_fom: power_w must be > 0 W, got {power_w} W')
    return float(sndr_db + 10.0 * np.log10(bw_hz / power_w))


def iip3_extrapolation(p_in_dbm: float, p_fund_dbm: float,
                       p_imd3_dbm: float) -> float:
    """Extrapolate the input third-order intercept from one two-tone measurement.

    Uses ``P_in + (P_fund - P_imd3)/2``. The measurement must be in the
    small-signal regime where the third-order product grows three times as
    fast in dB as the fundamental; this assumption is not checked.

    Parameters
    ----------
    p_in_dbm : float
        Input power of one tone in dBm.
    p_fund_dbm : float
        Output fundamental power of that tone in dBm.
    p_imd3_dbm : float
        Output third-order intermodulation product power in dBm.

    Returns
    -------
    float
        Input intercept power in dBm. NaN operands propagate ``nan``;
        infinities follow subtraction/addition, including ``nan`` for
        ``inf - inf``. No input validation or ``MetricWarning`` is emitted.

    Examples
    --------
    >>> from silk.metrics import iip3_extrapolation
    >>> iip3_extrapolation(-30.0, -30.0, -90.0)
    0.0
    """
    return p_in_dbm + 0.5 * (p_fund_dbm - p_imd3_dbm)


# ---------------------------------------------------------------------------
# Power conversion efficiency
# ---------------------------------------------------------------------------

def pae(p_rf_out_w: float, p_rf_in_w: float, p_dc_w: float) -> float:
    """Power-added efficiency, ``(P_RF_out - P_RF_in) / P_DC``.

    Parameters
    ----------
    p_rf_out_w : float
        Output RF power in W.
    p_rf_in_w : float
        Input RF power in W.
    p_dc_w : float
        DC supply power in W; must be positive.

    Returns
    -------
    float
        Dimensionless efficiency ratio, not percent. Equal RF powers give
        zero; output below input gives a negative result. NaN operands
        propagate ``nan`` and infinite values follow arithmetic. No
        ``MetricWarning`` is emitted and the result is not clipped.

    Raises
    ------
    ValueError
        If ``p_dc_w <= 0``, with the offending value in W.
    """
    if p_dc_w <= 0:
        raise ValueError(
            f'pae: p_dc_w must be > 0 W, got {p_dc_w} W')
    return (p_rf_out_w - p_rf_in_w) / p_dc_w


def dc_to_rf_efficiency(p_rf_w: float, p_dc_w: float) -> float:
    """Drain efficiency, ``P_RF_out / P_DC``.

    Parameters
    ----------
    p_rf_w : float
        Output RF power in W; zero and negative values are not rejected.
    p_dc_w : float
        DC supply power in W; must be positive.

    Returns
    -------
    float
        Dimensionless ratio, not percent, without clipping. Zero RF power
        gives zero. NaN operands propagate ``nan`` and infinite operands
        follow arithmetic. No ``MetricWarning`` is emitted.

    Raises
    ------
    ValueError
        If ``p_dc_w <= 0``, with the offending value in W.
    """
    if p_dc_w <= 0:
        raise ValueError(
            f'dc_to_rf_efficiency: p_dc_w must be > 0 W, got {p_dc_w} W')
    return p_rf_w / p_dc_w
