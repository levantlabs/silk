"""silk.metrics — analog and mixed-signal performance metrics.

Files group metrics by purpose. Every metric keeps its schema domain; its
input signal kinds describe the data it accepts. A file name does not select
a simulator analysis, and general tools accept axes beyond time.

* :mod:`silk.metrics.waveform` — General waveform tools that work on any x-axis (time, a DC sweep, frequency).
* :mod:`silk.metrics.statistics` — Waveform statistics and comparisons against a target waveform.
* :mod:`silk.metrics.timing` — Edge timing and response measurements on time waveforms.
* :mod:`silk.metrics.jitter` — Clock jitter and stability measurements on time waveforms.
* :mod:`silk.metrics.eye` — Serial-data eye diagrams, eye height and eye width from time waveforms.
* :mod:`silk.metrics.power` — Power and energy from paired voltage and current time waveforms.
* :mod:`silk.metrics.spectral` — FFT and power-spectrum metrics from clock-sampled time waveforms.
* :mod:`silk.metrics.rf` — RF compression, intercept, intermodulation, mixing and modulation metrics.
* :mod:`silk.metrics.converter` — Data-converter static linearity, effective resolution and jitter limits.
* :mod:`silk.metrics.fom` — Figures of merit from measured scalars: ADC FoMs, PA efficiency and IIP3.
* :mod:`silk.metrics.ac` — Small-signal frequency response: gain, bandwidth, margins and filter shape.
* :mod:`silk.metrics.noise` — Noise figure, integrated and input-referred noise, density and noise formulas.
* ``silk.metrics.diagnostics`` — measurement warnings and warning records.

Waveform functions take a :class:`silk.signals.Signal`; closed-form formulas
take numbers. Each function documents its failure contract. Malformed requests
raise; unavailable measurements usually return NaNs or empty arrays, often
with :class:`silk.metrics.diagnostics.MetricWarning`. Warning filters apply
normally, including promotion of a warning to an exception. Metrics do not
modify input waveform arrays; some returned signals share storage as documented.

Clock-sampled spectral functions, ENOB, and RF two-tone and mixing functions
accept ``(tclk, tstart, n_samples)`` for sample extraction. EVM functions accept
``tstart`` only; ``modulation_grid`` takes no sample-extraction arguments.
Consult each function for its sampling options.

User-facing reference: ``docs/metrics-library.md``.

Examples
--------
Construct a waveform without a simulator and measure its time-weighted mean:

>>> import numpy as np
>>> from silk import Signal
>>> from silk.metrics import average, peak_to_peak
>>> ramp = Signal('ramp', 'voltage', 'time', 'V',
...               np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
>>> average(ramp), peak_to_peak(ramp)
(1.0, 2.0)

A spectral analysis of an existing simulation signal can be composed as::

    from silk.metrics import sndr, enob, fft, rise_time, walden_fom

    sndr_db = sndr(adc_out_sig, tclk=10e-9, tstart=20e-9, n_samples=128)
    bits    = enob(adc_out_sig, tclk=10e-9, tstart=20e-9, n_samples=128)
    spec    = fft (adc_out_sig, tclk=10e-9, tstart=20e-9, n_samples=128)
    spec.plot()
    tr      = rise_time(step_response_sig)
    fom     = walden_fom(power_w=1e-3, fs_hz=1e8, enob_bits=bits)
"""

from silk.metrics import (
    waveform,
    statistics,
    timing,
    jitter,
    eye,
    power,
    spectral,
    rf,
    converter,
    fom,
    ac,
    noise,
)

# waveform.py
from silk.metrics.waveform import (
    crossing,
    crossings,
    value_at,
    final_value,
    argmin_x,
    argmax_x,
    peak,
    peak_to_peak,
    minimum,
    maximum,
    derivative,
    integral,
    running_integral,
    moving_average,
    clip,
    window,
    flip,
    lshift,
    rshift,
    sample,
    convolve,
    cross_correlation,
)

# statistics.py
from silk.metrics.statistics import (
    average,
    rms,
    stddev,
    variance,
    histogram,
    HistogramResult,
    error_norm,
    relative_error_norm,
)

# timing.py
from silk.metrics.timing import (
    rise_time,
    fall_time,
    slew_rate,
    rise_times,
    fall_times,
    slew_rates,
    settling_time,
    propagation_delay,
    propagation_delays,
    overshoot,
    undershoot,
    on_time,
    off_time,
    pulse_width,
    pulse_widths,
    average_duty,
    cycle_duty,
    average_period,
    average_frequency,
    cycle_period,
    cycle_frequency,
    hysteresis,
)

# jitter.py
from silk.metrics.jitter import (
    period_jitter,
    period_jitter_pp,
    cycle_to_cycle_jitter,
    cycle_to_cycle_jitter_pp,
    period_deviation,
    cycle_to_cycle_difference,
    crossing_jitter_rms,
    tie,
    long_term_jitter,
    allan_deviation,
)

# eye.py
from silk.metrics.eye import (
    eye_diagram,
    EyeResult,
    eye_height,
    eye_width,
)

# power.py
from silk.metrics.power import (
    average_power,
    dynamic_energy_per_cycle,
)

# spectral.py
from silk.metrics.spectral import (
    fft,
    psd,
    welch_psd,
    spectral_power,
    coherence,
    harmonic,
    harmonic_freqs,
    harmonic_level_dbc,
    harmonic_levels_db,
    thd,
    thd_percent,
    noise_floor_dbfs,
    snr,
    sndr,
    sinad,
    sfdr,
    sfdr_db,
    SFDR,
)

# rf.py
from silk.metrics.rf import (
    p1db,
    iip3_dbm,
    oip3_dbm,
    imd3_db,
    mix_product,
    evm_constellation,
    evm_percent,
    EVMResult,
    modulation_grid,
)

# converter.py
from silk.metrics.converter import (
    extract_code_transitions,
    inl,
    INL,
    inl_max_lsb,
    dnl,
    DNL,
    dnl_max_lsb,
    missing_codes,
    missing_code_count,
    is_monotonic,
    non_monotonic_steps,
    static_errors,
    StaticErrors,
    offset_error_lsb,
    gain_error_lsb,
    full_scale_error_lsb,
    enob,
    aperture_jitter_snr_db,
)

# fom.py
from silk.metrics.fom import (
    walden_fom,
    schreier_fom,
    iip3_extrapolation,
    pae,
    dc_to_rf_efficiency,
)

# ac.py
from silk.metrics.ac import (
    dc_gain_dB,
    peak_gain_dB,
    gain_at_freq,
    phase_at_freq,
    bandwidth_3dB,
    unity_gain_freq,
    gain_bw_product,
    phase_margin,
    gain_margin,
    gain_margins,
    group_delay,
    passband_ripple_db,
    stopband_attenuation_db,
    q_from_peak,
    psrr_db,
    cmrr_dB,
)

# noise.py
from silk.metrics.noise import (
    noise_figure,
    noise_figure_from_factor,
    noise_figure_at_freq,
    noise_figure_vs_freq,
    band_noise_figure,
    cascade_noise_figure,
    input_referred_noise,
    integrated_noise,
    noise_density_at_freq,
    one_over_f_corner,
    kT_over_C,
    sum_uncorrelated,
)

__all__ = [
    # Purpose modules
    "waveform",
    "statistics",
    "timing",
    "jitter",
    "eye",
    "power",
    "spectral",
    "rf",
    "converter",
    "fom",
    "ac",
    "noise",
    # waveform.py
    "crossing",
    "crossings",
    "value_at",
    "final_value",
    "argmin_x",
    "argmax_x",
    "peak",
    "peak_to_peak",
    "minimum",
    "maximum",
    "derivative",
    "integral",
    "running_integral",
    "moving_average",
    "clip",
    "window",
    "flip",
    "lshift",
    "rshift",
    "sample",
    "convolve",
    "cross_correlation",
    # statistics.py
    "average",
    "rms",
    "stddev",
    "variance",
    "histogram",
    "HistogramResult",
    "error_norm",
    "relative_error_norm",
    # timing.py
    "rise_time",
    "fall_time",
    "slew_rate",
    "rise_times",
    "fall_times",
    "slew_rates",
    "settling_time",
    "propagation_delay",
    "propagation_delays",
    "overshoot",
    "undershoot",
    "on_time",
    "off_time",
    "pulse_width",
    "pulse_widths",
    "average_duty",
    "cycle_duty",
    "average_period",
    "average_frequency",
    "cycle_period",
    "cycle_frequency",
    "hysteresis",
    # jitter.py
    "period_jitter",
    "period_jitter_pp",
    "cycle_to_cycle_jitter",
    "cycle_to_cycle_jitter_pp",
    "period_deviation",
    "cycle_to_cycle_difference",
    "crossing_jitter_rms",
    "tie",
    "long_term_jitter",
    "allan_deviation",
    # eye.py
    "eye_diagram",
    "EyeResult",
    "eye_height",
    "eye_width",
    # power.py
    "average_power",
    "dynamic_energy_per_cycle",
    # spectral.py
    "fft",
    "psd",
    "welch_psd",
    "spectral_power",
    "coherence",
    "harmonic",
    "harmonic_freqs",
    "harmonic_level_dbc",
    "harmonic_levels_db",
    "thd",
    "thd_percent",
    "noise_floor_dbfs",
    "snr",
    "sndr",
    "sinad",
    "sfdr",
    "sfdr_db",
    "SFDR",
    # rf.py
    "p1db",
    "iip3_dbm",
    "oip3_dbm",
    "imd3_db",
    "mix_product",
    "evm_constellation",
    "evm_percent",
    "EVMResult",
    "modulation_grid",
    # converter.py
    "extract_code_transitions",
    "inl",
    "INL",
    "inl_max_lsb",
    "dnl",
    "DNL",
    "dnl_max_lsb",
    "missing_codes",
    "missing_code_count",
    "is_monotonic",
    "non_monotonic_steps",
    "static_errors",
    "StaticErrors",
    "offset_error_lsb",
    "gain_error_lsb",
    "full_scale_error_lsb",
    "enob",
    "aperture_jitter_snr_db",
    # fom.py
    "walden_fom",
    "schreier_fom",
    "iip3_extrapolation",
    "pae",
    "dc_to_rf_efficiency",
    # ac.py
    "dc_gain_dB",
    "peak_gain_dB",
    "gain_at_freq",
    "phase_at_freq",
    "bandwidth_3dB",
    "unity_gain_freq",
    "gain_bw_product",
    "phase_margin",
    "gain_margin",
    "gain_margins",
    "group_delay",
    "passband_ripple_db",
    "stopband_attenuation_db",
    "q_from_peak",
    "psrr_db",
    "cmrr_dB",
    # noise.py
    "noise_figure",
    "noise_figure_from_factor",
    "noise_figure_at_freq",
    "noise_figure_vs_freq",
    "band_noise_figure",
    "cascade_noise_figure",
    "input_referred_noise",
    "integrated_noise",
    "noise_density_at_freq",
    "one_over_f_corner",
    "kT_over_C",
    "sum_uncorrelated",
]
