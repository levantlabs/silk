# silk.metrics — analog and mixed-signal measurements

Import measurements from `silk.metrics`. Waveform measurements take `Signal`
objects; closed-form formulas take numbers. You can measure a loaded simulation
result, a deferred testbench probe, or a waveform you construct from arrays.

## Start with a waveform

This example runs without a simulator or PDK:

```python
import numpy as np
from silk import Signal
from silk.metrics import average, peak_to_peak, rise_time

ramp = Signal('ramp', 'voltage', 'time', 'V',
              np.array([0.0, 1e-9, 2e-9]), np.array([0.0, 0.5, 1.0]))
assert np.isclose(average(ramp), 0.5)
assert np.isclose(peak_to_peak(ramp), 1.0)
assert np.isclose(rise_time(ramp), 1.6e-9, rtol=1e-12, atol=0)
```

For an existing testbench `tb`, register measurements before running it:

```python
from silk.metrics import average

vout = tb.probe('vout')
tb.add_metric('out_avg', average, sig=vout)
```

This fragment requires a testbench with a `vout` signal. The named `sig=` slot
resolves after simulation. Direct calls such as `average(vout)` need results
already available. See [probes](probes.md) for analysis selection, deferred
arithmetic, current probes and corner selection.

## Layout

The submodules group functions by their input data. All functions in the
reference below are also available from `silk.metrics`.

See [circuit-based metric examples](examples.md#metric-coverage-silkexamplesmetrics)
for worked measurement workflows.

| Module | Input and purpose |
|---|---|
| `silk.metrics.waveform` | General tools on any x-axis: crossings, values, extrema, calculus, sampling and transforms |
| `silk.metrics.statistics` | Waveform statistics and error norms; histogram is time-weighted and time-only |
| `silk.metrics.timing` | Time-waveform edges, delay, pulse, duty, period, frequency, response and hysteresis |
| `silk.metrics.jitter` | Clock jitter, TIE, long-term stability and Allan deviation |
| `silk.metrics.eye` | Serial-data eye folds, height and width |
| `silk.metrics.power` | Power and energy from paired voltage and current waveforms |
| `silk.metrics.spectral` | Clock-sampled spectra, harmonics, distortion, SNR, SNDR, SFDR and coherence |
| `silk.metrics.rf` | Power-sweep compression, two-tone intercept/intermodulation, mixing and I/Q EVM |
| `silk.metrics.converter` | Code-transition linearity, ENOB from sampled waveforms and scalar aperture-jitter limit |
| `silk.metrics.fom` | Figures of merit from measured scalars, including PA efficiency |
| `silk.metrics.ac` | Small-signal frequency response: gain, bandwidth, margins, rejection and filter shape |
| `silk.metrics.noise` | Noise-density queries and integration, noise figure and scalar noise formulas |
| `silk.metrics.diagnostics` | Measurement warnings and warning records |

Files group measurements by purpose; signal kinds define which inputs each metric
accepts.

## Sampling convention (spectral metrics)

Clock-sampled spectra take `tclk` (required period in seconds), `tstart=0.0`
(first sampling instant), and `n_samples=None` (the largest power of two that
fits after `tstart`). Select at least four samples; `tstart` cannot precede the trace.
Explicit `n_samples` need not be a power of two. Sampling occurs at `tstart + k*tclk`. Choose `tstart`
to skip startup and set the sampling phase together. Off-grid instants use
linear interpolation; exact stored-time matches keep their recorded values.
An `interpolated_samples` warning reports counts and the maximum distance to
the nearest recorded time. That distance is not a voltage-error estimate.

For a coherent record, the tone completes an integer number of cycles in
`n_samples*tclk`. A rectangular window is suitable for such records.

```python
import numpy as np
from silk import Signal
from silk.metrics import sndr, enob, sfdr, fft

tclk, n_samples = 1e-9, 256
k = np.arange(n_samples)
sig = Signal('tone', 'voltage', 'time', 'V', k*tclk,
             0.4*np.sin(2*np.pi*7*k/n_samples))
quality = sndr(sig, tclk=tclk, n_samples=n_samples)
bits = enob(sig, tclk=tclk, n_samples=n_samples)
spur = sfdr(sig, tclk=tclk, n_samples=n_samples)
spectrum = fft(sig, tclk=tclk, n_samples=n_samples, scaling='amplitude')
assert np.isfinite(quality) and np.isfinite(bits)
assert np.isclose(spur.fundamental_hz, 7/(n_samples*tclk))
assert np.isclose(np.max(spectrum.y), 0.4)
```

### Spectrum options: fundamental, guard bins, window, full scale

`fundamental_hz=None` selects the strongest non-DC bin, excluding the window's
DC main lobe when bins remain beyond it, in `sndr`, `sinad`,
`enob`, `snr`, `sfdr`, `sfdr_db`, `thd`, `thd_percent`, `noise_floor_dbfs`,
`harmonic_levels_db`, `harmonic_level_dbc`, `harmonic` and `harmonic_freqs`.
An explicit frequency selects the nearest bin; a selection at DC or beyond
available bins raises `ValueError`.

The power-ratio functions through `noise_floor_dbfs` in that list also take
`window='rectangular'` and `guard_bins=0`. The other supported windows are
`'hann'`, `'hamming'` and `'blackman'`. They apply to mean-removed samples.
Each tone includes the window's main lobe plus `guard_bins` on either side
of its centre. The main-lobe half-width is 0 bins for rectangular, 1 for
Hann/Hamming and 2 for Blackman. DC owns a band of the same width.
The noise sums in SNDR/SINAD/ENOB/SNR compensate for bins excluded by wider
bands. Increase guards for leakage, while retaining enough bins to measure
noise. Invalid guards raise `ValueError`. For metrics measuring residual noise,
guards that remove otherwise available residual bins also raise; an already-empty
selection is unavailable data. THD does not request residual noise bins. The four `harmonic*` queries listed above do not accept
`window` or `guard_bins`.

`sndr`, `sinad`, `enob`, `snr` and `noise_floor_dbfs` take `full_scale=None`.
An explicit `full_scale` is the peak amplitude of a reference sine in input
y units; ratios then use its power instead of the measured fundamental's.
A constant trace can therefore reach a numerical dBFS ceiling. Check waveform
activity separately when using this reference for a specification.

`fft` defaults to raw squared FFT magnitude. Use `scaling='amplitude'` for
sine peak amplitude or `'power'` for sine mean-square power; these correct
for window coherent gain. `psd` and `welch_psd` instead return density per Hz.
`harmonic` and `mix_product` return raw complex FFT coefficients.

Harmonics above Nyquist fold into the sampled spectrum. Collisions with DC or
the fundamental are unidentifiable. THD/SNR power sums count each remaining
bin once; per-harmonic level queries can report the same shared bin level for
multiple orders. THD describes sampled-record distortion; it cannot recover
an analog harmonic's amplitude or phase before aliasing.

Ratio metrics use one-sided sampled power: conjugate pairs have weight 2,
while DC and an even record's Nyquist bin have weight 1. Default `fft` and
`harmonic` expose raw FFT quantities instead. Account for this weighting when
comparing ratios near Nyquist; a 3 dB difference can be a normalization effect.

## Edge extraction and propagation delay

`crossings` returns an array; `crossing` returns one float. `n=1` selects the
first crossing, `n=-1` the last; other nonzero integers count from the same
ends. The default edge is `'rising'`; `'falling'` and `'any'` are also accepted.
A missing selected crossing returns `nan`, while a valid empty `crossings`
search returns an empty array. Crossings interpolate linearly between samples.

For transient edge functions, `threshold=None` means the midpoint of the
observed minimum and maximum inside `start`/`end`. It is not the time-weighted
average. A window selects events within its bounds.

`rise_time`, `fall_time` and `slew_rate` share an edge selector. `low=0.1`
and `high=0.9` are fractions of the swing measured from its lower level.
State `initial` and `final` when overshoot would distort inferred levels;
for a falling edge `initial` is the high level. `slew_rate` reports a positive
magnitude for either direction. The plural forms `rise_times`, `fall_times`
and `slew_rates` return every complete edge. An unavailable edge produces a
warning and `nan` in the singular form, an empty array in the plural form.

`propagation_delay` selects the `in_n`-th input crossing, then the `out_n`-th
output crossing at or after that trigger. Output events before the trigger
are excluded. `out_edge=None` follows `in_edge`; name `'falling'` for an
inverting stage driven by a rising input. Each side has its own threshold.
`propagation_delays` applies the same pairing rule to every input trigger.

Compose trip offset with `value_at(vin, crossing(vout, threshold))`, kickback
with `peak_to_peak(sig, start=t0, end=t1)`, and hold droop with
`value_at(sig, t0) - value_at(sig, t1)`. Choose the event, window and reference
for the measurement: a clocked output's first rising edge can be reset.
See the [comparator and sampler example](../silk/examples/metrics/comparator_and_sampler.py).

`settling_time` uses the last-out-of-band convention and reports elapsed time
from `start`. Its tolerance is `abs(tol)*abs(target-initial)`; `target=None` uses
the last sample. Overshoot and undershoot are fractions of the step swing.
Undershoot is measured only from the first sample at/past target onward.

## Period, duty and jitter

A period requires two same-direction edges in the window. `edge='any'`
reports full periods at both rising and falling edges, rather than half
periods. `cycle_period` places each period at its two edges' midpoint;
`average_period` averages those periods. `average_frequency` is the reciprocal
of the average period; `cycle_frequency` takes each period's reciprocal.

`average_duty` divides total high time by total time over full cycles, so
cycles are weighted by duration. `cycle_duty` returns each cycle's fraction.
For a burst or partial cycle use `on_time`/`off_time`, which estimate duration
by trapezoidal integration of thresholded samples; window bounds are
interpolated, but threshold crossings are not inserted.
`pulse_width`/`pulse_widths` require both edges of a complete high pulse
(`polarity='positive'`) or low pulse (`'negative'`).

Period jitter measures `d[k] = period[k] - reference[k]`. The default
`reference='mean'` includes slow drift. With `reference='moving_average'`,
`cycles=N` is required and must be an integer at least 2. The centred average
uses N periods for odd N; even N uses N+1 with half-weighted end periods.
The first and last `N//2` periods lack a full reference and are excluded.
Cycle-to-cycle jitter compares each deviation with the next cycle that starts
at its ending edge; with `edge='any'`, this is two entries later in the
interleaved sequence.
The scalar jitter functions return RMS or peak-to-peak as named;
`period_deviation` and `cycle_to_cycle_difference` return waveforms.

`tie` compares edge times with `phase + k*ideal_period`; for `edge='any'`,
the reference is `phase + k*ideal_period/2`. Its default
`reference='fit'` uses a least-squares line; `'endpoints'` uses the first and
last edges. Explicit period and phase override the fitted slope/intercept.
`long_term_jitter` returns the sample standard deviation of TIE (`ddof=1`).
Its `units` argument supports seconds, UI, radians or degrees; the schema
unit records the default seconds. `allan_deviation` measures clock stability
at averaging times `taus` and returns the actual whole-period times used.

For data with runs of identical bits, use `crossing_jitter_rms(period=ui)`.
It folds each crossing modulo the supplied UI, subtracts the circular mean,
and returns sample standard deviation (`ddof=1`). Missing edges do not advance
an assumed edge index. Frequency offset still contributes phase drift;
`edge='any'` also includes rising/falling offsets such as duty distortion.

## Waveform statistics and transforms

General statistics work on any x-axis. `start`/`end` default to the full
trace; `interpolate=True` inserts values at window boundaries. Means, RMS,
variance and integrals are weighted by x spacing, including nonuniform grids.
A raw arithmetic mean of simulation vertices can overweight fast edges.

`average`, `integral`, `value_at` and `final_value` preserve complex values.
`rms` measures magnitude; variance/stddev measure spread about the complex
mean. Extrema and `crossing` require a real projection of complex inputs:
choose `.real()`, `.imag()`, `.mag()` or a phase transform explicitly.
Complex window edges can make the usual variance/RMS identity differ because
RMS interpolates magnitudes there.

`error_norm` interpolates the goal onto the measurement's grid, holds the
goal's endpoints outside its coverage, subtracts complex values before taking
magnitude, then returns a max, RMS or mean-absolute norm. Compared full-record
NaNs are checked before window selection. `relative_error_norm` divides by
goal magnitude and excludes values below `floor` and exactly zero. Its
trapezoidal integration bridges gaps between retained points; it does not
integrate only the retained contiguous segments.

`sample` interpolates real or complex values onto `x_new` or `n` uniform
points. Its source axis must be finite and strictly monotonic in either
direction; targets retain their supplied order and clamp outside coverage.
`flip` reverses y and mirrors x as `x[0] + x[-1] - x[::-1]`, which matters on
nonuniform grids. `clip` limits y, while `window` restricts x.
Transforms can return arrays sharing storage with the input; copy arrays
before independently editing returned data.

`histogram` reports time fractions in value bins using piecewise-linear
segments, rather than counting recorded vertices. Integer `bins` spans the
observed range; explicit edges exclude time outside those edges.
For B bins, the returned `HistogramResult` exposes B+1 `edges` and B
`fractions`.

`average_power` and `dynamic_energy_per_cycle` take voltage and current
Signals on matching grids; recognized swapped types are refused. Energy
integration starts at the trace's first sample, discards a partial-cycle
tail, and interpolates power at the final complete-cycle boundary. A trace
shorter than one full period raises `ValueError`.

## Eye centering and original samples

The [simulator-free eye-centering example](../silk/examples/eye_centering.py)
shows how to plot the returned `result.traces` Signals.

`eye_diagram(sig, period=ui)` folds full UIs into chronological `Signal`
windows with original samples and values. It neither resamples nor synthesizes
endpoints. The default `auto_center=False` only folds. `n_traces=None` returns
all available full windows; an explicit count must fit. The input must span
at least two full periods, even if only one trace is requested; shorter inputs
raise `ValueError`.

With `auto_center=True`, centering uses all crossings of the full record's
time-weighted mean voltage. The equal-weight circular mean crossing phase
plus half a UI selects the centre, relative to the original `sig.x[0]`.
The windows shift so this centre appears at `ui/2` when plotted. Leading and
trailing partial spans are omitted, so centering can reduce window capacity.
This alignment is a display aid, not clock recovery or an optimum receiver
sampling point.

Nonfinite voltages, no crossings or a circular-mean magnitude at most `1e-12`
leave centering unavailable: the original fold is retained, phase is `nan`,
and `centering_unavailable` explains the result. The centre is also checked
at mean voltage ±5% of the observed swing. An undefined probe or a displacement
above 0.10 UI from the nominal centre emits `centering_sensitive`. The
original phase and fold remain, but dependent measurements are unavailable.
These thresholds are advisory heuristics, not a confidence score.

Supply an explicit scalar `threshold` and eligible centering to measure one
eye. At least two returned full windows are needed. Height is the nearest
centre observation above threshold minus the nearest below; width is the
clear interval around the centre between waveform contacts at that threshold
on the UI circle. Original piecewise-linear segments support the calculation,
including across window boundaries, without changing output samples.
An observation exactly at threshold gives zero height and width.

`eye_height` and `eye_width` require `threshold` and enable centering by
default. Missing or sensitive centering, insufficient samples, or an
unbracketed threshold leaves dependent measurements `nan`. For PAM, measure
each desired threshold separately and take the minimum yourself.
Without a threshold, scalar measurements are `nan` and per-opening tuples
are empty. These geometric gaps cannot detect overlapping symbol populations;
use a meaningful positive margin in a spec. See [eye limitations](known_issues.md#eye-measurements-do-not-detect-population-overlap).

## AC gain, bandwidth and noise

Frequency-response metrics take `xlabel='frequency'`, finite strictly
monotonic Hz coordinates and at least two points. Magnitude measurements
accept complex responses or real linear magnitudes. `group_delay` requires
complex y to recover phase. Pass the complex response to `phase_at_freq`,
`phase_margin`, `gain_margin` and `gain_margins`; real input emits
`real_response_no_phase` and uses only 0°/180° phase. Feed stability metrics the loop gain under the
negative-feedback convention (low-frequency phase near 0 degrees), choosing
the feedback sign explicitly when constructing the response.

`dc_gain_dB` reads the lowest frequency, not an extrapolation to zero.
`bandwidth_3dB(reference='dc')` uses that magnitude as its reference;
`reference='peak'` instead uses the sweep's peak. The result is the highest
sampled frequency at/above the threshold, not an interpolated crossing.
An unobserved upper roll-off returns `nan`. `gain_bw_product` multiplies
DC gain by the DC-relative bandwidth. `q_from_peak` uses the sampled
half-power span around a peak. Isolate one resonance first: it spans all
samples above the global peak-relative threshold, including disconnected lobes.
Sweep density limits all three.

`gain_at_freq` interpolates magnitude log-log; `phase_at_freq` interpolates
in degrees versus log frequency along the short arc. Set `interpolate=False`
to read the nearest point. `unity_gain_freq` interpolates a unity crossing
and returns `nan` if none is observed. `gain_margins` returns all phase
crossovers; `gain_margin` selects the margin closest to 0 dB.

Density integration and point queries accept real density in V/√Hz or A/√Hz.
`integrated_noise`
integrates squared density by trapezoid and takes its square root; band edges
snap inward to existing sweep points and require at least two selected points.
`noise_density_at_freq` interpolates log-log, or reads nearest with
`interpolate=False`; outside the sweep it returns `nan`.

The noise-figure functions require input-referred voltage amplitude density
in V/√Hz and a source resistance `rs` in series with the noise-analysis input
source. `noise_figure_at_freq` requires the actual simulation temperature
`temp_sim_k`. Its reference temperature defaults to
290 K. The source resistance and both temperatures must be finite and strictly
positive; otherwise the noise-figure functions raise `ValueError`. It subtracts the source thermal noise at the simulation temperature
before referring added noise to the reference temperature. `band_noise_figure`
uses gain-weighted integration, with gain from `noise_out²/noise_in²`.
Its input/output voltage densities must come from the same run on the same
frequency grid. A runtime kind check does not establish these physical conditions.
`input_referred_noise` divides an already integrated RMS output noise by
one flat gain; it is unsuitable for a frequency-dependent gain across a band.

A complex frequency response is refused as a noise density. Real frequency
responses and densities can share the same observable axis/dtype, so the
runtime kind guard cannot always distinguish them. Supply the appropriate
physical quantity even when a guard accepts the Signal. Schema consumers
must also filter by declared signal kind; see [schema export](schema-export.md).

## Signal kinds no analysis produces

Three catalog kinds require data you assemble: `code_transitions`,
`power_sweep` and `reference_waveform`. Their `produced_by` lists are empty:
no analysis directly returns these inputs.

For ADC static linearity, combine digital nets with `codes_from_bits` and
pass the resulting decimal-code transient plus the input ramp to
`extract_code_transitions`. Binary packing requires explicit `msb_first`;
thermometer packing counts asserted bits and refuses that argument.
Use `codes_from_bits(bits, weighting='thermometer', threshold=0.9)` for a
three-bit flash converter's seven comparator outputs. A bit is asserted
where `y > threshold`; the default threshold is 0.5.

```python
import numpy as np
from silk import Signal
from silk.signals import codes_from_bits
from silk.metrics import extract_code_transitions, inl, dnl, missing_codes

t = np.arange(8, dtype=float)*1e-9
vin = Signal('vin', 'voltage', 'time', 'V', t, np.arange(8)/8)
bits = [Signal(f'd{b}', 'voltage', 'time', 'V', t,
               ((np.arange(8) >> b) & 1).astype(float)) for b in (2, 1, 0)]
code = codes_from_bits(bits, msb_first=True)
transitions = extract_code_transitions(vin, code)
assert np.isclose(inl(transitions).max_abs_lsb, 0)
assert np.isclose(dnl(transitions).max_abs_lsb, 0)
assert len(missing_codes(transitions)) == 0
```

DNL requires consecutive integer codes; missing-code queries also require them
for finite transition data. An unavailable missing-code result is `[nan]`, not
an empty array. A transition table has x = integer code and y = transition voltage. The
lowest observed code has no transition row. A skipped code shares its
successor's voltage, producing DNL = −1. `is_monotonic` tests increasing
transition voltages and any `code_reversals` recorded by the extractor.
Pass the extractor's returned Signal directly: derived Signals do not
preserve that reversal metadata. `static_errors` requires `full_scale` and
`n_bits`; its ideal transition into k is `v_low + k*full_scale/2**n_bits`.

For compression, collect one input/output power pair per drive-level run.
`power_sweep` requires exactly two equal-length arrays, each with at least
two points, with strictly
ascending input dBm. `p1db` uses the first point's gain as its small-signal
reference and interpolates the first 1 dB reduction. This numerical example
needs no simulator:

```python
from silk.signals import power_sweep
from silk.metrics import p1db

curve = power_sweep([-30, -20, -10, 0], [-20, -10, 0, 8])
assert p1db(curve) == -5.0
```

Convert sinusoidal peak amplitude A at resistance R to dBm as
`10*log10(A**2/(2*R)/1e-3)`. `Signal.dbm()` instead converts each sample
using `abs(y)**2/R`; it does not infer a sinusoid's RMS from its peak.

For `reference_waveform`, construct a target Signal on your own arrays and
pass it as `goal` to an error norm. Real and complex targets are supported;
this role describes your intent, not a special stamped Signal type.

## What a metric does when it cannot measure

Malformed requests raise `ValueError` with context: an incompatible axis,
an empty input, a backwards window, or a band with insufficient points.
Python calling errors, such as an unknown keyword, can raise `TypeError`.
Read each function's contract for its specific minimum support.

Legitimate but unmeasurable data returns a number or an unavailable result
shape, often with `MetricWarning`. Examples include `nan` for a missing
crossing or unobserved bandwidth, and an empty array for no events.
A dead trace with default spectral reference receives a finite numerical
floor: SNDR is `-10*log10(n_samples/2-1)` dB for an even rectangular record,
SFDR is 0 dB. THD/SNR return `nan` if their required selection is empty.
An explicit `full_scale` changes the reference and can yield a ceiling
instead, as described above.

A point query outside coverage (`value_at`) returns `nan`; an aggregate
over an empty band raises. A legitimately flat passband still has 0 dB
ripple. A nonzero reference divided by a zero stopband or common-mode gain
can produce infinite rejection; zero/zero has no reference and returns `nan`.
`gain_margin` can be `+inf` when no phase crossover is observed and the
endpoint gain is below unity; above-unity endpoints remain unavailable.

`nan` fails ordinary numeric spec comparisons. The failing direction of a
finite value depends on the spec: low SNDR fails a minimum, while high THD
fails a maximum. Neither a numerical floor nor a numerical ceiling is a
measurement of circuit performance.

### Report capture and persistence of `MetricWarning`

`silk.metrics.diagnostics.MetricWarning` is a `UserWarning` with `metric`,
`code`, `message`, `details` and `suggestion` attributes. Details contain
counts, ranges and units relevant to the unavailable measurement. Codes such
as `missing_crossing`, `nan_waveform`, `no_harmonic_bins`,
`unattributable_imd_product`, `zero_gain` and `incomplete_constellation`
distinguish missing support, contamination and advisory conditions.

Normal warning filters apply: ignoring a warning preserves the result;
promoting it to an error interrupts the direct call. During testbench
extraction, warnings gain their output name and are kept in
`SpecReport.metric_warnings`. `store_report()` persists warnings for stored
numeric outputs under `conditions['silk_metric_warnings']` on the output's
simulation-result row. Use `SpecReport.save_json()` to preserve the complete report.
Signal-binding errors propagate as `MetricBindingError`; ambiguous branch
currents also propagate. Other extraction errors become `Error: ...` result
strings.
See [reports](reports.md) for reading and exporting evidence.
See [window, point and waveform-math diagnostics](metrics-transient-math-diagnostics.md)
for the checked-sample scope and warning conditions.

### AC measurement diagnostics

AC warnings distinguish missing unity crossings, unobserved upper roll-off,
missing DC references and contaminated magnitude/phase samples. Point queries
inspect their selected samples; unrelated NaNs elsewhere do not invalidate
those point values. Start stability sweeps well below crossover so phase
unwrapping has an appropriate low-frequency reference.

### Clock and eye-centering diagnostics

Clock warnings distinguish consumed NaNs from too few complete edges/cycles.
A valid zero jitter stays zero. Eye centering reports `centering_unavailable`
or advisory `centering_sensitive`; requested geometry can also report
`insufficient_eye_samples`, `eye_threshold_unbracketed` or
`eye_width_unavailable`. Ignoring these warnings does not make missing
height/width measurements available.

### DC measurement diagnostics

`p1db` reports `nan_power_sweep` for contaminated data and
`no_compression_observed` when no 1 dB reduction is seen. Transition metrics
use `nan_transition_voltage` to identify affected code levels. Extracted
transition diagnostics inspect the input-voltage samples actually selected
for transitions, rather than every ramp sample.

### Noise measurement diagnostics

Noise warnings identify contaminated densities, frequencies outside the
sweep, unavailable corner estimates and nonpositive noise factors.
`integrated_noise` consumes its selected band; an unrelated NaN outside it
does not invalidate the integral. Valid measured zero integrated noise
remains zero without a warning.

## Resolution — read before quoting a number

| Measurement | Practical limit |
|---|---|
| Bandwidth and Q | Existing sweep points set the resolution; increase density around the boundary or peak. |
| Spectral frequencies | One bin is `1/(n_samples*tclk)` Hz; `SFDR.bin_hz` reports this spacing. |
| Clean-tone ratios | Each shared ratio-spectrum bin carries a double-precision epsilon power floor. Exact tones reach a finite numerical ceiling, which depends on amplitude and sample count. |
| Edge timing and jitter | Linear interpolation cannot recover unresolved edge curvature. Edges faster than one timestep can read a timestep-derived rise/fall floor. |
| Settling | Reported on the sample grid, using the first qualifying sample after the last excursion. |
| Integrated noise | Inward-snapped band edges and trapezoidal integration limit accuracy. |
| ADC transitions | Input-ramp spacing sets transition resolution; divide by the LSB to express its effect on INL/DNL. |
| Group delay | A numerical derivative of unwrapped phase needs a sufficiently dense sweep, especially near sharp resonances. |

## Result objects and scalar wrappers

Some measurements return several related outputs. Use scalar wrappers when
you need one float; register the result function directly to expose all
its declared fields together.

| Result function | Declared scalar fields | Scalar wrappers |
|---|---|---|
| `sfdr` | `sfdr_db`, `fundamental_hz`, `spur_hz`, `bin_hz` | `sfdr_db` |
| `eye_diagram` | `height`, `width`, `sampling_instant_s` | `eye_height`, `eye_width` |
| `inl`, `dnl` | `max_abs_lsb` | `inl_max_lsb`, `dnl_max_lsb` |
| `static_errors` | `offset_error_lsb`, `gain_error_lsb`, `full_scale_error_lsb` | wrappers of the same names |
| `evm_constellation` | `evm_percent` | `evm_percent` |

For an existing `tb`, this fragment measures SFDR once and binds a spec to
one declared field:

```python
from silk.metrics import sfdr

tb.add_metric('dynamic', sfdr, sig=tb.probe('vout'), tclk=10e-9)
tb.add_spec('SFDR', 'dynamic.sfdr_db', '>=', 50.0)
```

The report keeps the result object under `dynamic` and adds dotted scalar
keys such as `dynamic.sfdr_db`. An extraction failure populates both the bare
name and declared outputs with the error string. Explicit names must not
collide with dotted outputs. Passing a lambda hides the output declaration.
Arrays and booleans are not automatically expanded into scalar report fields.

### EVM against a named modulation

`evm_percent` and `evm_constellation` sample at `tstart + (k+0.5)*symbol_period`; omitted `tstart` uses the
first input time, and off-grid values are interpolated.

`evm_percent` requires a modulation name: `bpsk`, `qpsk`, `8psk`, `16qam`,
`64qam`, `256qam`, `pam4` or `pam8`. `modulation_grid` returns its unit-RMS
grid. The fit removes one complex gain (scale and rotation), not a common
offset. It alternates nearest-point decisions and least-squares gain fitting
from two starting estimates, selecting the lower residual; this local search
does not guarantee a global optimum.

`evm_constellation` accepts exactly one of `modulation` and `ideal_points`.
Explicit ideal points are used as given, with no gain fit; `reference` requires
that path. Ideal points must be nonzero; `reference` supplies one complex ideal
value per sampled symbol, not point indices, and each must match an ideal point.
With named modulation, `fitted_gain` holds the fitted gain.

Visit the full constellation: a constant nonzero symbol can fit with zero
error while leaving most points untested. `incomplete_constellation` warns
when fitted points lack a sample within half the minimum fitted spacing.
The result remains available. All-zero symbols have no gain to fit and give
`nan` with `zero_gain`. Coverage alone does not prove fit accuracy.

## Signal methods

`Signal` also provides `.mag()`, `.real()`, `.imag()`, wrapped and unwrapped
`.phase_rad()`/`.phase_deg()` variants, `.db10()`, `.db20()` and `.dbm(z0=50.0)`.
The unwrapped names are `.phase_rad_unwrapped()` and `.phase_deg_unwrapped()`.
Magnitude/phase conversions return Signals suitable for `.plot()`.

| Method | Corresponding function or operation |
|---|---|
| `sig.shift(dx)` | `lshift(sig, dx)` |
| `sig.window(x1, x2)` | `window(sig, x1, x2)` |
| `sig.flip()` | `flip(sig)` |
| `sig.interpolate(x_new)` | Real-valued linear resampling; `sample` also accepts complex y and `n=` |
| `sig.derivative()` | `derivative(sig)` |
| `sig.integral()` | `running_integral(sig)` |
| `sig.fft(tclk=..., ...)` | `fft(sig, tclk=..., ...)` |

`interpolate` refuses complex y; interpolate real and imaginary parts
separately or use `sample`. Both accept ascending or descending strictly
monotonic finite source coordinates and clamp targets outside coverage.
`sig.integral()` preserves complex values. Arithmetic supports division of
Signals as well as addition, subtraction and multiplication.

## Schema export — which metrics are in the machine-readable catalog

Use the public `export_metric_schema` function rather than maintaining a
separate list. The command-line form is `python -m silk.signals.schema`.

```python
from silk import export_metric_schema

schema = export_metric_schema()
assert schema['_meta']['format_version'] == '2.0'
assert schema['_meta']['metric_count'] == len(schema) - 1
assert 'average' in schema
```

The catalog contains 80 scalar metrics. Result functions with declared
`outputs=` are registered for Python but omitted from this scalar export.
Signal-returning transforms, raw arrays/bools and closed-form helpers are
also outside the scalar catalog. The `@metric` decorator requires declared
signal kinds and supported annotations; it accepts `float | complex` scalar
returns. Schema eligibility does not make complex spec limits meaningful:
project a metric to a real value before comparing it with a limit.

Catalog entries describe signal slots, parameter defaults, returned units
and parameter units. `''` is dimensionless, `[y]` means the input y unit,
`[x]` the input x unit, and `[y]*sqrt([x])` gives integrated density units
(e.g. V from V/√Hz integrated over Hz). See [schema export](schema-export.md)
for declaration rules and filtering by `produced_by`.

## Function reference

Signatures below show every parameter and default; annotations on inputs
are omitted for readability. `*` marks keyword-only arguments. Return
`np.ndarray` means an array; result-class names denote structured objects.
Units follow the physical input (seconds for time, Hz for frequency) unless
a function explicitly returns dB, dBm, LSB, bits, a ratio or another unit.
The reference covers the public functions exported by `silk.metrics`;
function docstrings provide detailed validation and diagnostic conditions.

### `silk.metrics.waveform`

General tools on any x-axis: crossings, values, extrema, calculus, sampling and transforms.

| Function | Returns | Measurement or transform |
|---|---|---|
| `crossing(sig, threshold=None, *, edge='rising', n=1, start=None, end=None)` | `float` | X value at the n-th crossing of `threshold` in `edge` direction. |
| `crossings(sig, *, threshold=None, edge='rising', start=None, end=None)` | `np.ndarray` | X coordinates, in acquisition order, at which `sig` crosses `threshold` in `edge` direction. |
| `value_at(sig, x)` | `float \| complex` | Linearly interpolated y at the given x. |
| `final_value(sig)` | `float \| complex` | Value at the last acquired sample; does not check settling. |
| `argmin_x(sig, *, start=None, end=None, interpolate=True, refine=False)` | `float` | X value at which y is minimum in the window. |
| `argmax_x(sig, *, start=None, end=None, interpolate=True, refine=False)` | `float` | X value at which y is maximum in the window. |
| `peak(sig, *, start=None, end=None, interpolate=True)` | `float` | Largest-magnitude excursion: `max(\|y\|)` in the window. |
| `peak_to_peak(sig, *, start=None, end=None, interpolate=True)` | `float` | Peak-to-peak excursion: `max(y) − min(y)` in the window. |
| `minimum(sig, *, start=None, end=None, interpolate=True)` | `float` | Minimum y value in the window. |
| `maximum(sig, *, start=None, end=None, interpolate=True)` | `float` | Maximum y value in the window. |
| `derivative(sig)` | `Signal` | Numerical derivative dy/dx as a Signal, central differences. |
| `integral(sig, *, start=None, end=None, interpolate=True)` | `float \| complex` | Trapezoidal definite integral of y over x in the window. |
| `running_integral(sig)` | `Signal` | Cumulative trapezoidal integral as a Signal. |
| `moving_average(sig, *, window)` | `Signal` | Sliding-window time average as a Signal. |
| `clip(sig, *, low=None, high=None)` | `Signal` | Limit the y values of a signal to `[low, high]` (numpy semantics). |
| `window(sig, start, end)` | `Signal` | Restrict a signal to the x-window `[start, end]`. |
| `flip(sig)` | `Signal` | Time-reverse a signal: `y[i] = y_orig[N-1-i]`; x mirrored to match. |
| `lshift(sig, dx)` | `Signal` | Time-shift the signal left by `dx`: `x_new = x − dx`. |
| `rshift(sig, dx)` | `Signal` | Time-shift the signal right by `dx`: `x_new = x + dx`. |
| `sample(sig, *, x_new=None, n=None)` | `Signal` | Resample a signal onto a new x-grid via linear interpolation. |
| `convolve(a, b)` | `Signal` | Convolution of two Signals on a uniform shared dx grid. |
| `cross_correlation(a, b)` | `Signal` | Discrete cross-correlation of two Signals. |

### `silk.metrics.statistics`

Waveform statistics and error norms; histogram is time-weighted and time-only.

| Function | Returns | Measurement or transform |
|---|---|---|
| `average(sig, *, start=None, end=None, interpolate=True)` | `float \| complex` | Time-weighted (or x-weighted) mean: `∫ y dx / (b - a)`. |
| `rms(sig, *, start=None, end=None, interpolate=True)` | `float` | Time-weighted RMS: `√(∫ \|y\|² dx / (b - a))`. |
| `stddev(sig, *, start=None, end=None, interpolate=True)` | `float` | Time-weighted standard deviation: `√(∫ \|y − μ\|² dx / (b - a))`. |
| `variance(sig, *, start=None, end=None, interpolate=True)` | `float` | X-weighted variance about the mean, equal to stddev squared. |
| `histogram(sig, *, bins, start=None, end=None)` | `HistogramResult` | Fraction of time `sig` spends in each y-value bin over `[start, end]`. |
| `error_norm(sig, goal, *, kind='rms', start=None, end=None)` | `float` | Norm of (sig − goal). |
| `relative_error_norm(sig, goal, *, kind, floor, start=None, end=None)` | `float` | Dimensionless norm of `\|sig - goal\| / \|goal\|`. |

### `silk.metrics.timing`

Time-waveform edges, delay, pulse, duty, period, frequency, response and hysteresis.

| Function | Returns | Measurement or transform |
|---|---|---|
| `rise_time(sig, low=0.1, high=0.9, *, initial=None, final=None, n=1, start=None, end=None)` | `float` | Rise time from the `low` to the `high` threshold of a rising edge. |
| `fall_time(sig, low=0.1, high=0.9, *, initial=None, final=None, n=1, start=None, end=None)` | `float` | Fall time from the `high` to the `low` threshold of a falling edge. |
| `slew_rate(sig, low=0.1, high=0.9, *, edge='rising', initial=None, final=None, n=1, start=None, end=None)` | `float` | Average slew rate between the `low` and `high` thresholds. |
| `rise_times(sig, low=0.1, high=0.9, *, initial=None, final=None, start=None, end=None)` | `np.ndarray` | Every rise time: one per complete rising edge, in time order. |
| `fall_times(sig, low=0.1, high=0.9, *, initial=None, final=None, start=None, end=None)` | `np.ndarray` | Every fall time: one per complete falling edge, in time order. |
| `slew_rates(sig, low=0.1, high=0.9, *, edge='rising', initial=None, final=None, start=None, end=None)` | `np.ndarray` | Every slew rate: one positive magnitude per complete edge, in time order. |
| `settling_time(sig, initial, target=None, tol=0.01, start=0.0)` | `float` | Elapsed settling time using the last-out-of-band convention. |
| `propagation_delay(sig_in, sig_out, *, in_threshold=None, out_threshold=None, in_edge='rising', out_edge=None, in_n=1, out_n=1, start=None, end=None)` | `float` | Delay from the selected input trigger to an output crossing at or after it. |
| `propagation_delays(sig_in, sig_out, *, in_threshold=None, out_threshold=None, in_edge='rising', out_edge=None, out_n=1, start=None, end=None)` | `np.ndarray` | Every delay: one per `in_edge` crossing of `sig_in` in the window. |
| `overshoot(sig, initial, target)` | `float` | Peak excursion beyond target divided by the absolute step swing. |
| `undershoot(sig, initial, target)` | `float` | Opposite-direction excursion after reaching target, divided by the absolute step swing. |
| `on_time(sig, threshold=None, *, start=None, end=None)` | `float` | Estimated time above `threshold` by trapezoidal integration of thresholded samples; window bounds are interpolated, threshold crossings are not inserted. |
| `off_time(sig, threshold=None, *, start=None, end=None)` | `float` | Estimated time at or below `threshold`: the window duration − `on_time`, with the same sampled integration convention. |
| `pulse_width(sig, threshold=None, *, polarity='positive', n=1, start=None, end=None)` | `float` | Width in seconds of the `n`-th complete pulse in `[start, end]`. |
| `pulse_widths(sig, threshold=None, *, polarity='positive', start=None, end=None)` | `np.ndarray` | Widths in seconds of every complete pulse in `[start, end]`. |
| `average_duty(sig, threshold=None, *, edge='rising', start=None, end=None)` | `float` | Fraction of time above `threshold` over the full cycles in `[start, end]`. |
| `cycle_duty(sig, threshold=None, *, edge='rising', start=None, end=None)` | `Signal` | Per-cycle duty as a waveform: fraction high versus time. |
| `average_period(sig, threshold=None, *, edge='rising', start=None, end=None)` | `float` | Mean period in seconds over `[start, end]`. |
| `average_frequency(sig, threshold=None, *, edge='rising', start=None, end=None)` | `float` | `1 / average_period` in hertz — same core, same arguments. |
| `cycle_period(sig, threshold=None, *, edge='rising', start=None, end=None)` | `Signal` | Per-cycle period as a waveform: seconds versus time. |
| `cycle_frequency(sig, threshold=None, *, edge='rising', start=None, end=None)` | `Signal` | `1 / cycle_period` as a waveform: hertz versus time. |
| `hysteresis(sig_in_rising, sig_out_rising, sig_in_falling, sig_out_falling, *, out_threshold=None, n_rising=1, n_falling=1, start=None, end=None)` | `float` | Hysteresis = v_trip_rising − v_trip_falling. |

### `silk.metrics.jitter`

Clock jitter, TIE, long-term stability and Allan deviation.

| Function | Returns | Measurement or transform |
|---|---|---|
| `period_jitter(clock, *, threshold=None, edge='rising', start=None, end=None, reference='mean', cycles=None)` | `float` | RMS period jitter in seconds: the RMS of `period_deviation`. |
| `period_jitter_pp(clock, *, threshold=None, edge='rising', start=None, end=None, reference='mean', cycles=None)` | `float` | Peak-to-peak period jitter in seconds: `max - min` of `period_deviation`. |
| `cycle_to_cycle_jitter(clock, *, threshold=None, edge='rising', start=None, end=None, reference='mean', cycles=None)` | `float` | RMS cycle-to-cycle jitter in seconds: the RMS of `cycle_to_cycle_difference`. |
| `cycle_to_cycle_jitter_pp(clock, *, threshold=None, edge='rising', start=None, end=None, reference='mean', cycles=None)` | `float` | Peak-to-peak cycle-to-cycle jitter in seconds: `max - min` of `cycle_to_cycle_difference`. |
| `period_deviation(clock, *, threshold=None, edge='rising', start=None, end=None, reference='mean', cycles=None)` | `Signal` | Per-cycle period deviation as a waveform: seconds versus time. |
| `cycle_to_cycle_difference(clock, *, threshold=None, edge='rising', start=None, end=None, reference='mean', cycles=None)` | `Signal` | Per-cycle cycle-to-cycle difference as a waveform: seconds versus edge time. |
| `crossing_jitter_rms(sig, *, period, threshold=None, edge='rising', start=None, end=None)` | `float` | Sample standard deviation of wrapped crossing offsets from their circular mean. |
| `tie(clock, *, units='s', reference='fit', ideal_period=None, phase=None, threshold=None, edge='rising', start=None, end=None)` | `Signal` | Time Interval Error: deviation of each clock edge from its ideal time. |
| `long_term_jitter(clock, *, units='s', reference='fit', ideal_period=None, phase=None, threshold=None, edge='rising', start=None, end=None)` | `float` | Sample standard deviation of TIE, ddof=1. |
| `allan_deviation(clock, *, taus=None, threshold=None, edge='rising', start=None, end=None)` | `Signal` | Non-overlapping Allan deviation σ_y(τ) for a clock waveform. |

### `silk.metrics.eye`

Serial-data eye folds, height and width.

| Function | Returns | Measurement or transform |
|---|---|---|
| `eye_diagram(sig, *, period, n_traces=None, auto_center=False, threshold=None)` | `EyeResult` | Fold original transient samples, optionally aligning their crossing phase. |
| `eye_height(sig, *, period, threshold, n_traces=None, auto_center=True)` | `float` | Geometric opening height at an explicit threshold from `eye_diagram`. |
| `eye_width(sig, *, period, threshold, n_traces=None, auto_center=True)` | `float` | Opening width in seconds at an explicit threshold from `eye_diagram`. |

### `silk.metrics.power`

Power and energy from paired voltage and current waveforms.

| Function | Returns | Measurement or transform |
|---|---|---|
| `average_power(sig_v, sig_i)` | `float` | Time-averaged instantaneous power, ⟨v(t)·i(t)⟩ over the trace. |
| `dynamic_energy_per_cycle(sig_v, sig_i, period)` | `float` | Energy per clock cycle = (∫v·i dt over whole cycles) / cycle_count. |

### `silk.metrics.spectral`

Clock-sampled spectra, harmonics, distortion, SNR, SNDR, SFDR and coherence.

| Function | Returns | Measurement or transform |
|---|---|---|
| `fft(sig, *, tclk, tstart=0.0, n_samples=None, window='rectangular', scaling=None)` | `Signal` | Spectrum of a transient Signal, returned as a Signal. |
| `psd(sig, *, tclk, tstart=0.0, n_samples=None, window='hann')` | `Signal` | Single-sided power spectral density, returned as a Signal. |
| `welch_psd(sig, *, tclk, tstart=0.0, n_samples=None, segments=8, overlap=0.5, window='hann')` | `Signal` | Welch's segment-averaged single-sided PSD. |
| `spectral_power(sig, f1_hz, f2_hz, *, tclk, tstart=0.0, n_samples=None, window='hann')` | `float` | Total power in the band `[f1_hz, f2_hz]`, V² (assuming volts). |
| `coherence(a, b, *, tclk, tstart=0.0, n_samples=None, segments=8, overlap=0.5, window='hann')` | `Signal` | Magnitude-squared coherence `γ²(f) = \|S_xy\|² / (S_xx · S_yy)`. |
| `harmonic(sig, k, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None)` | `complex` | Complex value of the k-th harmonic FFT bin. |
| `harmonic_freqs(sig, *, tclk, tstart=0.0, n_samples=None, n=10, observed=False, fundamental_hz=None)` | `np.ndarray` | True harmonic frequencies; observed=True returns folded bin centres. |
| `harmonic_level_dbc(sig, *, n, tclk, tstart=0.0, n_samples=None, fundamental_hz=None)` | `float` | Harmonic `n >= 2` level relative to the fundamental, in dBc. |
| `harmonic_levels_db(sig, *, tclk, tstart=0.0, n_samples=None, n_harmonics=10, fundamental_hz=None)` | `np.ndarray` | Per-harmonic level relative to the fundamental, dB. |
| `thd(sig, *, tclk, tstart=0.0, n_samples=None, n_harmonics=5, fundamental_hz=None, guard_bins=0, window='rectangular')` | `float` | Total harmonic distortion in dB. |
| `thd_percent(sig, *, tclk, tstart=0.0, n_samples=None, n_harmonics=5, fundamental_hz=None, guard_bins=0, window='rectangular')` | `float` | THD as a percentage of the fundamental amplitude, with `thd`'s spectrum options. |
| `noise_floor_dbfs(sig, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None, guard_bins=0, window='rectangular', full_scale=None)` | `float` | Average non-fundamental, non-DC bin power in dBFS. |
| `snr(sig, *, tclk, tstart=0.0, n_samples=None, n_harmonics=10, fundamental_hz=None, guard_bins=0, window='rectangular', full_scale=None)` | `float` | Signal-to-noise ratio in dB, *excluding* harmonic distortion. |
| `sndr(sig, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None, guard_bins=0, window='rectangular', full_scale=None)` | `float` | Signal-to-noise-and-distortion ratio (dB). |
| `sinad(sig, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None, guard_bins=0, window='rectangular', full_scale=None)` | `float` | SINAD (dB). |
| `sfdr(sig, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None, guard_bins=0, window='rectangular')` | `SFDR` | Spurious-free dynamic range. |
| `sfdr_db(sig, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None, guard_bins=0, window='rectangular')` | `float` | Scalar SFDR in dB, from `sfdr`, with its spectrum options. |

### `silk.metrics.rf`

Power-sweep compression, two-tone intercept/intermodulation, mixing and I/Q EVM.

| Function | Returns | Measurement or transform |
|---|---|---|
| `p1db(sig)` | `float` | Input 1-dB compression point, in dBm. |
| `iip3_dbm(out_sig, in_sig, *, tclk, f1_hz, f2_hz, r_ref=50.0, tstart=0.0, n_samples=None)` | `float` | Input-referred 3rd-order intercept point in dBm: `OIP3 - gain`. |
| `oip3_dbm(sig, *, tclk, f1_hz, f2_hz, r_ref=50.0, tstart=0.0, n_samples=None)` | `float` | Output-referred 3rd-order intercept point in dBm from one two-tone spectrum. |
| `imd3_db(sig, *, tclk, f1_hz, f2_hz, tstart=0.0, n_samples=None)` | `float` | 3rd-order intermodulation distortion in dB, relative to the larger fundamental. |
| `mix_product(sig, m, n, *, f1_hz, f2_hz, tclk, tstart=0.0, n_samples=None)` | `complex` | Complex value of the FFT bin at frequency `m·f1 + n·f2`. |
| `evm_constellation(i_sig, q_sig, *, symbol_period, modulation=None, ideal_points=None, tstart=None, n_symbols=None, reference=None)` | `EVMResult` | Error Vector Magnitude from baseband I/Q transients. |
| `evm_percent(i_sig, q_sig, *, symbol_period, modulation, tstart=None, n_symbols=None)` | `float` | RMS error vector magnitude in percent, against a named modulation. |
| `modulation_grid(modulation)` | `np.ndarray` | Unit-RMS ideal constellation for a named modulation. |

### `silk.metrics.converter`

Code-transition linearity, ENOB from sampled waveforms and scalar aperture-jitter limit.

| Function | Returns | Measurement or transform |
|---|---|---|
| `extract_code_transitions(v_in, code)` | `Signal` | Build a code→voltage transition Signal from a slow-ramp capture. |
| `inl(sig, method='endpoint')` | `INL` | Integral non-linearity per code, in LSBs. |
| `inl_max_lsb(sig, method='endpoint')` | `float` | Maximum absolute INL in LSBs from `inl`. |
| `dnl(sig)` | `DNL` | Differential non-linearity per step, in LSBs. |
| `dnl_max_lsb(sig)` | `float` | Maximum absolute DNL in LSBs from `dnl`. |
| `missing_codes(sig)` | `np.ndarray` | Codes (in `sig.x`) whose computed DNL is at most −1. |
| `missing_code_count(sig)` | `float` | Number of missing codes from `missing_codes`, as a float. |
| `is_monotonic(sig)` | `bool` | True if the transfer curve is strictly increasing. |
| `non_monotonic_steps(sig)` | `float` | Count non-increasing transition steps plus recorded code reversals. |
| `static_errors(sig, *, full_scale, n_bits, v_low=0.0)` | `StaticErrors` | Offset, gain and full-scale error from code transitions, in LSBs. |
| `offset_error_lsb(sig, *, full_scale, n_bits, v_low=0.0)` | `float` | Offset error in LSBs from `static_errors`. |
| `gain_error_lsb(sig, *, full_scale, n_bits, v_low=0.0)` | `float` | Gain error in LSBs from `static_errors`. |
| `full_scale_error_lsb(sig, *, full_scale, n_bits, v_low=0.0)` | `float` | Full-scale error in LSBs from `static_errors`. |
| `enob(sig, *, tclk, tstart=0.0, n_samples=None, fundamental_hz=None, guard_bins=0, window='rectangular', full_scale=None)` | `float` | Effective number of bits = (SNDR − 1.76) / 6.02. |
| `aperture_jitter_snr_db(f_in, sigma_t)` | `float` | SNR limit from sampling-clock aperture jitter. |

### `silk.metrics.fom`

Figures of merit from measured scalars, including PA efficiency.

| Function | Returns | Measurement or transform |
|---|---|---|
| `walden_fom(power_w, fs_hz, enob_bits)` | `float` | Walden figure of merit, `P / (2**ENOB * fs)`. |
| `schreier_fom(sndr_db, bw_hz, power_w)` | `float` | Schreier figure of merit, `SNDR + 10*log10(BW/P)`, in dB. |
| `iip3_extrapolation(p_in_dbm, p_fund_dbm, p_imd3_dbm)` | `float` | Extrapolate the input third-order intercept from one two-tone measurement. |
| `pae(p_rf_out_w, p_rf_in_w, p_dc_w)` | `float` | Power-added efficiency, `(P_RF_out - P_RF_in) / P_DC`. |
| `dc_to_rf_efficiency(p_rf_w, p_dc_w)` | `float` | Drain efficiency, `P_RF_out / P_DC`. |

### `silk.metrics.ac`

Small-signal frequency response: gain, bandwidth, margins, rejection and filter shape.

| Function | Returns | Measurement or transform |
|---|---|---|
| `dc_gain_dB(sig)` | `float` | Magnitude in dB at the lowest sweep frequency. |
| `peak_gain_dB(sig)` | `float` | Peak magnitude in dB anywhere in an AC frequency-response signal. |
| `gain_at_freq(sig, freq, *, interpolate=True)` | `float` | Linear gain magnitude at *freq*, interpolated between sweep points. |
| `phase_at_freq(sig, freq, *, interpolate=True)` | `float` | Phase in degrees at *freq*, interpolated between sweep points. |
| `bandwidth_3dB(sig, *, reference='dc')` | `float` | 3 dB bandwidth: highest frequency where `\|gain\|` >= reference / sqrt(2). |
| `unity_gain_freq(sig)` | `float` | Frequency where `\|gain\|` crosses 1 (0 dB), interpolated. |
| `gain_bw_product(sig)` | `float` | Gain × −3 dB bandwidth. |
| `phase_margin(sig)` | `float` | Phase margin from the nearest stored unwrapped-phase sample at the lowest unity crossing, mapped to [−180°, +180°); multiple unity crossings warn. |
| `gain_margin(sig)` | `float` | Gain margin at the phase crossover whose margin is closest to 0 dB. |
| `gain_margins(sig)` | `list of (frequency, margin) pairs` | Every gain margin of the loop, one per phase crossover. |
| `group_delay(sig)` | `Signal` | Group delay τ_g(ω) = −dφ/dω of a complex AC response, per frequency. |
| `passband_ripple_db(sig, f_pass)` | `float` | Passband peak-to-valley magnitude ratio in dB. |
| `stopband_attenuation_db(sig, f_stop)` | `float` | Stopband attenuation = passband peak / stopband peak, in dB. |
| `q_from_peak(sig)` | `float` | Q-factor of a resonant peak: f_peak / Δf_3dB. |
| `psrr_db(v_supply, v_out)` | `Signal` | Power Supply Rejection Ratio in dB at each frequency. |
| `cmrr_dB(vdiff, vcm, freq=None)` | `float` | CMRR in dB: 20*log10(`\|diff_gain\|` / `\|cm_gain\|`) using each sweep's lowest frequency by default, or its independently nearest point to `freq`; use matching grids when an exact common frequency matters. |

### `silk.metrics.noise`

Noise-density queries and integration, noise figure and scalar noise formulas.

| Function | Returns | Measurement or transform |
|---|---|---|
| `noise_figure(snr_in_db, snr_out_db)` | `float` | Noise figure from input minus output signal-to-noise ratio. |
| `noise_figure_from_factor(f)` | `float` | Convert linear noise factor to noise figure, `10*log10(f)`. |
| `noise_figure_at_freq(noise_in, freq, rs, temp_sim_k, *, temp_ref_k=290.0, interpolate=True)` | `float` | Spot noise figure (dB) at *freq* from the input-referred noise density. |
| `noise_figure_vs_freq(noise_in, rs, temp_sim_k, *, temp_ref_k=290.0)` | `Signal` | Spot noise figure in dB at every input frequency. |
| `band_noise_figure(noise_in, noise_out, f1, f2, rs, temp_sim_k, *, temp_ref_k=290.0)` | `float` | Average (band) noise figure (dB) over `[f1, f2]`, gain-weighted. |
| `cascade_noise_figure(stage_nfs_db, stage_gains_db)` | `float` | Total noise figure of a cascade via the Friis formula. |
| `input_referred_noise(output_noise_v, gain_vv)` | `float` | Refer an output RMS noise total to the input using one flat gain. |
| `integrated_noise(sig, f1, f2)` | `float` | RMS integrated noise between f1 and f2 Hz. |
| `noise_density_at_freq(sig, freq, *, interpolate=True)` | `float` | Noise spectral density at *freq*, interpolated between sweep points. |
| `one_over_f_corner(noise_sig)` | `float` | Frequency at which 1/f noise equals the white-noise floor. |
| `kT_over_C(c_farads, temp_k=300.0)` | `float` | RMS sampled capacitor noise, `sqrt(k*T/C)`. |
| `sum_uncorrelated(*sigmas)` | `float` | Combine independent RMS noise contributions by root sum of squares. |
