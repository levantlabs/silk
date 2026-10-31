# ngspice `.measure` and silk metrics

Use this catalog to choose a `silk.metrics` function when comparing a
measurement with ngspice. Matching the formula is only the first step:
thresholds, edge selection, window boundaries and interpolation must match too.
The comparisons here use real-valued waveforms unless stated otherwise.

## Measurement forms

In the table, `sig` is a `Signal` with the same samples as the simulator
measurement. Import the functions with `import silk.metrics as m`.

| ngspice form | silk equivalent | Conditions and differences |
| --- | --- | --- |
| `WHEN v(out)=level RISE=n` | `m.crossing(sig, level, edge='rising', n=n)` | `FALL` uses `'falling'`; `CROSS` uses `'any'`. `n=1` selects the first crossing; `n=-1` selects the last, corresponding to `LAST`. |
| `WHEN v(a)=v(b)` | Crossings of the real difference signal | Compute `a - b`, then search at threshold zero. |
| `FIND v(out) AT=t` | `m.value_at(sig, t)` | silk interpolates linearly in the independent variable; it returns `nan` with `MetricWarning` outside the trace. |
| `FIND v(out) WHEN v(in)=level` | `m.value_at(out, m.crossing(inp, level))` | Select the same edge and occurrence in the crossing call. |
| `TRIG ... TARG ...` | `m.propagation_delay(inp, out, in_threshold=level, out_threshold=level)` | silk counts output crossings at or after the selected input crossing. ngspice can select an earlier target and return a negative delay. See [delay pairing](#delay-pairing). |
| `AVG` | `m.average(sig, start=a, end=b, interpolate=False)` | Weighted by the independent variable, not the arithmetic mean of samples. Snapped boundaries match ngspice's boundary selection. |
| `MIN`, `MAX`, `PP` | `m.minimum`, `m.maximum`, `m.peak_to_peak` | Use `start=a, end=b, interpolate=False` to retain only samples inside the window. |
| `MIN_AT`, `MAX_AT` | `m.argmin_x`, `m.argmax_x` | Return the coordinate of the extremum; use `interpolate=False` for snapped boundaries. |
| `RMS`, `INTEG` | `m.rms`, `m.integral` | Default `interpolate=True` inserts window endpoints. The integration rules still differ; see [windows](#windows-and-integration). |
| `PARAM` expression | Python arithmetic on metric results | Evaluate the component measurements first. |
| `DERIV` | `m.derivative(sig)`, then `m.value_at` if needed | The ngspice 42 comparison fixture reports this measurement function as unsupported. It supplies no derivative reference value. |
| `ERR`, `ERR1`, `ERR2`, `ERR3` | No direct mapping here | No reference comparison is supplied by the fixtures. |

Rise and fall times can be composed from two threshold crossings or a
`TRIG`/`TARG` pair on the same waveform. `m.rise_time(sig)` and
`m.fall_time(sig)` default to the 10% and 90% levels of the observed swing.
Use explicit levels on the ngspice side to match that swing.

Other measurements, including settling time, overshoot, jitter, spectral
metrics, ADC linearity and figures of merit, need more than a single
`.measure` primitive. Power can be composed by measuring a computed voltage
-times-current vector; it has no dedicated entry in this table.

## Crossings and qualifiers

silk linearly interpolates threshold crossings. A rising segment qualifies
when `a < level <= b`; a falling segment qualifies when `a >= level > b`.
A trace starting exactly on the level does not by itself supply a crossing.
Do not assume identical edge counts on exact plateaus or threshold ties
without checking the actual sampled waveform.

`m.crossings(sig, threshold=level, edge='any')` returns an array of all
crossings; `m.crossing` returns one coordinate. A missing requested crossing
returns `nan` with `MetricWarning`. Invalid requests, such as `n=0` or a
backwards window, raise `ValueError`.

`start` and `end` bound crossing coordinates inclusively. They are also
available on `propagation_delay` and the windowed reductions in the table.
They are silk's coordinate bounds, rather than a general parser for ngspice's
`TD`, `FROM` and `TO` qualifiers. In particular, do not translate an AC
`TD` qualifier into a silk frequency bound. For a single DC sweep the same
axis-based primitives work on the sweep coordinate. Split nested sweeps into
separate monotonic traces before measuring them.

## Windows and integration

The windowed reductions default to `interpolate=True`: a boundary between
samples adds a linearly interpolated endpoint inside the trace. Setting
`interpolate=False` keeps only existing samples inside the requested window.
A reduction needs at least two resulting samples; an empty, backwards or
undersampled window raises `ValueError`.

ngspice's `AVG`, `MIN`, `MAX`, `MIN_AT`, `MAX_AT` and `PP` use inside samples.
Its `RMS` and `INTEG` interpolate window boundaries. silk uses trapezoidal
integration; ngspice's RMS/integral implementation can combine Simpson and
trapezoidal rules. Matching endpoints therefore does not guarantee identical
answers for curved signals. Choose tolerances for the sample spacing and
curvature, and use on-grid boundaries when isolating integration error from
boundary selection.

This simulator-free example shows the difference on a linear ramp:

```python
import numpy as np
from silk.signals import Signal
import silk.metrics as m

sig = Signal('ramp', 'voltage', 'time', 'V',
             np.array([0., 1., 2., 3.]), np.array([0., 1., 2., 3.]))
assert m.integral(sig, start=0.5, end=2.5) == 3.0
assert m.integral(sig, start=0.5, end=2.5, interpolate=False) == 1.5
assert m.average(sig, start=0.5, end=2.5) == 1.5
assert m.crossing(sig, 1.5) == 1.5
assert m.value_at(sig, 1.5) == 1.5
```

## Delay pairing

`propagation_delay` defaults to rising input and output edges, with each
threshold set to that signal's own mid-swing. For an inverting stage, select
opposite edges explicitly:

```python
# Continue the simulator-free example above.
out = Signal('out', 'voltage', 'time', 'V',
             sig.x, np.array([3., 3., 0., 0.]))
delay = m.propagation_delay(sig, out, in_threshold=1.0,
                            out_threshold=1.5,
                            in_edge='rising', out_edge='falling')
assert delay == 0.5
```

`in_n` selects the input crossing; `out_n` selects an output crossing counted
from that trigger, including a simultaneous crossing. `start`/`end` bound
both searches. An output glitch before the input trigger is ignored. A
missing trigger or following target returns `nan` with `MetricWarning`.

ngspice searches the `TRIG` and `TARG` clauses independently. To compare a
waveform with early output glitches, gate the target search with a suitable
`TD`, or compare explicitly selected crossing coordinates rather than assuming
the default pairing rules agree.

## AC measurements

| ngspice composition | silk metric | Comparison convention |
| --- | --- | --- |
| `WHEN vdb(out)=reference_dB-3.0103` | `m.bandwidth_3dB(sig)` | Default `reference='dc'` uses the lowest-frequency magnitude; `'peak'` uses the largest magnitude. silk returns NaN with `upper_rolloff_not_observed` if the final point remains at/above the threshold; otherwise it returns the highest grid frequency still above the threshold, with no crossing interpolation. On a monotonic roll-off this can be one grid interval below an interpolated crossing. |
| `WHEN vm(out)=1` | `m.unity_gain_freq(sig)` | silk interpolates linear magnitude against log-frequency; ngspice interpolates against linear frequency. Dense sweeps reduce the difference. |
| `FIND vp(out) WHEN vdb(out)=0` | `m.phase_margin(sig)` | The ngspice result is phase in radians, not phase margin. Convert it with `(math.degrees(phi_rad) % 360.0) - 180.0`; for phases between -180 and 0 degrees this equals `180.0 + math.degrees(phi_rad)`. silk reads unwrapped phase at the sample nearest its unity-gain frequency and returns margin on the standard phase branch. Allow for both frequency interpolation and phase sampling. |
| Phase-crossover search | `m.gain_margin(sig)`, `m.gain_margins(sig)` | silk unwraps phase before searching. A literal `WHEN vp(out)=-pi` can miss a crossover at a wrapped phase discontinuity; it is not a reliable reference recipe. |

Use projections for scalar waveform measurements; pass the complex linear loop
gain to phase/gain-margin metrics, not its magnitude, dB or phase-only projection.
For complex signals, choose the projection deliberately. `average`, `integral`
and `value_at` preserve complex values; `rms` uses magnitude. Ordered reductions
and singular `crossing` require a real projection; plural `crossings` measures
magnitude for complex input. To compare with ngspice's
`vm`, `vr`, `vi`, `vp` or `vdb` measurement, construct a real-valued signal
with the corresponding magnitude, real part, imaginary part, phase or dB
values. Wrapped and unwrapped phase are different inputs.

## Reproducible comparisons

The repository contains these reference materials:

- `silk/metrics/waveform.py`, `silk/metrics/statistics.py`,
  `silk/metrics/timing.py` and `silk/metrics/ac.py`: current signatures,
  validation and interpolation contracts.
- `tests/fixtures/rawfiles/gen_e_meas.py`: transient and AC measurement decks
  and functions that capture ngspice results.
- `tests/fixtures/rawfiles/e_ng_tran_meas.json` and `e_ng_ac_meas.json`:
  captured reference values paired with the corresponding rawfiles.
- `tests/metrics/test_rawfile_realism.py`: comparisons with those values,
  including tolerances for the AC conventions above. Its slow regeneration
  test reruns the measurement decks and checks the saved values.

The transient comparisons cover rise/fall time, delay, crossings, point
queries and windowed reductions. The AC comparisons cover bandwidth,
unity-gain frequency and phase margin. These are measured comparisons for
those fixtures, rather than a promise that every `.measure` expression or
simulator version uses identical conventions.
