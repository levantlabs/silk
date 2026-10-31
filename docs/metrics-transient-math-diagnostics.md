# Window, point and waveform-math diagnostics

The window, point and waveform-math metrics in `silk.metrics.waveform`,
statistics in `silk.metrics.statistics`, and response measurements in
`silk.metrics.timing` emit
`silk.metrics.diagnostics.MetricWarning` when samples used in a calculation
contain NaNs, a point query falls outside the trace, or an undershoot
measurement has no post-transition region. The warning explains
the problem alongside the numeric result; it does not replace the result.
Waveform operations retain their finite samples and result shapes. Invalid
requests generally raise `ValueError`. For `error_norm`, NaN checking precedes
window validation, so contaminated data can return `nan` before invalid window
bounds are checked.

| Metrics | Observed condition |
|---|---|
| `average`, `rms`, `stddev`, `variance`, `peak`, `peak_to_peak`, `minimum`, `maximum`, `argmin_x`, `argmax_x`, `integral` | `nan_waveform`: the selected window contains NaNs in the values consumed by the metric (RMS uses magnitudes; ordering metrics require a real projection). |
| `overshoot`, `undershoot` | `nan_waveform`: the supplied waveform contains NaNs. `undershoot` also emits `target_not_reached` and returns `nan` if the trace never reaches the target. A NaN initial or target value is not diagnosed as waveform contamination. |
| `average_power`, `dynamic_energy_per_cycle` | `nan_power`: the integrated voltage-current products contain NaNs. Per-cycle energy checks complete cycles, including an interpolated final boundary when needed. |
| `value_at` | `point_outside_trace`: a requested point lies outside available coverage; `nan_waveform`: the exact queried sample or an interpolation bracket contains NaNs. |
| `running_integral`, `derivative`, `convolve`, `cross_correlation` | `nan_waveform`: consumed operands contain NaNs, leaving affected output samples unavailable. |
| `error_norm`, `relative_error_norm` | `nan_waveform`: compared samples contain NaNs, including the reference resampled onto the measured grid. Checking covers the full measured trace even when `start` and `end` select a smaller window. The payload identifies the comparison-axis range and units. |

Each warning identifies the originating function, a reason code, factual
sample counts and axis ranges, units and a suggested next action. Two-operand
waveform operations issue one warning containing both operands' counts and
ranges. Generic axes include their label and use `s` for time and `Hz` for
frequency; other axis units are `None` because Signal does not supply that
metadata. Sample-value units come from the Signal's `ylabel`. Outside-point
warnings report available coverage and zero consumed samples.

Checked samples depend on the metric; the table above identifies the scope. This example
needs no simulator:

```python
import warnings

import numpy as np
from silk import Signal
from silk.metrics import average, derivative, running_integral, value_at
from silk.metrics.diagnostics import MetricWarning

def trace(values):
    return Signal('vout', 'voltage', 'time', 'V',
                  np.arange(len(values), dtype=float), np.asarray(values))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always', MetricWarning)
    assert average(trace([1, 1, 1, np.nan]), start=0, end=2) == 1
    assert value_at(trace([1, np.nan]), 0) == 1
    np.testing.assert_array_equal(running_integral(trace([np.nan])).y, [0])
    assert not caught

    np.testing.assert_array_equal(derivative(trace([1, np.nan, 1])).y,
                                  [np.nan, 0, np.nan])
    assert len(caught) == 1
    assert caught[0].message.code == 'nan_waveform'
```

The windowed average excludes the final NaN. The exact point query does not
use its neighbour. A one-sample running integral starts at zero and has no
interval to integrate. The derivative uses neighbouring samples, so its centre
value is finite while both endpoints are NaN.

Unused reference samples remain quiet. Partial-cycle tails remain quiet unless
a sample is needed to interpolate the last complete-cycle boundary. Transforms
that select, move or resample values (`window`, `clip`, `flip`, `lshift`,
`sample`) do not emit these warnings. A zero or infinite result alone does
not trigger the NaN warnings described here.

Python warning filters apply normally: ignoring `MetricWarning` preserves the
result, and promoting it to an error raises the warning at the measurement
boundary. During testbench evaluation, warnings from an output extractor are
attributed to that output and collected in `SpecReport.metric_warnings`. Direct
calls raise when warnings are promoted to errors; testbench evaluation keeps the
warning record and records an `Error: ...` metric result. Ignoring display does
not remove the report record.
`store_report()` persists warnings for stored numeric outputs; standalone report
JSON preserves the complete warning mapping.
