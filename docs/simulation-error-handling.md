# Simulation error handling

Catch `SimulationError` when you want to handle a failed simulation. Catch its
`SimulationAborted` subclass when you also need structured information about
missing, empty or incomplete results. `ConvergenceError` is a subclass of
`SimulationAborted` for failures whose diagnostics name a convergence problem.
All three are available from `silk.exceptions`.

A simulator can exit successfully even when an analysis fails. silk checks the
results as well as the process exit status, so a zero exit code alone does not
make a simulation successful.

## What silk checks

For the built-in ngspice and Xyce backends, silk checks:

1. **The process succeeded.** A non-zero exit status fails. Xyce also needs
   its `End of Xyce` success banner in the captured transcript.
2. **The expected output exists and contains data.** Missing output, a rawfile
   with no data rows, or ngspice's synthetic `constants` plot without analysis
   data fails. Xyce noise uses a separate CSV file, which is checked for the
   required columns and data rows.
3. **A supported analysis reached its final grid point.** For requests silk
   can verify from its own generated and executed deck, transient, DC, AC and
   noise results must reach the expected endpoint. Discrete sweeps must also
   contain at least the expected number of points.

The endpoint tolerance is `1e-7` times the largest of the requested start
magnitude, stop magnitude and span. For discrete sweeps it is capped at a
quarter of the final grid interval. The expected endpoint follows the
simulator's grid, which can differ from the requested stop. Sub-decade ngspice
DEC AC sweeps use a version-specific grid; if the version is unknown, that
request receives no endpoint check.

Endpoint checks require an unambiguous association between the emitted request
and its results. External or custom decks retain the data-presence check but
receive no endpoint verdict. The same limit applies to opaque custom emission,
caller control commands, repeated analyses or ambiguous repeated plots,
continuation cards, `.step` and nested DC sweeps. See
[analysis completion](known_issues.md#analysis-completion) for the full boundary.

Log messages help explain a failure. They do not independently turn a
successful result into a failed run: a warning about a singular matrix during
a run that succeeds is not enough to raise `ConvergenceError`.

## Catching failures

In application code, put your `module.simulate()` or `module.run_simulator()`
call inside a `try` block. Catch the most specific exception first if you want
to handle convergence failures separately.

For registered tests, catch failures around `tb.run()` too. Declared corner
and Monte Carlo plans retain individual failures in `.failed` and raise when
every run fails. Preflight refusals still raise.

The following standalone example creates an exception to demonstrate the
handlers without requiring a simulator or PDK:

```python
from silk.exceptions import ConvergenceError, SimulationAborted, SimulationError

try:
    raise SimulationAborted(
        "No analysis data was produced",
        simulator="ngspice",
        data_rows=0,
    )
except ConvergenceError as err:
    print(f"Convergence failure: {err.matched_lines}")
except SimulationAborted as err:
    print(f"Simulation aborted: {err.reason or str(err)}")
    if err.log_path:
        print(f"See {err.log_path}")
except SimulationError as err:
    print(f"Simulation failed: {err}")
```

Output:

```text
Simulation aborted: No analysis data was produced
```

`SimulationAborted` provides these fields:

| Field | Meaning |
| --- | --- |
| `simulator` | Backend name, such as `'ngspice'` or `'xyce'` |
| `reason` | Diagnostic explanation, or `None`; use the exception message for the full failure description |
| `data_rows` | `0` for empty analysis data, the available axis length for an incomplete analysis, or `None` when no row count is available |
| `log_path` | Path to the associated simulator log, or `None`; both built-in SPICE backends can retain logs |

An incomplete result can have a positive `data_rows` value. The exception
message identifies the requested stop, expected final grid point and last
axis value reached.

## Convergence failures

When exit-status or success-banner checks fail, expected outputs are missing,
or rawfile analysis data is empty, silk matches the available diagnostics
against `silk.simulation.convergence.CONVERGENCE_PATTERNS`.
A match raises `ConvergenceError` instead of the generic failure.
Its `matched_lines` field contains up to ten distinct matching lines, in their
original order. Diagnostics include timestep collapse, singular matrices and
failed operating-point solves; the patterns differ by simulator.

Without a convergence match, a process failure raises `SimulationError` and
missing output or parsed results with no analysis rows raise `SimulationAborted`.
Zero-byte or malformed rawfiles raise `RawfileError` (a `SimulationError`)
during parsing; malformed Xyce noise CSV raises `SimulationError`. An empty Xyce noise CSV
and endpoint failures remain `SimulationAborted`; they are not reclassified
by convergence matching.

Detection does not diagnose the circuit, retry the simulation or change solver
tolerances. Read the exception message and retained log before changing the
circuit or simulation settings.

## Backend implementation

`SimulatorBackend.run` owns the shared result checks. A backend implements
`_launch(netlist_path, results_dir, *, debug=False, timeout=None)` and returns
a `RawResult`, or `(raw, log_path, reason)` for additional diagnostics.
Inside `_launch`, check the simulator's exit status and any required success
marker, and raise `SimulationError` when the process fails.
Concrete backends also implement `render_analysis` and `make_netlister`.
Defining `run` or `run_module` in a backend subclass raises `TypeError` when
the class is defined; these methods belong to the shared orchestration.

For a native result that does not use silk's nutmeg rawfile reader, either
define `silk_has_data() -> bool` on the native result's class or override
`has_data(self, native)` on the backend. The default predicate checks the
class-defined hook first, then the rawfile's `.npoints` or `.vectors`, excluding
the synthetic `constants` plot. An unsupported native shape raises
`NotSupportedError`; a hook that returns something other than a `bool` raises
`TypeError`. A Xyce noise-only result may have `native=None` because its CSV is
validated separately.

Loading saved results with `load_raw()` also checks for analysis data and
rejects empty results, including a synthetic `constants` plot. It does not
apply endpoint checks.

## See also

- [API reference](api-reference.md)
- [Known simulation limitations](known_issues.md)
