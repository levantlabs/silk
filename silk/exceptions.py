"""silk.exceptions — exception hierarchy."""


class CircuitError(Exception):
    """Base exception for all silk errors."""


class ConfigurationError(CircuitError):
    """Marker base: the CALLER'S SETUP is wrong, not the circuit.

    Multi-test runners fold a failing test into a results dictionary so that
    one circuit that will not converge does not abandon the rest of a sweep
    (``Module.run_all_tests``). That is right for a *result*: the circuit
    misbehaved, and the other circuits still have something to say. It is
    wrong for a *request*: if the call itself is mis-configured, every test in
    the tree fails for the same reason at once, and a dictionary full of error
    strings reads like output.

    An exception that inherits from this marker is therefore exempt from that
    swallow -- it propagates out of ``run_all_tests`` and stops the run.

    Opting in is **deliberate and by inheritance**, declared where the
    exception itself is defined; nothing is classified by matching message
    text, and no runner keeps a list of types. A new exception joins this
    class only by saying so. The marker is a second base, never a
    replacement: every class below keeps the base it already had, so existing
    ``except SimulationError`` / ``except DatabaseError`` callers are
    unaffected.

    Explicitly NOT configuration errors, because they describe a run rather
    than a request: :class:`SimulationAborted`, a simulator that fails to
    launch or exits nonzero, a malformed or missing rawfile, a backend whose
    launch did not confirm the temperature it was asked for, and a PDK model
    library or compiled OSDI plugin that is absent from this machine.
    """


class SettingsError(ConfigurationError):
    """The silk settings file (:mod:`silk.settings`) cannot be used: a line
    that is not ``KEY=value``, an unterminated quote, a key given twice, or a
    ``$SILK_SETTINGS`` that names a missing file. The message names the file
    and, for a bad line, its line number."""


class DatabaseError(CircuitError):
    """Base exception for persistent database operations."""


class DatabaseCompatibilityError(DatabaseError):
    """Raised when a database file is not a supported silk format."""


class UnsupportedSchemaVersionError(DatabaseCompatibilityError):
    """Raised when a silk database uses an unsupported schema version."""


class UnsupportedFingerprintSchemeError(DatabaseCompatibilityError):
    """Raised before comparing an unknown persisted fingerprint scheme."""


class EvidenceError(DatabaseError):
    """Base exception for persisted simulation or measurement evidence."""


class EvidenceNotFoundError(EvidenceError):
    """Raised when no compatible evidence exists for a requested context."""


class EvidenceAmbiguityError(EvidenceError):
    """Raised when a read selector identifies more than one evidence context."""


class EvidenceMismatchError(EvidenceError):
    """Raised when known evidence has an incompatible persisted identity."""


class LockedVersionError(DatabaseError):
    """Raised when a write targets an immutable circuit version."""


class ReportStorageError(DatabaseError):
    """Raised by ``Module.run_test(store=True)`` when a test ran to completion but
    persisting its ``SpecReport`` failed.

    The simulation is not lost: the completed report is on ``report`` (its
    ``storage`` receipt is ``None``), and the underlying storage error is chained
    as ``__cause__`` (an ``EvidenceMismatchError``, ``LockedVersionError``,
    ``sqlite3`` error, ...). ``store_report``'s per-report transaction has
    already rolled back every row it wrote; the live ``Measurement`` objects the
    run recorded on the module's specs are **not** undone, because they are the
    run's evidence rather than the database's.

    Parameters
    ----------
    message : str
        Error message.
    report : SpecReport, optional
        The completed report (keyword-only).

    Attributes
    ----------
    report : SpecReport or None
        The report whose storage failed.
    """

    def __init__(self, message, *, report=None):
        super().__init__(message)
        self.report = report


class CharacterizationContextError(DatabaseError, ConfigurationError):
    """Raised when a runtime characterization-context descriptor (owner path,
    test name, declared parameters, corner, temperature, voltage) is invalid
    or only partially supplied. A scoped context is all six fields or none —
    see ``silk.db.context``."""


class DuplicateNameError(CircuitError):
    """Raised when a test, element, net, source, or passive name already exists."""


class UndeclaredNetError(CircuitError):
    """Raised in strict_nets mode when a net has not been declared."""


class NotASubcircuitError(ConfigurationError):
    """Raised when a Module with no ``module_name`` is used where only a
    subcircuit will do: as a device under test, or emitted as a ``.subckt``
    definition or as an ``x`` instance line.

    ``module_name`` is what makes a Module emittable as a subcircuit. A Module
    without one is a top-level circuit, and SPICE has nothing to instantiate:
    there is no ``.subckt`` for an instance line to reference. Previously, that surfaced as a bare ``TypeError`` from concatenating ``None`` inside
    the netlister -- raised at emission time, far from the call that made the
    mistake, and naming neither the module nor the real problem.

    A :class:`ConfigurationError` (and so still a :class:`CircuitError`,
    which every emission-time caller already catches), because a device under
    test that is not a subcircuit is the caller's setup and not a result: it
    propagates out of ``Module.run_all_tests`` instead of being folded into
    the results dictionary as one test's error string.
    """


class SimulationError(CircuitError):
    """Simulation failure, with an optional public diagnostic ``log_path``.

    Attributes
    ----------
    failed : dict or None
        When every declared corner or Monte Carlo sample fails, maps every
        condition key to its failure in declaration order. Corner values are
        error strings; sample values are
        ``SampleFailure`` (error and seed),
        matching the corresponding results' ``failed`` map. Otherwise None.
        Retained across pickle and on ``TestFailure.exception`` when
        ``run_all_tests()`` collects the error.
    """

    def __init__(self, *args, log_path=None, failed=None):
        super().__init__(*args)
        self.log_path = log_path
        self.failed = failed


class SimulationConfigurationError(SimulationError, ConfigurationError):
    """A simulation request that is wrong before anything is solved.

    The deck that was prepared is not the deck the live configuration would
    emit now, the prepared deck is missing or already consumed, managed TEMP
    was asked for without one, the caller's own SPICE text declares a
    temperature that cannot be read, or a cached report is being asked to
    answer for conditions it was not run under. Every one of these is fixed by
    changing the script -- call ``generate_netlist()`` again, call
    ``clear_cache()``, remove the duplicate ``.temp`` -- and none of them says
    anything about the circuit.

    A :class:`SimulationError` subclass, so every existing
    ``except SimulationError`` handler (the optimizer's included) keeps
    catching these exactly as it did before; also a
    :class:`ConfigurationError`, so ``Module.run_all_tests`` propagates them
    instead of recording an error string per test.
    """


class LoopError(SimulationError):
    """Behavioral feedback cannot run under the selected execution or analysis."""


class ConflictError(SimulationError):
    """Two behavioral models assigned the same net voltage sample."""


class RewriteError(SimulationError):
    """A behavioral quantity was assigned more than once at a sample."""


class KCLError(SimulationError):
    """KCL was asked to fill more than one unknown pin current on a net."""


class SimulationAborted(SimulationError):
    """A simulator run completed at the process level but produced no usable
    result — aborted analysis, zero data rows, or a missing rawfile — even
    though the exit code may say success. Carries structured fields so a UI can
    render the failure reason instead of a misleading green.
    """

    def __init__(self, message, *, simulator, reason=None,
                 data_rows=None, log_path=None):
        super().__init__(message)
        self.simulator = simulator
        self.reason = reason          # e.g. 'Timestep too small', or None
        self.data_rows = data_rows    # 0, or None when no rawfile was produced
        self.log_path = log_path


class ConvergenceError(SimulationAborted):
    """A failed simulator run whose log names a convergence failure (timestep
    too small, singular matrix, gmin/source stepping failed, ...).

    Raised in place of the generic error the same failure raised before, only
    when a pattern in :data:`silk.simulation.convergence.CONVERGENCE_PATTERNS`
    matches; an unmatched failure is unchanged. A :class:`SimulationAborted`
    and therefore a :class:`SimulationError`, so every existing handler keeps
    catching it (``data_rows`` is ``None`` when the run exited non-zero
    before writing a rawfile). Detection only: nothing is diagnosed or retried.

    ``matched_lines`` is the list of distinct log lines that matched;
    ``log_path`` is the public simulator log, or ``None`` when no run log
    is associated with the error.
    """

    def __init__(self, message, *, simulator, matched_lines, reason=None,
                 data_rows=None, log_path=None):
        super().__init__(message, simulator=simulator, reason=reason,
                         data_rows=data_rows, log_path=log_path)
        self.matched_lines = list(matched_lines)


class PDKError(CircuitError):
    """Raised for PDK configuration errors (e.g., duplicate device in YAML)."""


class MissingLibrarySectionError(PDKError, ConfigurationError):
    """A PDK model library lacks a ``.lib`` section a request needs.

    Raised before any simulator starts (older open_pdks installs ship no
    statistical sections). A ConfigurationError, so multi-test runners
    propagate it instead of folding it into a per-test result.
    """


class VariationError(ConfigurationError, ValueError):
    """A statistical declaration is invalid; fix the model configuration."""


class MonteCarloError(ConfigurationError, ValueError):
    """A Monte Carlo request is invalid (bad settings, unsupported circuit,
    stale re-run). The caller's setup is wrong, not the circuit."""


class BusPairingError(ConfigurationError, ValueError):
    """Bus Signal maths was asked to combine buses of different length.

    Buses combine by position like numpy arrays; unequal widths raise this,
    naming both. A ``ValueError`` too, matching the mismatch errors scalar
    Signal maths raises.
    """


class NotSupportedError(ConfigurationError):
    """Raised when a backend cannot provide a requested capability (e.g. a
    simulator backend that does not expose device currents). Callers convert it
    into a user-facing message (spec 2026-07-03 §6.1)."""


class SweepSourceError(SimulationConfigurationError, ValueError):
    """A DC sweep or noise analysis names a source the bench does not have, or
    the bare name of an arrayed source. A sweep names one element
    (``vin[1]``). Raised before any deck is written; a ``ValueError`` too,
    matching the netlist-build refusal it now precedes."""


class RunArgumentError(ConfigurationError, TypeError):
    """Raised when the arguments a multi-test runner forwards do not fit the
    ``run`` method of the test they are aimed at — an unexpected keyword such
    as ``run_all_tests(store=True)``, or a required parameter left unsupplied.

    Python answers that with a bare :class:`TypeError` raised by the call
    itself, and a builtin cannot inherit :class:`ConfigurationError`, so such
    a typo used to become one ``{'error': ...}`` entry per test rather than a
    raised error. ``Module.run_all_tests`` therefore binds its
    arguments against each test's own ``run`` signature *before* invoking it
    and raises this instead. The classification still lives at a raise site —
    this one — and not in a list of builtin types kept by the runner.

    A :class:`TypeError` as well, so a caller that already wrote
    ``except TypeError`` around a mis-typed call keeps catching it.
    """


class UnboundSignalError(CircuitError):
    """Raised when ``x``, ``y``, ``xlabel`` or ``ylabel`` is read on an
    unbound :class:`~silk.signals.base.Signal` — a probe (``tb.probe(...)``)
    that names a signal before its bench has produced results, or whose
    bench has since been garbage-collected.

    Deliberately **not** an :class:`AttributeError` subclass. Two existing
    call sites read Signal attributes defensively to mean "this Signal
    carries no data": ``getattr(sig, 'y', None)`` in the signal-kind guard
    module's ``_is_complex`` helper and ``getattr(sig, 'xlabel', None)`` in
    ``silk.lint.rules.stress``. An ``AttributeError`` raised there would
    be silently read as exactly that, rather than surfaced as "not resolved
    yet" — a different fact, and one that must propagate rather than be
    swallowed by a defensive ``getattr``.
    """


class AmbiguousBranchCurrentError(ConfigurationError, ValueError):
    """``i(x)`` has two readings on this bench: ``x`` is a source's
    silk name AND the emitted instance name of a different source (sources
    ``vs`` and ``vvs``: ``i(vvs)`` is ``Vvvs``'s current by the first reading
    and ``Vvs``'s by the second). silk never picks one; the message names both.
    """


class AmbiguousSignalError(SimulationError):
    """A signal request matches more than one distinct vector of the selected
    plot: the as-given spelling and its simulator spelling reach
    different rows, for example a literal top-level net ``xx1[0].mid`` beside
    the hierarchical ``xx1<0>.mid``. silk never picks one; the message names
    the request, the plot and the raw candidates.
    """


class MetricBindingError(ConfigurationError):
    """A metric's Signal slot could not be resolved before the metric ran: a probe names a path, net, or terminal absent from the
    bench's results, or a recorded operation on an unbound Signal fails
    during resolution. The message names the metric.

    A :class:`ConfigurationError`, so ``Module.run_all_tests`` propagates it
    instead of folding it into that test's results as an ``{'error': str}``
    entry: the binding itself is wrong, not the circuit under
    test, and every test in the same tree would fail for the identical
    reason.

    The signal-kind guard's failure (``SignalKindError``, a
    :class:`ValueError` subclass raised by the guard in silk's signal-kinds
    module) is re-raised as this class by the evaluation loop when a
    metric's Signal input is the wrong kind — a transient trace handed to a
    metric that requires a frequency response, for example — so that
    failure raises from ``run()`` instead of being buried as a report
    string.
    """


class OutputPathError(ConfigurationError, ValueError):
    """Raised when a run's output location cannot be formed from the request:
    an invalid ``run_label``, a module or test name that is not a string, or a
    testbench directory that would resolve outside the DUT output root.

    Detected while selecting paths, before any directory is created, any deck
    is written or any simulator is launched. Every one of these is fixed by
    changing the call, and none of them says anything about the circuit, so
    the marker applies: a bad label handed to ``Module.run_all_tests`` stops
    the run instead of repeating itself once per test.

    A :class:`ValueError` as well, so the historical ``run()``/``run_test()``
    contract (an invalid label raised ``ValueError``) stays back-compatible —
    the same convention :class:`silk.pdk.registry.PDKRegistryError` and
    :class:`silk.circuit.port_roles.RoleRulesetError` already follow.
    """


class DeckTextError(ConfigurationError, ValueError):
    """Raised when a caller's string cannot be written into a SPICE card.

    silk writes each value it is given (a device or source parameter, a
    waveform field, a simulator option, an analysis field, a library or
    output path, a corner name) into ONE card. A line break inside that
    string would end the card early and turn the rest into deck text of its
    own: an injected ``.control`` block runs under ``ngspice -b``, and its
    ``shell`` command runs a program. Any character
    ``str.splitlines()`` splits on is refused, as is a corner name holding a
    control character or naming ``.`` or ``..`` (the corner is also an
    output folder name).

    The message names where the value was written. Raised at construction
    where silk validates values, and again when the card is rendered, so a
    value changed later is refused too. Never raised for explicit opaque
    content (``add_custom_line``, a custom ``to_spice()``), which silk writes
    verbatim.

    A configuration error (the call is wrong, not the circuit) and a
    :class:`ValueError`, the type silk already raised for a malformed value.
    """


class UnknownCornerError(ConfigurationError, ValueError):
    """Raised when a process corner name is not one the PDK declares.

    Raised by :meth:`silk.circuit.module_test.ModuleTestbench.set_corner <silk.circuit.ModuleTestbench.set_corner>`
    when the name is set, before any deck is written, and never answered by
    quietly falling back to the PDK's default corner. The message lists the
    PDK's valid corners. A configuration error, so a bad corner on one test
    stops :meth:`Module.run_all_tests <silk.circuit.Module.run_all_tests>` instead of being folded into the
    results; a :class:`ValueError` as well, the type ``Module.simulate``
    raises for an unknown corner.
    """



__all__ = [
    "CircuitError", "DatabaseError", "DatabaseCompatibilityError",
    "UnsupportedSchemaVersionError", "UnsupportedFingerprintSchemeError",
    "EvidenceError", "EvidenceNotFoundError", "EvidenceAmbiguityError",
    "EvidenceMismatchError",
    "LockedVersionError", "CharacterizationContextError", "DuplicateNameError",
    "UndeclaredNetError", "NotASubcircuitError", "SimulationError", "SimulationConfigurationError", "SweepSourceError",
    "ConflictError", "RewriteError", "KCLError", "LoopError",
    "SimulationAborted", "PDKError", "NotSupportedError", "ConfigurationError",
    "VariationError", "BusPairingError",
    "RunArgumentError", "OutputPathError", "UnknownCornerError", "DeckTextError",
    "UnboundSignalError", "MetricBindingError", "MonteCarloError",
    "MissingLibrarySectionError",
    "ReportStorageError", "AmbiguousSignalError",
]
