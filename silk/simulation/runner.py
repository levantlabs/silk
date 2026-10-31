"""silk.simulation.runner — NgspiceRunner: subprocess wrapper for ngspice.

Owns the ngspice invocation and raw-file loading. PyNetlist delegates to
this class so the simulation backend is independently testable and replaceable.
"""
import os
import subprocess
import tempfile
from silk.simulation._failure_detail import failure_detail
from silk.signals.rawfile import read_rawfile
from silk.simulation.executor import linux_preexec_fn
from silk.simulation.run_files import RunFiles
from silk.simulation._ngspice_compat import NgspiceCompatibility, warn_version_advice
from silk import settings as _settings
from silk.exceptions import SimulationError, SimulationAborted
from silk.simulation.convergence import convergence_error, read_log_text


# ngspice batch mode can exit 0 while the analysis aborted. These substrings in
# output.log identify that case; used ONLY to build a human-readable reason
# string, never as the pass/fail decision.
NGSPICE_ABORT_SENTINELS = (
    'Timestep too small',
    'run simulation(s) aborted',
    'simulation(s) aborted',
    'singular matrix',
    'no convergence',
    'Fatal error',
    # TASK-212: a multi-analysis deck's `write <path> tran1.all ac1.all` is
    # ALL-OR-NOTHING — MEASURED on ngspice-42, a write naming one plot that
    # does not exist prints this warning and writes NO FILE AT ALL, exit 0.
    # So when one of several declared analyses fails, the whole rawfile is
    # lost and this is the only line that says which plot was missing. Message
    # enrichment only, like every entry here — it decides nothing.
    'is not available or has zero length',
)


def ngspice_executable():
    """Resolve the ngspice binary: the ``SILK_NGSPICE_BIN`` setting (environment
    variable, then the settings file, see :mod:`silk.settings`) or ``ngspice``
    on PATH.

    Parallel to ``$SILK_XYCE_BIN`` (:mod:`silk.simulation.xyce_runner`), but
    kept as the bare name when unset so the emitted command is unchanged from
    before this hook existed. Setting the env var to an absolute path is the
    supported way to run one netlist against several ngspice versions without
    touching PATH, e.g. ``SILK_NGSPICE_BIN=~/ngspice/ngspice-47/bin/ngspice`` —
    which matters because ``save nosub`` needs ngspice >= 43 and
    ``save nointernals`` needs >= 45 while Ubuntu 24.04 still ships 42. Read
    at call time, so a test can switch versions per-run.
    """
    return _settings.get('SILK_NGSPICE_BIN')


# OSDI plugin loading (compiled Verilog-A, manifest `library.osdi:`). ngspice
# prints these when a `pre_osdi` line cannot be honoured. The version line is
# the one that matters in practice: openvaf-r compiles to OSDI v0.4, which
# ngspice accepts from release 44 on, while Ubuntu 24.04's ngspice-42 speaks
# only v0.3 and then reports every PSP model as "could not find a valid
# modelname" — a message that never mentions OSDI. Ordered by specificity:
# the first match wins, so a version mismatch is named as such rather than
# as the generic "couldn't be loaded" that follows it in the log.
OSDI_LOAD_SENTINELS = (
    'only supports OSDI v',
    'Error opening osdi lib',
    "couldn't be loaded",
)

OSDI_FIX = (
    "The plugins were compiled for a newer OSDI interface than this ngspice "
    f"understands (openvaf-r emits OSDI v0.4; ngspice needs release "
    f"{NgspiceCompatibility(None).since('osdi_v04')} or "
    "later). Point $SILK_NGSPICE_BIN at a newer ngspice binary, or compile "
    "the plugins with an openvaf that targets this ngspice."
)


def osdi_problem_from_text(text):
    """The first OSDI-load complaint in ngspice output, or None.

    Pure parser over a log/stdout string; the probe and the runner share it.
    Prefers the OSDI version-mismatch line over the generic load failure
    (see ``OSDI_LOAD_SENTINELS``). ngspice-42 prints the version complaints
    for every plugin on ONE line with no separator, so only the first
    plugin's complaint is returned — one is enough to name the cause.
    """
    if not text:
        return None
    lines = text.splitlines()
    for sentinel in OSDI_LOAD_SENTINELS:
        for line in lines:
            if sentinel in line:
                if sentinel == OSDI_LOAD_SENTINELS[0]:
                    # "NGSPICE only supports OSDI v0.3 but "<a>" targets
                    # v0.4!NGSPICE only supports ..." — keep the first clause.
                    head = line.find('NGSPICE only supports')
                    line = line[head:] if head >= 0 else line
                    cut = line.find('!', line.find('targets'))
                    if cut > 0:
                        line = line[:cut + 1]
                return line.strip()
    return None


def osdi_load_problem(osdi_paths, ngspice=None, timeout=30):
    """Ask the ngspice that silk would run whether it can load *osdi_paths*.

    Runs a circuit-free deck whose ``.control`` block is just the ``pre_osdi``
    lines silk emits, and returns the first complaint (see
    :func:`osdi_problem_from_text`) or None when every plugin loaded. ngspice
    exits 0 either way, so the output text is the only signal — which is
    also why this is a real run rather than a version-number comparison: the
    binary itself is the authority on what it can load. Returns None (not an
    error) when ngspice itself cannot be run; that is a different problem,
    reported elsewhere.
    """
    paths = list(osdi_paths or [])
    if not paths:
        return None
    ngspice = ngspice or ngspice_executable()
    deck = ('* silk OSDI load probe\n.control\n'
            + ''.join(f'pre_osdi {p}\n' for p in paths)
            + 'quit 0\n.endc\n.end\n')
    with tempfile.TemporaryDirectory(prefix='silk-osdi-probe-') as d:
        cir = os.path.join(d, 'probe.cir')
        with open(cir, 'w') as f:
            f.write(deck)
        try:
            out = subprocess.run([ngspice, '-b', cir], capture_output=True,
                                 text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            return None
    return osdi_problem_from_text(f'{out.stdout}\n{out.stderr}')


def abort_reason_from_log(log_path):
    """Return a timestep failure with preceding node context, else the first diagnostic.

    Retain the last singular-matrix or check-node line before that failure.
    Message enrichment only — never a decision signal.
    """
    first = None
    node_context = None
    try:
        with open(log_path) as f:
            for line in f:
                if 'Timestep too small' in line:
                    reason = line.strip()
                    return (f'{reason} (earlier: {node_context})'
                            if node_context else reason)
                lower = line.lower()
                if 'singular matrix' in lower or 'check node' in lower:
                    node_context = line.strip()
                if first is None and any(s in line for s in NGSPICE_ABORT_SENTINELS):
                    first = line.strip()
    except OSError:
        return None
    return first


def raise_no_rawfile(rawfile_path, logfile_path):
    """Raise the typed error for an ngspice run that exited 0 but wrote no
    rawfile: a convergence error when the log says so, else
    :class:`~silk.exceptions.SimulationAborted`. Shared by the one-process
    runner and the multi-sample launch (``silk.simulation.mc_chunk``) so a
    failed sample reads the same either way."""
    reason = abort_reason_from_log(logfile_path)
    message = (
        f"ngspice exited 0 but wrote no rawfile at {rawfile_path}"
        + (f": {reason}" if reason else "")
        + f". See {logfile_path}."
    )
    conv = convergence_error(
        'ngspice', read_log_text(logfile_path), message,
        reason=reason, data_rows=None, log_path=logfile_path)
    if conv is not None:
        raise conv
    raise SimulationAborted(
        message, simulator='ngspice', reason=reason, data_rows=None,
        log_path=logfile_path,
    )


class NgspiceRunner:
    """Runs an ngspice simulation from a netlist file and loads the results.

    The runner is a thin wrapper over one ``ngspice -b`` subprocess:
    :meth:`run` launches it, :meth:`load` parses the rawfile it wrote.
    Higher-level code normally goes through the simulator backend rather
    than constructing a runner directly.

    Parameters
    ----------
    netlist_path : str
        Path to the SPICE netlist file.
    results_dir : str
        Directory where the rawfile and the log are written. Ignored when
        ``run_files`` is given.
    debug : bool, optional
        When true, print progress messages and let ngspice output flow to
        the terminal instead of capturing it. Defaults to false.
    timeout : float, optional
        Seconds to wait for ngspice before the run is abandoned with a
        :class:`~silk.exceptions.SimulationError`. ``None`` (the default)
        waits indefinitely.
    run_files : RunFiles, optional
        Pre-resolved file locations for the run. When omitted they are
        resolved from ``netlist_path`` and ``results_dir`` with a log file.

    Examples
    --------
    Construction only resolves file locations; it neither creates files nor
    starts ngspice.

    >>> from silk.simulation import NgspiceRunner
    >>> runner = NgspiceRunner('deck.cir', 'out')
    >>> runner.rawfile_path, runner.logfile_path
    ('out/rawfile.raw', 'out/output.log')
    >>> runner.results is None
    True

    Attributes
    ----------
    run_files : RunFiles
        The file locations this runner uses.
    debug : bool
        The ``debug`` flag.
    timeout : float or None
        The ``timeout`` in seconds.
    stdout, stderr : bytes or None
        Captured ngspice output, set by :meth:`run` (``None`` when output was
        not captured, as with ``debug=True``). Not present before the first
        :meth:`run`.
    """

    def __init__(self, netlist_path: str, results_dir: str, debug: bool = False,
                 timeout: float = None, *, run_files: RunFiles = None):
        self.run_files = run_files or RunFiles.resolve(
            netlist_path, results_dir, log=True)
        self.debug = debug
        self.timeout = timeout
        self._results = None
        self._load_allowed = True  # A fresh runner may explicitly read a saved file.

    @property
    def netlist_path(self):
        """str: Path of the netlist this runner simulates (from ``run_files``)."""
        return self.run_files.netlist_path

    @property
    def rawfile_path(self):
        """str: Path of the rawfile ngspice writes (from ``run_files``)."""
        return self.run_files.rawfile_path

    @property
    def logfile_path(self):
        """str: Path of the ngspice log file (from ``run_files``)."""
        return self.run_files.logfile_path

    def run(self) -> None:
        """Run ngspice in batch mode on the netlist.

        The previous result and log files are removed, and any cached
        results are cleared, immediately before launch, so a failed attempt
        cannot be loaded as current. Exit code zero is not trusted: a run
        that exits cleanly without writing a rawfile is reported as an
        abort. Captured output is left on ``stdout`` and ``stderr``.

        Raises
        ------
        SimulationError
            If the ngspice executable cannot be found or launched, if the
            run exceeds ``timeout`` (the message names the log file), or if
            ngspice exits non-zero (the message carries a log excerpt, and
            the OSDI plugin-load cause when one is detected).
        SimulationAborted
            If ngspice exits 0 but writes no rawfile.

        Notes
        -----
        A non-zero exit, or a clean exit with no rawfile, whose log matches
        silk's convergence-failure patterns raises ``ConvergenceError``
        instead of the plain error; it is a subclass of
        ``SimulationAborted``.
        """
        previous = getattr(self, '_generation', None)
        try:
            return self._run()
        except BaseException as exc:
            from silk.simulation.generation import abandon, publish_failure_log, _active
            current = getattr(self, '_generation', None)
            if current is not previous or getattr(self, '_generation_managed', False):
                generation = current or _active.get()
                publish_failure_log(generation, exc)
                if generation is not None and getattr(generation, 'launched', False):
                    self.run_files = generation.public
                abandon(current)
            raise

    def _run(self) -> None:
        """Run ngspice in batch mode. Raises SimulationError on failure.

        Output is captured silently by default. Pass ``debug=True`` at
        construction to let ngspice output flow to the terminal.
        Captured output is available via ``self.stdout`` and ``self.stderr``.
        Immediately before launch, remove the exact previous result/log files
        and reset cached results. Failed attempts cannot be loaded as current.

        Stays a plain ``subprocess.run`` call — deliberately
        *not* routed through ``silk.simulation.executor.spawn_tracked``,
        so ngspice stays in the caller's process group exactly as before and
        every existing test that mocks ``subprocess.run`` keeps working.
        ``preexec_fn=linux_preexec_fn()`` still gets the process-death
        guarantee (``PR_SET_PDEATHSIG``): ngspice dies with whichever
        process launched it — a pool worker ``Pool.terminate()`` kills, or
        this process itself if it is killed outright. ``subprocess.run``'s
        own ``except: process.kill(); raise`` (stdlib) already covers a
        ``KeyboardInterrupt``/exception raised while this call is waiting.
        """
        ngspice = ngspice_executable()
        warn_version_advice(ngspice)
        if self.debug:
            print(f'Starting simulation with file {self.netlist_path}')
            print(f'Saving results in {self.rawfile_path}')
        if not getattr(self, '_generation_managed', False):
            from silk.simulation.generation import begin, abandon
            public = getattr(self, '_public_run_files', self.run_files)
            generation = begin(public)
            abandon(getattr(self, '_generation', None))
            self._public_run_files = public
            self._generation = generation
            self.run_files = generation.staged
        self._results = None
        self.stdout = self.stderr = None
        self._load_allowed = False
        self.run_files.clear_previous_outputs()
        from silk.simulation.generation import _active
        generation = getattr(self, '_generation', None) or _active.get()
        if generation is not None:
            # Also leave a diagnostic file for an otherwise silent cancellation.
            with open(self.logfile_path, 'w'):
                pass
            generation.launched = True
            from silk.simulation.launch_fact import note_launch
            note_launch(True)
        try:
            result = subprocess.run(
                [ngspice, '-b', '-r', self.rawfile_path,
                 '-o', self.logfile_path, self.netlist_path],
                capture_output=not self.debug,
                timeout=self.timeout,
                preexec_fn=linux_preexec_fn(),
            )
        except FileNotFoundError:
            if generation is not None:
                generation.launched = False
                from silk.simulation.launch_fact import note_launch
                note_launch(False)
            raise SimulationError(
                f"ngspice executable not found ({ngspice!r}). Set "
                f"SILK_NGSPICE_BIN (environment or the silk settings file, "
                f"see docs/settings.md) or put ngspice on PATH."
            )
        except subprocess.TimeoutExpired:
            raise SimulationError(
                f"ngspice simulation timed out after {self.timeout}s."
                f" Check {self.logfile_path} for progress."
            )
        except OSError as exc:
            if generation is not None:
                generation.launched = False
                from silk.simulation.launch_fact import note_launch
                note_launch(False)
            raise SimulationError(f'Cannot launch ngspice ({ngspice!r}): {exc}') from exc
        self.stdout = getattr(result, 'stdout', None)
        self.stderr = getattr(result, 'stderr', None)
        if result.returncode != 0:
            log_text = ''
            try:
                with open(self.logfile_path) as f:
                    log_text = f.read()
            except OSError:
                pass
            detail = failure_detail(log_text)
            # A refused OSDI plugin is the one failure whose log never names
            # its cause where the eye lands: the version complaint is a plain
            # line ngspice prints before the circuit banner, and the errors
            # that follow ("could not find a valid modelname") blame the
            # model. Name it, and say what to do.
            osdi = osdi_problem_from_text(log_text)
            if osdi:
                detail = (f"OSDI plugin load failed: {osdi}\n  {OSDI_FIX}"
                          f"\n  {detail}")
            message = (
                f"ngspice simulation failed (exit code {result.returncode})."
                f"\n  {detail}\nSee {self.logfile_path} for full output."
            )
            conv = convergence_error('ngspice', log_text, message,
                                     log_path=self.logfile_path)
            if conv is not None:
                raise conv
            raise SimulationError(message)
        # Exit code 0 is not sufficient: ngspice can abort the analysis and
        # still exit 0, writing no rawfile at all. Catch that here as a clean
        # abort rather than letting load() crash with FileNotFoundError.
        if not os.path.isfile(self.rawfile_path):
            raise_no_rawfile(self.rawfile_path, self.logfile_path)

        self._load_allowed = True

    def load(self):
        """Parse the rawfile written by the last successful :meth:`run`.

        Returns
        -------
        silk.signals.rawfile.RawFile
            The lazily-read rawfile. It is also kept on :attr:`results`.

        Raises
        ------
        SimulationError
            If the last :meth:`run` (or a previous :meth:`load`) failed, so
            there are no current results (run again first). A freshly
            constructed runner may read an existing rawfile.
        SimulationAborted
            If the rawfile carries no analysis data rows (for example an
            aborted run's synthetic ``constants`` plot).
        """
        try:
            return self._load_generation()
        except BaseException as exc:
            from silk.simulation.generation import abandon, publish_failure_log, _active
            generation = getattr(self, '_generation', None) or _active.get()
            publish_failure_log(generation, exc)
            if generation is not None and getattr(generation, 'launched', False):
                self.run_files = generation.public
            abandon(getattr(self, '_generation', None))
            raise

    def _load_generation(self):
        generation = getattr(self, '_generation', None)
        if generation is None or not generation.accepted:
            return self._load()
        from dataclasses import replace
        with generation.read():
            public = self.run_files
            self.run_files = replace(public, **{
                kind: generation.resolve(kind) for kind in
                ('rawfile_path', 'noise_path', 'logfile_path', 'temperature_path')})
            try:
                return self._load()
            finally:
                self.run_files = public

    def _load(self):
        """Parse the raw file and require usable saved data.

        Returns the parsed ``silk.signals.rawfile.RawFile`` after the
        data-presence check. The backend checks numerical endpoints for
        certified prepared runs before publishing their results.
        """
        if not self._load_allowed:
            raise SimulationError('No current ngspice results: the last attempt failed. Run again first.')
        self._results = None
        self._load_allowed = False
        results = read_rawfile(self.rawfile_path, lazy=True)
        # Local import avoids the backend/runner registration cycle.
        from silk.simulation.backend import RawResult, _require_data
        _require_data(RawResult(results, self.rawfile_path), simulator='ngspice',
                      log_path=self.logfile_path)
        generation = getattr(self, '_generation', None)
        if generation is not None:
            from silk.simulation.generation import offer, attach, accept
            from silk.simulation.backend import RawResult
            if not generation.accepted:
                raw = RawResult(results, self.rawfile_path, run_files=self.run_files)
                raw = attach(raw, offer(generation))
                accept(raw)
                self._generation = raw._generation
                self.run_files = self._public_run_files
            elif results is not None:
                attach(RawResult(results, self.rawfile_path), generation)
        self._results = results
        self._load_allowed = True
        return results

    @property
    def results(self):
        """The loaded results, or ``None`` before :meth:`load` has succeeded.

        Returns
        -------
        silk.signals.rawfile.RawFile or None
            The parsed rawfile from the last :meth:`load`. Reset to ``None``
            at the start of every :meth:`run` and :meth:`load`.
        """
        return self._results
