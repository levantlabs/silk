"""silk.simulation.xyce_runner — XyceRunner: subprocess wrapper for Xyce.

Mirrors :class:`~silk.simulation.runner.NgspiceRunner` but invokes the Xyce
CLI. silk's Xyce deck carries a ``.print <analysis> format=raw file=…/rawfile.raw``
directive (see :class:`~silk.circuit.xyce_netlister.XyceNetlister`), so Xyce
writes the rawfile itself; this runner just launches Xyce and reads the result
back through silk's own rawfile reader (Xyce's ``format=raw`` output is
nutmeg-compatible).

The Xyce executable is the ``SILK_XYCE_BIN`` setting (environment variable,
then the settings file; :mod:`silk.settings`), else ``Xyce`` on ``PATH``. An internal helper of :class:`~silk.simulation.backend.XyceBackend`,
not a public contract.
"""
import os
import subprocess
from silk.simulation._failure_detail import failure_detail

from silk.signals.rawfile import read_rawfile
from silk.simulation.executor import linux_preexec_fn
from silk.simulation.run_files import RunFiles
from silk import settings as _settings
from silk.exceptions import SimulationError, SimulationAborted
from silk.simulation.convergence import convergence_error, convergence_lines
import re

# Xyce's banner and timing statistics carry no cause; keep them out of the
# no-diagnostic fallback of a failure message.
_XYCE_BANNER = re.compile(
    r'^\s*(?:\*{5}\s*$|'
    r'\*{5}\s+(?:Welcome to the Xyce|This is version Xyce|Date:|Executing netlist|'
    r'Reading and parsing|Setting up |Device Count Summary|Number of Unknowns|'
    r'Initializing|Total (?:Simulation Solvers|Elapsed) Run Time|End of Xyce)|'
    r'Timing summary of )')
_XYCE_TABLE_HEAD = re.compile(r'^\s*Stats\s+Count\s+CPU Time')
_XYCE_TABLE_ROW = re.compile(
    r'^\s*(?:-{10,}(?:\s+-{3,})*|[A-Za-z][A-Za-z ]*?\s+\d+\s+'
    r'\d+\.\d+\s+\(\s*<?\s*[\d.]+%\)\s+\d+\.\d+\s+\(\s*<?\s*[\d.]+%\))\s*$')


def _XYCE_NOISE(lines):
    """Drop Xyce's banner lines and the timing table its header opens; any
    other line, inside or after the table, is kept."""
    kept = []
    in_table = False
    for line in lines:
        if _XYCE_TABLE_HEAD.match(line):
            in_table = True
        elif in_table and _XYCE_TABLE_ROW.match(line):
            pass
        elif _XYCE_BANNER.match(line):
            in_table = False
        else:
            in_table = False
            kept.append(line)
    return kept


def _xyce_executable() -> str:
    """Resolve the Xyce binary: the ``SILK_XYCE_BIN`` setting (environment
    variable, then the settings file, see :mod:`silk.settings`) or ``Xyce``
    on PATH."""
    return _settings.get('SILK_XYCE_BIN')


# Xyce abort markers in captured stdout/stderr. Message enrichment only —
# never the pass/fail decision.
XYCE_ABORT_SENTINELS = (
    'time step too small',
    'Abort',
    'aborted',
    'no convergence',
    'Fatal',
)


def xyce_abort_reason(text):
    """Return the first Xyce abort-sentinel line in *text* (stripped), or None."""
    for line in (text or '').splitlines():
        low = line.lower()
        if any(s.lower() in low for s in XYCE_ABORT_SENTINELS):
            return line.strip()
    return None


def read_xyce_noise_csv(path: str) -> dict:
    """Parse a Xyce ``.PRINT NOISE FORMAT=CSV`` file into column arrays.

    Xyce cannot write noise to a rawfile, so silk emits noise as CSV (header
    ``"FREQ","ONOISE","INOISE"``, comma-delimited, no footer). Returns a dict of
    upper-cased column name -> numpy array. ``ONOISE``/``INOISE`` are in
    **V²/Hz** (Xyce's documented units); callers convert to V/√Hz via sqrt.
    """
    import numpy as np
    from silk.exceptions import SimulationError

    if not os.path.exists(path):
        raise SimulationError(
            f"Xyce noise CSV not found at {path}. The deck needs a "
            f"'.print noise format=csv file=…' directive."
        )
    try:
        with open(path) as f:
            header = f.readline()
            cols = [c.strip().strip('"').upper() for c in header.split(',')]
            if not {'FREQ', 'ONOISE', 'INOISE'}.issubset(cols):
                raise ValueError('expected FREQ, ONOISE and INOISE columns')
            rows = [[float(x) for x in line.split(',')]
                    for line in f if line.strip()]
        if not rows:
            raise SimulationAborted(
                f'Xyce noise CSV contains no data rows: {path}',
                simulator='xyce', data_rows=0)
        if any(len(row) != len(cols) for row in rows):
            raise ValueError('data row width does not match the header')
        arr = np.asarray(rows, dtype=float)
    except (OSError, ValueError) as exc:
        raise SimulationError(f'Cannot read Xyce noise CSV {path!r}: {exc}') from exc
    return {name: arr[:, i] for i, name in enumerate(cols)}


class XyceRunner:
    """Runs a Xyce simulation from a netlist file and loads the results.

    Parameters
    ----------
    netlist_path : str
        Absolute path to the Xyce netlist (deck) file.
    results_dir : str
        Directory where the deck's ``.print … file=`` directive writes
        ``rawfile.raw``.
    debug : bool
        Let Xyce output flow to the terminal when True.
    timeout : float, optional
        Maximum seconds to wait for Xyce. None means no limit.
    extra_args : tuple of str
        Arguments appended after the netlist path (a Monte Carlo sample's
        ``-randseed <seed>``).
    """

    def __init__(self, netlist_path: str, results_dir: str, debug: bool = False,
                 timeout: float = None, *, run_files: RunFiles = None,
                 require_all_outputs: bool = None,
                 extra_args: tuple = ()):
        self.extra_args = tuple(str(a) for a in extra_args)
        self.run_files = run_files or RunFiles.resolve(
            netlist_path, results_dir, noise=True, log=True)
        self.debug = debug
        self.timeout = timeout
        self._results = None
        self._load_allowed = True
        self.stdout = self.stderr = None
        # Explicit legacy decks have candidate paths but no analysis inventory.
        self._require_all_outputs = (run_files is not None if require_all_outputs is None
                                     else require_all_outputs)

    @property
    def netlist_path(self):
        return self.run_files.netlist_path

    @property
    def rawfile_path(self):
        return self.run_files.rawfile_path

    @property
    def results_dir(self):
        return self.run_files.directory

    def run(self) -> None:
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
        """Run Xyce. Raises SimulationError on failure or timeout.

        Plain ``subprocess.run``, like
        :meth:`~silk.simulation.runner.NgspiceRunner.run` — see its
        docstring for why this isn't routed through
        :func:`~silk.simulation.executor.spawn_tracked`.
        """
        xyce = _xyce_executable()
        if self.debug:
            print(f'Starting Xyce on {self.netlist_path}')
            print(f'Expecting rawfile at {self.rawfile_path}')
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
        try:
            # Write as the process runs so cancellation retains partial output.
            log_path = (generation.staged.logfile_path if generation is not None
                        else self.run_files.logfile_path or os.path.join(self.results_dir, 'output.log'))
            with open(log_path, 'w') as log:
                if generation is not None:
                    generation.launched = True
                    from silk.simulation.launch_fact import note_launch
                    note_launch(True)
                result = subprocess.run(
                    [xyce, os.path.abspath(self.netlist_path), *self.extra_args],
                    cwd=self.results_dir,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                    text=True,
                    preexec_fn=linux_preexec_fn(),
                )
        except FileNotFoundError:
            if generation is not None:
                generation.launched = False
                from silk.simulation.launch_fact import note_launch
                note_launch(False)
            raise SimulationError(
                f"Xyce executable not found ({xyce!r}). Set SILK_XYCE_BIN "
                f"(environment or the silk settings file, see "
                f"docs/settings.md) or put Xyce on PATH."
            )
        except subprocess.TimeoutExpired:
            raise SimulationError(
                f"Xyce simulation timed out after {self.timeout}s."
            )
        except OSError as exc:
            if generation is not None:
                generation.launched = False
                from silk.simulation.launch_fact import note_launch
                note_launch(False)
            raise SimulationError(f'Cannot launch Xyce ({xyce!r}): {exc}') from exc
        self.stdout = getattr(result, 'stdout', None)
        self.stderr = getattr(result, 'stderr', None)
        # Test doubles or custom launch adapters may still return captured text.
        with open(log_path, 'a') as log:
            log.write(self.stdout or '')
            log.write(self.stderr or '')
        with open(log_path) as log:
            self.stdout = log.read()
        if self.debug:
            print(self.stdout, end='')
        # Xyce signals success with an "End of Xyce" banner on stdout; a zero
        # exit code alone is not always reliable across builds.
        ok = result.returncode == 0 and (self.stdout is None
                                         or 'End of Xyce' in (self.stdout or ''))
        if not ok:
            text = f'{self.stdout or ""}\n{self.stderr or ""}'
            detail = failure_detail(text, lead=convergence_lines('xyce', text),
                                    noise=_XYCE_NOISE)
            message = (f"Xyce simulation failed (exit code {result.returncode}).\n"
                       f"  {detail}")
            conv = convergence_error(
                'xyce', f'{self.stdout or ""}\n{self.stderr or ""}', message)
            if conv is not None:
                raise conv
            raise SimulationError(message)

        self._check_outputs()
        self._load_allowed = True

    def _check_outputs(self):
        """Require this prepared analysis's outputs, or a legacy candidate."""
        paths = [p for p in (self.rawfile_path, self.run_files.noise_path)
                 if p is not None]
        present = [os.path.isfile(p) for p in paths]
        if not any(present) or (self._require_all_outputs and not all(present)):
            reason = xyce_abort_reason(self.stdout or '')
            message = (f'Xyce wrote no fresh expected output: {paths}'
                       + (f'. Reason: {reason}' if reason else ''))
            conv = convergence_error('xyce', self.stdout, message,
                                     reason=reason, data_rows=None)
            if conv is not None:
                raise conv
            raise SimulationAborted(message, simulator='xyce', reason=reason,
                                    data_rows=None)

    def load(self):
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
        """Load usable raw/CSV results; reject output from a failed attempt.

        A fresh runner may read saved files. Noise-only results have no native
        rawfile; their CSV is validated here before the backend publishes them.
        This is a data-presence check. The backend checks numerical endpoints
        for certified prepared runs before publishing their results.
        """
        if not self._load_allowed:
            raise SimulationError('No current Xyce results: the last attempt failed. Run again first.')
        self._results = None
        self._load_allowed = False
        self._check_outputs()
        noise = self.run_files.noise_path
        if noise is not None and os.path.isfile(noise):
            read_xyce_noise_csv(noise)
        results = None
        if self.rawfile_path is not None and os.path.isfile(self.rawfile_path):
            results = read_rawfile(self.rawfile_path, lazy=True)
            # Local import avoids the backend/runner registration cycle.
            from silk.simulation.backend import RawResult, _require_data
            _require_data(RawResult(results, self.rawfile_path), simulator='xyce',
                          reason=xyce_abort_reason(self.stdout or ''),
                          diagnostic_text=self.stdout)
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
        """The loaded results, or None if load() has not been called."""
        return self._results
