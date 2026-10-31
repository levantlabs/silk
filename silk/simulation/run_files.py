"""Internal concrete paths shared by the consumers of one prepared deck.

Resolution is pure: ``prepare()`` creates directories and
``clear_previous_outputs()`` removes exact outputs before launch. The record
describes a location and optional generation-bound completion evidence, not a
configured output root or a simulation result.
It deliberately preserves path spelling so default emitted decks stay stable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
import stat
import tempfile

from silk.exceptions import SimulationError


@dataclass(frozen=True)
class RunFiles:
    """Immutable, serializable record of the files belonging to one prepared deck.

    Describes locations and optional completion evidence, not a simulation
    result. Resolution is pure: only :meth:`prepare`, :meth:`prepare_new`,
    :meth:`claim_new_attempt` and :meth:`clear_previous_outputs` touch the
    file system. A path a backend does not produce is ``None``. Path
    spelling is preserved as given, so default emitted decks stay stable.

    Parameters
    ----------
    directory : str
        Directory the run's outputs are written to.
    netlist_path : str
        Path of the input netlist (deck).
    rawfile_path : str, optional
        Path of the simulator's rawfile, or ``None`` when the backend writes
        none.
    logfile_path : str, optional
        Path of the simulator's log file, or ``None`` when there is none.
    noise_path : str, optional
        Path of a separate noise CSV (a backend that writes noise results
        outside the rawfile), or ``None``.
    temperature_path : str, optional
        Path of the backend-written runtime temperature record, or ``None``.
    completion : object, optional
        Generation-bound completion evidence; ``None`` for external or
        custom decks.
    completion_unverified_reason : str, optional
        Why completion could not be verified, when it could not.

    Examples
    --------
    >>> from silk.simulation.run_files import RunFiles
    >>> files = RunFiles.resolve('deck.cir', 'out', log=True)
    >>> files.rawfile_path, files.logfile_path, files.noise_path
    ('out/rawfile.raw', 'out/output.log', None)
    """

    #: Directory the run's outputs are written to.
    directory: str
    #: Path of the input netlist (deck).
    netlist_path: str
    #: Path of the simulator's rawfile; ``None`` when the backend writes none.
    rawfile_path: str | None = None
    #: Path of the simulator's log file; ``None`` when there is none.
    logfile_path: str | None = None
    #: Path of a separate noise CSV; ``None`` when noise results are not
    #: written outside the rawfile.
    noise_path: str | None = None
    #: Backend-written runtime record of the ambient temperature (degrees
    #: Celsius) actually applied, for a backend that has no log to scrape
    #: (Xyce). Present only when this run configures TEMP, so a
    #: temperature-free deck is unchanged.
    temperature_path: str | None = None
    #: Generation-bound completion evidence; absent for external/custom decks.
    completion: object | None = None
    #: Why completion could not be verified, when it could not; ``None``
    #: otherwise.
    completion_unverified_reason: str | None = None

    @classmethod
    def resolve(cls, netlist_path: str, directory: str, *,
                raw: bool = True, log: bool = False,
                noise: bool = False, temperature: bool = False) -> RunFiles:
        """Resolve the deterministic layout without creating or deleting files.

        Parameters
        ----------
        netlist_path : str or os.PathLike
            Path of the input netlist.
        directory : str or os.PathLike
            Output directory; each selected output is a fixed file name inside
            it.
        raw : bool, optional
            Include ``rawfile.raw``. Defaults to true.
        log : bool, optional
            Include ``output.log``. Defaults to false.
        noise : bool, optional
            Include ``noise.csv``. Defaults to false.
        temperature : bool, optional
            Include ``xyce_temperature.csv``, the record of the ambient
            temperature in degrees Celsius. Defaults to false.

        Returns
        -------
        RunFiles
            The layout; paths not selected are ``None``.
        """
        directory = os.fspath(directory)
        return cls(
            directory=directory,
            netlist_path=os.fspath(netlist_path),
            rawfile_path=directory + '/rawfile.raw' if raw else None,
            logfile_path=directory + '/output.log' if log else None,
            noise_path=directory + '/noise.csv' if noise else None,
            temperature_path=(directory + '/xyce_temperature.csv'
                              if temperature else None),
        )

    @classmethod
    def for_raw_source(cls, source_path: str) -> RunFiles:
        """Resolve companions for a legacy, explicitly loaded SPICE result.

        Parameters
        ----------
        source_path : str or os.PathLike
            The rawfile being loaded. Its directory becomes ``directory``
            (``'.'`` when it has none).

        Returns
        -------
        RunFiles
            A record whose ``rawfile_path`` is ``source_path``, whose
            ``noise_path`` is ``noise.csv`` beside it, and whose netlist path
            is empty.
        """
        directory = os.path.dirname(source_path) or '.'
        files = cls.resolve('', directory, raw=False, noise=True)
        return cls(directory, '', os.fspath(source_path),
                   noise_path=files.noise_path)

    def prepare(self) -> None:
        """Create the output directory, leaving existing files intact.

        Missing parent directories are created too. Raises ``OSError`` from
        the file system if creation fails.
        """
        os.makedirs(self.directory, exist_ok=True)

    def prepare_new(self) -> RunFiles:
        """Allocate an exclusive, persistent child and relocate this layout.

        ``mkdtemp`` retries collisions using exclusive mkdir; the directory
        survives failures and process exit. Resolution itself remains pure.

        Returns
        -------
        RunFiles
            A copy whose ``directory`` is a fresh ``run-*`` child of the
            original, and whose netlist, rawfile, log, noise and temperature
            paths are relocated into it (keeping their file names).

        Raises
        ------
        SimulationError
            If the parent or the child directory cannot be created.
        """
        try:
            self.prepare()
            directory = os.path.join(self.directory, os.path.basename(tempfile.mkdtemp(prefix='run-', dir=self.directory)))
        except OSError as exc:
            raise SimulationError(
                f'Cannot allocate a simulation run under {self.directory!r}: '
                f'{exc}. Check output directory permissions.'
            ) from exc

        def relocate(path):
            return (os.path.join(directory, os.path.basename(path))
                    if path is not None else None)

        return replace(
            self, directory=directory, netlist_path=relocate(self.netlist_path),
            rawfile_path=relocate(self.rawfile_path),
            logfile_path=relocate(self.logfile_path),
            noise_path=relocate(self.noise_path),
            temperature_path=relocate(self.temperature_path),
        )

    def claim_new_attempt(self) -> None:
        """Consume a unique preparation across copies and pickle round trips.

        Persisting the claim prevents an older copy of a prepared Module from
        launching into files that another copy has already used. This is a
        per-directory consumption guard, not a lock on the mutable circuit.

        Raises
        ------
        SimulationError
            If the claim marker (``.silk-run-started``) already exists or
            cannot be created; the message says to prepare the next run.
        """
        marker = os.path.join(self.directory, '.silk-run-started')
        try:
            with open(marker, 'x'):
                pass
        except OSError as exc:
            raise SimulationError(
                f'Cannot consume new-mode preparation {self.directory!r}: '
                f'{exc}. Call generate_netlist() again (or simulate()) to '
                'prepare the next run before calling run_simulator().'
            ) from exc

    def preflight_outputs(self) -> dict:
        """Check every cleanup target without changing any output.

        Parent directory symlinks are supported. Refuse output symlinks,
        non-regular files and deck aliases before replacement may begin.
        Return the ordered, deduplicated cleanup targets.

        Returns
        -------
        dict
            The selected output paths as keys (rawfile, noise, log,
            temperature order, absent ones skipped), each mapped to ``None``.

        Raises
        ------
        SimulationError
            If an output path aliases the input netlist, or exists and is
            not a regular file (including an output symlink). Nothing has
            been changed at that point.
        """
        paths = dict.fromkeys(p for p in (
            self.rawfile_path, self.noise_path, self.logfile_path,
            self.temperature_path) if p is not None)
        for path in paths:
            try:
                if os.path.abspath(path) == os.path.abspath(self.netlist_path):
                    raise OSError('output path aliases the input netlist; select distinct paths')
                try:
                    mode = os.lstat(path).st_mode
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(mode):
                    raise OSError('expected a regular file; move the directory, symlink or special file')
                if os.path.exists(self.netlist_path) and os.path.samefile(path, self.netlist_path):
                    raise OSError('output file aliases the input netlist; select distinct paths')
            except OSError as exc:
                raise SimulationError(
                    f'Cannot safely clear previous simulation output {path!r}: {exc}. '
                    'Simulator was not launched.'
                ) from exc
        return paths

    def clear_previous_outputs(self) -> None:
        """Preflight every target, then clear exactly the previous outputs.

        Missing files are fine; only the selected output paths are removed.

        Raises
        ------
        SimulationError
            If :meth:`preflight_outputs` refuses a target, or a file cannot
            be removed. The simulator has not been launched in either case.
        """
        paths = self.preflight_outputs()
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise SimulationError(
                    f'Cannot remove previous simulation output {path!r}: {exc}. '
                    'Check file/directory permissions. Simulator was not launched.'
                ) from exc
