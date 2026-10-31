"""Per-test run information: the launch record and the failure result.

:class:`RunRecord` is plain, picklable data naming the files of one simulator
launch. It is captured where the launch happens and carried by the existing
outcome/result machinery, so serial and parallel runs expose the same facts.
:class:`TestFailure` is what ``run_all_tests`` returns for a test that raised.
"""
import os
import pickle
from dataclasses import dataclass, fields, replace
from uuid import uuid4

__all__ = ['RunRecord', 'TestFailure']

_PATH_FIELDS = ('deck_path', 'log_path', 'rawfile_path', 'noise_path', 'directory')
_DETAIL_FIELDS = ('simulator', 'reason', 'data_rows', 'log_path')


@dataclass(frozen=True)
class RunRecord:
    """The files of one simulator launch, as absolute paths.

    ``deck_path`` is the prepared deck at its public path (not the private
    staged copy the simulator literally read). ``log_path`` is the public log,
    or, when a failed run's log could not be published, the retained private
    log (the same value as ``SimulationError.log_path``). A backend that writes
    no such file leaves it ``None``. ``launch_id`` is equal for every report
    that shares one physical launch. The files are replaced by the next run
    into the same folder.
    """
    launch_id: str
    deck_path: str = None
    log_path: str = None
    rawfile_path: str = None
    noise_path: str = None
    directory: str = None

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data, path='$'):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError(f'{path}: expected a run record with fields '
                             f'{[f.name for f in fields(cls)]}')
        for name, value in data.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f'{path}.{name}: expected a string or null')
        if not data['launch_id']:
            raise ValueError(f'{path}.launch_id: expected a non-empty string')
        return cls(**data)


def capture_run_record(netlist, *, launch_id=None, log_path=None, files=None):
    """Record the files of the current attempt now, as absolute paths.

    *files* are an in-process backend's own concrete files (the behavioral
    engine writes a rawfile and nothing else): the deck and log are then absent,
    and ``netlist._run_files`` (a possibly earlier SPICE preparation) is never
    consulted. Otherwise the module's public SPICE files are read.
    """
    in_process = files is not None
    if not in_process:
        files = getattr(netlist, '_run_files', None)

    def absolute(path):
        return None if path is None else os.path.abspath(os.fspath(path))

    return RunRecord(
        launch_id=launch_id or uuid4().hex,
        deck_path=None if in_process else absolute(getattr(files, 'netlist_path', None) or None),
        log_path=None if in_process else absolute(log_path or getattr(files, 'logfile_path', None)),
        rawfile_path=absolute(getattr(files, 'rawfile_path', None)),
        noise_path=absolute(getattr(files, 'noise_path', None)),
        directory=absolute(getattr(files, 'directory', None)))


def error_class_of(error):
    cls = type(error)
    return f'{cls.__module__}.{cls.__qualname__}'


def failure_facts(error):
    """``(error_type, error_class, details)`` read from the original error."""
    from silk.exceptions import SimulationAborted
    details = {}
    if isinstance(error, SimulationAborted):
        details = {name: getattr(error, name, None) for name in _DETAIL_FIELDS}
    return type(error).__name__, error_class_of(error), details


def pack_exception(error):
    """``(packed, fallback)``: *error* as ``(cls, args, attrs)`` that survives
    pickling, or the nearest reconstructible ancestor with only its text.

    *fallback* is True when the original class or attributes were given up;
    a stand-in never passes for the original. ``(None, True)`` when nothing
    could be carried.
    """
    cls = type(error)
    try:
        reduced = error.__reduce__()
        args = reduced[1] if reduced[0] is cls else error.args
        packed = (cls, args, dict(error.__dict__))
        pickle.dumps(packed)
        return packed, False
    except Exception:
        pass
    for base in cls.__mro__:
        if base is BaseException:
            base = RuntimeError
        candidate = (base, (str(error),), {})
        try:
            probe = base(str(error))
            restored = pickle.loads(pickle.dumps(probe))
            if str(restored) != str(error):
                continue
            pickle.dumps(candidate)
        except Exception:
            continue
        return candidate, True
    return None, True


def unpack_exception(packed):
    """Rebuild an exception from :func:`pack_exception`'s tuple."""
    if packed is None:
        return None
    cls, args, attrs = packed
    try:
        error = cls(*args)
    except TypeError:
        # SimulationAborted/ConvergenceError require keyword-only metadata;
        # it is restored below without changing their API.
        error = cls.__new__(cls, *args)
        BaseException.__init__(error, *args)
    error.__dict__.update(attrs)
    return error


class TestFailure(dict):
    """A test that raised: equal to ``{'error': str(error)}``, plus evidence.

    Mapping and equality readers of the old ``{'error': str}`` shape keep
    working. The attributes are failure evidence captured when the test
    failed; changing ``['error']`` does not change them, and plain
    ``json.dumps`` keeps only the message (use ``silk.circuit.save_results_json``
    to keep the rest).

    Attributes
    ----------
    error_type, error_class : str
        The original exception's class name and dotted path.
    details : dict
        ``simulator``, ``reason``, ``data_rows`` and ``log_path`` for a
        :class:`~silk.exceptions.SimulationAborted`; empty otherwise.
    exception : BaseException or None
        The exception, when it could be carried (``None`` after a JSON round
        trip or when it could not cross a worker process).
    exception_is_original : bool
        ``False`` when ``exception`` is a stand-in for a class that could not
        cross a worker process; ``error_type``/``error_class``/``details``
        still describe the original.
    run : RunRecord or None
        The failed attempt's files, or ``None`` when nothing was launched or
        the launch is unknown.
    """
    __test__ = False                 # not a pytest class

    def __init__(self, error, *, error_type=None, error_class=None, details=None,
                 exception=None, run=None, exception_is_original=True):
        super().__init__(error=error)
        self.error_type = error_type
        self.error_class = error_class
        self.details = {} if details is None else dict(details)
        self.exception = exception
        self.run = run
        self.exception_is_original = exception_is_original

    def __reduce__(self):
        packed, original = None, self.exception_is_original
        if self.exception is not None:
            packed, fallback = pack_exception(self.exception)
            original = original and not fallback
        return (_rebuild_failure, (self['error'], self.error_type, self.error_class,
                                   self.details, packed, self.run, original))

    @classmethod
    def from_exception(cls, error):
        """The failure for *error*, using the evidence attached when it was raised."""
        info = getattr(error, '_silk_failure', None)
        if info is None:
            error_type, error_class, details = failure_facts(error)
            info = dict(error_type=error_type, error_class=error_class,
                        details=details, run=None, original=True)
        return cls(str(error), error_type=info['error_type'],
                   error_class=info['error_class'], details=info['details'],
                   exception=error, run=info['run'],
                   exception_is_original=info['original'])

    @classmethod
    def from_outcome(cls, outcome, run=None):
        error = outcome.error
        return cls(str(error), error_type=outcome.error_type,
                   error_class=outcome.error_class, details=outcome.details,
                   exception=error, run=outcome.run if run is None else run,
                   exception_is_original=not outcome.error_fallback)


def _rebuild_failure(error, error_type, error_class, details, packed, run, original):
    return TestFailure(error, error_type=error_type, error_class=error_class,
                       details=details, exception=unpack_exception(packed), run=run,
                       exception_is_original=original)


def _with_final_log(error, run):
    """*run* naming the log the failure itself reports.

    A worker returns before publication; if the parent's acceptance then fails
    and the log stays private, ``error.log_path`` is the retained file. The
    launch identity and the other paths are kept.
    """
    log = getattr(error, 'log_path', None)
    if run is None or not log:
        return run
    log = os.path.abspath(os.fspath(log))
    return run if log == run.log_path else replace(run, log_path=log)


def attach_failure_info(error, outcome=None, run=None):
    """Remember, on the exception about to be raised, what the failure knew."""
    if outcome is not None:
        info = dict(error_type=outcome.error_type, error_class=outcome.error_class,
                    details=outcome.details, run=_with_final_log(error, outcome.run),
                    original=not outcome.error_fallback)
    else:
        error_type, error_class, details = failure_facts(error)
        info = dict(error_type=error_type, error_class=error_class,
                    details=details, run=_with_final_log(error, run), original=True)
    try:
        error._silk_failure = info
    except Exception:
        pass
