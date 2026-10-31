"""Parent-owned SPICE artifact generations and immutable worker offers.

Public paths are provenance. A generation resolves the actual artifact under a
read lock, including after replacement relocates it into private retention.
Workers can request a replacement and offer closed files; only the parent can
accept, publish, relocate or release them. Interrupted offers remain on disk.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
import os
import re
from pathlib import Path
import shutil
import tempfile
import threading
from uuid import uuid4
import weakref

from silk.exceptions import SimulationError

_KINDS = ('rawfile_path', 'noise_path', 'logfile_path', 'temperature_path')
_lock = threading.RLock()
_owner = os.getpid()
_origin_pid = _owner
_live = weakref.WeakValueDictionary()
_pending = {}
_endpoint = None
_owners = weakref.WeakKeyDictionary()
_replacements = {}
# Failed allocations this process knows about, pruned after a later accepted
# attempt in the same folder (TASK-586): {ident: record}. Parent-owned.
_failed = {}
_sequence = 0
_attempt = ContextVar('generation_attempt', default=None)
_replacing = ContextVar("generation_replacing", default=None)
_active = ContextVar('active_run_generation', default=None)
_FAILURE_RPC_TIMEOUT = 1.0
_SERVER_CONNECTION_TIMEOUT = 10.0


class _DeadlineConnection:
    """Small Connection-compatible frames with a deadline on every socket read.

    Connection.poll() alone cannot bound authentication or a partial frame.
    Failure cleanup exchanges only small method/identity and status messages.
    Use multiprocessing's own mutual-authentication implementation unchanged.
    """

    def __init__(self, sock, deadline):
        self.sock, self.deadline = sock, deadline

    def _remaining(self):
        from time import monotonic
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Generation failure cleanup timed out')
        self.sock.settimeout(remaining)

    def send_bytes(self, data):
        import struct
        self._remaining()
        self.sock.sendall(struct.pack('!i', len(data)) + data)

    def _read(self, size):
        data = bytearray()
        while len(data) < size:
            self._remaining()
            block = self.sock.recv(size - len(data))
            if not block:
                raise EOFError('Generation coordinator disconnected')
            data.extend(block)
        return bytes(data)

    def recv_bytes(self, maxlength=None):
        import struct
        size, = struct.unpack('!i', self._read(4))
        if size < 0 or size > min(maxlength or 1048576, 1048576):
            raise OSError('Invalid generation cleanup response length')
        return self._read(size)


def _failure_rpc(address, key, method, args):
    import pickle
    import socket
    from time import monotonic
    from multiprocessing.connection import answer_challenge, deliver_challenge

    deadline = monotonic() + _FAILURE_RPC_TIMEOUT
    with socket.create_connection(address, timeout=_FAILURE_RPC_TIMEOUT) as sock:
        connection = _DeadlineConnection(sock, deadline)
        answer_challenge(connection, key)
        deliver_challenge(connection, key)
        connection.send_bytes(pickle.dumps((method, args)))
        return pickle.loads(connection.recv_bytes())


def _hash(path):
    digest = sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(path):
    from silk.signals.rawfile import _file_identity
    return _file_identity(os.stat(path))


class Generation:
    def __init__(self, ident, public, staged, directory):
        self.ident, self.public, self.staged = ident, public, staged
        self.directory = directory
        self.paths = {kind: getattr(staged, kind) for kind in _KINDS}
        self.paths['netlist_path'] = staged.netlist_path
        self.paths['draws_path'] = os.path.join(staged.directory, os.path.basename(staged.netlist_path) + '.res')
        self.manifest = None
        self.accepted = False
        self.owner = _owner

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (_restore, (self.ident, self.__dict__))

    @contextmanager
    def read(self):
        if getattr(self, 'transferred', False) and os.getpid() == self.owner:
            raise SimulationError('Generation ownership was transferred; reload accepted evidence in its owner process.')
        if os.getpid() != self.owner and self.accepted:
            raise SimulationError('A worker cannot read retained parent evidence without a read lease.')
        with _lock:
            yield self

    def resolve(self, kind):
        path = self.paths.get(kind)
        if self.manifest is not None and self.manifest.get(kind) is None:
            return None
        if self.manifest is not None and self.manifest.get(kind) is not None:
            try:
                if _identity(path) != self.manifest[kind][2]:
                    raise OSError('artifact changed')
            except OSError as exc:
                raise SimulationError(
                    f'Generation {self.ident}, {kind}: unavailable retained artifact '
                    f'{path!r}: {exc}. Preserve original outputs or rerun.') from exc
        return path

    def __del__(self):
        # Unaccepted/crashed attempts remain identifiable. Public output lives
        # independently of its last reader; only private artifacts are released.
        if self.accepted and os.getpid() == self.owner and not getattr(self, 'transferred', False):
            try:
                shutil.rmtree(self.directory)
                Path(self.staged.netlist_path).unlink(missing_ok=True)
                Path(self.paths['draws_path']).unlink(missing_ok=True)
            except (OSError, ImportError):
                pass


def _restore(ident, state):
    if os.getpid() == _owner:
        existing = _live.get(ident) or _pending.get(ident)
        if existing is not None:
            return existing
    gen = Generation.__new__(Generation)
    gen.__dict__.update(state)
    return gen


def _invoke(method, *args):
    global _owner, _live, _pending, _lock
    if os.getpid() != _owner and _endpoint is None:
        # Independently launched processes may own fresh destinations, but
        # cannot relocate artifacts whose leases still belong to a parent.
        if method != 'begin':
            raise SimulationError('No generation coordinator owns this worker attempt.')
        targets = {os.path.realpath(p) for kind in _KINDS
                   if (p := getattr(args[0], kind)) is not None}
        inherited = [g for g in _live.values() if any(
            p and os.path.realpath(p) in targets for p in g.paths.values())]
        if inherited:
            raise SimulationError('Independent worker cannot replace parent-leased outputs; use run_tasks() or fresh output paths.')
        _owner = os.getpid()
        _live, _pending, _lock = weakref.WeakValueDictionary(), {}, threading.RLock()
        _failed.clear()
    if os.getpid() == _owner:
        with _lock:
            return globals()['_' + method](*args)
    if _endpoint is None:
        raise SimulationError('Worker has no parent generation coordinator; no simulation was launched.')
    from multiprocessing.connection import Client
    address, key = _endpoint
    if method in ('publish_failure_log', 'abandon'):
        ok, result = _failure_rpc(address, key, method, args)
    else:
        with Client(address, authkey=key) as connection:
            connection.send((method, args))
            ok, result = connection.recv()
    if not ok:
        error = SimulationError(result[0])
        error.replacement_started = result[1]
        raise error
    return result


@contextmanager
def worker_coordinator():
    """Serve artifact requests while this executor owns its worker pool."""
    global _endpoint
    if _endpoint is not None:
        yield lambda: None
        return
    from multiprocessing import AuthenticationError
    from multiprocessing.connection import deliver_challenge, answer_challenge
    import pickle
    import socket
    from time import monotonic
    key = os.urandom(32)
    # Workers can connect together while the service handles a prior request.
    # A full TCP accept queue can leave a client waiting forever for the
    # authentication challenge, even after the service drains the queue.
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(socket.SOMAXCONN)
    listener.settimeout(0.1)
    address = listener.getsockname()
    _endpoint = (address, key)
    prior_pending = set(_pending)
    prior_attempts = set(_replacements)
    stopping = threading.Event()
    def serve():
        while not stopping.is_set():
            try:
                client, _ = listener.accept()
            except OSError:
                # Persistent accept failures must not spin while awaiting stop.
                if stopping.wait(0.05):
                    break
                continue
            with client:
                connection = _DeadlineConnection(
                    client, monotonic() + _SERVER_CONNECTION_TIMEOUT)
                try:
                    if stopping.is_set():
                        break
                    deliver_challenge(connection, key)
                    answer_challenge(connection, key)
                    request = pickle.loads(connection.recv_bytes())
                    # Shutdown may have begun during authentication or a partial
                    # request. Never dispatch a late request after teardown.
                    if stopping.is_set() or request is None:
                        break
                    method, args = request
                    try:
                        result = _invoke(method, *args)
                        response = (True, result)
                    except Exception as exc:
                        response = (False, (str(exc), getattr(exc, 'replacement_started', False)))
                    # Artifact hashing may legitimately outlast the receive
                    # deadline; the response gets its own bounded write window.
                    connection.deadline = monotonic() + _SERVER_CONNECTION_TIMEOUT
                    connection.send_bytes(pickle.dumps(response))
                except (OSError, EOFError, AuthenticationError, pickle.UnpicklingError):
                    # A failed handshake/frame belongs to this client only.
                    stopping.wait(0.05)
    thread = threading.Thread(target=serve, daemon=True, name='silk-generation-coordinator')
    try:
        yield thread.start
    finally:
        stopping.set()
        if thread.is_alive():
            try:
                # Wake accept without authenticating or awaiting a response.
                # This remains bounded even if the service stopped answering.
                with socket.create_connection(address, timeout=1.0):
                    pass
            except OSError:
                pass
        if thread.ident is not None:
            thread.join(timeout=_SERVER_CONNECTION_TIMEOUT + 1.0)
        listener.close()
        _endpoint = None
        for ident in set(_pending) - prior_pending:
            _pending.pop(ident, None)
        for ident in set(_replacements) - prior_attempts:
            _replacements.pop(ident, None)


@contextmanager
def replacement_callback(callback):
    token = _replacing.set(callback)
    try:
        yield
    finally:
        _replacing.reset(token)


def begin(files):
    try:
        gen = _invoke('begin', files, _attempt.get())
    except Exception as exc:
        if getattr(exc, 'replacement_started', False) and _replacing.get():
            _replacing.get()()
        raise
    if _replacing.get():
        _replacing.get()()
    return gen


def _begin(files, attempt_id=None):
    global _sequence
    if files.logfile_path is None:
        files = replace(files, logfile_path=os.path.join(files.directory, 'output.log'))
    public = files
    files = replace(files, directory=os.path.abspath(files.directory),
                    netlist_path=os.path.abspath(files.netlist_path),
                    **{kind: os.path.abspath(path) if (path := getattr(files, kind)) is not None else None
                       for kind in _KINDS})
    files.preflight_outputs()
    try:
        original_bytes = Path(files.netlist_path).read_bytes()
        text = original_bytes.decode()
    except (OSError, UnicodeError) as exc:
        raise SimulationError(f'Cannot read prepared deck {files.netlist_path!r}: {exc}. Simulator was not launched.') from exc
    ident = uuid4().hex
    directory = tempfile.mkdtemp(prefix='.silk-generation-', dir=files.directory)
    _sequence += 1
    # Deletion targets are canonical at allocation: a parent symlink retargeted
    # later must never redirect cleanup (TASK-586).
    sequence, folder = _sequence, os.path.realpath(files.directory)
    # Keep the producing deck alongside its original so relative includes keep
    # both the original deck directory and the original simulator cwd.
    # Public names may already approach NAME_MAX; the private identity does
    # not need to repeat them (Generation.public retains their provenance).
    deck = os.path.join(os.path.dirname(files.netlist_path), '.silk-' + ident + '.cir')
    owned = (os.path.realpath(directory), os.path.join(os.path.realpath(os.path.dirname(deck)), os.path.basename(deck)))
    try:
        paths = {kind: (os.path.join(directory, os.path.basename(getattr(files, kind)))
                        if getattr(files, kind) is not None else None) for kind in _KINDS}
        rewrites = {}
        for kind in _KINDS:
            new = paths[kind]
            if new is not None:
                rewrites[getattr(public, kind)] = new
                rewrites[getattr(files, kind)] = new
        if rewrites:
            pattern = '|'.join(re.escape(path) for path in sorted(rewrites, key=len, reverse=True))
            text = re.sub(pattern, lambda match: rewrites[match.group()], text)
        Path(deck).write_text(text)
        completion = files.completion
        if completion is not None and completion.digest == sha256(original_bytes).digest():
            # The requests remain untouched; bind them to the actual staged bytes.
            completion = replace(completion, digest=sha256(Path(deck).read_bytes()).digest())
        staged = replace(files, netlist_path=deck, completion=completion, **paths)
        gen = Generation(ident, public, staged, directory)
        gen.targets = files
        gen.sequence, gen.folder, gen.owned = sequence, folder, owned
        gen.attempt_id = attempt_id
        if attempt_id is not None:
            _replacements[attempt_id] = True
        targets = {os.path.realpath(path) for kind in _KINDS
                   if (path := getattr(files, kind)) is not None}
        from silk.circuit.modules_sim import _invalidate_current_results
        for module, reference in list(_owners.items()):
            old = reference()
            if old is not None and any(path and os.path.realpath(path) in targets for path in old.paths.values()):
                if getattr(getattr(module, '_raw', None), '_generation', None) is old:
                    _invalidate_current_results(module)
                del _owners[module]
        try:
            # All safety checks already passed, before ANY public mutation.
            for old in list(_live.values()):
                for kind in _KINDS:
                    path = old.paths.get(kind)
                    target = getattr(files, kind)
                    if path and target and os.path.realpath(path) == os.path.realpath(target):
                        if old.manifest.get(kind) is not None:
                            record = old.manifest[kind]
                            intact = os.path.exists(path) and _identity(path) == record[2]
                            if not os.path.exists(path):
                                continue
                            private = os.path.join(old.directory, os.path.basename(path))
                            os.replace(path, private)
                            old.paths[kind] = private
                            size, digest, _ = old.manifest[kind]
                            if intact:
                                old.manifest[kind] = (size, digest, _identity(private))
            files.clear_previous_outputs()
        except Exception as exc:
            error = SimulationError(f'Cannot begin generation replacement: {exc}')
            error.replacement_started = True
            raise error from exc
        _pending[ident] = gen
        return gen
    except BaseException:
        # Whatever was allocated before the failure is owned by this attempt;
        # a later accepted attempt in the folder removes it (no globbing).
        _register_failed(ident, folder, sequence, *owned)
        raise


def offer(gen):
    manifest = {}
    for kind, path in gen.paths.items():
        if path is None or not os.path.exists(path):
            manifest[kind] = None
        else:
            before = _identity(path)
            digest = _hash(path)
            if before != _identity(path):
                raise SimulationError(f'Generation {gen.ident}: artifact changed while offering {path!r}')
            manifest[kind] = (before[2], digest, before)
    return _invoke('offer', gen.ident, manifest, getattr(gen, 'launched', False))


def _offer(ident, manifest, launched=False):
    gen = _pending[ident]
    if gen.manifest is not None and gen.manifest != manifest:
        raise SimulationError(f'Generation {ident} already offered different artifacts')
    if set(manifest) != set(gen.paths):
        raise SimulationError(f'Generation {ident}: incomplete artifact manifest.')
    for kind, record in manifest.items():
        path = gen.paths[kind]
        if record is not None and (_identity(path) != record[2] or _hash(path) != record[1]):
            raise SimulationError(f'Generation {ident}: invalid artifact offer {path!r}')
    gen.manifest = manifest
    gen.launched = getattr(gen, 'launched', False) or launched
    return gen


def accept(raw):
    gen = getattr(raw, '_generation', None)
    fresh = gen is not None and not gen.accepted
    try:
        result = _accept(raw)
    except BaseException as exc:
        gen = getattr(raw, '_generation', None)
        publish_failure_log(gen, exc)
        abandon(gen)
        if isinstance(exc, OSError):
            error = SimulationError(f'Cannot accept/publish simulation generation: {exc}. Previous public results were not restored.')
            publish_failure_log(gen, error)
            raise error from exc
        raise
    if fresh and gen.accepted and os.getpid() == _owner:
        # Only after publication and the native reload both succeeded.
        _prune_failed(gen)
    return result


def _register_failed(ident, folder, sequence, directory, deck):
    """Remember an abandoned allocation; repeated registration is harmless."""
    _failed.setdefault(ident, {'folder': folder, 'sequence': sequence,
                               'directory': directory, 'deck': deck})


def _prune_failed(accepted):
    """Remove earlier failed allocations of the accepted attempt's folder.

    The directory and the deck go independently; a record whose removal failed
    is kept and retried on the next success. A cleanup error never reaches the
    caller. Accepted and in-flight generations are never in ``_failed``.
    """
    with _lock:
        for ident, record in list(_failed.items()):
            if (record['folder'] != accepted.folder or record['sequence'] >= accepted.sequence
                    or ident in _live or ident in _pending):
                continue
            if record['directory'] is not None:
                try:
                    shutil.rmtree(record['directory'])
                except FileNotFoundError:
                    record['directory'] = None
                except OSError:
                    pass
                else:
                    record['directory'] = None
            if record['deck'] is not None:
                try:
                    Path(record['deck']).unlink(missing_ok=True)
                except OSError:
                    pass
                else:
                    record['deck'] = None
            if record['directory'] is None and record['deck'] is None:
                del _failed[ident]


def _accept(raw):
    gen = getattr(raw, '_generation', None)
    if gen is None:
        return raw
    if gen.accepted:
        if getattr(gen, 'transferred', False) and os.getpid() != gen.owner:
            with _lock:
                for kind, record in gen.manifest.items():
                    if record is not None and _hash(gen.resolve(kind)) != record[1]:
                        raise SimulationError(f'Generation {gen.ident}: transferred artifact changed.')
                gen.owner = os.getpid()
                gen.transferred = False
                _live[gen.ident] = gen
        return _load_native(raw)
    if os.getpid() != _owner:
        return raw  # worker offers; the parent consumes its outcome
    with _lock:
        gen = _pending.get(gen.ident, gen)
        # Worker-pool teardown releases pending metadata; the handed-off result
        # still owns this offer and must support failure-log publication here.
        _pending[gen.ident] = gen
        gen.targets.preflight_outputs()
        for kind in _KINDS:
            path = getattr(gen.targets, kind)
            if path and os.path.lexists(path):
                raise SimulationError(f'Generation {gen.ident}: intervening output at {path!r}; refusing publication.')
        for kind, record in gen.manifest.items():
            if record is not None and _hash(gen.resolve(kind)) != record[1]:
                raise SimulationError(f'Generation {gen.ident}: offered artifact changed before acceptance.')
        for kind in _KINDS:
            record = gen.manifest.get(kind)
            if record is not None:
                public = getattr(gen.targets, kind)
                os.replace(gen.paths[kind], public)
                gen.paths[kind] = public
                gen.manifest[kind] = (record[0], record[1], _identity(public))
        gen.accepted = True
        _live[gen.ident] = gen
        _pending.pop(gen.ident, None)
        raw._generation = gen
    return _load_native(raw)


def _load_native(raw):
    if getattr(raw, '_needs_generation_read', False):
        from silk.signals.rawfile import read_rawfile
        with raw._generation.read():
            raw.native = read_rawfile(raw._generation.resolve('rawfile_path'), lazy=True)
            attach(raw, raw._generation)
        raw._needs_generation_read = False
    return raw


def attach(raw, gen):
    raw._generation = gen
    native = raw.native
    if native is not None:
        native._generation = gen
        for plot in native.plots:
            for vector in plot.vectors:
                if vector._store is not None:
                    vector._store.generation = gen
    raw.run_files = gen.public
    raw.source_path = (gen.public.rawfile_path if native is not None else gen.public.noise_path)
    return raw


def resolve_artifact(raw, kind):
    gen = getattr(raw, '_generation', None)
    if gen is not None:
        return gen.resolve(kind)
    files = getattr(raw, 'run_files', None)
    return getattr(files, kind, None) if files is not None else None


def retained_source(owner):
    reader = owner._result_netlist() if hasattr(owner, '_result_netlist') else owner
    raw = getattr(reader, '_raw', None)
    if getattr(raw, '_generation', None) is not None:
        return resolve_artifact(raw, 'rawfile_path')
    return getattr(reader, 'results_file', None)


def read_retained(owner):
    """Eager view snapshot: hold the generation read lease through disk access."""
    from silk.signals.rawfile import read_rawfile, RawfileError
    reader = owner._result_netlist() if hasattr(owner, '_result_netlist') else owner
    raw = getattr(reader, '_raw', None)
    gen = getattr(raw, '_generation', None)
    try:
        if gen is not None:
            with gen.read():
                return read_rawfile(gen.resolve('rawfile_path'))
        return read_rawfile(retained_source(owner))
    except OSError as exc:
        raise RawfileError(f'Cannot read result association for {retained_source(owner)!r}: {exc}') from exc


@contextmanager
def artifact_read(raw, kind):
    """Hold a managed read lease throughout the caller's companion read."""
    gen = getattr(raw, '_generation', None)
    if gen is None:
        yield resolve_artifact(raw, kind)
    else:
        with gen.read():
            yield gen.resolve(kind)


@contextmanager
def attempt_scope(attempt_id):
    token = _attempt.set(attempt_id)
    try:
        yield
    finally:
        _attempt.reset(token)
        if os.getpid() == _owner:
            _replacements.pop(attempt_id, None)


def replacement_for(attempt_id):
    return _replacements.pop(attempt_id, None)


def register_owner(module, raw):
    gen = getattr(raw, '_generation', None)
    if gen is not None:
        _owners[module] = weakref.ref(gen)


def transfer_result(raw):
    """Relinquish only at the explicit independent-worker result handoff."""
    gen = getattr(raw, '_generation', None)
    if gen is not None and gen.accepted and os.getpid() == gen.owner and os.getpid() != _origin_pid:
        gen.transferred = True


def abandon(gen):
    """Release attempt metadata, preserving identifiable failed artifacts."""
    if (gen is not None and not gen.accepted
            and (os.getpid() == _owner or not getattr(gen, 'cleanup_unavailable', False))):
        try:
            _invoke('abandon', gen.ident)
        except Exception:
            # Cleanup diagnostics must never replace the original run failure.
            gen.cleanup_unavailable = True


def publish_failure_log(gen, error):
    """Publish only diagnostics, including for a cancelled worker attempt."""
    if gen is None or gen.accepted or not getattr(gen, 'launched', False):
        return
    outcome = getattr(gen, 'failure_log_outcome', None)
    if outcome is None:
        if getattr(gen, 'cleanup_unavailable', False):
            return
        try:
            public = (gen.public.logfile_path if getattr(gen, 'failure_log_published', False)
                      else _invoke('publish_failure_log', gen.ident))
        except (OSError, EOFError) as exc:
            # Never replace cancellation/the original simulation error with a
            # dead coordinator error. Leave the attempt identifiable on disk.
            path = gen.staged.logfile_path
            if os.getpid() == _owner:
                outcome = (path, f'log publication refused: {exc}; log retained at {path}')
            else:
                gen.cleanup_unavailable = True
                outcome = (path, f'Diagnostic publication unavailable; log retained at {path}.')
        except Exception as exc:
            # Refusals (including not-ok coordinator replies) are secondary to
            # the attempt's failure. Keep abandon reachable during cancellation.
            path = gen.staged.logfile_path
            outcome = (path, f'log publication refused: {exc}; log retained at {path}')
        else:
            if public is None:
                outcome = (None, 'no diagnostic log was written')
            else:
                gen.failure_log_published = True
                outcome = (public, None)
        # Nested runner/backend wrappers may encounter the same exception after
        # abandon has released coordinator metadata. Never retry publication.
        gen.failure_log_outcome = outcome
    if isinstance(error, SimulationError):
        path, note = outcome
        error.log_path = path
        message = str(error)
        if note is not None:
            if note not in message:
                message += f' {note}'
        else:
            message = message.replace(gen.staged.logfile_path, path)
            if path not in message:
                message += f' See {path} for full output.'
        error.args = (message, *error.args[1:])


def _publish_failure_log(ident):
    gen = _pending[ident]
    public = gen.public.logfile_path
    if not getattr(gen, 'failure_log_published', False):
        if not os.path.lexists(gen.staged.logfile_path):
            return None
        target = gen.targets.logfile_path
        gen.targets.preflight_outputs()
        if os.path.lexists(target):
            raise SimulationError(f'Intervening output at {target!r}; refusing log publication.')
        os.replace(gen.staged.logfile_path, target)
        gen.failure_log_published = True
    return public


def _abandon(ident):
    gen = _pending.pop(ident, None)
    if gen is not None and getattr(gen, 'folder', None) is not None:
        _register_failed(ident, gen.folder, gen.sequence, *gen.owned)
