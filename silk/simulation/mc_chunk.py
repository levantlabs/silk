"""Several Monte Carlo samples in one ngspice process.

A *chunk* is up to ``samples_per_process`` consecutive samples of one corner.
silk generates every member's ordinary single-sample deck as before, builds ONE
loop deck from the first member's deck, runs ngspice once, and then each member
goes through the unchanged per-sample pipeline with only the process launch
replaced by "take this member's private output from the chunk run"
(:meth:`ChunkRun.launch_member`, called from
:meth:`silk.simulation.backend.NgspiceBackend._launch_prepared`). Generation
staging, the data and completion gates, the offer and the acceptance are the
per-process code, not a copy of it.

The loop, per member (``remcirc; mc_source`` re-evaluates the circuit's
statistical parameters from the seed; ``destroy all`` drops the member's plots)::

    echo silk-mc BEGIN <chunk> <index>
    setseed <seed>
    remcirc
    mc_source
    <the generated saves> run
    echo silk-mc STATUS <chunk> <index> $sim_status
    <the generated set/option/setplot/write/showmod lines, write to a private file>
    echo silk-mc END <chunk> <index>
    destroy all

The chunk's transcript is cut at those markers by a strict parser; a member
whose segment is missing or unterminated is *unfinished* and silk reruns it in
its own process.
"""
import contextlib
import contextvars
from dataclasses import dataclass, field, replace
from hashlib import sha256
import os
from pathlib import Path
import re
import subprocess
from typing import NamedTuple

from silk.exceptions import NotSupportedError, SimulationError

MARK = 'silk-mc'
_MARKER = re.compile(rf'^{MARK} (BEGIN|STATUS|END) (\S+) (\d+)(?: (\S+))?\s*$')
_ALLOWED = ('save', 'set', 'option', 'setplot', 'showmod')
_current = contextvars.ContextVar('silk_mc_chunk', default=None)


class ChunkMember(NamedTuple):
    index: int      # the sample's global index in its corner
    seed: int


class ChunkUnfinished(Exception):
    """A member the chunk's process did not finish (crash, timeout, marker
    anomaly, or a standalone deck that differs from the chunk's). The caller
    reruns it once in its own process."""


@dataclass
class Segment:
    index: int
    status: str | None = None
    ended: bool = False
    lines: list = field(default_factory=list)

    @property
    def text(self):
        return ''.join(self.lines)


@dataclass
class Transcript:
    preamble: str
    segments: dict
    footer: str
    complete: bool


def _split_control(text):
    lines = text.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip().lower() == '.control')
        quit_at = next(i for i in range(start, len(lines))
                       if lines[i].strip().lower() in ('quit 0', 'quit'))
    except StopIteration:
        raise NotSupportedError(
            'samples_per_process: the generated deck has no .control block '
            'ending in quit; it cannot be turned into a multi-sample deck.') from None
    return lines[:start], lines[start + 1:quit_at]


def _strip_seed(head):
    out = []
    for line in head:
        if line.lstrip().lower().startswith('.options') and re.search(r'\bseed=\d+', line):
            line = re.sub(r'\s*\bseed=\d+', '', line)
            if len(line.split()) <= 1:
                continue
        out.append(line)
    return out


def build_chunk_deck(base_text, chunk_id, members, raw_paths):
    """The loop deck for *members*, built from the first member's deck text.

    Refuses (:class:`~silk.exceptions.NotSupportedError`) a control block that
    holds anything but silk's own ``save``/``run``/``set``/``option``/
    ``setplot``/``write``/``showmod`` lines, exactly one ``run`` and one
    ``write``.
    """
    head, body = _split_control(base_text)
    pre, post, runs, writes = [], [], 0, 0
    for line in body:
        word = line.split(None, 1)[0].lower() if line.strip() else ''
        if not word or word.startswith('*'):
            continue
        if word == 'run':
            runs += 1
        elif word == 'write':
            writes += 1
        elif word not in _ALLOWED:
            raise NotSupportedError(
                f'samples_per_process: the generated control block holds '
                f'{line.strip()!r}, which a multi-sample deck does not '
                f'support.')
        if word != 'run':
            (post if runs else pre).append(line)
    if runs != 1 or writes != 1:
        raise NotSupportedError(
            f'samples_per_process: expected one run and one write in the '
            f'control block, found {runs} and {writes}.')
    out = _strip_seed(head) + ['.control']
    for member in members:
        out += [f'echo {MARK} BEGIN {chunk_id} {member.index}',
                f'setseed {member.seed}', 'remcirc', 'mc_source']
        out += pre
        out += ['run', f'echo {MARK} STATUS {chunk_id} {member.index} $sim_status']
        for line in post:
            if line.split()[0].lower() == 'write':
                fields = line.split()
                line = ' '.join(['write', raw_paths[member.index]] + fields[2:])
            out.append(line)
        out += [f'echo {MARK} END {chunk_id} {member.index}', 'destroy all']
    out += ['quit 0', '.endc', '.end']
    return '\n'.join(out) + '\n'


def normalised_deck(text):
    """A member's deck without the two things that differ between members:
    the seed and the staged rawfile path."""
    text = re.sub(r'\bseed=\d+', 'seed=0', text)
    return re.sub(r'(?m)^write \S+', 'write X', text)


def parse_transcript(text, chunk_id, indices):
    """Cut a chunk log at its markers, strictly.

    Markers must come in order, once each, for *chunk_id*: ``BEGIN i``,
    ``STATUS i``, ``END i`` for each index. At the first anomaly (wrong chunk
    or index, duplicate, STATUS or END outside its segment) the parse stops;
    that member and every later one stay unfinished. The footer (text after
    the last ``END``) is kept only when every member ended cleanly.
    """
    expected = list(indices)
    preamble, footer = [], []
    segments = {}
    open_seg = None
    pos = 0
    anomaly = False
    for line in text.splitlines(keepends=True):
        m = _MARKER.match(line)
        if m is None:
            if anomaly:
                continue
            if open_seg is not None:
                open_seg.lines.append(line)
            elif pos == 0 and not segments:
                preamble.append(line)
            else:
                footer.append(line)
            continue
        if anomaly:
            continue
        kind, cid, idx, extra = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        if cid != chunk_id:
            anomaly = True
            continue
        if kind == 'BEGIN':
            if open_seg is not None or pos >= len(expected) or idx != expected[pos]:
                anomaly = True
                continue
            open_seg = segments[idx] = Segment(idx)
            footer = []
        elif kind == 'STATUS':
            if open_seg is None or open_seg.index != idx or open_seg.status is not None or extra is None:
                anomaly = True
                continue
            open_seg.status = extra
        else:
            if open_seg is None or open_seg.index != idx or open_seg.status is None:
                anomaly = True
                continue
            open_seg.ended = True
            open_seg = None
            pos += 1
    complete = (not anomaly and open_seg is None and pos == len(expected)
                and all(s.ended for s in segments.values()))
    if anomaly or not complete:
        footer = []
    return Transcript(''.join(preamble), segments, ''.join(footer), complete)


class ChunkRun:
    """One chunk's process and its members' private outputs."""

    def __init__(self, directory, members):
        self.directory = directory
        self.members = tuple(members)
        self.chunk_id = f'c{self.members[0].index}'
        self.current = None
        self.launched = False
        self.error = None
        self.transcript = None
        self.base = None            # normalised deck of the first member
        self.digest = None
        self.deck_path = os.path.join(directory, 'chunk.cir')
        self.log_path = os.path.join(directory, 'chunk.log')
        self.timed_out = False

    def raw_path(self, index):
        return os.path.join(self.directory, f's{index:04d}.raw')

    # -- launching ------------------------------------------------------
    def _launch_chunk(self, run_files, text, timeout, debug):
        from silk.simulation.executor import linux_preexec_fn
        from silk.simulation.runner import ngspice_executable, warn_version_advice
        self.launched = True
        plan = run_files.completion
        # Only a provably op-only deck is admitted without a plan. Campaigns
        # decide this before anything is written (ModuleTestbench.
        # _chunk_admission_reason); a deck that still arrives here without a
        # certificate is left to run one per process. A plan that does not
        # describe this very deck is never trusted.
        from silk.simulation.completion import OP_ONLY_REASON
        if plan is None and run_files.completion_unverified_reason != OP_ONLY_REASON:
            self.error = ChunkUnfinished(
                'the sample deck has no completion certificate '
                f'({run_files.completion_unverified_reason or "no reason recorded"}) '
                'and is not provably op-only, so it is run in its own process')
            return
        if plan is not None and (plan.simulator != 'ngspice'
                                 or plan.digest != sha256(text.encode()).digest()):
            self.error = NotSupportedError(
                'samples_per_process: the sample deck has a completion '
                'certificate that does not match its bytes, so a '
                'multi-sample run could not check that its analyses finished.')
            return
        os.makedirs(self.directory, exist_ok=True)
        for member in self.members:
            Path(self.raw_path(member.index)).unlink(missing_ok=True)
        Path(self.log_path).unlink(missing_ok=True)
        try:
            deck = build_chunk_deck(
                text, self.chunk_id, self.members,
                {m.index: self.raw_path(m.index) for m in self.members})
        except NotSupportedError as exc:
            self.error = exc
            return
        Path(self.deck_path).write_text(deck)
        self.digest = sha256(deck.encode()).digest()
        self.base = normalised_deck(text)
        ngspice = ngspice_executable()
        warn_version_advice(ngspice)
        budget = None if timeout is None else timeout * len(self.members)
        try:
            subprocess.run([ngspice, '-b', '-o', self.log_path, self.deck_path],
                           capture_output=not debug, timeout=budget,
                           preexec_fn=linux_preexec_fn())
        except FileNotFoundError:
            self.error = SimulationError(
                f"ngspice executable not found ({ngspice!r}). Set "
                f"SILK_NGSPICE_BIN (environment or the silk settings file, "
                f"see docs/settings.md) or put ngspice on PATH.")
            return
        except subprocess.TimeoutExpired:
            self.timed_out = True
        except OSError as exc:
            self.error = SimulationError(f'Cannot launch ngspice ({ngspice!r}): {exc}')
            return
        try:
            on_disk = sha256(Path(self.deck_path).read_bytes()).digest()
        except OSError:
            on_disk = None
        if on_disk != self.digest:
            self.error = SimulationError(
                f'The multi-sample deck {self.deck_path} changed while ngspice '
                f'ran; its results are not trusted. See {self.log_path}.')
            return
        try:
            log = Path(self.log_path).read_text(errors='replace')
        except OSError:
            log = ''
        self.transcript = parse_transcript(
            log, self.chunk_id, [m.index for m in self.members])

    def launch_member(self, run_files, *, debug=False, timeout=None):
        """Stand in for the ngspice launch of the current member: stage its
        log and rawfile from the chunk run and load them as the per-process
        runner would. Returns ``(RawResult, log_path, None)``."""
        from silk.simulation.backend import RawResult
        from silk.simulation.completion import CompletionPlan
        from silk.simulation.runner import NgspiceRunner, raise_no_rawfile
        index = self.current
        text = Path(run_files.netlist_path).read_text()
        if not self.launched:
            self._launch_chunk(run_files, text, timeout, debug)
        if self.error is not None:
            raise self.error
        member = next(m for m in self.members if m.index == index)
        seed = re.search(r'\bseed=(\d+)', text)
        if (normalised_deck(text) != self.base
                or seed is None or int(seed.group(1)) != member.seed):
            raise ChunkUnfinished(
                f'sample {index}: its deck differs from the multi-sample deck')
        segment = self.transcript.segments.get(index)
        t = self.transcript
        if segment is None or not segment.ended:
            partial = '' if segment is None else segment.text
            Path(run_files.logfile_path).write_text(t.preamble + partial)
            raise ChunkUnfinished(f'sample {index}: not finished by its process')
        Path(run_files.logfile_path).write_text(t.preamble + segment.text + t.footer)
        # The member really ran: let a failure publish its own mc_NNNN/output.log.
        from silk.simulation.generation import _active
        generation = _active.get()
        if generation is not None:
            generation.launched = True
        raw = self.raw_path(index)
        if os.path.isfile(raw) and os.path.getsize(raw) > 0:
            os.replace(raw, run_files.rawfile_path)
        if segment.status != '0':
            _raise_failed_status(segment.status, index, run_files)
        if not os.path.isfile(run_files.rawfile_path):
            raise_no_rawfile(run_files.rawfile_path, run_files.logfile_path)
        runner = NgspiceRunner(run_files.netlist_path, run_files.directory,
                               run_files=run_files)
        runner._generation_managed = True
        native = runner.load()
        bound = run_files
        if run_files.completion is not None:
            _require_one_plot_per_request(native, run_files.completion, index,
                                          run_files.rawfile_path)
            bound = replace(run_files, completion=replace(
                run_files.completion, digest=self.digest, deck_path=self.deck_path))
        result = RawResult(native=native, source_path=run_files.rawfile_path,
                           run_files=bound)
        return result, runner.logfile_path, None


def _raise_failed_status(status, index, run_files):
    """The sample-local typed failure for a member the simulator reported as
    failed: the convergence error when its log says so, else
    :class:`~silk.exceptions.SimulationAborted`. Data rows are not inspected
    here, so they are reported as unknown."""
    from silk.exceptions import SimulationAborted
    from silk.simulation.convergence import convergence_error, read_log_text
    from silk.simulation.runner import abort_reason_from_log
    log = run_files.logfile_path
    reason = abort_reason_from_log(log)
    message = (f'ngspice sample {index} of the multi-sample process ended with '
               f'simulator status {status}, not 0 (its rawfile, if any, is not '
               f'trusted)' + (f': {reason}' if reason else '') + f'. See {log}.')
    conv = convergence_error('ngspice', read_log_text(log), message,
                             reason=reason, data_rows=None, log_path=log)
    if conv is not None:
        raise conv
    raise SimulationAborted(message, simulator='ngspice', reason=reason,
                            data_rows=None, log_path=log)


def _require_one_plot_per_request(native, plan, index, rawfile_path):
    """Chunk acceptance needs every certified request tied to exactly one
    plot: with two matching plots the shared completion check gives no
    endpoint verdict, which a multi-sample run does not accept."""
    from silk.simulation.completion import plot_silk_analysis
    plots = getattr(native, 'plots', None) or []
    for request in plan.requests:
        matching = [p for p in plots if plot_silk_analysis(p) == request.kind]
        if request.kind == 'noise':
            matching = [p for p in matching if p.vectors
                        and p.vectors[0].name.lower().startswith('freq')]
        if len(matching) > 1:
            raise SimulationError(
                f'sample {index} wrote {len(matching)} {request.kind} plots, so '
                f'silk cannot tie its requested {request.kind} analysis to one '
                f'of them and cannot check that it finished. Rawfile: '
                f'{rawfile_path}.')


def current():
    return _current.get()


@contextlib.contextmanager
def active(chunk, index):
    """Make *chunk* the launch target, for member *index*."""
    chunk.current = index
    token = _current.set(chunk)
    try:
        yield chunk
    finally:
        _current.reset(token)
