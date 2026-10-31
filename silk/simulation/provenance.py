"""silk.simulation.provenance — one record of what produced a result.

"Why does this number differ from last month?" needs the silk version, the
simulator build, the PDK model files, the options and the deck that produced
it. :class:`RunProvenance` assembles those facts once per evaluation and is
attached to every testbench-built :class:`~silk.circuit.spec_report.SpecReport`
(``report.provenance``), rendered in circuit reports, compared by
:func:`~silk.circuit.report_diff.diff_reports`, and stored with each
measurement row in the database.

Format and stable contract
--------------------------
``scheme`` is the format version, ``'silk.provenance/v3'``. The field names in
:data:`FIELDS` are a stable contract for that scheme: a later scheme may add
fields, never rename or re-type these.

``run_origin`` says who ran the simulator:

- ``'silk-launch'`` — silk launched the simulator. ``simulator_binary``,
  ``simulator_binary_source`` and ``simulator_version`` describe the binary
  silk launched (``simulator_binary_source`` is ``'SILK_NGSPICE_BIN'``,
  ``'SILK_XYCE_BIN'``, ``'PATH'`` or ``'in-process'``).
- ``'external-rawfile'`` — the rawfile was handed to silk
  (``evaluate(rawfile=...)``, ``load_file()`` + ``read_raw_file()``). The
  three binary fields are ``None``: silk did not launch anything, so it does
  not claim a binary, and it never reads a local binary's version on this
  path. ``simulator`` is whatever the rawfile itself names (or ``None``).

``deck_sha256`` is the sha256 of the deck text silk PREPARED for this bench
(``generate_netlist()``). On a silk launch that is also the deck that ran; on
an external rawfile it is the last deck silk prepared for this bench, or
``None`` when silk prepared none. ``pdk_libraries``, ``simulator_options``,
``save_list`` and ``seed`` describe that same prepared deck.

``normalized_deck_sha256`` is an optional ``silk.normalized-deck/v1:<hex>``
identity of the accepted launch's UTF-8 deck emitted against fixed synthetic
RunFiles and the stable ``silk test`` fallback title. Only generated output
locations are normalized; absolute model/include paths remain. It is frozen
with launch facts, never recomputed when storing or evaluating loaded results.
``producing_test`` binds the logical report's exact ``owner_path``/``test_name``;
``launch_corner_label`` retains the requested launch label even on an ideal
circuit whose stored corner is none. v1/v2 records lack these facts and remain
not checked. MC, stochastic, opaque/custom, behavioral, external, manual and
failed launches supply no covered identity. Current sets are not checked.
Metric definitions/user Python, simulator versions and model contents are
outside the deck check. Ephemeral lookup verdicts are not provenance fields.

``pdk_libraries`` lists one entry per ``.lib "<file>" <section>`` line of the
prepared deck: ``file`` (relative to ``$PDK_ROOT`` when under it),
``section``, ``file_sha256`` (the bytes of that ONE file) and
``section_text_sha256`` (only the text between ``.lib <section>`` and
``.endl`` in that file). Neither hash covers the files that section itself
includes; they identify which library file and corner section were named, not
the whole model tree.

Cost
----
Nothing here touches a simulation. Environment facts are cached per process:
the silk ``git describe`` and Python/numpy versions once, a simulator's
version string once per (binary, mtime), and a library file's hashes once per
(path, mtime, size).

Privacy
-------
The record never holds a hostname, a username or an absolute home path.
Paths under ``$PDK_ROOT`` are made relative to it; any other path under the
home directory becomes ``~/...``. Inside ``simulator_options`` and
``save_list`` every string (keys and values, at any nesting depth) has EVERY
occurrence of the home directory rewritten to ``~``, mid-string included
(``'-I/home/u/x'`` -> ``'-I~/x'``), at a path boundary only (not followed
by a filename character, so ``/home/userX`` survives when home is
``/home/user``). A binary or library outside both
keeps its absolute path only under a system prefix (``/usr``, ``/opt``,
``/bin``, ``/nix``, ``/snap``); otherwise only its file name is kept.
"""
from __future__ import annotations

import datetime
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, fields
from typing import Any, List, Optional, Tuple

from silk import settings as _settings
from silk._json_codec import encode_json

SCHEME = 'silk.provenance/v3'

RUN_ORIGIN_SILK = 'silk-launch'
RUN_ORIGIN_EXTERNAL = 'external-rawfile'

#: Every field of a ``silk.provenance/v3`` record, in render order. A stable
#: contract: names are never renamed or re-typed within this scheme. v2 adds
#: ``corner_kind`` and ``supply_kind``; v3 adds producing deck/test/label facts.
#: Older records read absent facts as None, preserving their scheme.
FIELDS = (
    'scheme', 'run_origin',
    'silk_version', 'silk_git', 'python_version', 'numpy_version',
    'simulator', 'simulator_binary', 'simulator_binary_source',
    'simulator_version',
    'pdk_name', 'pdk_libraries',
    'corner', 'corner_kind', 'temperature_c', 'temperature_kind',
    'supply_voltage_v', 'supply_kind',
    'simulator_options', 'save_list', 'deck_sha256', 'seed', 'seedinfo',
    'timestamp_utc', 'normalized_deck_sha256', 'producing_test', 'launch_corner_label',
)

#: Fields left out of :meth:`RunProvenance.digest` and
#: :meth:`RunProvenance.diff` — they differ on every run by construction.
VOLATILE_FIELDS = frozenset({'timestamp_utc'})

_V3_FIELDS = frozenset({'normalized_deck_sha256', 'producing_test', 'launch_corner_label'})


def _report_field_names(record):
    """Preserve older report records without adding absent launch facts."""
    absent = (_V3_FIELDS if record.scheme in {'silk.provenance/v1', 'silk.provenance/v2'}
              else frozenset())
    if record.scheme == 'silk.provenance/v1':
        absent = absent | {'corner_kind', 'supply_kind'}
    return [field.name for field in fields(record)
            if field.name not in absent or getattr(record, field.name) is not None]

_SYSTEM_PREFIXES = ('/usr/', '/opt/', '/bin/', '/sbin/', '/nix/', '/snap/')
_GIT_TIMEOUT_S = 2
_VERSION_TIMEOUT_S = 20
_NGSPICE_RE = re.compile(r'ngspice-(\S+)')


@dataclass(frozen=True)
class RunProvenance:
    """What produced one result. See the module docstring for field meaning.

    Every field is plain data (str/int/float/bool/None, and lists/dicts of
    them), so a record pickles across the parallel executor and
    :meth:`to_dict` is directly ``json.dumps``-able.
    """
    run_origin: Optional[str] = None
    silk_version: Optional[str] = None
    silk_git: Optional[str] = None
    python_version: Optional[str] = None
    numpy_version: Optional[str] = None
    simulator: Optional[str] = None
    simulator_binary: Optional[str] = None
    simulator_binary_source: Optional[str] = None
    simulator_version: Optional[str] = None
    pdk_name: Optional[str] = None
    pdk_libraries: Optional[list] = None
    corner: Optional[str] = None
    corner_kind: Optional[str] = None
    temperature_c: Optional[float] = None
    temperature_kind: Optional[str] = None
    supply_voltage_v: Optional[float] = None
    supply_kind: Optional[str] = None
    simulator_options: Optional[dict] = None
    save_list: Optional[dict] = None
    deck_sha256: Optional[str] = None
    seed: Any = None
    seedinfo: Optional[str] = None
    timestamp_utc: Optional[str] = None
    normalized_deck_sha256: Optional[str] = None
    producing_test: Optional[dict] = None
    launch_corner_label: Optional[str] = None
    scheme: str = SCHEME

    def to_dict(self) -> dict:
        """JSON-plain fields for this scheme, in ``FIELDS`` order.

        v1/v2 retain the pre-v3 inventory, including its condition-kind
        defaults. The three producing fields belong only to v3 and later.
        """
        excluded = (_V3_FIELDS if self.scheme in {'silk.provenance/v1', 'silk.provenance/v2'}
                    else frozenset())
        return {name: _plain(getattr(self, name)) for name in FIELDS if name not in excluded}

    @classmethod
    def from_dict(cls, d: dict) -> 'RunProvenance':
        """Inverse of :meth:`to_dict`. Unknown keys are ignored (a newer
        scheme may add fields); missing ones are ``None``."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    def digest(self) -> str:
        """sha256 of canonical, non-volatile fields for the retained scheme.

        v1/v2 use exactly the pre-v3 input inventory and content addresses;
        producing facts affect only v3 addresses. Historical rows stay intact.
        """
        d = {k: v for k, v in self.to_dict().items() if k not in VOLATILE_FIELDS}
        text = json.dumps(encode_json(d), sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def diff(self, other: 'RunProvenance') -> List[Tuple[str, Any, Any]]:
        """``[(field, self value, other value), ...]`` for every differing
        field, in ``FIELDS`` order; ``timestamp_utc`` is never compared."""
        a, b = self.to_dict(), other.to_dict()
        return [(k, a.get(k), b.get(k)) for k in FIELDS
                if k not in VOLATILE_FIELDS and a.get(k) != b.get(k)]

    def summary_line(self) -> str:
        """One compact line: silk, simulator, PDK library[section], deck."""
        silk = f"silk {self.silk_version or '?'}"
        if self.silk_git:
            silk += f" ({self.silk_git})"
        if self.run_origin == RUN_ORIGIN_EXTERNAL:
            sim = f"external rawfile ({self.simulator or 'simulator unknown'})"
        else:
            sim = self.simulator_version or self.simulator or 'simulator unknown'
        parts = [silk, sim]
        libs = self.pdk_libraries or []
        if self.pdk_name or libs:
            lib_txt = ', '.join(
                f"{os.path.basename(e.get('file') or '?')}[{e.get('section')}] "
                f"{(e.get('file_sha256') or '?')[:10]}" for e in libs)
            parts.append(f"{self.pdk_name or 'PDK'}"
                         + (f" {lib_txt}" if lib_txt else ''))
        parts.append(f"deck {(self.deck_sha256 or 'none')[:12]}")
        if self.seed is not None:
            parts.append(f"seed {self.seed}")
        return ', '.join(parts)


def _plain(value):
    """Coerce to JSON-plain types (tuples -> lists, numpy scalars -> Python)."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_plain(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return _plain(item())
        except (TypeError, ValueError):
            pass
    return str(value)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def _home() -> Optional[str]:
    home = os.path.expanduser('~')
    return None if home in ('', '~', '/') else os.path.normpath(home)


def _pdk_root() -> Optional[str]:
    """Root silk resolves PDK library paths against (mirrors load_pdk)."""
    root = _settings.get('PDK_ROOT')
    return os.path.normpath(root) if root else None


def _under(path: str, root: Optional[str]) -> bool:
    return bool(root) and (path == root or path.startswith(root.rstrip('/') + '/'))


def home_relative(value: str) -> str:
    """``value`` with EVERY occurrence of the home directory rewritten to
    ``~``, wherever it sits in the string (``'-I/home/u/x'`` ->
    ``'-I~/x'``). Only at a path boundary: the match must not be followed by
    a filename character (letter, digit, ``.``, ``_``, ``-``), so ``/``, the
    end of the string, whitespace, quotes and separators all end it, and
    ``/home/userX`` is left alone when the home directory is ``/home/user``.
    (A bare ``/``-or-end rule would miss ``'-I/home/user -L...'``.)"""
    home = _home()
    if not home or not isinstance(value, str):
        return value
    return _home_pattern(home).sub('~', value)


@functools.lru_cache(maxsize=4)
def _home_pattern(home: str):
    return re.compile(re.escape(home) + r'(?![A-Za-z0-9._-])')


def share_path(path: Optional[str], *, pdk_relative: bool = False) -> Optional[str]:
    """A path safe to put in a shared record (see module docstring)."""
    if not path:
        return path
    path = os.path.normpath(os.fspath(path))
    if not os.path.isabs(path):
        return path
    if pdk_relative:
        root = _pdk_root()
        if _under(path, root) and path != root:
            return os.path.relpath(path, root)
    if _under(path, _home()):
        return home_relative(path)
    if path.startswith(_SYSTEM_PREFIXES):
        return path
    return os.path.basename(path)


def scrub(value):
    """Recursively rewrite every home-directory occurrence, in every string
    (keys included), to ``~`` -- see :func:`home_relative`."""
    if isinstance(value, str):
        return home_relative(value)
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Cached environment facts
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _environment() -> dict:
    try:
        from silk import __version__ as silk_version
    except Exception:
        silk_version = None
    numpy_version = None
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        pass
    return {
        'silk_version': silk_version,
        'silk_git': _git_describe(),
        'python_version': '.'.join(str(p) for p in sys.version_info[:3]),
        'numpy_version': numpy_version,
    }


def _git_describe() -> Optional[str]:
    """``git describe --always --dirty`` of the silk source tree, or None.

    None when git is missing, silk is not a git checkout (a wheel install),
    or the call fails or takes longer than two seconds. Never raises."""
    try:
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.exists(os.path.join(src, '.git')):
            return None
        git = shutil.which('git')
        if not git:
            return None
        text = _first_output([git, '-C', src, 'describe', '--always', '--dirty'],
                             _GIT_TIMEOUT_S)
        text = (text or '').strip()
        return text.splitlines()[0] if text else None
    except Exception:
        return None


_VERSION_CACHE: dict = {}


def _first_output(argv, timeout, *, require_ok=True) -> Optional[str]:
    """stdout+stderr of ``argv``, or None. Uses Popen directly, not
    ``subprocess.run``, so a provenance probe never shows up as a run."""
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                text=True)
    except Exception:
        return None
    try:
        out, err = proc.communicate(timeout=timeout)
    except Exception:
        proc.kill()
        proc.communicate()
        return None
    if require_ok and proc.returncode != 0:
        return None
    return f'{out}\n{err}'


def _binary_version(exe: str, simulator: str) -> Optional[str]:
    """The version string doctor reports for ``exe``; cached per (exe,
    mtime). A failed probe is not cached, so it is retried next launch."""
    try:
        key = (exe, os.path.getmtime(exe), simulator)
    except OSError:
        return None
    if key in _VERSION_CACHE:
        return _VERSION_CACHE[key]
    text = _first_output([exe, '--version' if simulator == 'ngspice' else '-v'],
                         _VERSION_TIMEOUT_S, require_ok=False)
    if text is None:
        return None
    version = None
    if simulator == 'ngspice':
        m = _NGSPICE_RE.search(text)
        version = m.group(0) if m else None
    if version is None:
        for line in text.splitlines():
            if line.strip().strip('*'):
                version = line.strip().lstrip('*').strip()
                break
    if version is not None:
        _VERSION_CACHE[key] = version
    return version


def launch_record(simulator: str) -> dict:
    """What silk launched for ``simulator``: name, binary, its source and
    version. Called once per successful silk launch. Never raises; any fact
    it cannot determine is None."""
    record = {'simulator': simulator, 'simulator_binary': None,
              'simulator_binary_source': None, 'simulator_version': None}
    try:
        if simulator == 'behavioral':
            # silk's own numpy engine (TASK-448.3): no binary is launched.
            record['simulator_binary_source'] = 'in-process'
            try:
                from importlib.metadata import version
                record['simulator_version'] = f"silk {version('circuitsilk')}"
            except Exception:
                pass
            return record
        if simulator == 'ngspice':
            from silk.simulation.runner import ngspice_executable
            requested, env = ngspice_executable(), 'SILK_NGSPICE_BIN'
        elif simulator == 'xyce':
            from silk.simulation.xyce_runner import _xyce_executable
            requested, env = _xyce_executable(), 'SILK_XYCE_BIN'
        else:
            return record
        record['simulator_binary_source'] = (env if _settings.is_configured(env)
                                             else 'PATH')
        exe = shutil.which(requested) or requested
        if exe and os.path.isfile(exe):
            exe = os.path.abspath(exe)
            record['simulator_binary'] = share_path(exe)
            record['simulator_version'] = _binary_version(exe, simulator)
    except Exception:
        pass
    return record


# ---------------------------------------------------------------------------
# PDK library files named by a prepared deck
# ---------------------------------------------------------------------------

_LIB_LINE_RE = re.compile(
    r'''^\s*\.lib\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+(\S+)\s*$''', re.IGNORECASE)


@functools.lru_cache(maxsize=64)
def _library_hashes(path: str, mtime: float, size: int) -> Tuple[str, dict]:
    """(file sha256, {section name lower: section text sha256}) for one file."""
    with open(path, 'rb') as f:
        data = f.read()
    sections = {}
    current, start = None, 0
    lines = data.decode('utf-8', errors='replace').splitlines(keepends=True)
    for i, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        head = tokens[0].lower()
        if head == '.lib' and len(tokens) == 2 and current is None:
            current, start = tokens[1].lower(), i + 1
        elif head == '.endl' and current is not None:
            text = ''.join(lines[start:i])
            sections.setdefault(current, hashlib.sha256(text.encode('utf-8')).hexdigest())
            current = None
    return hashlib.sha256(data).hexdigest(), sections


def deck_libraries(deck_text: str) -> list:
    """One entry per ``.lib "<file>" <section>`` line of ``deck_text``.

    ``file_sha256`` hashes that one file's bytes and ``section_text_sha256``
    only the text between ``.lib <section>`` and ``.endl``; neither covers
    the files the section includes. Hashes are None when the file cannot be
    read. Never raises."""
    entries = []
    for line in deck_text.splitlines():
        m = _LIB_LINE_RE.match(line)
        if not m:
            continue
        path = m.group(1) or m.group(2) or m.group(3)
        section = m.group(4)
        file_sha = section_sha = None
        try:
            st = os.stat(path)
            file_sha, sections = _library_hashes(
                os.path.abspath(path), st.st_mtime, st.st_size)
            section_sha = sections.get(section.lower())
        except Exception:
            pass
        entries.append({'file': share_path(path, pdk_relative=True),
                        'section': section, 'file_sha256': file_sha,
                        'section_text_sha256': section_sha})
    return entries


def prepared_deck_record(module, deck_text: str) -> dict:
    """Facts about a deck silk just prepared, stored on the module by
    ``generate_netlist()`` and read back by :func:`build`. Never raises."""
    try:
        options = dict(getattr(module, 'simulator_options', None) or {})
        save = getattr(module, '_save_list', None)
        return {
            'deck_sha256': hashlib.sha256(deck_text.encode()).hexdigest(),
            'pdk_libraries': deck_libraries(deck_text),
            'simulator_options': scrub(_plain(options)),
            'save_list': None if save is None else scrub(_plain(save)),
            'seed': _plain(options.get('seed')),
        }
    except Exception:
        return {}


def build(*, run_origin: str, launch: Optional[dict], prepared: Optional[dict],
          pdk_name: Optional[str], conditions: Any,
          simulator_hint: Optional[str] = None) -> RunProvenance:
    """Assemble one :class:`RunProvenance`. Never raises."""
    env = _environment()
    prepared = prepared or {}
    if run_origin == RUN_ORIGIN_SILK:
        sim = dict(launch or {})
        sim.setdefault('simulator', simulator_hint)
        prepared = sim.get('prepared', prepared)
    else:
        sim = {'simulator': simulator_hint}
    temperature_c = temperature_kind = supply = corner = None
    corner_kind = supply_kind = None
    if conditions is not None:
        corner = getattr(conditions, 'corner', None)
        temperature_kind = getattr(conditions, 'temperature_kind', None)
        temperature_c = getattr(conditions, 'temperature_c', None)
        supply = getattr(conditions, 'supply_voltage_v', None)
        corner_kind = getattr(conditions, 'corner_kind', None)
        supply_kind = getattr(conditions, 'supply_kind', None)
    return RunProvenance(
        run_origin=run_origin,
        silk_version=env['silk_version'],
        silk_git=env['silk_git'],
        python_version=env['python_version'],
        numpy_version=env['numpy_version'],
        simulator=sim.get('simulator'),
        simulator_binary=sim.get('simulator_binary'),
        simulator_binary_source=sim.get('simulator_binary_source'),
        simulator_version=sim.get('simulator_version'),
        pdk_name=pdk_name,
        pdk_libraries=prepared.get('pdk_libraries'),
        corner=None if corner is None else str(corner),
        corner_kind=corner_kind,
        temperature_c=_plain(temperature_c),
        temperature_kind=temperature_kind,
        supply_voltage_v=_plain(supply),
        supply_kind=supply_kind,
        simulator_options=prepared.get('simulator_options'),
        save_list=prepared.get('save_list'),
        deck_sha256=prepared.get('deck_sha256'),
        seed=prepared.get('seed'),
        seedinfo=None,
        normalized_deck_sha256=sim.get('normalized_deck_sha256'),
        launch_corner_label=sim.get('launch_corner_label'),
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec='seconds').replace('+00:00', 'Z'),
    )


def from_module(module, conditions: Any = None,
                pdk_details: Any = None) -> RunProvenance:
    """Build the record for the results currently loaded on ``module`` (a
    simulated Module / testbench netlist). ``'silk-launch'`` only when its
    conditions say silk ran the simulator; otherwise ``'external-rawfile'``.
    Never raises."""
    if conditions is None:
        conditions = getattr(module, 'operating_conditions', None)
    pdk = pdk_details if pdk_details is not None else getattr(module, 'pdk_details', None)
    silk_launched = conditions is not None and getattr(conditions, 'origin', None) == 'silk'
    return build(
        run_origin=RUN_ORIGIN_SILK if silk_launched else RUN_ORIGIN_EXTERNAL,
        launch=getattr(module, '_launch_record', None) if silk_launched else None,
        prepared=getattr(module, '_prepared_deck_record', None),
        pdk_name=(getattr(pdk, 'pdk_name', None) or getattr(pdk, 'name', None)),
        conditions=conditions,
        simulator_hint=(getattr(module, 'simulator', None) if silk_launched
                        else getattr(conditions, 'backend', None)),
    )


__all__ = [
    'SCHEME', 'FIELDS', 'VOLATILE_FIELDS', 'RUN_ORIGIN_SILK',
    'RUN_ORIGIN_EXTERNAL', 'RunProvenance', 'build', 'deck_libraries',
    'from_module',
    'launch_record', 'prepared_deck_record', 'share_path', 'scrub',
]
