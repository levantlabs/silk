"""silk.pdk.validate — vet that every PDK device can be simulated.

Static pass: cross-check each device's pinList arity + emission prefix against
its actual .subckt/.model in the PDK lib. Dynamic pass (opt-in): build a minimal
netlist per device and run ngspice.
"""
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass

import yaml

from silk.circuit.modules import Module
from silk.simulation import analysis
from silk import settings as _settings
from silk import sources
from silk.pdk.basepdk import parse_junction_rating

_SUBCKT_RE = re.compile(r'^\s*\.subckt\s+(\S+)\s+(.*)', re.IGNORECASE)
_MODEL_RE = re.compile(r'^\s*\.model\s+(\S+)\s+(\S+)', re.IGNORECASE)
_INCLUDE_RE = re.compile(r'^\s*\.(include|inc|lib)\s+(.*)', re.IGNORECASE)

# The fixed two-point transient the --simulate probe runs per device, in
# seconds (AUDIT-02: named; not user-configurable by design).
_PROBE_TRAN_STEP_S = 1e-9
_PROBE_TRAN_STOP_S = 2e-9
_SPICE_EXTS = ('.spice', '.cir', '.lib', '.mod')


def _subckt_terminals(rest: str) -> list:
    """Terminal names on a `.subckt <name> <rest>` line.

    Two things separate terminals from parameters. Bare ``k=v`` tokens are
    parameters wherever they appear. The optional ``params:`` keyword ends the
    terminal list outright — everything after it is parameters, including the
    spaced-out ``params: w = 1`` form whose tokens carry no ``=`` of their own,
    and the keyword itself is not a terminal.
    """
    terms = []
    for tok in rest.split():
        if tok.lower().startswith('params:'):
            break
        if '=' in tok:
            continue
        terms.append(tok)
    return terms


def scan_lib_files(paths) -> dict:
    """{name: {('subckt', n_terminals) | ('model', model_type), ...}} over `paths`.

    A subckt/model name can be defined multiple times *within one PDK* —
    once per corner/variant/parasitic file — often with different arities.
    Every definition seen is recorded (as a set), rather than picking a
    single last-wins/subckt-wins value, so callers can check a candidate
    arity against the full set of legitimate definitions instead of
    flagging false positives against whichever definition happened to be
    scanned last.

    That permissiveness is only sound when `paths` all belong to the SAME
    PDK — see :func:`resolve_lib_closure`, which is how callers should
    normally build the file list. Handing this a set of files spanning
    unrelated PDK trees lets a definition from a tree that is never
    netlisted launder a genuine mismatch in the tree that is.
    """
    defs = {}
    for path in paths:
        try:
            with open(path, errors='ignore') as fh:
                for line in fh:
                    m = _SUBCKT_RE.match(line)
                    if m:
                        terms = _subckt_terminals(m.group(2))
                        defs.setdefault(m.group(1), set()).add(('subckt', len(terms)))
                        continue
                    m = _MODEL_RE.match(line)
                    if m:
                        defs.setdefault(m.group(1), set()).add(('model', m.group(2).lower()))
        except OSError:
            continue
    return defs


def _include_target(line, base_dir):
    """Absolute path named by a `.include`/`.inc`/`.lib` line, or None.

    Handles the quoted and bare forms, and resolves relative paths against
    `base_dir` (the directory of the including file), which is how SPICE
    itself resolves them.

    A one-token `.lib X` is a section, not a file: it is either the header
    that OPENS a section (closed by `.endl`) or a bare reference to one, and
    in both cases X is a section name. Such a line is only followed when X
    looks like a path, so a file that happens to sit beside the lib under a
    section's name (a plain `tt` next to a lib full of `.lib tt`) is not
    pulled into the closure and cannot contribute definitions the simulator
    would never see. The two-token `.lib <file> <section>` form names a real
    file and is always followed.
    """
    m = _INCLUDE_RE.match(line)
    if not m:
        return None
    directive = m.group(1).lower()
    rest = m.group(2).strip()
    if rest[:1] in ('"', "'"):
        end = rest.find(rest[0], 1)
        tok = rest[1:end] if end > 0 else ''
        trailing = rest[end + 1:].split() if end > 0 else []
    else:
        words = rest.split()
        tok = words[0] if words else ''
        trailing = words[1:]
    if not tok or tok.startswith('*'):
        return None
    if directive == 'lib' and not trailing and not ('/' in tok or '.' in tok):
        return None
    tok = os.path.expanduser(os.path.expandvars(tok))
    if not os.path.isabs(tok):
        tok = os.path.join(base_dir, tok)
    return os.path.abspath(tok)


def resolve_lib_closure(lib_file: str, extra_includes=()) -> list:
    """Every file reachable from `lib_file` by following `.include`/`.lib`.

    This is the set of definitions the simulator can actually see for a
    netlist built against `lib_file`, so it is the right scope for the
    arity/emission cross-check. Sections are deliberately NOT honoured:
    the union over every corner section is taken, because a device's
    terminal list does not change with the corner, and the union is what
    preserves the legitimate "same cell, several arities across
    corner/parasitic files" tolerance documented on :func:`scan_lib_files`.

    `extra_includes` seeds the walk with any out-of-chain file the netlister
    also emits: the manifest's ``library.extra_includes``, AND every file named
    by a corner's ``includes:`` list (a PDK whose device families live in
    separate corner files reaches most of its devices only that way). So
    cells that live outside the lib's own include chain (e.g. sky130's
    ``cap_vpp_11p5x11p7``) are in scope rather than ``missing_definition``.

    Missing/unreadable includes are skipped rather than raised on: a PDK
    lib routinely references optional files, and a hard failure here would
    turn the validator into a liability. Cycles are handled by visiting
    each path at most once.
    """
    files = []
    seen = set()
    stack = [os.path.abspath(p) for p in (lib_file, *extra_includes)]
    while stack:
        path = stack.pop()
        if path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        files.append(path)
        base_dir = os.path.dirname(path)
        try:
            with open(path, errors='ignore') as fh:
                for line in fh:
                    target = _include_target(line, base_dir)
                    if target is not None:
                        stack.append(target)
        except OSError:
            continue
    return files


def scan_lib_definitions(root: str) -> dict:
    """:func:`scan_lib_files` over every SPICE file in the tree at `root`.

    The unscoped fallback for when no lib file is resolvable (see
    :func:`validate_pdk`). Prefer :func:`resolve_lib_closure`: a `$PDK_ROOT`
    commonly holds several unrelated PDK trees (an installed lib, a vendor
    source checkout, other technologies), and pooling their definitions
    lets one tree's arity vouch for a device that is netlisted against
    another.
    """
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        paths += [os.path.join(dirpath, fn) for fn in files if fn.endswith(_SPICE_EXTS)]
    return scan_lib_files(paths)


@dataclass
class DeviceIssue:
    device: str
    kind: str        # 'arity_mismatch' | 'wrong_emission' | 'missing_definition' | 'sim_failed'
                     # | 'unsourced_rating' | 'bad_provenance' | 'rating_convention'
    detail: str


class PDKValidationError(RuntimeError):
    pass


_MODEL_PRIMITIVE = {'d': 'D', 'r': 'R', 'c': 'C', 'l': 'L'}


def _lib_root(pdk, lib_root):
    """Resolve the PDK lib tree to walk: explicit arg > $PDK_ROOT > pdk.library_location.

    Only the *fallback* scope, used when no lib file is resolvable (see
    :func:`_collect_definitions`); a `$PDK_ROOT` is a directory of PDKs, not
    one PDK, so what it finds is not necessarily what gets netlisted.

    ``library_location`` is only trusted when it's a regular file (the lib
    spice file) — loaded PDKs commonly set it to ``/dev/null`` as a
    placeholder, and ``/dev/null`` is not a regular file (``os.path.isfile``
    is False for device nodes), so the placeholder is correctly skipped.
    ``PDK_ROOT`` is still preferred over it here because one lib file's
    directory is far too narrow to find all cell subckts by walking — the
    scoped answer to that is the include closure, not a wider walk.
    """
    if lib_root:
        return lib_root
    root = _settings.get('PDK_ROOT') if _settings.is_configured('PDK_ROOT') else None
    if root and os.path.isdir(root):
        return root
    loc = getattr(pdk, 'library_location', None)
    if loc and os.path.isfile(loc):
        return os.path.dirname(os.path.abspath(loc))
    raise PDKValidationError(
        'no PDK lib root: pass lib_root= or set PDK_ROOT (or a real library_location)')


def _is_placeholder_lib(path):
    """True for the documented "this PDK has no lib file" placeholder.

    ``os.devnull`` (``/dev/null``) is what a PDK under construction, and every
    PDK-free netlisting example, sets ``library_location`` to. It is a
    deliberate declaration that there is no lib to scope against — unlike a
    path that was *meant* to name a lib file and doesn't.
    """
    return os.path.realpath(path) == os.path.realpath(os.devnull)


def _reject_unusable_lib(lib_file):
    """Refuse to validate against a lib path that names no readable lib file.

    Always raises :class:`PDKValidationError`. Validation must not silently
    degrade such a path into the unscoped ``$PDK_ROOT`` walk: that walk pools
    every SPICE file under a directory that routinely holds several unrelated
    PDK trees, so one tree's definitions can vouch for a device netlisted
    against another (see :func:`scan_lib_definitions`) and a typo'd ``--lib``
    reports a clean bill instead of a wrong path.
    """
    if os.path.isdir(lib_file):
        raise PDKValidationError(
            f'library location {lib_file!r} is a directory; point --lib / '
            'library_location at the PDK lib file itself '
            '(e.g. .../sky130A/libs.tech/ngspice/sky130.lib.spice)')
    raise PDKValidationError(
        f'PDK lib file {lib_file!r} does not exist or is not readable; '
        'validation will not fall back to an unscoped $PDK_ROOT scan, which '
        'can report a clean bill using an unrelated PDK tree. Fix the path, '
        f'or set library_location={os.devnull!r} if this PDK has no lib yet')


def _collect_definitions(pdk, lib_root):
    """``(definitions, scope)`` to cross-check `pdk`'s devices against.

    `scope` is a one-line description of what was actually scanned, returned
    alongside the definitions so the CLI can disclose it (see :func:`main`).
    A clean verdict means nothing without it: "no issues" from the include
    closure and "no issues" from the unscoped ``$PDK_ROOT`` walk are very
    different claims, and printing them identically is what let the
    laundering hole sit unnoticed.

    Scoped to the PDK actually in use wherever possible: ``library_location``
    is the lib spice file the netlister writes a ``.lib`` line for (the
    manifest's ``library.lib_path``, resolved by ``load_pdk``), so the
    closure reachable from it — plus the manifest's ``extra_includes`` — is
    exactly what the simulator can bind. Checking against that, instead of
    every SPICE file under ``$PDK_ROOT``, is what stops an unrelated tree
    (a vendor source checkout alongside the installed PDK) from vouching
    for an arity the in-chain lib does not define.

    A ``library_location`` that names no readable lib file is an error here
    (:func:`_reject_unusable_lib`), not a reason to widen the scope — the
    fallback would launder exactly the mismatch the closure was introduced to
    catch. Only the ``/dev/null`` placeholder, which declares that there is no
    lib to start a closure from, still falls back to the unscoped
    ``$PDK_ROOT`` walk. This strictness is confined to validation: building a
    PDK against a fake lib stays legal, since netlisting never reads it.

    An explicit `lib_root` keeps meaning "walk exactly this tree" and wins
    outright.
    """
    if lib_root:
        return scan_lib_definitions(lib_root), f'explicit tree walk of {lib_root}'
    lib_file = getattr(pdk, 'library_location', None)
    if lib_file and not _is_placeholder_lib(lib_file):
        if not (os.path.isfile(lib_file) and os.access(lib_file, os.R_OK)):
            _reject_unusable_lib(lib_file)
        # Seed the walk with every file any CORNER names, not just lib_path.
        # A PDK may split device families across separate corner files — IHP
        # SG13G2 keeps its high-voltage FETs in cornerMOShv.lib while lib_path
        # points at cornerMOSlv.lib — and the netlister emits one `.lib` line
        # per entry, so those definitions ARE visible to the simulator. Without
        # this the validator reported `sg13_hv_nmos` as missing from a deck
        # that resolves it perfectly well: a false failure on the one tool
        # whose job is telling you the PDK is wired up correctly.
        seeds = list(getattr(pdk, 'extra_includes', ()) or ())
        for meta in (getattr(pdk, 'corners', {}) or {}).values():
            for inc in (meta or {}).get('includes') or ():
                if inc.get('file') and inc['file'] not in seeds:
                    seeds.append(inc['file'])
        files = resolve_lib_closure(lib_file, seeds)
        return scan_lib_files(files), f'closure of {lib_file}, {len(files)} files'
    root = _lib_root(pdk, None)
    return (scan_lib_definitions(root),
            f'unscoped walk of $PDK_ROOT ({root}) — no lib file')


# ---------------------------------------------------------------------------
# Junction-ratings provenance (TASK-48)
# ---------------------------------------------------------------------------

# The two heuristic-rule forms a `junction_ratings_provenance.heuristic` entry
# may open with. Each is re-checked against the numbers, so a rule that no
# longer matches its value is reported instead of silently trusted:
#   "= Vds_max"                -> equal to that other rating of the same device
#   "1.1 x nominal 6.0 V"      -> factor times the stated nominal voltage
_HEUR_EQUAL_RE = re.compile(r'^=\s*([A-Za-z]+_max)\b')
_HEUR_FACTOR_RE = re.compile(
    r'^([0-9]*\.?[0-9]+)\s*x\s*nominal\s*([0-9]*\.?[0-9]+)\s*V?\b', re.IGNORECASE)
_RATING_REL_TOL = 1e-6


def _fmt_range(lo, hi):
    """``1.95`` for the symmetric (magnitude) form, ``[-1.95, +0.3]`` otherwise."""
    if lo == -hi:
        return f'{hi:g}'
    return f'[{lo:+g}, {hi:+g}]'


def _range_convention_issue(key, lo, hi):
    """``None`` when ``(lo, hi)`` is a usable operating range, else why not.

    A magnitude ``m`` arrives here as ``(-m, m)``, so the legacy rule
    ("a positive finite magnitude") is the ``lo < hi`` check on its
    normalised form: ``m = 0`` is the empty range ``(0, 0)`` and ``m < 0``
    is inverted. A signed range must be finite at both ends with
    ``min < max``; a ``{min: 0.3, max: -1.95}`` (the sky130 NMOS Vbs bounds
    written in the order the SkyWater page lists them, "+0.3 to -1.95 V")
    is exactly the mistake this catches.
    """
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return f'{key}=({lo!r}, {hi!r}) is not a pair of voltages'
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f'{key}={_fmt_range(lo, hi)} is not a finite voltage range'
    if not lo < hi:
        if lo == -hi:
            return f'{key}={hi:g} is not a positive finite voltage magnitude'
        return f'{key}={_fmt_range(lo, hi)} has min >= max (a signed range is {{min: lo, max: hi}} with lo < hi)'
    return None


def _ranges_of(dev, ratings):
    """``{key: (lo, hi)}`` for a device — its ``junction_ranges`` when the
    loader built them, else the magnitudes expanded through the ONE parser
    (``parse_junction_rating``) so a hand-built device carrying only
    ``junction_ratings`` is judged by the same rule as a catalog one."""
    ranges = getattr(dev, 'junction_ranges', None)
    if isinstance(ranges, dict) and ranges:
        return dict(ranges)
    out = {}
    for key, val in ratings.items():
        try:
            out[key] = parse_junction_rating(key, val)
        except (TypeError, ValueError):
            out[key] = (val, val)      # reported by _range_convention_issue
    return out


def _check_heuristic_rule(key, text, ranges):
    """``None`` when `text` is a recognised rule that agrees with
    ``ranges[key]``; otherwise the one-line reason it is rejected.

    ``= <other>_max`` means the two pairs share the SAME signed range (both
    ends compared, so a magnitude cannot be claimed equal to an asymmetric
    range); ``<factor> x nominal <V> V`` derives a magnitude, so it can only
    vouch for the symmetric form.
    """
    if not isinstance(text, str) or not text.strip():
        return f'heuristic rule for {key} is empty'
    text = text.strip()
    lo, hi = ranges[key]
    m = _HEUR_EQUAL_RE.match(text)
    if m:
        other = m.group(1)
        if other == key:
            return f'heuristic rule for {key} refers to itself'
        if other not in ranges:
            return f'heuristic rule for {key} refers to {other}, which this device does not rate'
        olo, ohi = ranges[other]
        if not (math.isclose(lo, olo, rel_tol=_RATING_REL_TOL, abs_tol=_RATING_REL_TOL)
                and math.isclose(hi, ohi, rel_tol=_RATING_REL_TOL, abs_tol=_RATING_REL_TOL)):
            return (f'heuristic rule for {key} says "= {other}" but {key}={_fmt_range(lo, hi)} '
                    f'and {other}={_fmt_range(olo, ohi)}')
        return None
    m = _HEUR_FACTOR_RE.match(text)
    if m:
        expect = float(m.group(1)) * float(m.group(2))
        if lo != -hi:
            return (f'heuristic rule for {key} says "{m.group(0)}", a magnitude, '
                    f'but {key}={_fmt_range(lo, hi)} is a signed range')
        if not math.isclose(hi, expect, rel_tol=_RATING_REL_TOL):
            return (f'heuristic rule for {key} says "{m.group(0)}" = {expect:g} '
                    f'but {key}={hi:g}')
        return None
    return (f'heuristic rule for {key} does not open with a recognised form '
            f'("= <key>_max" or "<factor> x nominal <V> V"): {text[:60]!r}')


def check_junction_provenance(pdk) -> list:
    """Catalog-only pass: every ``junction_ratings`` value must be sourced.

    Reads nothing but the loaded catalog (no lib, no ``$PDK_ROOT``), so it is
    part of every static run. For each device that carries ratings it flags:

    - ``unsourced_rating`` — no ``junction_ratings_provenance`` block, or a
      rating key covered neither by ``source`` nor by a ``heuristic`` entry.
    - ``bad_provenance`` — the block is malformed: ``source`` is not a
      non-empty string, a ``heuristic`` key names a pair the device does not
      rate, or its rule does not open with one of the two recognised forms /
      disagrees with the numbers it claims to derive (``= Vds_max`` when the
      two values differ, ``1.1 x nominal 6.0 V`` when the value is not 6.6).
    - ``rating_convention`` — a rating that is not a usable operating range:
      a magnitude that is not a positive finite number, or a signed range
      ``{min, max}`` that is not finite at both ends with
      ``min < max``. Both forms are judged on their normalised ``(lo, hi)``
      (``parse_junction_rating``), which is what ``silk.lint.rules.stress``
      scores the signed pair voltage against.

    A device with an empty ``junction_ratings`` and no provenance is fine
    (passives are unrated by design); a provenance block on an unrated device
    is ``bad_provenance`` because it can only be stale. The derivation rule
    behind all of this is written once, in docs/adding-a-pdk.md.
    """
    issues = []
    for name, dev in pdk._devices.items():
        ratings = dict(getattr(dev, 'junction_ratings', None) or {})
        prov = getattr(dev, 'junction_ratings_provenance', None)
        if not ratings:
            if prov is not None:
                issues.append(DeviceIssue(name, 'bad_provenance',
                    'junction_ratings_provenance present but the device declares no junction_ratings'))
            continue
        ranges = _ranges_of(dev, ratings)
        for key, (lo, hi) in ranges.items():
            why = _range_convention_issue(key, lo, hi)
            if why:
                issues.append(DeviceIssue(name, 'rating_convention', why))
        if prov is None:
            issues.append(DeviceIssue(name, 'unsourced_rating',
                f'junction_ratings {sorted(ratings)} carry no junction_ratings_provenance '
                '(cite the PDK document / model parameter, or tag each key HEURISTIC with its rule)'))
            continue
        source = getattr(prov, 'source', None)
        heur = dict(getattr(prov, 'heuristic', None) or {})
        if source is not None and (not isinstance(source, str) or not source.strip()):
            issues.append(DeviceIssue(name, 'bad_provenance',
                f'junction_ratings_provenance.source must be a non-empty citation, got {source!r}'))
            source = None
        for key, text in heur.items():
            if key not in ratings:
                issues.append(DeviceIssue(name, 'bad_provenance',
                    f'heuristic entry {key!r} names a pair this device does not rate {sorted(ratings)}'))
                continue
            why = _check_heuristic_rule(key, text, ranges)
            if why:
                issues.append(DeviceIssue(name, 'bad_provenance', why))
        if source is None:
            uncovered = sorted(k for k in ratings if k not in heur)
            if uncovered:
                issues.append(DeviceIssue(name, 'unsourced_rating',
                    f'{uncovered} have no source and no heuristic rule'))
    return issues


_LOG_SUBSTR_MARKERS = ('unknown subckt', 'singular matrix')
_LOG_KEYWORD_MARKERS = ('Error', 'error', 'fatal')


def _scan_sim_log(text):
    """Scan ngspice ``output.log`` text for the "silent mis-parse" failure
    mode (spec 2026-07-10-pdk-device-vetting-design.md §3.2 / Bug A1): a
    mis-bound subckt can make ngspice exit 0 while still logging
    ``unknown subckt`` / ``singular matrix``, or another error/fatal line —
    so a non-zero exit code alone isn't a reliable enough signal.

    Returns the first offending line (stripped, truncated to ~200 chars),
    or ``None`` if the log looks clean.
    """
    if not text:
        return None
    for line in text.splitlines():
        low = line.lower()
        if any(m in low for m in _LOG_SUBSTR_MARKERS):
            return line.strip()[:200]
        if any(kw in line for kw in _LOG_KEYWORD_MARKERS):
            return line.strip()[:200]
    return None


def _device_parameters(dev):
    """Build a minimal valid parameter dict for `dev` from its param_list
    constraints, using each parameter's ``minval`` — a legitimate value for
    every shipped device (no device has an unbounded/-1 minval; ``maxval ==
    -1`` just means "unbounded above"). Devices with no param_list (or a
    fake test double lacking the attribute) get ``{}``.
    """
    params = {}
    for pname, constraint in getattr(dev, 'param_list', {}).items():
        minval = constraint.minval
        try:
            params[pname] = float(minval)
        except (TypeError, ValueError):
            params[pname] = minval  # symbolic minval — pass through as-is
    return params


def _pdk_label(pdk):
    """Label for the throwaway vetting ``Module``: the loaded PDK's name
    (``load_pdk('gf180mcu')`` sets ``pdk_name``), or the PDK class name for a
    bare constructor. Cosmetic — the label names the deck's origin in the
    output directory — but a fixed ``'PDK_skywater130'`` for every ``--pdk``
    was a lie about which PDK was vetted."""
    return getattr(pdk, 'pdk_name', None) or type(pdk).__name__


def _simulate_devices(pdk):
    """Dynamic pass (opt-in): per-device ngspice smoke-sim.

    For each device the PDK exposes, builds a throwaway top-level `Module`
    giving every pin its own node. There is no ideal-resistor primitive in
    ``silk.sources`` to tie nodes to ground through a series resistor (the
    original design sketch), so instead each node is driven independently
    to ground by its own distinct small DC ``VoltageSource`` — this fully
    constrains every node (no floating nodes, no singular matrix) without
    needing a resistor primitive. A tiny `.tran` stands in for an
    operating-point smoke test (no ``OperatingPoint`` analysis class
    exists).

    Flags a ``sim_failed`` `DeviceIssue` for any device that:
    - raises while being built (bad/missing params, wiring errors) or while
      being simulated (``SimulationError`` on non-zero ngspice exit — its
      message already includes the offending ngspice log lines); or
    - simulates with exit 0 but whose ``output.log`` shows the A1-style
      silent mis-parse (see :func:`_scan_sim_log`); or
    - completes but its recorded run log cannot be read.

    A clean device contributes nothing to the returned list.
    """
    issues = []
    for name, dev in pdk._devices.items():
        device_cls = getattr(pdk, name, None)
        if device_cls is None:
            continue
        tmpdir = tempfile.mkdtemp(prefix=f'pdkvet_{name}_')
        try:
            detail = None
            try:
                tb = Module(_pdk_label(pdk), tmpdir, f'vet_{name}.cir',
                            pdk_details=pdk, module_name=None)
                nets = {}
                for i, pin in enumerate(dev.pin_list):
                    netname = f'n_{pin}'
                    tb.add_net(netname)
                    tb.add_source(sources.VoltageSource(
                        f'v_{pin}', positive=netname, negative='0',
                        dc=0.1 * (i + 1)))
                    nets[pin] = netname
                tb.add_element(device_cls, 'DUT', nets=nets,
                                parameters=_device_parameters(dev))
                tb.add_sim_type(analysis.Transient(step=_PROBE_TRAN_STEP_S, stop=_PROBE_TRAN_STOP_S))
                try:
                    tb.simulate(corner=getattr(pdk, 'default_corner', 'tt'))
                except Exception as exc:
                    detail = str(exc)[:200]
                if detail is None:
                    files = tb._run_files
                    logpath = None if files is None else files.logfile_path
                    if logpath is None:
                        detail = 'Cannot read simulator log: the completed run has no recorded log path'
                    else:
                        try:
                            with open(logpath, errors='replace') as fh:
                                detail = _scan_sim_log(fh.read())
                        except OSError as exc:
                            detail = f'Cannot read simulator log {logpath!r}: {exc}'
            except Exception as exc:
                # Build-time failure (missing/invalid params, wiring, ...)
                # is also a "cannot be simulated" signal.
                detail = str(exc)[:200]
            if detail is not None:
                issues.append(DeviceIssue(name, 'sim_failed', detail))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return issues


def validate_pdk(pdk, *, lib_root=None, simulate=False):
    """Static cross-check of every device in `pdk._devices` against the PDK lib.

    "The PDK lib" here means the definitions reachable from the PDK's own
    lib file — see :func:`_collect_definitions` for how that scope is
    resolved and why it is not simply everything under ``$PDK_ROOT``.

    For each device, flags:
    - ``missing_definition``: `.pdk_device` isn't a `.subckt`/`.model` anywhere in the lib.
    - ``arity_mismatch``: a `.subckt` device whose `pinList` length != the `.subckt` terminal count.
    - ``wrong_emission``: a `.subckt` device emitted with a non-X prefix, or a `.model` device
      emitted with an X prefix (i.e. instantiated as a subcircuit call instead of a primitive line).
    - ``unsourced_rating`` / ``bad_provenance`` / ``rating_convention``: a
      ``junction_ratings`` value with no cited source or heuristic rule, a
      malformed or self-contradicting provenance block, or a rating that is
      not a positive finite magnitude — see :func:`check_junction_provenance`.

    With ``simulate=True``, also runs the dynamic smoke-sim pass (`_simulate_devices`).

    Raises `PDKValidationError` when no trustworthy scope can be resolved:
    the PDK's ``library_location`` names a file that isn't there (a typo'd
    ``--lib``), or there is no lib and no ``$PDK_ROOT`` to fall back to. A
    validator that guesses a wider scope is worse than one that stops.
    """
    return _validate_with_scope(pdk, lib_root=lib_root, simulate=simulate)[0]


def _validate_with_scope(pdk, *, lib_root=None, simulate=False):
    """:func:`validate_pdk` plus the scope it checked: ``(issues, scope)``.

    The CLI prints the scope beside its verdict. It comes back from the same
    call that produced the issues rather than from a second resolution pass,
    so the disclosed scope cannot drift from the scope actually used — a
    disclosure that can lie is worse than none.
    """
    defs, scope = _collect_definitions(pdk, lib_root)
    issues = []
    for name, dev in pdk._devices.items():
        pd = dev.pdk_device
        npins = len(dev.pin_list)
        # The prefix the netlister will actually write: the catalog
        # declaration carried on the PDKDevice (default XM). Read from the
        # device, not re-derived, so the verdict cannot drift from emission.
        prefix = dev.spice_prefix
        d = defs.get(pd)
        if not d:
            issues.append(DeviceIssue(name, 'missing_definition',
                f'{pd!r} is not a .subckt/.model in the lib'))
            continue
        subckt_arities = {n for (k, n) in d if k == 'subckt'}
        has_model = any(k == 'model' for (k, _n) in d)
        if subckt_arities:
            if npins not in subckt_arities:
                issues.append(DeviceIssue(name, 'arity_mismatch',
                    f'pinList has {npins} pins but .subckt {pd} has terminal counts '
                    f'{sorted(subckt_arities)} in the lib'))
            if not str(prefix).startswith('X'):
                issues.append(DeviceIssue(name, 'wrong_emission',
                    f'.subckt {pd} but spice_prefix={prefix!r} (expected X/XM)'))
        elif has_model:
            if str(prefix).startswith('X'):
                model_types = {n for (k, n) in d if k == 'model'}
                info = next(iter(model_types))
                want = _MODEL_PRIMITIVE.get(info, info.upper())
                issues.append(DeviceIssue(name, 'wrong_emission',
                    f'.model {pd} ({info}) but emits an X line; expected a {want} primitive'))
    issues += check_junction_provenance(pdk)   # TASK-48: catalog-only, no lib needed
    if simulate:
        issues += _simulate_devices(pdk)   # Task 4
    return issues, scope


# ---------------------------------------------------------------------------
# CLI: `python -m silk.pdk.validate [--simulate]`
# ---------------------------------------------------------------------------

def _default_yaml():
    """Package-relative path to the shipped sky130 device-details yaml (cwd-independent)."""
    import silk
    return os.path.join(os.path.dirname(silk.__file__), 'pdk', 'data',
                        'skywater_130_device_details.yaml')


def _default_lib():
    """Best-effort sky130 lib path from $PDK_ROOT, for the ad-hoc ``--yaml``
    path (a generic BasePDK has no manifest to resolve one from). It is the
    root of the static pass's include closure as well as the lib the
    ``--simulate`` netlists bind against."""
    root = _settings.get('PDK_ROOT')
    return os.path.join(root, 'sky130A', 'libs.tech', 'ngspice', 'sky130.lib.spice')


def _catalog_identifier_errors(path):
    """Collect every bad Python name without constructing a fail-fast PDK."""
    from silk.elements._catalog import _device_name_errors
    from silk.pdk.basepdk import _StrictSafeLoader

    with open(path) as stream:
        catalog = yaml.load(stream, Loader=_StrictSafeLoader)
    return [error for name, spec in catalog.items()
            for error in _device_name_errors(name, spec, path)]


def main(argv=None):
    """Validate any registered PDK (or an ad-hoc device-details yaml).

    Two modes:
    - ``--pdk NAME`` (default ``sky130``): load the PDK through the registry
      (``silk.pdk.load_pdk``), so the device catalog, lib, corners, and
      ``extra_includes`` all come from its manifest. Works for any registered
      PDK — see ``silk.pdk.available()``.
    - ``--yaml PATH``: validate a device-details yaml directly with a generic
      ``BasePDK``, bypassing the registry. Use this when authoring a PDK before
      its manifest exists.

    Exit code 0 = no issues, 1 = validation issues, 2 = PDK/lib not resolvable —
    so this doubles as a CI/pre-commit gate for a new or edited PDK.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog='python -m silk.pdk.validate',
        description='Vet a PDK: every device\'s pinList arity and emission must match '
                    'its real .subckt/.model, and (with --simulate) every device must '
                    'bind and simulate in ngspice.')
    parser.add_argument('--pdk', default='sky130',
                        help='registered PDK name to validate (default: sky130; see '
                             'silk.pdk.available()). Resolves catalog/lib/corners/'
                             'extra_includes from its manifest.')
    parser.add_argument('--yaml', default=None,
                        help='validate an ad-hoc device-details yaml directly, bypassing '
                             'the registry (for authoring a PDK before its manifest exists); '
                             'constructs a generic BasePDK.')
    parser.add_argument('--lib', default=None,
                        help='override the library_location (…/sky130.lib.spice); default '
                             'resolves from the manifest via $PDK_ROOT.')
    parser.add_argument('--simulate', action='store_true',
                        help='also run the per-device ngspice smoke-sim (needs ngspice + PDK)')
    args = parser.parse_args(argv)

    from silk.exceptions import PDKError
    from silk.pdk.registry import PDKRegistryError, manifest
    try:
        if args.yaml:
            catalog_path = args.yaml
        else:
            info = manifest(args.pdk)
            catalog_path = os.path.join(info['_manifest_dir'], info['device_details'])
        name_errors = _catalog_identifier_errors(catalog_path)
        if name_errors:
            for error in name_errors:
                print(f'validate: {error}')
            return 1
        if args.yaml:
            from silk.pdk.basepdk import BasePDK
            pdk = BasePDK(args.yaml, library_location=args.lib or _default_lib())
            label = args.yaml
        else:
            from silk import load_pdk
            pdk = load_pdk(args.pdk, library_location=args.lib)
            label = args.pdk
    except (PDKError, yaml.YAMLError) as exc:
        print(f'validate: {exc}')
        return 1
    except (PDKRegistryError, OSError) as exc:
        print(f'validate: {exc}')
        return 2

    try:
        issues, scope = _validate_with_scope(pdk, simulate=args.simulate)
    except PDKValidationError as exc:
        # Every PDKValidationError message is self-contained (it names the
        # offending path and what to do about it), so print it as one line
        # rather than wrapping it in a guess about which failure it was.
        print(f'validate: {exc}')
        return 2

    # The verdict always carries the scope it was reached in: a clean bill
    # from the lib's include closure and one from the unscoped $PDK_ROOT walk
    # are different claims, and printing them identically hides which files an
    # arity was vouched for by.
    n = len(pdk._devices)
    if not issues:
        print(f'OK: {label}: {n} devices, no issues (scope: {scope}).')
        return 0
    print(f'{label}: {len(issues)} issue(s) across {n} devices (scope: {scope}):')
    for i in issues:
        print(f'  {i.device:24s} {i.kind:18s} {i.detail}')
    return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
