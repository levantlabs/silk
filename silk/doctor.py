"""silk.doctor — one-shot check that this environment can actually run silk.

Answers the first question a new user or agent has ("is my setup the problem,
or my circuit?") in one command: interpreter + dependencies, the silk settings
(file path, then every setting with its value and source — see
:mod:`silk.settings`), the ngspice binary, ``PDK_ROOT`` and the model library
of every *registered* PDK manifest, plus the optional extras (Xyce, a usable
matplotlib backend).

Required checks gate the exit code (0 = ready, 1 = something to fix); optional
ones report ``[skip]`` when absent and never gate. Every failure prints one
indented, copy-pasteable fix.

Deliberately dependency-light and import-late: the doctor has to run *in* a
broken environment, so nothing heavier than the stdlib is imported at module
level and every check catches its own ImportError.

``python -m silk.doctor`` still imports the ``silk`` package itself, which
needs numpy — if that import is what's broken you get numpy's traceback before
any check runs. Run the file directly (``python silk/doctor.py``) to bypass the
package import and get the full report anyway.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

if __package__:
    from silk import settings as _settings
    from silk._json_codec import encode_json
else:
    # Direct script mode must work even if numpy is missing.
    import settings as _settings
    from _json_codec import encode_json

OK = 'ok'
FAIL = 'FAIL'
SKIP = 'skip'
# Real, actionable, but NOT gating: silk works, this one thing doesn't. Used
# for a registered PDK that isn't installed on this machine (registered means
# silk ships its catalog, not that you need it) and for a PDK installed but
# not finished (OSDI models declared, not compiled). FAIL had made doctor —
# and CI running it — red on every machine missing any one of the three PDKs.
WARN = 'warn'

def _min_python():
    """The floor from the package's own metadata (requires-python), so doctor
    cannot drift from pyproject — it used to carry a second copy of the
    number, and both were wrong together (3.9, on which `import silk` dies).
    Falls back to the known floor when silk is not installed as a dist
    (e.g. run from a bare checkout with PYTHONPATH)."""
    try:
        from importlib.metadata import metadata
        m = re.match(r'>=\s*(\d+)\.(\d+)',
                     metadata('circuitsilk')['Requires-Python'] or '')
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass
    return (3, 10)


MIN_PYTHON = _min_python()
# module name -> pip distribution name (they differ for pyyaml)
REQUIRED_IMPORTS = [('numpy', 'numpy'), ('matplotlib', 'matplotlib'),
                    ('yaml', 'pyyaml'), ('pydantic', 'pydantic>=2')]
_NGSPICE_RE = re.compile(r'ngspice-(\S+)')


@dataclass
class Check:
    """One environment check. ``fix`` is empty unless ``status`` is FAIL."""
    name: str
    status: str
    detail: str = ''
    fix: str = ''


def _pip_fix(dist):
    """`pip install` line naming *this* interpreter, so the fix lands in the
    venv the doctor was run under rather than whatever `pip` resolves to."""
    return f'{sys.executable} -m pip install {dist!r}'


def check_python():
    ver = '.'.join(str(n) for n in sys.version_info[:3])
    want = '.'.join(str(n) for n in MIN_PYTHON)
    if sys.version_info[:2] >= MIN_PYTHON:
        return [Check('python', OK, f'{ver} (>= {want}) at {sys.executable}')]
    return [Check('python', FAIL, f'{ver} is older than {want}',
                  f'run silk under python{want} or newer '
                  f'(current interpreter: {sys.executable})')]


def check_imports():
    """One line per hard dependency — a precise fix beats a bulk reinstall."""
    import importlib
    checks = []
    for mod, dist in REQUIRED_IMPORTS:
        try:
            m = importlib.import_module(mod)
        except ImportError as exc:
            checks.append(Check(f'import {mod}', FAIL, str(exc), _pip_fix(dist)))
        else:
            ver = getattr(m, '__version__', 'unknown version')
            checks.append(Check(f'import {mod}', OK, f'{mod} {ver}'))
    return checks


def _version_line(argv, pattern=None):
    """First useful line of `argv --version` output, or None if it won't run."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    text = f'{out.stdout}\n{out.stderr}'
    if pattern:
        m = pattern.search(text)
        if m:
            return m.group(0)
    for line in text.splitlines():
        if line.strip().strip('*'):
            return line.strip().lstrip('*').strip()
    return None


_SOURCE_LABEL = {'env': 'environment', 'file': 'settings file',
                 'default': 'built-in default'}


def _settings_file_phrase():
    """Where a user should write a setting, for a fix hint: the actual file
    path, or why there is none."""
    path, state = _settings.settings_file_status()
    if path is None:
        return ('the silk settings file (currently disabled by an empty '
                'SILK_SETTINGS; see docs/settings.md)')
    return f'{path} (see docs/settings.md)'


def _set_it(key, example):
    """'add KEY=example to <settings file>' — the common fix tail."""
    return f'add {key}={example} to {_settings_file_phrase()}'


def check_settings():
    """The settings file (path and state), then every setting with its value
    and source. Never FAIL except for a file silk cannot use; an unknown key
    is a [warn], so a misspelling is never silently ignored."""
    path, state = _settings.settings_file_status()
    if path is None:
        head = 'disabled by SILK_SETTINGS (environment and defaults only)'
    else:
        head = f'{path} ({state})'
    try:
        _settings.read_settings_file()
    except _settings.SettingsError as exc:
        return [Check('settings file', FAIL, str(exc),
                      'correct the file (KEY=value lines, see docs/settings.md), '
                      'or set SILK_SETTINGS to an empty value to ignore it')]
    checks = [Check('settings file', OK, head)]
    known = ', '.join(_settings.KNOWN_SETTINGS)
    for key, line in _settings.unknown_keys():
        checks.append(Check(f'setting {key}', WARN,
                            f'unknown key at {path}:{line}; silk does not use it',
                            f'remove it or correct the spelling (known settings: '
                            f'{known})'))
    for st in _settings.all_settings():
        if st.value is None:
            checks.append(Check(f'setting {st.name}', SKIP,
                                f'not set ({_settings.KNOWN_SETTINGS[st.name]}; '
                                f'optional)'))
            continue
        checks.append(Check(f'setting {st.name}', OK,
                            f'{st.value}  (source: {st.source})'))
    return checks


def check_region_thresholds():
    """Validate the selected customer policy before any simulation."""
    path = _settings.get('SILK_REGION_THRESHOLDS')
    if not path:
        return [Check('region thresholds', SKIP, 'no user policy selected')]
    try:
        from silk.circuit.region_thresholds import load_thresholds_file, RegionThresholdsError
    except ImportError as exc:
        return [Check('region thresholds', FAIL, str(exc),
                      'install silk and its dependencies before validating policy')]
    try:
        load_thresholds_file(path)
    except RegionThresholdsError as exc:
        return [Check('region thresholds', FAIL, str(exc),
                      'correct the threshold file (docs/operating-point.md), '
                      'or unset SILK_REGION_THRESHOLDS to use shipped policy')]
    return [Check('region thresholds', OK, f'{path}: valid per-PDK policy')]


def check_ngspice():
    # Resolve the same binary the runner will invoke, so doctor never reports a
    # different ngspice than the one that runs simulations (SILK_NGSPICE_BIN).
    from silk.simulation.runner import ngspice_executable
    requested = ngspice_executable()
    exe = shutil.which(requested)
    st = _settings.lookup('SILK_NGSPICE_BIN')
    if not exe:
        if st.source != 'default':
            return [Check('ngspice', FAIL,
                          f'SILK_NGSPICE_BIN={requested!r} (from the '
                          f'{_SOURCE_LABEL[st.source]}) is not executable',
                          'Point SILK_NGSPICE_BIN at an ngspice binary, or unset it '
                          'to fall back to PATH. ' + _where('SILK_NGSPICE_BIN', st))]
        return [Check('ngspice', FAIL, 'not found on PATH',
                      'sudo apt install ngspice   # macOS: brew install ngspice; or '
                      + _set_it('SILK_NGSPICE_BIN', '/path/to/ngspice'))]
    ver = _version_line([exe, '--version'], _NGSPICE_RE)
    via = _via('SILK_NGSPICE_BIN', st)
    checks = [Check('ngspice', OK, f'{ver or "version unknown"} at {exe}{via}')]
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    policy = NgspiceCompatibility(ver.removeprefix('ngspice-') if ver else None)
    if policy.advice:
        checks.append(Check('ngspice support', WARN if policy.below_minimum else OK,
                            policy.advice, 'update ngspice when possible'))
    return checks


def _where(key, st):
    """Name where *key*'s current value came from, for a fix hint."""
    if st.source == 'file':
        return f'It is set in {_settings_file_phrase()}.'
    if st.source == 'env':
        return (f'It is set in the environment, which overrides '
                f'{_settings_file_phrase()}.')
    return 'Or ' + _set_it(key, '...') + '.'


def _via(key, st):
    """' (via KEY)' for an environment value, ' (via KEY in the settings
    file)' for a file value, '' for the built-in default."""
    if st.source == 'env':
        return f' (via {key})'
    if st.source == 'file':
        return f' (via {key} in the settings file)'
    return ''


def _pdk_root():
    """Root silk will resolve lib paths against — mirrors silk.pdk.load_pdk."""
    return _settings.get('PDK_ROOT')


def _pdk_fix():
    return ('export PDK_ROOT=/path/to/pdk   # directory that holds your installed PDKs '
            '(for example ~/pdk, which holds sky130A/) — see docs/installing-the-pdk.md; '
            'or, to keep it, ' + _set_it('PDK_ROOT', '/path/to/pdk'))


def _lib_fix():
    return (f'install the PDK under PDK_ROOT ({_pdk_root()}; '
            f'github.com/levantlabs/pdksetup: `pdk install sky130`) — see '
            f'docs/installing-the-pdk.md; to use another directory, '
            + _set_it('PDK_ROOT', '/path/to/pdk'))


def check_pdk_root():
    st = _settings.lookup('PDK_ROOT')
    root = st.value
    if not os.path.isdir(root):
        return [Check('PDK_ROOT', FAIL,
                      f'{root!r} (from the {_SOURCE_LABEL[st.source]}) is not '
                      f'a directory', _pdk_fix())]
    return [Check('PDK_ROOT', OK, f'{root} ({_SOURCE_LABEL[st.source]})')]


def check_pdks():
    """Every registered manifest's model library, resolved like load_pdk does."""
    try:
        from silk.pdk import registry
    except ImportError as exc:
        return [Check('pdk manifests', FAIL,
                      f'cannot import silk.pdk.registry: {exc}',
                      f'install silk and its dependencies: '
                      f'{sys.executable} -m pip install -e .')]
    try:
        names = registry.available()
    except Exception as exc:                     # malformed/duplicate manifest
        return [Check('pdk manifests', FAIL, f'{type(exc).__name__}: {exc}',
                      'fix the offending silk/pdk/data/*.pdk.yaml '
                      '(see docs/adding-a-pdk.md §2)')]
    if not names:
        return [Check('pdk manifests', FAIL, 'no *.pdk.yaml in silk/pdk/data',
                      'add a PDK manifest — see docs/adding-a-pdk.md')]
    root = _pdk_root()
    checks = []
    for name in names:
        library = registry.manifest(name)['library']
        lib = library['lib_path']
        path = lib if os.path.isabs(lib) else os.path.join(root, lib)
        if not os.path.isfile(path):
            checks.append(Check(f'pdk {name}', WARN,
                                f'not installed (lib not found at {path})',
                                _lib_fix()))
            continue
        # Present is not usable: a PDK whose models are compiled Verilog-A
        # (manifest `library.osdi:`) cannot simulate until the plugins are
        # compiled. This used to report `ok` for SG13G2 on a machine where no
        # SG13G2 run could succeed — the lib file existed, the models did not.
        osdi = [p if os.path.isabs(p) else os.path.join(root, p)
                for p in library.get('osdi', [])]
        missing = [p for p in osdi if not os.path.isfile(p)]
        if missing:
            checks.append(Check(
                f'pdk {name}', WARN,
                f'installed, but {len(missing)}/{len(osdi)} OSDI models not '
                f'compiled (cannot simulate): {missing[0]}',
                "install openvaf-r, then run the PDK's compile script (for "
                "SG13G2: $PDK_ROOT/ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/"
                "install.py)"))
            continue
        # Compiled is not loadable either: the plugins carry an OSDI interface
        # version and the ngspice binary decides whether it speaks it
        # (openvaf-r emits v0.4; ngspice-42 stops at v0.3 and then reports
        # every model as unknown). Ask the same binary the runner would use —
        # the only authority — rather than comparing version numbers.
        problem = _osdi_load_problem(osdi)
        if problem:
            checks.append(Check(
                f'pdk {name}', WARN,
                f'installed and compiled, but {_ngspice_label()} cannot load '
                f'its OSDI plugins (cannot simulate): {problem}',
                _osdi_version_fix()))
            continue
        loaded = f' ({len(osdi)} OSDI plugins load)' if osdi else ''
        checks.append(Check(f'pdk {name}', OK, f'{path}{loaded}'))
        checks.extend(_statistical_checks(name, library, path, root))
    return checks


def _statistical_checks(name, library, path, root):
    """One [warn] per statistical library section the manifest relies on that
    the installed library lacks. Never FAIL: Monte Carlo is optional, and the
    run itself refuses with MissingLibrarySectionError."""
    block = library.get('statistical')
    if not block:
        return []
    from silk.exceptions import PDKError
    from silk.pdk.statistical import missing_sections, parse_statistical
    try:
        st = parse_statistical(block, root, name)
        missing = missing_sections(path, st.required_sections())
    except (PDKError, OSError) as exc:
        return [Check(f'pdk {name} statistical', WARN, str(exc),
                      'fix the manifest library.statistical block or the '
                      'PDK install')]
    hint = (f' (needs open_pdks {st.min_open_pdks} or newer)'
            if st.min_open_pdks else '')
    return [Check(f'pdk {name} section {sec}', WARN,
                  f'library has no {sec!r} section; Monte Carlo cannot run '
                  f'on {name}{hint}',
                  'update the PDK install (docs/installing-the-pdk.md); '
                  'open_pdks releases before the statistical sections lack it')
            for sec in missing]


def _osdi_version_fix():
    from silk.simulation._ngspice_compat import NgspiceCompatibility
    release = NgspiceCompatibility(None).since('osdi_v04')
    return ('use an ngspice that speaks the plugins\' OSDI version (openvaf-r '
            f'emits v0.4, accepted from ngspice {release} on): SILK_NGSPICE_BIN=/path/'
            'to/newer/ngspice for the commands that need it — or, to use it for '
            'every run, ' + _set_it('SILK_NGSPICE_BIN', '/path/to/newer/ngspice')
            + ' — or recompile the plugins for this ngspice')


def _osdi_load_problem(osdi_paths):
    """The resolved ngspice's first complaint about loading *osdi_paths*, or
    None (also None when there is no runnable ngspice — check_ngspice owns
    that failure). Indirection so tests can stub the probe."""
    from silk.simulation.runner import ngspice_executable, osdi_load_problem
    exe = shutil.which(ngspice_executable())
    if not exe:
        return None
    return osdi_load_problem(osdi_paths, ngspice=exe)


def _ngspice_label():
    """'ngspice-42' style tag of the resolved binary, for a message."""
    from silk.simulation.runner import ngspice_executable
    exe = shutil.which(ngspice_executable())
    ver = _version_line([exe, '--version'], _NGSPICE_RE) if exe else None
    return ver or 'the resolved ngspice'


def check_xyce():
    """Optional second simulator; resolved as silk.simulation.xyce_runner does."""
    st = _settings.lookup('SILK_XYCE_BIN')
    exe = st.value if st.source != 'default' else shutil.which('Xyce')
    if not exe or not (os.path.isfile(exe) or shutil.which(exe)):
        return [Check('xyce', SKIP, 'not found (optional: ngspice is the default '
                                    'simulator; to use Xyce, '
                                    + _set_it('SILK_XYCE_BIN', '/path/to/Xyce')
                                    + ' or export SILK_XYCE_BIN)')]
    ver = _version_line([exe, '-v'])
    via = _via('SILK_XYCE_BIN', st)
    return [Check('xyce', OK, f'{ver or "version unknown"} at {exe}{via}')]


def check_matplotlib_backend():
    """Optional: can we actually draw? Headless boxes need the Agg fallback."""
    try:
        import matplotlib
    except ImportError:
        return [Check('matplotlib backend', SKIP, 'matplotlib not installed '
                                                  '(see the import check above)')]
    try:
        import matplotlib.pyplot  # noqa: F401
    except Exception:
        try:
            matplotlib.use('Agg')
            import matplotlib.pyplot  # noqa: F401,F811
        except Exception as exc:
            return [Check('matplotlib backend', SKIP,
                          f'pyplot unusable even with Agg ({type(exc).__name__}: '
                          f'{exc}); plotting will fail, simulation still works')]
        return [Check('matplotlib backend', OK, 'usable via the Agg fallback '
                                                '(no display); plots must be saved to file')]
    return [Check('matplotlib backend', OK, f'{matplotlib.get_backend()} backend')]


REQUIRED_CHECKS = [check_python, check_settings, check_imports, check_region_thresholds, check_ngspice,
                   check_pdk_root, check_pdks]
OPTIONAL_CHECKS = [check_xyce, check_matplotlib_backend]


def run_checks():
    """Every check, in report order. Optional checks never return FAIL."""
    checks = []
    for fn in REQUIRED_CHECKS + OPTIONAL_CHECKS:
        try:
            checks.extend(fn())
        except _settings.SettingsError as exc:
            # check_settings already reported the file; every check that
            # needs a setting cannot run until it is fixed.
            if fn is check_settings:
                raise
            name = fn.__name__.replace('check_', '').replace('_', ' ')
            checks.append(Check(name, FAIL,
                                f'cannot resolve settings: {exc}',
                                'fix the settings file first (see the '
                                '"settings file" line above)'))
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m silk.doctor',
        description='Check that this environment can run silk: interpreter, '
                    'dependencies, silk settings, ngspice, PDK_ROOT and every '
                    'registered PDK '
                    'library, plus optional extras. Exit 0 = ready, 1 = fix needed.')
    parser.add_argument('--json', action='store_true',
                        help='emit the results as a JSON list of '
                             '{name, status, detail, fix} instead of text')
    args = parser.parse_args(argv)

    checks = run_checks()
    failed = [c for c in checks if c.status == FAIL]
    if args.json:
        print(json.dumps(encode_json([{'name': c.name, 'status': c.status,
                           'detail': c.detail, 'fix': c.fix} for c in checks]), allow_nan=False,
                         indent=2))
        return 1 if failed else 0

    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f'[{c.status:^4s}] {c.name:{width}s}  {c.detail}')
        if c.status in (FAIL, WARN) and c.fix:
            print(f'{"":{width + 9}s}fix: {c.fix}')
    n_ok = sum(1 for c in checks if c.status == OK)
    n_skip = sum(1 for c in checks if c.status == SKIP)
    n_warn = sum(1 for c in checks if c.status == WARN)
    tail = f'{n_ok} ok, {n_skip} skipped (optional).'
    if n_warn:
        tail = f'{n_ok} ok, {n_warn} warning(s), {n_skip} skipped (optional).'
    if failed:
        print(f'\ndoctor: silk is not ready — {len(failed)} failed, {tail}')
        return 1
    print(f'\ndoctor: silk is ready — {tail}')
    return 0


if __name__ == '__main__':
    if not __package__:
        # Run as a plain script (`python silk/doctor.py`) — put the repo root on
        # sys.path so the PDK check can still find silk.pdk.
        sys.path.insert(
            0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
