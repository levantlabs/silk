"""silk.settings — where silk finds its tools and PDKs.

Six locations are silk *settings*: the PDK root and the external programs
silk launches. Each is named by the same variable it has always been read
from::

    PDK_ROOT            directory holding the installed PDKs
    SILK_NGSPICE_BIN    the ngspice binary
    SILK_XYCE_BIN       the Xyce binary (optional simulator)
    SILK_EXAMPLES_OUT   where the worked examples write
    SILK_FIREFOX        a headless-capable Firefox (optional browser smoke)
    SILK_CHROME         a headless-capable Chrome/Chromium (same)

A value resolves as **environment variable > settings file > built-in
default**. The settings file is optional; with none, silk behaves exactly as
it does with environment variables alone.

The file is ``$SILK_SETTINGS`` when that is set, else
``$XDG_CONFIG_HOME/silk/silk.env``, else ``~/.config/silk/silk.env``.
``SILK_SETTINGS`` set to an empty value, or to ``/dev/null``, disables the
file: environment and built-in defaults only (what CI and test runners want).

Format — ``KEY=value`` lines::

    # comments and blank lines are ignored
    PDK_ROOT=~/pdk
    export SILK_XYCE_BIN="/opt/xyce/bin/Xyce"   # optional 'export ' prefix
    SILK_EXAMPLES_OUT='~/silk runs'             # quoted values keep spaces

A leading ``~`` is expanded. An unquoted value ends at whitespace followed by
``#``. An empty value counts as not set. A line that is not ``KEY=value``,
an unterminated quote, or a key given twice raises :class:`SettingsError`
naming the file and line. A key silk does not know is kept and reported by
``python -m silk.doctor`` as a warning, never silently dropped.

Stdlib only, like :mod:`silk.doctor`, which must run in a broken environment.
See docs/settings.md.
"""
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from silk.exceptions import SettingsError
except ImportError:      # `python silk/doctor.py` with silk itself unimportable
    from exceptions import SettingsError  # type: ignore[no-redef]

__all__ = ['KNOWN_SETTINGS', 'Setting', 'SettingsError', 'settings_path',
           'settings_file_status', 'read_settings_file', 'lookup', 'get',
           'is_configured', 'unknown_keys', 'all_settings']

#: Every setting silk reads, with a one-line description (doctor prints it).
KNOWN_SETTINGS: Dict[str, str] = {
    'SILK_REGION_THRESHOLDS': 'per-PDK operating-region thresholds YAML',
    'PDK_ROOT': 'directory holding the installed PDKs',
    'SILK_NGSPICE_BIN': 'the ngspice binary',
    'SILK_XYCE_BIN': 'the Xyce binary (optional simulator)',
    'SILK_EXAMPLES_OUT': 'where the worked examples write',
    'SILK_FIREFOX': 'Firefox for the optional browser smoke test',
    'SILK_CHROME': 'Chrome/Chromium for the optional browser tests',
}

SETTINGS_ENV = 'SILK_SETTINGS'
_DISABLED = ('', os.devnull)


def _default(name: str) -> Optional[str]:
    """The built-in default, or None when the setting has none."""
    if name == 'PDK_ROOT':
        return os.path.expanduser('~/pdk')
    if name == 'SILK_NGSPICE_BIN':
        return 'ngspice'
    if name == 'SILK_XYCE_BIN':
        return shutil.which('Xyce') or 'Xyce'
    if name == 'SILK_EXAMPLES_OUT':
        return os.path.join(os.getcwd(), 'silk-output')
    return None


@dataclass(frozen=True)
class Setting:
    """One resolved setting. ``source`` is ``'env'``, ``'file'`` or
    ``'default'``; ``value`` is None only for an unset setting with no
    built-in default."""
    name: str
    value: Optional[str]
    source: str


def settings_path() -> Optional[str]:
    """The settings file silk reads, or None when ``$SILK_SETTINGS``
    disables it (empty or ``/dev/null``). The path need not exist."""
    explicit = os.environ.get(SETTINGS_ENV)
    if explicit is not None:
        if explicit.strip() in _DISABLED:
            return None
        return os.path.abspath(os.path.expanduser(explicit))
    xdg = os.environ.get('XDG_CONFIG_HOME')
    base = xdg if xdg else os.path.join(os.path.expanduser('~'), '.config')
    return os.path.join(base, 'silk', 'silk.env')


def settings_file_status() -> Tuple[Optional[str], str]:
    """``(path, state)`` with state ``'found'``, ``'not found'`` or
    ``'disabled'`` (by ``$SILK_SETTINGS``)."""
    path = settings_path()
    if path is None:
        return None, 'disabled'
    return path, 'found' if os.path.exists(path) else 'not found'


_LINE_RE = re.compile(r'^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$')
_CACHE: Dict[str, Tuple[tuple, Dict[str, Tuple[str, int]]]] = {}


def _parse_value(raw: str, path: str, lineno: int) -> str:
    if raw[:1] in ('"', "'"):
        quote = raw[0]
        end = raw.find(quote, 1)
        if end < 0:
            raise SettingsError(f'{path}:{lineno}: unterminated {quote} quote')
        rest = raw[end + 1:].strip()
        if rest and not rest.startswith('#'):
            raise SettingsError(
                f'{path}:{lineno}: unexpected text after the closing quote: '
                f'{rest!r}')
        value = raw[1:end]
    else:
        value = re.split(r'\s+#', raw, maxsplit=1)[0].strip()
        if value[:1] == '#':
            value = ''
    return os.path.expanduser(value) if value.startswith('~') else value


def _parse(text: str, path: str) -> Dict[str, Tuple[str, int]]:
    entries: Dict[str, Tuple[str, int]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        m = _LINE_RE.match(stripped)
        if not m:
            raise SettingsError(
                f'{path}:{lineno}: expected KEY=value, got {stripped!r}')
        key = m.group(1)
        if key in entries:
            raise SettingsError(
                f'{path}:{lineno}: {key} is already set on line '
                f'{entries[key][1]}; give each setting once')
        entries[key] = (_parse_value(m.group(2), path, lineno), lineno)
    return entries


def read_settings_file(path: Optional[str] = None) -> Dict[str, Tuple[str, int]]:
    """Parse the settings file into ``{key: (value, line)}``, unknown keys
    included. ``{}`` when there is no file or it is disabled.

    Raises
    ------
    SettingsError
        For a malformed line (naming the file and line), or when
        ``$SILK_SETTINGS`` names a file that does not exist.
    """
    explicit = path is not None
    if path is None:
        path = settings_path()
        explicit = path is not None and bool(os.environ.get(SETTINGS_ENV))
    if path is None:
        return {}
    try:
        st = os.stat(path)
    except FileNotFoundError:
        if explicit:
            raise SettingsError(
                f'settings file {path} does not exist (named by '
                f'${SETTINGS_ENV}); create it, or set {SETTINGS_ENV} to an '
                f'empty value to use no settings file') from None
        return {}
    key = (st.st_mtime_ns, st.st_size, st.st_ino)
    cached = _CACHE.get(path)
    if cached and cached[0] == key:
        return cached[1]
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise SettingsError(f'cannot read settings file {path}: {exc}') from exc
    entries = _parse(text, path)
    _CACHE[path] = (key, entries)
    return entries


def _check_known(name: str) -> None:
    if name not in KNOWN_SETTINGS:
        raise KeyError(f'{name!r} is not a silk setting; known: '
                       f'{", ".join(KNOWN_SETTINGS)}')


def lookup(name: str) -> Setting:
    """Resolve *name*: environment variable > settings file > default.

    An empty environment variable or file value counts as not set.
    """
    _check_known(name)
    env = os.environ.get(name)
    if env:
        return Setting(name, env, 'env')
    entry = read_settings_file().get(name)
    if entry and entry[0]:
        return Setting(name, entry[0], 'file')
    return Setting(name, _default(name), 'default')


def get(name: str) -> Optional[str]:
    """The resolved value of *name* (see :func:`lookup`)."""
    return lookup(name).value


def is_configured(name: str) -> bool:
    """True when *name* comes from the environment or the settings file,
    i.e. the user chose it rather than silk's built-in default."""
    return lookup(name).source != 'default'


def unknown_keys() -> List[Tuple[str, int]]:
    """``(key, line)`` for every key in the settings file silk does not know."""
    return [(k, line) for k, (_, line) in read_settings_file().items()
            if k not in KNOWN_SETTINGS]


def all_settings() -> List[Setting]:
    """Every known setting, resolved, in :data:`KNOWN_SETTINGS` order."""
    return [lookup(name) for name in KNOWN_SETTINGS]
