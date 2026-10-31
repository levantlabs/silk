"""Read ngspice release facts from the simulator YAML.

Unknown versions enable no features and conservatively retain known bugs.
A development token (e.g. ``47+``) uses its numbered release's facts; it does
not promise the next release's features. Exact waveform verification tokens
are intentionally stricter until later emission forms are measured.
"""
import os
import re
import shutil
import sys
import warnings
from threading import Lock
from contextvars import ContextVar
from functools import wraps

from silk.simulators import load_simulator_config

_SELECTED = object()
_emission_version = ContextVar('silk_ngspice_emission_version', default=None)
_advised_binaries = set()
_advice_lock = Lock()


def _release(token):
    if token is None or not re.fullmatch(r'\d+(?:\.\d+)*\+?', str(token)):
        return None
    parts = [int(p) for p in str(token).rstrip('+').split('.')]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _with_ngspice_version(fn):
    """Scope a target release around the entire recursive generation operation."""
    @wraps(fn)
    def emit(module, *args, **kwargs):
        version = kwargs.get('ngspice_version')
        if version is not None:
            if not isinstance(version, str) or _release(version) is None:
                raise ValueError(f'Invalid ngspice_version {version!r}: expected a release string')
            if module.simulator != 'ngspice':
                raise ValueError('ngspice_version requires the ngspice simulator')
        token = _emission_version.set(version or _emission_version.get())
        try:
            return fn(module, *args, **kwargs)
        finally:
            _emission_version.reset(token)
    return emit


class NgspiceCompatibility:
    """Facts for an explicit release, or the cached selected binary by default."""

    def __init__(self, version=_SELECTED):
        if version is _SELECTED:
            from silk.simulation._ngspice_version import ngspice_version
            version = _emission_version.get()
            if version is None:
                version = ngspice_version()
        self.version = version
        self.release = _release(version)
        self.policy = load_simulator_config('ngspice').version_policy

    @property
    def minimum(self):
        return self.policy['minimum']

    @property
    def below_minimum(self):
        return self.release is not None and self.release < _release(self.minimum)

    @property
    def waveforms_verified(self):
        return self.version in self.policy['waveforms_verified']

    @property
    def verified_waveforms_label(self):
        return ', '.join(self.policy['waveforms_verified'])

    def since(self, name):
        """First release for a named fact (None means all known releases)."""
        return self.policy['behaviours'][name]['since']

    def has(self, name):
        """Whether the release has a named feature; unknown means False."""
        row = self.policy['behaviours'][name]
        if row['kind'] != 'feature':
            raise ValueError(f'{name} is a bug; use has_bug()')
        return self.release is not None and self.release >= _release(row['since'])

    def has_bug(self, name):
        """Whether a bug applies; unknown versions conservatively return True."""
        row = self.policy['behaviours'][name]
        if row['kind'] != 'bug':
            raise ValueError(f'{name} is a feature; use has()')
        return self.release is None or (
            (row['since'] is None or self.release >= _release(row['since']))
            and (row['fixed_in'] is None or self.release < _release(row['fixed_in'])))

    @property
    def advice(self):
        """Shared doctor/runtime information; no speculative advice for unknowns."""
        if self.release is None:
            return None
        bugs = [f"{row['user_message']} (fixed in {row['fixed_in']})"
                for name, row in self.policy['behaviours'].items()
                if row['kind'] == 'bug' and row['fixed_in'] is not None
                and self.has_bug(name)]
        if self.below_minimum:
            message = (f'ngspice {self.version} is older than the minimum version '
                       f'silk supports (silk supports releases from {self.minimum} onward).')
            if bugs:
                message += ' A newer release fixes '
        elif bugs:
            message = f'ngspice {self.version} works with silk, but a newer release fixes '
        else:
            return None
        if bugs:
            message += ('a bug that affects it: ' if len(bugs) == 1
                        else 'bugs that affect it: ')
            message += '; '.join(bugs) + '.'
        return message + ' Update ngspice when possible.'



class NgspiceVersionWarning(UserWarning):
    """Filterable information about a selected ngspice release, not a run failure."""


def _advice_stacklevel():
    """Find the caller outside silk internals, retaining example call sites.

    Walk frames explicitly for Python 3.10, which has no skip_file_prefixes.
    Examples are user programs even when imported as silk.examples modules.
    """
    frame = sys._getframe(1)
    level = 1
    try:
        while frame is not None:
            module = frame.f_globals.get('__name__', '')
            if (module != 'silk' and not module.startswith('silk.')) or (
                    module == 'silk.examples' or module.startswith('silk.examples.')):
                break
            frame = frame.f_back
            level += 1
        return level
    finally:
        del frame


def warn_version_advice(executable):
    """Emit nonfatal update information once per canonical binary path.

    Honor warning filters without allowing escalation of this advisory to
    abort a run. Unrelated warning-handler failures still propagate.
    """
    resolved = shutil.which(executable)
    if resolved is None:
        return
    key = os.path.realpath(resolved)
    with _advice_lock:
        if key in _advised_binaries:
            return
        advice = NgspiceCompatibility().advice
        if advice:
            advisory = NgspiceVersionWarning(advice)
            try:
                warnings.warn(advisory, stacklevel=_advice_stacklevel())
            except NgspiceVersionWarning as exc:
                # An error filter raises the very warning instance passed in.
                # Catch only our advisory, not errors from a custom handler.
                if exc is not advisory:
                    raise
        _advised_binaries.add(key)
