"""Cached version of the same ngspice executable selected by the runner."""
from functools import lru_cache
import os
import re
import shutil

from silk.simulation.runner import ngspice_executable


def ngspice_version():
    """Return the release token, or None when unavailable/unrecognizable.

    Resolve selection on every call so environment/settings/PATH changes take
    effect. Probe each canonical executable path only once per process; restart
    the process after replacing a binary in place.
    """
    executable = shutil.which(ngspice_executable())
    if executable is None:
        return None
    return _binary_version(os.path.realpath(executable))


@lru_cache(maxsize=None)
def _binary_version(executable):
    # Reuse doctor's bounded subprocess probe, including stderr and failures.
    from silk.doctor import _version_line, _NGSPICE_RE
    label = _version_line([executable, '--version'], _NGSPICE_RE)
    match = re.fullmatch(r'ngspice-(\d+(?:\.\d+)*\+?)', label or '')
    return match.group(1) if match else None
