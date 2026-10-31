"""Runnable demo scripts (``python -m silk.examples.<name>``).

This package ships in the wheel, so every example runs from an installed
copy as well as from a checkout. That is why the one thing this module
exports is a *writable* output directory: an example must never write
beside its own source file, which for an installed user is inside
``site-packages`` — unrecorded by pip, left behind by ``pip uninstall``,
and read-only on a system-wide install.
"""
import os

from silk import settings as _settings

__all__ = ['SILK_EXAMPLES_OUT_ENV', 'DEFAULT_OUTPUT_DIRNAME',
           'ExampleOutputError', 'example_output_dir']

#: Environment variable that redirects everything the examples write.
SILK_EXAMPLES_OUT_ENV = 'SILK_EXAMPLES_OUT'

#: Directory created under the current working directory when that variable
#: is unset. Hyphenated on purpose: it is not a legal Python identifier, so
#: it can never shadow an importable package sitting on ``sys.path``.
DEFAULT_OUTPUT_DIRNAME = 'silk-output'


class ExampleOutputError(OSError):
    """The examples' output directory could not be created or written to."""


def example_output_dir(*parts, create=True):
    """Return the directory this example should write into.

    The ``SILK_EXAMPLES_OUT`` setting (environment variable, then the settings
    file; :mod:`silk.settings`) when set, otherwise ``./silk-output`` resolved
    against the **current working directory** — never against
    ``os.path.dirname(__file__)``, which is inside the installed package.

    Parameters
    ----------
    *parts : str
        Optional subdirectory components, e.g.
        ``example_output_dir('design_loop', 'work')``.
    create : bool, default True
        Create the directory (and its parents). Pass ``False`` for a
        module-level constant, so that merely *importing* an example does
        not touch the filesystem; the first call that actually writes
        creates it.

    Raises
    ------
    ExampleOutputError
        If ``create`` is set and the directory cannot be made. The message
        names the path and the variable that moves it, so the failure reads
        as "point the examples somewhere writable", not as a silk bug.
    """
    root = _settings.get(SILK_EXAMPLES_OUT_ENV)
    path = os.path.join(os.path.abspath(root), *parts)
    if create:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            raise _unwritable(path, exc.strerror or str(exc)) from exc
        # An existing but read-only directory passes makedirs() and fails
        # much later, deep inside silk, where it reads as a library bug.
        if not os.access(path, os.W_OK):
            raise _unwritable(path, 'directory is not writable')
    return path


def _unwritable(path, why):
    return ExampleOutputError(
        f'cannot write the examples output directory {path!r}: {why}. Set '
        f'{SILK_EXAMPLES_OUT_ENV} (environment or the silk settings file, see '
        f'docs/settings.md) to a writable directory, or run from one '
        f'— by default the examples write to ./{DEFAULT_OUTPUT_DIRNAME} '
        f'under the current working directory.')
