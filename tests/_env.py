"""What this machine can actually run — shared skip markers.

The suite needs two things that are not Python dependencies: an **ngspice
binary** and an **installed sky130 PDK**. Neither arrives with
``pip install -e ".[dev]"``, so on a bare checkout a large slice of the suite
cannot pass no matter how correct the code is.

Failing is the wrong signal there. A failure means "silk is broken"; what is
actually true is "this machine has no PDK". These markers say the second thing,
so ``pytest`` on a fresh clone produces a usable result with honest skips
instead of a wall of ``FileNotFoundError``.

Usage::

    from tests._env import needs_pdk

    pytestmark = needs_pdk              # whole module
    @needs_pdk                          # one test
    def test_something(): ...

Both probes run **once** per session, at import. Every location is resolved
through :mod:`silk.settings` (environment variable, then the settings file,
then the built-in default), so the suite and silk agree on what is available.
"""
import os
import shutil

import pytest

from silk import settings


def pdk_available(name='sky130'):
    """True when *name*'s model library actually resolves on this machine.

    Deliberately goes through ``load_pdk`` for the *resolution* ($PDK_ROOT,
    then the registry manifest's lib_path), so the probe cannot disagree with
    reality about WHERE the library would be — then stats that path itself.

    The stat is the load-bearing half. ``load_pdk`` no longer raises for a
    missing library (TASK-71: the catalog ships in-package, so a PDK that is
    not installed still loads usefully and PDK-free decks still run), which
    means "``load_pdk`` did not raise" now answers True on a machine with no
    PDK at all — it would turn every ``needs_pdk`` skip into a failure.
    """
    try:
        from silk import load_pdk
        lib = load_pdk(name).library_location
    except Exception:
        return False
    return bool(lib) and os.path.isfile(lib)


def _ngspice_available():
    """True when the ngspice binary silk would invoke actually exists.

    Resolved through :mod:`silk.settings`, as silk resolves it — the
    ``SILK_NGSPICE_BIN`` setting first, else ``ngspice`` on ``PATH`` — so the
    probe follows the same binary the run would use.
    """
    st = settings.lookup('SILK_NGSPICE_BIN')
    if st.source != 'default':
        return os.path.isfile(st.value) and os.access(st.value, os.X_OK)
    return shutil.which('ngspice') is not None


def xyce_executable():
    """The Xyce binary silk would invoke (the ``SILK_XYCE_BIN`` setting, else
    ``Xyce`` on PATH), or None when it does not exist."""
    exe = settings.get('SILK_XYCE_BIN')
    return shutil.which(exe) if exe else None


def browser_executable():
    """The browser named by the ``SILK_FIREFOX`` setting, else
    ``SILK_CHROME``, or None. Explicit opt-in only: a browser on PATH is not
    assumed to run headless."""
    return settings.get('SILK_FIREFOX') or settings.get('SILK_CHROME')


def chrome_setting():
    """The ``SILK_CHROME`` setting, or None."""
    return settings.get('SILK_CHROME')


HAVE_PDK = pdk_available()
HAVE_NGSPICE = _ngspice_available()
HAVE_XYCE = xyce_executable() is not None

needs_pdk = pytest.mark.skipif(
    not HAVE_PDK,
    reason='needs an installed sky130 PDK (PDK_ROOT setting); see '
           'docs/installing-the-pdk.md')

#: Per-PDK gate. sky130 is installed on the dev machine but a second PDK may
#: not be, and a test for one must not fail because the other is missing.
def needs_pdk_named(name):
    return pytest.mark.skipif(
        not pdk_available(name),
        reason=f'needs the {name} PDK installed under the PDK_ROOT setting')


needs_ngspice = pytest.mark.skipif(
    not HAVE_NGSPICE,
    reason='needs the ngspice binary (SILK_NGSPICE_BIN setting or ngspice on PATH)')


def osdi_runnable(name):
    """True when *name*'s compiled Verilog-A plugins actually load in the
    ngspice silk would run — the whole chain a real OSDI simulation needs.

    Three things can be missing, and each is a different skip reason on a
    developer's box: the PDK install, the compiled ``.osdi`` files (openvaf-r
    is a separate install), and an ngspice new enough for their OSDI version
    (Ubuntu's ngspice-42 refuses openvaf-r's v0.4 plugins). Only the ngspice
    binary knows the last one, so this asks it — the same probe
    ``python -m silk.doctor`` runs — rather than comparing version numbers.
    Cached per name: the probe is one ngspice launch.
    """
    if name in _OSDI_RUNNABLE:
        return _OSDI_RUNNABLE[name]
    ok = False
    if HAVE_NGSPICE and pdk_available(name):
        from silk import load_pdk
        from silk.simulation.runner import osdi_load_problem
        osdi = list(getattr(load_pdk(name), 'osdi_files', None) or [])
        ok = bool(osdi) and all(os.path.isfile(p) for p in osdi) \
            and osdi_load_problem(osdi) is None
    _OSDI_RUNNABLE[name] = ok
    return ok


_OSDI_RUNNABLE = {}


def needs_osdi_runnable(name):
    """Skip unless *name* is installed, compiled AND loadable by the resolved
    ngspice — see :func:`osdi_runnable`."""
    return pytest.mark.skipif(
        not osdi_runnable(name),
        reason=f'needs the {name} PDK installed, its OSDI plugins compiled, '
               f'and an ngspice that loads them ($SILK_NGSPICE_BIN; see '
               f'docs/installing-the-pdk.md)')


# ---------------------------------------------------------------------------
# Scratch directories (TASK-453)
# ---------------------------------------------------------------------------

_scratch_factory = None


def scratch_dir(name='scratch'):
    """A fresh empty directory (as ``str``) under pytest's basetemp.

    Use this where a test helper (not a test function, so it cannot take the
    ``tmp_path`` fixture) needs a ``dir_folder``. It replaces an uncleaned
    ``mkdtemp`` call: pytest prunes its own basetemp, which honours
    ``TMPDIR``. ``tests/conftest.py`` binds the session's
    ``tmp_path_factory`` before any test or fixture runs.
    """
    if _scratch_factory is None:
        raise RuntimeError(
            'scratch_dir() needs the pytest session fixture from '
            'tests/conftest.py (tmp_path_factory is not bound yet)')
    return str(_scratch_factory.mktemp(name))
