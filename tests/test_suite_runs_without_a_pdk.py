"""The suite must COLLECT on a machine with no PDK and no ngspice.

This is guarded here because the failure mode is silent and total: one module-level ``load_pdk()``
anywhere in an imported module makes pytest fail at **collection**, and pytest
then exits before running a single test. That is what happened —
``silk/examples/ota_5t_guide.py`` loaded the PDK and built a Module at import
time, four test files imported it, and the entire suite refused to start on any
machine without a PDK installed.

Individual tests may of course *need* a PDK or a simulator. They must say so
with ``@needs_pdk`` / ``@needs_ngspice`` from ``tests._env`` and skip, never
fail and never break collection.

Collection-only, in a subprocess, ~1 s.
"""
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _collect(env_overrides):
    env = dict(os.environ, MPLBACKEND='Agg', **env_overrides)
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', '--collect-only',
         '-p', 'no:cacheprovider', 'tests'],
        cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=600)


def test_collection_succeeds_without_a_pdk(tmp_path):
    """No PDK_ROOT, and HOME redirected so `load_pdk`'s ~/pdk fallback misses.

    The fallback is why this rots quietly on a dev machine: `~/pdk` exists
    there, so a module-level `load_pdk()` looks harmless locally and only
    breaks for everyone else.
    """
    cp = _collect({'PDK_ROOT': str(tmp_path / 'nopdk'), 'HOME': str(tmp_path)})
    assert cp.returncode == 0, (
        'the suite does not collect without a PDK — something imports a PDK at '
        f'module scope:\n{cp.stdout[-3000:]}\n{cp.stderr[-2000:]}')


def test_collection_succeeds_without_ngspice(tmp_path):
    empty = tmp_path / 'bin'
    empty.mkdir()
    cp = _collect({'PATH': f'{empty}{os.pathsep}/usr/bin{os.pathsep}/bin',
                   'SILK_NGSPICE_BIN': str(empty / 'absent')})
    assert cp.returncode == 0, (
        'the suite does not collect without ngspice:\n'
        f'{cp.stdout[-3000:]}\n{cp.stderr[-2000:]}')
