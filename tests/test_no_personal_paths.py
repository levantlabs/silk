"""No tracked shipping file may carry an absolute home-directory path.

A PDK or tool location is a setting (``$PDK_ROOT``, ``$SILK_NGSPICE_BIN``,
``$SILK_XYCE_BIN``), never a baked-in path. The pattern is generic: it matches
``/home/<anyone>/`` and ``/Users/<anyone>/``, not a particular user.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PATTERN = re.compile(rb'/(?:home|Users)/[A-Za-z0-9_.-]+(?:/|\b)')

# Frozen records and the task board; whether they ship is a release-manifest
# decision. (AGENTS.md was excluded until TASK-452 moved its tool paths into
# the owner's settings file.)
EXCLUDED_PREFIXES = ('docs/archive/', 'docs/superpowers/', 'docs/backlog/',
                     'backlog/')
EXCLUDED_FILES = set()
# Files whose subject is home-path handling: they need literal illustrative
# paths (``/home/user/x``) as test data or in docstrings.
ILLUSTRATIVE_FILES = {'silk/simulation/provenance.py', 'tests/test_provenance.py',
                      'docs/settings.md',
                      'docs/reports.md', 'tests/test_ci_workflow_runner.py',
                      'tests/test_no_personal_paths.py'}


def _tracked():
    try:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=REPO, check=True,
                             capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('not a git checkout')
    return [p for p in out.decode().split('\0') if p]


def _exempt(rel):
    return (rel in EXCLUDED_FILES or rel in ILLUSTRATIVE_FILES
            or rel.startswith(EXCLUDED_PREFIXES))


def test_pattern_is_generic():
    for user in ('alice', 'bob.smith', 'x_1'):
        assert PATTERN.search(f'/home/{user}/pdk'.encode())
        assert PATTERN.search(f'/Users/{user}/pdk'.encode())
    assert not PATTERN.search(b'~/pdk and $PDK_ROOT/sky130A')


def test_no_home_directory_paths_in_tracked_files():
    offenders = []
    for rel in _tracked():
        if _exempt(rel):
            continue
        path = REPO / rel
        if not path.is_file() or path.is_symlink():
            continue
        if PATTERN.search(path.read_bytes()):
            offenders.append(rel)
    assert not offenders, (
        'absolute home-directory paths in tracked files (use $PDK_ROOT, '
        '$SILK_NGSPICE_BIN, $SILK_XYCE_BIN or ~): ' + ', '.join(offenders))
