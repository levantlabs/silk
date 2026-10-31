"""Structural guards on ``silk/examples/``.

``pyproject.toml`` has ``include = ["silk*"]``, so **``silk.examples`` ships in
the wheel** and everything here reaches installed users. Three properties were
broken before the 0.2.0 release audit and are pinned here so they cannot
regress:

* **A1** — examples imported their siblings by bare module name
  (``import ldo_regulator_example``), which only resolves because running a
  file as a script puts its own directory on ``sys.path[0]``. From an
  installed package, ``from silk.examples import pll_example`` raised
  ``ModuleNotFoundError``.
* **A2** — 19 of them inserted the repo root into ``sys.path``, a
  source-checkout hack that, inside a wheel, prepends an unrelated directory
  two levels above ``site-packages/silk/examples``.
* **A4** — 7 had no ``__main__`` guard, so *importing* one ran it.
  ``import silk.examples.source_examples`` spawned ngspice and had not
  finished after 25 seconds.

The first three tests are static and cost nothing. The last actually imports
every one of them with a fake simulator on ``PATH``, which is the only way to
catch a sim-on-import that a source scan would miss.

All four cover the topic subpackages under ``silk/examples/`` as well as the
flat files — ``silk/examples/metrics/`` (the metric-coverage suite, TASK-187)
ships in the wheel the same way, so a subdirectory must not be a way round
these rules.
"""
import ast
import glob
import os
import subprocess
import sys
import textwrap

import pytest

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_EXAMPLES = os.path.join(_REPO, 'silk', 'examples')


def _example_files():
    """Every example file, including the ones in topic subpackages.

    ``silk/examples/metrics/`` (the metric-coverage suite, TASK-187) ships in
    the wheel exactly like the flat examples do, so it inherits all three
    guarantees below; a subdirectory must not be a way round them. Only
    directories that are real packages count — ``silk/examples/output/`` is
    the gitignored run-output directory and holds generated scripts.
    """
    files = sorted(glob.glob(os.path.join(_EXAMPLES, '*.py'))
                   + [f for f in glob.glob(os.path.join(_EXAMPLES, '*', '*.py'))
                      if os.path.exists(os.path.join(os.path.dirname(f),
                                                     '__init__.py'))])
    files = [f for f in files if os.path.basename(f) != '__init__.py']
    assert files, f'no examples found under {_EXAMPLES!r} — glob is broken'
    return files


def _module_stems():
    return {os.path.basename(f)[:-3] for f in _example_files()}


def test_no_example_imports_a_sibling_by_bare_module_name():
    """A1. Sibling imports must be package-relative, or an installed user
    cannot import the PLL/SerDes/SAR/LDO chain at all."""
    stems, offenders = _module_stems(), []
    for path in _example_files():
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in stems:
                        offenders.append(f'{os.path.basename(path)}:{node.lineno} '
                                         f'import {alias.name}')
            elif (isinstance(node, ast.ImportFrom) and node.level == 0
                  and node.module and node.module.split('.')[0] in stems):
                offenders.append(f'{os.path.basename(path)}:{node.lineno} '
                                 f'from {node.module}')
    assert offenders == [], (
        'these resolve only when the file is run as a script from '
        'silk/examples/; use silk.examples.<module>:\n  '
        + '\n  '.join(offenders))


def test_no_example_manipulates_sys_path():
    """A2. The repo-root insertion is a checkout hack that ships in the wheel."""
    offenders = [f'{os.path.basename(f)}' for f in _example_files()
                 if 'sys.path' in open(f).read()]
    assert offenders == [], offenders


def test_every_example_has_a_main_guard():
    """A4. Without one, importing the module runs it — and one of them ran a
    simulator. Checked as a real ``if __name__ == '__main__'`` comparison in
    the AST, not a substring, so a mention in a docstring does not count."""
    offenders = []
    for path in _example_files():
        tree = ast.parse(open(path).read())
        guarded = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == '__name__'
            and any(isinstance(c, ast.Constant) and c.value == '__main__'
                    for c in node.test.comparators)
            for node in tree.body)
        if not guarded:
            offenders.append(os.path.basename(path))
    assert offenders == [], offenders


def test_net_current_persistence_entrypoint_does_not_mask_failures():
    """The release example may explain missing setup in documentation, but an
    arbitrary implementation/database failure must still exit non-zero."""
    path = os.path.join(_EXAMPLES, 'net_current_persistence_example.py')
    tree = ast.parse(open(path).read())
    main_guard = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == '__name__'
        and any(
            isinstance(value, ast.Constant) and value.value == '__main__'
            for value in node.test.comparators
        )
    )
    assert not any(isinstance(node, ast.Try) for node in ast.walk(main_guard))


def _pdk_available():
    """A3 is still open — 46 examples load a PDK at module level — so importing
    them needs a real one. Until that closes, this check cannot also assert
    that a PDK-free import works; it skips instead of failing."""
    try:
        from silk import load_pdk
        load_pdk('sky130')
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _pdk_available(),
                    reason='examples load a PDK at import time (audit item A3)')
def test_importing_every_example_runs_no_simulator(tmp_path):
    """A4, behaviourally. Imports every example from a foreign cwd with
    a fake ngspice/Xyce first on ``PATH``; any invocation leaves a marker.

    A static scan cannot catch this — the call is reached through
    ``Module.simulate()``, several frames from anything an example spells out.
    """
    marker = tmp_path / 'invoked.txt'
    shim = tmp_path / 'bin'
    shim.mkdir()
    for name in ('ngspice', 'Xyce'):
        exe = shim / name
        exe.write_text(f'#!/bin/sh\necho {name} >> {marker}\nexit 1\n')
        exe.chmod(0o755)

    script = textwrap.dedent('''
        import importlib, pkgutil, sys
        import silk.examples as pkg
        # walk_packages, not iter_modules: the topic subpackages
        # (silk.examples.metrics) must be imported file by file too.
        names = sorted(x.name for x in pkgutil.walk_packages(
            pkg.__path__, prefix='silk.examples.') if not x.ispkg)
        for m in names:
            try:
                importlib.import_module(m)
            except Exception as exc:
                print(f'{m}: {type(exc).__name__}: {exc}', file=sys.stderr)
                raise SystemExit(1)
    ''')
    env = dict(os.environ,
               PATH=f'{shim}{os.pathsep}' + os.environ.get('PATH', ''),
               MPLBACKEND='Agg')
    cp = subprocess.run([sys.executable, '-c', script], cwd=str(tmp_path),
                        env=env, capture_output=True, text=True, timeout=600)

    assert cp.returncode == 0, f'an example failed to import:\n{cp.stderr[-2000:]}'
    assert not marker.exists(), (
        'importing an example invoked a simulator: '
        + marker.read_text().strip().replace('\n', ', '))
