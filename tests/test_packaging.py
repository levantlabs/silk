"""Packaging smoke tests.

The regular suite imports silk from the source tree, so a wheel that omits
subpackages or data files looks perfectly green in CI and then fails on
``pip install``. That exact failure shipped once: a hand-maintained
``packages`` list in pyproject.toml lagged the tree by ~13 subpackages and
all but one data YAML, and the installed package could not even
``import silk``. These tests exercise the *built artifact*.
"""

import os
import re
import subprocess
import sys
import tarfile
import zipfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pyproject_version():
    text = open(os.path.join(REPO_ROOT, 'pyproject.toml')).read()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, 'no version in pyproject.toml'
    return m.group(1)


def _is_runtime_data_file(rel_path):
    """True if a non-.py file under silk/ is runtime package data the wheel
    must ship, as opposed to a dev-only or example-output artifact.

    Each exclusion follows the runtime package-data globs, the packages.find
    exclude for view/assets/samples/ frontend fixtures, or a structural
    convention (superseded .xlsx source spreadsheets, docs, generated example
    output), so this stays a rule, not a second hand-maintained list.
    """
    parts = rel_path.split('/')
    if rel_path.endswith('.md'):
        return False  # dev-only docs, e.g. silk/constraints/DESIGN.md
    if rel_path.endswith('.xlsx'):
        return False  # superseded source spreadsheet; the derived .yaml ships instead
    if 'samples' in parts:
        return False  # frontend dev fixtures under view/assets/samples/
    if 'examples' in parts:
        return False  # example code ships as .py; anything else here is run output
    return True


def test_version_matches_pyproject():
    import silk
    assert silk.__version__ == _pyproject_version(), (
        'silk.__version__ and pyproject.toml disagree — bump both together')


@pytest.mark.slow
def test_wheel_is_complete_and_importable(tmp_path):
    """Build the wheel, then import silk FROM THE WHEEL in a subprocess
    whose cwd is outside the repo (so the source tree can't shadow it)."""
    build = pytest.importorskip('build', reason="the 'build' package "
                                "(dev extra) is required to build the wheel")
    del build

    dist = tmp_path / 'dist'
    subprocess.run(
        [sys.executable, '-m', 'build', '--wheel', '-o', str(dist), REPO_ROOT],
        check=True, capture_output=True, text=True)
    wheels = list(dist.glob('*.whl'))
    assert len(wheels) == 1, f'expected one wheel, got {wheels}'
    wheel = wheels[0]

    # -- static contents: every silk subpackage and runtime data file ships --
    names = set(zipfile.ZipFile(wheel).namelist())
    src_pkgs = set()
    required_data = []
    for dirpath, dirnames, filenames in os.walk(os.path.join(REPO_ROOT, 'silk')):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        if '__init__.py' in filenames:
            rel = os.path.relpath(dirpath, REPO_ROOT).replace(os.sep, '/')
            src_pkgs.add(rel + '/__init__.py')
        for filename in filenames:
            if filename.endswith(('.py', '.pyc', '.pyo')):
                continue
            rel = os.path.relpath(
                os.path.join(dirpath, filename), REPO_ROOT).replace(os.sep, '/')
            if _is_runtime_data_file(rel):
                required_data.append(rel)
    missing_pkgs = sorted(p for p in src_pkgs if p not in names)
    assert not missing_pkgs, f'wheel is missing subpackages: {missing_pkgs}'

    # Every non-.py file under silk/ is required package data UNLESS it falls
    # into one of the excluded categories below (TASK-245: a hand-maintained
    # list of 8 data files silently missed the whole waveform-explorer JS
    # bundle). The exclusions are structural, not filenames, so a new doc,
    # generated example image, dev fixture or superseded spreadsheet is
    # skipped automatically instead of requiring an edit to this test.
    assert len(required_data) >= 20, (
        'exclusion rules in _is_runtime_data_file() are filtering out '
        f'real runtime data ({len(required_data)} files found); this guard '
        'degraded to a near-empty check instead of covering the wheel')
    missing_data = sorted(d for d in required_data if d not in names)
    assert not missing_data, f'wheel is missing data files: {missing_data}'

    # -- dynamic: the installed layout actually imports and finds its data --
    extract = tmp_path / 'site'
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
         '--target', str(extract), str(wheel)],
        check=True, capture_output=True, text=True)
    probe = (
        "import silk\n"
        "assert silk.__file__.startswith(r'%s'), silk.__file__\n"
        "import silk.sources          # reads source_metadata.yaml at import\n"
        "import silk.lint, silk.metrics, silk.faults, silk.specs\n"
        "import silk.codegen, silk.db, silk.layout, silk.simulators\n"
        "from silk.pdk.registry import available\n"
        "assert 'sky130' in available(), available()\n"
        "from silk.circuit.port_roles import RoleRuleset\n"
        "RoleRuleset.default()        # reads port-roles.default.yaml\n"
        "from silk.simulators._loader import load_simulator_config\n"
        "import warnings\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('error')  # missing YAML would warn\n"
        "    assert load_simulator_config('ngspice').presets\n"
        "    load_simulator_config('xyce')\n"
        "    load_simulator_config('behavioral')\n"
        "from silk import sources\n"
        "from silk.sources.schema import export_source_schema\n"
        "from silk.sources import _bits\n"
        "assert _bits.__file__.startswith(r'%s')\n"
        "schema = export_source_schema()\n"
        "for kind, params in [('bitpattern', {'bits': '101', 'repeat': 2}),\n"
        "                     ('prbs', {'order': 7, 'length': 16, 'seed': 3})]:\n"
        "    for quantity in ['voltage', 'current']:\n"
        "        src = getattr(getattr(sources, quantity), kind)(\n"
        "            'data', p='out', n='0', v1=0, v2=1, ui=1e-6,\n"
        "            tr=1e-7, tf=2e-7, **params)\n"
        "        assert 'PWL(' in src.to_spice()\n"
        "        assert len(src.waveform.stream) == (6 if kind == 'bitpattern' else 16)\n"
        "        assert len(src.to_numpy([0, 2e-6])) == 2\n"
        "        assert schema[type(src).__name__]['waveforms'][kind]['kwarg'] == 'waveform'\n"
        "assert sources.prbs_bits(7, 8) == (0, 0, 0, 0, 0, 0, 1, 1)\n"
        "import sys\n"
        "from pathlib import Path\n"
        "package_root = Path(silk.__file__).resolve().parent\n"
        "outside = {name: mod.__file__ for name, mod in sys.modules.items()\n"
        "           if (name == 'silk' or name.startswith('silk.'))\n"
        "           and getattr(mod, '__file__', None)\n"
        "           and not Path(mod.__file__).resolve().is_relative_to(package_root)}\n"
        "assert not outside, outside\n"
        "print('WHEEL-OK')\n" % (extract, extract))
    env = dict(os.environ, PYTHONPATH=str(extract))
    result = subprocess.run(
        [sys.executable, '-c', probe],
        cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert result.returncode == 0, (
        f'import-from-wheel failed:\n{result.stdout}\n{result.stderr}')
    assert 'WHEEL-OK' in result.stdout


@pytest.mark.slow
def test_sdist_omits_repository_tests_and_keeps_package(tmp_path):
    """The source package ships the library; tests run from a repository clone."""
    pytest.importorskip('build', reason="the 'build' dev dependency is required")
    dist = tmp_path / 'sdist'
    subprocess.run(
        [sys.executable, '-m', 'build', '--sdist', '-o', str(dist), REPO_ROOT],
        check=True, capture_output=True, text=True)
    sdists = list(dist.glob('*.tar.gz'))
    assert len(sdists) == 1, f'expected one sdist, got {sdists}'
    with tarfile.open(sdists[0]) as archive:
        members = archive.getnames()
    roots = {name.split('/', 1)[0] for name in members}
    assert len(roots) == 1, f'expected one archive root, got {roots}'
    relative = {name.split('/', 1)[1] for name in members if '/' in name}
    assert not any(name == 'tests' or name.startswith('tests/') for name in relative)
    assert {'silk/__init__.py', 'README.md', 'LICENSE'} <= relative
