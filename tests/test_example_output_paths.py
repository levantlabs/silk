"""``silk/examples/`` must never write beside its own source.

``pyproject.toml`` has ``include = ["silk*"]``, so the examples ship in the
wheel. An example that builds its output directory from
``os.path.dirname(__file__)`` therefore writes inside ``site-packages``.
Measured over a full sweep on a real installed layout (2026-09-19,
TASK-243): the installed ``silk/`` tree went from 9,283,142 to
1,174,210,201 bytes — 126x, one example alone contributing 1.11 GB — with
pip keeping no record of any of it, so ``pip uninstall`` left it behind. On
a read-only install the same line is an outright crash.

The fix is one helper, ``silk.examples.example_output_dir``. These guards
stop the old idiom coming back in a new example, and stop the helper's
default drifting back inside the package.

Two idioms are deliberately NOT routed through the helper and are asserted
to stay as they are:

* the relative ``dir_folder='testSpice'`` examples — resolved against the
  working directory, they already write outside the package;
* context-managed ``TemporaryDirectory`` scratch directories, removed
  automatically when their context exits.
"""
import ast
import glob
import os
import re
import subprocess
import sys

import pytest

from silk.examples import (DEFAULT_OUTPUT_DIRNAME, ExampleOutputError,
                           SILK_EXAMPLES_OUT_ENV, example_output_dir)

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_EXAMPLES = os.path.join(_REPO, 'silk', 'examples')


def _example_files():
    """Every example file, flat and in topic subpackages — the same set the
    other structural guards use (tests/test_examples_importable.py)."""
    files = sorted(glob.glob(os.path.join(_EXAMPLES, '*.py'))
                   + [f for f in glob.glob(os.path.join(_EXAMPLES, '*', '*.py'))
                      if os.path.exists(os.path.join(os.path.dirname(f),
                                                     '__init__.py'))])
    files = [f for f in files if os.path.basename(f) != '__init__.py']
    assert files, f'no examples found under {_EXAMPLES!r} — glob is broken'
    return files


# --------------------------------------------------------------------------- #
# The idiom that caused it
# --------------------------------------------------------------------------- #
def _file_derived_paths(path):
    """Lines where a ``__file__``-derived directory is joined with anything.

    Catches both spellings the sweep found — ``os.path.join(_HERE, 'output')``
    and ``Path(__file__).resolve().parent / 'output'`` — by looking for the
    join itself rather than for the literal ``'output'``, so a new example
    picking a different subdirectory name is caught too.
    """
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)

    # Names bound to a __file__-derived directory anywhere in the module.
    derived = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            seg = ast.get_source_segment(src, node.value) or ''
            if '__file__' in seg:
                derived.add(node.targets[0].id)

    offenders = []
    for node in ast.walk(tree):
        # os.path.join(<derived or __file__ expr>, ...)
        if isinstance(node, ast.Call):
            func = ast.get_source_segment(src, node.func) or ''
            if func.endswith('path.join') and node.args:
                first = ast.get_source_segment(src, node.args[0]) or ''
                if '__file__' in first or first.strip() in derived:
                    offenders.append(node.lineno)
        # Path(__file__)... / 'something'
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = ast.get_source_segment(src, node.left) or ''
            if '__file__' in left or left.strip() in derived:
                offenders.append(node.lineno)
    return sorted(set(offenders))


# ``_HERE`` legitimately locates data that SHIPS with the package — read-only
# reads, not writes. Each exemption names the file and what it reads.
_READ_ONLY_FILE_JOINS = {}


def test_no_example_derives_an_output_path_from_its_own_location():
    """The regression itself. A new example must call
    ``example_output_dir()``, never build a path from ``__file__``."""
    offenders = []
    for path in _example_files():
        name = os.path.basename(path)
        lines = _file_derived_paths(path)
        if lines and name not in _READ_ONLY_FILE_JOINS:
            offenders.append(f'{name}:{",".join(map(str, lines))}')
    assert offenders == [], (
        'these build a path from their own file location, which for an '
        'installed user is inside site-packages; use '
        'silk.examples.example_output_dir() instead:\n  '
        + '\n  '.join(offenders))


def test_the_read_only_exemptions_are_still_read_only():
    """The exemption list above is not a general escape hatch: every file on
    it must still be free of an ``output``-flavoured join, and must not have
    quietly lost its ``__file__`` use (a stale exemption hides the next
    offender behind a name that no longer needs it)."""
    stale, writing = [], []
    for name in sorted(_READ_ONLY_FILE_JOINS):
        path = os.path.join(_EXAMPLES, name)
        assert os.path.exists(path), f'exemption names a missing file: {name}'
        src = open(path, encoding='utf-8').read()
        if not _file_derived_paths(path):
            stale.append(name)
        if re.search(r"__file__|_HERE|_REPO", src) and re.search(
                r"(?:_HERE|_REPO|__file__)[^\n]*['\"]output['\"]", src):
            writing.append(name)
    assert writing == [], (
        'an exempted file now joins its own location with an output '
        f'directory: {writing}')
    assert stale == [], (
        'these no longer derive any path from __file__ — drop them from '
        f'_READ_ONLY_FILE_JOINS: {stale}')


def test_every_example_that_writes_uses_the_helper():
    """An example that names an output directory must reach it through the
    helper — or be one of the two idioms that are already correct."""
    offenders = []
    for path in _example_files():
        src = open(path, encoding='utf-8').read()
        if 'example_output_dir' in src or 'TemporaryDirectory' in src:
            continue
        # The relative-CWD idiom: dir_folder='testSpice' and friends.
        if re.search(r"dir_folder\s*=\s*['\"][^/'\"]+['\"]", src):
            continue
        if re.search(r"out_?dir|OUT_DIR|output_dir", src):
            offenders.append(os.path.basename(path))
    assert offenders == [], (
        'these name an output directory without going through '
        f'example_output_dir(), TemporaryDirectory or a relative dir_folder: {offenders}')


def test_metrics_examples_use_their_declared_output_dirs():
    """Metrics results are retained under the helper, except managed scratch."""
    metrics = sorted(glob.glob(os.path.join(_EXAMPLES, 'metrics', '*.py')))
    metrics = [f for f in metrics if os.path.basename(f) != '__init__.py']
    assert metrics, 'silk/examples/metrics/ is empty — glob is broken'
    missing = []
    for path in metrics:
        name = os.path.basename(path)
        expected = ('tempfile.TemporaryDirectory(' if name == 'ring_osc_jitter.py'
                    else 'example_output_dir(')
        if expected not in open(path, encoding='utf-8').read():
            missing.append(name)
    assert missing == [], missing


# --------------------------------------------------------------------------- #
# The helper's contract
# --------------------------------------------------------------------------- #
def test_default_is_under_the_cwd_and_never_inside_the_package(tmp_path,
                                                               monkeypatch):
    monkeypatch.delenv(SILK_EXAMPLES_OUT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    got = example_output_dir(create=False)
    assert got == os.path.join(str(tmp_path), DEFAULT_OUTPUT_DIRNAME)
    assert os.path.dirname(os.path.dirname(__file__)) not in got
    assert 'site-packages' not in got


def test_default_dirname_cannot_shadow_an_importable_package():
    """Hyphenated on purpose: a ``silk_output/`` or ``output/`` left in a
    user's working directory is importable and can shadow a real module."""
    assert not DEFAULT_OUTPUT_DIRNAME.isidentifier()


def test_environment_variable_wins(tmp_path, monkeypatch):
    target = tmp_path / 'elsewhere'
    monkeypatch.setenv(SILK_EXAMPLES_OUT_ENV, str(target))
    monkeypatch.chdir(tmp_path)
    assert example_output_dir() == str(target)
    assert target.is_dir()


def test_parts_are_appended(tmp_path, monkeypatch):
    monkeypatch.setenv(SILK_EXAMPLES_OUT_ENV, str(tmp_path))
    assert example_output_dir('design_loop', 'work', create=False) == \
        os.path.join(str(tmp_path), 'design_loop', 'work')


def test_create_false_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(SILK_EXAMPLES_OUT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    example_output_dir('sub', create=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores the write bit')
def test_an_unwritable_directory_is_diagnosed_here_not_deep_inside_silk(
        tmp_path, monkeypatch):
    """AC #2. The traceback must name the output directory and the variable
    that moves it, not bottom out in silk/circuit/modules.py."""
    ro = tmp_path / 'ro'
    ro.mkdir()
    ro.chmod(0o555)
    try:
        monkeypatch.setenv(SILK_EXAMPLES_OUT_ENV, str(ro))
        with pytest.raises(ExampleOutputError) as exc:
            example_output_dir()
        assert str(ro) in str(exc.value)
        assert SILK_EXAMPLES_OUT_ENV in str(exc.value)
    finally:
        ro.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason='root ignores the write bit')
def test_an_uncreatable_directory_is_diagnosed_here_too(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    parent.mkdir()
    parent.chmod(0o555)
    try:
        monkeypatch.setenv(SILK_EXAMPLES_OUT_ENV, str(parent / 'child'))
        with pytest.raises(ExampleOutputError) as exc:
            example_output_dir()
        assert 'child' in str(exc.value)
        assert SILK_EXAMPLES_OUT_ENV in str(exc.value)
    finally:
        parent.chmod(0o755)


def test_a_pdk_free_example_writes_under_the_cwd_and_not_into_the_package(
        tmp_path):
    """End to end, in a subprocess from a foreign working directory: the
    package tree must be byte-for-byte the same size afterwards."""
    pkg = os.path.join(_REPO, 'silk')

    def _size():
        return sum(os.path.getsize(os.path.join(d, f))
                   for d, _, fs in os.walk(pkg) for f in fs
                   if not f.endswith(('.pyc', '.pyo')))

    before = _size()
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONPATH=_REPO)
    env.pop(SILK_EXAMPLES_OUT_ENV, None)
    cp = subprocess.run(
        [sys.executable, '-m', 'silk.examples.testbench_codegen_example'],
        cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert cp.returncode == 0, cp.stderr[-2000:]
    assert (tmp_path / DEFAULT_OUTPUT_DIRNAME).is_dir(), (
        f'nothing appeared under {tmp_path}/{DEFAULT_OUTPUT_DIRNAME}')
    assert _size() == before, 'the example wrote into the silk package tree'


def test_no_example_hard_codes_a_tmp_output_directory():
    """Shared /tmp paths let unrelated example runs overwrite SPICE results."""
    offenders = []
    for path in _example_files():
        with open(path, encoding='utf-8') as source:
            tree = ast.parse(source.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value == '/tmp' or node.value.startswith('/tmp/'):
                    offenders.append(f'{os.path.basename(path)}:{node.lineno}')
    assert offenders == [], (
        'hard-coded temporary paths bypass SILK_EXAMPLES_OUT; use '
        'example_output_dir(<example name>) instead:\n  '
        + '\n  '.join(offenders))


@pytest.mark.parametrize(('name', 'factory'), [
    ('bandgap_example', 'build_bandgap'),
    ('ldo_regulator_example', 'build_ldo'),
    ('current_mirror_example', 'build_current_mirror'),
    ('source_follower_example', 'build_source_follower'),
    ('ota_5t_guide', 'build_5t_ota'),
    ('two_stage_miller_ota_example', 'build_two_stage_ota'),
    ('constraints_external_load', 'build_shell'),
    ('constraints_diff_pair', 'build_diff_pair'),
    ('constraints_5t_ota', 'build_5t_ota'),
    ('constraints_current_mirror', 'build_current_mirror'),
])
def test_example_netlists_honour_the_output_root(name, factory, tmp_path,
                                               monkeypatch):
    """Building a circuit must place its SPICE work in its own example folder."""
    import importlib

    monkeypatch.setenv(SILK_EXAMPLES_OUT_ENV, str(tmp_path))
    module = importlib.import_module(f'silk.examples.{name}')
    dut = getattr(module, factory)()
    expected = tmp_path / name
    assert dut.dir_folder == str(expected)
    assert expected.is_dir()
    assert os.path.dirname(dut.netlistname) == ''
    assert os.path.dirname(os.path.join(dut.dir_folder, dut.netlistname)) == str(expected)
    for bench in dut._tests.values():
        assert os.path.commonpath(
            [str(expected), os.path.join(bench.netlist.dir_folder, bench.netlist.netlistname)]) == str(expected)


def _unmanaged_temp_calls(source):
    """Reject unmanaged allocation and temp-root paths, including import aliases."""
    tree = ast.parse(source)
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'tempfile':
                    names[alias.asname or alias.name] = 'tempfile'
        elif isinstance(node, ast.ImportFrom) and node.module == 'tempfile':
            for alias in node.names:
                names[alias.asname or alias.name] = 'tempfile.' + alias.name
    managed = {id(item.context_expr) for node in ast.walk(tree)
               if isinstance(node, ast.With) for item in node.items}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        root, dot, rest = name.partition('.')
        name = names.get(root, root) + dot + rest
        if name in {'tempfile.mkdtemp', 'tempfile.mktemp',
                    'tempfile.gettempdir', 'tempfile.gettempdirb'} or (
                name == 'tempfile.TemporaryDirectory' and (
                    id(node) not in managed or any(
                        kw.arg == 'delete' and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False for kw in node.keywords))):
            offenders.append(node.lineno)
    return offenders


def test_no_example_leaves_unmanaged_temporary_output():
    offenders = []
    for path in _example_files():
        with open(path, encoding='utf-8') as source:
            lines = _unmanaged_temp_calls(source.read())
        offenders.extend(f'{os.path.relpath(path, _EXAMPLES)}:{line}'
                         for line in lines)
    assert offenders == [], (
        'unmanaged temporary output bypasses SILK_EXAMPLES_OUT; use '
        'example_output_dir(<example name>) instead:\n  '
        + '\n  '.join(offenders))


@pytest.mark.parametrize('source', [
    'import tempfile; out = tempfile.mkdtemp()',
    'import tempfile as tmp; out = tmp.mkdtemp()',
    'from tempfile import mkdtemp as alloc; out = alloc()',
    'from tempfile import mktemp; out = mktemp()',
    'import tempfile; out = tempfile.gettempdir() + "/example"',
    'import tempfile; out = tempfile.TemporaryDirectory().name',
    'from tempfile import TemporaryDirectory\n'
    'with TemporaryDirectory(delete=False) as out:\n    pass',
])
def test_unmanaged_temp_guard_detects_allocations(source):
    assert len(_unmanaged_temp_calls(source)) == 1


def test_unmanaged_temp_guard_allows_context_managed_scratch():
    assert _unmanaged_temp_calls(
        'from tempfile import TemporaryDirectory as Scratch\n'
        'with Scratch() as out:\n    pass\n') == []
