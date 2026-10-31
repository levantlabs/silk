"""The generated API reference builds clean and covers the manifest.

TASK-305.1 (A1). The pure parts (what the manifest expects, what is left out
on purpose) run everywhere; the end-to-end build needs the ``docs`` extra and
skips without Sphinx. CI's docs job installs the extra and runs the build and
``scripts/api_reference.py`` directly.

Fast suite: no PDK models, no simulator.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import api_reference as ref

REPO = Path(__file__).resolve().parent.parent


def test_catalog_generated_classes_are_not_rows():
    rendered, skipped = ref.build_rows()
    assert not [e for e in rendered if e.kind in ref.NOT_RENDERED]
    assert sum(e.kind == 'factory' for e in skipped) == 36


def test_every_rendered_row_has_an_expected_object_name():
    rendered, _ = ref.build_rows()
    expected = ref.expected_objects()
    assert len(expected) == len(rendered)
    names = [n for n, _ in expected]
    assert len(set(names)) == len(names), 'two rows share one documented name'
    assert 'silk.circuit.Module.add_net' in names


def test_pages_cover_every_package_and_are_idempotent(tmp_path):
    names = ref.write_pages(tmp_path)
    assert 'silk_circuit' in names and 'silk_metrics' in names
    stamp = {p.name: p.stat().st_mtime_ns for p in tmp_path.glob('*.md')}
    ref.write_pages(tmp_path)
    assert stamp == {p.name: p.stat().st_mtime_ns for p in tmp_path.glob('*.md')}


BANNED_CONF_NAMES = {'nitpick_ignore', 'nitpick_ignore_regex',
                     'suppress_warnings', 'autodoc_mock_imports'}


def _banned_assignments(source):
    """Banned config names assigned in *source*, plain or annotated."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        found |= {t.id for t in targets if isinstance(t, ast.Name)} & BANNED_CONF_NAMES
    return found


def test_conf_has_no_blanket_suppressions():
    conf = (REPO / 'docs' / 'reference' / 'conf.py').read_text()
    assert _banned_assignments(conf) == set()


def test_intersphinx_is_online_and_bounded():
    conf = (REPO / 'docs' / 'reference' / 'conf.py').read_text()
    tree = ast.parse(conf)
    assigned = {t.id: n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert isinstance(assigned['intersphinx_timeout'], ast.Constant)
    assert 0 < assigned['intersphinx_timeout'].value <= 60
    assert 'External dependency' in ast.get_docstring(tree)


@pytest.mark.parametrize('line', [
    "nitpick_ignore = [('py:class', 'x')]",
    "suppress_warnings: list = ['ref.class']",
    "autodoc_mock_imports += ['silk']",
])
def test_suppression_ban_catches_every_assignment_form(line):
    assert _banned_assignments(line) != set()


def test_instance_attributes_are_rows_not_exclusions():
    expected = dict(ref.expected_objects())
    for name in ('silk.signals.Signal.x', 'silk.signals.Signal.ylabel',
                 'silk.circuit.Module.results_file'):
        assert expected[name] == 'attribute'
    _, skipped = ref.build_rows()
    assert not [e for e in skipped if e.kind == 'attribute']


def test_a_manifest_attribute_missing_from_the_build_fails_the_check(tmp_path):
    names = {n for n, _ in ref.expected_objects()}
    present = names - {'silk.signals.Signal.x'}
    inv = tmp_path / 'objects.inv'
    inv.write_bytes(_inventory(present))
    assert ref.missing_from_build(tmp_path) == ['silk.signals.Signal.x']


def test_counts_report_rendered_and_excluded_separately():
    counts = ref.row_counts()
    assert counts['excluded'] == {'factory': 36, 'alias': 4}
    assert counts['rendered'] == len(ref.expected_objects())


def _inventory(names):
    import zlib
    body = '\n'.join(f'{n} py:data 1 x.html#$ -' for n in sorted(names))
    head = (b'# Sphinx inventory version 2\n# Project: t\n# Version: 1\n'
            b'# The remainder of this file is compressed using zlib.\n')
    return head + zlib.compress(body.encode())


def test_reference_builds_with_no_warnings_and_every_row(tmp_path):
    pytest.importorskip('sphinx')
    pytest.importorskip('myst_parser')
    out = tmp_path / 'html'
    run = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-E', '-n', '-W', '--keep-going', '-q',
         '-b', 'html', '-d', str(tmp_path / 'doctrees'),
         str(REPO / 'docs' / 'reference'), str(out)],
        capture_output=True, text=True, cwd=REPO, timeout=900)
    assert run.returncode == 0, run.stderr[-3000:]
    assert ref.missing_from_build(out) == []
