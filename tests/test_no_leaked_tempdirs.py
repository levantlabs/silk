"""TASK-453: tests must not leak temp directories into the system temp dir.

``tempfile.mkdtemp`` / ``mkstemp`` / ``NamedTemporaryFile(delete=False)`` are
never cleaned up by pytest. Use ``tmp_path`` / ``tmp_path_factory`` (via
``tests._env.scratch_dir`` from a helper) or a context-managed
``tempfile.TemporaryDirectory``.
"""
import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent

#: Standalone fixture generators (run by hand, not collected). Each one
#: unlinks its deck in a ``finally`` — they do not leak — so they are the only
#: sanctioned ``delete=False`` users.
_ALLOWED = {
    'fixtures/rawfiles/gen_a.py', 'fixtures/rawfiles/gen_b.py',
    'fixtures/rawfiles/gen_c.py', 'fixtures/rawfiles/gen_d.py',
    'fixtures/rawfiles/gen_e.py', 'fixtures/rawfiles/gen_e_meas.py',
}


def _offences(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.split('.')[-1] in ('mkdtemp', 'mkstemp'):
            yield node.lineno, name
        elif name.split('.')[-1] == 'NamedTemporaryFile':
            for kw in node.keywords:
                if (kw.arg == 'delete' and isinstance(kw.value, ast.Constant)
                        and kw.value.value is False):
                    yield node.lineno, f'{name}(delete=False)'


def test_tests_do_not_call_uncleaned_tempfile_apis():
    found = []
    for path in sorted(TESTS.rglob('*.py')):
        rel = path.relative_to(TESTS).as_posix()
        if rel in _ALLOWED or rel == Path(__file__).name:
            continue
        for line, what in _offences(ast.parse(path.read_text())):
            found.append(f'tests/{rel}:{line} {what}')
    assert not found, 'uncleaned temp-dir calls in tests: ' + '; '.join(found)
