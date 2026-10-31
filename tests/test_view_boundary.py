"""Import-boundary enforcement for silk.signals.view (design spec §3.2, §9.3).

Every module under ``silk/signals/view/`` may import, at any scope: the
standard library, numpy, exactly ``silk.signals.rawfile`` and the shared
stdlib-only leaves ``silk._json_codec``, ``silk.spice_names`` and
``silk.vector_match``, and sibling ``silk.signals.view`` modules.
The codec is the TASK-274 package-wide JSON contract; it must stay a leaf.
No ``silk.exceptions`` or ``silk.circuit`` dependency is permitted, so a
standalone ``rawview`` needs only the viewer, rawfile reader and JSON codec.
"""
import ast
import sys
from pathlib import Path

import pytest

VIEW_DIR = Path(__file__).parent.parent / 'silk' / 'signals' / 'view'

_STDLIB = set(sys.stdlib_module_names)
_ALLOWED_EXACT = {'numpy', 'silk.signals.rawfile', 'silk._json_codec',
                  'silk.spice_names', 'silk.vector_match'}


def _view_py_files():
    return sorted(p for p in VIEW_DIR.rglob('*.py') if 'assets' not in p.parts)


def _top_module(dotted):
    return dotted.split('.')[0]


def _is_allowed(module_name, *, level=0):
    if level > 0:
        # A relative import (`from . import encode`) resolves inside the
        # package itself — always allowed.
        return True
    if module_name in _ALLOWED_EXACT:
        return True
    if module_name.startswith('silk.signals.view'):
        return True
    if _top_module(module_name) == 'numpy':
        return True
    if _top_module(module_name) in _STDLIB:
        return True
    return False


def _collect_imports(tree):
    """Yield (module_name, level, lineno) for every Import/ImportFrom node at
    any scope, including inside functions and TYPE_CHECKING blocks."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` -> module is None, level > 0.
            yield (node.module or ''), node.level, node.lineno


class TestImportBoundary:
    @pytest.mark.parametrize('path', _view_py_files(), ids=lambda p: p.name)
    def test_every_import_is_allowed(self, path):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        violations = []
        for module_name, level, lineno in _collect_imports(tree):
            if not _is_allowed(module_name, level=level):
                violations.append((path.name, lineno, module_name))
        assert not violations, (
            f'disallowed import(s) found: {violations} — silk.signals.view '
            f'may only import stdlib, numpy, silk.signals.rawfile, '
            f'silk._json_codec, and sibling silk.signals.view modules'
        )

    def test_shared_json_codec_remains_stdlib_only(self):
        # This exact exception must not become a back door to the circuit or
        # database packages. A standalone viewer can copy the leaf unchanged.
        path = VIEW_DIR.parents[1] / '_json_codec.py'
        imports = list(_collect_imports(ast.parse(path.read_text())))
        assert imports, 'The codec source must be inspected, not silently skipped'
        assert all(level == 0 and _top_module(name) in _STDLIB
                   for name, level, _ in imports)

    def test_spice_names_codec_remains_stdlib_only(self):
        path = VIEW_DIR.parents[1] / 'spice_names.py'
        imports = list(_collect_imports(ast.parse(path.read_text())))
        assert all(level == 0 and _top_module(name) in _STDLIB
                   for name, level, _ in imports)

    def test_no_silk_exceptions_or_matplotlib_in_source_text(self):
        for path in _view_py_files():
            text = path.read_text(encoding='utf-8')
            assert 'matplotlib' not in text, f'{path} references matplotlib'
            assert 'silk.exceptions' not in text, f'{path} references silk.exceptions'

    def test_view_error_is_plain_exception_not_circuit_error(self):
        from silk.signals.view.emit import ViewError
        assert ViewError.__bases__ == (Exception,)

    def test_files_discovered(self):
        # Guard against the glob silently matching nothing (a path typo would
        # make every test above vacuously pass).
        names = {p.name for p in _view_py_files()}
        assert {'emit.py', 'encode.py', 'serialize.py', 'template.py',
               'serve.py', '__init__.py', '__main__.py'} <= names


class TestVectorMatchLeaf:
    def test_allowlist_names_the_module(self):
        assert 'silk.vector_match' in _ALLOWED_EXACT

    def test_vector_match_imports_only_stdlib_and_spice_names(self):
        path = VIEW_DIR.parents[1] / 'vector_match.py'
        imports = list(_collect_imports(ast.parse(path.read_text())))
        assert imports, 'the leaf must be inspected, not silently skipped'
        for name, level, _ in imports:
            assert level == 0
            assert _top_module(name) in _STDLIB or name == 'silk.spice_names', name

    def test_view_layer_never_imports_the_reader_module(self):
        for path in _view_py_files():
            for name, _level, _ in _collect_imports(ast.parse(path.read_text())):
                assert not name.startswith('silk.signals.spice'), (path.name, name)
