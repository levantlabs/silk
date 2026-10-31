"""Catch JavaScript tests/helpers left outside the pytest reference chain."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def unreferenced_modules(root):
    tests = root / 'tests'
    modules = set(tests.joinpath('js').rglob('*.mjs'))
    modules.update(tests.joinpath('browser').rglob('*.mjs'))
    sources = set(tests.rglob('test_*.py')) | set(tests.rglob('*_test.py'))
    sources.update(tests.rglob('*.mjs'))
    sources.update(tests.rglob('*.html'))
    # Installed dependencies are neither our test modules nor their wrappers.
    modules = {path for path in modules if 'node_modules' not in path.parts}
    sources = {path for path in sources if 'node_modules' not in path.parts}
    # This guard's own diagnostic/fixture strings must not register a script.
    sources.discard(Path(__file__).resolve())
    text = {path: path.read_text(encoding='utf-8') for path in sources}
    missing = []
    for module in sorted(modules):
        # Wrappers often construct paths from directory + basename. Require a
        # whole filename, so "old_example.mjs" cannot register "example.mjs".
        reference = re.compile(r'(?<![\w.-])' + re.escape(module.name) + r'(?![\w.-])')
        if not any(path != module and reference.search(content)
                   for path, content in text.items()):
            missing.append(module.relative_to(root).as_posix())
    return missing


def test_javascript_modules_are_referenced():
    missing = unreferenced_modules(ROOT)
    assert not missing, (
        'JavaScript modules have no reference from a pytest file or another '
        '.mjs/.html module; wire them into a test or remove them:\n'
        + '\n'.join(missing)
    )


def test_reference_guard_accepts_wrappers_and_helpers_but_not_self_references(tmp_path):
    tests = tmp_path / 'tests'
    (tests / 'js').mkdir(parents=True)
    (tests / 'browser').mkdir()
    (tests / 'test_wrapper.py').write_text("script = 'entry.mjs'\n")
    (tests / 'js/entry.mjs').write_text("import './helper.mjs';\n")
    (tests / 'js/helper.mjs').write_text('export const value = 1;\n')
    (tests / 'browser/index.html').write_text('<script src="page.mjs"></script>')
    (tests / 'browser/page.mjs').write_text('')
    (tests / 'js/orphan.mjs').write_text("// orphan.mjs names itself\n")
    (tests / 'test_unrelated.py').write_text("script = 'old_orphan.mjs'\n")
    assert unreferenced_modules(tmp_path) == ['tests/js/orphan.mjs']


def test_reference_guard_ignores_installed_packages(tmp_path):
    tests = tmp_path / 'tests'
    (tests / 'js').mkdir(parents=True)
    # Cover both candidate trees and sources outside those trees, including
    # nested installations. Dependency references must not hide real orphans.
    for index, directory in enumerate(('browser/node_modules',
                                       'js/nested/node_modules', 'node_modules')):
        installed = tests / directory
        installed.mkdir(parents=True)
        for suffix in ('mjs', 'html', 'py'):
            orphan = f'orphan_{index}_{suffix}.mjs'
            (tests / 'js' / orphan).write_text('')
            name = {'mjs': 'x.mjs', 'html': 'index.html', 'py': 'test_vendor.py'}[suffix]
            (installed / name).write_text(f'"{orphan}"')
    assert unreferenced_modules(tmp_path) == [
        f'tests/js/orphan_{index}_{suffix}.mjs'
        for index in range(3) for suffix in ('html', 'mjs', 'py')
    ]
