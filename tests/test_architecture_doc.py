"""Currency guards on ``docs/architecture.md`` (TASK-19).

The architecture document is only useful while it describes what *is*. Three
properties are pinned here so it cannot drift silently:

* **Package map is complete and not stale.** Every top-level ``silk/*``
  subpackage (a directory with ``__init__.py``) and every top-level module
  (``silk/*.py``) has a row in the package map, and every row names something
  that still exists.
* **Every path it cites resolves.** Any backticked ``silk/…``, ``tests/…``,
  ``scripts/…`` or ``docs/…`` reference (with or without a ``:line`` suffix)
  points at a real file, so a rename or deletion fails here instead of leaving
  a dead pointer for the next maintainer.
* **The two front doors cross-link.** ``docs/agent-guide.md`` (usage) points at
  ``docs/architecture.md`` (internals) and vice versa, and the doc states its
  own same-PR maintenance rule.

Fast suite, no PDK or simulator, so CI runs it on every push.
"""
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SILK = _REPO / 'silk'
_DOC = _REPO / 'docs' / 'architecture.md'
_AGENT_GUIDE = _REPO / 'docs' / 'agent-guide.md'


def _doc_text():
    assert _DOC.is_file(), f'{_DOC} is missing'
    return _DOC.read_text()


def _package_map_section(text):
    """The text between the package-map heading and the next top-level heading."""
    match = re.search(r'^## 1\..*?(?=^## 2\.)', text, flags=re.MULTILINE | re.DOTALL)
    assert match, 'docs/architecture.md must have "## 1. …" (package map) followed by "## 2. …"'
    return match.group(0)


def _subpackages():
    names = sorted(p.name for p in _SILK.iterdir()
                   if p.is_dir() and (p / '__init__.py').is_file())
    assert names, f'no subpackages found under {_SILK}'
    return names


def _top_level_modules():
    """Every ``silk/*.py`` including ``__init__`` (the façade deserves a row too)."""
    return sorted(p.stem for p in _SILK.glob('*.py'))


def _map_entries(section):
    """Package-map rows are keyed by a backticked ``silk/<name>/`` or ``silk/<name>.py``."""
    return set(re.findall(r'`silk/([A-Za-z_][A-Za-z0-9_]*)/`', section)), \
        set(re.findall(r'`silk/([A-Za-z_][A-Za-z0-9_]*)\.py`', section))


def test_every_subpackage_has_a_package_map_row():
    dirs, _ = _map_entries(_package_map_section(_doc_text()))
    missing = [n for n in _subpackages() if n not in dirs]
    assert not missing, (
        f'silk subpackage(s) with no row in docs/architecture.md §1 package map: '
        f'{missing}. Add a `silk/<name>/` row (responsibility, entry points, '
        f'must-not) in the same PR that adds the package.')


def test_every_top_level_module_has_a_package_map_row():
    _, mods = _map_entries(_package_map_section(_doc_text()))
    missing = [n for n in _top_level_modules() if n not in mods]
    assert not missing, (
        f'top-level silk module(s) with no row in docs/architecture.md §1: {missing}')


def test_package_map_has_no_stale_rows():
    dirs, mods = _map_entries(_package_map_section(_doc_text()))
    stale_dirs = sorted(dirs - set(_subpackages()))
    stale_mods = sorted(mods - set(_top_level_modules()))
    assert not stale_dirs and not stale_mods, (
        f'docs/architecture.md §1 names packages that no longer exist: '
        f'dirs={stale_dirs} modules={stale_mods}')


_PATH_REF = re.compile(
    r'`((?:silk|tests|scripts|docs)/[A-Za-z0-9_./\-]+?)(?::\d+(?:-\d+)?)?`')


def test_every_cited_repo_path_exists():
    text = _doc_text()
    refs = sorted(set(_PATH_REF.findall(text)))
    assert refs, 'docs/architecture.md cites no repo paths — the regex or the doc is broken'
    dead = [r for r in refs if not (_REPO / r).exists()]
    assert not dead, (
        f'docs/architecture.md cites path(s) that do not exist: {dead}. '
        f'Update the doc in the same PR as the rename/removal.')


def test_doc_states_its_maintenance_rule():
    text = _doc_text()
    assert re.search(r'same PR', text), (
        'docs/architecture.md must state that it is updated in the same PR as '
        'any change that alters a responsibility or invariant')


def test_front_doors_cross_link():
    assert 'architecture.md' in _AGENT_GUIDE.read_text(), (
        'docs/agent-guide.md must point maintainers at docs/architecture.md')
    assert 'agent-guide.md' in _doc_text(), (
        'docs/architecture.md must point users at docs/agent-guide.md')
