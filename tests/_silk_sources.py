"""The Python files under ``silk/`` that a source-walking guard should read.

``silk/examples/output/`` is gitignored and example runs write generated
``.py`` files into it. A guard that globs ``silk/`` and parametrizes per file
would collect a different number of tests depending on what earlier runs left
on disk, so every source walk goes through here instead of a bare ``rglob``.
Generated output is not source: it is excluded by path, not by git state, so
the result is the same in a fresh checkout, an archive, and a used tree.
"""
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SILK = _REPO / 'silk'
_GENERATED = _SILK / 'examples' / 'output'


def silk_py_files(root=_SILK):
    """Sorted ``*.py`` under *root*, minus generated example output and caches."""
    return sorted(p for p in Path(root).rglob('*.py')
                  if _GENERATED not in p.parents and '__pycache__' not in p.parts)
