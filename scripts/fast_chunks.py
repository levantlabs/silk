"""The fast-suite chunks (TASK-444): one table, one partition check.

A single-process run of the whole fast suite can be killed for memory on the
shared machine, so agents and the release gate run it in four chunks. Each
chunk is a list of globs relative to ``tests/``; a glob may match files or
whole directories. Sizes were balanced by ``pytest --collect-only`` counts, so
re-balance the table when one chunk drifts above about a third of the suite.

    .venv/bin/python scripts/fast_chunks.py 1        # print chunk 1's paths
    .venv/bin/python -m pytest -q $(.venv/bin/python scripts/fast_chunks.py 1)

``tests/test_fast_chunks.py`` fails when a test file falls outside every chunk
or inside two, so a new directory or file prefix cannot go unrun.
"""
from __future__ import annotations

from pathlib import Path
import sys

CHUNKS = (
    ('metrics/test_[a-e]*.py', 'metrics/test_[f-n]*.py',
     'test_[a-c]*.py', 'test_[d-f]*.py'),
    ('metrics/test_[o-s]*.py', 'metrics/test_[t-z]*.py', 'test_[g-m]*.py',
     'circuit', 'constraints', 'coverage', 'faults', 'layout'),
    ('test_[n-p]*.py', 'test_[q-r]*.py', 'db', 'lint', 'refactor', 'specs'),
    ('test_[s-z]*.py',),
)


def chunk_paths(root, chunks=CHUNKS):
    """Return, per chunk, the sorted tests/-relative paths its globs match."""
    tests = Path(root) / 'tests'
    out = []
    for patterns in chunks:
        found = set()
        for pattern in patterns:
            found.update(p.relative_to(root).as_posix()
                         for p in tests.glob(pattern))
        out.append(sorted(found))
    return out


def test_files(root):
    """Every Python test file pytest would collect from ``tests/``."""
    tests = Path(root) / 'tests'
    files = set(tests.rglob('test_*.py')) | set(tests.rglob('*_test.py'))
    return {p.relative_to(root).as_posix() for p in files}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    paths = chunk_paths(root)
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() \
            or not 1 <= int(sys.argv[1]) <= len(paths):
        sys.exit(f'usage: fast_chunks.py <1..{len(paths)}>')
    print(' '.join(paths[int(sys.argv[1]) - 1]))

# tests/test_sampling_options.py is registered by chunk 4's test_[s-z]*.py glob.
# tests/metrics/test_input_guards_517.py is registered by chunk 1's test_[f-n]*.py glob.
# tests/test_docstring_examples.py is registered by chunk 1's test_[d-f]*.py glob.
# tests/test_js_references.py is registered by chunk 2's test_[g-m]*.py glob.

# TASK-549 node-path and selector tests are covered by chunk 4 test_[s-z]*.py.
# TASK-598.2 SG13G2 update regressions use chunk 4's test_[s-z]*.py glob.
# TASK-598.1 GF180MCU update tests are covered by chunk 4 test_[s-z]*.py.

# tests/test_codegen_contract_doc_examples.py is registered by chunk 1 test_[a-c]*.py.
# tests/test_metric_example_{adc,bindings,spectral,sweep,timing,transforms}.py
# are registered by chunk 2's test_[g-m]*.py glob.
