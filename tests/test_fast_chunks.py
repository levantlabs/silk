"""The fast-suite chunks partition what pytest collects (TASK-444)."""
from pathlib import Path
import subprocess
import sys

from scripts.fast_chunks import CHUNKS, chunk_paths

ROOT = Path(__file__).resolve().parents[1]


def _collected_files():
    """Test file -> collected test count, from a real ``--collect-only`` run."""
    out = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests', '--collect-only', '-q',
         '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    counts = {}
    for line in out.splitlines():
        if '::' in line:
            path = line.split('::')[0]
            counts[path] = counts.get(path, 0) + 1
    return counts


def _chunk_of(path, chunks):
    return [i for i, paths in enumerate(chunks)
            if any(path == p or path.startswith(p + '/') for p in paths)]


def test_chunks_partition_the_collected_fast_suite():
    counts = _collected_files()
    chunks = chunk_paths(ROOT)
    assert len(chunks) == len(CHUNKS)
    sizes = [0] * len(chunks)
    for path, n in counts.items():
        owners = _chunk_of(path, chunks)
        assert len(owners) == 1, (
            f'{path} is in chunks {[o + 1 for o in owners]}; each test file '
            'must be in exactly one: edit CHUNKS in scripts/fast_chunks.py')
        sizes[owners[0]] += n
    total = sum(counts.values())
    assert sum(sizes) == total
    for number, size in enumerate(sizes, 1):
        assert size > 0, f'chunk {number} collects no tests'
        assert size <= total * 0.4, (
            f'chunk {number} holds {size} of {total} tests (limit 40%); re-balance CHUNKS')
