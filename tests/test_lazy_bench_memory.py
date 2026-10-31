"""Bench evaluation must retain requested vectors, not every rawfile column."""
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from silk import Module, load_pdk
from silk.simulation import Transient
from silk.circuit.circuit_elements import PortDir


def _write_raw(path, points, columns, binary):
    header = ('Title: memory regression\nDate: today\n'
              'Plotname: Transient Analysis\nFlags: real\n'
              f'No. Variables: {columns}\nNo. Points: {points}\nVariables:\n'
              '0\ttime\ttime\n' + ''.join(
                  f'{i}\tv(n{i})\tvoltage\n' for i in range(1, columns)))
    with path.open('wb') as handle:
        handle.write((header + ('Binary:\n' if binary else 'Values:\n')).encode())
        for start in range(0, points, 256):
            axis = np.arange(start, min(start + 256, points), dtype=float)
            rows = axis[:, None] + 100 * np.arange(columns)[None, :]
            if binary:
                handle.write(rows.astype('<f8').tobytes())
            else:
                for index, row in zip(range(start, start + len(rows)), rows):
                    handle.write((f'{index}\t{row[0]}\n' + ''.join(
                        f'\t{value}\n' for value in row[1:])).encode())


def _last(sig):
    return float(sig.y[-1])


def _benches(root, points):
    dut = Module('sky130', str(root), 'dut.cir', load_pdk('sky130'),
                 module_name='dut')
    dut.define_module_ports({f'n{i}': PortDir.OUT for i in range(1, 17)})
    benches = []
    for i in range(15):
        bench = dut.create_test(f'memory_{i}')
        bench.add_module_under_test({f'n{j}': f'n{j}' for j in range(1, 17)})
        bench.add_sim_type(Transient(1, points - 1))
        bench.add_metric('common', _last, sig=bench.probe('n1'))
        bench.add_metric('distinct', _last, sig=bench.probe(f'n{i + 2}'))
        benches.append(bench)
    return dut, benches


def _cached_bytes(raw):
    # Observe allocations without invoking RawVector.data and changing them.
    return sum(v._data.nbytes for p in raw.plots for v in p.vectors
               if v._data is not None)


@pytest.mark.parametrize('binary', [False, True])
def test_fifteen_evaluations_cache_only_requested_vectors(tmp_path, binary):
    points, columns = 256, 48
    path = tmp_path / 'shared.raw'
    _write_raw(path, points, columns, binary)
    dut, benches = _benches(tmp_path, points)
    reports = [bench.evaluate(rawfile=str(path)) for bench in benches]
    for index, (bench, report) in enumerate(zip(benches, reports)):
        assert report.metrics == {'common': 355., 'distinct': 455. + 100 * index}
        raw = bench.netlist._raw.native
        assert raw is bench.netlist.simResultsAll
        assert _cached_bytes(raw) == 3 * points * 8
        assert sum(v._data is not None for v in raw.vectors) == 3
    # Retaining reports and ordinary deepcopy must not materialize unread slots.
    original = benches[0].netlist._raw.native
    copied, alias = copy.deepcopy((original, original.vectors[1].data))
    assert copied.vectors[1].data is alias
    alias[0] = 123
    assert original.vectors[1].data[0] == 100
    assert _cached_bytes(copied) == 3 * points * 8
    assert dut.get_test('memory_0') is benches[0]


def _private_bytes():
    fields = Path('/proc/self/smaps_rollup').read_text().splitlines()
    return 1024 * sum(int(line.split()[1]) for line in fields
                      if line.startswith(('Private_Clean:', 'Private_Dirty:',
                                          'Private_Hugetlb:')))


def _memory_probe(root):
    import silk
    print(silk.__file__)
    points, columns = 16384, 96
    path = root / 'large.raw'
    _write_raw(path, points, columns, True)
    dut, benches = _benches(root, points)
    reports = [benches[0].evaluate(rawfile=str(path))]
    gc.collect()
    before = _private_bytes()
    reports.extend(bench.evaluate(rawfile=str(path)) for bench in benches[1:])
    gc.collect()
    growth = _private_bytes() - before
    cache = sum(_cached_bytes(b.netlist._raw.native) for b in benches)
    assert len(reports) == 15 and len(dut._tests) == 15
    print('MEMORY_RESULT=' + json.dumps(dict(growth=growth, cache=cache,
          raw_bytes=path.stat().st_size, points=points)))


@pytest.mark.skipif(not Path('/proc/self/smaps_rollup').is_file(),
                    reason='private-memory measurement requires Linux procfs')
def test_additional_benches_have_bounded_private_memory(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
               MPLBACKEND='Agg')
    result = subprocess.run([sys.executable, __file__, str(tmp_path)],
                            capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith('MEMORY_RESULT='))
    measured = json.loads(line.partition('=')[2])
    print(line)
    # 14 additional eager loads cost 168 MiB. Requested arrays cost 5.25 MiB;
    # allow interpreter/allocator metadata without letting even two full copies in.
    assert measured['growth'] < 24 * 1024 * 1024, measured
    assert measured['cache'] == 15 * 3 * 16384 * 8, measured


if __name__ == '__main__':
    _memory_probe(Path(sys.argv[1]))
