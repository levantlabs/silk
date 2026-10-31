"""TASK-608: which vector name a bus bit / array-copy net reaches the explorer as,
and that the copied code reads it back.

Recorded real runs (``tests/fixtures/rawfiles/gen_e_bus.py``: ngspice 42, Xyce
7.10, the behavioral engine) of a bench with a bus ``d[1:0]`` and an arrayed
copy ``amp1[k]`` whose copies hold different values. Every producer writes
such a net wrapped (``v(...)``), so the explorer's bare-name refusal is never
met, and the viewer payload yields copyable ``tb.probe`` code for the bus bits
and the array-copy nets on the raw path and on the designer-address path.
The tests need no simulator: each bench is rebuilt and loaded with its
recorded rawfile. A refusal never counts as success.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from silk.circuit import view_node_context
from silk.circuit.view_nodes import _capture_addresses
from silk.signals.rawfile import read_rawfile
from silk.signals.view import emit_view_data
from silk.simulation.backend import RawResult
from tests.test_view_trace_identity import _dump

FIXTURES = Path(__file__).resolve().parent / 'fixtures/rawfiles'

_spec = importlib.util.spec_from_file_location('gen_e_bus', FIXTURES / 'gen_e_bus.py')
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# fixture suffix -> (bench builder, vector names the producer wrote for the
# bus bits and array copies, raw-path code, designer-path code, last value)
CASES = {
    'ng': (lambda tmp: gen.build('ngspice', tmp), [
        ('v(d[0])', "tb.probe('d[0]')", "tb.probe('v(d[0])')", 1.0),
        ('v(d[1])', "tb.probe('d[1]')", "tb.probe('v(d[1])')", 2.0),
        ('v(xdut.xamp1<0>.mid)', "tb.probe('xdut.xamp1<0>.mid')", "tb.probe('v(dut.amp1[0].mid)')", 1.0),
        ('v(xdut.xamp1<1>.mid)', "tb.probe('xdut.xamp1<1>.mid')", "tb.probe('v(dut.amp1[1].mid)')", 1.5),
    ]),
    'xy': (lambda tmp: gen.build('xyce', tmp), [
        ('V(D[0])', "tb.probe('d[0]')", "tb.probe('v(d[0])')", 1.0),
        ('V(D[1])', "tb.probe('d[1]')", "tb.probe('v(d[1])')", 2.0),
        ('V(XDUT:XAMP1<0>:MID)', "tb.probe('xdut.xamp1<0>.mid')", "tb.probe('v(dut.amp1[0].mid)')", 1.0),
        ('V(XDUT:XAMP1<1>:MID)', "tb.probe('xdut.xamp1<1>.mid')", "tb.probe('v(dut.amp1[1].mid)')", 1.5),
    ]),
    'bh': (gen.build_behavioral, [
        ('v(d[0])', "tb.probe('d[0]')", "tb.probe('v(d[0])')", 1.0),
        ('v(d[1])', "tb.probe('d[1]')", "tb.probe('v(d[1])')", 2.0),
        ('v(xdut.xamp1<0>.mid)', "tb.probe('xdut.xamp1<0>.mid')", "tb.probe('v(dut.amp1[0].mid)')", 2.0),
        ('v(xdut.xamp1<1>.mid)', "tb.probe('xdut.xamp1<1>.mid')", "tb.probe('v(dut.amp1[1].mid)')", 6.0),
    ]),
}


def _loaded_bench(suffix, tmp_path):
    build, _ = CASES[suffix]
    tb = build(str(tmp_path))
    path = str(FIXTURES / f'e_bus_array_{suffix}.raw')
    tb.netlist.load_file(path)
    tb.netlist.read_raw_file()
    tb.netlist.results_file = path
    tb.netlist._raw = RawResult(read_rawfile(path), path)
    tb.netlist._raw._view_addresses = _capture_addresses(tb.netlist)
    return tb, path


@pytest.mark.parametrize('suffix', sorted(CASES))
def test_producer_writes_bus_bits_and_array_copies_wrapped(suffix):
    names = [v.name for v in read_rawfile(str(FIXTURES / f'e_bus_array_{suffix}.raw')).plots[0].vectors]
    for vector, *_ in CASES[suffix][1]:
        assert vector in names, (vector, names)
    # no vector of the run is a bare bracketed or angled name
    assert all(n.lower().startswith(('v(', 'i(')) for n in names[1:]), names


@pytest.mark.parametrize('mode', ['raw', 'designer'])
@pytest.mark.parametrize('suffix', sorted(CASES))
def test_copied_code_for_bus_bits_and_array_copies_reads_the_traced_vector(suffix, mode, tmp_path):
    tb, path = _loaded_bench(suffix, tmp_path)
    if mode == 'raw':
        payload = emit_view_data(path)
    else:
        ctx = view_node_context(tb)
        assert ctx is not None
        payload = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    entries = {e['name']: e for e in _dump(payload)}
    for vector, raw_code, designer_code, value in CASES[suffix][1]:
        entry = entries[vector]
        code = entry['code']
        assert code['ok'] is True, (vector, code)
        assert entry['identity']['probe'] is not None, vector
        assert code['code'] == (raw_code if mode == 'raw' else designer_code), vector
        signal = eval(code['code'], {'tb': tb})    # exactly as copied
        np.testing.assert_allclose(np.asarray(signal.y), value, err_msg=f'{suffix} {mode} {code["code"]}')
