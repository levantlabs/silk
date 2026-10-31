"""write_rawfile: round-trips through silk's own reader (TASK-448.2).

Expectations are the format itself: what is written must read back with the
same vectors, types, attributes, plot name, header and values. The slow test
closes the loop on a real ngspice run, through ``NgspiceSignal.from_rawfile``
(the path ``load_signal`` takes).
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from silk.signals.rawfile import (
    RawFile, RawPlot, RawVector, RawfileError, read_rawfile, write_rawfile)
from silk.signals.spice import NgspiceSignal
from tests._env import needs_ngspice

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'


def _real_plot():
    t = np.linspace(0.0, 1e-6, 7)
    header = {'title': 'rt', 'date': 'today', 'plotname': 'Transient Analysis',
              'flags': 'real', 'command': 'silk behavioural'}
    vecs = [RawVector('time', 'time', (), t),
            RawVector('v(out)', 'voltage', (), np.sin(t * 1e7) / 3.0),
            RawVector('i(vdd)', 'current', (), -t * 1e3 + 0.1)]
    return RawPlot(header, vecs, 7, True, None)


def _complex_plot():
    f = np.logspace(0, 6, 5)
    header = {'title': 'ac', 'date': 'today', 'plotname': 'AC Analysis',
              'flags': 'complex'}
    vecs = [RawVector('frequency', 'frequency', ('grid=3',), f),
            RawVector('v(out)', 'voltage', (), 1 / (1 + 1j * f / 1e3))]
    return RawPlot(header, vecs, 5, False, None)


def _assert_same(a, b):
    assert a.plotname == b.plotname and a.real == b.real
    assert a.npoints == b.npoints and a.nvars == b.nvars
    for key in ('title', 'date', 'plotname', 'command'):
        assert a.header.get(key, '') == b.header.get(key, '')
    for va, vb in zip(a.vectors, b.vectors):
        assert (va.name, va.type, va.attrs) == (vb.name, vb.type, vb.attrs)
        assert va.data.dtype == vb.data.dtype
        np.testing.assert_array_equal(va.data, vb.data)


@pytest.mark.parametrize('binary', [True, False], ids=['binary', 'ascii'])
@pytest.mark.parametrize('make', [_real_plot, _complex_plot],
                         ids=['real', 'complex'])
def test_round_trip(tmp_path, make, binary):
    plot = make()
    path = tmp_path / 'x.raw'
    write_rawfile(path, plot, binary=binary)
    back = read_rawfile(path)
    assert len(back.plots) == 1
    _assert_same(plot, back.current_plot)
    text = path.read_bytes()
    assert (b'\nBinary:\n' in text) is binary
    assert (b'\nValues:\n' in text) is (not binary)


def test_multi_plot_order_and_current(tmp_path):
    path = tmp_path / 'm.raw'
    write_rawfile(path, [_complex_plot(), _real_plot()])
    back = read_rawfile(path)
    assert [p.plotname for p in back.plots] == [
        'AC Analysis', 'Transient Analysis']
    assert back.current_plot.plotname == 'Transient Analysis'


@pytest.mark.parametrize('name', ['a_tran_bin.raw', 'a_ac_bin.raw',
                                  'a_tran_ascii.raw', 'd_multiplot_ac_tran.raw',
                                  'b_tran_bin_offset.raw'])
def test_fixture_rewrite_is_a_fixed_point(tmp_path, name):
    """Real simulator files survive read -> write -> read, both encodings
    (including an Offset: file, whose offset is already in vector 0)."""
    src = read_rawfile(FIXTURES / name)
    for binary in (True, False):
        path = tmp_path / f'{binary}.raw'
        write_rawfile(path, src, binary=binary)
        back = read_rawfile(path)
        assert len(back.plots) == len(src.plots)
        for a, b in zip(src.plots, back.plots):
            _assert_same(a, b)


def test_dims0_vector_reads_back_empty(tmp_path):
    plot = _real_plot()
    plot.vectors[2] = RawVector('v(none)', 'voltage', ('dims=0',))
    write_rawfile(tmp_path / 'd.raw', plot)
    back = read_rawfile(tmp_path / 'd.raw').current_plot
    assert len(back.vectors[2].data) == 0
    np.testing.assert_array_equal(back.vectors[1].data, plot.vectors[1].data)


def test_zero_points(tmp_path):
    plot = RawPlot({'plotname': 'Transient Analysis', 'flags': 'real'},
                   [RawVector('time', 'time')], 0, True, None)
    write_rawfile(tmp_path / 'z.raw', plot)
    assert read_rawfile(tmp_path / 'z.raw').current_plot.npoints == 0


def test_refusals(tmp_path):
    p = tmp_path / 'bad.raw'
    with pytest.raises(RawfileError, match='no plots'):
        write_rawfile(p, [])
    short = _real_plot()
    short.vectors[1].data = short.vectors[1].data[:3]
    with pytest.raises(RawfileError, match='expected 7'):
        write_rawfile(p, short)
    cplx = _real_plot()
    cplx.vectors[1].data = cplx.vectors[1].data.astype(complex)
    with pytest.raises(RawfileError, match='complex'):
        write_rawfile(p, cplx)
    nl = _real_plot()
    nl.header['title'] = 'a\nb'
    with pytest.raises(RawfileError, match='line break'):
        write_rawfile(p, nl)


_NETLIST = """rc
V1 in 0 DC 0 AC 1 PULSE(0 1 0 1u 1u 1 2)
R1 in out 1k
C1 out 0 1n
.control
tran 0.1u 5u
write {tran} v(out) v(in)
ac dec 5 1k 10meg
write {ac} v(out)
.endc
.end
"""


@pytest.mark.slow
@needs_ngspice
@pytest.mark.parametrize('binary', [True, False], ids=['binary', 'ascii'])
def test_real_ngspice_result_round_trips_through_load_signal(tmp_path, binary):
    exe = os.environ.get('SILK_NGSPICE_BIN') or shutil.which('ngspice')
    cir = tmp_path / 'rc.cir'
    tran, ac = tmp_path / 'tran.raw', tmp_path / 'ac.raw'
    cir.write_text(_NETLIST.format(tran=tran, ac=ac))
    subprocess.run([exe, '-b', str(cir)], check=True, capture_output=True,
                   timeout=120)
    for src_path, names in ((tran, ('out', 'in')), (ac, ('out',))):
        src = read_rawfile(src_path)
        out = tmp_path / f'rewritten_{src_path.name}'
        write_rawfile(out, src, binary=binary)
        back = read_rawfile(out)
        for name in names:
            a = NgspiceSignal.from_rawfile(src, name, 'V')
            b = NgspiceSignal.from_rawfile(back, name, 'V')
            np.testing.assert_array_equal(a.x, b.x)
            np.testing.assert_array_equal(a.y, b.y)
            assert (a.xlabel, a.ylabel) == (b.xlabel, b.ylabel)
