"""Vector-preservation measurements for the certified primitive model bundle."""
from pathlib import Path

import numpy as np
import pytest

from silk.examples.ota_5t_guide import build_5t_ota
from silk.signals.rawfile import read_rawfile
from silk.simulation._ngspice_compat import NgspiceCompatibility
from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


@pytest.mark.parametrize('analysis,capture', [
    ('tran', 'plain'), ('ac', 'plain'), ('noise', 'plain'),
    ('tran', 'params'), ('tran', 'ammeter'), ('tran', 'savecurrents'),
    ('tran', 'current-source')])
def test_preserve_circuit_vectors(tmp_path, monkeypatch, analysis, capture):
    if not NgspiceCompatibility().has('save_nointernals'):
        pytest.skip('filter measurement requires ngspice with save_nointernals')
    results = []
    for keep in (True, False):
        monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path / str(keep)))
        dut = build_5t_ota()
        dut.omit_device_internal_nodes = not keep
        dut.save_device_params = capture == 'params'
        dut.save_device_currents = capture in ('ammeter', 'savecurrents')
        if capture == 'ammeter':
            dut.terminal_current_method = 'ammeter_wrapper'
        tb = dut.get_test(analysis)
        probes = []
        if capture == 'current-source':
            from silk import sources
            tb.add_source(sources.CurrentSource(
                'load', positive='vout', negative='0', dc=1e-6,
                sine=sources.Sine(voffset=0, vampl=1e-7, freq=1e6)))
        if analysis != 'noise':
            probes = [tb.probe('vout'), tb.probe('xdut.vint'), tb.probe('i(vdd)')]
            if capture == 'ammeter':
                probes.append(tb.probe('xdut.M1:d'))
            elif capture != 'savecurrents':
                probes.append(tb.probe('M1:gm'))
            if capture == 'current-source':
                probes.append(tb.probe('i(load)'))
        tb.run()
        path = Path(tb.results_file)
        raw = read_rawfile(path)
        results.append((path.stat().st_size, raw, [p.y.copy() for p in probes]))
    before, after = results
    assert len(before[1].plots) == len(after[1].plots)
    removed = 0
    for old, new in zip(before[1].plots, after[1].plots):
        old_vectors = {v.name: v.data for v in old.vectors}
        new_vectors = {v.name: v.data for v in new.vectors}
        dropped = set(old_vectors) - set(new_vectors)
        assert all(name.endswith(('#body)', '#dbody)', '#sbody)')) for name in dropped), dropped
        assert set(new_vectors) <= set(old_vectors)
        removed += len(dropped)
        for name, values in new_vectors.items():
            np.testing.assert_array_equal(old_vectors[name], values, err_msg=name)
    for old, new in zip(before[2], after[2]):
        np.testing.assert_array_equal(old, new)
    if analysis != 'noise':
        assert removed > 0
        assert after[0] < before[0]
    print(f'{analysis}/{capture}: {before[0]} -> {after[0]} bytes; removed {removed} body vectors')
