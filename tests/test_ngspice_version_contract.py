"""Tiny ideal-device release contracts; select the binary with SILK_NGSPICE_BIN.

These test raw simulator facts, not silk's emission (TASK-545 owns new source
forms). Runtime advice is also checked across serial and forked launches.
OSDI needs a real compiled plugin and remains in test_osdi_models.py.
"""
import subprocess

import numpy as np
import pytest

from silk.signals.rawfile import read_rawfile
from silk.simulation._ngspice_compat import NgspiceCompatibility
from silk.simulation.runner import ngspice_executable
from tests._env import needs_ngspice

pytestmark = [pytest.mark.slow, needs_ngspice]


@pytest.fixture
def policy():
    result = NgspiceCompatibility()
    if result.release is None:
        pytest.skip('unrecognized ngspice version; cannot assert release facts')
    return result


def deck(tmp_path, circuit, commands, *, expect_data=True):
    path = tmp_path / 'contract.cir'
    path.write_text('* version contract\n' + circuit + '\n.control\n'
                    'set filetype=ascii\n' + commands
                    + '\nwrite result.raw all\nquit\n.endc\n.end\n')
    run = subprocess.run([ngspice_executable(), '-n', '-b', str(path)],
                         cwd=tmp_path, capture_output=True, text=True, timeout=5)
    log = run.stdout + run.stderr
    raw = tmp_path / 'result.raw'
    if not expect_data:
        return log
    assert run.returncode == 0 and raw.exists(), log
    return read_rawfile(raw), log


def values(raw, name):
    return next(v.data for v in raw.vectors if v.name == name)


@pytest.mark.parametrize('kind', ['AM', 'SFFM'])
def test_modulated_source_order(tmp_path, policy, kind):
    # Distinct frequencies make a swapped pair observable. Zero phases avoid
    # the separate zero-time differences; assert the positive-time waveform.
    if kind == 'AM':
        if policy.has('am_new_order'):
            card = 'AM(0.4 0.3 0.2 100 1000 0 0 0)'
            expected = lambda t: .4 + (.3 + .2*np.sin(2*np.pi*100*t))*np.sin(2*np.pi*1000*t)
        else:
            card = 'AM(0.2 0.3 100 1000 0)'
            expected = lambda t: .2*(.3 + np.sin(2*np.pi*100*t))*np.sin(2*np.pi*1000*t)
    else:
        if policy.has('sffm_carrier_first'):
            card = 'SFFM(0.4 0.2 1000 0.3 100 0 0 0)'
        elif policy.has('sffm_modulator_first'):
            card = 'SFFM(0.4 0.2 100 0.3 1000 0 0 0)'
        else:
            card = 'SFFM(0.4 0.2 1000 0.3 100 0 0)'
        expected = lambda t: .4 + .2*np.sin(2*np.pi*1000*t + .3*np.sin(2*np.pi*100*t))
    raw, _ = deck(tmp_path, f'V1 out 0 {card}\nR1 out 0 1k', 'tran 1u 100u')
    t = raw.vectors[0].data
    np.testing.assert_allclose(values(raw, 'v(out)')[t > 0], expected(t[t > 0]), atol=1e-10)


def test_zero_resistor(tmp_path, policy):
    raw, _ = deck(tmp_path, 'I1 0 out 1\nR1 out 0 0', 'op')
    expected = 1e-12 if policy.has('zero_resistor_clamp') else 1e-3
    assert values(raw, 'v(out)')[0] == pytest.approx(expected, rel=1e-6, abs=1e-18)


def test_save_nosub(tmp_path, policy):
    raw, _ = deck(tmp_path, 'V1 out 0 1\nX1 out load\n.subckt load in\n'
                  'R1 in inside 1k\nR2 inside 0 1k\n.ends', 'save all nosub\nop')
    names = {v.name for v in raw.vectors}
    assert ('v(x1.inside)' not in names) == policy.has('save_nosub')
    assert 'v(out)' in names


def test_save_nointernals(tmp_path, policy):
    # A hash-named node exercises the filter without a transistor model.
    # save all already drops the special #internal suffix on older releases.
    raw, _ = deck(tmp_path, 'V1 out 0 0.5\nR1 out out#private 1k\nR2 out#private 0 1k',
                  'save all nointernals\nop')
    names = {v.name for v in raw.vectors}
    assert ('v(out#private)' not in names) == policy.has('save_nointernals')
    assert 'v(out)' in names


def test_strict_dc_syntax(tmp_path, policy):
    # Missing arguments exercise the parser without executing a zero-step loop.
    log = deck(tmp_path, 'V1 out 0 1\nR1 out 0 1k\n.dc V1 0', 'op', expect_data=False)
    assert ('Bad syntax' in log) == policy.has('strict_dc_syntax')


def test_current_pwl_timing(tmp_path, policy):
    circuit = 'I1 0 out PWL(0 0 1u 1m 2u 0) r=0 td=1u\nR1 out 0 1k'
    if not policy.has('current_pwl_timing'):
        log = deck(tmp_path, circuit, 'tran 10n 5u', expect_data=False)
        assert 'unknown parameter' in log.lower(), log
    else:
        raw, _ = deck(tmp_path, circuit, 'tran 10n 5u')
        t = raw.vectors[0].data
        y = values(raw, 'v(out)')
        assert np.interp(.5e-6, t, y) == pytest.approx(0, abs=1e-8)
        for peak in (2e-6, 4e-6):
            assert np.interp(peak, t, y) == pytest.approx(1, abs=.015)


def test_ac_lin_two(tmp_path, policy):
    raw, _ = deck(tmp_path, 'V1 out 0 DC 0 AC 1\nR1 out 0 1k', 'ac lin 2 1k 2k')
    assert raw.npoints == (1 if policy.has_bug('ac_lin_two') else 2)


def test_dec_subdecade_grid(tmp_path, policy):
    # Two or more steps on every release: no runaway deck is launched. This
    # measures the same fixed branch with a safe span, including on 42.
    raw, _ = deck(tmp_path, 'V1 out 0 DC 0 AC 1\nR1 out 0 1k', 'ac dec 10 1k 2k')
    if policy.has_bug('dec_subdecade_runaway'):
        expected = 1000 * 2**(np.arange(4)/3)
    else:
        expected = 1000 * 10**(np.arange(4)/10)
    np.testing.assert_allclose(raw.vectors[0].data, expected, rtol=1e-9)


@pytest.mark.parametrize('n_workers', [1, 2])
def test_launch_advice_once_at_user_call(tmp_path, monkeypatch, policy, n_workers):
    """Record child warnings too: parent-only catch_warnings misses repeats."""
    import inspect
    import json
    import multiprocessing as mp
    import os
    import warnings
    from silk import Module, sources
    from silk.elements import elements
    from silk.simulation import Transient, _ngspice_compat as compat

    if not policy.advice:
        pytest.skip('selected ngspice has no update advisory')
    if n_workers > 1 and 'fork' not in mp.get_all_start_methods():
        pytest.skip('requires the fork executor')
    monkeypatch.setattr(compat, '_advised_binaries', set())
    warning_log = tmp_path / 'warnings.jsonl'
    def record(message, category, filename, lineno, file=None, line=None):
        if issubclass(category, compat.NgspiceVersionWarning):
            # Each process appends one short record with one write.
            with warning_log.open('a') as stream:
                stream.write(json.dumps([os.getpid(), str(message), filename, lineno]) + '\n')

    dut = Module(None, str(tmp_path / 'dut'), 'parallel.cir', pdk_details=None, module_name='rc')
    dut.define_module_ports({'in': 'in', 'out': 'out', 'vss': 'ground'})
    dut.add(elements.resistor('r1', p='in', n='out', value=1000.0))
    dut.add(elements.capacitor('c1', p='out', n='vss', value=1e-9))
    for index in range(2):
        tb = dut.create_test(f'bench{index}')
        tb.add_module_under_test({'in': 'vin', 'out': 'vout', 'vss': '0'})
        tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=index + 1.0))
        tb.add_sim_type(Transient(step=1e-9, stop=20e-9))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('always', compat.NgspiceVersionWarning)
            monkeypatch.setattr(warnings, 'showwarning', record)
            caller_line = inspect.currentframe().f_lineno + 1
            results = dut.run_all_tests(n_workers=n_workers)
        assert len(results) == 2
        assert all(not isinstance(result, dict) for result in results.values()), results
        records = [json.loads(line) for line in warning_log.read_text().splitlines()]
        assert records == [[os.getpid(), policy.advice, __file__, caller_line]]
    finally:
        dut.close()
