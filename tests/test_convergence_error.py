"""TASK-316: convergence failures are recognised (detection only).

The fixtures under ``tests/fixtures/convergence_logs`` are logs captured from
ngspice 42, ngspice 47 and Xyce 7.10 on small deliberately failing decks
(``decks/``).
"""
import os
import shutil
from pathlib import Path

import pytest

from silk.exceptions import (ConvergenceError, SimulationAborted,
                             SimulationError)
from silk.simulation.convergence import (CONVERGENCE_PATTERNS,
                                         convergence_lines)

FIX = Path(__file__).parent / 'fixtures' / 'convergence_logs'


def _copied_deck(tmp_path, name):
    """Copy a fixture deck into tmp_path: failed runs leave hidden decks beside it."""
    return str(Path(shutil.copy(FIX / 'decks' / name, tmp_path)))

NG_CONVERGENCE = ['tran_op_fail', 'dc_op_fail', 'vsource_loop']
XY_CONVERGENCE = ['xyce_tran_dcop_fail', 'xyce_dc_op_fail',
                  'xyce_tran_step_collapse', 'vsource_loop']


def _text(*parts):
    return FIX.joinpath(*parts).read_text(errors='replace')


@pytest.mark.parametrize('folder', ['ngspice-42', 'ngspice-47'])
@pytest.mark.parametrize('case', NG_CONVERGENCE)
def test_ngspice_logs_are_convergence(folder, case):
    assert convergence_lines('ngspice', _text(folder, case + '.log'))


@pytest.mark.parametrize('case', XY_CONVERGENCE)
def test_xyce_logs_are_convergence(case):
    assert convergence_lines('xyce', _text('xyce-7.10', case + '.log'))


@pytest.mark.parametrize('folder', ['ngspice-42', 'ngspice-47'])
def test_ngspice_non_convergence_failure_is_not(folder):
    assert convergence_lines('ngspice',
                             _text(folder, 'unknown_subckt.log')) == []


def test_xyce_non_convergence_failure_is_not():
    assert convergence_lines('xyce',
                             _text('xyce-7.10', 'xyce_unknown_subckt.log')) == []


def test_specific_lines_and_dedup():
    lines = convergence_lines('ngspice', _text('ngspice-42', 'vsource_loop.log'))
    assert lines[0].startswith('Warning: singular matrix')
    assert len(lines) == len(set(lines))
    lines = convergence_lines('xyce', _text('xyce-7.10',
                                            'xyce_tran_step_collapse.log'))
    assert any('Time step too small' in ln for ln in lines)


def test_unknown_simulator_and_empty_text():
    assert convergence_lines('behavioral', 'singular matrix') == []
    assert convergence_lines('ngspice', '') == []
    assert convergence_lines('xyce', None) == []


def test_clean_xyce_summary_is_not_a_match():
    assert convergence_lines(
        'xyce', '\tNumber Nonlinear Convergence Failures:\t0') == []


def test_pattern_table_is_per_simulator_data():
    assert set(CONVERGENCE_PATTERNS) == {'ngspice', 'xyce'}


def test_convergence_error_is_a_simulation_aborted():
    assert issubclass(ConvergenceError, SimulationAborted)
    assert issubclass(ConvergenceError, SimulationError)
    err = ConvergenceError('m', simulator='xyce', matched_lines=['a'],
                           log_path='p')
    assert (err.simulator, err.matched_lines, err.log_path) == ('xyce', ['a'], 'p')


# ---------------------------------------------------------------- real runs

def _ngspice_bins():
    bins = {}
    if shutil.which('ngspice'):
        bins['system'] = shutil.which('ngspice')
    env = os.environ.get('SILK_NGSPICE_BIN')
    if env and os.path.exists(env):
        bins['env'] = env
    return bins


@pytest.mark.parametrize('which', ['system', 'env'])
@pytest.mark.parametrize('deck,expect', [('tran_op_fail', True),
                                         ('vsource_loop', True),
                                         ('unknown_subckt', False)])
def test_ngspice_runner_raises_convergence_error(which, deck, expect,
                                                 tmp_path, monkeypatch):
    bins = _ngspice_bins()
    if which not in bins:
        pytest.skip('ngspice binary not available')
    monkeypatch.setenv('SILK_NGSPICE_BIN', bins[which])
    from silk.simulation.runner import NgspiceRunner
    runner = NgspiceRunner(_copied_deck(tmp_path, deck + '.cir'), str(tmp_path))
    with pytest.raises(SimulationError) as ei:
        runner.run()
    assert isinstance(ei.value, ConvergenceError) is expect
    if expect:
        assert ei.value.simulator == 'ngspice'
        assert ei.value.matched_lines
        assert ei.value.log_path == runner.logfile_path
    else:
        assert type(ei.value) is SimulationError


@pytest.mark.parametrize('deck,expect', [('xyce_dc_op_fail', True),
                                         ('xyce_tran_dcop_fail', True),
                                         ('xyce_unknown_subckt', False)])
def test_xyce_runner_raises_convergence_error(deck, expect, tmp_path):
    from silk.simulation.xyce_runner import XyceRunner, _xyce_executable
    exe = _xyce_executable()
    if not (shutil.which(exe) or os.path.exists(exe)):
        pytest.skip('Xyce binary not available')
    runner = XyceRunner(_copied_deck(tmp_path, deck + '.cir'), str(tmp_path))
    with pytest.raises(SimulationError) as ei:
        runner.run()
    assert isinstance(ei.value, ConvergenceError) is expect
    if expect:
        assert ei.value.simulator == 'xyce'
        assert ei.value.matched_lines


# ------------------------------------------- device-probe attribution path

def test_device_probe_attribution_preserves_convergence_error():
    from types import SimpleNamespace
    from silk.circuit.module_test import _attribute_abort_to_device_probe
    netlist = SimpleNamespace(_pending_device_probes=[('m1', 'cgso')])
    exc = ConvergenceError('write failed on m1[cgso]', simulator='ngspice',
                           matched_lines=['Warning: singular matrix'],
                           reason='r', data_rows=None, log_path='/x.log')
    out = _attribute_abort_to_device_probe(exc, netlist)
    assert type(out) is ConvergenceError
    assert out is not exc and 'm1:cgso' in str(out)
    assert out.matched_lines == ['Warning: singular matrix']
    assert (out.simulator, out.reason, out.log_path) == ('ngspice', 'r', '/x.log')


def test_device_probe_attribution_plain_abort_stays_plain():
    from types import SimpleNamespace
    from silk.circuit.module_test import _attribute_abort_to_device_probe
    netlist = SimpleNamespace(_pending_device_probes=[('m1', 'cgso')])
    exc = SimulationAborted('m1[cgso]', simulator='ngspice')
    assert type(_attribute_abort_to_device_probe(exc, netlist)) is SimulationAborted


@pytest.mark.slow
@pytest.mark.requires_xyce
def test_xyce_singular_matrix_message_leads_with_cause(tmp_path):
    from silk.simulation.xyce_runner import XyceRunner, _xyce_executable
    exe = _xyce_executable()
    if not (shutil.which(exe) or os.path.exists(exe)):
        pytest.skip('Xyce binary not available')
    runner = XyceRunner(_copied_deck(tmp_path, 'vsource_loop.cir'), str(tmp_path))
    with pytest.raises(ConvergenceError) as ei:
        runner.run()
    assert 'singular matrix' in str(ei.value)[:240].lower()
