"""Xyce Layer 3 reason extraction, and the gate wired for Xyce."""
import pytest
from silk.signals.rawfile import read_rawfile
from silk.simulation.xyce_runner import xyce_abort_reason, XYCE_ABORT_SENTINELS
from silk.simulation.backend import _require_data, RawResult
from silk.exceptions import SimulationAborted


def test_xyce_reason_finds_sentinel():
    text = "Reading netlist\nXyce Abort: time step too small\n"
    reason = xyce_abort_reason(text)
    assert reason is not None
    assert 'time step too small' in reason.lower()


def test_xyce_reason_none_when_clean():
    assert xyce_abort_reason("End of Xyce(TM) Simulation\n") is None


def test_gate_accepts_precomputed_reason(tmp_path):
    raw_text = (
        "Title: t\nDate: d\nPlotname: Transient Analysis\nFlags: real\n"
        "No. Variables: 2\nNo. Points: 0\n"
        "Variables:\n\t0\ttime\ttime\n\t1\tv(out)\tvoltage\nValues:\n"
    )
    p = tmp_path / 'rawfile.raw'
    p.write_text(raw_text)
    raw = RawResult(native=read_rawfile(str(p)), source_path=str(p))
    with pytest.raises(SimulationAborted) as ei:
        _require_data(raw, simulator='xyce', reason='time step too small')
    assert ei.value.reason == 'time step too small'


@pytest.mark.parametrize('fixture', ['xyce_duplicate_stdout.txt', 'xyce_duplicate_long_stdout.txt',
                                     'xyce_duplicate_error_path_stdout.txt'])
def test_nonzero_exit_keeps_stdout_diagnostic_when_stderr_has_abort_summary(tmp_path, fixture):
    from pathlib import Path
    from unittest.mock import patch
    from subprocess import CompletedProcess
    from silk.simulation.xyce_runner import XyceRunner
    from silk.exceptions import SimulationError
    fixtures = Path(__file__).parent / 'fixtures/simulator_failures'
    stdout = (fixtures / fixture).read_text()
    stderr = (fixtures / 'xyce_duplicate_stderr.txt').read_text()
    (tmp_path / 'failure.cir').write_text('Diagnostic fixture\n.end\n')
    runner = XyceRunner(str(tmp_path / 'failure.cir'), str(tmp_path))
    with patch('subprocess.run', return_value=CompletedProcess([], 1, stdout, stderr)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    message = str(error.value)
    displayed = ' '.join(message.split())
    assert displayed.index('Netlist error in file') < displayed.index('near line 5') < displayed.index('Duplicate device XDUT:R1')
    assert 'Simulation aborted due to error' in message


def test_nonzero_exit_diagnostic_precedes_large_trailing_output(tmp_path):
    from unittest.mock import patch
    from subprocess import CompletedProcess
    from silk.simulation.xyce_runner import XyceRunner
    from silk.exceptions import SimulationError
    stdout = 'Netlist error in file failure.cir at or\n near line 5\n Duplicate device R1\n' + 'trailing output\n' * 1000
    (tmp_path / 'failure.cir').write_text('Diagnostic fixture\n.end\n')
    runner = XyceRunner(str(tmp_path / 'failure.cir'), str(tmp_path))
    with patch('subprocess.run', return_value=CompletedProcess([], 1, stdout, 'abort summary')):
        with pytest.raises(SimulationError) as error:
            runner.run()
    assert 'Duplicate device R1' in str(error.value)
    assert len(str(error.value)) < 4100


@pytest.mark.parametrize('padding', [0, 3500, 3850])
def test_wrapped_xyce_location_and_reason_are_atomic(padding):
    from pathlib import Path
    from silk.simulation._failure_detail import failure_detail
    log = (Path(__file__).parent / 'fixtures/simulator_failures/xyce_duplicate_long_stdout.txt').read_text()
    prefix = ('Error: ' + 'x' * 493 + '\n') * (padding // 500)
    if padding % 500:
        prefix += 'Error: ' + 'x' * (padding % 500 - 7) + '\n'
    # Add another location wrap without changing the diagnostic meaning.
    log = log.replace('/evidence/xyce/', '/evidence/\n xyce/').replace('line 5', 'line\n 5')
    detail = failure_detail(prefix + log)
    assert len(detail) <= 4000
    assert ('Netlist error in file' in detail) == ('Duplicate device XDUT:R1' in detail)
    if not padding:
        assert 'Duplicate device XDUT:R1' in detail

    if padding == 3850:
        assert 'Netlist error in file' not in detail
        assert 'truncated' in detail


@pytest.mark.parametrize('filename', ['error-cases', 'issue-cases', 'fatal-cases'])
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
@pytest.mark.parametrize('padding', [0, 3500, 3850])
def test_measured_filename_continuation_stays_in_atomic_location(filename, newline, padding):
    from pathlib import Path
    from silk.simulation._failure_detail import failure_detail
    measured = (Path(__file__).parent / 'fixtures/simulator_failures/xyce_duplicate_error_path_stdout.txt').read_text()
    # Change only the same-width filename word: wrapping remains identical.
    # Focus the budget on the diagnostic block; the runner replay above
    # separately covers the full captured stream, including its banner.
    measured = measured[measured.index('Netlist error in file'):]
    log = measured.replace('error-cases', filename).replace('\n', newline)
    prefix = ('Error: ' + 'x' * 493 + newline) * (padding // 500)
    if padding % 500:
        prefix += 'Error: ' + 'x' * (padding % 500 - 7) + newline
    detail = failure_detail(prefix + log)
    assert len(detail) <= 4000
    assert ('Netlist error in file' in detail) == ('Duplicate device XDUT:R1' in detail)
    if padding == 3850:
        assert 'Netlist error in file' not in detail
        assert 'truncated' in detail
    else:
        assert filename in detail
        assert 'Duplicate device XDUT:R1' in detail


@pytest.mark.parametrize('header', ['FATAL: stopped', 'User error: broken', 'Fatal error: stopped'])
@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_incomplete_xyce_location_still_stops_at_genuine_diagnostic(header, newline):
    from silk.simulation._failure_detail import failure_detail
    lines = ['Netlist error in file', ' /tmp/incomplete', header, 'next-card', 'next-reason']
    assert failure_detail(newline.join(lines)) == '\n  '.join(lines)


def _singular_text():
    from pathlib import Path
    return (Path(__file__).parent / 'fixtures/convergence_logs/xyce-7.10/vsource_loop.log').read_text()


def test_singular_matrix_failure_leads_with_its_cause(tmp_path):
    from subprocess import CompletedProcess
    from unittest.mock import patch
    from silk.exceptions import ConvergenceError
    from silk.simulation.xyce_runner import XyceRunner
    (tmp_path / 'f.cir').write_text('x\n.end\n')
    runner = XyceRunner(str(tmp_path / 'f.cir'), str(tmp_path))
    with patch('subprocess.run', return_value=CompletedProcess([], 1, _singular_text(), '')):
        with pytest.raises(ConvergenceError) as error:
            runner.run()
    message = str(error.value)
    assert 'singular matrix' in message[:240].lower()
    assert 'Welcome to the Xyce' not in message


def test_fallback_without_cause_skips_banner_and_timing():
    from silk.simulation._failure_detail import failure_detail
    from silk.simulation.xyce_runner import _XYCE_NOISE
    text = '\n'.join(l for l in _singular_text().splitlines() if 'singular' not in l.lower())
    detail = failure_detail(text, noise=_XYCE_NOISE)
    assert 'Welcome' not in detail and 'Wall Time' not in detail
    assert 'Netlist warning: No print specified' in detail
    assert len(detail) <= 4000


def test_lead_lines_respect_total_bound_and_marker():
    from silk.simulation._failure_detail import failure_detail
    from silk.simulation.convergence import convergence_lines
    for count in (1, 8, 10):
        for newline in ('\n', '\r\n'):
            text = newline.join(
                f'Netlist warning: Numerically singular matrix at node n{i} ' + 'x' * 600
                for i in range(count))
            detail = failure_detail(text, lead=convergence_lines('xyce', text))
            assert len(detail) <= 4000
            assert detail.startswith('Netlist warning: Numerically singular matrix at node n0')
            assert 'truncated' in detail


def test_noise_filter_keeps_real_lines_and_never_restores_banner():
    from silk.simulation._failure_detail import failure_detail
    from silk.simulation.xyce_runner import _XYCE_NOISE
    text = 'Netlist warning: voltage limit exceeded at VOUT (12.5%)\n***** Voltage limit exceeded at VOUT\nSimulation stopped'
    detail = failure_detail(text, noise=_XYCE_NOISE)
    assert 'VOUT (12.5%)' in detail and '***** Voltage limit' in detail
    only_noise = '***** Welcome to the Xyce(TM) Parallel Electronic Simulator\nTiming summary of 1 processor'
    detail = failure_detail(only_noise, noise=_XYCE_NOISE)
    assert 'Welcome' not in detail and 'Timing summary' not in detail and detail


@pytest.mark.parametrize('text', ['solver stopped\n\n  node n1 has no reference',
                                  'a  \r\nb\r\n\r\nc', 'solver ' * 700 + '\n\nend'],
                         ids=['blank-line', 'crlf', 'over-limit'])
def test_default_fallback_is_literal_text(text):
    from silk.simulation._failure_detail import failure_detail
    detail = failure_detail(text)
    if len(text.strip()) <= 4000:
        assert detail == text.strip()
    else:
        assert detail.endswith(text.strip()[-100:]) and len(detail) <= 4000


def test_stream_shaped_launch_leads_with_cause(tmp_path):
    from unittest.mock import patch
    from subprocess import CompletedProcess
    from silk.exceptions import ConvergenceError
    from silk.simulation.xyce_runner import XyceRunner
    (tmp_path / 'f.cir').write_text('x\n.end\n')
    runner = XyceRunner(str(tmp_path / 'f.cir'), str(tmp_path))
    log_text = _singular_text()

    def launch(*args, stdout=None, **kwargs):
        stdout.write(log_text)
        return CompletedProcess(args, 1, None, None)
    with patch('subprocess.run', side_effect=launch):
        with pytest.raises(ConvergenceError) as error:
            runner.run()
    assert 'singular matrix' in str(error.value)[:240].lower()
    assert error.value.matched_lines
    assert error.value.log_path
    assert open(error.value.log_path).read() == log_text


def test_timing_rows_dropped_only_inside_the_table():
    from silk.simulation._failure_detail import failure_detail
    from silk.simulation.xyce_runner import _XYCE_NOISE
    warning = 'Netlist warning: residual at VOUT: 1 0.100 (12.5%) 0.200 (25.0%)'
    table = ('Timing summary of 1 processor\n'
             'Stats    Count   CPU Time   Wall Time\n'
             '---------------------------------------- ----- ---------\n'
             'Xyce                                         1        0.009 (100.0%)        0.003 (100.0%)\n'
             '  Late Initialization                        1        0.000 ( 0.00%)        0.000 (10.85%)\n')
    before = failure_detail(warning + '\nSimulation stopped', noise=_XYCE_NOISE)
    assert warning in before
    inside = failure_detail(table + warning + '\n', noise=_XYCE_NOISE)
    assert warning in inside and 'Late Initialization' not in inside
    after = failure_detail(table + 'Netlist warning: after table\n', noise=_XYCE_NOISE)
    assert 'after table' in after and 'Xyce  ' not in after


@pytest.mark.parametrize('name', ['xyce_dc_op_fail.log', 'xyce_tran_dcop_fail.log'])
def test_captured_timing_section_alone_gives_placeholder(name):
    from pathlib import Path
    from silk.simulation._failure_detail import failure_detail
    from silk.simulation.xyce_runner import _XYCE_NOISE
    text = (Path(__file__).parent / 'fixtures/convergence_logs/xyce-7.10' / name).read_text()
    section = text[text.index('Timing summary'):]
    detail = failure_detail(section, noise=_XYCE_NOISE)
    assert detail.startswith('(no diagnostic lines found')
    assert 'Wall Time' not in detail and 'Verify Devices' not in detail
    kept = failure_detail(section + 'Netlist warning: after table\n', noise=_XYCE_NOISE)
    assert kept == 'Netlist warning: after table'


def test_table_ends_at_non_row_and_second_table_is_dropped():
    from silk.simulation._failure_detail import failure_detail
    from silk.simulation.xyce_runner import _XYCE_NOISE
    head = 'Stats    Count   CPU Time   Wall Time\n'
    row = 'Xyce          1        0.009 (100.0%)        0.003 (<0.01%)\n'
    text = (head + row + 'Solver note\n' + row + head + row + row)
    detail = failure_detail(text, noise=_XYCE_NOISE)
    assert detail == 'Solver note\n' + row.rstrip()
