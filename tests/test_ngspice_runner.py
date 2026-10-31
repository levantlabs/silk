"""Tests for NgspiceRunner — mocks subprocess so no ngspice binary needed."""
from pathlib import Path

import pytest
from unittest.mock import patch, MagicMock
from silk.simulation.runner import (
    NgspiceRunner, abort_reason_from_log, NGSPICE_ABORT_SENTINELS,
    ngspice_executable,
)
from silk.exceptions import SimulationError, SimulationAborted


pytestmark = pytest.mark.usefixtures('mock_ngspice_version')


@pytest.fixture(autouse=True)
def prepared_decks(tmp_path):
    for name in ('test.spi', 'test.cir'):
        (tmp_path / name).write_text('Runner fixture\n.end\n')


def _launch_writes(tmp_path, *, raw=None, log=None, returncode=0):
    def launch(argv, **kwargs):
        if raw is not None:
            Path(argv[argv.index('-r') + 1]).write_text(raw)
        if log is not None:
            Path(argv[argv.index('-o') + 1]).write_text(log)
        return MagicMock(returncode=returncode)
    return launch


class TestNgspiceRunnerRun:
    def test_run_success(self, tmp_path):
        (tmp_path / 'rawfile.raw').write_text("stub")  # old output must be replaced during launch
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = _launch_writes(tmp_path, raw="stub")
            runner.run()  # should not raise
        mock_run.assert_called_once()

    def test_run_failure_raises_simulation_error(self, tmp_path):
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            with pytest.raises(SimulationError, match='ngspice simulation failed'):
                runner.run()

    def test_run_extracts_error_lines_from_log(self, tmp_path):
        log = tmp_path / 'output.log'
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = _launch_writes(
                tmp_path, log='normal line\nError: something broke\n', returncode=1)
            with pytest.raises(SimulationError, match='something broke'):
                runner.run()

    def test_correct_command_built(self, tmp_path, monkeypatch):
        # This pins the DEFAULT argv, so isolate it from a SILK_NGSPICE_BIN
        # exported in the developer's shell (the version-matrix workflow).
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        (tmp_path / 'rawfile.raw').write_text("stub")  # old output must be replaced during launch
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = _launch_writes(tmp_path, raw="stub")
            runner.run()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == 'ngspice'
        assert '-b' in cmd
        launched_deck = Path(cmd[-1])
        assert launched_deck.parent == tmp_path
        assert launched_deck != tmp_path / 'test.spi'
        assert launched_deck.read_bytes() == (tmp_path / 'test.spi').read_bytes()


class TestNgspiceRunnerLoad:
    """Tests for NgspiceRunner.load() — parsing raw files."""

    def test_load_parses_raw_file(self, tmp_path, capsys):
        """Verify load() correctly parses a minimal valid ngspice raw file."""
        # Create a minimal valid ngspice ASCII raw file
        raw_content = """\
Title: Test Circuit
Date: Mon Jan  1 00:00:00 2024
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
\t0\ttime\ttime
\t1\tv(out)\tvoltage
Values:
\t0\t0.000000e+00
\t\t1.000000e+00
\t1\t1.000000e-09
\t\t1.800000e+00
"""
        rawfile = tmp_path / 'rawfile.raw'
        rawfile.write_text(raw_content)

        runner = NgspiceRunner(str(tmp_path / 'test.spi'), str(tmp_path))

        # Load should parse the file and populate results
        results = runner.load()

        # Verify results object is not None
        assert results is not None
        assert runner.results is results

        # Verify vectors were parsed
        assert len(results.vectors) > 0

        # Verify metadata was parsed
        assert results.current_plot.title == 'Test Circuit'
        assert results.current_plot.plotname == 'Transient Analysis'

        # Debug prints have been removed as a fix (Issue #54)
        # Previous version checked for 'KEYEWORD is: title' in stdout

    def test_load_sets_results_property(self, tmp_path):
        """Verify load() sets the results property."""
        raw_content = """\
Title: Test
Date: Mon Jan  1 00:00:00 2024
Plotname: Transient Analysis
Flags: real
No. Variables: 1
No. Points: 1
Variables:
\t0\ttime\ttime
Values:
\t0\t0.000000e+00
"""
        rawfile = tmp_path / 'rawfile.raw'
        rawfile.write_text(raw_content)

        runner = NgspiceRunner(str(tmp_path / 'test.spi'), str(tmp_path))

        assert runner.results is None
        runner.load()
        assert runner.results is not None


class TestNgspiceAbortReason:
    def test_reason_from_log_finds_timestep_too_small(self, tmp_path):
        log = tmp_path / 'output.log'
        log.write_text("Doing analysis\nTimestep too small; time = 1e-18\n"
                       "run simulation(s) aborted\n")
        reason = abort_reason_from_log(str(log))
        assert reason is not None
        assert 'Timestep too small' in reason

    def test_reason_from_log_none_when_clean(self, tmp_path):
        log = tmp_path / 'output.log'
        log.write_text("Everything is fine\nNo. of Data Rows : 101\n")
        assert abort_reason_from_log(str(log)) is None

    def test_reason_from_missing_log_is_none(self, tmp_path):
        assert abort_reason_from_log(str(tmp_path / 'nope.log')) is None

    def test_sentinels_include_known_aborts(self):
        joined = ' '.join(NGSPICE_ABORT_SENTINELS)
        assert 'aborted' in joined
        assert 'Timestep too small' in joined


class TestNgspiceRunnerMissingRawfile:
    def test_rc0_but_no_rawfile_raises_aborted(self, tmp_path):
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = _launch_writes(
                tmp_path, log="Timestep too small; time = 1e-18\nrun simulation(s) aborted\n")
            # rawfile.raw does NOT exist in tmp_path
            with pytest.raises(SimulationAborted) as ei:
                runner.run()
        err = ei.value
        assert err.simulator == 'ngspice'
        assert 'Timestep too small' in (err.reason or '')

    def test_rc0_with_rawfile_present_does_not_raise(self, tmp_path):
        (tmp_path / 'rawfile.raw').write_text("stub")   # old output
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = _launch_writes(tmp_path, raw="stub")
            runner.run()  # must not raise


class TestNgspiceRunnerAttributes:
    def test_importable_from_silk_simulation(self):
        from silk.simulation import NgspiceRunner
        assert NgspiceRunner is not None

    def test_results_none_before_load(self, tmp_path):
        runner = NgspiceRunner(str(tmp_path / 'test.spi'), str(tmp_path))
        assert runner.results is None

    def test_rawfile_path_set_correctly(self, tmp_path):
        runner = NgspiceRunner('netlist.spi', str(tmp_path))
        assert runner.rawfile_path == str(tmp_path) + '/rawfile.raw'

    def test_logfile_path_set_correctly(self, tmp_path):
        runner = NgspiceRunner('netlist.spi', str(tmp_path))
        assert runner.logfile_path == str(tmp_path) + '/output.log'


class TestNgspiceExecutableOverride:
    """SILK_NGSPICE selects the binary, so one netlist can be run against
    several ngspice versions without touching PATH."""

    def test_defaults_to_path_lookup(self, monkeypatch):
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        assert ngspice_executable() == 'ngspice'

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv('SILK_NGSPICE_BIN', '/opt/ngspice-47/bin/ngspice')
        assert ngspice_executable() == '/opt/ngspice-47/bin/ngspice'

    def test_empty_env_falls_back(self, monkeypatch):
        monkeypatch.setenv('SILK_NGSPICE_BIN', '')
        assert ngspice_executable() == 'ngspice'

    def test_runner_invokes_the_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SILK_NGSPICE_BIN', '/opt/ngspice-47/bin/ngspice')
        (tmp_path / 'rawfile.raw').write_text("stub")
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = _launch_writes(tmp_path, raw="stub")
            runner.run()
        assert mock_run.call_args[0][0][0] == '/opt/ngspice-47/bin/ngspice'

    def test_resolved_at_call_time_not_import_time(self, monkeypatch):
        monkeypatch.setenv('SILK_NGSPICE_BIN', '/first/ngspice')
        assert ngspice_executable() == '/first/ngspice'
        monkeypatch.setenv('SILK_NGSPICE_BIN', '/second/ngspice')
        assert ngspice_executable() == '/second/ngspice'


class TestMissingBinary:
    """An unusable binary must arrive as a SimulationError that names the
    override, not as a bare FileNotFoundError: an unexpanded '~' or a typo in
    $SILK_NGSPICE_BIN otherwise escapes every SimulationError handler in silk
    with a message that never mentions the variable that caused it.
    """

    def test_missing_binary_raises_simulation_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SILK_NGSPICE_BIN', '/nonexistent/ngspice')
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with pytest.raises(SimulationError) as ei:
            runner.run()           # real exec attempt -> real FileNotFoundError
        msg = str(ei.value)
        assert '/nonexistent/ngspice' in msg
        assert 'SILK_NGSPICE_BIN' in msg

    def test_missing_binary_on_path_also_names_the_variable(self, tmp_path, monkeypatch):
        """Same failure with the var unset (nothing named 'ngspice' on PATH):
        the fix hint is still the actionable one."""
        monkeypatch.delenv('SILK_NGSPICE_BIN', raising=False)
        runner = NgspiceRunner(
            netlist_path=str(tmp_path / 'test.spi'),
            results_dir=str(tmp_path),
        )
        with patch('subprocess.run', side_effect=FileNotFoundError(2, 'No such file')):
            with pytest.raises(SimulationError, match='ngspice executable not found'):
                runner.run()


@pytest.mark.parametrize('abort', [
    'doAnalyses: TRAN: Timestep too small; initial timepoint: cause unrecorded.',
    'doAnalyses: AC: Timestep too small; frequency = 0: cause unrecorded.',
])
@pytest.mark.parametrize('warning_before', [True, False])
def test_abort_reason_retains_only_preceding_node_warning(tmp_path, abort, warning_before):
    warning = 'Warning: singular matrix: check node vt_vexp__silk_wave#branch'
    lines = [warning, abort] if warning_before else [abort, warning]
    log = tmp_path / 'output.log'
    log.write_text('\n'.join(lines) + '\n')
    expected = f'{abort} (earlier: {warning})' if warning_before else abort
    assert abort_reason_from_log(log) == expected


@pytest.mark.parametrize('abort', [
    'doAnalyses: TRAN: Timestep too small; initial timepoint: cause unrecorded.',
    'doAnalyses: AC: Timestep too small; frequency = 0: cause unrecorded.',
])
def test_abort_reason_without_node_warning_is_unchanged(tmp_path, abort):
    log = tmp_path / 'output.log'
    log.write_text('Doing analysis\n' + abort + '\nrun simulation(s) aborted\n')
    assert abort_reason_from_log(log) == abort


@pytest.mark.parametrize('warnings', [
    ['Warning: singular matrix: check node old#branch', 'Check node final#branch'],
    ['Check node old#branch', 'Warning: SINGULAR MATRIX: check node final#branch'],
])
def test_abort_reason_retains_last_preceding_node_line(tmp_path, warnings):
    abort = 'doAnalyses: TRAN: Timestep too small; time = 1e-9'
    log = tmp_path / 'output.log'
    log.write_text('\n'.join(warnings + ['Trying another step', abort,
                                          'Warning: singular matrix: check node later#branch']))
    assert abort_reason_from_log(log) == f'{abort} (earlier: {warnings[-1]})'


@pytest.mark.parametrize('case,card,reason', [
    ('duplicate', 'r.xdut.r1 out 0 1k', 'device already exists, bail out'),
    ('unknown_model', 'd1 out 0 missing_model', 'could not find a valid modelname'),
    ('malformed', 'r1 out 0 1k garbage', 'unknown parameter (garbage)'),
])
def test_nonzero_exit_keeps_measured_card_and_reason(tmp_path, case, card, reason):
    from pathlib import Path
    log = (Path(__file__).parent / 'fixtures/simulator_failures' /
           f'ngspice_{case}.txt').read_text()
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(tmp_path, log=log, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    message = str(error.value)
    assert message.index('Error on line') < message.index(card) < message.index(reason)
    assert message.index(reason) < message.index('Simulation interrupted')


def test_nonzero_exit_keeps_order_without_duplicate_context_and_bounds_detail(tmp_path):
    log = ('Error on line:\n\n  r1 out 0 1k\nreason one\n'
           'Error on line 9:\n  d1 out 0 bad\nreason two\n' +
           'Error: repeated diagnostic\n' * 1000)
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(tmp_path, log=log, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    detail = str(error.value).split('\n  ', 1)[1].rsplit('\nSee ', 1)[0]
    assert detail.index('r1 out') < detail.index('reason one') < detail.index('d1 out') < detail.index('reason two')
    assert detail.count('Error on line 9:') == 1
    assert len(detail) <= 4000
    assert 'truncated' in detail


def test_nonzero_exit_long_card_leaves_room_for_reason(tmp_path):
    log = 'Error on line:\n  r1 ' + 'x' * 10000 + '\ndevice already exists, bail out\n'
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(tmp_path, log=log, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    assert 'r1 ' in str(error.value)
    assert 'device already exists, bail out' in str(error.value)
    assert len(str(error.value)) < 4500


def test_timeout_message_unchanged(tmp_path):
    import subprocess
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path), timeout=2)
    with patch('subprocess.run', side_effect=subprocess.TimeoutExpired('ngspice', 2)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    assert str(error.value) == f'ngspice simulation timed out after 2s. Check {runner.logfile_path} for progress.'


@pytest.mark.parametrize('log,expected', [
    ('', '(no error lines found)'),
    ('Error on line:', 'Error on line:'),
    ('FATAL: stopped\n  card\nreason', 'FATAL: stopped\n    card\n  reason'),
    ('unrecognized failure', 'unrecognized failure'),
])
def test_nonzero_exit_short_or_unrecognized_log(tmp_path, log, expected):
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(tmp_path, log=log, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    assert str(error.value).split('\n  ', 1)[1].rsplit('\nSee ', 1)[0] == expected


def test_nonzero_exit_unrecognized_long_log_is_bounded(tmp_path):
    log = 'progress\n' * 1000 + 'unrecognized failure'
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(tmp_path, log=log, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    detail = str(error.value).split('\n  ', 1)[1].rsplit('\nSee ', 1)[0]
    assert len(detail) <= 4000
    assert 'truncated' in detail
    assert detail.endswith('unrecognized failure')


def test_diagnostic_log_does_not_change_success_verdict(tmp_path):
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(
            tmp_path, log='Error on line:\ncard\nreason', raw='current output')):
        runner.run()


@pytest.mark.parametrize('incomplete_prefix', ['', 'Error on line:\ncard-with-no-reason\n'])
def test_detail_budget_never_leaves_a_partial_card_diagnostic(tmp_path, incomplete_prefix):
    earlier = ('Error: ' + 'x' * 493 + '\n') * 7
    block = 'Error on line 99:\n  r1 ' + 'y' * 495 + '\ndevice already exists, bail out\n'
    runner = NgspiceRunner(str(tmp_path / 'test.cir'), str(tmp_path))
    with patch('subprocess.run', side_effect=_launch_writes(tmp_path, log=earlier + incomplete_prefix + block, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    detail = str(error.value).split('\n  ', 1)[1].rsplit('\nSee ', 1)[0]
    assert len(detail) <= 4000
    assert 'truncated' in detail
    assert 'Error on line 99:' not in detail or 'device already exists, bail out' in detail
    assert 'Error: ' in detail


@pytest.mark.parametrize('prefix', [
    'Error on line:\n', 'Error on line:\ncard-with-no-reason\n',
    'Netlist error in file\n /tmp/incomplete-location\n',
    'Netlist error in file x.cir at or near line 1\n',
])
@pytest.mark.parametrize('header', ['Error on line 9:', 'Netlist error in file x.cir at or near line 9'])
def test_incomplete_diagnostic_stops_at_next_header(prefix, header):
    from silk.simulation._failure_detail import failure_detail
    context = ('  r9 out 0 1k\nreason-for-r9' if header.startswith('Error')
               else 'Duplicate device R9')
    text = prefix + header + '\n' + context + '\n'
    detail = failure_detail(text)
    assert detail == '\n  '.join(text.rstrip().splitlines())


@pytest.mark.parametrize('text', ['Error on line:', 'Error on line:\n  r1 out 0 1k',
                                  'Netlist error in file\n /tmp/incomplete'])
def test_eof_incomplete_diagnostic_is_preserved_without_invented_context(text):
    from silk.simulation._failure_detail import failure_detail
    assert failure_detail(text) == text.replace('\n', '\n  ')


def test_original_long_log_path_does_not_push_reason_out_of_short_display(tmp_path):
    from pathlib import Path
    directory = tmp_path / 'synthetic-long-project-work-directory' / '2f5974e2-4a1d-4bcd-bf90-f49364714f48' / 'scratchpad/failsafe/plan1/run/t3-stage/measure-twin-run'
    directory.mkdir(parents=True)
    log = (Path(__file__).parent / 'fixtures/simulator_failures/ngspice_original_duplicate.txt').read_text()
    (directory / 'test_twin_div.cir').write_text('Diagnostic fixture\n.end\n')
    runner = NgspiceRunner(str(directory / 'test_twin_div.cir'), str(directory))
    with patch('subprocess.run', side_effect=_launch_writes(directory, log=log, returncode=1)):
        with pytest.raises(SimulationError) as error:
            runner.run()
    message = str(error.value)
    displayed = ' '.join(message.split())
    assert 'r.xdut.rr1 out 0 1k' in displayed[:240]
    assert 'device already exists, bail out' in displayed[:240]
    assert message.index('device already exists, bail out') < message.index('See ')
    assert message.endswith(f'See {runner.logfile_path} for full output.')
