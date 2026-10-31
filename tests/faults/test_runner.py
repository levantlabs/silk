"""Tests for silk.faults.runner.run_one (Task 7).

``run_one`` is the per-run worker invoked by ``FaultCampaign``'s
multiprocessing pool. These tests mock out ``_run_ngspice`` and
``_reconcile_specs`` so they run in milliseconds and don't require a
real simulator.
"""

import pickle

import pytest

from tests._mock_rawfile import write_complete_rawfile
from unittest.mock import patch

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.faults import FaultSet, stuck_open
from silk.faults.campaign import RunResult
from silk.faults.runner import run_one, _run_ngspice


@pytest.fixture
def fake_module(tmp_dir, pdk):
    sub = Module('PDK_skywater130', tmp_dir, 'sub.cir',
                 pdk_details=pdk, module_name='sub')
    sub.define_module_nets({'d': PortDir.INOUT, 'g': PortDir.INOUT,
                            's': PortDir.INOUT, 'b': PortDir.INOUT})
    with sub:
        pdk.nmos1p8('mn1', d='d', g='g', s='s', b='b',
                    w=1.0, l=0.18, m=1)
    top = Module('PDK_skywater130', tmp_dir, 'top.cir',
                 pdk_details=pdk)
    for n in ('g1', 'd1', 'gnd'):
        top.add_net(n)
    top.add_module(sub, 'u1',
                    nets={'g': 'g1', 'd': 'd1', 's': 'gnd', 'b': 'gnd'})
    return top


def test_run_one_produces_runresult_with_status(fake_module, tmp_path):
    fs = FaultSet([stuck_open('xu1.mn1')])
    run_dir = tmp_path / 'run_0000'
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.return_value = {
            'rawfile_path': str(run_dir / 'rawfile.raw'),
            'status': 'ok',
        }
        result = run_one(
            module_pickle=pickle.dumps(fake_module),
            fault_set=fs,
            seed=42,
            run_id=0,
            run_dir=str(run_dir),
            archive_waveforms=False,
            reconcile_specs=True,
        )
    assert isinstance(result, RunResult)
    assert result.run_id == 0
    assert result.seed == 42
    # The worker only simulates; the parent scores the rawfile.
    assert result.status == 'simulated', result.error
    assert result.spec_outcomes == []
    assert result.backend == 'ngspice'
    assert result.requested_corner == 'tt'
    assert result.no_pdk_device is False


def test_run_one_writes_netlist_and_manifest(fake_module, tmp_path):
    fake_module.set_output_options(mode='overwrite')
    fs = FaultSet([stuck_open('xu1.mn1')])
    run_dir = tmp_path / 'run_0000'
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.return_value = {'rawfile_path': None, 'status': 'ok'}
        run_one(
            module_pickle=pickle.dumps(fake_module),
            fault_set=fs, seed=1, run_id=0,
            run_dir=str(run_dir),
            archive_waveforms=False, reconcile_specs=False,
        )
    assert (run_dir / 'netlist.cir').exists()
    assert (run_dir / 'manifest.json').exists()


def test_run_one_catches_ngspice_failure_produces_errored_status(
        fake_module, tmp_path):
    fs = FaultSet()
    run_dir = tmp_path / 'run_0000'
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.side_effect = RuntimeError('ngspice crashed')
        result = run_one(
            module_pickle=pickle.dumps(fake_module),
            fault_set=fs, seed=1, run_id=0,
            run_dir=str(run_dir),
            archive_waveforms=False, reconcile_specs=False,
        )
    assert result.status == 'errored'
    assert 'crashed' in result.error


def test_run_one_leaves_rawfile_for_the_parent_to_score(fake_module, tmp_path):
    fs = FaultSet()
    run_dir = tmp_path / 'run_0000'
    run_dir.mkdir()
    (run_dir / 'rawfile.raw').write_bytes(b'dummy')
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.return_value = {
            'rawfile_path': str(run_dir / 'rawfile.raw'), 'status': 'ok'}
        result = run_one(
            module_pickle=pickle.dumps(fake_module),
            fault_set=fs, seed=1, run_id=0,
            run_dir=str(run_dir),
            archive_waveforms=False, reconcile_specs=False,
        )
    # Deleting before evaluation would destroy the evidence being scored.
    assert (run_dir / 'rawfile.raw').exists()
    assert result.rawfile_path == str(run_dir / 'rawfile.raw')


def test_run_ngspice_uses_backend_runner(tmp_path):
    """_run_ngspice must run via backend.run() -> RawResult, not by importing
    a runner directly. The rawfile path comes from RawResult.source_path."""
    from unittest.mock import MagicMock
    from silk.simulation.backend import RawResult

    fake_rawfile = str(tmp_path / 'sim.raw')
    fake_backend = MagicMock()
    fake_backend.run.return_value = RawResult(native='RAW', source_path=fake_rawfile)

    result = _run_ngspice(
        netlist_path=str(tmp_path / 'netlist.cir'),
        run_dir=str(tmp_path),
        backend=fake_backend,
    )

    fake_backend.run.assert_called_once_with(
        netlist_path=str(tmp_path / 'netlist.cir'),
        results_dir=str(tmp_path),
        debug=False,
    )
    assert result == {'rawfile_path': fake_rawfile, 'status': 'ok'}


def test_run_one_keeps_rawfile_when_archive_true(fake_module, tmp_path):
    fs = FaultSet()
    run_dir = tmp_path / 'run_0000'
    run_dir.mkdir()
    (run_dir / 'rawfile.raw').write_bytes(b'dummy')
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.return_value = {
            'rawfile_path': str(run_dir / 'rawfile.raw'), 'status': 'ok'}
        result = run_one(
            module_pickle=pickle.dumps(fake_module),
            fault_set=fs, seed=1, run_id=0,
            run_dir=str(run_dir),
            archive_waveforms=True, reconcile_specs=False,
        )
    assert (run_dir / 'rawfile.raw').exists()
    assert result.rawfile_path is not None


@pytest.mark.parametrize('archive_waveforms', [False, True])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_worker_launches_shared_files_without_replacing_faulted_deck(
        fake_module, tmp_path, monkeypatch, archive_waveforms, mode, mock_ngspice_version):
    from dataclasses import replace
    import json
    from pathlib import Path
    import subprocess

    from silk.simulation.backend import NgspiceBackend

    # A pre-generated parent may already carry prepared paths when pickled.
    fake_module.set_output_options(mode=mode)
    fake_module.generate_netlist()
    original = fake_module._run_files
    original_deck = Path(original.netlist_path).read_bytes()
    run_dir = tmp_path / 'worker'
    expected_raw = run_dir / 'worker.rawfile'
    expected_log = run_dir / 'worker.log'
    resolve = NgspiceBackend._resolve_run_files
    prepared = []
    launched = []

    def worker_files(self, netlist_path, results_dir, **kwargs):
        if results_dir == '/__silk_execution__':
            return resolve(self, netlist_path, results_dir, **kwargs)
        files = replace(resolve(self, netlist_path, results_dir, **kwargs),
                        rawfile_path=str(expected_raw), logfile_path=str(expected_log))
        prepared.append(files)
        return files

    launch = NgspiceBackend._launch_prepared

    def launch_worker(self, files, **kwargs):
        assert files.netlist_path.endswith('/netlist.cir')
        if mode == 'overwrite':
            # Generation enriches the immutable paths with completion provenance.
            assert replace(files, completion=None, completion_unverified_reason=None) == prepared[-1]
        else:
            assert Path(files.directory).parent == run_dir
        launched.append(files)
        return launch(self, files, **kwargs)

    def fixture_run(argv, **kwargs):
        expected_raw = Path(launched[-1].rawfile_path)
        expected_log = Path(launched[-1].logfile_path)
        deck = Path(argv[-1]).read_text()
        assert 'FAULT: stuck_open on xu1.mn1' in deck
        assert f'write {expected_raw}\n' in deck
        assert str(expected_raw) == argv[argv.index('-r') + 1]
        assert str(expected_log) == argv[argv.index('-o') + 1]
        assert any(line.startswith('.lib ') and line.endswith(' tt')
                   for line in deck.splitlines())
        fixture = Path(__file__).parents[1] / 'fixtures/rawfiles/e_ng_tran.raw'
        write_complete_rawfile(expected_raw, deck, template=fixture)
        expected_log.write_text('fixture worker log\n')
        return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

    monkeypatch.setattr(NgspiceBackend, '_resolve_run_files', worker_files)
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', launch_worker)
    monkeypatch.setattr(subprocess, 'run', fixture_run)
    result = run_one(
        pickle.dumps(fake_module), FaultSet([stuck_open('xu1.mn1')]),
        seed=37, run_id=9, run_dir=str(run_dir),
        archive_waveforms=archive_waveforms, reconcile_specs=False,
    )
    assert result.status == 'simulated', result.error
    expected_raw = Path(launched[-1].rawfile_path)
    expected_log = Path(launched[-1].logfile_path)
    assert (result.run_id, result.seed) == (9, 37)
    assert len(prepared) == len(launched) == 1
    # The worker never deletes: the parent scores the rawfile first.
    assert result.rawfile_path == str(expected_raw)
    assert expected_raw.exists()
    assert expected_log.is_file()
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    assert manifest['seed'] == 37
    assert manifest['applied']
    assert not (run_dir / 'outcome.json').exists()  # written by the parent
    assert fake_module._run_files is original
    assert Path(original.netlist_path).read_bytes() == original_deck


def test_outcome_json_retains_placement_identity():
    from silk.specs import Spec, Measurement, SpecOutcome
    from silk.faults.runner import _outcome_to_json
    rows = []
    for owner in ('$/x1', '$/x2', '$/x1'):
        outcome = SpecOutcome(Spec('Gain', None, 'gain', '>=', 60, key='stable'),
                              Measurement('user', 70), 'PASS')
        outcome.owner_path = owner
        outcome.report_key = owner + ':leaf.gain'
        rows.append(_outcome_to_json(outcome))
    assert [r['owner_path'] for r in rows] == ['$/x1', '$/x2', '$/x1']
    assert [r['spec_key'] for r in rows] == ['stable'] * 3
    assert rows[0]['report_key'] == '$/x1:leaf.gain'
    assert rows[0] == rows[2]


def test_outcome_json_unknown_identity_is_not_inferred():
    from types import SimpleNamespace
    from silk.faults.runner import _outcome_to_json
    result = _outcome_to_json(SimpleNamespace(spec=SimpleNamespace(name='legacy')))
    assert result['owner_path'] is None
    assert result['report_key'] is None
    assert result['spec_key'] is None


def test_worker_invalid_numeric_fault_is_errored_without_launch(fake_module, tmp_path):
    from silk.faults import w_variation
    with patch('silk.faults.runner._run_ngspice') as launch:
        result = run_one(
            module_pickle=pickle.dumps(fake_module),
            fault_set=FaultSet([w_variation('xu1.mn1', sigma=float('nan'))]),
            seed=1, run_id=0, run_dir=str(tmp_path / 'invalid'),
            archive_waveforms=False, reconcile_specs=False)
    assert result.status == 'errored'
    assert 'sigma' in result.error
    launch.assert_not_called()
