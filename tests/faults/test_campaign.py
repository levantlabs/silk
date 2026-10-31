"""FaultCampaign orchestration + CampaignResult accessors (Task 8;
TASK-301.6 moved orchestration onto the shared silk.simulation.executor)."""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.faults import FaultCampaign, FaultSet, stuck_open
from silk.faults.campaign import CampaignResult, RunResult, _derive_run_seed
from silk.faults.types import AppliedFault, FaultManifest


def _mock_outcome(spec_name: str, status: str, value: float = 0.0):
    o = MagicMock()
    o.spec.name = spec_name
    o.spec.key = spec_name
    o.owner_path = '$'
    o.status = status
    o.measurement.value = value
    return o


def _manifest(seed: int, applied=None):
    return FaultManifest(seed=seed, input_spec={}, applied=applied or [])


def _run(run_id: int, status: str, spec_outcomes=None, applied=None):
    return RunResult(
        run_id=run_id, seed=run_id,
        status=status,
        manifest=_manifest(run_id, applied),
        spec_outcomes=spec_outcomes or [],
    )


# --- CampaignResult accessors ---

def test_pass_fraction_all_pass():
    runs = [_run(i, 'passed',
                 [_mock_outcome('adc_sndr', 'PASS', 25.0)])
            for i in range(10)]
    result = CampaignResult(
        campaign_id='x', module_name='adc', n_runs=10,
        fault_set=FaultSet(), seed=0, runs=runs, out_dir='/tmp')
    assert result.pass_fraction('adc_sndr') == 1.0


def test_pass_fraction_mixed():
    runs = [_run(i, 'passed' if i < 8 else 'failed',
                 [_mock_outcome('adc_sndr', 'PASS' if i < 8 else 'FAIL',
                                25.0 if i < 8 else 15.0)])
            for i in range(10)]
    result = CampaignResult('x', 'adc', 10, FaultSet(), 0, runs, '/tmp')
    assert result.pass_fraction('adc_sndr') == pytest.approx(0.8)


def test_measurement_distribution_returns_array():
    runs = [_run(i, 'passed',
                 [_mock_outcome('adc_sndr', 'PASS', 20.0 + i)])
            for i in range(5)]
    result = CampaignResult('x', 'adc', 5, FaultSet(), 0, runs, '/tmp')
    dist = result.measurement_distribution('adc_sndr')
    assert isinstance(dist, np.ndarray)
    assert len(dist) == 5
    assert dist.min() == 20.0 and dist.max() == 24.0


def test_runs_failing_returns_only_failures():
    runs = [_run(0, 'passed',
                 [_mock_outcome('s', 'PASS', 1.0)]),
            _run(1, 'failed',
                 [_mock_outcome('s', 'FAIL', 0.0)]),
            _run(2, 'errored', [])]
    result = CampaignResult('x', 'adc', 3, FaultSet(), 0, runs, '/tmp')
    failing = result.runs_failing('s')
    assert len(failing) == 1
    assert failing[0].run_id == 1


def test_correlate_faults_with_failure_signals_real_correlation():
    def run_with_fault(run_id, has_fault, spec_fail):
        applied = ([AppliedFault(
            target='xcomp1.mn_tail', fault_type='stuck_open',
            declared_params={}, realized_params={},
            netlist_comment='',
        )] if has_fault else [])
        status = 'failed' if spec_fail else 'passed'
        out = _mock_outcome('s', 'FAIL' if spec_fail else 'PASS', 0.0)
        return _run(run_id, status, [out], applied)

    runs = [run_with_fault(i, has_fault=(i < 5), spec_fail=(i < 5))
            for i in range(10)]
    result = CampaignResult('x', 'adc', 10, FaultSet(), 0, runs, '/tmp')
    corr = result.correlate_faults_with_failure('s')
    assert len(corr) >= 1
    key = next(k for k in corr if 'stuck_open' in k and 'mn_tail' in k)
    assert corr[key] >= 0.9


# --- FaultCampaign orchestration ---

@pytest.fixture
def mini_module(tmp_dir, pdk):
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
    from silk.simulation.analysis import Transient
    top.add_sim_type(Transient(step=1e-9, stop=1e-7))
    return top


def test_faultcampaign_rejects_module_without_analysis(tmp_dir, pdk, tmp_path):
    """A module with no add_sim_type must fail loudly at construction —
    otherwise every run emits a netlist with no analysis command and dies
    N times with 'no data rows' (the flash-ADC smoke-test regression)."""
    top = Module('PDK_skywater130', tmp_dir, 'top.cir', pdk_details=pdk)
    with pytest.raises(ValueError, match='add_sim_type'):
        FaultCampaign(top, FaultSet(), n_runs=1,
                      out_dir=str(tmp_path / 'campaign'))


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_faultcampaign_n_workers_1_runs_n_times(mini_module, tmp_path, mode):
    mini_module.set_output_options(mode=mode)
    fs = FaultSet([stuck_open('xu1.mn1')])
    campaign = FaultCampaign(
        mini_module, fs, n_runs=3, seed=0,
        out_dir=str(tmp_path / 'camp'),
        n_workers=1, archive_waveforms=False,
        reconcile_specs=False, progress=False,
    )
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.return_value = {'rawfile_path': None, 'status': 'ok'}
        result = campaign.run()
    assert len(result.runs) == 3
    assert all(r.run_id in (0, 1, 2) for r in result.runs)
    for i in range(3):
        run_dir = tmp_path / 'camp' / 'runs' / f'run_{i:04d}'
        decks = list(run_dir.glob('run-*/netlist.cir' if mode == 'new' else 'netlist.cir'))
        assert len(decks) == 1
        assert (run_dir / 'manifest.json').is_file()
        assert (run_dir / 'outcome.json').is_file()


def test_faultcampaign_seed_derivation_stable():
    assert _derive_run_seed(42, 7) == _derive_run_seed(42, 7)
    assert _derive_run_seed(42, 7) != _derive_run_seed(43, 7)


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_faultcampaign_failed_run_caught_not_crashed(mini_module, tmp_path, mode):
    mini_module.set_output_options(mode=mode)
    fs = FaultSet()
    campaign = FaultCampaign(
        mini_module, fs, n_runs=2, seed=0,
        out_dir=str(tmp_path / 'camp'),
        n_workers=1, archive_waveforms=False,
        reconcile_specs=False, progress=False,
    )
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.side_effect = RuntimeError('boom')
        result = campaign.run()
    assert len(result.runs) == 2
    assert all(r.status == 'errored' for r in result.runs)
    assert all('boom' in r.error for r in result.runs)


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_faultcampaign_resume_skips_completed_runs(mini_module, tmp_path, mode):
    mini_module.set_output_options(mode=mode)
    fs = FaultSet()
    campaign = FaultCampaign(
        mini_module, fs, n_runs=3, seed=0,
        out_dir=str(tmp_path / 'camp'),
        n_workers=1, archive_waveforms=False,
        reconcile_specs=False, progress=False,
    )
    kept = _run(0, 'simulated', [])
    kept.evaluation = 'execution-only'
    prior = CampaignResult(
        'campaign_0', 'mod', 3, FaultSet(), 0,
        runs=[kept], out_dir=str(tmp_path / 'camp'))
    with patch('silk.faults.runner._run_ngspice') as mock_ng:
        mock_ng.return_value = {'rawfile_path': None, 'status': 'ok'}
        result = campaign.resume(prior)
    assert len(result.runs) == 3
    assert any(r.run_id == 0 and r.status == 'simulated' for r in result.runs)
    assert mock_ng.call_count == 2
    assert not (tmp_path / 'camp' / 'runs' / 'run_0000').exists()
    assert {r.seed for r in result.runs if r.run_id != 0} == {
        _derive_run_seed(0, 1), _derive_run_seed(0, 2)}


# --- TASK-301.6: shared executor, serial default, n_workers spellings ---

def test_faultcampaign_default_n_workers_is_serial(mini_module, tmp_path):
    """No n_workers kwarg at all — BREAKING default change from the old
    os.cpu_count() — must resolve to 1 (serial, in-process) at run()."""
    fs = FaultSet([stuck_open('xu1.mn1')])
    campaign = FaultCampaign(
        mini_module, fs, n_runs=2, seed=0,
        out_dir=str(tmp_path / 'camp'),
        archive_waveforms=False, reconcile_specs=False, progress=False,
    )
    assert campaign.n_workers is None
    with patch('silk.faults.campaign.run_tasks') as mock_run_tasks:
        mock_run_tasks.return_value = []
        campaign._execute(missing_ids=[])
    assert mock_run_tasks.call_args.kwargs['n_workers'] == 1


@pytest.mark.parametrize('requested, expected', [
    ('all', os.cpu_count() or 1),
    (1000000, os.cpu_count() or 1),
    (1, 1),
])
def test_faultcampaign_n_workers_spellings_resolve(
        mini_module, tmp_path, requested, expected):
    fs = FaultSet([stuck_open('xu1.mn1')])
    campaign = FaultCampaign(
        mini_module, fs, n_runs=2, seed=0,
        out_dir=str(tmp_path / 'camp'), n_workers=requested,
        archive_waveforms=False, reconcile_specs=False, progress=False,
    )
    with patch('silk.faults.campaign.run_tasks') as mock_run_tasks:
        mock_run_tasks.return_value = []
        campaign._execute(missing_ids=[])
    assert mock_run_tasks.call_args.kwargs['n_workers'] == expected


def test_faultcampaign_delegates_to_shared_executor(mini_module, tmp_path):
    """FaultCampaign._execute builds silk.simulation.executor.RunTask specs
    and calls run_tasks — the point of TASK-301.6. run_tasks' own
    KeyboardInterrupt/pool-termination behaviour is exercised generically by
    tests/test_run_executor.py::TestKeyboardInterruptCleanup; that test's
    tracked-child cleanup only reaches FaultCampaign's ngspice children once
    TASK-301.3 switches silk/simulation/runner.py to launch simulators via
    spawn_tracked() — until then, Ctrl-C here still stops only the pool
    workers, not their ngspice children."""
    from silk.simulation.executor import RunTask

    fs = FaultSet([stuck_open('xu1.mn1')])
    campaign = FaultCampaign(
        mini_module, fs, n_runs=3, seed=5,
        out_dir=str(tmp_path / 'camp'), n_workers=1,
        archive_waveforms=False, reconcile_specs=False, progress=False,
    )
    with patch('silk.faults.campaign.run_tasks') as mock_run_tasks:
        mock_run_tasks.return_value = []
        campaign._execute(missing_ids=[0, 1, 2])
    (tasks_arg,), kwargs = mock_run_tasks.call_args
    assert len(tasks_arg) == 3
    for run_id, task in enumerate(tasks_arg):
        assert isinstance(task, RunTask)
        assert task.run_dir == str(
            tmp_path / 'camp' / 'runs' / f'run_{run_id:04d}')
    assert kwargs['n_workers'] == 1


@pytest.mark.parametrize('mode', ['overwrite', 'new'])
@pytest.mark.parametrize('random_faults', [False, True])
def test_faultcampaign_serial_and_pooled_produce_identical_results(
        mini_module, tmp_path, mode, random_faults):
    """AC#1: same seed, n_workers=1 vs n_workers=2 must produce identical
    run folders, manifests, outcomes and summary.json — the shared executor
    only changes scheduling, never results."""
    mini_module.set_output_options(mode=mode)
    fs = FaultSet([stuck_open('xu1.mn1')])
    if random_faults:
        from silk.faults import w_variation
        pdk = mini_module.pdk_details
        mini_module.add(pdk.nmos1p8('mn2', d='d1', g='g1', s='gnd', b='gnd', w=2, l=.18))
        fs = FaultSet([w_variation('xu1.mn1', sigma=.1),
                       w_variation('mn2', sigma=.2), stuck_open('xu1.mn1')])
        fs.sprinkle(r'.*', w_variation(sigma=.3), rate=.5, seed=17)
        fs.sprinkle(r'.*', w_variation(sigma=.4), rate=.5)

    def _run(n_workers, out_dir):
        campaign = FaultCampaign(
            mini_module, fs, n_runs=4, seed=7,
            out_dir=str(out_dir), n_workers=n_workers,
            archive_waveforms=False, reconcile_specs=False, progress=False,
        )
        with patch('silk.faults.runner._run_ngspice') as mock_ng:
            mock_ng.return_value = {'rawfile_path': None, 'status': 'ok'}
            return campaign.run()

    serial = _run(1, tmp_path / 'serial')
    pooled = _run(2, tmp_path / 'pooled')

    assert len(serial.runs) == len(pooled.runs) == 4
    serial_by_id = {r.run_id: r for r in serial.runs}
    pooled_by_id = {r.run_id: r for r in pooled.runs}
    assert set(serial_by_id) == set(pooled_by_id)
    for run_id in serial_by_id:
        s, p = serial_by_id[run_id], pooled_by_id[run_id]
        assert s.error is None, s.error
        assert p.error is None, p.error
        assert [r.target for r in s.manifest.applied] == (
            ['xu1.mn1', 'mn2'] if random_faults else ['xu1.mn1'])
        assert s.seed == p.seed
        assert s.status == p.status
        assert s.manifest.to_json() == p.manifest.to_json()
        assert s.rawfile_path == p.rawfile_path

    for run_id in range(4):
        run_name = f'run_{run_id:04d}'
        for fname in ('manifest.json', 'outcome.json'):
            s_text = (tmp_path / 'serial' / 'runs' / run_name / fname).read_text()
            p_text = (tmp_path / 'pooled' / 'runs' / run_name / fname).read_text()
            assert s_text == p_text, f'{fname} differs for {run_name}'

    serial_summary = json.loads((tmp_path / 'serial' / 'summary.json').read_text())
    pooled_summary = json.loads((tmp_path / 'pooled' / 'summary.json').read_text())
    assert serial_summary == pooled_summary


def test_successive_campaigns_keep_only_requested_faults_and_authored_edits(
        mini_module, tmp_path, monkeypatch):
    from pathlib import Path
    import silk.simulation.completion as completion
    from silk.faults import short_gd

    module = mini_module
    module.set_output_options(mode='overwrite')
    target = 'xu1.mn1'
    device = module._silk_resolve_fault_target(target)
    attrs = ('_silk_pending_fault', '_silk_fault_rng', '_silk_fault_sink',
             '_silk_fault_realization')
    original_params = dict(device.params)
    original_nets = dict(device.nets)
    module.circuit_name = 'successive campaigns'
    module.generate_netlist()
    baseline = Path(module._run_files.netlist_path).read_bytes()
    with pytest.raises(KeyError):
        module.generate_netlist(fault_set=FaultSet([short_gd(target), stuck_open('missing')]))

    decks = []
    real_certify = completion.certify_generated
    emissions = []
    fail_emission = {2}
    fail_backend = {2}

    def certify(mod, text):
        emissions.append(text)
        if len(emissions) - 1 in fail_emission:
            raise RuntimeError('post-emission sentinel')
        return real_certify(mod, text)

    def backend(netlist_path, run_dir, selected_backend, *, run_files):
        text = Path(netlist_path).read_text()
        decks.append(text)
        if len(decks) - 1 in fail_backend:
            raise RuntimeError('backend sentinel')
        return {'rawfile_path': None, 'status': 'ok'}

    monkeypatch.setattr(completion, 'certify_generated', certify)
    monkeypatch.setattr('silk.faults.runner._run_ngspice', backend)

    def campaign(name, faults, n_runs):
        return FaultCampaign(
            module, faults, n_runs=n_runs, seed=37,
            out_dir=str(tmp_path / name), n_workers=1,
            archive_waveforms=False, reconcile_specs=False, progress=False).run()

    nominal = campaign('nominal-after-failed-preparation', FaultSet(), 1)
    assert nominal.runs[0].status == 'simulated'
    assert 'FAULT:' not in decks[0]
    assert nominal.runs[0].manifest.applied == []
    first = campaign('first', FaultSet([stuck_open(target)]), 4)
    assert [run.status for run in first.runs] == ['simulated', 'errored', 'errored', 'simulated']
    assert 'post-emission sentinel' in first.runs[1].error
    assert first.runs[1].manifest.applied == []
    assert 'backend sentinel' in first.runs[2].error
    assert first.runs[2].manifest.applied
    assert all(run.manifest.applied for run in (first.runs[0], first.runs[3]))
    assert len(decks) == 4
    assert all('FAULT: stuck_open on xu1.mn1' in deck and 'short_gd' not in deck
               for deck in decks[1:])
    assert not any(hasattr(device, attr) for attr in attrs)
    assert dict(device.params) == original_params
    assert dict(device.nets) == original_nets
    module.generate_netlist()
    assert Path(module._run_files.netlist_path).read_bytes() == baseline

    device.w = 3.0
    second = campaign('second', FaultSet([short_gd(target)]), 1)
    third = campaign('third', FaultSet(), 1)
    assert second.runs[0].status == third.runs[0].status == 'simulated'
    assert 'FAULT: short_gd on xu1.mn1' in decks[-2]
    assert 'stuck_open' not in decks[-2]
    assert 'FAULT:' not in decks[-1]
    assert 'w=3 ' in decks[-2] and 'w=3 ' in decks[-1]
    assert device.w == 3.0
    assert dict(device.nets) == original_nets
    assert not any(hasattr(device, attr) for attr in attrs)
