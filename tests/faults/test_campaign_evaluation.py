"""Fault campaigns score every run from that run's own recorded output.

Simulator-free: only ``backend.run`` is replaced (it copies a recorded rawfile
into the run directory and parses it); reconciliation, evaluation and
reporting run for real, through ``FaultCampaign.run()``/``resume()``.
"""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import MetricBindingError
from silk.faults import FaultCampaign, FaultSet
from silk.metrics.waveform import final_value
from silk.simulation import Transient
from silk.simulation.backend import SimulatorBackend
from silk.specs.spec import Measurement
from tests._env import needs_ngspice, needs_pdk

RAWS = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'rawfiles')
A_TRAN = os.path.join(RAWS, 'a_tran_ascii.raw')
OTHER = os.path.join(RAWS, 'e_ng_tran.raw')  # a different recorded v(out)
FRESH = 0.9914561752448197


def _final_out(path, simulator='ngspice'):
    raw = SimulatorBackend.get(simulator).load_raw(path, simulator=simulator)
    (vec,) = [v for v in raw.native.vectors if v.name.lower() in ('v(out)', 'out')]
    return float(vec.data[-1])


@pytest.fixture
def launch(monkeypatch):
    """Replace only the backend launch: copy a recorded rawfile per run."""
    state = {'source': {None: A_TRAN}, 'launches': []}

    def fake_run(self, netlist_path, results_dir, **kwargs):
        run = next((part for part in reversed(Path(results_dir).parts)
                    if part.startswith('run_')), Path(results_dir).name)
        src = state['source'].get(run, state['source'][None])
        dst = os.path.join(results_dir, 'recorded.raw')
        shutil.copy(src, dst)
        state['launches'].append(run)
        return self.load_raw(dst, simulator=self.name)

    real_popen = subprocess.Popen

    def no_subprocess(argv, *a, **k):
        # Version probes are not simulations; anything else is.
        if list(argv)[-1:] == ['--version']:
            return real_popen(argv, *a, **k)
        raise AssertionError('a simulator subprocess was launched')

    for name in ('ngspice', 'xyce'):
        monkeypatch.setattr(type(SimulatorBackend.get(name)), 'run', fake_run)
    monkeypatch.setattr(subprocess, 'Popen', no_subprocess)
    return state


def _module(tmp_path, ideal=True):
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'rc.cir', pdk_details=pdk, module_name='rc')
    dut.define_module_ports({'out': 'output', 'vss': 'ground'})
    dut.add(elements.resistor('r1', p='out', n='vss', value=1e3))
    return dut


def _top(tmp_path):
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.add_source(sources.VoltageSource('v1', positive='out', negative='0', dc=1.0))
    top.add(elements.resistor('r1', p='out', n='0', value=1e3))
    top.add_sim_type(Transient(step=1e-9, stop=40e-9))
    return top


def _bench(tmp_path, *, op='>=', target=2.0, name='resp', extractor=None):
    dut = _module(tmp_path)
    tb = dut.create_test(name)
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-9, stop=40e-9))
    calls = []

    def final_vout(ckt):
        calls.append(1)
        return final_value(ckt.load_signal('out'))

    tb.add_metric('final_vout', extractor(calls) if extractor else final_vout)
    spec = tb.add_spec('vout check', 'final_vout', op, target)
    return dut, tb, spec, calls


def _campaign(module, tmp_path, *, n_runs=2, archive=True, reconcile=True,
              n_workers=None, fault_set=None):
    return FaultCampaign(
        module, fault_set or FaultSet(), n_runs=n_runs, seed=7,
        out_dir=str(tmp_path / 'campaign'), n_workers=n_workers,
        archive_waveforms=archive, reconcile_specs=reconcile, progress=False)


# --- the confirmed reproductions ------------------------------------------

def test_fresh_failing_output_fails_with_the_fresh_value(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path)
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['failed', 'failed']
    assert len(calls) == 2  # the extractor ran once per run
    for run in result.runs:
        (row,) = run.spec_outcomes
        assert (row.status, row.value) == ('FAIL', FRESH)
        assert row.attempt_kind == 'measured'
        assert run.evaluation == 'fresh-v1'
    assert list(result.measurement_distribution('vout check')) == [FRESH, FRESH]
    assert result.pass_fraction('vout check') == 0.0


def test_old_passing_value_cannot_hide_a_fresh_failure(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path)
    spec.record_measurement(Measurement('spice', 3.0, passes=True))
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['failed', 'failed']
    assert list(result.measurement_distribution('vout check')) == [FRESH, FRESH]
    assert 'pass   0.0%' in result.generate_report().summary() or \
        '0.0%' in result.generate_report().summary()


def test_old_failing_value_cannot_hide_a_fresh_pass(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    spec.record_measurement(Measurement('spice', 0.0, passes=False))
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['passed', 'passed']
    assert list(result.measurement_distribution('vout check')) == [FRESH, FRESH]


def test_caller_evidence_is_never_overwritten_or_leaked(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path)
    old = Measurement('spice', 3.0, passes=True)
    spec.record_measurement(old)
    before = list(spec.measurements.items()) if isinstance(
        spec.measurements, dict) else list(spec.measurements)
    _campaign(tb.netlist, tmp_path).run()
    after = list(spec.measurements.items()) if isinstance(
        spec.measurements, dict) else list(spec.measurements)
    assert after == before
    assert getattr(tb.netlist, '_raw', None) is None
    assert getattr(tb.netlist, 'results_file', None) is None
    assert getattr(tb.netlist, 'operating_conditions', None) is None
    assert '_raw_simulator_hint' not in tb.netlist.__dict__
    assert '_test_evidence' not in dut.__dict__
    assert tb._sweep_table == {}


def test_declared_spec_with_no_bench_is_unverified_not_passed(tmp_path, launch):
    dut = _top(tmp_path)
    spec = dut.declare_spec(name='vout check', metric='final_vout', op='>=',
                            target=2.0)
    spec.record_measurement(Measurement('spice', 3.0, passes=True))
    result = _campaign(dut, tmp_path).run()
    assert [r.status for r in result.runs] == ['unverified', 'unverified']
    for run in result.runs:
        (row,) = run.spec_outcomes
        assert (row.status, row.attempt_kind, row.value) == (
            'UNMEASURED', 'unbound', None)
        assert row.checked is False
        (excl,) = run.excluded
        assert excl['spec_name'] == 'vout check' and excl['checked'] is False
        assert 'no metric binding' in excl['reason']
    rows = json.load(open(tmp_path / 'campaign' / 'runs' / 'run_0000' / 'outcome.json'))
    assert [(r['spec_name'], r['checked'], r['attempt_kind']) for r in rows] == [
        ('vout check', False, 'unbound')]
    assert 'no metric binding' in result.generate_report().summary()
    assert len(result.measurement_distribution('vout check')) == 0
    assert math.isnan(result.pass_fraction('vout check'))
    assert 'pass' not in ''.join(
        l for l in result.generate_report().summary().splitlines()
        if l.strip().startswith('vout check'))  # no pass % line for it


def test_no_specs_at_all_is_unverified(tmp_path, launch):
    dut = _top(tmp_path)
    result = _campaign(dut, tmp_path).run()
    assert [r.status for r in result.runs] == ['unverified', 'unverified']


def test_execution_only_is_simulated_not_passed(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path)
    result = _campaign(tb.netlist, tmp_path, reconcile=False).run()
    assert [r.status for r in result.runs] == ['simulated', 'simulated']
    assert all(r.evaluation == 'execution-only' for r in result.runs)
    assert calls == []
    summary = result.generate_report().summary()
    assert 'simulated: 2' in summary and 'passed: 0' in summary


# --- binding ----------------------------------------------------------------

def test_sibling_bench_specs_are_not_checked(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    other = dut.create_test('other')
    other.add_module_under_test({'out': 'out', 'vss': '0'})
    other.add_sim_type(Transient(step=1e-9, stop=40e-9))
    other.add_metric('m', lambda ckt: 99.0)
    other.add_spec('other spec', 'm', '<', 0.0)  # would fail if it were checked
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['passed', 'passed']
    for run in result.runs:
        assert [o.spec_name for o in run.spec_outcomes] == ['vout check']
        (excl,) = run.excluded
        assert excl['spec_name'] == 'other spec' and excl['checked'] is False
        assert "owned by test 'other'" in excl['reason']
    rows = json.load(open(tmp_path / 'campaign' / 'runs' / 'run_0000' / 'outcome.json'))
    checked = [r for r in rows if r['checked']]
    assert [r['spec_name'] for r in checked] == ['vout check']
    (skipped,) = [r for r in rows if not r['checked']]
    assert skipped['spec_name'] == 'other spec'
    assert skipped['measurement_value'] is None and skipped['status'] is None
    assert "owned by test 'other'" in skipped['reason']
    report = result.generate_report()
    for text in (report.summary(), report._render_md()):
        assert 'other spec' in text and "owned by test 'other'" in text
    # not-checked rows never count as evidence
    assert list(result.measurement_distribution('vout check')) == [FRESH, FRESH]
    assert math.isnan(result.pass_fraction('other spec'))


def test_only_the_benchs_dut_placement_is_checked(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    tb.add_module(dut, 'xother', {'out': 'aux', 'vss': '0'})
    result = _campaign(tb.netlist, tmp_path).run()
    for run in result.runs:
        assert len(run.spec_outcomes) == 1
        assert run.spec_outcomes[0].owner_path == '$/dut'
        (excl,) = run.excluded
        assert excl['owner_path'] == '$/xother' and excl['checked'] is False
        assert 'not the' in excl['reason'] and "DUT placement" in excl['reason']
    assert len(calls) == 2
    rows = json.load(open(tmp_path / 'campaign' / 'runs' / 'run_0000' / 'outcome.json'))
    assert [(r['owner_path'], r['checked']) for r in rows] == [
        ('$/dut', True), ('$/xother', False)]
    assert '$/xother' in result.generate_report().summary()


# --- classification ----------------------------------------------------------

def test_extractor_error_is_a_failure_with_no_value(tmp_path, launch):
    def make(calls):
        def boom(ckt):
            calls.append(1)
            raise RuntimeError('extractor exploded')
        return boom
    dut, tb, spec, calls = _bench(tmp_path, extractor=make)
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['failed', 'failed']
    (row,) = result.runs[0].spec_outcomes
    assert (row.status, row.attempt_kind, row.value) == ('FAIL', 'extractor_error', None)
    assert 'extractor exploded' in row.reason
    assert len(result.measurement_distribution('vout check')) == 0


def test_nan_is_unavailable_failure_never_in_the_distribution(tmp_path, launch):
    dut, tb, spec, calls = _bench(
        tmp_path, extractor=lambda calls: (lambda ckt: float('nan')))
    result = _campaign(tb.netlist, tmp_path).run()
    (row,) = result.runs[0].spec_outcomes
    assert (row.status, row.attempt_kind, row.value) == ('FAIL', 'unavailable', None)
    assert len(result.measurement_distribution('vout check')) == 0


def test_infinity_is_a_real_measurement(tmp_path, launch):
    dut, tb, spec, calls = _bench(
        tmp_path, extractor=lambda calls: (lambda ckt: float('inf')))
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['passed', 'passed']
    assert list(result.measurement_distribution('vout check')) == [math.inf] * 2


def test_bad_inspectable_binding_raises_before_any_launch(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path)
    # A metric bound before a second analysis is declared is ambiguous.
    from silk.simulation import AC
    tb.add_metric('avg', type='average', signal='out')
    tb.add_spec('avg ok', 'avg', '>=', 0.0)
    tb.add_sim_type(AC(variation='dec', points=5, fstart=1.0, fstop=1e6))
    with pytest.raises(ValueError, match='more than one analysis'):
        _campaign(tb.netlist, tmp_path).run()
    assert launch['launches'] == []


# --- isolation -----------------------------------------------------------------

def test_lambda_captured_probe_reads_the_fault_run_not_nominal(tmp_path, launch):
    dut = _module(tmp_path)
    tb = dut.create_test('resp')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-9, stop=40e-9))
    vout = tb.probe('out')
    tb.add_metric('final_vout', lambda ckt: final_value(vout))
    tb.add_spec('vout check', 'final_vout', '>=', 0.5)
    tb.evaluate(rawfile=OTHER)  # nominal evidence on the authored bench
    nominal = {
        'raw': tb.netlist._raw, 'file': tb.netlist.results_file,
        'conditions': tb.netlist.operating_conditions,
        'hint': getattr(tb.netlist, '_raw_simulator_hint', None),
        'table': dict(tb._sweep_table),
        'metrics': dict(tb._metric_values),
        'deck': getattr(tb.netlist, '_prepared_deck_record', None),
    }
    result = _campaign(tb.netlist, tmp_path).run()
    assert list(result.measurement_distribution('vout check')) == [FRESH, FRESH]
    assert tb.netlist._raw is nominal['raw']
    assert tb.netlist.results_file == nominal['file']
    assert tb.netlist.operating_conditions is nominal['conditions']
    assert getattr(tb.netlist, '_raw_simulator_hint', None) == nominal['hint']
    assert tb._sweep_table == nominal['table']
    assert tb._metric_values == nominal['metrics']
    assert getattr(tb.netlist, '_prepared_deck_record', None) is nominal['deck']


def test_failed_extraction_also_leaves_the_authored_bench_alone(tmp_path, launch):
    def make(calls):
        def boom(ckt):
            raise RuntimeError('nope')
        return boom
    dut, tb, spec, calls = _bench(tmp_path, extractor=make)
    _campaign(tb.netlist, tmp_path).run()
    assert getattr(tb.netlist, '_raw', None) is None
    assert getattr(tb.netlist, 'results_file', None) is None
    assert '_test_evidence' not in dut.__dict__


def test_each_run_is_scored_from_its_own_rawfile(tmp_path, launch):
    launch['source'].update({'run_0000': A_TRAN, 'run_0001': OTHER,
                             'run_0002': A_TRAN})
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    result = _campaign(tb.netlist, tmp_path, n_runs=3).run()
    values = [r.spec_outcomes[0].value for r in result.runs]
    assert values[0] == values[2] == FRESH
    assert values[1] == pytest.approx(_final_out(OTHER))
    assert values[1] != values[0]


# --- rawfile lifetime ------------------------------------------------------------

@pytest.mark.parametrize('archive', [False, True])
def test_rawfiles_removed_only_after_evaluation(tmp_path, launch, archive):
    seen = []

    def make(calls):
        def extractor(ckt):
            seen.append(sorted(
                p for p in os.listdir(os.path.dirname(ckt.results_file))
                if p.endswith('.raw')))
            if len(seen) == 2:
                raise RuntimeError('middle run fails')
            return final_value(ckt.load_signal('out'))
        return extractor
    dut, tb, spec, calls = _bench(tmp_path, extractor=make)
    result = _campaign(tb.netlist, tmp_path, n_runs=3, archive=archive).run()
    assert all(s == ['recorded.raw'] for s in seen) and len(seen) == 3
    assert [r.status for r in result.runs] == ['failed'] * 3  # all below target
    for run in result.runs:
        if archive:
            assert run.rawfile_path and os.path.exists(run.rawfile_path)
        else:
            assert run.rawfile_path is None
    present = [bool(list((tmp_path / 'campaign' / 'runs' / f'run_000{i}').rglob('recorded.raw')))
               for i in range(3)]
    assert present == [archive] * 3


# --- statistics ------------------------------------------------------------------

def test_same_named_specs_are_separate_identities(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    tb.add_spec('vout check', 'final_vout', '>=', 2.0, key='second')
    result = _campaign(tb.netlist, tmp_path).run()
    with pytest.raises(ValueError, match='ambiguous'):
        result.pass_fraction('vout check')
    # Identity enumeration is private (owner Q31 O2: no public selector yet).
    assert not hasattr(result, 'spec_identities')
    with pytest.raises(TypeError):
        result.pass_fraction(('$/dut', 'second'))
    md = result.generate_report()._render_md()
    assert '[second]' in md or ('vout check' in md and 'second' in md)
    rep = result.generate_report()
    fractions = {label: result._pass_fraction(ident)
                 for label, ident in rep._spec_labels()}
    assert sorted(fractions.values()) == [0.0, 1.0]


def test_errored_runs_are_not_counted_as_passes(tmp_path, launch):
    from silk.faults import RunResult
    from silk.faults.types import FaultManifest, AppliedFault
    from silk.faults.evaluation import CampaignSpecOutcome

    def row(status):
        return CampaignSpecOutcome(
            owner_path='$/dut', spec_key='s', spec_name='s', test_name='t',
            run_id=0, seed=0, backend='ngspice', requested_corner='tt',
            proved_conditions=None, status=status,
            value=None if status == 'UNMEASURED' else 1.0,
            attempt_kind='measured')
    applied = FaultManifest(seed=0, input_spec={}, applied=[
        AppliedFault(fault_type='stuck_open', target='m1', declared_params={}, realized_params={}, netlist_comment='')])
    clean = FaultManifest(seed=0, input_spec={}, applied=[])
    from silk.faults import CampaignResult
    runs = [
        RunResult(0, 0, 'failed', applied, [row('FAIL')]),
        RunResult(1, 0, 'errored', applied, [], error='x'),
        RunResult(2, 0, 'passed', clean, [row('PASS')]),
        RunResult(3, 0, 'unverified', clean, [row('UNMEASURED')]),
    ]
    result = CampaignResult('c', 'top', 4, FaultSet(), 0, runs, 'out')
    assert result.pass_fraction('s') == 0.5
    assert result.correlate_faults_with_failure('s') == {'stuck_open@m1': 1.0}
    assert [r.run_id for r in result.runs_failing('s')] == [0]


# --- resume ------------------------------------------------------------------------

def test_resume_reruns_prefix_results_and_finalizes_merged(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    campaign = _campaign(tb.netlist, tmp_path, n_runs=3)
    prior = campaign.run()
    assert len(launch['launches']) == 3
    # A result recorded before fresh evaluation existed is never trusted.
    prior.runs[1].evaluation = None
    resumed = campaign.resume(prior)
    assert len(launch['launches']) == 4 and launch['launches'][-1] == 'run_0001'
    assert [r.run_id for r in resumed.runs] == [0, 1, 2]
    assert resumed.runs[1].evaluation == 'fresh-v1'
    summary = json.load(open(tmp_path / 'campaign' / 'summary.json'))
    assert summary['runs'] == [
        {'run_id': r.run_id, 'status': r.status, 'seed': r.seed,
         'evaluation': r.evaluation} for r in resumed.runs]
    assert [r['evaluation'] for r in summary['runs']] == ['fresh-v1'] * 3
    assert summary['checked_scope'] == {'test_name': 'resp', 'owner_path': '$/dut'}


def test_resume_all_kept_still_writes_the_summary(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    campaign = _campaign(tb.netlist, tmp_path, n_runs=2)
    prior = campaign.run()
    os.remove(tmp_path / 'campaign' / 'summary.json')
    resumed = campaign.resume(prior)
    assert len(launch['launches']) == 2
    assert [r.run_id for r in resumed.runs] == [0, 1]
    summary = json.load(open(tmp_path / 'campaign' / 'summary.json'))
    assert summary['runs'] == [
        {'run_id': r.run_id, 'status': r.status, 'seed': r.seed,
         'evaluation': r.evaluation} for r in resumed.runs]
    assert [r['status'] for r in summary['runs']] == ['passed', 'passed']


def test_resume_mode_mismatch_reruns(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    prior = _campaign(tb.netlist, tmp_path, n_runs=2, reconcile=False).run()
    assert all(r.status == 'simulated' for r in prior.runs)
    resumed = _campaign(tb.netlist, tmp_path, n_runs=2).resume(prior)
    assert [r.status for r in resumed.runs] == ['passed', 'passed']


# --- attribution ---------------------------------------------------------------------

def test_pdk_free_deck_labels_tt_but_proves_corner_none(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    tb.set_corner('ff')
    result = _campaign(tb.netlist, tmp_path).run()
    (row,) = result.runs[0].spec_outcomes
    assert row.requested_corner == 'tt'
    assert row.proved_conditions['corner_kind'] == 'none'
    assert row.proved_conditions['origin'] == 'external'
    assert row.proved_conditions['backend'] == 'ngspice'
    assert row.proved_conditions['temperature_kind'] == 'unknown'
    assert 'ff' not in json.dumps(row.proved_conditions)
    on_disk = json.load(open(tmp_path / 'campaign' / 'runs' / 'run_0000' / 'outcome.json'))
    assert on_disk[0]['requested_corner'] == 'tt'
    assert on_disk[0]['backend'] == 'ngspice'


# --- serial versus parallel ----------------------------------------------------------

def test_serial_and_two_workers_agree(tmp_path, launch):
    launch['source'].update({'run_0000': A_TRAN, 'run_0001': OTHER,
                             'run_0002': A_TRAN, 'run_0003': OTHER})
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)

    def summarize(result):
        return [(r.run_id, r.seed, r.status, r.evaluation,
                 [(o.owner_path, o.spec_key, o.status, o.value, o.backend)
                  for o in r.spec_outcomes],
                 r.manifest.to_json()) for r in result.runs]

    serial = FaultCampaign(tb.netlist, FaultSet(), n_runs=4, seed=7,
                           out_dir=str(tmp_path / 's'), archive_waveforms=True,
                           progress=False).run()
    parallel = FaultCampaign(tb.netlist, FaultSet(), n_runs=4, seed=7,
                             out_dir=str(tmp_path / 'p'), n_workers=2,
                             archive_waveforms=True, progress=False).run()
    assert summarize(serial) == summarize(parallel)


# --- cross-backend reader boundary and corner-probe route ------------------------------

XYCE = os.path.join(RAWS, 'c_tran_ascii_cli.raw')  # recorded Xyce output


def test_xyce_fault_run_after_ngspice_nominal_reads_its_own_file(tmp_path, launch):
    dut = _module(tmp_path)
    tb = dut.create_test('resp')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-9, stop=40e-9))
    vout = tb.probe('out')
    tb.add_metric('plain', lambda ckt: final_value(vout))
    tb.add_metric('at_tt', lambda ckt: final_value(vout.corner('tt')))
    tb.add_spec('plain >= 0', 'plain', '>=', 0.0)
    tb.add_spec('at_tt >= 0', 'at_tt', '>=', 0.0)
    tb.evaluate(rawfile=A_TRAN)  # nominal ngspice evidence
    nominal_raw = tb.netlist._raw
    nominal_hint = tb.netlist._raw_simulator_hint
    nominal_conditions = tb.netlist.operating_conditions
    nominal_deck = getattr(tb.netlist, '_prepared_deck_record', None)

    launch['source'][None] = XYCE
    tb.netlist.simulator = 'xyce'
    result = _campaign(tb.netlist, tmp_path).run()
    expected = _final_out(XYCE, 'xyce')
    for run in result.runs:
        assert run.status == 'passed', run.error
        assert {o.spec_name: o.value for o in run.spec_outcomes} == {
            'plain >= 0': pytest.approx(expected),
            'at_tt >= 0': pytest.approx(expected)}
        assert {o.backend for o in run.spec_outcomes} == {'xyce'}
        assert {o.proved_conditions['backend'] for o in run.spec_outcomes} == {'xyce'}
    assert tb.netlist._raw is nominal_raw
    assert tb.netlist._raw_simulator_hint == nominal_hint == 'ngspice'
    assert tb.netlist.operating_conditions is nominal_conditions
    assert getattr(tb.netlist, '_prepared_deck_record', None) is nominal_deck


# --- real simulator control -------------------------------------------------------------

def _real_bench(tmp_path):
    """A resistor-loaded NMOS pull-down: on, the output sits low; deleted
    (stuck_open) the output rises to the supply."""
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'pull.cir', pdk_details=pdk,
                 module_name='pull')
    dut.define_module_ports({'out': 'output', 'vdd': 'power', 'vg': 'input',
                             'vss': 'ground'})
    dut.add(elements.resistor('rl', p='vdd', n='out', value=10e3))
    with dut:
        pdk.nmos1p8('mn1', d='out', g='vg', s='vss', b='vss', w=4.0, l=0.5)
    tb = dut.create_test('level')
    tb.add_module_under_test({'out': 'out', 'vdd': 'vdd', 'vg': 'vg', 'vss': '0'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-9, stop=20e-9))
    tb.add_metric('final_vout', lambda ckt: final_value(ckt.load_signal('out')))
    tb.add_spec('output low', 'final_vout', '<', 0.9)
    return tb


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_real_fault_moves_the_fresh_metric_across_the_spec(tmp_path):
    from silk.faults import stuck_open
    tb = _real_bench(tmp_path)
    tb.netlist.simulator = 'ngspice'
    control = FaultCampaign(
        tb.netlist, FaultSet(), n_runs=1, seed=3,
        out_dir=str(tmp_path / 'control'), progress=False).run()
    faulted = FaultCampaign(
        tb.netlist, FaultSet([stuck_open('xdut.mn1')]), n_runs=1, seed=3,
        out_dir=str(tmp_path / 'faulted'), progress=False).run()
    # Both results first, through the interface the old and new rows share.
    (ok_row,) = control.runs[0].spec_outcomes
    (bad_row,) = faulted.runs[0].spec_outcomes
    assert control.runs[0].error is None and faulted.runs[0].error is None
    assert ok_row.measurement is not None and bad_row.measurement is not None, \
        'a run reported no fresh measurement'
    assert ok_row.measurement.value < 0.9
    assert (ok_row.status, control.runs[0].status) == ('PASS', 'passed')
    assert bad_row.measurement.value == pytest.approx(1.8, abs=0.05)  # pull-down deleted
    assert (bad_row.status, faulted.runs[0].status) == ('FAIL', 'failed')
    assert faulted.runs[0].manifest.applied


@pytest.mark.slow
@needs_pdk
@needs_ngspice
def test_real_fault_rows_carry_fresh_attribution(tmp_path):
    from silk.faults import stuck_open
    tb = _real_bench(tmp_path)
    tb.netlist.simulator = 'ngspice'
    run = FaultCampaign(
        tb.netlist, FaultSet([stuck_open('xdut.mn1')]), n_runs=1, seed=3,
        out_dir=str(tmp_path / 'faulted'), progress=False).run().runs[0]
    (row,) = run.spec_outcomes
    assert row.attempt_kind == 'measured' and row.requested_corner == 'tt'
    assert row.backend == 'ngspice' and run.evaluation == 'fresh-v1'


# --- review 599e: lifecycle, resume, API surface, regression contracts ------------------

def _rows_file(tmp_path, run=0, root='campaign'):
    return tmp_path / root / 'runs' / f'run_{run:04d}' / 'outcome.json'


def test_failed_rerun_invalidates_the_previous_outcome_file(tmp_path, launch, monkeypatch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    first = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in first.runs] == ['passed', 'passed']
    assert json.load(open(_rows_file(tmp_path)))[0]['status'] == 'PASS'

    def broken(self, netlist_path, results_dir, **kw):
        raise RuntimeError('simulator exploded')
    for name in ('ngspice', 'xyce'):
        monkeypatch.setattr(type(SimulatorBackend.get(name)), 'run', broken)
    second = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in second.runs] == ['errored', 'errored']
    for i in range(2):
        path = _rows_file(tmp_path, i)
        assert not path.exists() or json.load(open(path)) == []
    summary = json.load(open(tmp_path / 'campaign' / 'summary.json'))
    assert [r['status'] for r in summary['runs']] == ['errored', 'errored']


def test_scoring_error_also_leaves_no_stale_passing_outcome(tmp_path, launch):
    state = {'boom': False}

    def make(calls):
        def extractor(ckt):
            if state['boom']:
                raise MetricBindingError('slot cannot bind')
            return final_value(ckt.load_signal('out'))
        return extractor
    dut, tb, spec, calls = _bench(tmp_path, target=0.5, extractor=make)
    assert [r.status for r in _campaign(tb.netlist, tmp_path).run().runs] == ['passed'] * 2
    state['boom'] = True
    again = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in again.runs] == ['errored', 'errored']
    assert 'MetricBindingError' in again.runs[0].error
    for i in range(2):
        path = _rows_file(tmp_path, i)
        assert not path.exists() or json.load(open(path)) == []


def test_cleanup_failure_is_contained_per_run(tmp_path, launch, monkeypatch):
    real_remove = os.remove

    def flaky(path, *a, **k):
        if 'run_0001' in str(path) and str(path).endswith('recorded.raw'):
            raise OSError('disk says no')
        return real_remove(path, *a, **k)
    monkeypatch.setattr(os, 'remove', flaky)
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    result = _campaign(tb.netlist, tmp_path, n_runs=3, archive=False).run()
    assert [r.run_id for r in result.runs] == [0, 1, 2]
    assert [r.status for r in result.runs] == ['passed', 'errored', 'passed']
    assert 'disk says no' in result.runs[1].error
    assert 'run_0001' in result.runs[1].error and 'recorded.raw' in result.runs[1].error
    assert result.runs[0].rawfile_path is None and result.runs[2].rawfile_path is None
    assert not os.path.exists(tmp_path / 'campaign' / 'runs' / 'run_0002' / 'recorded.raw')
    summary = json.load(open(tmp_path / 'campaign' / 'summary.json'))
    assert [r['status'] for r in summary['runs']] == ['passed', 'errored', 'passed']
    path = _rows_file(tmp_path, 1)
    assert not path.exists() or json.load(open(path)) == []


def test_resume_reruns_after_in_place_fault_set_mutation(tmp_path, launch):
    from silk.faults import stuck_open
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    fs = FaultSet()
    campaign = _campaign(tb.netlist, tmp_path, n_runs=2, fault_set=fs)
    prior = campaign.run()
    launched = len(launch['launches'])
    fs.add(stuck_open('xdut.r1'))
    resumed = campaign.resume(prior)
    assert len(launch['launches']) == launched + 2  # both runs redone
    assert all(r.manifest.input_spec['faults'] for r in resumed.runs)


def test_resume_reruns_after_fault_set_replacement_and_seed_change(tmp_path, launch):
    from silk.faults import stuck_open
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    prior = _campaign(tb.netlist, tmp_path, n_runs=2).run()
    n = len(launch['launches'])
    other = _campaign(tb.netlist, tmp_path, n_runs=2,
                      fault_set=FaultSet([stuck_open('xdut.r1')]))
    other.resume(prior)
    assert len(launch['launches']) == n + 2
    campaign = FaultCampaign(tb.netlist, FaultSet(), n_runs=2, seed=8,
                             out_dir=str(tmp_path / 'campaign'), progress=False)
    campaign.resume(prior)
    assert len(launch['launches']) == n + 4


def test_resume_reruns_when_the_checked_test_changed(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    campaign = _campaign(tb.netlist, tmp_path, n_runs=2)
    prior = campaign.run()
    for r in prior.runs:
        r.test_name = 'some other bench'
    n = len(launch['launches'])
    campaign.resume(prior)
    assert len(launch['launches']) == n + 2


def test_measurement_view_is_a_detached_measurement(tmp_path, launch):
    nominal = Measurement('spice', 3.0, passes=True)
    dut, tb, spec, calls = _bench(tmp_path)
    spec.record_measurement(nominal)
    result = _campaign(tb.netlist, tmp_path).run()
    meas = result.runs[0].spec_outcomes[0].measurement
    assert isinstance(meas, Measurement) and meas is not nominal
    assert (meas.source, meas.value, meas.passes) == ('spice', FRESH, False)
    from silk.simulation.conditions import ConditionState
    assert meas.corner is ConditionState.NONE  # PDK-free deck: corner proved none
    assert result.runs[0].spec_outcomes[0].spec.key == 'vout check'
    nan_run = _bench(tmp_path / 'n', extractor=lambda c: (lambda ckt: float('nan')))
    # a missing value has no measurement at all
    assert _campaign(nan_run[1].netlist, tmp_path / 'n').run() \
        .runs[0].spec_outcomes[0].measurement is None


def test_statistics_exclude_errored_and_simulated_runs_with_residual_rows():
    from silk.faults import CampaignResult, RunResult
    from silk.faults.evaluation import CampaignSpecOutcome
    from silk.faults.types import FaultManifest

    def row(status, value):
        return CampaignSpecOutcome(
            owner_path='$/dut', spec_key='s', spec_name='s', test_name='t',
            run_id=0, seed=0, backend='ngspice', requested_corner='tt',
            proved_conditions=None, status=status, value=value,
            attempt_kind='measured')
    m = FaultManifest(seed=0, input_spec={}, applied=[])
    runs = [RunResult(0, 0, 'passed', m, [row('PASS', 5.0)]),
            RunResult(1, 0, 'errored', m, [row('PASS', 99.0)], error='x'),
            RunResult(2, 0, 'simulated', m, [row('FAIL', -9.0)])]
    result = CampaignResult('c', 'top', 3, FaultSet(), 0, runs, 'out')
    assert list(result.measurement_distribution('s')) == [5.0]
    assert result.pass_fraction('s') == 1.0
    assert result.runs_failing('s') == []


def test_evaluation_runs_inside_a_suppressed_attempt_frame(tmp_path, launch):
    from silk.specs.attempt import _evaluation_attempt
    seen = []

    def make(calls):
        def extractor(ckt):
            frame = _evaluation_attempt.get()
            seen.append(None if frame is None else frame[0]['suppressed'])
            return final_value(ckt.load_signal('out'))
        return extractor
    dut, tb, spec, calls = _bench(tmp_path, extractor=make)
    _campaign(tb.netlist, tmp_path).run()
    assert seen == [True, True]
    assert _evaluation_attempt.get() is None


def test_callable_binding_error_makes_the_run_errored(tmp_path, launch):
    def make(calls):
        def extractor(ckt):
            raise MetricBindingError('probe belongs to another bench')
        return extractor
    dut, tb, spec, calls = _bench(tmp_path, extractor=make)
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['errored', 'errored']
    assert 'MetricBindingError' in result.runs[0].error
    assert result.runs[0].spec_outcomes == []


def _spy_reports(monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    reports = []
    original = ModuleTestbench.evaluate

    def spy(self, *a, **k):
        report = original(self, *a, **k)
        reports.append(report)
        return report
    monkeypatch.setattr(ModuleTestbench, 'evaluate', spy)
    return reports


def test_fault_report_carries_the_run_label_and_no_nominal_provenance(
        tmp_path, launch, monkeypatch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    tb.set_corner('ff')
    nominal_deck = {'deck_sha256': 'nominal-deck-hash', 'seed': 4242,
                    'simulator_options': {'nominal': 1}, 'save_list': {'x': 1},
                    'pdk_libraries': ['nominal.lib']}
    tb.netlist._prepared_deck_record = nominal_deck
    tb.netlist.simulator = 'ngspice'
    reports = _spy_reports(monkeypatch)
    _campaign(tb.netlist, tmp_path).run()
    assert len(reports) == 2
    for report in reports:
        assert report.corner == 'tt'
        conditions = report.operating_conditions
        assert conditions.origin == 'external'
        assert conditions.corner_kind == 'none'
        assert conditions.temperature_kind == 'unknown'
        assert conditions.supply_kind == 'unknown'
        prov = report.provenance
        assert prov.run_origin == 'external-rawfile'
        assert (prov.deck_sha256, prov.seed, prov.simulator_options,
                prov.save_list, prov.pdk_libraries) == (None,) * 5
        assert 'ff' not in json.dumps(conditions.to_dict())
    assert tb.netlist._prepared_deck_record is nominal_deck


def test_failed_extraction_keeps_provenance_clean_and_nominal_untouched(
        tmp_path, launch, monkeypatch):
    def make(calls):
        def boom(ckt):
            raise RuntimeError('extract failed')
        return boom
    dut, tb, spec, calls = _bench(tmp_path, extractor=make)
    nominal_deck = {'deck_sha256': 'nominal-deck-hash', 'seed': 4242}
    tb.netlist._prepared_deck_record = nominal_deck
    reports = _spy_reports(monkeypatch)
    result = _campaign(tb.netlist, tmp_path).run()
    assert [r.status for r in result.runs] == ['failed', 'failed']
    assert all(r.provenance.deck_sha256 is None for r in reports)
    assert tb.netlist._prepared_deck_record is nominal_deck
    assert '_raw_simulator_hint' not in tb.netlist.__dict__
    assert '_test_evidence' not in dut.__dict__


THRESH = 0.995  # nominal OTHER (1.0000003) passes, the fault run (0.99146) fails


def _selected_setup(tmp_path):
    """A leaf with captured-probe, corner-probe and function-slot metrics,
    evaluated through a selected hierarchy placement on OTHER (all PASS)."""
    from silk.circuit.test_placement import evidence_index
    mod = Module(None, str(tmp_path), 'leaf.cir', None, module_name='leaf')
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('r', p='out', n='vss', value=1000))
    tb = mod.create_test('step')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('rail', positive='out', negative='0', dc=0.5))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    vout = tb.probe('out')
    state = {'fail': False}

    def captured(ckt):
        if state['fail']:
            raise RuntimeError('captured probe failed')
        return final_value(vout)

    tb.add_metric('captured', captured)
    tb.add_metric('at_corner', lambda ckt: final_value(vout.corner('tt')))
    tb.add_metric('slot', final_value, sig=vout)
    specs = {n: tb.add_spec(f'{n} ok', n, '>=', THRESH)
             for n in ('captured', 'at_corner', 'slot')}
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_module(mod, 'one', out='a', vss='0')
    selected = top.get_test('step', instance='one')
    nominal = selected.evaluate(rawfile=OTHER)
    # The promised opposite verdicts: every nominal metric is a real value
    # above the threshold and every nominal spec passed.
    for name in ('captured', 'at_corner', 'slot'):
        value = nominal.metrics[name]
        assert isinstance(value, float) and value > THRESH, (name, value)
    assert nominal.results and all(r.passed for r in nominal.results)
    assert {r.spec_key for r in nominal.results} == {
        sp.key for sp in specs.values()}
    return mod, tb, top, selected, specs, state, evidence_index(top)


_ABSENT = object()


def _raw_content(raw):
    """A frozen copy of every waveform vector a result object holds."""
    import numpy as np
    native = getattr(raw, 'native', raw)
    vectors = getattr(native, 'vectors', None)
    if vectors is None:
        return None
    return {v.name: np.array(v.data, copy=True) for v in vectors}


def _same_content(a, b):
    import numpy as np
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _freeze(value):
    """Mutable contents, detached from the live object (identity is checked
    separately). Plain data is deep-copied; anything else is frozen as its
    repr, which is stable while the object is unchanged."""
    import copy
    if value is _ABSENT:
        return _ABSENT
    plain = (str, int, float, bool, bytes, type(None))

    def is_plain(v):
        if isinstance(v, plain):
            return True
        if isinstance(v, (list, tuple, set, frozenset)):
            return all(is_plain(x) for x in v)
        if isinstance(v, dict):
            return all(is_plain(k) and is_plain(x) for k, x in v.items())
        return False
    return copy.deepcopy(value) if is_plain(value) else repr(value)


def _snapshot(tb, selected, specs, index, *, extra=()):
    """Identities and, separately, frozen contents of everything a campaign
    must not touch."""
    import copy

    def spec_state(sp):
        measurements = list(sp.measurements.all_items())  # every owner entry
        attempts = [(scope, a) for scope, a in sp.__dict__.get('_attempts', ())]
        return {
            'measurement_ids': [(k, m) for k, m in measurements],
            'measurement_contents': [(k, repr(m)) for k, m in measurements],
            'attempt_ids': attempts,
            'attempt_contents': [(sc, a.attempt_id, copy.deepcopy(a.as_dict()))
                                 for sc, a in attempts],
            'extra': sorted(k for k in sp.__dict__ if k.startswith('_')),
        }

    netlist = tb.netlist
    attrs = ('_raw', '_raw_simulator_hint', 'operating_conditions',
             'results_file', '_prepared_deck_record')
    live = {n: netlist.__dict__.get(n, _ABSENT) for n in attrs}
    return {
        'ids': {k: (v, v.report, v.results, v.run_id) for k, v in index.records.items()},
        'record_contents': {k: copy.deepcopy(dict(v.values))
                            for k, v in index.records.items()},
        'corner_readers': {k: dict(v.corners) for k, v in index.records.items()},
        'reader_ids': {k: {n: x for n, x in vars(v.results).items()}
                       for k, v in index.records.items()},
        'reader_waveforms': {k: _raw_content(getattr(v.results, '_raw', None))
                             for k, v in index.records.items()},
        'reader_contents': {k: {n: _freeze(x) for n, x in vars(v.results).items()
                                if n != '_raw'}
                            for k, v in index.records.items()},
        'selected_file': selected.results_file,
        'bench': {k: copy.deepcopy(tb.__dict__.get(k)) for k in
                  ('_metric_values', '_sweep_table', '_sweep_generations', '_cache',
                   '_mc_samples')},
        'specs': {n: spec_state(sp) for n, sp in specs.items()},
        'netlist_ids': live,
        'netlist_contents': {n: _freeze(v) for n, v in live.items() if n != '_raw'},
        'netlist_waveform': _raw_content(None if live['_raw'] is _ABSENT
                                         else live['_raw']),
        'extra': {n: _freeze(v) for n, v in extra},
        'extra_ids': {n: v for n, v in extra},
    }


def _assert_unchanged(before, after):
    assert after['ids'].keys() == before['ids'].keys()
    for key, objs in before['ids'].items():
        assert all(a is b for a, b in zip(after['ids'][key], objs))
    assert after['record_contents'] == before['record_contents']
    assert after['corner_readers'].keys() == before['corner_readers'].keys()
    for key, mapping in before['corner_readers'].items():
        assert after['corner_readers'][key].keys() == mapping.keys()
        assert all(after['corner_readers'][key][n] is r for n, r in mapping.items())
    for key, fields in before['reader_ids'].items():
        assert after['reader_ids'][key].keys() == fields.keys()
        assert all(after['reader_ids'][key][n] is x for n, x in fields.items())
    for key, wave in before['reader_waveforms'].items():
        assert _same_content(after['reader_waveforms'][key], wave), \
            'a retained nominal waveform changed'
    assert after['reader_contents'].keys() == before['reader_contents'].keys()
    for key, fields in before['reader_contents'].items():
        assert _eq_frozen(after['reader_contents'][key], fields)
    assert after['selected_file'] == before['selected_file']
    assert after['bench'] == before['bench']
    for name, state in before['specs'].items():
        now = after['specs'][name]
        assert now['measurement_contents'] == state['measurement_contents']
        assert all(a[1] is b[1] for a, b in zip(now['measurement_ids'],
                                                state['measurement_ids']))
        assert len(now['measurement_ids']) == len(state['measurement_ids'])
        assert now['attempt_contents'] == state['attempt_contents']
        assert all(a[1] is b[1] for a, b in zip(now['attempt_ids'], state['attempt_ids']))
        assert now['extra'] == state['extra']
    for n, value in before['netlist_ids'].items():
        assert after['netlist_ids'][n] is value, n
    assert _eq_frozen(after['netlist_contents'], before['netlist_contents']), \
        'an authored netlist attribute changed in place'
    assert _same_content(after['netlist_waveform'], before['netlist_waveform'])
    assert all(after['extra_ids'][n] is v for n, v in before['extra_ids'].items())
    assert _eq_frozen(after['extra'], before['extra']), \
        'a protected nominal object changed in place'


def _eq_frozen(a, b):
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


@pytest.mark.parametrize('fail', [False, True])
def test_selected_hierarchy_nominal_and_fault_evidence_never_mix(tmp_path, launch, fail):
    from silk.specs.spec import CalibrationBinding
    mod, tb, top, selected, specs, state, index = _selected_setup(tmp_path)
    mod._behavioral_params = {'gain': 7.0}
    specs['slot'].calibration = CalibrationBinding(mod, 'gain')
    before = _snapshot(tb, selected, specs, index)
    state['fail'] = fail
    result = FaultCampaign(tb.netlist, FaultSet(), n_runs=2, seed=7,
                           out_dir=str(tmp_path / 'campaign'), archive_waveforms=True,
                           progress=False).run()
    assert [r.status for r in result.runs] == ['failed', 'failed']
    for run in result.runs:
        by_name = {o.spec_name: o for o in run.spec_outcomes}
        assert set(by_name) == {'captured ok', 'at_corner ok', 'slot ok'}
        assert all(o.status == 'FAIL' for o in by_name.values())  # nominal passed
        if fail:
            assert by_name['captured ok'].attempt_kind == 'extractor_error'
            assert by_name['captured ok'].value is None
        else:
            assert by_name['captured ok'].value == FRESH
        assert by_name['at_corner ok'].value == FRESH  # explicit corner probe
        assert by_name['slot ok'].value == FRESH      # function-form slot
        assert {o.requested_corner for o in by_name.values()} == {'tt'}
    _assert_unchanged(before, _snapshot(tb, selected, specs, index))
    assert mod._behavioral_params == {'gain': 7.0}  # calibration target untouched


def test_sweep_table_key_is_the_requested_label_during_a_fault_evaluation(tmp_path, launch):
    dut = _module(tmp_path)
    tb = dut.create_test('resp')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-9, stop=40e-9))
    tb.set_corner('ff')
    seen = []

    def metric(ckt):
        seen.append(dict(tb.sweep_corners))
        return final_value(ckt.load_signal('out'))
    tb.add_metric('m', metric)
    tb.add_spec('m ok', 'm', '>=', 0.5)
    _campaign(tb.netlist, tmp_path).run()
    assert [list(t) for t in seen] == [['tt'], ['tt']]
    assert tb._sweep_table == {}


def test_copied_netlist_binds_its_own_copied_bench_not_the_original(tmp_path, launch):
    import copy
    dut, tb, spec, calls = _bench(tmp_path)
    spec.record_measurement(Measurement('spice', 3.0, passes=True))
    clone = copy.deepcopy(tb.netlist)
    result = _campaign(clone, tmp_path).run()
    # The copy carries its own bench and specs: scored from the fault output,
    # never from the original's recorded 3.0, and the original is untouched.
    assert [r.status for r in result.runs] == ['failed', 'failed']
    assert list(result.measurement_distribution('vout check')) == [FRESH, FRESH]
    assert [m.value for m in spec.measurements.values()] == [3.0] \
        if isinstance(spec.measurements, dict) else True
    assert tb._sweep_table == {} and getattr(tb.netlist, '_raw', None) is None


def test_function_form_ambiguity_is_a_metric_binding_error(tmp_path, launch):
    dut = _module(tmp_path)
    tb = dut.create_test('resp')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_sim_type(Transient(step=1e-9, stop=40e-9))
    vout = tb.probe('out')
    tb.add_metric('slot', final_value, sig=vout)
    tb.add_spec('slot ok', 'slot', '>=', 0.5)
    from silk.simulation import AC
    tb.add_sim_type(AC(variation='dec', points=5, fstart=1.0, fstop=1e6))
    with pytest.raises(MetricBindingError):
        _campaign(tb.netlist, tmp_path).run()
    assert launch['launches'] == []


@pytest.mark.parametrize('fail', [False, True])
def test_xyce_fault_after_retained_ngspice_selected_evidence(
        tmp_path, launch, monkeypatch, fail):
    mod, tb, top, selected, specs, state, index = _selected_setup(tmp_path)
    nominal_deck = {'deck_sha256': 'nominal-deck-hash', 'seed': 4242,
                    'simulator_options': {'nominal': 1}, 'save_list': {'x': 1},
                    'pdk_libraries': ['nominal.lib']}
    tb.netlist._prepared_deck_record = nominal_deck
    before = _snapshot(tb, selected, specs, index,
                       extra=[('nominal_deck', nominal_deck)])
    state['fail'] = fail
    launch['source'][None] = XYCE
    tb.netlist.simulator = 'xyce'
    reports = _spy_reports(monkeypatch)
    result = _campaign(tb.netlist, tmp_path).run()
    expected = _final_out(XYCE, 'xyce')
    for run in result.runs:
        by_name = {o.spec_name: o for o in run.spec_outcomes}
        assert {o.backend for o in by_name.values()} == {'xyce'}
        assert by_name['at_corner ok'].value == pytest.approx(expected)
        assert by_name['slot ok'].value == pytest.approx(expected)
        captured = by_name['captured ok']
        if fail:
            assert (captured.value, captured.attempt_kind) == (None, 'extractor_error')
            assert 'captured probe failed' in captured.reason
        else:
            assert captured.value == pytest.approx(expected)  # not the nominal 1.0000003
            assert captured.attempt_kind == 'measured'
    assert len(reports) == 2
    for report in reports:
        assert report.operating_conditions.backend == 'xyce'
        prov = report.provenance
        assert prov.run_origin == 'external-rawfile'
        assert (prov.deck_sha256, prov.seed, prov.simulator_options,
                prov.save_list, prov.pdk_libraries) == (None,) * 5
    _assert_unchanged(before, _snapshot(tb, selected, specs, index,
                                        extra=[('nominal_deck', nominal_deck)]))
    assert tb.netlist._prepared_deck_record is nominal_deck


def test_artifact_write_and_cleanup_failures_are_contained_per_run(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    # A directory where the temp artifact belongs: the write fails and so does
    # the cleanup (os.remove cannot delete a directory).
    (tmp_path / 'campaign' / 'runs' / 'run_0001' / 'outcome.json.tmp').mkdir(parents=True)
    result = _campaign(tb.netlist, tmp_path, n_runs=3).run()
    assert [r.run_id for r in result.runs] == [0, 1, 2]
    assert [r.status for r in result.runs] == ['passed', 'errored', 'passed']
    assert 'outcome.json' in result.runs[1].error  # the original failure and path
    assert 'could not remove' in result.runs[1].error  # and the cleanup failure
    summary = json.load(open(tmp_path / 'campaign' / 'summary.json'))
    assert [r['status'] for r in summary['runs']] == ['passed', 'errored', 'passed']
    assert _rows_file(tmp_path, 2).exists()


def test_distributions_skip_unchecked_and_unmeasured_numeric_rows():
    from silk.faults import CampaignResult, RunResult
    from silk.faults.evaluation import CampaignSpecOutcome
    from silk.faults.types import FaultManifest

    def row(status, value, checked=True):
        return CampaignSpecOutcome(
            owner_path='$/dut', spec_key='s', spec_name='s', test_name='t',
            run_id=0, seed=0, backend='ngspice', requested_corner='tt',
            proved_conditions=None, status=status, value=value,
            attempt_kind='measured', checked=checked)
    m = FaultManifest(seed=0, input_spec={}, applied=[])
    runs = [RunResult(0, 0, 'passed', m, [row('PASS', 5.0)]),
            RunResult(1, 0, 'failed', m, [row('FAIL', math.inf)]),
            RunResult(2, 0, 'unverified', m, [row('PASS', 11.0, checked=False)]),
            RunResult(3, 0, 'unverified', m, [row('UNMEASURED', 12.0)])]
    result = CampaignResult('c', 'top', 4, FaultSet(), 0, runs, 'out')
    assert list(result.measurement_distribution('s')) == [5.0, math.inf]
    assert result.pass_fraction('s') == 0.5
    assert [r.run_id for r in result.runs_failing('s')] == [1]


def test_rendered_report_rows_carry_each_specs_own_statistics(tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    tb.add_spec('vout check', 'final_vout', '>=', 2.0, key='second')
    md = _campaign(tb.netlist, tmp_path).run().generate_report()._render_md()
    rows = {l.split('|')[1].strip(): l for l in md.splitlines()
            if l.startswith('| `') and 'vout check' in l}
    (first,) = [l for k, l in rows.items() if '[vout check]' in k]
    (second,) = [l for k, l in rows.items() if '[second]' in k]
    assert '100.0%' in first and '2/2' in first
    assert '0.0%' in second and '100.0%' not in second


def test_excluded_json_rows_carry_run_attribution_without_claiming_evaluation(
        tmp_path, launch):
    dut, tb, spec, calls = _bench(tmp_path, target=0.5)
    other = dut.create_test('other')
    other.add_module_under_test({'out': 'out', 'vss': '0'})
    other.add_sim_type(Transient(step=1e-9, stop=40e-9))
    other.add_metric('m', lambda ckt: 1.0)
    other.add_spec('other spec', 'm', '<', 0.0)
    result = _campaign(tb.netlist, tmp_path).run()
    rows = json.load(open(_rows_file(tmp_path, 1)))
    (skipped,) = [r for r in rows if not r['checked']]
    assert (skipped['run_id'], skipped['seed'], skipped['backend'],
            skipped['requested_corner']) == (1, result.runs[1].seed, 'ngspice', 'tt')
    assert skipped['proved_conditions'] is None  # never evaluated
    assert set(skipped) == set(rows[0])  # same projection keys as checked rows
