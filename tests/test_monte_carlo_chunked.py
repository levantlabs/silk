"""Several Monte Carlo samples in one ngspice process (TASK-551).

The pure parts (deck builder, transcript parser, budget, refusals) run
without a simulator; the campaigns compare ``samples_per_process`` > 1 with one
process per sample on ngspice + sky130.
"""
import os
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from silk import load_pdk, sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.circuit.monte_carlo import MonteCarloSettings
from silk.exceptions import MonteCarloError, NotSupportedError
from silk.metrics import final_value
from silk.simulation import AC, DC, Noise, Op, Transient, mc_chunk
from silk.simulation.completion import NO_REQUEST_REASON, OP_ONLY_REASON
from tests._env import needs_ngspice, needs_pdk

BASE_DECK = """*** Circuit: x.cir
.lib "/pdk/sky130.lib.spice" tt_mm
.options seed=11 reltol=1e-4

.op

.control
save all
run
set wr_singlescale
set wr_vecnames
option numdgt=3
write /stage/rawfile.raw
showmod 'm.xdut.XMm1.msky130_fd_pr__nfet_01v8' : vth0
quit 0
.endc
.end
"""
MEMBERS = [mc_chunk.ChunkMember(4, 40), mc_chunk.ChunkMember(5, 50)]


def _deck(members=MEMBERS):
    return mc_chunk.build_chunk_deck(
        BASE_DECK, 'c4', members, {m.index: f'/priv/s{m.index}.raw' for m in members})


# -- pure ------------------------------------------------------------------

def test_loop_deck_has_one_control_block_and_no_seed_option():
    deck = _deck()
    assert deck.count('.control') == 1 and deck.count('quit 0') == 1
    assert 'seed=11' not in deck and '.options reltol=1e-4' in deck
    for index, seed in ((4, 40), (5, 50)):
        block = deck.split(f'echo silk-mc BEGIN c4 {index}')[1].split(f'END c4 {index}')[0]
        assert block.index(f'setseed {seed}') < block.index('remcirc') < block.index('mc_source')
        assert block.index('mc_source') < block.index('save all') < block.index('\nrun\n')
        assert f'write /priv/s{index}.raw' in block and '/stage/' not in block
        assert 'showmod' in block
    assert deck.count('destroy all') == 2
    assert 'reset' not in deck


def test_options_line_with_only_a_seed_is_dropped():
    deck = mc_chunk.build_chunk_deck(
        BASE_DECK.replace(' reltol=1e-4', ''), 'c4', MEMBERS, {4: 'a', 5: 'b'})
    assert '.options' not in deck


@pytest.mark.parametrize('line', ['osdi x.osdi', 'pre_osdi x.osdi', 'op', 'echo hi'])
def test_unknown_control_command_is_refused(line):
    text = BASE_DECK.replace('save all\n', f'save all\n{line}\n')
    with pytest.raises(NotSupportedError, match='multi-sample'):
        mc_chunk.build_chunk_deck(text, 'c4', MEMBERS, {4: 'a', 5: 'b'})


def _log(*parts):
    return ''.join(parts)


def _seg(i, body='', status='0', end=True, chunk='c4'):
    return (f'silk-mc BEGIN {chunk} {i}\n{body}silk-mc STATUS {chunk} {i} {status}\n'
            + (f'silk-mc END {chunk} {i}\n' if end else ''))


def test_transcript_cut_into_preamble_segments_and_footer():
    t = mc_chunk.parse_transcript(
        'parse note\n' + _seg(4, 'a\n') + _seg(5, 'b\n') + 'ngspice-42 done\n', 'c4', [4, 5])
    assert t.complete and t.preamble == 'parse note\n' and t.footer == 'ngspice-42 done\n'
    assert t.segments[4].text.startswith('a\n') and 'b' not in t.segments[4].text
    assert t.segments[5].ended and t.segments[5].status == '0'


def test_failed_status_is_a_finished_sample():
    t = mc_chunk.parse_transcript(_seg(4, status='1') + _seg(5), 'c4', [4, 5])
    assert t.segments[4].ended and t.segments[4].status == '1' and t.complete


@pytest.mark.parametrize('text, ended', [
    (_seg(4) + 'silk-mc BEGIN c4 5\n', [4]),                         # crash in 5
    (_seg(4) + _seg(5, end=False)[:-4], [4]),                        # truncated marker
    (_seg(4) + _seg(4) + _seg(5), [4]),                              # duplicate
    (_seg(5) + _seg(4), []),                                         # out of order
    (_seg(4, chunk='c9') + _seg(5), []),                             # other chunk
    (_seg(4) + 'silk-mc END c4 5\n' + _seg(5), [4]),                 # END without BEGIN
    ('silk-mc BEGIN c4 4\nsilk-mc END c4 4\n' + _seg(5), []),        # END without STATUS
    ('', []),                                                        # no output at all
])
def test_marker_anomalies_leave_the_rest_unfinished(text, ended):
    t = mc_chunk.parse_transcript(text, 'c4', [4, 5])
    assert not t.complete and t.footer == ''
    assert sorted(i for i, s in t.segments.items() if s.ended) == ended


def test_a_pre_begin_failure_stays_in_the_preamble_only():
    t = mc_chunk.parse_transcript('Error: parse failure\n' + _seg(4) + _seg(5), 'c4', [4, 5])
    assert 'parse failure' in t.preamble
    assert 'parse failure' not in t.segments[4].text + t.segments[5].text


def test_deck_comparison_ignores_only_seed_and_rawfile_path():
    other = BASE_DECK.replace('seed=11', 'seed=99').replace('/stage/', '/other/')
    assert mc_chunk.normalised_deck(other) == mc_chunk.normalised_deck(BASE_DECK)
    assert (mc_chunk.normalised_deck(BASE_DECK.replace('.op', '.tran 1n 2n'))
            != mc_chunk.normalised_deck(BASE_DECK))


def test_chunk_budget_is_the_per_sample_timeout_times_members(tmp_path, monkeypatch):
    seen = {}

    monkeypatch.setattr(mc_chunk, 'warn_version_advice', lambda *a: None, raising=False)
    import silk.simulation.runner as runner
    monkeypatch.setattr(runner, 'warn_version_advice', lambda *a: None)

    def fake_run(cmd, **kwargs):
        seen['timeout'] = kwargs['timeout']
        with open(cmd[cmd.index('-o') + 1], 'w') as f:
            f.write(_seg(4, 'ok\n') + 'silk-mc BEGIN c4 5\n')
        raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])

    monkeypatch.setattr(mc_chunk.subprocess, 'run', fake_run)
    chunk = mc_chunk.ChunkRun(str(tmp_path / 'chunk'), MEMBERS)
    chunk._launch_chunk(SimpleNamespace(completion=None, completion_unverified_reason=OP_ONLY_REASON), BASE_DECK, 7.0, False)
    assert seen['timeout'] == 14.0 and chunk.timed_out and chunk.error is None
    assert [i for i, s in chunk.transcript.segments.items() if s.ended] == [4]
    chunk2 = mc_chunk.ChunkRun(str(tmp_path / 'chunk2'), MEMBERS)
    chunk2._launch_chunk(SimpleNamespace(completion=None, completion_unverified_reason=OP_ONLY_REASON), BASE_DECK, None, False)
    assert seen['timeout'] is None


def test_settings_validation():
    assert MonteCarloSettings(n=3).samples_per_process == 1
    for bad in (0, -1, 2.0, True, '2', None):
        with pytest.raises(MonteCarloError, match='samples_per_process'):
            MonteCarloSettings(n=3, samples_per_process=bad)


# -- the benches -----------------------------------------------------------

def _build(tmp_path, analysis, *, simulator=None, ac=None, probe_analysis=None,
           vout_metric=True):
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', str(tmp_path), 'b.cir', pdk_details=pdk,
                 module_name='b')
    if simulator:
        mod.simulator = simulator
    mod.define_module_ports({'d': PortDir.INOUT, 'g': PortDir.IN,
                             's': PortDir.GROUND})
    with mod:
        pdk.nmos1p8('m1', d='d', g='g', s='s', b='s', w=1.0, l=0.5)
    tb = mod.create_test('t')
    tb.add_module_under_test({'d': 'vout', 'g': 'vg', 's': '0'})
    tb.add(sources.vdc('vd', positive='vd', negative='0', value=1.8))
    from silk.elements import elements
    tb.add(elements.resistor('rl', p='vd', n='vout', value=20e3))
    tb.add(sources.vdc('vg', positive='vg', negative='0', value=0.9,
                       **({} if ac is None else {'ac': ac})))
    tb.add_sim_type(analysis)
    if vout_metric:
        out = tb.probe('vout', analysis=probe_analysis) if probe_analysis else tb.probe('vout')
        tb.add_metric('vout_V', final_value, sig=out)
        tb.add_spec('vout', 'vout_V', '<', 1.7)
    return tb


def _op(tmp_path, **kw):
    return _build(tmp_path, Op(), **kw)


def _tran(tmp_path, **kw):
    return _build(tmp_path, Transient(step=0.1e-9, stop=5e-9), **kw)


def _raw(report):
    with open(report.rawfile_path, 'rb') as f:
        return re.sub(rb'^Date:[^\n]*\n', b'', f.read(), flags=re.M)  # one per plot


def _run(tb, n, *, spp, workers=None, allow_fallback=False, **kw):
    """Run a campaign; unless *allow_fallback*, silk falling back to one
    process per sample (its ceiling warning) is a test failure, so equality
    controls cannot pass without sharing processes."""
    import warnings
    with warnings.catch_warnings():
        if not allow_fallback:
            warnings.filterwarnings('error', message='.*is a ceiling')
        return tb.run_monte_carlo(n, samples_per_process=spp, n_workers=workers, **kw)


def assert_same_campaign(a, b):
    assert sorted(a.reports) == sorted(b.reports) and a.seeds == b.seeds
    assert a.failed.keys() == b.failed.keys()
    for key in a.reports:
        assert a.reports[key].metrics == b.reports[key].metrics, key
        assert _raw(a.reports[key]) == _raw(b.reports[key]), key
    if a.draws is not None:
        assert a.draws.values.keys() == b.draws.values.keys()
        for key in a.draws.values:
            assert np.array_equal(a.draws.values[key], b.draws.values[key])
    assert a.yield_fraction() == b.yield_fraction()
    if a.draws is not None:
        key = next(iter(a.reports))
        fa = a.reports[key].rawfile_path.rsplit('/', 1)[0].rsplit('/', 1)[0]
        fb = b.reports[key].rawfile_path.rsplit('/', 1)[0].rsplit('/', 1)[0]
        assert os.path.isfile(fa + '/draws.csv') and os.path.isfile(fb + '/draws.csv')
        assert open(fa + '/draws.csv').read() == open(fb + '/draws.csv').read()


# -- refusals ---------------------------------------------------------------

@pytest.mark.parametrize('output_mode', ['overwrite', 'new'])
def test_samples_per_process_needs_ngspice_before_any_deck(tmp_path, output_mode):
    tb = _op(tmp_path, simulator='xyce')
    with pytest.raises(NotSupportedError, match='ngspice'):
        tb.run_monte_carlo(2, samples_per_process=2, output_mode=output_mode)
    tb = _op(tmp_path / 'b', simulator='behavioral')
    with pytest.raises(NotSupportedError, match='ngspice'):
        tb.run_monte_carlo(2, samples_per_process=2, output_mode=output_mode)
    assert not list(tmp_path.rglob('*.cir'))


@pytest.mark.parametrize('output_mode', ['overwrite', 'new'])
def test_samples_per_process_refuses_save_device_currents(tmp_path, output_mode):
    tb = _op(tmp_path)
    tb.module.save_device_currents = True
    with pytest.raises(MonteCarloError, match='save_device_currents'):
        tb.run_monte_carlo(2, samples_per_process=2, output_mode=output_mode)


@pytest.mark.parametrize('output_mode', ['overwrite', 'new'])
def test_samples_per_process_refuses_custom_lines(tmp_path, output_mode):
    tb = _op(tmp_path)
    tb.netlist.add_custom_line('* custom')
    with pytest.raises(NotSupportedError, match='custom'):
        tb.run_monte_carlo(2, samples_per_process=2, output_mode=output_mode)


# -- campaigns (need ngspice + sky130) ---------------------------------------

CAMPAIGN = [pytest.mark.slow, needs_ngspice, needs_pdk]




@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('make', [_op, _tran])
@pytest.mark.parametrize('variation', ['mismatch', 'both'])
def test_chunked_campaign_equals_one_process_per_sample(tmp_path, make, variation):
    tb = make(tmp_path / 'pp')
    per = _run(tb, 3, spp=1, variation=variation, seed=5, record_draws=True)
    tb2 = make(tmp_path / 'ch')
    chunked = _run(tb2, 3, spp=2, variation=variation, seed=5, record_draws=True)
    assert chunked.rerun_samples == ()
    assert_same_campaign(per, chunked)
    for key, report in chunked.reports.items():
        folder = report.rawfile_path.rsplit('/', 1)[0]
        assert folder.endswith(f'mc_{key[1]:04d}')
        log = open(folder + '/output.log').read()
        assert 'silk-mc BEGIN' not in log and 'silk-mc END' not in log
    # rerun(key) still runs that one sample alone and reproduces it.
    again = chunked.rerun(('tt', 2))
    assert again.metrics == per.reports[('tt', 2)].metrics
    assert _raw(again) == _raw(per.reports[('tt', 2)])


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_chunked_campaign_runs_in_parallel_chunks(tmp_path):
    per = _run(_op(tmp_path / 'pp'), 4, spp=1, seed=2)
    chunked = _run(_op(tmp_path / 'ch'), 4, spp=2, workers=2, seed=2)
    assert chunked.rerun_samples == ()
    assert_same_campaign(per, chunked)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_setting_samples_per_process_does_not_block_replay(tmp_path):
    tb = _op(tmp_path)
    tb.set_monte_carlo(2, seed=3, samples_per_process=2)
    mc = tb.module.run_all_tests()['b.t']
    tb.set_monte_carlo(2, seed=3, samples_per_process=1)
    assert mc.rerun(('tt', 1)).metrics == mc.reports[('tt', 1)].metrics


@pytest.fixture
def patched_deck(monkeypatch):
    """Let a test edit one member's block of the generated loop deck."""
    state = {}
    real = mc_chunk.build_chunk_deck

    def build(base, chunk_id, members, raw_paths):
        text = real(base, chunk_id, members, raw_paths)
        return state['edit'](text) if 'edit' in state else text

    monkeypatch.setattr(mc_chunk, 'build_chunk_deck', build)
    return state


def _after(text, marker, extra):
    return text.replace(marker, marker + extra, 1)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_crashed_chunk_reruns_unfinished_samples_one_per_process(tmp_path, patched_deck):
    per = _run(_op(tmp_path / 'pp'), 4, spp=1, seed=9)
    # ngspice leaves the script after sample 1's END: 2 and 3 never run.
    patched_deck['edit'] = lambda t: _after(t, 'echo silk-mc END c0 1\n', 'quit 0\n')
    with pytest.warns(RuntimeWarning, match='rerun one per process'):
        chunked = _run(_op(tmp_path / 'ch'), 4, spp=4, seed=9)
    assert chunked.rerun_samples == (('tt', 2), ('tt', 3))
    assert 'rerun one per process' in chunked.summary()
    assert chunked.corner('tt').rerun_samples == chunked.rerun_samples
    assert_same_campaign(per, chunked)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_sample_the_process_dies_inside_is_rerun_not_lost(tmp_path, patched_deck):
    patched_deck['edit'] = lambda t: _after(t, 'echo silk-mc BEGIN c0 1\n', 'quit 0\n')
    with pytest.warns(RuntimeWarning):
        chunked = _run(_op(tmp_path), 3, spp=3, seed=1)
    assert chunked.rerun_samples == (('tt', 1), ('tt', 2)) and not chunked.failed


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_failed_sample_is_its_own_typed_failure(tmp_path, patched_deck):
    # Sample 1's circuit is removed before it runs: no analysis, no plot.
    patched_deck['edit'] = lambda t: re.sub(
        r'(silk-mc BEGIN c0 1\n(?:.*\n)*?)(run\n)', r'\1remcirc\n\2', t, count=1)
    chunked = _run(_op(tmp_path), 3, spp=3, seed=1)
    assert list(chunked.failed) == [('tt', 1)]
    assert sorted(chunked.reports) == [('tt', 0), ('tt', 2)]
    assert chunked.rerun_samples == ()
    assert 'ngspice' in chunked.failed[('tt', 1)].error


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_status_zero_truncated_transient_fails_that_sample_only(tmp_path, patched_deck):
    patched_deck['edit'] = lambda t: re.sub(
        r'(silk-mc BEGIN c0 1\n(?:.*\n)*?)(run\n)', r'\1stop when time > 2n\n\2', t, count=1)
    chunked = _run(_tran(tmp_path), 3, spp=3, seed=1)
    assert list(chunked.failed) == [('tt', 1)]
    message = chunked.failed[('tt', 1)].error
    assert 'tran analysis aborted' in message and 'expected final grid point' in message
    assert sorted(chunked.reports) == [('tt', 0), ('tt', 2)]


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_altered_chunk_deck_is_refused(tmp_path, monkeypatch):
    real = mc_chunk.subprocess.run

    def tamper(cmd, **kwargs):
        done = real(cmd, **kwargs)
        with open(cmd[-1], 'a') as f:
            f.write('* changed\n')
        return done

    monkeypatch.setattr(mc_chunk.subprocess, 'run', tamper)
    chunked_run = lambda: _run(_op(tmp_path), 2, spp=2, seed=1)
    with pytest.raises(Exception, match='every declared run failed|changed while ngspice ran'):
        chunked_run()


# -- review of the diff (P2-1 .. P2-5) -------------------------------------

def _run_files(**kw):
    base = dict(completion=None, completion_unverified_reason=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_missing_certificate_is_only_admitted_for_a_provably_op_only_deck(tmp_path):
    for reason in (NO_REQUEST_REASON, 'opaque custom content', None,
                   'emitted analysis cards do not match silk analysis records'):
        chunk = mc_chunk.ChunkRun(str(tmp_path / 'c'), [mc_chunk.ChunkMember(0, 1)])
        chunk._launch_chunk(_run_files(completion_unverified_reason=reason),
                            BASE_DECK, None, False)
        # Not a refusal: the member is left to run in its own process.
        assert isinstance(chunk.error, mc_chunk.ChunkUnfinished), reason
        assert 'not provably op-only' in str(chunk.error)
        assert not (tmp_path / 'c').exists()


def test_op_only_is_proved_from_the_declared_and_emitted_analyses(tmp_path):
    from silk.simulation.completion import certify_generated
    tb = _op(tmp_path)
    assert certify_generated(tb.netlist, '.options a=1\n.op\n') == (None, OP_ONLY_REASON)
    # Declared op but no op card emitted: not proven.
    assert certify_generated(tb.netlist, '.options a=1\n') == (None, NO_REQUEST_REASON)
    tran = _tran(tmp_path / 't')
    # Declared transient but only an op card emitted: not op-only.
    assert certify_generated(tran.netlist, '.op\n') == (None, NO_REQUEST_REASON)
    tb.netlist.sim_analysis_dict.clear()           # nothing declared at all
    assert certify_generated(tb.netlist, '.op\n') == (None, NO_REQUEST_REASON)


def test_the_campaign_admission_is_decided_from_the_declared_analyses(tmp_path):
    assert _op(tmp_path / 'a')._chunk_admission_reason() is None
    assert _tran(tmp_path / 'b')._chunk_admission_reason() is None
    empty = _op(tmp_path / 'c')
    empty.netlist.sim_analysis_dict.clear()
    assert 'no analysis' in empty._chunk_admission_reason()


def test_two_plots_for_one_certified_request_fail_that_sample():
    from silk.exceptions import ConfigurationError, SimulationError
    from silk.simulation.completion import CompletionPlan, CompletionRequest
    plot = SimpleNamespace(name='tran1', plotname='Transient Analysis',
                           vectors=[SimpleNamespace(name='time', data=[0.0])])
    for unverified in (None, 'resolution'):
        plan = CompletionPlan('ngspice', b'x', (CompletionRequest(
            'tran', 0.0, 1e-9, 1e-9, endpoint_unverified_reason=unverified),))
        with pytest.raises(SimulationError, match='2 tran plots') as caught:
            mc_chunk._require_one_plot_per_request(
                SimpleNamespace(plots=[plot, plot]), plan, 3, '/x.raw')
        assert not isinstance(caught.value, ConfigurationError)
        mc_chunk._require_one_plot_per_request(SimpleNamespace(plots=[plot]), plan, 3, '/x.raw')


def test_a_failed_status_keeps_the_convergence_classification(tmp_path):
    from silk.exceptions import ConvergenceError, SimulationAborted
    log = tmp_path / 'output.log'
    files = SimpleNamespace(logfile_path=str(log))
    log.write_text('doAnalyses: TRAN: Timestep too small; time = 1e-9\n')
    with pytest.raises(ConvergenceError) as caught:
        mc_chunk._raise_failed_status('1', 4, files)
    assert caught.value.matched_lines and caught.value.data_rows is None
    assert 'sample 4' in str(caught.value) and caught.value.log_path == str(log)
    log.write_text('some other message\n')
    with pytest.raises(SimulationAborted) as other:
        mc_chunk._raise_failed_status('1', 4, files)
    assert not isinstance(other.value, ConvergenceError) and other.value.data_rows is None


def _files(root):
    """Every file under *root* with its bytes."""
    return sorted((str(p.relative_to(root)), p.read_bytes())
                  for p in root.rglob('*') if p.is_file())


def test_save_device_params_is_checked_on_the_module_not_a_stale_netlist(tmp_path):
    settings = MonteCarloSettings(n=2, samples_per_process=2)
    tb = _op(tmp_path / 'a')
    tb.module.save_device_params = True          # the netlist still says False
    before = _files(tmp_path / 'a')
    with pytest.raises(NotSupportedError, match='save_device_params'):
        tb.run_monte_carlo(2, samples_per_process=2)
    assert _files(tmp_path / 'a') == before
    stale = _op(tmp_path / 'b')
    stale.netlist.save_device_params = True      # the module cleared it
    stale._check_samples_per_process(settings)   # does not refuse


def test_osdi_models_are_refused_before_any_file(tmp_path, monkeypatch):
    tb = _op(tmp_path)
    monkeypatch.setattr(tb.netlist.pdk_details, 'osdi_files', ['m.osdi'], raising=False)
    before = _files(tmp_path)
    with pytest.raises(NotSupportedError, match='OSDI'):
        tb.run_monte_carlo(2, samples_per_process=2)
    assert _files(tmp_path) == before


def _status_edit(index, status):
    """Make the transcript report simulator status *status* for sample *index*."""
    return lambda t: t.replace(
        f'echo silk-mc STATUS c0 {index} $sim_status',
        f'echo silk-mc STATUS c0 {index} {status}', 1)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_nonzero_status_fails_that_sample_even_with_a_complete_rawfile(tmp_path, patched_deck):
    # Sample 1 completed with a readable rawfile but the simulator said 1;
    # then the process dies inside sample 3: the completed failure is not retried.
    patched_deck['edit'] = lambda t: _after(_after(
        _status_edit(1, 1)(t), 'echo silk-mc BEGIN c0 3\n', 'quit 0\n'),
        'echo silk-mc BEGIN c0 1\n', 'echo doAnalyses: TRAN: Timestep too small\n')
    with pytest.warns(RuntimeWarning, match='rerun one per process'):
        chunked = _run(_op(tmp_path), 4, spp=4, seed=1)
    assert list(chunked.failed) == [('tt', 1)]
    assert sorted(chunked.reports) == [('tt', 0), ('tt', 2), ('tt', 3)]
    assert chunked.rerun_samples == (('tt', 3),)
    message = chunked.failed[('tt', 1)].error
    assert 'sample 1' in message and 'simulator status 1' in message
    assert 'Timestep too small' in message
    # P2-3: its own, public, sample-local log; no rawfile published as current.
    folder = os.path.join(str(tmp_path), 'test_b~t', 'tt', 'mc_0001')
    log = open(os.path.join(folder, 'output.log')).read()
    assert 's0001.raw' in log and 's0000.raw' not in log and 's0002.raw' not in log
    assert not os.path.exists(os.path.join(folder, 'rawfile.raw'))


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_failed_sample_publishes_its_log_in_serial_and_worker_runs(tmp_path, patched_deck):
    patched_deck['edit'] = _status_edit(1, 1)
    for workers, sub in ((None, 's'), (2, 'w')):
        root = tmp_path / sub
        chunked = _run(_op(root), 2, spp=2, workers=workers, seed=1)
        assert list(chunked.failed) == [('tt', 1)]
        folder = root / 'test_b~t' / 'tt' / 'mc_0001'
        assert 's0001.raw' in (folder / 'output.log').read_text()
        assert not (folder / 'rawfile.raw').exists()


# -- equality controls beyond the DC-biased bench (P2-5) --------------------

def _rich(tmp_path, analyses, *, pulse=False, temperature=None, save=None,
          metric_analysis=None, cl_value=1e-12):
    """The MOS stage with a driven, coupled RC node; ``analyses`` is a list."""
    from silk.elements import elements
    small = any(isinstance(a, (AC, Noise)) for a in analyses)
    tb = _build(tmp_path, analyses[0], ac=1 if small else None,
                probe_analysis=metric_analysis,
                vout_metric=not any(isinstance(a, Noise) for a in analyses))
    for extra in analyses[1:]:
        tb.add_sim_type(extra)
    tb.add(elements.capacitor('cl', p='vout', n='0', value=cl_value))
    tb.add(elements.resistor('rc', p='vout', n='vc', value=5e3))
    tb.add(elements.capacitor('cc', p='vc', n='0', value=2e-12))
    if pulse:
        tb.add(sources.vpulse('vp', positive='vp', negative='0', v1=0, v2=1.8,
                              tr=1e-10, tf=1e-10, pw=2e-9, per=4e-9))
        tb.add(elements.resistor('rp', p='vp', n='vc', value=8e3))
    if temperature is not None:
        tb.set_temperature(temperature)
        tb.module.set_simulator_options(reltol=2e-4)
    if save is not None:
        tb.netlist.set_save_list(save)
    kw = {'analysis': metric_analysis} if metric_analysis else {}
    name = 'onoise_spectrum' if any(isinstance(a, Noise) for a in analyses) else 'vout'

    def level(ckt):
        return float(abs(np.asarray(ckt.load_signal(name, **kw).y).ravel()[-1]))

    tb.add_metric('level', level)
    return tb


CASES = {
    'tran_uic_driven_temp_options': lambda d: _rich(
        d, [Transient(step=0.1e-9, stop=5e-9, uic=True)], pulse=True, temperature=60),
    'tran_explicit_initial_value': lambda d: _rich(
        d, [Transient(step=0.1e-9, stop=5e-9, uic=True)], cl_value='1p IC=0.25'),
    'tran_restricted_saves': lambda d: _rich(
        d, [Transient(step=0.1e-9, stop=5e-9)], save=['vout', 'vc']),
    'dc_sweep': lambda d: _rich(d, [DC('vg', 0.6, 1.2, 0.2)]),
    'ac': lambda d: _rich(d, [AC('dec', 3, 1e3, 1e6)]),
    'noise': lambda d: _rich(d, [Noise('vout', '0', 'vg', 'dec', 3, 1e3, 1e6)]),
    'tran_and_ac': lambda d: _rich(
        d, [Transient(step=0.1e-9, stop=5e-9), AC('dec', 3, 1e3, 1e6)],
        metric_analysis='tran'),
}


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('case', sorted(CASES))
def test_chunked_equals_per_process_on_other_benches(tmp_path, case):
    per = _run(CASES[case](tmp_path / 'pp'), 2, spp=1, variation='both', seed=4,
               record_draws=True)
    chunked = _run(CASES[case](tmp_path / 'ch'), 2, spp=2, variation='both', seed=4,
                   record_draws=True)
    assert not per.failed and not chunked.failed, (per.failed, chunked.failed)
    assert chunked.rerun_samples == ()
    assert_same_campaign(per, chunked)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_restricted_saves_keep_their_vector_identity(tmp_path):
    chunked = _run(CASES['tran_restricted_saves'](tmp_path), 2, spp=2, seed=4)
    raw = _raw(chunked.reports[('tt', 1)])
    assert b'v(vout)' in raw.lower() or b'vout' in raw.lower()
    assert b'vd' not in raw.split(b'Binary:')[0].lower().replace(b'vdd', b'')


def _same_number(x, y):
    import math
    if x is None or y is None:
        return x is None and y is None
    return (math.isnan(x) and math.isnan(y)) or x == y


def _assert_same_sensitivity(a, b):
    assert (a.method, a.n, a.r_squared) == (b.method, b.n, b.r_squared)
    assert a.rows and len(a.rows) == len(b.rows)
    for ra, rb in zip(a.rows, b.rows):
        assert ra.name == rb.name
        for field in ('effect_per_sigma', 'share', 'correlation'):
            assert _same_number(getattr(ra, field), getattr(rb, field)), (ra, rb)
    assert len(a.correlated) == len(b.correlated)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_process_variation_alone_is_identical_chunked(tmp_path):
    per = _run(_tran(tmp_path / 'pp'), 2, spp=1, variation='process', seed=6)
    chunked = _run(_tran(tmp_path / 'ch'), 2, spp=2, variation='process', seed=6)
    assert not chunked.failed and chunked.rerun_samples == ()
    assert_same_campaign(per, chunked)


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('make', [_op, _tran])
@pytest.mark.parametrize('variation', ['mismatch', 'process'])
def test_campaign_draws_csv_and_sensitivity_are_identical_chunked(tmp_path, make, variation):
    kw = dict(variation=variation, seed=8, record_draws=True)
    per = _run(make(tmp_path / 'pp'), 6, spp=1, **kw)
    chunked = _run(make(tmp_path / 'ch'), 6, spp=3, **kw)
    assert not chunked.failed and chunked.rerun_samples == () and per.rerun_samples == ()
    assert per.draws is not None and chunked.draws is not None
    assert_same_campaign(per, chunked)       # results, draws, draws.csv, yield
    _assert_same_sensitivity(per.sensitivity('vout_V'), chunked.sensitivity('vout_V'))


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_test_that_cannot_be_certified_runs_one_per_process_with_one_warning(tmp_path, monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    monkeypatch.setattr(ModuleTestbench, '_chunk_admission_reason', lambda self: 'a stand-in reason')

    def forbidden(*a, **k):
        raise AssertionError('no multi-sample process may start')

    monkeypatch.setattr(mc_chunk, 'ChunkRun', forbidden)
    with pytest.warns(RuntimeWarning, match='a stand-in reason') as caught:
        mc = _run(_op(tmp_path), 2, spp=2, seed=2, allow_fallback=True)
    assert len([w for w in caught if 'ceiling' in str(w.message)]) == 1
    assert len(mc.reports) == 2 and not mc.failed and mc.rerun_samples == ()
    assert not list(tmp_path.rglob('.mc_chunk_*'))


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_duplicate_plot_member_fails_alone_without_aborting_the_campaign(tmp_path, monkeypatch):
    from silk.simulation.runner import NgspiceRunner
    real = NgspiceRunner.load

    def load(self, *a, **k):
        native = real(self, *a, **k)
        if 'mc_0001' in str(self.rawfile_path) or '.s0001' in str(self.rawfile_path):
            native.plots = list(native.plots) * 2
        return native

    monkeypatch.setattr(NgspiceRunner, 'load', load)
    chunked = _run(_tran(tmp_path), 3, spp=3, seed=1)
    assert list(chunked.failed) == [('tt', 1)]
    assert sorted(chunked.reports) == [('tt', 0), ('tt', 2)]
    assert chunked.rerun_samples == ()
    assert '2 tran plots' in chunked.failed[('tt', 1)].error


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_correlate_is_kept_and_its_ranking_refused_in_both_modes(tmp_path):
    from silk.exceptions import NotSupportedError
    results = []
    for spp, sub in ((1, 'pp'), (2, 'ch')):
        tb = _op(tmp_path / sub)
        mc = _run(tb, 2, spp=spp, seed=3, record_draws=True,
                  correlate=[tb.probe('vout')])
        assert len(mc.settings.correlate) == 1 and not mc.failed
        with pytest.raises(NotSupportedError, match='correlate'):
            mc.sensitivity('vout_V')
        results.append(mc)
    assert all(r.rerun_samples == () for r in results)
    assert_same_campaign(*results)


# -- the early decision (round 3) -------------------------------------------

def test_preflight_applies_the_certification_rules_without_emitting(tmp_path):
    from silk.simulation.completion import uncertifiable_reason
    assert uncertifiable_reason([AC('dec', 3, 1e3, 1e6)]) is None
    assert uncertifiable_reason([Transient(step=1e-9, stop=5e-9), AC('dec', 3, 1e3, 1e6)]) is None
    assert uncertifiable_reason([DC('vg', 0.6, 1.2, 0.2)]) is None
    assert uncertifiable_reason([Noise('vout', '0', 'vg', 'dec', 3, 1e3, 1e6)]) is None
    assert uncertifiable_reason([Op()]) is None
    bad = uncertifiable_reason([AC('lin', 2, 1, 1)])
    assert bad and 'unresolvable' in bad
    assert uncertifiable_reason([AC('dec', 3, 1e3, 1e6), AC('dec', 3, 1e3, 1e6)])  # repeated


@needs_pdk          # variation='both' resolves the real process library
def test_an_uncertifiable_request_is_routed_before_the_first_write(tmp_path, monkeypatch):
    import silk.circuit.module_test as module_test
    tb = _rich(tmp_path, [AC('lin', 2, 1, 1)])
    keep = tmp_path / 'test_b~t'
    keep.mkdir(exist_ok=True)
    (keep / 'keep.txt').write_bytes(b'existing output')
    before = _files(tmp_path)
    seen = {}

    class Stop(Exception):
        pass

    def first_write(*args, **kwargs):
        seen['files'] = _files(tmp_path)
        raise Stop

    monkeypatch.setattr(module_test, '_write_wrapper', first_write)
    with pytest.warns(RuntimeWarning, match='ceiling.*unresolvable') as caught:
        with pytest.raises(Stop):
            tb.run_monte_carlo(2, samples_per_process=2, variation='both')
    assert seen['files'] == before                   # nothing written or replaced
    assert len([w for w in caught if 'ceiling' in str(w.message)]) == 1


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_a_deck_found_uncertifiable_late_falls_back_once_and_keeps_its_reason(tmp_path, monkeypatch):
    import silk.simulation.completion as completion
    monkeypatch.setattr(completion, 'certify_generated',
                        lambda module, text: (None, 'a stand-in late reason'))
    with pytest.warns(RuntimeWarning) as caught:
        mc = _run(_tran(tmp_path), 3, spp=3, seed=1)
    texts = [str(w.message) for w in caught if 'rerun one per process' in str(w.message)]
    assert len(texts) == 1
    assert 'fell back before one launched' in texts[0] and 'a stand-in late reason' in texts[0]
    assert mc.rerun_samples == (('tt', 0), ('tt', 1), ('tt', 2))
    assert len(mc.reports) == 3 and not mc.failed
    assert 'fell back' in mc.summary()
    assert not list(tmp_path.rglob('chunk.log'))      # no multi-sample process ran


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('spp,workers', [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize('variation,corners', [('mismatch', ['tt', 'ss']),
                                             ('both', ['tt'])])
def test_new_campaign_keeps_both_variation_files(tmp_path, spp, workers, variation, corners):
    from pathlib import Path
    tb = _op(tmp_path)
    tb.set_corners(corners)
    baseline = _run(tb, 2, spp=spp, workers=workers, variation=variation,
                    record_draws=True, seed=4)
    first = _run(tb, 2, spp=spp, workers=workers, variation=variation,
                 record_draws=True, seed=4, output_mode='new')
    roots = {corner: Path(first[(corner, 0)].run.deck_path).parent.parent
             for corner in corners}
    saved = {p: p.read_bytes() for root in roots.values()
             for p in root.rglob('*') if p.is_file()
             and not any(part.startswith('.silk') for part in p.relative_to(root).parts)}
    second = _run(tb, 2, spp=spp, workers=workers, variation=variation,
                  record_draws=True, seed=4, output_mode='new')
    assert not baseline.failed and not first.failed and not second.failed
    assert {k: r.metrics for k, r in baseline.items()} == {k: r.metrics for k, r in first.items()}
    assert {k: r.metrics for k, r in second.items()} == {k: r.metrics for k, r in first.items()}
    assert baseline.draws.columns == first.draws.columns == second.draws.columns
    assert baseline.draws.raw == first.draws.raw == second.draws.raw
    for key in baseline.draws.values:
        np.testing.assert_array_equal(baseline.draws.values[key], first.draws.values[key])
        np.testing.assert_array_equal(baseline.draws.values[key], second.draws.values[key])
    for corner, root in roots.items():
        assert root.name.startswith('run-') and root.parent.name == corner
        assert (root / 'draws.csv').is_file()
        assert (root / 'mc_both.lib.spice').is_file() == (variation == 'both')
        assert Path(first[(corner, 1)].run.deck_path).parent.parent == root
        assert Path(second[(corner, 0)].run.deck_path).parent.parent != root
    assert all(p.read_bytes() == data for p, data in saved.items())


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_labelled_declared_both_campaign_retains_all_files(tmp_path):
    from pathlib import Path
    tb = _op(tmp_path)
    tb.set_monte_carlo(2, variation='both', seed=4, record_draws=True,
                       output_mode='new')
    first = tb.run(run_label='baseline')
    root = Path(first[('tt', 0)].run.deck_path).parent.parent
    assert root.parent.parent.name.endswith('~run~baseline')
    assert root.parent.name == 'tt' and root.name.startswith('run-')
    wrapper = root / 'mc_both.lib.spice'
    assert wrapper.is_file() and (root / 'draws.csv').is_file()
    unlabelled = root.parent.parent.with_name(root.parent.parent.name.removesuffix('~run~baseline'))
    assert not list(unlabelled.rglob('run-*'))
    for index in range(2):
        report = first[('tt', index)]
        for path in (report.run.deck_path, report.run.log_path, report.run.rawfile_path):
            assert Path(path).is_file() and Path(path).parent.parent == root
        assert f'.lib "{wrapper}" mc_both\n' in Path(report.run.deck_path).read_text()
    saved = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()
             and not any(part.startswith('.silk') for part in p.relative_to(root).parts)}
    second = tb.run(run_label='baseline')
    assert Path(second[('tt', 0)].run.deck_path).parent.parent != root
    assert {key: r.metrics for key, r in first.items()} == {key: r.metrics for key, r in second.items()}
    for key in first.draws.values:
        np.testing.assert_array_equal(first.draws.values[key], second.draws.values[key])
    assert all(p.read_bytes() == data for p, data in saved.items())
