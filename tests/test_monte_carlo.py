"""Monte Carlo run wiring (TASK-427, Monte Carlo plan task 3c).

No simulator: ``ModuleTestbench._simulate_run`` is replaced by a fake that
writes the real deck (so its section and seed lines can be read back) and a
small rawfile, then returns a real ``_RunSimResult``; the real
``_evaluate_run``, metric extraction, planning and collection run. The
model library's section check is stubbed so the fast suite needs no PDK
install; ``test_the_section_check_runs_before_the_first_sample`` pins where
it is called.
"""
import os

import pytest

from tests._mock_rawfile import write_complete_rawfile

from silk.circuit.monte_carlo import MonteCarloResults, derive_sample_seed
from silk.exceptions import (MissingLibrarySectionError, MonteCarloError,
                             NotSupportedError, SimulationError)


def _gain(corner, index):
    base = {'tt': 10.0, 'ss': 8.0, 'ff': 12.0}[corner]
    return base + (0.0 if index is None else 0.25 * index)


def _metric(ckt):
    sample = ckt._mc_sample
    return _gain(ckt._last_sim_label, None if sample is None else sample.index)


@pytest.fixture
def events(monkeypatch):
    """Ordered record of section checks and simulator launches."""
    from silk.pdk import statistical
    log = []

    def check(pdk, sections, *, simulator='ngspice'):
        log.append(('check', tuple(sections), simulator))
    monkeypatch.setattr(statistical, 'require_library_sections', check)
    return log


def _bench(tmp_path, pdk_name, device):
    from silk import load_pdk, sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.simulation import analysis
    pdk = load_pdk(pdk_name)
    mod = Module(pdk_name, str(tmp_path), 'amp.cir', pdk_details=pdk,
                 module_name='amp')
    mod.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                             's': PortDir.GROUND})
    if device:
        mod.add(getattr(pdk, device)('M1', d='d', g='g', s='s', b='s',
                                     w=1.0, l=0.5))
    else:
        from silk.elements import elements
        mod.add(elements.resistor('r1', p='d', n='s', value=1e3))
    tb = mod.create_test('op')
    tb.add_module_under_test({'d': 'vout', 'g': 'vg', 's': '0'})
    tb.add(sources.vdc('vd', positive='vout', negative='0', value=1.0))
    tb.add(sources.vdc('vg', positive='vg', negative='0', value=1.0))
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=2e-9))
    tb.add_metric('gain', _metric)
    tb.add_spec('gain_ok', 'gain', '>', 8.5)
    tb.first_spec_name = 'gain_ok'
    return tb


@pytest.fixture
def sky130_tb(tmp_path, events):
    return _bench(tmp_path, 'sky130', 'nmos1p8')


@pytest.fixture
def gf180_tb(tmp_path, events):
    return _bench(tmp_path, 'gf180mcu', 'nmos3p3')


@pytest.fixture
def ideal_rc_tb(tmp_path, events):
    return _bench(tmp_path, 'sky130', None)


class _FakeSim:
    def __init__(self, events):
        self.events = events
        self.failing = set()
        self.decks = {}

    def fail_on(self, *keys):
        self.failing.update(keys)

    def deck_text(self, key):
        with open(self.decks[key]) as f:
            return f.read()

    @staticmethod
    def change_a_device_width(tb):
        tb.module.elem_dict['M1']['device'].params['w'] = 2.0


@pytest.fixture
def fake_sim(monkeypatch, events):
    from silk.circuit.module_test import ModuleTestbench, _RunSimResult
    from silk.simulation.conditions import (OperatingConditions,
                                            _launched_corner)
    fake = _FakeSim(events)

    def simulate(spec):
        netlist = spec.netlist
        sample = netlist._mc_sample
        key = (spec.corner_str, None if sample is None else sample.index)
        events.append(('sim', key))
        netlist_deck_path = netlist.generate_netlist(corner=spec.corner_str)
        fake.decks[key] = netlist_deck_path
        if key in fake.failing:
            raise SimulationError(f'fake failure at {key}')
        raw = os.path.join(spec.tb_dir, 'fake.raw')
        from pathlib import Path
        write_complete_rawfile(raw, Path(netlist_deck_path).read_text(), values={'v(vout)': _gain(*key)})
        netlist.results_file = raw
        netlist._raw = netlist.backend.load_raw(raw)
        netlist.simResultsAll = netlist._raw.native
        conditions = OperatingConditions(
            'silk', netlist.simulator, **_launched_corner(spec.corner_str, False))
        netlist.operating_conditions = conditions
        netlist.sim_results_loaded = True
        return _RunSimResult(netlist=netlist, corner_str=spec.corner_str,
                             conditions=conditions, invocation=spec.invocation,
                             cache_key=spec.cache_key)

    monkeypatch.setattr(ModuleTestbench, '_simulate_run', staticmethod(simulate))
    original = ModuleTestbench._prepare_run

    def prepare(self, *args, **kwargs):
        spec = original(self, *args, **kwargs)
        sample = kwargs.get('sample')
        key = (spec.corner_str, None if sample is None else sample.index)
        fake.decks[key] = os.path.join(spec.netlist.dir_folder, spec.netlist.netlistname)
        events.append(('prepare', key))
        return spec
    monkeypatch.setattr(ModuleTestbench, '_prepare_run', prepare)
    return fake


def _key(tb):
    return f'{tb.module.module_name}.{tb.test_name}'


def _sample_dirs(tb):
    found = []
    for root, dirs, _files in os.walk(tb.module.dir_folder):
        found += [os.path.join(root, d) for d in dirs if d.startswith('mc_')]
    return found


# --- the setting ------------------------------------------------------------

@pytest.mark.parametrize('kwargs', [dict(n=0), dict(n=-1), dict(n=2.5),
                                    dict(n=10, min_yield=0),
                                    dict(n=10, min_yield=1.5),
                                    dict(n=10, variation='global')])
def test_the_setter_validates_through_the_settings(sky130_tb, kwargs):
    with pytest.raises(MonteCarloError):
        sky130_tb.set_monte_carlo(**kwargs)
    assert sky130_tb.monte_carlo is None


def test_the_setter_stores_returns_self_and_none_clears(sky130_tb):
    assert sky130_tb.set_monte_carlo(5, seed=3) is sky130_tb
    assert (sky130_tb.monte_carlo.n, sky130_tb.monte_carlo.seed) == (5, 3)
    sky130_tb.set_monte_carlo(None)
    assert sky130_tb.monte_carlo is None


# --- refusals before any deck is written ------------------------------------

def _refuses_before_any_deck(tb, error, match, events):
    with pytest.raises(error, match=match):
        tb.run()
    assert _sample_dirs(tb) == []
    assert not any(e[0] in ('prepare', 'sim') for e in events)


def test_process_variation_off_typical_raises(sky130_tb, fake_sim, events):
    sky130_tb.set_corners(['ss'])
    sky130_tb.set_monte_carlo(n=5, variation='process')
    _refuses_before_any_deck(sky130_tb, MonteCarloError, "'tt'", events)


def test_both_variation_off_typical_raises(sky130_tb, fake_sim, events):
    sky130_tb.set_corner('ff')
    sky130_tb.set_monte_carlo(n=5, variation='both')
    _refuses_before_any_deck(sky130_tb, MonteCarloError, "'tt'", events)


def test_pdk_free_deck_raises(ideal_rc_tb, fake_sim, events):
    ideal_rc_tb.set_monte_carlo(n=5)
    _refuses_before_any_deck(ideal_rc_tb, MonteCarloError, 'no PDK device',
                             events)


def test_a_pdk_without_statistics_raises_naming_it(gf180_tb, fake_sim, events):
    gf180_tb.set_monte_carlo(n=2)
    _refuses_before_any_deck(gf180_tb, NotSupportedError, 'gf180', events)


def test_duplicate_spec_names_raise(sky130_tb, fake_sim, events):
    sky130_tb.add_spec(sky130_tb.first_spec_name, 'gain', '>', 0.0,
                       key='second')
    sky130_tb.set_monte_carlo(n=2)
    _refuses_before_any_deck(sky130_tb, MonteCarloError,
                             sky130_tb.first_spec_name, events)


def test_a_missing_library_section_raises_before_any_deck(
        sky130_tb, fake_sim, events, monkeypatch):
    from silk.pdk import statistical

    def missing(pdk, sections, *, simulator='ngspice'):
        raise MissingLibrarySectionError(f'no {sections[0]!r} section')
    monkeypatch.setattr(statistical, 'require_library_sections', missing)
    sky130_tb.set_monte_carlo(n=2)
    _refuses_before_any_deck(sky130_tb, MissingLibrarySectionError, 'tt_mm',
                             events)


def test_the_section_check_runs_before_the_first_sample(
        sky130_tb, fake_sim, events):
    sky130_tb.set_corners(['tt', 'ss'])
    sky130_tb.set_monte_carlo(n=2)
    sky130_tb.run()
    assert events[0] == ('check', ('tt_mm', 'ss_mm'), 'ngspice')
    assert [e[0] for e in events].count('check') == 1


def test_store_true_refuses_declared_runs(sky130_tb, fake_sim, events):
    sky130_tb.set_monte_carlo(n=5)
    with pytest.raises(MonteCarloError, match='store'):
        sky130_tb.module.run_test(sky130_tb.test_name, store=True)
    sky130_tb.set_monte_carlo(None)
    sky130_tb.set_corners(['tt', 'ss'])
    with pytest.raises(MonteCarloError, match='store'):
        sky130_tb.module.run_test(sky130_tb.test_name, store=True)
    assert not any(e[0] == 'sim' for e in events)


# --- decks, folders and results ---------------------------------------------

def test_each_sample_deck_carries_its_section_and_seed(sky130_tb, fake_sim):
    sky130_tb.set_corners(['ss'])
    sky130_tb.set_monte_carlo(n=2, seed=0)
    sky130_tb.run()
    deck = fake_sim.deck_text(('ss', 1))
    assert ' ss_mm\n' in deck
    assert f'.options seed={derive_sample_seed(0, 1)}' in deck


def test_process_and_both_decks(sky130_tb, fake_sim):
    st = sky130_tb.pdk_details.statistical
    sky130_tb.run_monte_carlo(1, variation='process', seed=4)
    assert f' {st.process_section}\n' in fake_sim.deck_text(('tt', 0))
    sky130_tb.run_monte_carlo(1, variation='both', seed=4)
    deck = fake_sim.deck_text(('tt', 0))
    wrapper = os.path.join(os.path.dirname(os.path.dirname(
        fake_sim.decks[('tt', 0)])), 'mc_both.lib.spice')
    assert f'.lib "{wrapper}" mc_both\n' in deck
    assert os.path.isfile(wrapper)


def test_a_plain_run_after_monte_carlo_has_no_sample_lines(sky130_tb, fake_sim):
    sky130_tb.run_monte_carlo(2)
    report = sky130_tb.run()
    deck = fake_sim.deck_text(('tt', None))
    assert 'seed' not in deck.lower() and '_mm' not in deck
    assert report.metrics['gain'] == _gain('tt', None)


@pytest.mark.parametrize('n_workers', [None, 2])
def test_run_all_tests_returns_every_sample(sky130_tb, fake_sim, n_workers):
    sky130_tb.set_corners(['tt', 'ss'])
    sky130_tb.set_monte_carlo(n=3)
    res = sky130_tb.module.run_all_tests(n_workers=n_workers)
    mc = res[_key(sky130_tb)]
    assert isinstance(mc, MonteCarloResults)
    assert sorted(mc.reports) == [('ss', 0), ('ss', 1), ('ss', 2),
                                  ('tt', 0), ('tt', 1), ('tt', 2)]
    assert {k: r.metrics['gain'] for k, r in mc.reports.items()} == {
        k: _gain(*k) for k in mc.reports}
    assert mc.seeds == {k: derive_sample_seed(0, k[1]) for k in mc.reports}
    assert all(r.corner == k[0] for k, r in mc.reports.items())


def test_samples_are_prepared_in_worker_sized_batches(sky130_tb, fake_sim,
                                                      monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    batches = []
    original = ModuleTestbench._prepare_plan

    def spy(self, batch, **kwargs):
        batches.append([run.key for run in batch])
        return original(self, batch, **kwargs)
    monkeypatch.setattr(ModuleTestbench, '_prepare_plan', spy)
    sky130_tb.set_monte_carlo(n=5)
    sky130_tb.module.run_all_tests(n_workers=2)
    assert batches == [[('tt', 0), ('tt', 1)], [('tt', 2), ('tt', 3)],
                       [('tt', 4)]]
    batches.clear()
    sky130_tb.run()        # a plain run() simulates one sample at a time
    assert batches == [[('tt', 0)], [('tt', 1)], [('tt', 2)], [('tt', 3)],
                       [('tt', 4)]]


def test_run_monte_carlo_leaves_the_declared_setting(sky130_tb, fake_sim):
    sky130_tb.set_monte_carlo(7, seed=9)
    mc = sky130_tb.run_monte_carlo(2, seed=1)
    assert sorted(mc.reports) == [('tt', 0), ('tt', 1)]
    assert mc.settings.seed == 1
    assert (sky130_tb.monte_carlo.n, sky130_tb.monte_carlo.seed) == (7, 9)


def test_monte_carlo_without_a_corner_list_uses_the_test_corner(
        sky130_tb, fake_sim):
    sky130_tb.set_corner('ff')
    sky130_tb.set_monte_carlo(n=2)
    assert sorted(sky130_tb.run().reports) == [('ff', 0), ('ff', 1)]
    assert sorted(sky130_tb.run(corner='ss').reports) == [('ss', 0), ('ss', 1)]


def test_sample_folders_ignore_output_mode(sky130_tb, fake_sim):
    sky130_tb.module.set_output_options(mode='new')
    sky130_tb.set_monte_carlo(n=2)
    first = sky130_tb.run()
    second = sky130_tb.run()
    for key in first.reports:
        assert first.reports[key].rawfile_path == second.reports[key].rawfile_path
        assert f'/tt/mc_{key[1]:04d}/' in first.reports[key].rawfile_path
    assert len(_sample_dirs(sky130_tb)) == 2


def test_a_failed_sample_is_kept_out_and_counted(sky130_tb, fake_sim):
    fake_sim.fail_on(('tt', 1))
    sky130_tb.set_monte_carlo(n=4)
    mc = sky130_tb.run()
    assert ('tt', 1) not in mc.reports and ('tt', 1) in mc.failed
    assert mc.failed[('tt', 1)].seed == derive_sample_seed(0, 1)
    assert 'fake failure' in mc.failed[('tt', 1)].error
    assert '1 failed' in mc.summary()


@pytest.mark.parametrize('through_module,n_workers',
                         [(False, None), (True, None), (True, 2)])
def test_every_sample_failing_raises(sky130_tb, fake_sim, n_workers,
                                    through_module):
    import pickle
    from silk.circuit.monte_carlo import SampleFailure
    fake_sim.fail_on(('tt', 0), ('tt', 1))
    sky130_tb.set_monte_carlo(n=2)
    if through_module:
        result = sky130_tb.module.run_all_tests(n_workers=n_workers)
        failure = next(iter(result.values()))
        assert failure.exception_is_original
        error = pickle.loads(pickle.dumps(failure)).exception
    else:
        with pytest.raises(SimulationError,
                           match='every declared run failed') as caught:
            sky130_tb.run()
        error = pickle.loads(pickle.dumps(caught.value))
    assert type(error) is SimulationError
    assert list(error.failed) == [('tt', 0), ('tt', 1)]
    for index in (0, 1):
        entry = error.failed[('tt', index)]
        assert isinstance(entry, SampleFailure)
        assert entry.seed == derive_sample_seed(0, index)
        assert f"fake failure at ('tt', {index})" in entry.error
    assert str(error).startswith("test 'op': every declared run failed")
    assert "1 other failure(s)" in str(error)


def test_provenance_records_the_sample(sky130_tb, fake_sim):
    sky130_tb.set_corners(['ss'])
    mc = sky130_tb.run_monte_carlo(2, seed=5)
    prov = mc.reports[('ss', 1)].provenance
    assert prov.seed == derive_sample_seed(5, 1)
    assert prov.seedinfo == ('monte-carlo variation=mismatch section=ss_mm '
                             'sample=1')
    assert mc.reports[('ss', 1)].corner == 'ss'


def test_rerun_reproduces_a_sample(sky130_tb, fake_sim):
    mc = sky130_tb.run_monte_carlo(3)
    report = mc.rerun(('tt', 2))
    assert report.metrics['gain'] == mc.reports[('tt', 2)].metrics['gain']
    assert report.rawfile_path == mc.reports[('tt', 2)].rawfile_path
    assert f'seed={derive_sample_seed(0, 2)}' in fake_sim.deck_text(('tt', 2))


def test_rerun_refuses_a_changed_circuit(sky130_tb, fake_sim):
    sky130_tb.set_monte_carlo(n=2)
    mc = sky130_tb.run()
    fake_sim.change_a_device_width(sky130_tb)
    with pytest.raises(MonteCarloError, match='changed'):
        mc.rerun(('tt', 0))


@pytest.mark.parametrize('selected', [False, True])
def test_probe_reads_one_sample_and_corner_refuses(sky130_tb, fake_sim, tmp_path, selected):
    from silk.circuit.modules import Module
    top = Module('sky130', str(tmp_path), 'top.cir', sky130_tb.pdk_details)
    top.add_module(sky130_tb.module, 'child', d='d', g='g', s='0')
    tb = top.get_test(sky130_tb.test_name, instance='child') if selected else sky130_tb
    vout = tb.probe('vout')
    tb.set_monte_carlo(n=2)
    tb.run()
    assert list(vout.sample(('tt', 1)).y) == [_gain('tt', 1)] * 2
    assert list(vout.sample(('tt', 0)).y) == [_gain('tt', 0)] * 2
    with pytest.raises(Exception, match=r'\.sample\('):
        vout.corner('tt')
    with pytest.raises(Exception, match='no sample'):
        vout.sample(('tt', 7))
    top.close()


def test_sample_refuses_without_a_monte_carlo_run(sky130_tb, fake_sim):
    vout = sky130_tb.probe('vout')
    sky130_tb.set_corners(['tt', 'ss'])
    sky130_tb.run()
    with pytest.raises(Exception, match='no Monte Carlo samples'):
        vout.sample(('tt', 0))
    assert list(vout.corner('ss').y) == [_gain('ss', None)] * 2


def test_samples_record_no_design_stage_measurement(sky130_tb, fake_sim):
    sky130_tb.set_monte_carlo(n=3)
    sky130_tb.run()
    specs = [s['spec'] for s in sky130_tb._specs]
    assert specs and all(not spec.measurements for spec in specs)
    sky130_tb.set_monte_carlo(None)
    sky130_tb.run()
    assert all(spec.measurements for spec in specs)


@pytest.mark.parametrize('selected', [False, True])
def test_retained_sample_rerun_restores_its_original_selection(
        sky130_tb, fake_sim, tmp_path, selected):
    from silk.circuit.modules import Module
    tb = sky130_tb
    top = Module('sky130', str(tmp_path), 'top.cir', tb.pdk_details)
    for name in ('a', 'b'):
        top.add_module(tb.module, name, d=f'd_{name}', g=f'g_{name}', s='0')
    a, b = top.list_test_entries()
    origin = a.test if selected else tb
    owner = origin.owner_path
    mc = origin.run_monte_carlo(2)
    b.test.run()
    for sample in (1, 0):
        replay = mc.rerun(('tt', sample))
        assert replay.owner_path == owner
        assert replay.metrics == mc.reports[('tt', sample)].metrics
        assert b.test.owner_path == b.owner_path
    assert tb.owner_path == '$'
    top.close()


@pytest.mark.parametrize('n_workers', [None, 2])
def test_collection_captures_registered_inventory(sky130_tb, fake_sim, monkeypatch, n_workers):
    import math
    from silk.metrics.diagnostics import MetricWarning
    from silk.signals import schema as S
    from tests.test_multi_output_metrics import Pair
    from silk.circuit.module_test import ModuleTestbench

    monkeypatch.setattr(S, '_REGISTRY', S._REGISTRY.copy())

    @S.metric(domain='tran', outputs={'height': 'V', 'width': 's'})
    def inventory_pair(sig: S.Transient) -> Pair:
        return Pair(2.0, 3.0)

    sky130_tb.add_metric('eye', inventory_pair, sig=sky130_tb.probe('vout'))
    sky130_tb.add_metric('unspecced', lambda nr: 7.0)
    sky130_tb.add_spec('wide', 'eye.width', '>', 1.0)
    sky130_tb.set_corners(['tt', 'ss'])
    sky130_tb.set_monte_carlo(n=1)
    fake_sim.fail_on(('ss', 0))
    evaluate = ModuleTestbench._evaluate_run

    def partial_report(self, *args, **kwargs):
        report = evaluate(self, *args, **kwargs)
        # Collection must use declarations even if no successful observation
        # contains the metric/check (for example incomplete external evidence).
        report.metrics.clear()
        report._results.clear()
        return report

    monkeypatch.setattr(ModuleTestbench, '_evaluate_run', partial_report)
    mc = sky130_tb.module.run_all_tests(n_workers=n_workers)[_key(sky130_tb)]
    ss = mc.corner('ss')
    assert set(ss.yield_fraction()) == {'gain_ok', 'wide'}
    assert all(fraction == 0.0 for fraction, _ in ss.yield_fraction().values())
    for name in ('gain', 'unspecced', 'eye.width', 'eye.height'):
        with pytest.warns(MetricWarning, match='every sample failed'):
            assert math.isnan(ss.stat(name, 'mean'))
    assert set(ss.table()[0]) >= {'metric.gain', 'metric.unspecced', 'metric.eye', 'metric.eye.width', 'metric.eye.height'}


def test_collection_rejects_report_conflicting_with_declaration(sky130_tb, fake_sim, monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    evaluate = ModuleTestbench._evaluate_run

    def conflicting_report(self, *args, **kwargs):
        report = evaluate(self, *args, **kwargs)
        report._results[0].target = 100.0
        return report

    monkeypatch.setattr(ModuleTestbench, '_evaluate_run', conflicting_report)
    with pytest.raises(ValueError) as caught:
        sky130_tb.run_monte_carlo(1)
    message = str(caught.value)
    assert 'gain_ok' in message and '8.5' in message and '100.0' in message


@pytest.mark.parametrize('n_workers', [1, 2])
@pytest.mark.parametrize('replacement_started', [False, True])
def test_selected_campaign_refusal_and_rerun_keep_publication_rules(
        sky130_tb, fake_sim, tmp_path, monkeypatch, n_workers, replacement_started):
    from silk import Module
    from silk.exceptions import MonteCarloError
    tb = sky130_tb
    from silk.circuit.module_test import ModuleTestbench
    simulate = ModuleTestbench._simulate_run

    def loaded_result(spec):
        result = simulate(spec)
        # Transport the same loaded waveform state a real launch publishes.
        path = result.netlist.results_file
        result.netlist.load_file(path)
        result.netlist.read_raw_file()
        result.netlist.results_file = path
        return result

    monkeypatch.setattr(ModuleTestbench, '_simulate_run', staticmethod(loaded_result))
    top = Module('sky130', str(tmp_path), 'selected.cir', tb.pdk_details)
    top.add_module(tb.module, 'a', d='out', g='gate', s='0')
    selected = top.get_test('op', instance='a')
    probe = selected.probe('vout')
    try:
        campaign = selected.run_monte_carlo(2, n_workers=n_workers)
        values = [probe.sample(('tt', i)).y[-1] for i in (0, 1)]
        with pytest.raises(MonteCarloError):
            selected.run_monte_carlo(0)
        assert selected.report is campaign
        assert [probe.sample(('tt', i)).y[-1] for i in (0, 1)] == values
        assert all(not spec.measurements for spec in (entry['spec'] for entry in selected._specs))
        with pytest.raises(MonteCarloError):
            campaign.rerun(('tt', 99))
        assert selected.report is campaign
        replay = campaign.rerun(('tt', 0))
        assert selected.report is replay
        assert replay.metrics == campaign.reports[('tt', 0)].metrics
        assert all(not spec.measurements for spec in (entry['spec'] for entry in selected._specs))

        def fail(spec):
            if replacement_started:
                from silk.circuit.modules_sim import _begin_result_replacement
                _begin_result_replacement(spec.netlist)
            raise SimulationError('rerun boundary control')

        monkeypatch.setattr(tb, '_simulate_run', fail)
        with pytest.raises(SimulationError, match='rerun boundary control'):
            campaign.rerun(('tt', 0))
        assert selected.report is (None if replacement_started else replay)
        assert all(not spec.measurements for spec in (entry['spec'] for entry in selected._specs))
    finally:
        top.close()


@pytest.mark.parametrize('method', ['set_monte_carlo', 'run_monte_carlo'])
@pytest.mark.parametrize('mode', ['invalid', None, '', 1, [], {}])
def test_invalid_output_mode(sky130_tb, method, mode):
    with pytest.raises(ValueError, match="'overwrite'.*'new'"):
        getattr(sky130_tb, method)(2, output_mode=mode)
    sky130_tb.set_monte_carlo(None, output_mode='ignored')
    assert sky130_tb.monte_carlo is None


def test_new_campaign_keeps_corner_artifacts(sky130_tb, fake_sim):
    from pathlib import Path
    tb = sky130_tb
    tb.module.set_output_options(mode='new')
    tb.set_corners(['tt', 'ss'])
    baseline = tb.set_monte_carlo(2, seed=13).run()
    baseline_paths = dict(fake_sim.decks)
    repeated = tb.run()
    assert dict(fake_sim.decks) == baseline_paths
    tb.set_monte_carlo(2, seed=13, output_mode='new')
    first = tb.run()
    first_paths = dict(fake_sim.decks)
    folders = {corner: Path(first_paths[(corner, 0)]).parent.parent
               for corner in ('tt', 'ss')}
    snapshots = {p: p.read_bytes() for root in folders.values()
                 for p in root.rglob('*') if p.is_file()}
    second = tb.run()
    for corner, root in folders.items():
        assert root.name.startswith('run-')
        assert root.parent.name == corner
        assert Path(first_paths[(corner, 1)]).parent.parent == root
        assert Path(fake_sim.decks[(corner, 0)]).parent.parent != root
    assert all(p.read_bytes() == data for p, data in snapshots.items())
    assert {k: r.metrics for k, r in first.items()} == {k: r.metrics for k, r in baseline.items()}


@pytest.mark.parametrize('variation', ['mismatch', 'both'])
@pytest.mark.parametrize('output_mode', ['overwrite', 'new'])
def test_labelled_declared_campaign_placement(sky130_tb, fake_sim, variation, output_mode):
    from pathlib import Path
    tb = sky130_tb.set_monte_carlo(2, variation=variation, output_mode=output_mode)
    result = tb.run(run_label='baseline')
    assert not result.failed
    deck, = Path(tb.module.dir_folder).glob('**/mc_0000/*.cir')
    root = deck.parent.parent
    corner = root.parent if output_mode == 'new' else root
    assert corner == Path(tb.module.dir_folder) / 'test_amp~op~run~baseline' / 'tt'
    assert not list((Path(tb.module.dir_folder) / 'test_amp~op').rglob('run-*'))
    for index in range(2):
        sample_deck, = root.glob(f'mc_{index:04d}/*.cir')
        assert sample_deck.parent.parent == root
        if variation == 'both':
            wrapper = root / 'mc_both.lib.spice'
            assert wrapper.is_file()
            assert f'.lib "{wrapper}" mc_both\n' in sample_deck.read_text()


@pytest.mark.parametrize('variation', ['mismatch', 'both'])
def test_invalid_campaign_label_allocates_no_run_folder(sky130_tb, fake_sim, variation):
    from pathlib import Path
    from silk.exceptions import OutputPathError
    sky130_tb.set_monte_carlo(2, variation=variation, output_mode='new')
    with pytest.raises(OutputPathError):
        sky130_tb.run(run_label='../invalid')
    assert not list(Path(sky130_tb.module.dir_folder).rglob('run-*'))


def test_grid_campaign_retains_order_seeds_folders_and_replay(sky130_tb, fake_sim):
    tb = sky130_tb.set_corners(['ss', 'tt'])
    tb.set_temperatures([125, -40]).set_supply_voltages([1.62, 1.98], source='vd')
    tb.set_monte_carlo(2, seed=17)
    result = tb.run()
    points = [(c, t, v) for c in ['ss', 'tt'] for t in [125.0, -40.0] for v in [1.62, 1.98]]
    expected = [(point, i) for point in points for i in range(2)]
    assert list(result.reports) == expected
    assert result._keys() == expected
    assert result.corners == ['ss', 'tt']
    assert all(result.seeds[k] == derive_sample_seed(17, k[1]) for k in expected)
    assert len({r.rawfile_path for r in result.reports.values()}) == 16
    assert 'temperature-125.0/supply-1.62' in result[expected[0]].rawfile_path
    assert tb._sample_corner(expected[0]) == expected[0]
    assert result.rerun(expected[0]).metrics == result[expected[0]].metrics
    tb.set_temperatures([27])
    with pytest.raises(MonteCarloError, match='condition axes'):
        result.rerun(expected[0])


def test_corner_only_campaign_keeps_its_existing_banner(sky130_tb, fake_sim, capsys):
    sky130_tb.set_corners(['ss','tt']).set_monte_carlo(1).run()
    assert 'Monte Carlo: 2 corners × 1 samples = 2 simulations' in capsys.readouterr().out


@pytest.mark.parametrize('n_workers',[None,2])
@pytest.mark.parametrize('output_mode',['overwrite','new'])
def test_grid_campaign_run_all_keeps_plan_order_and_point_folders(
        sky130_tb,fake_sim,n_workers,output_mode):
    from pathlib import Path
    tb=sky130_tb.set_corners(['ss','tt'])
    tb.set_temperatures([125,-40]).set_supply_voltages([1.62],source='vd')
    tb.set_monte_carlo(2,seed=17,output_mode=output_mode)
    fake_sim.fail_on(('ss',0))
    result=tb.module.run_all_tests(n_workers=n_workers,run_label='grid')[_key(tb)]
    keys=[((c,t,1.62),i) for c in ['ss','tt'] for t in [125.0,-40.0] for i in range(2)]
    assert result._keys()==keys
    assert list(result.failed)==[(('ss',125.0,1.62),0),(('ss',-40.0,1.62),0)]
    assert [row['run.status'] for row in result.table()]==['failed','FAIL','failed','FAIL','PASS','PASS','PASS','PASS']
    for (point,index),report in result.items():
        path=Path(report.rawfile_path)
        assert path.parent.name==f'mc_{index:04d}'
        campaign=path.parent.parent
        point_dir=campaign.parent if output_mode=='new' else campaign
        assert point_dir.name=='supply-1.62'
        assert point_dir.parent.name=='temperature-'+repr(point[1])
        assert point_dir.parent.parent.name==point[0]
        assert 'run~grid' in str(path)
        assert result.seeds[(point,index)]==derive_sample_seed(17,index)


@pytest.mark.parametrize('shared', [False, True])
def test_sample_expression_keeps_owners_separate(tmp_path, events, fake_sim, monkeypatch, shared):
    import sys
    import numpy as np
    from silk import Module
    first = _bench(tmp_path / 'first', 'sky130', 'nmos1p8')
    second = first if shared else _bench(tmp_path / 'second', 'sky130', 'nmos1p8')
    top = Module(None, str(tmp_path / 'top'), 'top.cir', None)
    try:
        if shared:
            top.add_module(first.module, 'one', d='a', g='g', s='0')
            top.add_module(first.module, 'two', d='b', g='g', s='0')
            benches = [top.get_test('op', instance=name) for name in ('one', 'two')]
        else:
            benches = [first, second]
        for bench, value in zip(benches, (1.0, 9.0)):
            monkeypatch.setattr(sys.modules[__name__], '_gain', lambda *args, value=value: value)
            bench.set_monte_carlo(n=1, output_mode="new")
            bench.run()
        a, b = [bench.probe('vout') for bench in benches]
        key = ('tt', 0)
        for left, right, difference in [(a, b, -8), (b, a, 8)]:
            np.testing.assert_allclose((left.sample(key) + right.sample(key)).y, 10)
            np.testing.assert_allclose((left + right).sample(key).y, 10)
            np.testing.assert_allclose((left - right).sample(key).y, difference)
    finally:
        top.close()
        first.module.close()
        if not shared:
            second.module.close()


@pytest.mark.parametrize('plotter', ['probe', 'bench'])
def test_grid_sample_plot_preserves_successful_samples(sky130_tb, fake_sim, events, plotter):
    import matplotlib.pyplot as plt
    import numpy as np
    tb = sky130_tb.set_temperatures([125, -40])
    tb.set_monte_carlo(3, seed=17)
    fake_sim.fail_on(('tt', 1))
    probe = tb.probe('vout', analysis='tran')
    expression = (probe + probe) / 2
    results = tb.run()
    before = list(events)
    retained = dict(tb.sweep_corners)
    from pathlib import Path
    files = {key: Path(path).read_bytes() for key, path in retained.items()}
    reports = dict(results.reports)
    native = getattr(tb.netlist, 'simResultsAll', None)
    keys = [((('tt', t, None)), i) for t in [125.0, -40.0] for i in [0, 2]]
    assert list(retained) == keys
    try:
        ax = expression.plot() if plotter == 'probe' else tb.plot(expression, show=False).axes[0]
        assert [line.get_label() for line in ax.lines] == [str(key) for key in keys]
        for line, value in zip(ax.lines, [10.0, 10.5, 10.0, 10.5]):
            np.testing.assert_allclose(line.get_ydata(), value)
        assert list(events) == before
        assert tb.sweep_corners == retained
        assert results.reports == reports
        assert getattr(tb.netlist, 'simResultsAll', None) is native
        assert {key: Path(path).read_bytes() for key, path in retained.items()} == files
        assert len(results.failed) == 2
    finally:
        plt.close('all')


def test_reference_first_sample_and_overlay(sky130_tb, fake_sim, monkeypatch):
    import numpy as np
    import matplotlib.pyplot as plt
    tb = sky130_tb
    tb.set_monte_carlo(n=2)
    tb.run()
    probe = tb.probe('vout', analysis='tran')
    reference = probe.sample(('tt', 0)) * 0 + 2
    expression = reference + probe
    monkeypatch.setattr(type(tb), '_simulate_run', staticmethod(
        lambda *a, **k: pytest.fail('readback launched a simulation')))
    try:
        for key, expected in [(('tt', 0), 12), (('tt', 1), 12.25)]:
            np.testing.assert_allclose(expression.sample(key).y, expected)
        ax = expression.plot()
        assert len(ax.lines) == 2
        for line, key, expected in zip(ax.lines, [('tt', 0), ('tt', 1)], [12, 12.25]):
            np.testing.assert_allclose(line.get_ydata(), expected)
            np.testing.assert_allclose(line.get_ydata(), expression.sample(key).y)
            assert line.get_label() == f'tt/mc_{key[1]:04d}'
    finally:
        plt.close('all')


@pytest.mark.parametrize('index', [0, 2])
@pytest.mark.parametrize('variation', ['mismatch', 'process', 'both'])
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_prepare_selected_sample(sky130_tb, tmp_path, monkeypatch, events,
                                 index, variation, mode):
    from pathlib import Path
    from unittest.mock import Mock
    tb = sky130_tb.set_monte_carlo(3, seed=17, variation=variation,
                                   output_mode=mode)
    sample_spec = Mock(wraps=tb._sample_spec)
    monkeypatch.setattr(tb, '_sample_spec', sample_spec)
    evidence = object()
    tb._metric_values = {'old': evidence}
    tb._cache = {'old': evidence}
    tb._cache_enabled = True
    monkeypatch.setattr(tb, '_fingerprint', lambda: pytest.fail('cache accessed'))
    monkeypatch.setattr(tb.netlist, 'simulate', lambda *a, **k: pytest.fail('launched'))
    deck = Path(tb.prepare(corner='tt', sample=index, run_label='external'))
    text = deck.read_text()
    assert sample_spec.call_count == 1
    assert sample_spec.call_args.args[2] == index
    assert deck.parent.name == f'mc_{index:04d}'
    assert any(part.endswith('~run~external') for part in deck.parts)
    assert f'.options seed={derive_sample_seed(17, index)}' in text
    assert tb.netlist._mc_sample.index == index
    assert tb.netlist._mc_sample.seed == derive_sample_seed(17, index)
    assert len(list(tmp_path.rglob('*.cir'))) == 1
    assert len(list(tmp_path.rglob('run-*'))) == (1 if mode == 'new' else 0)
    if variation == 'both':
        wrapper = Path(tb.netlist._mc_sample.wrapper_path)
        assert wrapper.is_file()
        assert any(part.endswith('~run~external') for part in wrapper.parts)
        assert f'.lib "{wrapper}" mc_both' in text
    elif variation == 'mismatch':
        assert 'tt_mm' in text
    else:
        assert tb.pdk_details.statistical.process_section in text
    raw = deck.parent / 'rawfile.raw'
    raw.write_text('previous evidence')
    tb.prepare(corner='tt', sample=index, run_label='external')
    assert raw.read_text() == 'previous evidence'
    assert tb._metric_values == {'old': evidence}
    assert tb._cache == {'old': evidence}
    assert not any(e[0] == 'sim' for e in events)


@pytest.mark.parametrize('index', [-1, 3, True, 1.0, '1'])
def test_prepare_invalid_sample_before_allocation(sky130_tb, tmp_path, index):
    sky130_tb.set_monte_carlo(3, output_mode='new')
    with pytest.raises(MonteCarloError, match='sample'):
        sky130_tb.prepare(corner='tt', sample=index)
    assert not list(tmp_path.rglob('run-*'))
    assert not list(tmp_path.rglob('*.cir'))


def test_prepare_ambiguous_sample_before_allocation(sky130_tb, tmp_path):
    sky130_tb.set_temperatures([0, 85]).set_monte_carlo(3, output_mode='new')
    with pytest.raises(NotSupportedError, match='one'):
        sky130_tb.prepare(corner='tt', sample=1)
    assert not list(tmp_path.rglob('run-*'))
    assert not list(tmp_path.rglob('*.cir'))


def test_prepare_sample_requires_declaration(sky130_tb):
    with pytest.raises(MonteCarloError, match='declar'):
        sky130_tb.prepare(sample=0)


def test_prepare_sample_refuses_overridden_run(sky130_tb, monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    sky130_tb.set_monte_carlo(3)
    # Match the planner's subclass-override admission without a new class.
    monkeypatch.setattr(ModuleTestbench, '_run_declarations', lambda *a: (None, None))
    with pytest.raises(MonteCarloError, match='sample'):
        sky130_tb.prepare(sample=1)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('variation', ['mismatch', 'process', 'both'])
def test_prepare_matches_managed_sample(sky130_tb, fake_sim, simulator, variation):
    from pathlib import Path
    tb = sky130_tb
    tb.module.simulator = simulator
    tb.set_monte_carlo(3, seed=19, variation=variation)
    tb.run()
    expected = fake_sim.deck_text(('tt', 2))
    assert Path(tb.prepare(corner='tt', sample=2)).read_text() == expected


def test_selected_prepare_sample_inherits_and_restores_point(sky130_tb, tmp_path):
    from pathlib import Path
    from silk import Module, Variable
    tb = sky130_tb
    supply = Variable('rail', value=1.0)
    tb.netlist.sig_sources_dict['vd'].dc = supply
    tb.set_variable_values(supply, [1.8])
    tb.set_temperatures([85]).set_monte_carlo(3, seed=17)
    top = Module('sky130', str(tmp_path), 'selected.cir', tb.pdk_details)
    top.add_module(tb.module, 'a', d='out', g='gate', s='0')
    selected = top.get_test('op', instance='a')
    try:
        text = Path(selected.prepare(corner='tt', sample=2)).read_text()
        assert '.options temp=85.0' in text
        assert 'Vvd vout 0 DC 1.8' in text
        assert f'seed={derive_sample_seed(17, 2)}' in text
        assert supply.value == 1.0
        assert tb.netlist.sig_sources_dict['vd'].dc is supply
        assert getattr(selected, '_metric_values', {}) == {}
    finally:
        top.close()


def test_retained_mc_logical_currents_read_each_samples_operands(sky130_tb, fake_sim, monkeypatch):
    import numpy as np
    from silk import sources
    tb = sky130_tb
    tb.add_source(sources.CurrentSource('bias', 'vout', '0', dc=99, pwl=[0, .5, 1, 1]))
    tb.probe('i(bias)')
    tb.add_metric('logical', lambda c: float(c.load_signal('i(bias)').y[-1]))
    tb.add_metric('string', type='average', signal='i(bias)')
    writer = write_complete_rawfile

    def write_currents(path, deck, *, values):
        level = values['v(vout)']
        writer(path, deck, values={**values, 'i(vvd)': -level,
                                  '@ibias__silk_pwl[current]': level,
                                  '@ibias__silk_dc[current]': -.1})

    monkeypatch.setattr(__name__ + '.write_complete_rawfile',
                        write_currents)
    reports = tb.run_monte_carlo(2, seed=13)
    values = []
    for key, report in reports.items():
        reader = report.sim_results
        level = 10 + .25 * key[1]
        single = reader.load_signal('i(vd)')
        physical = reader.load_signal('vvd', 'I')
        np.testing.assert_array_equal(single.x, physical.x)
        np.testing.assert_array_equal(single.y, physical.y)
        np.testing.assert_allclose(single.y, -level)
        logical = reader.load_signal('i(bias)', 'I')
        np.testing.assert_allclose(logical.y, level - .1)
        assert report.metrics['logical'] == pytest.approx(level - .1)
        assert report.metrics['string'] == pytest.approx(level - .1)
        values.append(float(logical.y[-1]))
        assert all(vec._data is None for plot in reader.simResultsAll.plots for vec in plot.vectors)
    assert len(set(values)) == 2


def _add_advisory_resistors(tb, kinds):
    for i, kind in enumerate(kinds):
        tb.module.add(getattr(tb.pdk_details, kind)(
            f'r{i}', a='d', b='s', c='s', w=1.0, l=1.0))


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('variation', ['process', 'both', 'mismatch'])
@pytest.mark.parametrize('kinds', [('res_hpo',), ('res_xhpo',),
    ('res_xhpo', 'res_hpo', 'res_hpo'), ()])
def test_process_advisory_matches_catalog_types(
        sky130_tb, fake_sim, simulator, variation, kinds):
    import warnings
    tb = sky130_tb
    tb.module.simulator = simulator
    _add_advisory_resistors(tb, kinds)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = tb.run_monte_carlo(2, seed=19, variation=variation, record_draws=False)
    advisories = [w for w in caught if 'process-model advisory' in str(w.message)]
    assert len(advisories) == int(bool(kinds) and variation != 'mismatch')
    if advisories:
        assert advisories[0].category is UserWarning
        assert str(advisories[0].message) == (
            f"Monte Carlo process-model advisory for PDK 'sky130', test {tb.report_key!r}, "
            f"device types {', '.join(sorted(set(kinds)))}: "
            + tb.pdk_details.statistical.process_warnings[0][1])
    assert list(result.reports) == [('tt', 0), ('tt', 1)]
    assert result.seeds == {('tt', i): derive_sample_seed(19, i) for i in range(2)}
    assert [r.metrics['gain'] for r in result.reports.values()] == [10., 10.25]


@pytest.mark.parametrize('route', ['direct', 'declared', 'serial', 'parallel', 'prepare'])
def test_process_advisory_precedes_preparation_once_per_plan(
        sky130_tb, fake_sim, events, monkeypatch, route):
    import warnings
    tb = sky130_tb
    _add_advisory_resistors(tb, ['res_hpo', 'res_hpo', 'res_xhpo'])
    tb.set_monte_carlo(3, seed=19, variation='process')
    if route != 'prepare':
        tb.set_temperatures([25, 85])
    original = warnings.showwarning
    def show(message, *args, **kwargs):
        events.append(('warning', str(message)))
    monkeypatch.setattr(warnings, 'showwarning', show)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        for _ in range(2):
            events.clear()
            if route == 'direct':
                tb.run_monte_carlo(3, seed=19, variation='process')
            elif route == 'declared':
                tb.run()
            elif route in ('serial', 'parallel'):
                results = tb.module.run_all_tests(n_workers=2 if route == 'parallel' else None)
                assert all(isinstance(r, MonteCarloResults) for r in results.values())
            else:
                tb.prepare(sample=1)
            labels = [e[0] for e in events]
            assert labels.count('warning') == 1
            assert max(i for i, e in enumerate(labels) if e == 'check') < labels.index('warning')
            assert labels.index('warning') < labels.index('prepare')
    monkeypatch.setattr(warnings, 'showwarning', original)


def test_process_advisory_filters_and_preflight(sky130_tb, fake_sim, events):
    import warnings
    from pathlib import Path
    tb = sky130_tb
    _add_advisory_resistors(tb, ['res_hpo'])
    before = set(Path(tb.module.dir_folder).rglob('*'))
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Monte Carlo process-model advisory')
        with pytest.raises(UserWarning, match='res_hpo'):
            tb.run_monte_carlo(2, variation='process', output_mode='new')
    assert set(Path(tb.module.dir_folder).rglob('*')) == before
    assert not any(e[0] in ('prepare', 'sim') for e in events)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(MonteCarloError):
            tb.set_corners(['ss']).run_monte_carlo(2, variation='process')
    assert not caught
    tb.set_corners(['tt'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Monte Carlo process-model advisory')
        result = tb.run_monte_carlo(2, variation='process')
    assert len(result.reports) == 2


@pytest.mark.parametrize('n_workers', [None, 2])
def test_process_advisory_recursive_and_selected_placements(
        sky130_tb, fake_sim, tmp_path, n_workers):
    import warnings
    from silk import Module
    tb = sky130_tb
    _add_advisory_resistors(tb, ['res_hpo', 'res_xhpo'])
    tb.set_monte_carlo(2, variation='process')
    top = Module('sky130', str(tmp_path), 'top.cir', tb.pdk_details)
    for name in ('a', 'b'):
        top.add_module(tb.module, name, d=f'd_{name}', g=f'g_{name}', s='0')
    try:
        entries = top.list_test_entries()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            results = top.run_all_tests(n_workers=n_workers)
        messages = [str(w.message) for w in caught if 'process-model advisory' in str(w.message)]
        assert len(messages) == 2
        for entry in entries:
            assert sum(f'test {entry.report_key!r},' in m for m in messages) == 1
        assert all(isinstance(r, MonteCarloResults) for r in results.values())
        selected = top.get_test('op', instance='a')
        with pytest.warns(UserWarning, match='process-model advisory') as caught:
            selected.prepare(sample=1)
        assert len(caught) == 1
        assert f'test {entries[0].report_key!r},' in str(caught[0].message)
    finally:
        top.close()


@pytest.mark.parametrize('record_draws', [False, True])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_process_advisory_preserves_prepared_deck(sky130_tb, record_draws, simulator):
    import warnings
    from dataclasses import replace
    from pathlib import Path
    tb = sky130_tb
    tb.module.simulator = simulator
    _add_advisory_resistors(tb, ['res_hpo'])
    tb.set_monte_carlo(3, seed=23, variation='both', record_draws=record_draws)
    with pytest.warns(UserWarning, match='process-model advisory') as caught:
        text = Path(tb.prepare(sample=2)).read_text()
    assert len(caught) == 1
    st = tb.pdk_details.statistical
    try:
        tb.pdk_details.statistical = replace(st, process_warnings=())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            baseline = Path(tb.prepare(sample=2)).read_text()
        assert not caught
        assert text == baseline
    finally:
        tb.pdk_details.statistical = st


@pytest.mark.parametrize('identity', ['ideal', 'custom', 'pdk'])
def test_process_advisory_generic_manifest_and_identity(ideal_rc_tb, fake_sim, identity):
    from dataclasses import replace
    from silk.elements import elements
    import warnings
    tb = ideal_rc_tb
    st = tb.pdk_details.statistical
    # A different catalog type and message exercise data-driven matching.
    tb.pdk_details.statistical = replace(st, process_warnings=((('nmos1p8',), 'custom reason'),))
    if identity != 'pdk':
        tb.module.add(elements.resistor('nmos1p8', p='d', n='s', value=1000))
        if identity == 'custom':
            device = tb.module.elem_dict['nmos1p8']['device']
            device.device_name = 'nmos1p8'
            device._is_pdk_device = True
            device._pdk_device = ''
        # Keep a real, unmatched PDK device so preflight still admits the plan.
        _add_advisory_resistors(tb, ['res_hpo'])
    else:
        tb.module.add(tb.pdk_details.nmos1p8('M1', d='d', g='g', s='s', b='s', w=1., l=.5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        tb.run_monte_carlo(2, variation='process')
    messages = [str(w.message) for w in caught if 'process-model advisory' in str(w.message)]
    assert len(messages) == int(identity == 'pdk')
    if messages:
        assert messages[0].endswith('device types nmos1p8: custom reason')


def test_process_advisory_batch_plan(sky130_tb, fake_sim, monkeypatch, events):
    tb = sky130_tb
    _add_advisory_resistors(tb, ['res_hpo'])
    tb.set_monte_carlo(5, seed=19, variation='process', samples_per_process=2)
    with pytest.warns(UserWarning, match='process-model advisory') as caught:
        planned = tb._plan_runs()
    assert len(caught) == 1
    assert [r.key for r in planned] == [('tt', i) for i in range(5)]
    assert [r.sample.seed for r in planned] == [derive_sample_seed(19, i) for i in range(5)]
    from silk.circuit import module_test
    batches = []
    def simulate_chunk(run_dir, specs, chunk_dir):
        batches.append(len(specs))
        return [module_test._simulate_run_outcome(spec) for spec in specs], []
    monkeypatch.setattr(module_test, '_chunk_simulate_task', simulate_chunk)
    with pytest.warns(UserWarning, match='process-model advisory') as caught:
        result = tb.run()
    assert len(caught) == 1
    assert batches == [2, 2, 1]
    assert list(result.reports) == [('tt', i) for i in range(5)]
    assert sum(e[0] == 'sim' for e in events) == 5


def test_process_advisory_skips_behavioral_with_matching_manifest(tmp_path):
    import warnings
    from dataclasses import replace
    from silk import load_pdk
    from silk.simulation.variation import Normal
    from tests._variation_bench import chip
    pdk = load_pdk('sky130')
    pdk.statistical = replace(pdk.statistical, process_warnings=((('nmos1p8',), 'reason'),))
    top, tb = chip(pdk, tmp_path, {
        'offset': Normal(nominal=0., sigma=.01, variation='process', name='offset')})
    block = top.elem_dict['c0']['device']
    block.add(pdk.nmos1p8('m1', d='y', g='a', s='a', b='a', w=1., l=.5))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = tb.run_monte_carlo(2, variation='process')
        assert not any('process-model advisory' in str(w.message) for w in caught)
        assert len(result.reports) == 2
    finally:
        top.close()


@pytest.mark.parametrize('record_draws', [False, True])
def test_process_advisory_preserves_fake_campaign_results(sky130_tb, fake_sim, monkeypatch, record_draws):
    from dataclasses import replace
    import warnings
    tb = sky130_tb
    _add_advisory_resistors(tb, ['res_hpo', 'res_xhpo'])
    from silk.circuit import module_test
    monkeypatch.setattr(module_test, '_read_sample_draws',
                        lambda netlist: {('dut.M1', 'vth0'): .5 + netlist._mc_sample.index * .01})
    with pytest.warns(UserWarning, match='process-model advisory'):
        advised = tb.run_monte_carlo(3, seed=31, variation='both', record_draws=record_draws)
    decks = {key: fake_sim.deck_text(key) for key in advised.reports}
    st = tb.pdk_details.statistical
    try:
        tb.pdk_details.statistical = replace(st, process_warnings=())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            baseline = tb.run_monte_carlo(3, seed=31, variation='both', record_draws=record_draws)
        assert not caught
        assert advised.seeds == baseline.seeds
        assert list(advised.reports) == list(baseline.reports)
        assert advised.draws == baseline.draws
        if record_draws:
            assert advised.draws.skipped_types == ('res_xhpo',)
            assert len(advised.draws.values) == 3
        for key in advised.reports:
            assert decks[key] == fake_sim.deck_text(key)
            assert advised[key].metrics == baseline[key].metrics
            assert advised[key].metric_warnings == baseline[key].metric_warnings
            advised_report = advised[key].to_dict()
            baseline_report = baseline[key].to_dict()
            advised_report.pop('provenance')
            baseline_report.pop('provenance')
            assert advised_report == baseline_report
    finally:
        tb.pdk_details.statistical = st


@pytest.mark.parametrize('entry', ['direct', 'module'])
@pytest.mark.parametrize('output_mode', ['overwrite', 'new'])
def test_declared_run_workers_keep_sample_identity(sky130_tb, fake_sim, monkeypatch, entry, output_mode):
    from silk.circuit import module_test
    monkeypatch.setattr(module_test, '_read_sample_draws',
                        lambda *args, **kwargs: {('M1', 'vth0'): 0.5})
    tb = sky130_tb.set_corners(['tt', 'ss']).set_monte_carlo(
        3, seed=19, record_draws=True, output_mode=output_mode)
    settings = tb.monte_carlo
    serial = tb.run()
    run = tb.run if entry == 'direct' else lambda **kw: tb.module.run_test('op', **kw)
    parallel = run(n_workers=2)
    assert list(parallel.reports) == list(serial.reports)
    assert parallel.seeds == serial.seeds
    assert parallel.settings == serial.settings == settings
    assert parallel.draws.columns == serial.draws.columns
    assert parallel.draws.raw == serial.draws.raw
    assert parallel.draws.skipped_types == serial.draws.skipped_types
    for key in serial.draws.values:
        assert parallel.draws.values[key] == pytest.approx(serial.draws.values[key])
    assert parallel.structure_hash == serial.structure_hash
    assert {key: r.metrics for key, r in parallel.items()} == {
        key: r.metrics for key, r in serial.items()}
    assert tb.monte_carlo is settings
    assert parallel.rerun(('tt', 1)).metrics == serial[('tt', 1)].metrics


def test_declared_run_workers_keep_chunk_path(sky130_tb, fake_sim, monkeypatch):
    from silk.simulation import executor
    tb = sky130_tb.set_monte_carlo(5, seed=19, samples_per_process=2)
    dispatch = executor.run_tasks
    batches = []

    def recording(tasks, **kwargs):
        batches.append([len(task.args[0]) for task in tasks])
        return dispatch(tasks, **kwargs)

    monkeypatch.setattr(executor, 'run_tasks', recording)
    serial = tb.run()
    assert batches == [[2], [2], [1]]
    batches.clear()
    monkeypatch.setattr(executor.os, 'cpu_count', lambda: 2)
    parallel = tb.run(n_workers=2)
    assert batches == [[2, 2], [1]]
    assert list(parallel.reports) == [('tt', i) for i in range(5)]
    assert parallel.seeds == serial.seeds
    assert parallel.settings == serial.settings
    assert [r.metrics for r in parallel.values()] == [r.metrics for r in serial.values()]
