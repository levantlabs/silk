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
