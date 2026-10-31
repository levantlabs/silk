from pathlib import Path
from silk import PortDir, sources
from silk.metrics import final_value
from silk.signals.rawfile import read_rawfile
from silk.exceptions import VariationError
from silk.simulation.variation import bench_key
import numpy as np
import pytest
from silk.circuit.modules import Module
from silk.exceptions import MonteCarloError
from silk.simulation import DC, Transient
from silk.simulation.variation import Normal, Uniform, draw_u, mismatch_key, process_key
from tests._variation_bench import add_offset, adder, chip

SIGMA = 1e-3


def values(mc, n=3):
    return {key: [mc[key].metrics[f'y{k}'] for k in range(n)] for key in mc.keys()}


def expected(seed, index, segments, nominal=0.0, sigma=SIGMA):
    return Normal(nominal=nominal, sigma=sigma).resolve().at(
        draw_u(mismatch_key(seed, index, segments, 'offset')))


def test_each_instance_draws_its_own_value_by_the_scheme(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    mc = tb.set_monte_carlo(n=4, seed=3).run()
    got = values(mc)
    assert len(got) == 4
    for (corner, index), ys in got.items():
        assert len(set(ys)) == 3
        assert ys == pytest.approx([expected(3, index, (f'c{k}',)) for k in range(3)],
                                   rel=1e-9, abs=1e-15)


def test_same_seed_reproduces_and_another_seed_differs(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    a = values(tb.set_monte_carlo(n=3, seed=1).run())
    b = values(tb.set_monte_carlo(n=3, seed=1).run())
    c = values(tb.set_monte_carlo(n=3, seed=2).run())
    assert a == b and a != c


def test_unrelated_block_changes_no_draw(pdk, tmp_path):
    _, tb3 = chip(pdk, tmp_path / 'three', {'offset': Normal(nominal=0.0, sigma=SIGMA)}, n=3)
    _, tb4 = chip(pdk, tmp_path / 'four', {'offset': Normal(nominal=0.0, sigma=SIGMA)}, n=4)
    three = values(tb3.set_monte_carlo(n=3).run())
    four = values(tb4.set_monte_carlo(n=3).run(), n=4)
    assert {k: v[:3] for k, v in four.items()} == three


def test_process_is_shared_and_filtered(pdk, tmp_path):
    params = {'offset': Normal(nominal=0.0, sigma=SIGMA),
              'shift': Normal(nominal=0.5, sigma=SIGMA, variation='process', name='shift')}
    _, tb = chip(pdk, tmp_path, params)
    proc = values(tb.set_monte_carlo(n=3, variation='process').run())
    for (corner, index), ys in proc.items():
        assert ys[0] == ys[1] == ys[2]                     # mismatch nominal, one shared shift
        assert ys[0] == pytest.approx(Normal(nominal=0.5, sigma=SIGMA).resolve().at(
            draw_u(process_key(0, index, 'shift'))))
    mism = values(tb.set_monte_carlo(n=3, variation='mismatch').run())
    for (corner, index), ys in mism.items():
        assert ys == pytest.approx([0.5 + expected(0, index, (f'c{k}',)) for k in range(3)])
    both = values(tb.set_monte_carlo(n=3, variation='both').run())
    assert all(len(set(ys)) == 3 for ys in both.values())


def test_set_mismatch_off_keeps_nominal_and_process_still_draws(pdk, tmp_path):
    params = {'offset': Normal(nominal=0.0, sigma=SIGMA),
              'shift': Normal(nominal=0.0, sigma=SIGMA, variation='process', name='shift')}
    _, tb = chip(pdk, tmp_path, params, mismatch=False)
    with pytest.raises(MonteCarloError, match='no active'):
        tb.set_monte_carlo(n=2, variation='mismatch').run()
    both = values(tb.set_monte_carlo(n=2, variation='both').run())
    assert all(ys[0] == ys[1] == ys[2] for ys in both.values())


@pytest.mark.parametrize('params', [
    {'offset': 0.0},
    {'offset': Normal(nominal=0.0, sigma=0.0)},
    {'offset': Uniform(low=0.2, high=0.2)},
])
def test_no_active_declaration_refused(pdk, tmp_path, params):
    _, tb = chip(pdk, tmp_path, params)
    with pytest.raises(MonteCarloError, match='no active'):
        tb.set_monte_carlo(n=2).run()



def setup_bench(top, name, outputs):
    tb = top.create_test(name)
    tb.add_module_under_test({p: p for p in top.pin_list})
    tb.add_source(sources.VoltageSource('input', positive='a', negative='gnd', dc=0.0))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    for output in outputs:
        tb.add_metric(output, final_value, sig=tb.probe(output))
    return tb


def waveforms(report):
    return [{v.name: v.data for v in plot.vectors}
            for plot in read_rawfile(report.rawfile_path).plots]


def test_dc_sweep_holds_one_draw(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)},
                 analyses=[DC('input', 0.0, 0.2, 0.05)])
    mc = tb.set_monte_carlo(n=2).run()
    for key in mc.keys():
        assert mc[key].metrics['y0'] == pytest.approx(0.2 + expected(0, key[1], ('c0',)))
        plot, = waveforms(mc[key])
        np.testing.assert_allclose(plot['v(y0)'] - plot['v(a)'], expected(0, key[1], ('c0',)))


def test_dc_and_transient_share_one_draw(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb._metrics.clear()
    for k in range(3):
        tb.add_metric(f'y{k}', final_value, sig=tb.probe(f'y{k}', analysis='tran'))
    tb.add_sim_type(DC('input', 0.0, 0.0, 0.1))
    mc = tb.set_monte_carlo(n=2).run()
    for key in mc.keys():
        plots = waveforms(mc[key])
        assert len(plots) == 2
        for plot in plots:
            np.testing.assert_allclose(plot['v(y0)'], expected(0, key[1], ('c0',)))


def test_workers_do_not_change_draws(pdk, tmp_path):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb.set_monte_carlo(n=3)
    serial = values(next(iter(top.run_all_tests(n_workers=1).values())))
    parallel = values(next(iter(top.run_all_tests(n_workers=2).values())))
    assert serial == parallel


def test_corners_repeat_the_same_chips(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb.set_corners(['tt', 'ss'])
    with pytest.warns(UserWarning, match='identical'):
        got = values(tb.set_monte_carlo(n=2).run())
    assert got[('tt', 0)] == got[('ss', 0)] and got[('tt', 1)] == got[('ss', 1)]


def test_tests_of_one_circuit_share_a_chip(pdk, tmp_path):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb2 = setup_bench(top, 'again', ['y0', 'y1', 'y2'])
    tb.set_monte_carlo(n=2)
    tb2.set_monte_carlo(n=2)
    results = top.run_all_tests()
    assert values(results['chip.offsets']) == values(results['chip.again'])


def test_root_and_child_tests_name_the_same_instance(pdk, tmp_path):
    stage, tb = chip(pdk, tmp_path, {'offset': Normal(0.0, SIGMA)}, n=2, module_name='stage')
    tb.set_monte_carlo(n=2)
    root = Module('PDK_skywater130', str(tmp_path / 'root'), 'root.cir',
                  pdk_details=pdk, module_name='root')
    root.define_module_ports({'a': PortDir.IN, 's1': PortDir.OUT, 's2': PortDir.OUT})
    for k in (1, 2):
        root.add_net(f'unused{k}')
        root.add_module(stage, f'stage{k}', nets={'a': 'a', 'y0': f's{k}', 'y1': f'unused{k}'})
    root.simulator = 'behavioral'
    root_tb = setup_bench(root, 'root_offsets', ['s1', 's2'])
    root_tb.set_monte_carlo(n=2)
    results = root.run_all_tests(recursive=True)
    root_mc = results['root.root_offsets']
    for k in (1, 2):
        child = results[f'$/stage{k}:stage.offsets']
        for key in root_mc.keys():
            want = expected(0, key[1], (f'stage{k}', 'c0'))
            assert root_mc[key].metrics[f's{k}'] == pytest.approx(want)
            assert child[key].metrics['y0'] == pytest.approx(want)
    for key in root_mc.keys():
        assert root_mc[key].metrics['s1'] != root_mc[key].metrics['s2']


def test_x_prefixed_instance_names_are_kept(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)}, names=['xa', 'a'])
    got = values(tb.set_monte_carlo(n=1).run(), n=2)
    (ys,) = got.values()
    assert ys == pytest.approx([expected(0, 0, ('xa',)), expected(0, 0, ('a',))])
    assert ys[0] != ys[1]


def test_bench_helpers_use_the_bench_namespace(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': 0.0})
    helper = adder(pdk, tmp_path, {'offset': Uniform(-0.1, 0.1)}, name='helper')
    tb.add_module(helper, 'stim', nets={'a': 'a', 'y': 'helper_y'})
    tb.add_metric('helper', final_value, sig=tb.probe('helper_y'))
    mc = tb.set_monte_carlo(n=2, seed=17).run()
    for key in mc.keys():
        want = Uniform(-0.1, 0.1).resolve().at(draw_u(
            bench_key(17, key[1], tb.owner_path, tb.test_name, ('stim',), 'offset')))
        assert mc[key].metrics['helper'] == pytest.approx(want)
        assert mc[key].metrics['y0'] == 0.0


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('monte_carlo', [False, True])
def test_process_conflicts_across_tests_before_any_model(pdk, tmp_path, workers, monte_carlo):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0.0, SIGMA)})
    tb2 = setup_bench(top, 'again', ['y0'])
    for bench, sigma in [(tb, 0.1), (tb2, 0.2)]:
        helper = adder(pdk, tmp_path, {'offset': Normal(0, sigma, variation='process', name='shared')},
                       name=f'helper_{bench.test_name}')
        bench.add_module(helper, 'helper', nets={'a': 'a', 'y': 'helper_y'})
        if monte_carlo:
            bench.set_monte_carlo(2, variation='mismatch')  # conflict even when filtered out
    with pytest.raises(VariationError, match='offsets.*again'):
        top.run_all_tests(n_workers=workers)
    assert not list(tmp_path.rglob('*.raw'))


def test_full_seed_and_index_not_simulator_seed(pdk, tmp_path):
    from dataclasses import replace
    from silk.circuit.monte_carlo import MonteCarloSettings
    from silk.simulation._behavioral_variation import resolve_run, VariationScope
    from silk.simulation.behavioral import BehavioralEngine
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0.0, SIGMA)})
    settings = MonteCarloSettings(n=1, seed=2**80 + 19)
    tb._check_monte_carlo(settings, ['tt'])
    sample = tb._sample_spec(settings, 'tt', 55170, None)
    blocks, _ = BehavioralEngine()._flatten(tb.netlist)
    scope = VariationScope(dut_instance='dut')
    resolution = resolve_run(blocks, scope=scope, sample=sample)
    assert resolution.draws[('dut.c0', 'offset')] == expected(settings.seed, 55170, ('c0',))
    assert resolve_run(blocks, scope=scope, sample=replace(sample, seed=7)).draws == resolution.draws
    assert resolve_run(blocks, scope=scope, sample=replace(sample, index=50699)).draws != resolution.draws


def test_samples_keep_their_folders_and_record_no_measurement(pdk, tmp_path, monkeypatch):
    from silk.specs import Spec
    top, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb.add_spec('y0 small', 'y0', '<=', 1.0, key='y0_small')
    monkeypatch.setattr(Spec, 'record_measurement', lambda *a, **kw: pytest.fail('sample recorded measurement'))
    mc = tb.set_monte_carlo(n=2).run()
    paths = []
    for key in mc.keys():
        path = Path(mc[key].rawfile_path)
        assert path.is_file()
        assert path.parent.name == f'mc_{key[1]:04d}'
        assert path.parent.parent.name == key[0]
        assert mc[key].operating_conditions.backend == 'behavioral'
        paths.append(path)
    assert len(set(paths)) == 2


def test_campaign_freezes_variable_fields_until_next_campaign(pdk, tmp_path, monkeypatch):
    from silk.circuit.variable import Variable
    sigma = Variable('sigma', value=SIGMA, bounds=(0.0, 1.0))
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0.0, sigma)})
    from silk.simulation.behavioral_backend import BehavioralBackend
    simulate = BehavioralBackend._launch_module

    def change_between_samples(backend, module, **kwargs):
        result = simulate(backend, module, **kwargs)
        sigma.value = 2 * SIGMA
        return result

    monkeypatch.setattr(BehavioralBackend, '_launch_module', change_between_samples)
    first = values(tb.run_monte_carlo(2))
    second = values(tb.run_monte_carlo(2))
    for key, ys in first.items():
        assert ys == pytest.approx([expected(0, key[1], (f'c{k}',)) for k in range(3)])
        assert second[key] == pytest.approx([2 * y for y in ys])


def test_ancestor_mismatch_switch_and_no_pdk_statistics(pdk, tmp_path):
    params = {'offset': Normal(0.0, SIGMA),
              'shift': Normal(0.5, SIGMA, variation='process', name='shift')}
    top, _ = chip(pdk, tmp_path, params)
    top.set_mismatch(False)
    tb = setup_bench(top, 'disabled', ['y0', 'y1', 'y2'])
    pdk.statistical = None
    with pytest.raises(MonteCarloError, match='no active'):
        tb.run_monte_carlo(2, variation='mismatch')
    tb.set_corners(['ss'])
    mc = tb.run_monte_carlo(2, variation='process')
    for key, ys in values(mc).items():
        want = params['shift'].resolve().at(draw_u(process_key(0, key[1], 'shift')))
        assert ys == pytest.approx([want] * 3)
    assert not list(tmp_path.rglob('mc_both.lib.spice'))


def test_duplicate_spec_names_refused(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    tb.add_spec('same', 'y0', '<=', 1.0)
    tb.add_spec('same', 'y1', '<=', 1.0, key='other')
    with pytest.raises(MonteCarloError, match='more than one spec'):
        tb.run_monte_carlo(2)


@pytest.mark.parametrize('entry', ['run', 'serial', 'parallel'])
def test_preflight_honors_inherited_supply_without_side_effects(pdk, tmp_path, entry):
    import warnings
    from silk.elements import elements
    from silk.simulation._behavioral_variation import resolve_test
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    tb.add_source(sources.VoltageSource('power', positive='vdd', negative='gnd', dc=1.2))
    tb.netlist.add(elements.capacitor('load', p='y0', n='vdd', value=1e-12))
    top.set_supply_voltage(1.8, source='power')
    before = (tb.netlist._supply_voltage, tb.netlist._supply_source,
              tb.netlist.sig_sources_dict['power'].dc,
              getattr(tb.netlist, '_behavioral_element_notes', None))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert resolve_test(tb).records
    assert not caught
    assert before == (tb.netlist._supply_voltage, tb.netlist._supply_source,
                      tb.netlist.sig_sources_dict['power'].dc,
                      getattr(tb.netlist, '_behavioral_element_notes', None))
    tb.set_monte_carlo(2)
    with pytest.warns(UserWarning, match='load.*ignored') as caught:
        if entry == 'run':
            mc = tb.run()
        else:
            mc = top.run_all_tests(n_workers=1 if entry == 'serial' else 2)['chip.offsets']
    assert len(caught) == 2  # one notice per sample, none during preflight
    assert len(list(mc.keys())) == 2


def test_empty_bench_reports_no_active_declaration(pdk, tmp_path):
    top = Module('PDK_skywater130', str(tmp_path), 'empty.cir', pdk_details=pdk)
    top.simulator = 'behavioral'
    with pytest.raises(MonteCarloError, match='no active'):
        top.create_test('empty').run_monte_carlo(2)


def test_draw_table_columns_and_values(pdk, tmp_path):
    import csv
    params = {'offset': Normal(0, SIGMA),
              'shift': Normal(0, SIGMA, variation='process', name='shift')}
    _, tb = chip(pdk, tmp_path, params)
    tb.set_corners(['tt', 'ss'])
    with pytest.warns(UserWarning, match='identical'):
        mc = tb.set_monte_carlo(n=3, variation='both', record_draws=True).run()
    cols = tuple((f'dut.c{k}', 'offset') for k in range(3)) + (('process', 'shift'),)
    assert mc.draws.columns == cols
    assert mc.draws.raw is True
    assert mc.draws.skipped_types == ()
    for key in mc.keys():
        row = dict(zip(cols, mc.draws.values[key]))
        for k in range(3):
            assert mc[key].metrics[f'y{k}'] == pytest.approx(
                row[(f'dut.c{k}', 'offset')] + row[('process', 'shift')])
    paths = list(tmp_path.rglob('draws.csv'))
    assert {p.parent.name for p in paths} == {'tt', 'ss'}
    for path in paths:
        rows = list(csv.reader(path.open()))
        assert rows[0] == ['corner', 'sample', 'seed', *[f'{a}:{b}' for a, b in cols]]
        assert len(rows) == 4


def test_draw_recording_without_pdk_preserves_samples_and_writes_csv(tmp_path):
    import csv
    from silk.circuit.monte_carlo import MonteCarloResults

    dut = Module(None, str(tmp_path), 'offset.cir', pdk_details=None,
                 module_name='offset')
    dut.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT})
    dut.set_behavioral_model(
        add_offset,
        params={'offset': Normal(0, SIGMA)}, execution='vector')
    dut.simulator = 'behavioral'
    tb = setup_bench(dut, 'offsets', ['y'])
    assert dut.pdk_details is None
    assert tb.pdk_details is None
    assert tb.netlist.pdk_details is None

    plain = tb.set_monte_carlo(n=3, seed=5, record_draws=False).run()
    assert isinstance(plain, MonteCarloResults)
    assert len(plain.reports) == 3
    assert plain.failed == {}
    assert plain.draws is None
    recorded = tb.set_monte_carlo(n=3, seed=5, record_draws=True).run()
    assert isinstance(recorded, MonteCarloResults)
    assert len(recorded.reports) == 3
    assert recorded.failed == {}
    assert recorded.seeds == plain.seeds
    assert recorded.draws.columns == (('dut', 'offset'),)
    assert recorded.draws.raw is True
    assert recorded.draws.skipped_types == ()
    assert set(recorded.draws.values) == set(recorded.reports) == set(plain.reports)
    for key, row in recorded.draws.values.items():
        assert row == pytest.approx([recorded[key].metrics['y']])
        assert recorded[key].metrics == plain[key].metrics
    assert len({r.metrics['y'] for r in recorded.reports.values()}) == 3

    paths = list(tmp_path.rglob('draws.csv'))
    assert len(paths) == 1
    assert paths[0].parent.name == 'tt'
    with paths[0].open() as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ['corner', 'sample', 'seed', 'dut:offset']
    assert len(rows) == 4
    for row, key in zip(rows[1:], sorted(recorded.reports)):
        assert row[:3] == [key[0], str(key[1]), str(recorded.seeds[key])]
        assert float(row[3]) == pytest.approx(recorded.draws.values[key][0])


def test_sensitivity_rows_and_device_grouping(pdk, tmp_path):
    params = {'offset': Normal(0, SIGMA),
              'shift': Normal(0, SIGMA, variation='process', name='shift'),
              'trim': Normal(0, SIGMA, variation='process', name='trim')}
    _, tb = chip(pdk, tmp_path, params, n=2)
    mc = tb.set_monte_carlo(n=40, variation='both', record_draws=True).run()
    assert {r.name for r in mc.sensitivity('y0').rows} == {
        'dut.c0:offset', 'dut.c1:offset', 'process:shift', 'process:trim'}
    assert {r.name for r in mc.sensitivity('y0', by='device').rows} == {
        'dut.c0', 'dut.c1', 'process'}


def test_column_collision_refused_before_running(pdk, tmp_path):
    _, tb = chip(pdk, tmp_path, {
        'offset': Normal(0, SIGMA, variation='process', name='shift')})
    helper = adder(pdk, tmp_path, {'offset': 0.0, 'shift': Normal(0, SIGMA)}, name='helper')
    tb.netlist.add_module(helper, 'process', nets={'a': 'a', 'y': 'helper_out'})
    with pytest.raises(MonteCarloError, match='column.*process.*shift'):
        tb.set_monte_carlo(n=2, variation='both', record_draws=True).run()
    assert not list(tmp_path.rglob('mc_0000'))


@pytest.mark.parametrize('record', [True, False])
def test_rerun_reproduces(pdk, tmp_path, record):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    mc = tb.set_monte_carlo(n=3, seed=5, record_draws=record).run()
    key = list(mc.keys())[1]
    assert mc.rerun(key).metrics == pytest.approx(mc[key].metrics)


@pytest.mark.parametrize('change,label', [
    ('plain_param', 'model'), ('set_mismatch', 'mismatch'),
    ('source_dc', 'source'), ('transient_stop', 'analysis'),
    ('sigma_number', 'declaration'), ('sigma_variable', 'declaration')])
def test_rerun_refuses_after_each_signature_change(pdk, tmp_path, change, label):
    from silk.circuit.variable import Variable
    s = Variable('s', value=SIGMA, bounds=(0.0, 1.0))
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0, s), 'gain': 1.0})
    mc = tb.set_monte_carlo(n=2).run()
    block = top.elem_dict['c0']['device']
    if change == 'plain_param':
        block._behavioral_params['gain'] = 2.0
    elif change == 'set_mismatch':
        block.set_mismatch(False)
    elif change == 'source_dc':
        tb.netlist.sig_sources_dict['input'].dc = 0.1
    elif change == 'transient_stop':
        tb.add_sim_type(Transient(step=1e-9, stop=20e-9))
    elif change == 'sigma_number':
        block._behavioral_params['offset'] = Normal(0, 2 * SIGMA)
    else:
        s.value = 2 * SIGMA
    with pytest.raises(MonteCarloError, match=f'changed.*{label}'):
        mc.rerun(list(mc.keys())[0])


def test_signature_captured_before_first_sample(pdk, tmp_path, monkeypatch):
    from silk.circuit.variable import Variable
    sigma = Variable('sigma', value=SIGMA, bounds=(0, 1))
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0, sigma)})
    prepare = tb._prepare_plan

    def change_during_preparation(*args, **kwargs):
        sigma.value = 2 * SIGMA
        return prepare(*args, **kwargs)

    monkeypatch.setattr(tb, '_prepare_plan', change_during_preparation)
    mc = tb.set_monte_carlo(n=2).run()
    for key in mc.keys():
        assert mc[key].metrics['y0'] == pytest.approx(expected(0, key[1], ('c0',)))
    with pytest.raises(MonteCarloError, match='changed declaration'):
        mc.rerun(next(iter(mc.keys())))


def test_rerun_with_inherited_supply(pdk, tmp_path):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    tb.add_source(sources.VoltageSource('power', positive='vdd', negative='gnd', dc=1.2))
    top.set_supply_voltage(1.8, source='power')
    mc = tb.set_monte_carlo(n=2).run()
    key = next(iter(mc.keys()))
    assert mc.rerun(key).metrics == pytest.approx(mc[key].metrics)
    top.set_supply_voltage(1.9, source='power')
    with pytest.raises(MonteCarloError, match='changed'):
        mc.rerun(key)


def test_rerun_refuses_changed_controlled_source_gain(pdk, tmp_path):
    from silk.sources import VCVS
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    tb.add_source(VCVS('amp', 'amplified', 'gnd', 'y0', 'gnd', 2))
    tb.add_metric('amplified', final_value, sig=tb.probe('amplified'))
    mc = tb.set_monte_carlo(n=2).run()
    key = next(iter(mc.keys()))
    assert mc.rerun(key).metrics == pytest.approx(mc[key].metrics)
    tb.netlist.sig_sources_dict['amp'].gain = 3
    with pytest.raises(MonteCarloError, match='changed.*amp'):
        mc.rerun(key)


@pytest.mark.parametrize('variation, mismatch, columns', [
    ('mismatch', True, (('dut.c0', 'offset'),)),
    ('process', True, (('process', 'shift'),)),
    ('both', False, (('process', 'shift'),)),
])
def test_draw_columns_only_include_active_declarations(pdk, tmp_path, variation, mismatch, columns):
    _, tb = chip(pdk, tmp_path, {
        'offset': Normal(0, SIGMA), 'constant': Uniform(1, 1),
        'shift': Normal(0, SIGMA, variation='process', name='shift')},
        n=1, mismatch=mismatch)
    mc = tb.set_monte_carlo(n=2, variation=variation, record_draws=True).run()
    assert mc.draws.columns == columns


def test_selected_rerun_refuses_changed_inherited_supply(pdk, tmp_path):
    top, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    tb.add_source(sources.VoltageSource('power', positive='vdd', negative='gnd', dc=1.2))
    top.set_supply_voltage(1.8, source='power')
    tb.set_monte_carlo(n=2)
    mc = top.run_all_tests()['chip.offsets']
    key = next(iter(mc.keys()))
    assert mc.rerun(key).metrics == pytest.approx(mc[key].metrics)
    top.set_supply_voltage(1.9, source='power')
    with pytest.raises(MonteCarloError, match='changed'):
        mc.rerun(key)


def test_replay_identity_distinguishes_source_and_model_with_same_name(pdk, tmp_path):
    from silk.sources import VCVS
    _, tb = chip(pdk, tmp_path, {'offset': Normal(0, SIGMA)})
    helper = adder(pdk, tmp_path, {'offset': 0.0}, name='helper')
    tb.netlist.add_module(helper, 'amp', nets={'a': 'a', 'y': 'helper_out'})
    tb.add_source(VCVS('amp', 'amplified', 'gnd', 'y0', 'gnd', 2))
    mc = tb.set_monte_carlo(n=2).run()
    helper._behavioral_params['offset'] = 0.1
    with pytest.raises(MonteCarloError, match='changed model.*amp'):
        mc.rerun(next(iter(mc.keys())))


@pytest.mark.parametrize('workers', [1, 2])
def test_new_campaign_retains_seeded_samples_and_draws(pdk, tmp_path, workers):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb.set_corners(['tt', 'ss'])
    baseline = tb.run_monte_carlo(2, seed=9, record_draws=True, n_workers=workers)
    first = tb.run_monte_carlo(2, seed=9, record_draws=True,
                               n_workers=workers, output_mode='new')
    roots = {corner: Path(first[(corner, 0)].run.rawfile_path).parent.parent
             for corner in ('tt', 'ss')}
    saved = {p: p.read_bytes() for root in roots.values()
             for p in root.rglob('*') if p.is_file()
             and not any(part.startswith('.silk') for part in p.relative_to(root).parts)}
    second = tb.run_monte_carlo(2, seed=9, record_draws=True,
                                n_workers=workers, output_mode='new')
    assert values(first) == values(second) == values(baseline)
    assert baseline.draws.columns == first.draws.columns == second.draws.columns
    assert baseline.draws.raw == first.draws.raw == second.draws.raw
    for key in baseline.draws.values:
        np.testing.assert_array_equal(baseline.draws.values[key], first.draws.values[key])
        np.testing.assert_array_equal(baseline.draws.values[key], second.draws.values[key])
    for corner, root in roots.items():
        assert root.name.startswith('run-') and root.parent.name == corner
        assert (root / 'draws.csv').is_file()
        assert Path(first[(corner, 1)].run.rawfile_path).parent.parent == root
        assert Path(second[(corner, 0)].run.rawfile_path).parent.parent != root
    assert all(p.read_bytes() == data for p, data in saved.items())


@pytest.mark.parametrize('workers', [None, 1, 2])
def test_labelled_declared_campaign_retains_files(pdk, tmp_path, workers):
    _, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=0.0, sigma=SIGMA)})
    tb.set_corners(['tt', 'ss'])
    tb.set_monte_carlo(2, seed=9, record_draws=True, output_mode='new')
    if workers is None:
        first = tb.run(run_label='baseline')
    else:
        first, = tb.module.run_all_tests(n_workers=workers, run_label='baseline').values()
    roots = {corner: Path(first[(corner, 0)].run.rawfile_path).parent.parent
             for corner in ('tt', 'ss')}
    saved = {p: p.read_bytes() for root in roots.values() for p in root.rglob('*')
             if p.is_file() and not any(part.startswith('.silk') for part in p.relative_to(root).parts)}
    if workers is None:
        second = tb.run(run_label='baseline')
    else:
        second, = tb.module.run_all_tests(n_workers=workers, run_label='baseline').values()
    for corner, root in roots.items():
        assert root.parent.parent.name.endswith('~run~baseline')
        assert root.parent.name == corner and root.name.startswith('run-')
        labelled = root.parent.parent
        unlabelled = labelled.with_name(labelled.name.removesuffix('~run~baseline'))
        assert not list(unlabelled.rglob('run-*'))
        assert (root / 'draws.csv').is_file()
        assert Path(first[(corner, 1)].run.rawfile_path).parent.parent == root
        assert Path(second[(corner, 0)].run.rawfile_path).parent.parent != root
    assert values(first) == values(second)
    assert all(p.read_bytes() == data for p, data in saved.items())


@pytest.mark.parametrize('workers', [1, 2])
def test_swept_variable_resolves_behavioral_declarations_per_point(pdk, tmp_path, workers):
    from silk import Variable
    nominal = Variable('nominal', value=0.25)
    top, tb = chip(pdk, tmp_path, {'offset': Normal(nominal=nominal, sigma=SIGMA)})
    tb.set_variable_values(nominal, [0.1, 0.7]).set_monte_carlo(n=2, seed=3)
    mc = next(iter(top.run_all_tests(n_workers=workers).values()))
    assert len(mc.reports) == 4
    for (point, index), report in mc.items():
        chosen = dict(point[3])['nominal']
        assert report.metrics['y0'] == pytest.approx(
            expected(3, index, ('c0',), nominal=chosen))
        assert report.parameters['silk_run_variables'] == {'nominal': chosen}
    assert nominal.value == 0.25
    assert [row['run.nominal'] for row in mc.table()] == [0.1, 0.1, 0.7, 0.7]
    nominal.value = 0.5
    replay = mc.rerun(next(iter(mc.keys())))
    assert replay.metrics['y0'] == pytest.approx(next(iter(mc.values())).metrics['y0'])
    assert replay.parameters['silk_run_variables'] == {'nominal': 0.1}
    assert nominal.value == 0.5
    tb.set_variable_values(nominal, [0.1, 0.9])
    with pytest.raises(MonteCarloError, match='changed'):
        mc.rerun(next(iter(mc.keys())))
    assert nominal.value == 0.5


@pytest.mark.parametrize('route', ['run', 'run_monte_carlo', 'serial', 'parallel'])
def test_process_conflicts_check_every_swept_point_before_walk(pdk, tmp_path, route):
    from silk import Variable
    sigma = Variable('sigma', value=SIGMA)
    top, tb = chip(pdk, tmp_path, {
        'shift': Normal(nominal=0, sigma=sigma, variation='process', name='shared')})
    tb.set_variable_values(sigma, [SIGMA, 2 * SIGMA]).set_monte_carlo(n=2, variation='process')
    with pytest.raises(VariationError, match='shared.*conflict|conflict.*shared'):
        if route == 'run':
            tb.run()
        elif route == 'run_monte_carlo':
            tb.run_monte_carlo(2, variation='process')
        else:
            top.run_all_tests(n_workers=1 if route == 'serial' else 2)
    assert sigma.value == SIGMA


@pytest.mark.parametrize('workers', [1, 2])
def test_variable_sources_and_metrics_execute_at_the_same_point(pdk, tmp_path, workers):
    from silk import Variable
    knob = Variable('rail', value=1.8)
    top, tb = chip(pdk, tmp_path, {'offset': 0})
    source = tb.netlist.sig_sources_dict['input']
    source.dc = knob
    source.negative = '0'
    tb.add_metric('live_knob', lambda _ckt: knob.value)
    tb.set_variable_values(knob, [1.62, 1.98])
    result = next(iter(top.run_all_tests(n_workers=workers).values()))
    assert [report.metrics['y0'] for report in result.values()] == [1.62, 1.98]
    assert [report.metrics['live_knob'] for report in result.values()] == [1.62, 1.98]
    assert [row['observed.supply'] for row in result.table()] == [1.62, 1.98]
    assert list(tb._sweep_table) == list(result.keys()), [(key, report.rawfile_path) for key, report in result.items()]
    for point in result.keys():
        signal = tb.probe('y0').point(point)
        assert signal.y[-1] == dict(point[3])['rail']
    assert knob.value == 1.8 and source.dc is knob


def test_two_tests_sharing_a_variable_keep_their_own_points(pdk, tmp_path):
    from silk import Variable
    knob = Variable('offset', value=0.25)
    top, first = chip(pdk, tmp_path, {'offset': 0})
    second = setup_bench(top, 'again', ['y0', 'y1', 'y2'])
    first.netlist.sig_sources_dict['input'].dc = knob
    second.netlist.sig_sources_dict['input'].dc = knob
    first.set_variable_values(knob, [0.1, 0.2])
    second.set_variable_values(knob, [0.7, 0.8])
    results = top.run_all_tests(n_workers=2)
    assert [r.metrics['y0'] for r in results['chip.offsets'].values()] == [0.1, 0.2]
    assert [r.metrics['y0'] for r in results['chip.again'].values()] == [0.7, 0.8]
    assert knob.value == 0.25
    for collection in results.values():
        assert [r.parameters['silk_run_variables']['offset'] for r in collection.values()] == [
            dict(point[3])['offset'] for point in collection.keys()]


@pytest.mark.parametrize('workers', [1, 2])
def test_two_placements_sharing_a_swept_variable_keep_distinct_evidence(pdk, tmp_path, workers):
    from silk import Variable
    knob = Variable('level', value=0.25)
    child, tb = chip(pdk, tmp_path / 'child', {'offset': 0})
    binding = tb.netlist.sig_sources_dict['input']
    binding.dc = knob
    tb.set_variable_values(knob, [0.1, 0.7])
    top = Module('PDK_skywater130', str(tmp_path / 'root'), 'root.cir', pdk_details=pdk)
    top.simulator = 'behavioral'
    for name in ('left', 'right'):
        top.add_module(child, name, nets={port: name + '_' + port for port in child.pin_list})
    selected = [top.get_test('offsets', instance=name) for name in ('left', 'right')]
    assert selected[0].netlist is selected[1].netlist is tb.netlist
    results = top.run_all_tests(n_workers=workers)
    assert len(results) == 2
    reports = list(results.values())
    for collection in reports:
        assert [r.metrics['y0'] for r in collection.values()] == [0.1, 0.7]
        assert [r.parameters['silk_run_variables'] for r in collection.values()] == [
            {'level': 0.1}, {'level': 0.7}]
    assert {r.owner_path for c in reports for r in c.values()} == {'$/left', '$/right'}
    assert set(r.rawfile_path for r in reports[0].values()).isdisjoint(
        r.rawfile_path for r in reports[1].values())
    for bench in selected:
        for point in reports[0].keys():
            assert bench.probe('y0').point(point).y[-1] == dict(point[3])['level']
    assert knob.value == 0.25 and binding.dc is knob
    top.close()
    child.close()
