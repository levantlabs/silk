"""Drawn-parameter parsers and sensitivity ranking (TASK-435).

Parsers are tested on saved simulator output: the ngspice showmod log from the
design probes, and a one-sample Xyce ``.res`` regenerated from the saved probe
deck ``xy_block_mismatch_off.cir`` (see ``tests/data/README.md``).
"""
import os
import shlex
import warnings

import numpy as np
import pytest

from silk.circuit.monte_carlo_draws import (DrawTable, Sensitivity,
                                            SensitivityRow, parse_showmod,
                                            parse_xyce_res, rank_sensitivity,
                                            showmod_lines)
from silk.exceptions import SimulationError
from silk.metrics.diagnostics import MetricWarning

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, 'tests', 'data')
RES = os.path.join(ROOT, 'tests/data/xy_block_mismatch_off_one_sample.res')


def _probe_log():
    with open(os.path.join(PROBES, 'ng_showmod_draws.txt')) as fh:
        return fh.read()


# --- showmod -----------------------------------------------------------------

def test_showmod_lines_use_the_verified_form():
    [line] = showmod_lines([('xm1', 'sky130_fd_pr__nfet_01v8', ('vth0', 'toxe', 'voff'))])
    assert line == "showmod 'm.xm1.msky130_fd_pr__nfet_01v8' : vth0 toxe voff"


def test_showmod_lines_preserve_saved_probe_selectors_and_parameters():
    with open(os.path.join(PROBES, 'ng_showmod_draws.sp')) as fh:
        deck = fh.read().splitlines()
    lines = showmod_lines([
        ('xm1', 'sky130_fd_pr__nfet_01v8', ('vth0', 'toxe', 'voff')),
        ('xm2', 'sky130_fd_pr__nfet_01v8', ('vth0', 'toxe', 'voff')),
    ])
    assert [' '.join(shlex.split(line)) for line in lines] == deck[10:12]


def test_showmod_lines_refuse_a_device_with_no_parameters():
    with pytest.raises(ValueError, match='xm1'):
        showmod_lines([('xm1', 'model', ())])


def test_parse_showmod_matches_by_request_order():
    got = parse_showmod(_probe_log(), [('M1', ('vth0',)), ('M2', ('vth0',))])
    assert got[('M1', 'vth0')] == pytest.approx(0.524649)
    assert got[('M2', 'vth0')] == pytest.approx(0.522452)


def test_parse_showmod_ignores_the_truncated_printed_name():
    # The log prints 'xm1:sky130_fd_pr__nfe'; designer names never appear
    # there, and swapping the request order swaps the values.
    got = parse_showmod(_probe_log(), [('st1.MB', ('vth0', 'toxe', 'voff')),
                                       ('st1.MA', ('vth0', 'toxe', 'voff'))])
    assert got[('st1.MB', 'vth0')] == pytest.approx(0.524649)
    assert got[('st1.MB', 'toxe')] == pytest.approx(4.15519e-09)
    assert got[('st1.MA', 'voff')] == pytest.approx(-0.0848793)


def test_parse_showmod_refuses_a_block_count_mismatch():
    with pytest.raises(SimulationError, match='2 showmod block.*3 were requested'):
        parse_showmod(_probe_log(), [('a', ('vth0',)), ('b', ('vth0',)),
                                     ('c', ('vth0',))])


def test_parse_showmod_refuses_a_missing_parameter():
    with pytest.raises(SimulationError, match="'M1' has no 'u0'"):
        parse_showmod(_probe_log(), [('M1', ('u0',)), ('M2', ('vth0',))])


# --- Xyce .res -----------------------------------------------------------------

def test_parse_xyce_res_maps_columns_to_designer_names():
    got = parse_xyce_res(RES, {'XON:XM1': 'st1.MN'})
    assert set(got) == {('st1.MN', 'toxe'), ('st1.MN', 'voff'), ('st1.MN', 'vth0')}
    with open(RES) as fh:
        header, row = fh.readline().split(), fh.readline().split()
    col = header.index('XON:XM1:SKY130_FD_PR__NFET_01V8__MODEL.12:VTH0')
    assert got[('st1.MN', 'vth0')] == float(row[col])


def test_parse_xyce_res_name_map_is_case_insensitive():
    a = parse_xyce_res(RES, {'xoff:xm2': 'p.M2'})
    b = parse_xyce_res(RES, {'XOFF:XM2': 'p.M2'})
    assert a == b and len(a) == 3


def test_parse_xyce_res_refuses_an_unknown_device(tmp_path):
    with pytest.raises(SimulationError, match='XDUT:XM9'):
        parse_xyce_res(RES, {'XDUT:XM9': 'st1.MN'})


def test_parse_xyce_res_refuses_more_than_one_sample(tmp_path):
    with open(RES) as fh:
        header, row = fh.readline(), fh.readline()
    path = tmp_path / 'two.res'
    path.write_text(header + row + row.replace('0 ', '1 ', 1))
    with pytest.raises(SimulationError, match='2 sample row'):
        parse_xyce_res(str(path), {'XON:XM1': 'st1.MN'})


# --- DrawTable -------------------------------------------------------------------

def test_draw_table_refuses_misaligned_rows():
    with pytest.raises(ValueError, match='shape'):
        DrawTable(columns=(('M1', 'a'),), values={('tt', 0): [1.0, 2.0]},
                  raw=True, skipped_types=())


# --- ranking -----------------------------------------------------------------

def _table(x, cols=(('M1', 'a'), ('M1', 'b'), ('M2', 'c'))):
    return DrawTable(columns=cols, values={('tt', i): x[i] for i in range(len(x))},
                     raw=True, skipped_types=())


def _values(y):
    return {('tt', i): y[i] for i in range(len(y))}


def test_least_squares_recovers_known_effects():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((400, 3))
    y = 3 * x[:, 0] - 1 * x[:, 1] + 0.01 * rng.standard_normal(400)
    s = rank_sensitivity('y', _values(y), _table(x))
    assert isinstance(s, Sensitivity) and isinstance(s.rows[0], SensitivityRow)
    assert s.method == 'least_squares' and s.r_squared > 0.99
    assert (s.n, s.p) == (400, 3)
    first, second, third = s.rows
    assert first.name == 'M1:a' and first.effect_per_sigma == pytest.approx(3, rel=0.05)
    assert second.name == 'M1:b'
    assert second.effect_per_sigma == pytest.approx(-1, rel=0.05)
    assert first.share == pytest.approx(0.9, abs=0.03)
    assert second.share == pytest.approx(0.1, abs=0.03)
    assert third.name == 'M2:c' and third.share == pytest.approx(0.0, abs=0.01)
    assert first.correlation > 0.9 and second.correlation < 0


def test_raw_values_and_normalised_draws_rank_alike():
    # ngspice gives actual values, Xyce normalised draws: standardising each
    # column makes the effect per sigma independent of the column's scale.
    rng = np.random.default_rng(4)
    z = rng.standard_normal((200, 3))
    y = 2 * z[:, 0] + z[:, 2]
    raw = z * np.array([0.002, 1e-11, 0.003]) + np.array([0.52, 4.1e-9, -0.087])
    a = rank_sensitivity('y', _values(y), _table(z))
    b = rank_sensitivity('y', _values(y), _table(raw))
    for ra, rb in zip(a.rows, b.rows):
        assert ra.name == rb.name
        assert ra.effect_per_sigma == pytest.approx(rb.effect_per_sigma, rel=1e-6)


def test_by_device_sums_each_device():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((300, 3))
    y = 3 * x[:, 0] - 1 * x[:, 1]
    s = rank_sensitivity('y', _values(y), _table(x), by='device')
    assert [r.name for r in s.rows] == ['M1', 'M2']
    assert s.rows[0].share == pytest.approx(1.0, abs=0.01)
    assert s.rows[0].effect_per_sigma == pytest.approx(np.sqrt(10), rel=0.05)
    assert s.rows[0].correlation == pytest.approx(1.0, abs=1e-6)


def test_by_must_be_parameter_or_device():
    with pytest.raises(ValueError, match="'parameter' or 'device'"):
        rank_sensitivity('y', {('tt', 0): 1.0}, _table(np.zeros((1, 3))), by='net')


def test_more_parameters_than_samples_falls_back_to_correlation():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        s = rank_sensitivity('y', {('tt', 0): 1.0, ('tt', 1): 2.0}, _table(x))
    mw = [w for w in caught if issubclass(w.category, MetricWarning)]
    assert len(mw) == 1 and len(caught) == 1
    assert mw[0].message.details == {'n': 2, 'p': 3}
    assert 'n=2' in str(mw[0].message) and 'p=3' in str(mw[0].message)
    assert s.method == 'correlation' and s.r_squared is None
    assert all(r.share is None and r.effect_per_sigma is None for r in s.rows)


def test_correlation_fallback_sorts_by_absolute_correlation():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((3, 3))
    y = -x[:, 2]
    with pytest.warns(MetricWarning, match='n=3.*p=3'):
        s = rank_sensitivity('y', _values(y), _table(x))
    assert s.rows[0].name == 'M2:c' and s.rows[0].correlation == pytest.approx(-1)


def test_correlated_quantities_get_correlation_only():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((100, 3))
    y = x[:, 0]
    vth_diff = {('tt', i): 2 * x[i, 0] + 0.1 for i in range(100)}
    noise = {('tt', i): rng.standard_normal() for i in range(100)}
    s = rank_sensitivity('y', _values(y), _table(x),
                         correlated={'noise': noise, 'vth1 - vth2': vth_diff})
    assert [r.name for r in s.correlated] == ['vth1 - vth2', 'noise']
    top = s.correlated[0]
    assert top.share is None and top.effect_per_sigma is None
    assert top.correlation == pytest.approx(1.0)
    # The quantities never enter the fit.
    assert s.p == 3 and {r.name for r in s.rows} == {'M1:a', 'M1:b', 'M2:c'}


def test_a_constant_column_stays_out_of_the_fit():
    # A device whose mismatch is off draws the same value every sample.
    rng = np.random.default_rng(7)
    x = rng.standard_normal((50, 3))
    x[:, 2] = 0.52
    y = x[:, 0] + 0.5 * x[:, 1]
    s = rank_sensitivity('y', _values(y), _table(x))
    assert s.method == 'least_squares' and s.p == 2
    last = s.rows[-1]
    assert last.name == 'M2:c' and last.share == 0.0
    assert last.effect_per_sigma is None and np.isnan(last.correlation)


def test_a_nan_metric_is_a_caller_error():
    x = np.random.default_rng(8).standard_normal((10, 3))
    y = np.arange(10.0)
    y[4] = np.nan
    with pytest.raises(ValueError, match='not finite'):
        rank_sensitivity('y', _values(y), _table(x))


def test_a_sample_without_draws_is_a_caller_error():
    x = np.random.default_rng(9).standard_normal((10, 3))
    values = _values(np.arange(10.0))
    values[('ss', 0)] = 1.0
    with pytest.raises(ValueError, match="no recorded draws"):
        rank_sensitivity('y', values, _table(x))


def test_a_constant_metric_reports_no_shares():
    # Without the guard a fit ranks round-off (shares like 57/43 of nothing).
    x = np.random.default_rng(10).standard_normal((20, 3))
    with pytest.warns(MetricWarning, match='same in all 20 samples'):
        s = rank_sensitivity('y', _values(np.ones(20)), _table(x))
    assert np.isnan(s.r_squared)
    assert all(r.effect_per_sigma == 0.0 and np.isnan(r.share) for r in s.rows)


# ---------------------------------------------------------------------------
# TASK-429: record_draws wired into ngspice runs
# ---------------------------------------------------------------------------

from silk.circuit.monte_carlo import (MonteCarloResults, MonteCarloSettings,  # noqa: E402
                                      _SampleSpec, build_draw_table,
                                      derive_sample_seed, draw_devices,
                                      write_draws_csv)
from silk.exceptions import MonteCarloError, NotSupportedError  # noqa: E402
from tests.test_monte_carlo import (_bench, _sample_dirs, events,  # noqa: E402,F401
                                    fake_sim, sky130_tb)
from tests.test_monte_carlo_mismatch_off import build_two_pairs  # noqa: E402

SAMPLE_LOG = os.path.join(os.path.dirname(__file__), 'data',
                          'ng_showmod_sample.log')
RECORDED = ('vth0', 'toxe', 'voff', 'nfactor')


def _sample(record_draws=True):
    return _SampleSpec(index=0, seed=7, variation='mismatch', section='tt_mm',
                       wrapper_path=None, record_draws=record_draws)


def _control_block(tmp_path, sample):
    top = build_two_pairs(tmp_path, off_pair=False)
    top._mc_sample = sample
    top.generate_netlist(corner='tt')
    with open(top._run_files.netlist_path) as f:
        deck = f.read()
    return deck[deck.index('.control'):deck.index('.endc')].splitlines()


def test_a_record_draws_deck_shows_every_pdk_transistor_after_the_analysis(
        tmp_path):
    block = _control_block(tmp_path, _sample())
    shows = [line for line in block if line.startswith('showmod')]
    assert shows == [
        f"showmod 'm.{inst}.{dev}.msky130_fd_pr__nfet_01v8' : "
        'vth0 toxe voff nfactor'
        for inst in ('xxoff', 'xxon') for dev in ('XMm1', 'XMm2')]
    first = block.index(shows[0])
    assert block.index('run') < first
    assert block[first - 1].startswith('write ')
    assert block[-1] == 'quit 0'


def test_a_sample_without_record_draws_has_no_showmod(tmp_path):
    assert not any('showmod' in line
                   for line in _control_block(tmp_path, _sample(False)))


def test_record_draws_with_save_device_currents_raises_on_the_deck(tmp_path):
    from silk.simulation.analysis import Transient
    top = build_two_pairs(tmp_path, off_pair=False)
    top.sim_analysis_dict.clear()
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    top.save_device_currents = True
    top._mc_sample = _sample()
    with pytest.raises(MonteCarloError, match='save_device_currents'):
        top.generate_netlist(corner='tt')


def test_record_draws_with_save_device_currents_refuses_before_any_deck(
        sky130_tb, fake_sim, events):
    sky130_tb.module.save_device_currents = True
    sky130_tb.set_monte_carlo(n=3, record_draws=True)
    with pytest.raises(MonteCarloError, match='save_device_currents'):
        sky130_tb.run()
    assert _sample_dirs(sky130_tb) == []
    assert not any(e[0] in ('prepare', 'sim') for e in events)


def test_record_draws_with_no_recorded_device_type_refuses(tmp_path, events):
    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    st = tb.pdk_details.statistical
    from dataclasses import replace
    tb.pdk_details.statistical = replace(st, record={})
    try:
        tb.set_monte_carlo(n=2, record_draws=True)
        with pytest.raises(MonteCarloError, match='nmos1p8'):
            tb.run()
    finally:
        tb.pdk_details.statistical = st


def test_draw_devices_names_designer_paths_and_skipped_types(tmp_path):
    from dataclasses import replace
    top = build_two_pairs(tmp_path, off_pair=False)
    st = top.pdk_details.statistical
    devices, skipped = draw_devices(top.elem_dict, st)
    assert [d.name for d in devices] == ['xoff.m1', 'xoff.m2',
                                         'xon.m1', 'xon.m2']
    assert devices[0].path == 'xxoff.XMm1'
    assert devices[0].params == RECORDED and skipped == ()
    none, skipped = draw_devices(top.elem_dict, replace(st, record={}))
    assert none == [] and skipped == ('nmos1p8',)


def test_the_saved_log_reads_back_under_designer_names(tmp_path):
    """The real ngspice -o log (blank line after each 'model' header)."""
    from silk.circuit.module_test import _read_sample_draws

    class Ran:          # the netlist a sample ran on: deck and run files
        simulator = 'ngspice'

    tb = _bench(tmp_path, 'sky130', 'nmos1p8')
    tb.module.add(tb.pdk_details.nmos1p8('M2', d='d', g='g', s='s', b='s',
                                         w=1.0, l=0.5))
    tb.netlist.elem_dict['dut']['device'] = tb.module
    ran = Ran()
    ran.elem_dict = tb.netlist.elem_dict
    ran.pdk_details = tb.pdk_details
    ran._run_files = type('F', (), {'logfile_path': SAMPLE_LOG})()
    got = _read_sample_draws(ran)
    assert list(got)[:4] == [('dut.M1', p) for p in RECORDED]
    assert got[('dut.M1', 'vth0')] == pytest.approx(0.522291)
    assert got[('dut.M2', 'voff')] == pytest.approx(-0.088229)
    ran.simulator = 'xyce'
    with pytest.raises(NotSupportedError, match='ngspice'):
        _read_sample_draws(ran)


def _fake_showmod_log(netlist):
    """Write a showmod log for a fake run: vth0 grows with the sample."""
    index = netlist._mc_sample.index
    devices, _ = draw_devices(netlist.elem_dict,
                              netlist.pdk_details.statistical)
    text = ['Circuit: fake', '']
    for k, d in enumerate(devices):
        text += [' BSIM4v5 models (Berkeley Short Channel IGFET Model-4)',
                 f'      model {d.path.lower()}:sky130_fd_p', '']
        values = (0.5 + 0.001 * index + 0.01 * k, 4.1e-9, -0.08, 2.5)
        text += [f'{p:>11} {v:>21}' for p, v in zip(d.params, values)]
        text.append('')
    with open(netlist._run_files.logfile_path, 'w') as f:
        f.write('\n'.join(text) + '\n')


@pytest.fixture
def showmod_sim(fake_sim, monkeypatch):
    from silk.circuit.module_test import ModuleTestbench
    simulate = ModuleTestbench._simulate_run

    def with_log(spec):
        result = simulate(spec)
        if spec.netlist._mc_sample.record_draws:
            _fake_showmod_log(spec.netlist)
        return result
    monkeypatch.setattr(ModuleTestbench, '_simulate_run',
                        staticmethod(with_log))
    return fake_sim


def test_a_run_records_draws_and_writes_draws_csv_per_corner(
        sky130_tb, showmod_sim):
    sky130_tb.set_corners(['tt', 'ss'])
    sky130_tb.set_monte_carlo(n=3, record_draws=True)
    mc = sky130_tb.run()
    assert mc.draws.columns == tuple(('dut.M1', p) for p in RECORDED)
    assert mc.draws.raw is True and mc.draws.skipped_types == ()
    assert sorted(mc.draws.values) == sorted(mc.reports)
    assert mc.draws.values[('ss', 2)][0] == pytest.approx(0.502)
    assert sorted(mc.corner('ss').draws.values) == [('ss', i) for i in range(3)]
    for corner in ('tt', 'ss'):
        path = os.path.join(sky130_tb.module.dir_folder,
                            f'test_{sky130_tb.module.module_name}~op',
                            corner, 'draws.csv')
        with open(path) as f:
            rows = f.read().splitlines()
        assert rows[0] == ('corner,sample,seed,dut.M1:vth0,dut.M1:toxe,'
                           'dut.M1:voff,dut.M1:nfactor')
        assert len(rows) == 4
        assert rows[1].startswith(f'{corner},0,{derive_sample_seed(0, 0)},')


def test_a_run_without_record_draws_has_no_draws(sky130_tb, fake_sim):
    mc = sky130_tb.run_monte_carlo(2)
    assert mc.draws is None
    with pytest.raises(ValueError, match='record_draws=True'):
        mc.sensitivity('gain')


def _synthetic_mc(n=60, **settings):
    from silk.circuit.spec_report import SpecReport
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n, 3))
    y = 3 * x[:, 0] - x[:, 1]
    keys = [('tt', i) for i in range(n)]
    reports = {k: SpecReport('t', 'm', {'y': float(y[i])})
               for i, k in enumerate(keys)}
    draws = build_draw_table(
        {k: {('M1', 'a'): x[i, 0], ('M1', 'b'): x[i, 1], ('M2', 'a'): x[i, 2]}
         for i, k in enumerate(keys)})
    return MonteCarloResults(
        reports, settings=MonteCarloSettings(n=n, record_draws=True, **settings),
        seeds={k: derive_sample_seed(0, k[1]) for k in keys}, draws=draws)


def test_sensitivity_ranks_the_recorded_draws():
    mc = _synthetic_mc()
    s = mc.sensitivity('y')
    assert s.method == 'least_squares' and s.r_squared > 0.999
    assert [r.name for r in s.rows[:2]] == ['M1:a', 'M1:b']
    assert s.rows[0].effect_per_sigma == pytest.approx(3, rel=0.15)
    assert s.rows[0].share == pytest.approx(0.9, abs=0.05)
    by_device = mc.sensitivity('y', by='device')
    assert by_device.rows[0].name == 'M1'
    assert by_device.rows[0].share == pytest.approx(1.0, abs=0.01)


def test_sensitivity_refuses_a_multi_corner_result_and_per_corner_ranks():
    from silk.circuit.spec_report import SpecReport
    rng = np.random.default_rng(5)
    reports, vals, seeds = {}, {}, {}
    for c, shift in (('tt', 0.0), ('ss', 50.0)):
        for i in range(40):
            x = rng.standard_normal(3)
            k = (c, i)
            reports[k] = SpecReport('t', 'm', {'y': float(3 * x[0] - x[1])})
            vals[k] = {('M1', 'a'): shift + x[0], ('M1', 'b'): shift + x[1],
                       ('M2', 'a'): shift + x[2]}
            seeds[k] = derive_sample_seed(0, i)
    mc = MonteCarloResults(
        reports, settings=MonteCarloSettings(n=40, record_draws=True),
        seeds=seeds, draws=build_draw_table(vals))
    with pytest.raises(ValueError, match=r"\['ss', 'tt'\].*mc\.corner\('ss'\)"):
        mc.sensitivity('y')
    for c in ('tt', 'ss'):
        s = mc.corner(c).sensitivity('y')
        assert [r.name for r in s.rows[:2]] == ['M1:a', 'M1:b']


def test_sensitivity_skips_nan_samples_with_a_counted_warning():
    mc = _synthetic_mc()
    mc.reports[('tt', 5)].metrics['y'] = float('nan')
    with pytest.warns(MetricWarning, match='1 gave nan'):
        s = mc.sensitivity('y')
    assert s.n == 59


def test_sensitivity_with_correlate_declared_is_not_supported_yet():
    with pytest.raises(NotSupportedError, match='correlate'):
        _synthetic_mc(correlate=['vth_diff']).sensitivity('y')


def test_write_draws_csv(tmp_path):
    table = build_draw_table({('tt', 1): {('M1', 'vth0'): 0.5},
                              ('tt', 0): {('M1', 'vth0'): 0.25}})
    path = tmp_path / 'draws.csv'
    write_draws_csv(path, table, {('tt', 0): 7, ('tt', 1): 9})
    assert path.read_text().splitlines() == [
        'corner,sample,seed,M1:vth0', 'tt,0,7,0.25', 'tt,1,9,0.5']


# --- integration with TASK-428 (Xyce .res draws) ---------------------------

def _xyce_ran(tmp_path):
    """A netlist that ran a Xyce sample: build_two_pairs and a .res file
    respelled for silk's own instance names (as test_monte_carlo_decks)."""
    top = build_two_pairs(tmp_path, off_pair=False)
    top.simulator = 'xyce'
    # Only the run's file locations matter here (an Op() over current
    # sources has no Xyce deck form); _attach_xyce_draws reads the .res
    # next to the netlist path.
    from silk.simulation.run_files import RunFiles
    files = top._run_files = RunFiles.resolve(
        str(tmp_path / 'top.cir'), str(tmp_path))
    res = os.path.join(files.directory,
                       os.path.basename(files.netlist_path) + '.res')
    saved = os.path.join(os.path.dirname(__file__), 'data',
                         'xy_block_mismatch_off_one_sample.res')
    with open(saved) as f, open(res, 'w') as out:
        out.write(f.read().replace('XOFF:XM', 'XXOFF:XMM')
                  .replace('XON:XM', 'XXON:XMM'))
    return top


def test_each_simulator_reads_its_draws_by_one_path(tmp_path):
    from silk.circuit.module_test import _attach_sample_draws

    class Report:
        pass
    top = _xyce_ran(tmp_path)
    xy = Report()
    _attach_sample_draws(xy, top, _sample())
    assert xy._mc_draws_raw is False
    assert xy._mc_draws[('xoff.m1', 'vth0')] == pytest.approx(2.01835591)


def test_a_run_holding_both_kinds_of_draws_raises(sky130_tb, showmod_sim):
    from silk.circuit.module_test import ModuleTestbench
    original = ModuleTestbench._evaluate_run

    def evaluate(self, result):
        report = original(self, result)
        if result.netlist._mc_sample.index == 1:
            report._mc_draws_raw = False        # as a Xyce sample would
        return report
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ModuleTestbench, '_evaluate_run', evaluate)
        with pytest.raises(MonteCarloError, match='both'):
            sky130_tb.run_monte_carlo(2, record_draws=True)


def test_xyce_draws_make_a_normalised_table(sky130_tb, showmod_sim):
    from silk.circuit.module_test import ModuleTestbench
    original = ModuleTestbench._evaluate_run

    def evaluate(self, result):
        report = original(self, result)
        report._mc_draws_raw = False            # as a Xyce sample would
        return report
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ModuleTestbench, '_evaluate_run', evaluate)
        mc = sky130_tb.run_monte_carlo(2, record_draws=True)
    assert mc.draws.raw is False


def test_xyce_record_draws_is_not_refused_at_planning(sky130_tb):
    """Xyce reads its .res: save_device_currents does not stop it, and a
    deck with a recorded device type passes the planning check."""
    sky130_tb.module.simulator = 'xyce'
    sky130_tb.module.save_device_currents = True
    st = sky130_tb.pdk_details.statistical
    sky130_tb._check_record_draws(st, 'test')
    from dataclasses import replace
    with pytest.raises(MonteCarloError, match='nmos1p8'):
        sky130_tb._check_record_draws(replace(st, record={}), 'test')


def test_failed_corner_sensitivity_reports_no_samples_for_known_metric():
    from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings, SampleFailure
    from silk.circuit.spec_report import SpecReport
    from silk.metrics.diagnostics import MetricWarning
    mc = MonteCarloResults(
        {('tt', 0): SpecReport('t', 'm', {'gain': 2.0})},
        settings=MonteCarloSettings(n=1), seeds={('tt', 0): 1, ('ss', 0): 1},
        failed={('ss', 0): SampleFailure('no data', 1)},
        draws=DrawTable((('M1', 'vth'),), {('tt', 0): np.array([1.0])}, raw=False))
    with pytest.warns(MetricWarning, match='every sample failed') as caught:
        with pytest.raises(ValueError, match='no samples for metric'):
            mc.corner('ss').sensitivity('gain')
    assert len(caught) == 1


def test_grid_draw_csv_keeps_physical_coordinates(tmp_path):
    from silk.circuit.monte_carlo import write_draws_csv
    from silk.circuit.monte_carlo_draws import DrawTable
    import csv
    point = ('ss', 125.0, 1.62)
    key = (point, 0)
    path = tmp_path / 'draws.csv'
    table = DrawTable((('M1','vth0'),), {key: [0.4]}, raw=True)
    write_draws_csv(path, table, {key: 17})
    rows = list(csv.DictReader(path.open()))
    assert rows == [dict(corner='ss', temperature='125.0', supply='1.62', sample='0', seed='17', **{'M1:vth0':'0.4'})]


def test_failed_grid_point_draw_csv_keeps_grid_header(tmp_path):
    import csv
    from silk.circuit.monte_carlo_draws import DrawTable
    path = tmp_path / 'failed_draws.csv'
    write_draws_csv(path, DrawTable((('M1','vth0'),), {}, raw=True),
                    {(('tt',125.0,1.62),0):17})
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['corner','temperature','supply','sample','seed','M1:vth0']
        assert list(reader) == []


def test_variable_coordinates_have_matching_draw_csv_columns(tmp_path):
    import csv
    from silk.circuit.monte_carlo import write_draws_csv
    points = [('tt', 27.0, None, (('trim', value), ('f_clk', 1e9))) for value in (1, 3)]
    keys = [(point, 0) for point in points]
    table = DrawTable((('M1', 'vth'),), {key: [0.1 + i] for i, key in enumerate(keys)}, raw=True)
    path = tmp_path / 'variable-draws.csv'
    write_draws_csv(path, table, {key: 7 for key in keys})
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ['corner', 'temperature', 'supply', 'trim', 'f_clk', 'sample', 'seed', 'M1:vth']
    assert [row['trim'] for row in rows] == ['1', '3']
    assert [row['M1:vth'] for row in rows] == ['0.1', '1.1']
