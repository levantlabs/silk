"""Declared corner lists: ``tb.set_corners([...])`` and the run plan both
``run_all_tests`` walks expand (TASK-426, Monte Carlo plan task 2).

No simulator: ``ModuleTestbench._simulate_run`` is replaced by a fake that
returns a real ``_RunSimResult``, so the real ``_evaluate_run`` and metric
extraction run, and each corner yields a known metric value.
"""
import math

import pytest

from silk.circuit.spec_report import CornerResults
from silk.exceptions import (ConfigurationError, SimulationError,
                             UnknownCornerError)
from silk.metrics.diagnostics import MetricWarning
from tests._env import needs_ngspice, needs_pdk

GAIN = {'tt': 10.0, 'ss': 8.0, 'ff': 12.0, 'sf': 9.0, 'fs': 11.0}


@pytest.fixture
def sky130_tb(tmp_path):
    from silk import load_pdk, sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.simulation import analysis
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', str(tmp_path), 'nfet.cir',
                 pdk_details=pdk, module_name='nfet')
    mod.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                             's': PortDir.GROUND, 'b': PortDir.GROUND})
    mod.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15))
    tb = mod.create_test('op')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    tb.add(sources.vdc('vd', positive='vd', negative='0', value=1.0))
    tb.add(sources.vdc('vg', positive='vg', negative='0', value=1.0))
    tb.add_sim_type(analysis.Op())
    tb.add_metric('gain', lambda ckt: GAIN[ckt._last_sim_label])
    tb.add_spec('gain_ok', 'gain', '>', 8.5)
    return tb


class _FakeSim:
    def __init__(self):
        self.failing = set()
        self.calls = []

    def fail_on(self, *corners):
        self.failing.update(corners)

    @staticmethod
    def mean_of(*corners):
        return sum(GAIN[c] for c in corners) / len(corners)


@pytest.fixture
def fake_sim(monkeypatch):
    from silk.circuit.module_test import ModuleTestbench, _RunSimResult
    from silk.simulation.conditions import (OperatingConditions,
                                            _launched_corner)
    fake = _FakeSim()

    def simulate(spec):
        fake.calls.append(spec.corner_str)
        if spec.corner_str in fake.failing:
            raise SimulationError(f'fake failure at {spec.corner_str}')
        netlist = spec.netlist
        conditions = OperatingConditions(
            'silk', netlist.simulator, **_launched_corner(spec.corner_str, False))
        netlist.operating_conditions = conditions
        netlist.sim_results_loaded = True
        return _RunSimResult(netlist=netlist, corner_str=spec.corner_str,
                             conditions=conditions, invocation=spec.invocation,
                             cache_key=spec.cache_key)

    monkeypatch.setattr(ModuleTestbench, '_simulate_run',
                        staticmethod(simulate))
    return fake


def _key(tb):
    return f'{tb.module.module_name}.{tb.test_name}'


# --- the setting ------------------------------------------------------------

def test_set_corners_validates_each_name(sky130_tb):
    with pytest.raises(UnknownCornerError):
        sky130_tb.set_corners(['tt', 'not_a_corner'])
    assert sky130_tb.corners is None


def test_set_corners_stores_returns_self_and_none_clears(sky130_tb):
    assert sky130_tb.corners is None
    assert sky130_tb.set_corners(['tt', 'ss']) is sky130_tb
    assert sky130_tb.corners == ('tt', 'ss')
    sky130_tb.set_corners(None)
    assert sky130_tb.corners is None


def test_set_corners_refuses_a_bare_string_an_empty_or_duplicate_list(sky130_tb):
    with pytest.raises(TypeError):
        sky130_tb.set_corners('tt')
    with pytest.raises(TypeError):
        sky130_tb.set_corners(['tt', 3])
    with pytest.raises(ConfigurationError):
        sky130_tb.set_corners([])
    assert sky130_tb.corners is None
    sky130_tb.set_corners(['tt', 'ss', 'tt'])
    with pytest.raises(ConfigurationError):
        sky130_tb._plan_runs()


# --- running a declared list ------------------------------------------------

def _flat(cr):
    return {c: (dict(r.metrics), r.passed, r.corner) for c, r in cr.items()}


def test_run_all_tests_runs_every_declared_corner_serial_and_parallel(
        sky130_tb, fake_sim):
    sky130_tb.set_corners(['tt', 'ss', 'ff'])
    by_workers = {}
    for n_workers in (None, 2):
        results = sky130_tb.module.run_all_tests(n_workers=n_workers)
        cr = results[_key(sky130_tb)]
        assert isinstance(cr, CornerResults)
        assert list(cr.reports) == ['tt', 'ss', 'ff']
        assert cr.failed == {}
        by_workers[n_workers] = _flat(cr)
    assert by_workers[None] == by_workers[2]
    assert by_workers[None]['ss'][0] == {'gain': 8.0}
    assert by_workers[None]['ss'][1] is False


def test_run_returns_corner_results_in_declared_order(sky130_tb, fake_sim):
    sky130_tb.set_corners(['ff', 'tt'])
    cr = sky130_tb.run()
    assert list(cr.reports) == ['ff', 'tt']
    assert cr['ff'].metrics['gain'] == 12.0
    assert cr.passed is True
    # Each corner ran in its own folder, as run_corners() does.
    paths = sky130_tb._sweep_table
    assert set(paths) <= {'ff', 'tt'}


def test_explicit_corner_restricts_the_plan(sky130_tb, fake_sim):
    sky130_tb.set_corners(['tt', 'ss', 'ff'])
    cr = sky130_tb.run(corner='ss')
    assert list(cr.reports) == ['ss']
    assert fake_sim.calls == ['ss']
    results = sky130_tb.module.run_all_tests(corner='ff', n_workers=2)
    assert list(results[_key(sky130_tb)].reports) == ['ff']


def test_no_declaration_takes_todays_single_run_path(sky130_tb, fake_sim):
    from silk.circuit.spec_report import SpecReport
    sky130_tb.set_corner('ss')
    report = sky130_tb.run()
    assert isinstance(report, SpecReport)
    assert report.metrics['gain'] == 8.0
    for n_workers in (None, 2):
        value = sky130_tb.module.run_all_tests(n_workers=n_workers)[_key(sky130_tb)]
        assert isinstance(value, SpecReport)


@pytest.mark.parametrize('n_workers', [None, 2])
def test_plain_and_planned_tests_share_one_walk(sky130_tb, fake_sim, n_workers):
    from silk import sources
    from silk.circuit.spec_report import SpecReport
    from silk.simulation import analysis
    mod = sky130_tb.module
    plain = mod.create_test('plain')
    plain.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    plain.add(sources.vdc('vd', positive='vd', negative='0', value=1.0))
    plain.add(sources.vdc('vg', positive='vg', negative='0', value=1.0))
    plain.add_sim_type(analysis.Op())
    plain.add_metric('gain', lambda ckt: GAIN[ckt._last_sim_label])
    sky130_tb.set_corners(['ss', 'ff'])
    results = mod.run_all_tests(n_workers=n_workers)
    assert list(results) == [_key(sky130_tb), _key(plain)]
    assert list(results[_key(sky130_tb)].reports) == ['ss', 'ff']
    assert isinstance(results[_key(plain)], SpecReport)
    assert results[_key(plain)].metrics['gain'] == GAIN['tt']


# --- failures (owner ruling D2) ---------------------------------------------

@pytest.mark.parametrize('n_workers', [None, 2])
def test_a_failed_corner_is_kept_out_and_counted(sky130_tb, fake_sim, n_workers):
    fake_sim.fail_on('ss')
    sky130_tb.set_corners(['tt', 'ss', 'ff'])
    cr = sky130_tb.module.run_all_tests(n_workers=n_workers)[_key(sky130_tb)]
    assert list(cr.reports) == ['tt', 'ff']
    assert list(cr.failed) == ['ss']
    assert 'fake failure at ss' in cr.failed['ss']
    assert cr.passed is False
    assert '1 of 3' in cr.summary()
    with pytest.warns(MetricWarning, match='1 of 3') as record:
        assert cr.stat('gain', 'mean') == pytest.approx(fake_sim.mean_of('tt', 'ff'))
    assert len([w for w in record if issubclass(w.category, MetricWarning)]) == 1


def test_stat_and_worst_skip_unavailable_values_with_one_counted_warning(
        sky130_tb, fake_sim):
    fake_sim.fail_on('ss')
    sky130_tb.set_corners(['tt', 'ss', 'ff'])
    cr = sky130_tb.run()
    # Replace one usable value by nan after the run: a nan metric.
    cr.reports['ff'].metrics['gain'] = float('nan')
    with pytest.warns(MetricWarning, match='1 of 3') as record:
        assert cr.stat('gain', 'max') == 10.0
    warned = [w for w in record if issubclass(w.category, MetricWarning)]
    assert len(warned) == 1
    assert 'nan' in str(warned[0].message) and "'ff'" in str(warned[0].message)
    with pytest.warns(MetricWarning, match='1 of 3'):
        assert tuple(cr.worst('gain')) == (10.0, 'tt')


def test_every_corner_failing_raises(sky130_tb, fake_sim):
    fake_sim.fail_on('tt', 'ss')
    sky130_tb.set_corners(['tt', 'ss'])
    with pytest.raises(SimulationError, match='fake failure at tt') as caught:
        sky130_tb.run()
    import pickle
    for error in (caught.value, pickle.loads(pickle.dumps(caught.value))):
        assert type(error) is SimulationError
        assert list(error.failed) == ['tt', 'ss']
        assert 'fake failure at tt' in error.failed['tt']
        assert 'fake failure at ss' in error.failed['ss']


@pytest.mark.parametrize('n_workers', [None, 2])
def test_every_corner_failing_folds_into_the_run_all_tests_error_entry(
        sky130_tb, fake_sim, n_workers):
    fake_sim.fail_on('tt', 'ss')
    sky130_tb.set_corners(['tt', 'ss'])
    value = sky130_tb.module.run_all_tests(n_workers=n_workers)[_key(sky130_tb)]
    assert set(value) == {'error'}
    assert 'fake failure at tt' in value['error']
    import pickle
    for failure in (value, pickle.loads(pickle.dumps(value))):
        assert failure.exception_is_original
        assert type(failure.exception) is SimulationError
        assert list(failure.exception.failed) == ['tt', 'ss']
        assert 'fake failure at ss' in failure.exception.failed['ss']


def test_a_configuration_error_still_stops_the_run(sky130_tb, fake_sim):
    sky130_tb.set_corners(['tt', 'ss'])
    with pytest.raises(ConfigurationError):
        sky130_tb.run(run_label='../escape')
    assert fake_sim.calls == []


# --- run_corners() is unchanged ----------------------------------------------

def test_run_corners_is_unchanged(sky130_tb, fake_sim):
    from silk.circuit.spec_report import CornerRunError
    sky130_tb.set_corners(['ff'])            # ignored by the explicit sweep
    fake_sim.fail_on('ss')
    with pytest.raises(CornerRunError) as err:
        sky130_tb.run_corners(['tt', 'ss'])
    assert list(err.value.results.reports) == ['tt']
    assert err.value.results.failed == {}
    cr = sky130_tb.run_corners(['tt', 'ff'])
    cr.reports['ff'].metrics['gain'] = float('nan')
    with pytest.warns(MetricWarning, match='is nan'):
        assert math.isnan(cr.stat('gain', 'mean'))


def test_a_subclass_that_overrides_run_keeps_its_own_corner_attribute(
        sky130_tb, fake_sim):
    """The PVTTestbench example pattern: assigns self.corners and loops
    ModuleTestbench.run(self, corner=c) itself."""
    from silk.circuit.module_test import ModuleTestbench
    from silk.circuit.spec_report import SpecReport

    class Looping(ModuleTestbench):
        def run(self, corner=None):
            return CornerResults({c: ModuleTestbench.run(self, corner=c)
                                  for c in self.corners})

    sky130_tb.__class__ = Looping
    sky130_tb.corners = ('tt', 'ss')
    assert sky130_tb.corners == ('tt', 'ss')
    with pytest.raises(UnknownCornerError):
        sky130_tb.corners = ('tt', 'bogus')
    cr = sky130_tb.run()
    assert all(isinstance(r, SpecReport) for r in cr.values())
    assert list(cr.reports) == ['tt', 'ss']


@pytest.mark.parametrize('axis,scalar,plural,values', [
    ('temperature', 'set_temperature', 'set_temperatures', [125.0, -40.0]),
    ('supply', 'set_supply_voltage', 'set_supply_voltages', [1.98, 1.62]),
])
def test_condition_list_precedence_and_explicit_clear(sky130_tb, axis, scalar, plural, values):
    tb = sky130_tb
    owner = tb.module
    kwargs = {'source': 'vd'} if axis == 'supply' else {}
    def coords():
        plan = tb._plan_runs()
        return [getattr(run, axis) for run in plan] if plan is not None else None
    getattr(owner, plural)(values, **kwargs)
    assert coords() == values
    getattr(owner, plural)(values[::-1], **kwargs)
    assert coords() == values[::-1]  # live owner, not a copy
    getattr(tb, scalar)(values[0], **kwargs)
    assert coords() is None
    getattr(tb, plural)(values, **kwargs)
    assert coords() == values
    getattr(tb, scalar)(values[1], **kwargs)
    assert coords() is None
    getattr(tb, plural)(values, **kwargs)
    getattr(tb, plural)(None)
    assert coords() is None  # explicit clear does not resume the owner's list


@pytest.mark.parametrize('values', [[], [True], [float('nan')], [-273.15], [None]])
def test_temperature_list_rejects_atomically(sky130_tb, values):
    sky130_tb.set_temperatures([27])
    with pytest.raises(ConfigurationError):
        sky130_tb.set_temperatures(values)
    assert [r.temperature for r in sky130_tb._plan_runs()] == [27.0]


def test_grid_order_and_readable_distinct_folders(sky130_tb):
    tb = sky130_tb.set_corners(['ss', 'tt'])
    tb.set_temperatures([125, -40]).set_supply_voltages([1.62, 1.6200000000000003], source='vd')
    plan = tb._plan_runs()
    assert [r.key for r in plan] == [(c, t, v) for c in ['ss', 'tt']
                                    for t in [125.0, -40.0] for v in [1.62, 1.6200000000000003]]
    assert all(r.corner == r.key[0] for r in plan)
    assert plan[0].corner_subdir == ('ss', 'temperature-125.0', 'supply-1.62')
    assert len({r.corner_subdir for r in plan}) == 8


def test_native_temp_option_replaces_list_only_when_temp_given(sky130_tb):
    tb = sky130_tb.set_temperatures([27, 85])
    tb.set_simulator_options(reltol=1e-5)
    assert len(tb._plan_runs()) == 2
    tb.set_simulator_options(temp=40)
    assert tb._plan_runs() is None


def test_grid_preparation_applies_scalars_and_restores_authored_settings(sky130_tb):
    tb = sky130_tb.set_temperatures([-40, 125]).set_supply_voltages([1.62, 1.98], source='vd')
    before = (dict(tb.netlist.simulator_options), tb.netlist.sig_sources_dict['vd'].dc)
    plan = tb._plan_runs()
    specs, errors = tb._prepare_plan(plan)
    assert errors == {}
    assert [(s.netlist.simulator_options['temp'], s.netlist.sig_sources_dict['vd'].dc)
            for s in specs.values()] == [(-40.0, 1.62), (-40.0, 1.98), (125.0, 1.62), (125.0, 1.98)]
    assert (dict(tb.netlist.simulator_options), tb.netlist.sig_sources_dict['vd'].dc) == before


def test_temperature_list_refuses_behavioral_and_dc_temp_before_allocating(sky130_tb):
    from silk.simulation import DC
    tb = sky130_tb.set_temperatures([27])
    tb.netlist.simulator = 'behavioral'
    with pytest.raises(ConfigurationError, match='temperature'):
        tb._plan_runs()
    tb.netlist.simulator = 'ngspice'
    tb.add_sim_type(DC(source='temp', start=-40, stop=125, step=5))
    with pytest.raises(ConfigurationError, match='conflicts'):
        tb._plan_runs()


def test_grid_run_keeps_keys_failures_and_authored_values(sky130_tb, fake_sim):
    tb = sky130_tb.set_corners(['ss', 'tt']).set_temperatures([125, -40])
    tb.set_supply_voltages([1.62, 1.98], source='vd')
    fake_sim.fail_on('ss')
    before = (dict(tb.netlist.simulator_options), tb.netlist.sig_sources_dict['vd'].dc)
    results = tb.run()
    keys = [(c,t,v) for c in ['ss','tt'] for t in [125.0,-40.0] for v in [1.62,1.98]]
    assert [row['run.corner'] for row in results.table()] == ['ss']*4 + ['tt']*4
    assert list(results.failed) == keys[:4]
    assert list(results.reports) == keys[4:]
    assert all(report.corner == 'tt' for report in results.reports.values())
    assert (dict(tb.netlist.simulator_options), tb.netlist.sig_sources_dict['vd'].dc) == before


@pytest.mark.parametrize('n_workers', [None, 2])
def test_grid_run_all_parity(sky130_tb, fake_sim, n_workers):
    tb = sky130_tb.set_corners(['ss', 'tt'])
    tb.module.set_temperatures([125, -40])
    tb.module.set_supply_voltages([1.62, 1.98], source='vd')
    cr = tb.module.run_all_tests(n_workers=n_workers)[_key(tb)]
    assert not isinstance(cr, dict), cr
    assert list(cr.reports) == [(c,t,v) for c in ['ss','tt']
                                for t in [125.0,-40.0] for v in [1.62,1.98]]
    assert tb.netlist.simulator_options == {}
    assert tb.netlist.sig_sources_dict['vd'].dc == 1.0


@pytest.mark.parametrize('axis,plural,scalar,first,second', [
    ('temperature','set_temperatures','set_temperature',27.0,85.0),
    ('supply','set_supply_voltages','set_supply_voltage',1.62,1.98),
])
def test_clear_uses_scalar_without_reinheriting_list(sky130_tb, axis, plural, scalar, first, second):
    tb = sky130_tb
    kwargs = {'source':'vd'} if axis == 'supply' else {}
    getattr(tb.module, scalar)(first, **kwargs)
    getattr(tb.module, plural)([second], **kwargs)
    getattr(tb, plural)(None)
    # Force another axis to keep a point plan, exposing the scalar selected after clear.
    if axis == 'supply':
        tb.set_temperatures([27])
    else:
        tb.set_supply_voltages([1.8], source='vd')
    assert getattr(tb._plan_runs()[0], axis) == first
    getattr(tb, scalar)(second, **kwargs)
    getattr(tb, plural)([first], **kwargs)
    getattr(tb, plural)(None)
    assert getattr(tb._plan_runs()[0], axis) == second


def test_supply_grid_refuses_same_dc_source_but_allows_other(sky130_tb):
    from silk.simulation import DC
    tb = sky130_tb.set_supply_voltages([1.6, 1.8], source='vd')
    tb.add_sim_type(DC(source='vg', start=0, stop=1, step=0.1))
    assert len(tb._plan_runs()) == 2
    tb.add_sim_type(DC(source='vd', start=0, stop=1, step=0.1))
    with pytest.raises(ConfigurationError, match='Supply list conflicts'):
        tb._plan_runs()


def test_grid_refuses_scalar_consumers(sky130_tb):
    tb = sky130_tb.set_temperatures([27, 85])
    with pytest.raises(ConfigurationError, match='Optimizer'):
        tb._require_single_run()
    with pytest.raises(ConfigurationError, match='store=True'):
        tb.module.run_test(tb.test_name, store=True)
    with pytest.raises(ConfigurationError, match='requires one deck'):
        tb.prepare()
    tb.module.set_temperatures([27])
    with pytest.raises(ConfigurationError, match='Module.simulate'):
        tb.module.simulate()


@pytest.mark.parametrize('values', [[0.2], [0.2, 0.4]])
def test_variable_lists_name_the_scalar_consumer_refusal(sky130_tb, values):
    from silk import Variable
    from silk.optimize import Optimizer, VariableSpace
    knob = Variable('knob', value=0.5, bounds=(0, 1))
    tb = sky130_tb.set_variable_values(knob, values)
    optimizer = Optimizer(tb, VariableSpace([knob]))
    with pytest.raises(ConfigurationError, match='Variable lists'):
        optimizer.evaluate({'knob': 0.3})
    with pytest.raises(ConfigurationError, match='Variable lists'):
        optimizer.sweep(n=2)
    with pytest.raises(ConfigurationError, match='Variable lists'):
        tb.module.run_test(tb.test_name, store=True)
    assert knob.value == 0.5


def test_supply_only_grid_preserves_native_temperature_basis(sky130_tb):
    tb = sky130_tb
    tb.set_simulator_options(temp='85')
    tb.set_supply_voltages([1.8], source='vd')
    plan = tb._plan_runs()
    specs, errors = tb._prepare_plan(plan)
    assert errors == {}
    assert next(iter(specs.values())).netlist._temperature_basis == 'native-and-observed'


def test_supply_grid_does_not_rewrite_an_undeclared_temperature_axis(sky130_tb):
    tb = sky130_tb.set_supply_voltages([1.8], source='vd')
    tb.netlist.add_custom_line('.temp 85')
    plan = tb._plan_runs()
    assert plan[0].temperature == 85.0
    specs, errors = tb._prepare_plan(plan)
    assert errors == {}
    netlist = next(iter(specs.values())).netlist
    assert 'temp' not in netlist.simulator_options
    from silk.simulation.conditions import prepare_temperature
    request=prepare_temperature(netlist)
    assert request.temperature==85.0 and request.basis=='raw-and-observed'


@pytest.mark.parametrize('plural,scalar,values,single,kwargs', [
    ('set_temperatures','set_temperature',[-40,125],85,{}),
    ('set_supply_voltages','set_supply_voltage',[1.62,1.98],1.8,{'source':'vd'}),
])
def test_test_list_overrides_owner_scalar(sky130_tb, plural, scalar, values, single, kwargs):
    tb=sky130_tb
    getattr(tb.module, scalar)(single, **kwargs)
    getattr(tb, plural)(values, **kwargs)
    axis='temperature' if plural == 'set_temperatures' else 'supply'
    assert [getattr(run,axis) for run in tb._plan_runs()] == values


def test_single_element_axis_still_returns_collection(sky130_tb, fake_sim):
    tb=sky130_tb.set_temperatures([27])
    result=tb.run()
    assert isinstance(result,CornerResults)
    assert list(result.reports)==[('tt',27.0,None)]


def test_inherited_supply_list_resolves_live_source(sky130_tb):
    tb=sky130_tb
    tb.module.set_supply_voltages([1.8],source='vd')
    assert tb._plan_runs()[0].supply_source=='vd'
    tb.module.set_supply_voltages([1.8],source='vg')
    plan=tb._plan_runs()
    assert plan[0].supply_source=='vg'
    specs,errors=tb._prepare_plan(plan)
    assert not errors
    netlist=next(iter(specs.values())).netlist
    assert netlist.sig_sources_dict['vg'].dc==1.8
    assert netlist.sig_sources_dict['vd'].dc==1.0
    assert tb.netlist.sig_sources_dict['vg'].dc==1.0


def test_invalid_grid_preflight_does_not_allocate_campaign_folders(sky130_tb, tmp_path):
    from silk.simulation import DC
    tb=sky130_tb.set_temperatures([27]).set_monte_carlo(2,output_mode='new')
    tb.add_sim_type(DC(source='temp',start=0,stop=100,step=1))
    before=set(tmp_path.rglob('*'))
    with pytest.raises(ConfigurationError,match='conflicts'):
        tb.run()
    assert set(tmp_path.rglob('*'))==before


def test_axis_inheritance_is_scoped_to_selected_placement(sky130_tb, tmp_path):
    from silk import Module
    tb=sky130_tb
    pdk=tb.module.pdk_details
    top=Module('sky130',str(tmp_path),'top.cir',pdk_details=pdk)
    parents=[]
    try:
        for name,temps,volts in [('low',[-40,27],[1.62]),('high',[85,125],[1.98])]:
            parent=Module('sky130',str(tmp_path),name+'.cir',pdk_details=pdk)
            parent.set_temperatures(temps).set_supply_voltages(volts,source='vd')
            parent.define_module_name(name)
            parent.define_module_ports({})
            parent.add_module(tb.module,'cell',d='d',g='g',s='0',b='0')
            top.add_module(parent,name,nets={})
            parents.append(parent)
        low=top.get_test('op',instance='low.cell')
        high=top.get_test('op',instance='high.cell')
        assert [r.key for r in low._plan_runs()]==[('tt',-40.0,1.62),('tt',27.0,1.62)]
        assert [r.key for r in high._plan_runs()]==[('tt',85.0,1.98),('tt',125.0,1.98)]
        assert [r.key for r in low._plan_runs()]==[('tt',-40.0,1.62),('tt',27.0,1.62)]
        assert tb._plan_runs() is None
    finally:
        top.close()
        for parent in parents:
            parent.close()


def test_point_preparation_failure_restores_scalar_inputs(sky130_tb, fake_sim, monkeypatch):
    tb=sky130_tb.set_temperatures([27]).set_supply_voltages([1.62,1.98],source='vd')
    original=tb._prepare_run
    before=(dict(tb.netlist.simulator_options),tb.netlist._temperature_explicit,
            tb.netlist._supply_explicit,tb.netlist.sig_sources_dict['vd'].dc)
    def prepare(*args,**kwargs):
        spec=original(*args,**kwargs)
        if tb.netlist._supply_voltage==1.98:
            raise SimulationError('preparation failed after applying supply')
        return spec
    monkeypatch.setattr(tb,'_prepare_run',prepare)
    result=tb.run()
    assert list(result.failed)==[('tt',27.0,1.98)]
    assert (dict(tb.netlist.simulator_options),tb.netlist._temperature_explicit,
            tb.netlist._supply_explicit,tb.netlist.sig_sources_dict['vd'].dc)==before


def test_one_grid_point_can_prepare_one_external_deck(sky130_tb):
    from pathlib import Path
    tb=sky130_tb.set_temperatures([85]).set_supply_voltages([1.62],source='vd')
    deck=Path(tb.prepare())
    assert deck.is_file()
    text=deck.read_text()
    assert 'temp=85' in text.lower()
    assert 'temperature-85.0/supply-1.62' in str(deck)
    assert tb.netlist.sig_sources_dict['vd'].dc==1.0
    assert 'temp' not in tb.netlist.simulator_options


def test_direct_module_list_refusal_is_a_preflight_failure(sky130_tb):
    from silk.circuit.modules_sim import _track_attempt_stage
    sky130_tb.module.set_temperatures([27])
    with _track_attempt_stage() as stage:
        with pytest.raises(ConfigurationError,match='Module.simulate'):
            sky130_tb.module.simulate()
    assert stage[0]=='preflight'


def test_store_refusal_names_the_condition_list(sky130_tb):
    sky130_tb.set_temperatures([27])
    with pytest.raises(ConfigurationError,match='temperature/supply'):
        sky130_tb.module.run_test(sky130_tb.test_name,store=True)


def test_metric_extractors_see_the_current_planned_scalars(sky130_tb, fake_sim):
    tb=sky130_tb.set_temperatures([-40,125]).set_supply_voltages([1.62,1.98],source='vd')
    tb.add_metric('applied_temperature',lambda c:c.simulator_options.get('temp'))
    tb.add_metric('applied_supply',lambda c:c.sig_sources_dict['vd'].dc)
    result=tb.run()
    for point,report in result.items():
        assert report.metrics['applied_temperature']==point[1]
        assert report.metrics['applied_supply']==point[2]
    assert tb.netlist.simulator_options=={}
    assert tb.netlist.sig_sources_dict['vd'].dc==1.0


def test_a_plan_keeps_its_scalar_snapshot_and_next_invocation_inherits_live(sky130_tb):
    tb=sky130_tb.set_supply_voltages([1.62],source='vd')
    tb.module.set_temperature(27)
    plan=tb._plan_runs()
    tb.module.set_temperature(85)
    specs,errors=tb._prepare_plan(plan)
    assert not errors
    assert next(iter(specs.values())).netlist.simulator_options['temp']==27.0
    assert tb._plan_runs()[0].temperature==85.0


def test_corner_only_table_retains_requested_scalar_conditions(sky130_tb, fake_sim, tmp_path):
    import csv
    from silk.circuit._results_json import results_data
    tb=sky130_tb.set_temperature(85).set_supply_voltage(1.8,source='vd').set_corners(['ss','tt'])
    assert [(run.key, run.corner_subdir) for run in tb._plan_runs()] == [('ss', 'ss'), ('tt', 'tt')]
    result=tb.run()
    assert list(result.reports)==['ss','tt']
    path = tmp_path / 'legacy.csv'
    result.to_csv(path)
    with path.open() as stream:
        assert [name for name in csv.DictReader(stream).fieldnames if name.startswith('run.')] == [
            'run.corner', 'run.temperature', 'run.supply', 'run.supply_source', 'run.status', 'run.error']
    assert 'planned points:' not in result.summary()
    wire = results_data(result)
    assert wire['schema_version'] == 3
    assert all(set(run) == {'key', 'corner', 'temperature', 'supply', 'supply_source'}
               for run in wire['run_inventory'])
    for row in result.table():
        assert row['run.temperature']==85.0
        assert row['run.supply']==1.8
        assert row['run.supply_source']=='vd'


def test_behavioral_supply_list_refuses_its_dc_sweep_before_allocating(tmp_path):
    from silk import load_pdk, sources
    from silk.simulation import DC
    from tests._variation_bench import chip
    _,tb=chip(load_pdk('sky130'),tmp_path,{'offset':0.0},
              analyses=[DC(source='rail',start=1.0,stop=2.0,step=0.1)])
    tb.add_source(sources.VoltageSource('rail',positive='vdd',negative='gnd',dc=1.0))
    tb.set_supply_voltages([1.6,1.8],source='rail')
    with pytest.raises(ConfigurationError,match='Supply list conflicts'):
        tb._plan_runs()
    assert tb.netlist.sig_sources_dict['rail'].dc==1.0


def test_grouped_variable_plan_six_points_and_authored_restoration(sky130_tb):
    from silk import Variable
    tb = sky130_tb
    vdd = Variable('vdd', value=1.8, bounds=(1.5, 2.0))
    freq = Variable('f_clk', value=1e9)
    bits = [Variable(f'b{k}', value=0) for k in range(3)]
    tb.set_corners(['ss', 'ff']).set_temperatures([-40, 125])
    assert tb.set_variable_values(vdd, [1.62, 1.98]) is tb
    tb.set_variable_values(freq, [0.6e9, 1.2e9, 2.4e9])
    for var, values in zip(bits, ([1, 1, 1], [1, 0, 1], [0, 1, 1])):
        tb.set_variable_values(var, values)
    assert tb.set_groups([['corner', 'temperature', vdd], [freq, *bits]]) is tb
    plan = tb._plan_runs()
    assert [r.key for r in plan] == [
        (corner, temp, None, tuple(zip(['vdd', 'f_clk', 'b0', 'b1', 'b2'],
                                     [supply, clock, *code])))
        for corner, temp, supply in [('ss', -40.0, 1.62), ('ff', 125.0, 1.98)]
        for clock, code in [(0.6e9, [1, 1, 0]), (1.2e9, [1, 0, 1]),
                            (2.4e9, [1, 1, 1])]]
    assert [v.value for v in [vdd, freq, *bits]] == [1.8, 1e9, 0, 0, 0]
    assert len({r.corner_subdir for r in plan}) == 6
    assert len(tb._plan_runs(corner='ss')) == 3
    with pytest.raises(ConfigurationError, match='corner'):
        tb._plan_runs(corner='tt')


def test_grouped_repeats_select_complete_rows_and_clear(sky130_tb):
    tb = sky130_tb
    tb.set_corners(['ss', 'ss', 'ff']).set_temperatures([-40, -40, 125])
    tb.set_supply_voltages([1.62, 1.8, 1.98], source='vd')
    tb.set_groups([['corner', 'temperature', 'supply']])
    assert [r.key for r in tb._plan_runs(corner='ss')] == [
        ('ss', -40.0, 1.62), ('ss', -40.0, 1.8)]
    tb.set_groups(None)
    with pytest.raises(ConfigurationError, match='repeat|duplicate'):
        tb._plan_runs()


@pytest.mark.parametrize('groups,match', [
    ([['nonsense']], 'nonsense'), ([['temperature']], 'declared'),
    ([['corner'], ['corner']], 'group'),
])
def test_group_validation_before_emission(sky130_tb, groups, match):
    sky130_tb.set_corners(['ss', 'ff'])
    with pytest.raises((ConfigurationError, ValueError), match=match):
        sky130_tb.set_groups(groups)._plan_runs()


def test_variable_values_replace_clear_bounds_and_snap(sky130_tb):
    from silk import Variable
    knob = Variable('knob', value=0.5, bounds=(0, 1), resolution=0.1)
    tb = sky130_tb
    tb.set_variable_values(knob, [0.2, 0.3])
    assert [dict(r.key[3])['knob'] for r in tb._plan_runs()] == [0.2, 0.3]
    tb.set_variable_values(knob, [0.7])
    assert dict(tb._plan_runs()[0].key[3]) == {'knob': 0.7}
    with pytest.raises(ValueError, match='knob'):
        tb.set_variable_values(knob, [2])
    assert knob.value == 0.5
    tb.set_variable_values(knob, None)
    assert tb._plan_runs() is None


def test_variable_supply_override_expression_conflict(sky130_tb):
    from silk import Variable
    knob = Variable('vdd', value=0.9)
    sky130_tb.netlist.sig_sources_dict['vd'].dc = 2 * knob
    sky130_tb.set_supply_voltage(1.8, source='vd')
    sky130_tb.set_variable_values(knob, [0.8, 0.9])
    with pytest.raises(ConfigurationError, match='supply|Supply'):
        sky130_tb._plan_runs()
    assert knob.value == 0.9


def test_ungrouped_temperature_repeats_are_refused_on_planning(sky130_tb):
    sky130_tb.set_temperatures([27, 27.0])
    with pytest.raises(ConfigurationError, match='repeated'):
        sky130_tb._plan_runs()


@pytest.mark.parametrize('entries', [[True, 1], [1, 1.0]])
def test_categorical_run_identity_uses_python_equality(sky130_tb, entries):
    from silk import Variable
    mode = Variable('mode', value='1', var_type='categorical', categories=[True, 1, 1.0, '1'])
    sky130_tb.set_variable_values(mode, entries)
    with pytest.raises(ConfigurationError, match='repeat|Repeated'):
        sky130_tb._plan_runs()
    sky130_tb.set_variable_values(mode, [1, '1'])
    assert [dict(run.key[3])['mode'] for run in sky130_tb._plan_runs()] == [1, '1']
    assert mode.value == '1'


def test_grouping_order_replacement_and_variable_removal(sky130_tb):
    from silk import Variable
    first, last = Variable('first', value=0), Variable('last', value=0)
    tb = sky130_tb.set_corners(['ss', 'ff']).set_temperatures([0, 27])
    tb.set_variable_values(first, [1, 2]).set_variable_values(last, [3, 4])
    tb.set_groups([['corner', first], [last]])
    assert [(run.corner, run.temperature, dict(run.key[3])['last']) for run in tb._plan_runs()] == [
        (corner, temp, value) for corner in ('ss', 'ff') for temp in (0, 27) for value in (3, 4)]
    tb.set_variable_values(first, [5, 6])
    assert list(dict(tb._plan_runs()[0].key[3])) == ['first', 'last']
    tb.set_groups(None).set_variable_values(first, None).set_variable_values(first, [5, 6])
    assert list(dict(tb._plan_runs()[0].key[3])) == ['last', 'first']


@pytest.mark.parametrize('text', ['/', '%', '~', 'μ', '%2F', '%7E', 'μ' * 130])
def test_variable_folder_names_encode_once_at_output_boundary(sky130_tb, text):
    from silk import Variable
    from urllib.parse import quote
    from pathlib import Path
    import hashlib
    knob = Variable('mode/' + text, value=text, var_type='categorical', categories=[text])
    sky130_tb.set_variable_values(knob, [text])
    run, = sky130_tb._plan_runs()
    raw = run.corner_subdir[-1]
    assert raw == f'variable-mode/{text}-s-{text}'
    directory, _ = sky130_tb._output_paths(corner_subdir=run.corner_subdir)
    encoded = quote(raw, safe='').replace('~', '%7E')
    expected = 'test~' + hashlib.sha256(encoded.encode('ascii')).hexdigest() if len(encoded) > 240 else encoded
    assert Path(directory).name == expected
    assert str(Path(directory).resolve()).startswith(str(Path(sky130_tb.module.dir_folder).resolve()))


def test_variable_stimulus_decks_follow_each_point_and_restore(sky130_tb):
    from silk import Variable, sources
    from pathlib import Path
    tb = sky130_tb
    vdd = Variable('vdd', value=1.8)
    tb.netlist.sig_sources_dict['vd'].dc = vdd
    tb.netlist.sig_sources_dict['vg'].dc = vdd - 0.4
    tb.add_source(sources.VoltageSource('clock', positive='clk', negative='0',
        pulse=sources.Pulse(v1=0, v2=vdd, tr=1e-11, tf=1e-11, pw=8e-10, per=1.6e-9)))
    tb.add_source(sources.VoltageSource('bits', positive='bits', negative='0',
        waveform=sources.BitPattern([1, 0, 1], v1=0, v2=vdd, ui=1e-9, tr=1e-11, tf=1e-11)))
    tb.add_source(sources.VoltageSource('prbs', positive='prbs', negative='0',
        waveform=sources.PRBS(7, 8, v1=0, v2=vdd, ui=1e-9, tr=1e-11, tf=1e-11)))
    binding = tb.netlist.sig_sources_dict['vg'].dc
    decks = []
    for voltage, offset in ((1.62, '1.22'), (1.98, '1.58')):
        tb.set_variable_values(vdd, [voltage])
        deck = Path(tb.prepare()).read_text()
        decks.append(deck)
        assert f'DC {voltage}' in deck and f'DC {offset}' in deck
        assert f'PULSE(0 {voltage}' in deck
        for name in ('Vbits', 'Vprbs'):
            lines = deck.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith(name))
            stop = next((i for i in range(start + 1, len(lines)) if not lines[i].startswith('+')), len(lines))
            assert str(voltage) in ' '.join(lines[start:stop])
        assert vdd.value == 1.8
        assert tb.netlist.sig_sources_dict['vd'].dc is vdd
        assert tb.netlist.sig_sources_dict['vg'].dc is binding
    assert decks[0] != decks[1]


@pytest.mark.parametrize('stage', ['prepare', 'simulate', 'evaluate'])
def test_swept_values_restore_after_each_failure_stage(sky130_tb, fake_sim, monkeypatch, stage):
    from silk import Variable
    from silk.circuit.module_test import ModuleTestbench
    knob = Variable('rail', value=1.8)
    tb = sky130_tb
    tb.netlist.sig_sources_dict['vd'].dc = knob
    expression = knob - 0.4
    tb.netlist.sig_sources_dict['vg'].dc = expression
    tb.set_variable_values(knob, [1.62, 1.98])
    name = {'prepare': '_prepare_run', 'simulate': '_simulate_run', 'evaluate': '_evaluate_run'}[stage]
    def refuse(*args, **kwargs):
        raise SimulationError('injected failure')
    if stage == 'simulate':
        monkeypatch.setattr(ModuleTestbench, name, staticmethod(refuse))
    else:
        monkeypatch.setattr(tb, name, refuse)
    with pytest.raises(SimulationError, match='injected failure'):
        tb.run()
    assert knob.value == 1.8
    assert tb.netlist.sig_sources_dict['vd'].dc is knob
    assert tb.netlist.sig_sources_dict['vg'].dc is expression


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('workers', [1, 2])
def test_spice_variable_points_preserve_worker_values(sky130_tb, workers):
    from silk import Variable
    from silk.metrics import final_value
    tb = sky130_tb
    knob = Variable('rail', value=1.8)
    tb.netlist.sig_sources_dict['vd'].dc = knob
    tb.add_metric('rail_voltage', final_value, sig=tb.probe('vd'))
    tb.set_variable_values(knob, [1.62, 1.98])
    results = tb.module.run_all_tests(n_workers=workers)[_key(tb)]
    assert [report.metrics['rail_voltage'] for report in results.values()] == pytest.approx([1.62, 1.98])
    assert knob.value == 1.8 and tb.netlist.sig_sources_dict['vd'].dc is knob


@pytest.mark.parametrize('bad', [[], [None], [True], ['1'], [complex(1, 2)],
                                  [float('nan')], [float('inf')], [2]])
def test_bad_variable_lists_leave_the_prior_declaration_and_value(sky130_tb, bad):
    from silk import Variable
    knob = Variable('knob', value=0.5, bounds=(0, 1))
    sky130_tb.set_variable_values(knob, [0.2])
    with pytest.raises((ValueError, TypeError), match='knob'):
        sky130_tb.set_variable_values(knob, bad)
    assert dict(sky130_tb._plan_runs()[0].key[3]) == {'knob': 0.2}
    assert knob.value == 0.5


@pytest.mark.parametrize('bad', ['0.2', 0.2, {'value': 0.2}])
def test_variable_list_container_type_error_is_atomic(sky130_tb, bad):
    from silk import Variable
    knob = Variable('knob', value=0.5)
    sky130_tb.set_variable_values(knob, [0.2])
    with pytest.raises(TypeError, match='knob'):
        sky130_tb.set_variable_values(knob, bad)
    assert dict(sky130_tb._plan_runs()[0].key[3]) == {'knob': 0.2}
    assert knob.value == 0.5


def test_variable_list_refuses_nonfinite_effective_value(sky130_tb):
    from silk import Variable
    knob = Variable('knob', value=0, resolution=1e308)
    sky130_tb.set_variable_values(knob, [0])
    with pytest.warns(UserWarning, match='knob'):
        with pytest.raises(ValueError, match='knob'):
            sky130_tb.set_variable_values(knob, [1.7e308])
    assert dict(sky130_tb._plan_runs()[0].key[3]) == {'knob': 0}
    assert knob.value == 0


@pytest.mark.parametrize('case', ['unequal', 'missing', 'duplicate_name', 'reserved', 'whole_repeat', 'snapped_repeat'])
def test_invalid_complete_plans_refuse_before_preparation(sky130_tb, monkeypatch, case):
    from silk import Variable
    tb = sky130_tb
    knob = Variable('status' if case == 'reserved' else 'knob', value=0.5, resolution=0.1)
    tb.set_variable_values(knob, [0.2, 0.4])
    if case == 'unequal':
        tb.set_temperatures([27]).set_groups([['temperature', knob]])
    elif case == 'missing':
        tb.set_groups([[Variable('absent', value=0)]])
    elif case == 'duplicate_name':
        tb.set_variable_values(Variable('knob', value=0), [0.2])
    elif case in ('whole_repeat', 'snapped_repeat'):
        tb.set_variable_values(knob, [0.2, 0.2] if case == 'whole_repeat' else [0.201, 0.202])
        tb.set_groups([[knob]])
    def no_preparation(*args, **kwargs):
        pytest.fail('An invalid plan reached preparation')
    monkeypatch.setattr(tb, '_prepare_run', no_preparation)
    with pytest.raises(ConfigurationError):
        tb.run()
    assert knob.value == 0.5


def test_prepared_worker_pickle_freezes_variable_expressions(sky130_tb):
    import pickle
    from pathlib import Path
    from silk import Variable, sources
    knob = Variable('rail', value=1.8)
    tb = sky130_tb
    tb.netlist.sig_sources_dict['vd'].dc = knob
    expression = knob - 0.4
    tb.netlist.sig_sources_dict['vg'].dc = expression
    tb.add_source(sources.VoltageSource('clock', positive='clk', negative='0',
        pulse=sources.Pulse(v1=0, v2=knob, tr=1e-11, tf=1e-11, pw=8e-10, per=1.6e-9)))
    tb.set_variable_values(knob, [1.62, 1.98])
    plan = tb._plan_runs()
    prepared, errors = tb._prepare_plan(plan)
    assert not errors
    for run, offset in zip(plan, ('1.22', '1.58')):
        worker_spec = pickle.loads(pickle.dumps(prepared[run.key]))
        text = Path(worker_spec.netlist.generate_netlist(corner=run.corner)).read_text()
        voltage = dict(run.key[3])['rail']
        assert f'DC {voltage}' in text and f'DC {offset}' in text
        assert f'PULSE(0 {voltage}' in text
        assert knob.value == 1.8
        assert tb.netlist.sig_sources_dict['vd'].dc is knob
        assert tb.netlist.sig_sources_dict['vg'].dc is expression


@pytest.mark.parametrize('expression', [False, True])
@pytest.mark.parametrize('inherited', [False, True])
def test_behavioral_supply_list_checks_original_variable_binding(tmp_path, monkeypatch, expression, inherited):
    from silk import Variable, load_pdk, sources
    from tests._variation_bench import chip
    from silk.circuit.module_test import ModuleTestbench
    top, tb = chip(load_pdk('sky130'), tmp_path, {'offset': 0.0})
    rail = Variable('rail', value=0.9)
    binding = 2 * rail if expression else rail
    tb.add_source(sources.VoltageSource('rail', positive='vdd', negative='gnd', dc=binding))
    (top if inherited else tb).set_supply_voltages([1.6, 1.8], source='rail')
    tb.set_variable_values(rail, [0.8, 0.9])
    def unexpected(*args, **kwargs):
        pytest.fail('invalid plan reached preparation')
    monkeypatch.setattr(ModuleTestbench, '_prepare_plan', unexpected)
    with pytest.raises(ConfigurationError, match='Supply override conflicts'):
        tb.run()
    assert rail.value == 0.9
    assert tb.netlist.sig_sources_dict['rail'].dc is binding


@pytest.mark.parametrize('route', ['run', 'walk', 'prepare'])
def test_public_grouped_corner_selection_keeps_complete_rows(sky130_tb, fake_sim, route):
    from pathlib import Path
    from silk import Variable
    from silk.exceptions import NotSupportedError
    tb = sky130_tb
    knob = Variable('trim', value=7)
    tb.set_corners(['ss', 'ss', 'ff']).set_variable_values(knob, [1, 3, 5])
    tb.set_groups([['corner', knob]])
    if route == 'run':
        result = tb.run(corner='ss')
    elif route == 'walk':
        result = tb.module.run_all_tests(corner='ss')[_key(tb)]
    else:
        with pytest.raises(NotSupportedError, match='one deck'):
            tb.prepare(corner='ss')
        path = Path(tb.prepare(corner='ff', ngspice_version='42'))
        assert path.is_file() and 'variable-trim-i-5' in str(path)
        result = None
    if result is not None:
        assert [dict(point[3])['trim'] for point in result.keys()] == [1.0, 3.0]
        assert fake_sim.calls == ['ss', 'ss']
    before = list(fake_sim.calls)
    with pytest.raises(ConfigurationError, match='absent'):
        if route == 'walk':
            tb.module.run_all_tests(corner='tt')
        else:
            getattr(tb, route)(corner='tt')
    assert fake_sim.calls == before and knob.value == 7
