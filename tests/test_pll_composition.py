"""The PLL executes its placed component models and live calibrations."""
import numpy as np
import pytest


def test_pll_executes_replaced_component_model(monkeypatch, tmp_path):
    from silk.examples import pll_example as pll

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    original = pll.build_ldo

    def fixed_ldo(**kwargs):
        mod = original(**kwargs)
        def fixed(ports, t, params):
            ports.VREG.v = 1.2
        mod.set_behavioral_model(fixed, execution='sample', params=mod._behavioral_params)
        return mod

    monkeypatch.setattr(pll, 'build_ldo', fixed_ldo)
    mod = pll.build_pll(v_init=1.4)
    bench = pll.attach_lock_test(mod, t_stop=40e-9, t_step=0.2e-9)
    bench.run()
    np.testing.assert_array_equal(bench.netlist.load_signal('vreg').y, 1.2)


def test_pll_component_calibration_and_repeat_initialization(monkeypatch, tmp_path):
    from silk.examples import pll_example as pll

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    mod = pll.build_pll(v_init=1.4)
    bench = pll.attach_lock_test(mod, t_stop=80e-9, t_step=0.2e-9)
    bench.run()
    before = bench.netlist.load_signal('vreg').y.copy()
    ldo = mod.elem_dict['ldo']['device']
    ldo.calibrate_behavioral('gain', 0.5)
    bench.run()
    changed = bench.netlist.load_signal('vreg').y.copy()
    assert changed[0] == before[0] == 1.4
    assert changed[-1] < before[-1] - 0.01
    bench.run()
    np.testing.assert_array_equal(bench.netlist.load_signal('vreg').y, changed)


@pytest.mark.parametrize('first', ['dc', 'transient'])
@pytest.mark.parametrize('kind', ['ldo', 'ring'])
def test_component_dispatch_resets_and_independent_placements(monkeypatch, tmp_path, first, kind):
    from silk import Module, sources
    from silk.simulation import DC, Transient
    from silk.examples.ldo_regulator_example import build_ldo
    from silk.examples.ring_oscillator_example import build_ring_oscillator
    from silk.examples.pll_example import pdk

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    mod = Module('PDK_skywater130', str(tmp_path), 'dispatch.cir', pdk_details=pdk)
    component = (build_ldo if kind == 'ldo' else build_ring_oscillator)(with_tests=False)
    for index, value in enumerate([1.2, 1.5]):
        supply = f'supply{index}'
        out = f'out{index}'
        if kind == 'ring':
            nets = {'VDD': supply, 'VSS': '0', 'EN': 'enable', 'OUT': out}
        else:
            nets = {'VDD': 'rail', 'VSS': '0', 'VCTRL': supply,
                    'IBIAS_AMP': 'bias', 'VREG': out}
        mod.add_module(component, f'component{index}', nets=nets)
        mod.add_source(sources.VoltageSource(supply, positive=supply, negative='0', dc=value))
    mod.add_source(sources.VoltageSource('rail', positive='rail', negative='0', dc=3.3))
    mod.add_source(sources.VoltageSource('enable', positive='enable', negative='0', dc=1.8))
    mod.add_source(sources.CurrentSource('bias', positive='bias', negative='0', dc=20e-6))
    analyses = {'dc': DC('supply0', 1.1, 1.3, 0.1),
                'transient': Transient(step=0.2e-9, stop=20.05e-9)}
    for name in (first, 'dc' if first == 'transient' else 'transient'):
        mod.add_sim_type(analyses[name])
    mod.simulator = 'behavioral'
    mod.simulate()
    waveforms = [mod.load_signal(f'out{k}', analysis='tran').y.copy() for k in (0, 1)]
    assert not np.array_equal(*waveforms)
    for index, initial in enumerate([1.2, 1.5]):
        dc = mod.load_signal(f'out{index}', analysis='dc').y
        if kind == 'ring':
            np.testing.assert_array_equal(dc, 0)
            assert waveforms[index][0] == 0
            rises = np.count_nonzero(np.diff((waveforms[index] > 0.6).astype(int)) == 1)
            assert rises == [8, 15][index]
        else:
            np.testing.assert_allclose(waveforms[index], initial * component._behavioral_params['gain'])
            expected = np.array([1.1, 1.2, 1.3]) if index == 0 else np.full(3, 1.5)
            np.testing.assert_allclose(dc, expected * component._behavioral_params['gain'])
    mod.simulate()
    for index, old in enumerate(waveforms):
        np.testing.assert_array_equal(mod.load_signal(f'out{index}', analysis='tran').y, old)


def test_ring_calibration_changes_the_assembled_clock(monkeypatch, tmp_path):
    from silk.examples import pll_example as pll

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    mod = pll.build_pll(v_init=1.4)
    bench = pll.attach_lock_test(mod, t_stop=40e-9, t_step=0.2e-9)
    bench.run()
    assert np.any(bench.netlist.load_signal('out').y > 0.9)
    mod.elem_dict['ring']['device'].calibrate_behavioral('vdd_min', 10)
    from silk.metrics.diagnostics import MetricWarning
    with pytest.warns(MetricWarning, match='Too few crossings'):
        bench.run()
    np.testing.assert_array_equal(bench.netlist.load_signal('out').y, 0)


def test_controller_uses_actual_previous_interval_at_shortened_step():
    from types import SimpleNamespace as NS
    from silk.simulation._analysis_context import analysis_scope
    from silk.examples.pll_example import _PLLController, _PFD_CP_P, design_loop_filter

    filt = design_loop_filter(verbose=False)
    params = NS(**{k: filt[k] for k in ('r_ohm', 'c1_f', 'c2_f')},
                n_div=1, vdd_cp=1.8, v_ctrl0=0., pfd_t_reset_s=4e-9,
                cp_i_up_a=_PFD_CP_P['i_up_a'], cp_i_dn_a=_PFD_CP_P['i_dn_a'],
                cp_i_leak_a=_PFD_CP_P['i_leak_a'],
                cp_v_comp_lo=_PFD_CP_P['v_comp_lo'], cp_v_comp_hi=_PFD_CP_P['v_comp_hi'])
    model = _PLLController()
    for _ in range(2):
        with analysis_scope('transient'):
            for t, clock, ref in [(0., 0., 0.), (1e-9, 0., 1.8),
                                  (2e-9, 0., 1.8), (2.1e-9, 1.8, 1.8)]:
                ports = NS(CLK=NS(v=NS(previous_value=clock)), REF=NS(v=ref),
                           VCTRL=NS(), PHERR=NS())
                model(ports, t, params)
        # REF crosses at 0.5 ns; delayed OUT crosses physical [1, 2] ns at 1.5 ns.
        assert ports.PHERR.v == pytest.approx(1000.)
        assert model.count == 0


def test_fresh_pump_calibration_still_reaches_the_loop(monkeypatch, tmp_path):
    from silk.examples import pll_example as pll

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    fresh = dict(pll._PFD_CP_P, i_up_a=pll._PFD_CP_P['i_up_a'] * 1.1)
    traces = []
    for calibration in (None, fresh):
        mod = pll.build_pll(v_init=1.4, pfd_cp=calibration)
        result = pll.run_lock(mod, t_stop=80e-9, t_step=0.2e-9)
        traces.append(result['vctrl'])
    assert traces[0][0] == traces[1][0] == 1.4
    assert np.max(np.abs(traces[1] - traces[0])) > 1e-4


def test_controller_refuses_a_step_as_long_as_reset():
    from types import SimpleNamespace as NS
    from silk.simulation._analysis_context import analysis_scope
    from silk.exceptions import SimulationError
    from silk.examples.pll_example import _PLLController

    params = NS(v_ctrl0=0., pfd_t_reset_s=0.2e-9)
    ports = NS(CLK=NS(v=NS(previous_value=0.)), REF=NS(v=0.),
               VCTRL=NS(), PHERR=NS())
    model = _PLLController()
    with analysis_scope('transient'):
        model(ports, 0., params)
        with pytest.raises(SimulationError, match='smaller.*reset'):
            model(ports, 0.2e-9, params)
