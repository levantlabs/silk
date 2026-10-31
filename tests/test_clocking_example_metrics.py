"""Regression checks for the clocking examples' catalog-metric migration."""
import numpy as np


def test_dead_behavioral_ring_has_no_frequency(monkeypatch, tmp_path):
    from silk.examples.ring_calibrate import _behavioral_freq
    from silk.examples.ring_oscillator_example import build_ring_oscillator

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    ring = build_ring_oscillator(with_tests=False)
    assert not np.isfinite(_behavioral_freq(ring, 0.8))


def test_pll_lock_test_records_behavioral_system_specs(monkeypatch, tmp_path):
    from silk.examples.pll_example import (
        add_lock_specs, attach_lock_test, build_pll, run_lock,
        FERR_METRIC, LOCKTIME_METRIC, PHERR_METRIC, VCTRL_METRIC,
    )
    from silk.simulation.conditions import ConditionState

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    pll = build_pll(v_init=1.4)
    bench = attach_lock_test(pll, t_stop=4e-6, t_step=0.2e-9)
    specs = add_lock_specs(bench)
    result = run_lock(pll, testbench=bench)

    assert 500e6 < result['f_locked'] < 700e6
    assert 1.0 < result['v_ctrl_locked'] < 1.8
    expected = {
        LOCKTIME_METRIC: result['lock_time_us'],
        FERR_METRIC: abs(result['f_err_pct']),
        VCTRL_METRIC: result['v_ctrl_locked'],
        PHERR_METRIC: abs(result['pherr_ps']),
    }
    for metric, spec in specs.items():
        measurement = spec.latest(source='behavioral')
        assert measurement is not None
        assert measurement.corner is ConditionState.NONE
        assert measurement.value == expected[metric]
        assert spec.latest(source='spice') is None


def test_ring_low_supply_metrics_use_the_bench_supply(monkeypatch, tmp_path):
    from silk.examples.ring_oscillator_example import (
        build_ring_oscillator, attach_osc_test, F_METRIC, DUTY_METRIC, JIT_METRIC,
    )

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    ring = build_ring_oscillator(with_tests=False)
    bench = attach_osc_test(ring, vdd=0.85)
    ring._behavioral_params['vdd_min'] = 0.0
    ring._behavioral_params['f0_mhz'] = 1000.0
    ring._behavioral_params['kvco_mhz_per_v'] = 0.0
    bench.netlist.simulator = 'behavioral'
    report = bench.run()
    assert 990 < report.metrics[F_METRIC] < 1010
    assert 49 < report.metrics[DUTY_METRIC] < 51
    assert np.isfinite(report.metrics[JIT_METRIC])


def test_ring_tuning_propagates_simulation_failure(monkeypatch, tmp_path):
    import pytest
    from silk.circuit.module_test import ModuleTestbench
    from silk.exceptions import SimulationError
    from silk.examples.ring_oscillator_example import attach_tuning_sweep

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))

    def fail_run(self):
        raise SimulationError('simulator unavailable')

    monkeypatch.setattr(ModuleTestbench, 'run', fail_run)
    with pytest.raises(SimulationError, match='simulator unavailable'):
        attach_tuning_sweep(vdds=(1.8,))


def test_dead_ring_retains_catalog_diagnostics(monkeypatch, tmp_path):
    from silk.examples.ring_oscillator_example import (
        build_ring_oscillator, attach_osc_test, JIT_METRIC,
    )

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    ring = build_ring_oscillator(with_tests=False)
    bench = attach_osc_test(ring, vdd=0.8)
    bench.netlist.simulator = 'behavioral'
    report = bench.run()
    assert np.isnan(report.metrics[JIT_METRIC])
    assert not report.passed
    assert not report.metric_errors
    assert report.metric_warnings['period_jitter_s']


def test_ring_tuning_records_unmeasurable_frequency(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from silk.circuit.module_test import ModuleTestbench
    from silk.examples.ring_oscillator_example import attach_tuning_sweep, F_METRIC

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    monkeypatch.setattr(ModuleTestbench, 'run',
                        lambda self: SimpleNamespace(metrics={F_METRIC: np.nan}))
    assert attach_tuning_sweep(vdds=(0.8,)) == [(0.8, None)]


def test_vco_rejection_preserves_unknown_jitter_in_specs(monkeypatch, tmp_path):
    from silk import load_pdk
    from silk.circuit.modules import Module
    from silk.examples import supply_regulated_vco_example as vco
    from silk.specs import Measurement

    for j_reg, j_raw, expected in (
        (float('nan'), 100.0, 'FAIL'),
        (1.0, float('nan'), 'FAIL'),
        (1.0, 100.0, 'PASS'),
    ):
        monkeypatch.setattr(vco, '_regulated_jitter_ps', lambda *a, **kw: j_reg)
        monkeypatch.setattr(vco, '_unregulated_jitter_ps', lambda **kw: j_raw)
        mod = Module('sky130', str(tmp_path), 'rejection.cir',
                     pdk_details=load_pdk('sky130'))
        reg, raw, rejection = vco.measure_rejection(mod)
        np.testing.assert_equal((reg, raw), (j_reg, j_raw))
        if expected == 'FAIL':
            assert np.isnan(rejection)
        else:
            assert rejection == j_raw / j_reg
        spec = mod.declare_spec(
            f'jitter rejection > {vco.REJ_MIN:.0f}x @ {vco._REJ_FM/1e6:.0f} MHz',
            vco.REJ_METRIC, '>', vco.REJ_MIN)
        spec.record_measurement(Measurement(
            source='spice', value=rejection, corner='tt', timestamp=0.0))
        assert mod.reconcile_specs().outcomes[0].status == expected


def test_ring_tuning_fit_uses_only_the_default_linear_range():
    from silk.examples.ring_oscillator_example import fit_tuning

    rows = [(0.8, 10_000.0), (1.0, 200.0), (1.2, 400.0),
            (1.4, 600.0), (1.6, 800.0), (1.8, 1000.0), (2.0, 10_000.0)]
    np.testing.assert_allclose(fit_tuning(rows), (-800.0, 1000.0, 1.0))
    rows[1] = (1.0, None)
    np.testing.assert_allclose(fit_tuning(rows), (-800.0, 1000.0, 1.2))


def test_ring_calibration_keeps_below_range_rows(monkeypatch, tmp_path):
    from silk.examples.ring_calibrate import calibrate
    from silk.examples.ring_oscillator_example import build_ring_oscillator

    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    ring = build_ring_oscillator(with_tests=False)
    before = dict(ring._behavioral_params)
    rows = [(0.8, 200.0)] + [
        (v, before['f0_mhz'] + before['kvco_mhz_per_v'] * v)
        for v in (1.0, 1.2, 1.4, 1.6, 1.8)
    ]
    measured_rows, params = calibrate(ring, rows=rows)
    assert measured_rows is rows
    np.testing.assert_allclose(params, tuple(before.values()))
    np.testing.assert_allclose(tuple(ring._behavioral_params.values()),
                               tuple(before.values()))
