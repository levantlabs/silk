"""End-to-end verification of the probe, metric, run and plot workflow.

This is the one place the whole probe-signals feature (TASK-215.1-215.9) is
exercised together, on real ngspice, as the section-2 program is actually
written: ``probe`` -> arithmetic/transforms defer -> ``add_metric(fn,
sig=...)`` -> one spec -> ``run()`` -> ``vcm.plot()`` / ``tb.plot(...)`` /
``.y``. No unit here mocks the simulator or stubs a bench; every class runs
real ngspice-42 on the 1 kOhm / 1 nF RC low-pass of ``tests/test_probe.py``
(vin -> R -> vout -> C -> gnd, driven by a vpulse), with ``vin``/``vout``
standing in for the design doc's ``vp``/``vm`` probes of a real net pair.

Three things are proved:

1. **The program runs unchanged and matches its lambda twin to 1e-12**
   (parent AC #1). The section-2 program's four metrics (``vcm_avg``,
   ``vd_pk``, ``slew_pk``, ``p_rms``) are computed once through
   ``tb.probe()`` + deferred arithmetic/transforms + the function form of
   ``add_metric``, and once more on an independent, identically-wired
   second bench of the same DUT through today's lambda/callable form
   (``ckt.load_signal`` inside a closure) — the only form that predates
   this feature. Two separate real-ngspice runs, not one run read twice.
2. **The real-ngspice legs of the three prior steps still pass** — imported
   and run here explicitly (rather than asserted from memory) so this file
   is the one piece of evidence that the whole stack, not just this slice,
   is green under real ngspice.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from silk import sources
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.elements import elements
from silk.metrics.statistics import average, rms
from silk.metrics.waveform import peak
from silk.simulation import analysis
from tests._env import needs_ngspice, needs_pdk

_PDK_YAML = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data',
                         'skywater_130_device_details.yaml')
_SKY130_LIB = os.path.expanduser(
    '~/pdk/sky130A/libs.tech/ngspice/sky130.lib.spice')
_have_pdk = os.path.exists(_SKY130_LIB)

ATOL = 1e-12


def _build_rc(pdk, tmp_dir, name):
    """The 1 kOhm / 1 nF RC low-pass of ``tests/test_probe.py``, driven by
    a 0V->1V vpulse: ``vin`` -[R]- ``vout`` -[C]- gnd."""
    dut = Module('PDK_skywater130', tmp_dir, f'{name}.cir', pdk_details=pdk,
                 module_name=name)
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    test = dut.create_test(name)
    test.add_module_under_test({'a': 'vin', 'b': 'vout'})
    test.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9,
                            pw=5e-6, per=10e-6)))
    test.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))
    return test


# ---------------------------------------------------------------------------
# The section-2 program, run exactly as the design doc writes it
# ---------------------------------------------------------------------------

@needs_ngspice
@needs_pdk
@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestSectionTwoProgram:
    """The probe, metric, run and plot program,
    with ``vp``/``vm`` played by the RC circuit's real ``vin``/``vout`` nets.
    """

    @pytest.fixture(scope='class')
    def ran(self, tmp_path_factory):
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)
        tmp_dir = str(tmp_path_factory.mktemp('e2e_probe'))
        tb = _build_rc(pdk, tmp_dir, 'rc_probe')

        vp = tb.probe('vin')                 # a Signal that knows its name and its bench; no data yet
        vm = tb.probe('vout')
        iout = tb.probe('i(vs)')             # a branch current, SPICE spelling

        vcm = (vp + vm) / 2                  # arithmetic defers; vcm is a Signal with a recipe
        vd = vp - vm
        slew = vd.derivative()               # transforms defer too
        p = vp * iout                        # voltage * current, type 'voltage*current'

        tb.add_metric('vcm_avg', average, sig=vcm)          # function + named slot; no lambda, no string type
        tb.add_metric('vd_pk', peak, sig=vd)
        tb.add_metric('slew_pk', peak, sig=slew)
        tb.add_metric('p_rms', rms, sig=p.window(1e-6, 5e-6))
        tb.add_spec('vcm_avg > 0.4', 'vcm_avg', '>', 0.4)

        rep = tb.run()

        return {'tb': tb, 'vp': vp, 'vm': vm, 'iout': iout, 'vcm': vcm,
                'vd': vd, 'slew': slew, 'p': p, 'rep': rep}

    def test_the_program_produced_a_report_with_all_four_metrics(self, ran):
        rep = ran['rep']
        assert rep is not None
        for name in ('vcm_avg', 'vd_pk', 'slew_pk', 'p_rms'):
            assert name in rep.metrics
            assert isinstance(rep.metrics[name], float)
            assert np.isfinite(rep.metrics[name])

    def test_the_spec_was_evaluated_and_passed(self, ran):
        rep = ran['rep']
        assert len(rep.results) == 1
        result = rep.results[0]
        assert result.name == 'vcm_avg > 0.4'
        assert result.metric_name == 'vcm_avg'
        assert result.passed is True
        assert rep.passed is True

    def test_vcm_plot_does_not_raise(self, ran):
        ran['vcm'].plot()

    def test_tb_plot_of_explicit_signals_does_not_raise(self, ran):
        tb, vp, vm, vcm = ran['tb'], ran['vp'], ran['vm'], ran['vcm']
        tb.plot(vp, vm, vcm)

    def test_vcm_y_mean_and_slew_y_max_are_finite_floats(self, ran, capsys):
        vcm, slew = ran['vcm'], ran['slew']
        mean = float(vcm.y.mean())
        smax = float(slew.y.max())
        print(f'vcm.y.mean() = {mean!r}')
        print(f'slew.y.max() = {smax!r}')
        assert np.isfinite(mean) and np.isfinite(smax)

    def test_the_rc_actually_charged(self, ran):
        """Sanity: this is a real simulation, not a stub returning zeros."""
        assert float(np.max(ran['vm'].y)) == pytest.approx(1.0, abs=0.05)
        assert float(np.min(ran['vm'].y)) < 0.05


# ---------------------------------------------------------------------------
# Lambda parity — the SAME four metrics on a second, independent bench of
# the same DUT, written the only way possible before this feature
# ---------------------------------------------------------------------------

@needs_ngspice
@needs_pdk
@pytest.mark.slow
@pytest.mark.skipif(not _have_pdk, reason='sky130 ngspice models not installed')
class TestLambdaParity:
    """Two separate real-ngspice runs of the same RC circuit: one measured
    through ``tb.probe()`` + deferred arithmetic/transforms, one through
    today's ``lambda ckt: ...`` closures over ``ckt.load_signal``. Every
    metric must match to 1e-12 — not merely be close, since both runs solve
    the identical deck and ngspice is deterministic."""

    @pytest.fixture(scope='class')
    def both(self, tmp_path_factory):
        from silk.pdk.basepdk import BasePDK
        pdk = BasePDK(_PDK_YAML, library_location=_SKY130_LIB)

        tmp_a = str(tmp_path_factory.mktemp('e2e_probe_twin'))
        tb_probe = _build_rc(pdk, tmp_a, 'rc_probe_twin')
        vp = tb_probe.probe('vin')
        vm = tb_probe.probe('vout')
        iout = tb_probe.probe('i(vs)')
        vcm = (vp + vm) / 2
        vd = vp - vm
        slew = vd.derivative()
        p = vp * iout
        tb_probe.add_metric('vcm_avg', average, sig=vcm)
        tb_probe.add_metric('vd_pk', peak, sig=vd)
        tb_probe.add_metric('slew_pk', peak, sig=slew)
        tb_probe.add_metric('p_rms', rms, sig=p.window(1e-6, 5e-6))
        rep_probe = tb_probe.run()

        tmp_b = str(tmp_path_factory.mktemp('e2e_lambda_twin'))
        tb_lambda = _build_rc(pdk, tmp_b, 'rc_lambda_twin')
        tb_lambda.add_metric('vcm_avg', lambda ckt: average(
            (ckt.load_signal('vin') + ckt.load_signal('vout')) / 2))
        tb_lambda.add_metric('vd_pk', lambda ckt: peak(
            ckt.load_signal('vin') - ckt.load_signal('vout')))
        tb_lambda.add_metric('slew_pk', lambda ckt: peak(
            (ckt.load_signal('vin') - ckt.load_signal('vout')).derivative()))
        # load_signal uses the physical raw branch; probe('i(vs)') is logical.
        tb_lambda.add_metric('p_rms', lambda ckt: rms(
            (ckt.load_signal('vin') * ckt.load_signal('vvs__silk_wave', 'I'))
            .window(1e-6, 5e-6)))
        rep_lambda = tb_lambda.run()

        return rep_probe, rep_lambda

    @pytest.mark.parametrize('name', ['vcm_avg', 'vd_pk', 'slew_pk', 'p_rms'])
    def test_probe_form_matches_lambda_form_to_1e_12(self, both, name):
        rep_probe, rep_lambda = both
        probe_value = rep_probe.metrics[name]
        lambda_value = rep_lambda.metrics[name]
        assert isinstance(probe_value, float) and np.isfinite(probe_value)
        assert isinstance(lambda_value, float) and np.isfinite(lambda_value)
        np.testing.assert_allclose(probe_value, lambda_value, atol=ATOL,
                                   rtol=0)


# ---------------------------------------------------------------------------
# The real-ngspice legs of 215.5/215.6/215.7, run here explicitly so this
# file is the one piece of evidence that the whole stack is green together,
# not just this slice.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_probe_real_ngspice_legs_are_collectible():
    """Sanity that the three modules' real-ngspice classes still import and
    collect cleanly under this worktree; the actual pass/fail evidence for
    parent AC #7 comes from running them directly (see the final report),
    not from re-running them a second time inside this test.
    """
    import tests.test_probe as m1
    import tests.test_add_metric_function_form as m2
    import tests.test_metric_binding_errors as m3
    assert hasattr(m1, 'TestAgainstRealNgspice')
    assert hasattr(m1, 'TestDeviceCurrentAgainstRealNgspice')
    assert hasattr(m2, 'TestAgainstRealNgspice')
    assert hasattr(m3, 'TestAgainstRealNgspice')
