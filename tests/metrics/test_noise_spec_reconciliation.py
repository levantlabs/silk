"""TASK-170 AC#2 — a Spec on a noise-domain metric reconciles pass/fail from
a REAL Noise analysis run, for both ``noise_density_at_freq`` and the
pre-existing ``integrated_noise``.

Reuses the real ``.noise`` fixture ``tests/metrics/test_rawfile_realism.py``
already established (TASK-74): ``e_ng_noise.raw`` / ``e_xy_noise/noise.csv``
are genuine ngspice / Xyce ``.noise`` analysis output — not a hand-built
Signal — loaded through the same real path (``SimulatorBackend.load_noise``,
see that module's docstring) each backend's ``load_noise`` verb uses. The
closed-form oracle (``onoise_true`` / ``integrated_true``) comes from the
same ``gen_e.CIRCUIT`` component values so the deck and the oracle cannot
drift apart.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.noise import integrated_noise, noise_density_at_freq
from silk.signals.base import Signal
from silk.simulation.backend import RawResult, SimulatorBackend

FIXTURES = Path(__file__).resolve().parent.parent / 'fixtures' / 'rawfiles'


def _load_gen_e():
    spec = importlib.util.spec_from_file_location('gen_e', FIXTURES / 'gen_e.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_N = _load_gen_e().CIRCUIT['noise']
K_B = 1.380649e-23
Q_E = 1.602176634e-19
VT = K_B * _N['T_kelvin'] / Q_E
RD = VT / _N['I_bias']                              # diode small-signal r
RP = RD * _N['R1'] / (RD + _N['R1'])                # node impedance
WHITE_I2 = 4 * K_B * _N['T_kelvin'] / _N['R1'] + 2 * Q_E * _N['I_bias']   # A^2/Hz
FLICKER_I2 = _N['KF'] * _N['I_bias'] ** _N['AF']                          # A^2*Hz/Hz at 1 Hz

BACKENDS = ('ngspice', 'xyce')


def onoise_true(f):
    f = np.asarray(f, dtype=float)
    return _N['G'] * RP * np.sqrt(WHITE_I2 + FLICKER_I2 / f)


def integrated_true(f1, f2):
    return _N['G'] * RP * math.sqrt(WHITE_I2 * (f2 - f1) + FLICKER_I2 * math.log(f2 / f1))


def load_noise(backend: str) -> Signal:
    """The REAL noise path: SimulatorBackend.load_signal(raw, 'noise_out')."""
    be = SimulatorBackend.get(backend)
    if backend == 'ngspice':
        raw = be.load_raw(FIXTURES / 'e_ng_noise.raw')
    else:
        raw = RawResult(native=None,
                        source_path=str(FIXTURES / 'e_xy_noise' / 'rawfile.raw'))
    return be.load_signal(raw, 'noise_out')


@pytest.fixture(params=BACKENDS)
def noise_out(request) -> Signal:
    return load_noise(request.param)


class TestIntegratedNoiseSpecFromARealRun:

    def test_spec_passes_and_fails_against_the_real_measured_band(self, noise_out):
        f1, f2 = float(noise_out.x[0]), float(noise_out.x[-1])
        value = integrated_noise(noise_out, f1, f2)
        expect = integrated_true(f1, f2)
        assert value == pytest.approx(expect, rel=2e-4)

        report = SpecReport('noise', 'dut', {'integrated_noise': value})
        passing = report.check('output rms noise', 'integrated_noise', '<', expect * 1.5)
        assert passing.passed

        report = SpecReport('noise', 'dut', {'integrated_noise': value})
        failing = report.check('output rms noise', 'integrated_noise', '<', expect * 0.5)
        assert not failing.passed

    def test_spec_cannot_pass_on_an_unavailable_measurement(self, noise_out):
        # A NaN density inside the requested band is the degenerate-data
        # case decision-1 requires a numeric (nan) result for, not a raise;
        # a Spec built on it must never report a pass, whatever the operator.
        y = np.array(noise_out.y, dtype=float)
        y[0] = np.nan
        poisoned = Signal(noise_out.name, noise_out.type, noise_out.xlabel,
                          noise_out.ylabel, noise_out.x, y)
        f1, f2 = float(poisoned.x[0]), float(poisoned.x[-1])
        with pytest.warns(MetricWarning) as caught:
            value = integrated_noise(poisoned, f1, f2)
        assert type(value) is float and math.isnan(value)
        assert caught[0].message.code == 'nan_density'

        for op in ('<', '<=', '>', '>=', '=='):
            report = SpecReport('noise', 'dut', {'integrated_noise': value})
            assert not report.check('output rms noise', 'integrated_noise', op, 0).passed


class TestNoiseDensityAtFreqSpecFromARealRun:

    def test_spec_passes_and_fails_against_the_real_measured_density(self, noise_out):
        f_mid = float(noise_out.x[len(noise_out.x) // 2])
        value = noise_density_at_freq(noise_out, f_mid)
        expect = float(onoise_true(f_mid))
        assert value == pytest.approx(expect, rel=2e-4)

        report = SpecReport('noise', 'dut', {'noise_density_at_freq': value})
        passing = report.check('output density', 'noise_density_at_freq', '<', expect * 1.5)
        assert passing.passed

        report = SpecReport('noise', 'dut', {'noise_density_at_freq': value})
        failing = report.check('output density', 'noise_density_at_freq', '<', expect * 0.5)
        assert not failing.passed

    def test_spec_cannot_pass_when_the_requested_frequency_is_outside_the_sweep(self, noise_out):
        f_hi = float(noise_out.x[-1])
        with pytest.warns(MetricWarning) as caught:
            value = noise_density_at_freq(noise_out, f_hi * 10)
        assert type(value) is float and math.isnan(value)
        assert caught[0].message.code == 'frequency_outside_sweep'

        for op in ('<', '<=', '>', '>=', '=='):
            report = SpecReport('noise', 'dut', {'noise_density_at_freq': value})
            assert not report.check('output density', 'noise_density_at_freq', op, 0).passed
