"""TASK-326 — the zeta=0.2 second-order low-pass reproducer.

Same signal as docs/archive/reviews/task-295-headline-findings-recheck.py
section A2, shared by bandwidth_3dB(reference=...) and schema tests
so they measure the same fixture rather than re-deriving the transfer function.

    H(s) = w0^2 / (s^2 + 2*zeta*w0*s + w0^2),  zeta = 0.2,  f0 = 10 kHz

is a standard underdamped second-order low-pass: DC gain 0 dB, one
resonant peak above DC (+8.13 dB, at f0*sqrt(1 - 2*zeta^2)), then a
two-pole roll-off. reference='peak' reads the -3 dB point relative to
that peak; reference='dc' reads it relative to the 0 dB DC gain, which is
what a designer means by "the -3 dB bandwidth" of a low-pass.

PEAK_BW_HZ and DC_BW_HZ are this silk tree's measured values on this
exact grid (1201 points, 6 decades, i.e. 200 points/decade) — both are
grid-snapped per bandwidth_3dB's documented resolution, not interpolated.
DC_ANALYTIC_BW_HZ is the closed-form DC-referenced -3 dB point, solving
|H(jw)|^2 = 1/2 for x = (w/w0)^2: x^2 - (2 - 4*zeta^2)*x - 1 = 0.
"""
import math

import numpy as np

from silk.signals.base import Signal

ZETA = 0.2
F0_HZ = 10e3

# 1201 points, 10 Hz to 10 MHz: the same np.logspace(1, 7, 1201) as the
# TASK-295 A2 reproducer -- 200 points/decade.
_F = np.logspace(1, 7, 1201)


def signal() -> Signal:
    """The simulated-equivalent AC Signal: x = Hz, y = complex H(jw)."""
    w0 = 2 * math.pi * F0_HZ
    s = 1j * 2 * math.pi * _F
    h = w0 ** 2 / (s ** 2 + 2 * ZETA * w0 * s + w0 ** 2)
    return Signal('h', 'voltage', 'frequency', 'V', _F, h)


# Measured on this grid (see tests/metrics/test_analytic_fixtures.py); both
# are the last grid point still at/above their respective -3 dB threshold.
PEAK_BW_HZ = 11350.108156723143
DC_BW_HZ = 14962.356560944358

# Closed form: with x = (w/w0)^2, |H(jw)|^2 = 1 / ((1-x)^2 + 4*zeta^2*x).
# Setting this to 1/2 gives x^2 - (2 - 4*zeta^2)*x - 1 = 0; the positive
# root is the DC-referenced -3 dB point.
_A = 2.0 - 4.0 * ZETA ** 2
_X_3DB = (_A + math.sqrt(_A ** 2 + 4.0)) / 2.0
DC_ANALYTIC_BW_HZ = F0_HZ * math.sqrt(_X_3DB)

# +8.13 dB: the resonant peak above the 0 dB DC gain.
PEAK_ABOVE_DC_DB = -10.0 * math.log10(4.0 * ZETA ** 2 * (1.0 - ZETA ** 2))
