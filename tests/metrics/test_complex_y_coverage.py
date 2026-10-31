"""TASK-63 — completeness guard: every AC metric gets a complex-y Signal.

F12 (HIGH) was a single line — ``mag = np.abs(np.asarray(sig.y,
dtype=float))`` — that silently discarded the imaginary part of every
simulator AC Signal, because the one pre-existing test for
``gain_bw_product`` happened to hand-build the REAL magnitude instead of
the complex transfer function ``from_rawfile`` actually produces.
Coverage existed; the input dtype was wrong, and nothing forced it to be
right.

This file closes that class of gap mechanically: every public callable
defined in ``silk.metrics.ac`` is exercised at
least once with a Signal whose ``y`` is complex128 (the shape every
ngspice/Xyce ``.ac`` sweep loads as), asserting only that the call
completes without raising ``ComplexWarning`` (or any other exception).
Value-correctness fixtures live in ``test_analytic_fixtures.py`` and
``test_ac.py`` — this file is deliberately narrow: dtype coverage, not
another copy of the numeric checks.

``test_every_ac_metric_has_complex_y_coverage`` enumerates the public
surface of both modules mechanically (mirrors the ``COVERED`` guard in
``test_analytic_fixtures.py``), so a metric added to either module
without a complex-y exercise here fails loudly instead of silently
joining F12's blind spot.

``group_delay`` was the one deliberate exclusion under TASK-63 (its
contract then was a real unwrapped-phase array) — and the exclusion hid
exactly the F12 defect: it cast complex ``y`` to float and differentiated
Re ``H``. TASK-99 made the complex ``H(jω)`` its ONE contract, so it is
covered here like every other metric; ``EXCLUDED`` is empty and stays a
registry only so a future exclusion must be written down with a reason.
"""
import warnings

import numpy as np
import pytest

from silk.signals.base import Signal
import silk.metrics.ac as AC
import silk.metrics.waveform as TR
import silk.metrics as M


def _freq(f, y):
    return Signal('h', 'voltage', 'frequency', 'V',
                  np.asarray(f, dtype=float), np.asarray(y, dtype=complex))


# ---------------------------------------------------------------------------
# Coverage registry (drives the completeness guard)
# ---------------------------------------------------------------------------

COVERED: set = set()


def covers(*names):
    def deco(fn):
        COVERED.update(names)
        return fn
    return deco


def _no_complex_warning(fn):
    """Run fn() with ComplexWarning promoted to an error; return the result."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', np.exceptions.ComplexWarning)
        return fn()


# Single pole and a 3-pole loop, both complex — enough to give the
# phase-crossover metrics (gain_margin(s)) a real crossing to find.
A0, FP = 100.0, 1e4
F = np.logspace(1, 7, 601)
H1 = A0 / (1 + 1j * F / FP)
F3 = np.logspace(1, 7, 1201)
H3 = A0 / (1 + 1j * F3 / FP) ** 3


class TestAcModuleComplexCoverage:

    @covers('passband_ripple_db')
    def test_passband_ripple_db(self):
        _no_complex_warning(lambda: AC.passband_ripple_db(_freq(F, H1), f_pass=1e3))

    @covers('stopband_attenuation_db')
    def test_stopband_attenuation_db(self):
        _no_complex_warning(lambda: AC.stopband_attenuation_db(_freq(F, H1), f_stop=1e5))

    @covers('q_from_peak')
    def test_q_from_peak(self):
        _no_complex_warning(lambda: AC.q_from_peak(_freq(F, H1)))

    @covers('gain_bw_product')
    def test_gain_bw_product(self):
        got = _no_complex_warning(lambda: AC.gain_bw_product(_freq(F, H1)))
        assert got == pytest.approx(A0 * FP, rel=0.05)

    @covers('psrr_db')
    def test_psrr_db(self):
        r = _no_complex_warning(lambda: AC.psrr_db(_freq(F, H1), _freq(F, H1)))
        assert isinstance(r, Signal)


class TestSignalsMetricsComplexCoverage:

    @covers('dc_gain_dB')
    def test_dc_gain_dB(self):
        _no_complex_warning(lambda: M.dc_gain_dB(_freq(F, H1)))

    @covers('peak_gain_dB')
    def test_peak_gain_dB(self):
        _no_complex_warning(lambda: M.peak_gain_dB(_freq(F, H1)))

    @covers('gain_at_freq')
    def test_gain_at_freq(self):
        _no_complex_warning(lambda: M.gain_at_freq(_freq(F, H1), FP))

    @covers('bandwidth_3dB')
    def test_bandwidth_3dB(self):
        got = _no_complex_warning(lambda: M.bandwidth_3dB(_freq(F, H1)))
        assert got == pytest.approx(FP, rel=0.05)

    @covers('unity_gain_freq')
    def test_unity_gain_freq(self):
        _no_complex_warning(lambda: M.unity_gain_freq(_freq(F, H1)))

    @covers('phase_at_freq')
    def test_phase_at_freq(self):
        _no_complex_warning(lambda: M.phase_at_freq(_freq(F, H1), FP))

    @covers('phase_margin')
    def test_phase_margin(self):
        _no_complex_warning(lambda: M.phase_margin(_freq(F, H1)))

    @covers('gain_margins')
    def test_gain_margins(self):
        margins = _no_complex_warning(lambda: M.gain_margins(_freq(F3, H3)))
        assert len(margins) == 1

    @covers('gain_margin')
    def test_gain_margin(self):
        _no_complex_warning(lambda: M.gain_margin(_freq(F3, H3)))

    @covers('cmrr_dB')
    def test_cmrr_dB(self):
        _no_complex_warning(lambda: M.cmrr_dB(_freq(F, H1), _freq(F, 0.01 * H1)))

    @covers('integrated_noise')
    def test_integrated_noise_now_REFUSES_complex_y(self):
        """CHANGED by TASK-197 (T-E). ``integrated_noise`` declares a
        ``noise_density`` input, and a complex ``y`` RULES THAT KIND OUT
        — ``.ac`` is the only producer of complex samples, and
        ``sqrt(int |H|^2 df)`` over a transfer function is not an RMS
        voltage (design §3.4 measured 1.25e+04 "V" on a 100 V/V single
        pole). It used to return that number here.

        This is a STRONGER form of the F12 guarantee this module exists
        for, not a weaker one: the metric can no longer read the wrong
        half of a complex array because it no longer accepts one.
        """
        with pytest.raises(ValueError, match='must be a noise_density Signal'):
            M.integrated_noise(_freq(F, H1), F[0], F[-1])

    @covers('group_delay')
    def test_group_delay(self):
        """FIXED (TASK-99): raised ComplexWarning here pre-fix (the
        ``dtype=float`` cast) and returned the gradient of Re(H)."""
        gd = _no_complex_warning(lambda: AC.group_delay(_freq(F3, H3)))
        assert isinstance(gd, Signal) and gd.y.dtype == np.float64


class TestTransientDerivativeComplexCoverage:
    """``derivative`` lives in ``silk.metrics.waveform``, outside the
    ``AC`` surface the completeness guard below enumerates —
    it is one of transient's axis-agnostic primitives, not an AC metric
    — so it is pinned here explicitly rather than through ``@covers``.

    FIXED (TASK-99 report / TASK-106 fix): before TASK-106 this was the
    one axis-agnostic primitive still doing ``np.asarray(sig.y,
    dtype=float)`` unconditionally — the same F12 defect class as
    ``group_delay`` above — so a complex AC Signal handed to it raised
    ``ComplexWarning`` and came back as d/dx(Re(H)) only.
    """

    def test_derivative_of_complex_H(self):
        d = _no_complex_warning(lambda: TR.derivative(_freq(F, H1)))
        assert isinstance(d, Signal) and np.iscomplexobj(d.y)

    def test_derivative_of_real_y_stays_real(self):
        real_sig = Signal('h', 'voltage', 'frequency', 'V',
                           np.asarray(F, dtype=float),
                           np.asarray(np.abs(H1), dtype=float))
        d = _no_complex_warning(lambda: TR.derivative(real_sig))
        assert isinstance(d, Signal) and d.y.dtype == np.float64


# ---------------------------------------------------------------------------
# completeness guard
# ---------------------------------------------------------------------------

EXCLUDED: dict = {}      # name -> reason; empty since TASK-99 (see module docstring)


def _public_names(module):
    return {n for n, o in vars(module).items()
            if callable(o) and not n.startswith('_')
            and getattr(o, '__module__', '') == module.__name__}


def test_every_ac_metric_has_complex_y_coverage():
    names = _public_names(AC) - set(EXCLUDED)
    missing = sorted(names - COVERED)
    assert not missing, f'AC metrics without complex-y coverage: {missing}'


def test_excluded_metrics_have_a_reason():
    for name in EXCLUDED:
        assert name in _public_names(AC), (
            f'{name} is in EXCLUDED but no longer exists — drop the entry')
