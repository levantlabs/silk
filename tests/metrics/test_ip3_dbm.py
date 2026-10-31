"""oip3_dbm / iip3_dbm (TASK-397): one two-tone spectrum -> IP3 in dBm."""
import numpy as np
import pytest

from silk.metrics.fom import iip3_extrapolation
from silk.metrics.rf import imd3_db, oip3_dbm, iip3_dbm
from tests.metrics.test_spectral import _trans, TCLK, N

F1, F2 = 101 / (N * TCLK), 103 / (N * TCLK)
T = np.arange(N) * TCLK
A_IN, G, A3 = 0.01, 4.0, -300.0


def _two_tone(a):
    return a * (np.sin(2 * np.pi * F1 * T) + np.sin(2 * np.pi * F2 * T))


def _out():
    x = _two_tone(A_IN)
    return G * x + A3 * x ** 3


def test_oip3_matches_cubic_intercept():
    # y = G x + a3 x^3, two equal tones of amplitude A: fundamental
    # (G A + 9/4 a3 A^3), IM3 3/4 |a3| A^3. Intercept from the measured
    # powers (the one-point extrapolation) must match the closed form
    # computed from the same tone amplitudes.
    p = lambda amp: 10 * np.log10(amp ** 2 / 2 / 50 / 1e-3)
    fund = G * A_IN + 2.25 * A3 * A_IN ** 3
    im3 = 0.75 * abs(A3) * A_IN ** 3
    expect = p(fund) + 0.5 * (p(fund) - p(im3))
    got = oip3_dbm(_trans(_out()), tclk=TCLK, f1_hz=F1, f2_hz=F2, n_samples=N)
    assert got == pytest.approx(expect, abs=1e-6)
    # small-signal limit: OIP3 amplitude sqrt(4/3 G/|a3|)
    a_ip3 = np.sqrt(4 * G / (3 * abs(A3)))
    assert got == pytest.approx(p(G * a_ip3), abs=0.5)


def test_oip3_is_fund_dbm_minus_half_imd3():
    y = _trans(_out())
    kw = dict(tclk=TCLK, f1_hz=F1, f2_hz=F2, n_samples=N)
    p_fund = 10 * np.log10((G * A_IN + 2.25 * A3 * A_IN ** 3) ** 2 / 2 / 50 / 1e-3)
    assert oip3_dbm(y, **kw) == pytest.approx(p_fund - imd3_db(y, **kw) / 2,
                                              abs=1e-6)
    # r_ref scales the dBm reference
    assert (oip3_dbm(y, r_ref=100.0, **kw)
            == pytest.approx(oip3_dbm(y, **kw) - 10 * np.log10(2), abs=1e-9))


def test_iip3_is_oip3_minus_gain_and_matches_fom():
    kw = dict(tclk=TCLK, f1_hz=F1, f2_hz=F2, n_samples=N)
    out, inp = _trans(_out()), _trans(_two_tone(A_IN))
    got = iip3_dbm(out, inp, **kw)
    p = lambda amp: 10 * np.log10(amp ** 2 / 2 / 50 / 1e-3)
    fund = G * A_IN + 2.25 * A3 * A_IN ** 3
    gain_db = p(fund) - p(A_IN)
    assert got == pytest.approx(oip3_dbm(out, **kw) - gain_db, abs=1e-6)
    ref = iip3_extrapolation(p(A_IN), p(fund), imd3_db(out, **kw) + p(fund))
    assert got == pytest.approx(ref, abs=1e-6)


def test_degenerate_and_caller_errors():
    kw = dict(tclk=TCLK, f1_hz=F1, f2_hz=F2, n_samples=N)
    lin = _trans(G * _two_tone(A_IN))
    assert oip3_dbm(lin, **kw) > 100.0              # no IM3: intercept far out
    with pytest.raises(ValueError, match='r_ref'):
        oip3_dbm(lin, r_ref=0.0, **kw)
    with pytest.raises(ValueError, match='r_ref'):
        iip3_dbm(lin, lin, r_ref=-1.0, **kw)
    silent = _trans(np.zeros(N))
    with pytest.warns(Warning):
        assert np.isnan(oip3_dbm(silent, **kw))
    with pytest.warns(Warning):
        assert np.isnan(iip3_dbm(lin, silent, **kw))
