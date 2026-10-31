"""Unavailable data must fail specs without aborting a sweep."""
import math

import numpy as np
import pytest

from silk import metrics as M
from silk.circuit.spec_report import SpecReport
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def _tone(n, fundamental, *, toggle=False):
    k = np.arange(n)
    y = np.where(k % 2 == 0, 1.2, 0.0) if toggle else np.cos(
        2 * np.pi * fundamental * k / n)
    return Signal('v', 'voltage', 'time', 'V', k * 1e-9, y)


@pytest.mark.parametrize('n, fundamental, toggle', [
    (256, 128, False), (252, 84, False), (256, 128, True),
])
def test_empty_harmonic_selection_is_nan_and_fails_specs(n, fundamental, toggle):
    sig = _tone(n, fundamental, toggle=toggle)
    values = {fn.__name__: fn(sig, tclk=1e-9, n_samples=n, n_harmonics=5)
              for fn in (M.thd, M.thd_percent)}
    assert all(isinstance(v, float) and math.isnan(v) for v in values.values())
    report = SpecReport('tb', 'dut', values)
    report.check('thd < -60 dB', 'thd', '<', -60)
    report.check('thd_percent < 1%', 'thd_percent', '<', 1)
    assert all(not result.passed for result in report.results)


def _analytic_signal(n, fundamental):
    k = np.arange(n)
    # 1 V RMS fundamental; 30 and 40 mV RMS harmonics: sqrt(.03²+.04²)=.05.
    y = math.sqrt(2) * (
        np.cos(2 * np.pi * fundamental * k / n)
        + 0.03 * np.cos(2 * np.pi * 2 * fundamental * k / n)
        + 0.04 * np.cos(2 * np.pi * 3 * fundamental * k / n))
    return Signal('v', 'voltage', 'time', 'V', k * 1e-9, y)


@pytest.mark.parametrize('bad_n, bad_bin', [(256, 128), (252, 84)])
def test_valid_unavailable_valid_sweep_continues(bad_n, bad_bin):
    signals = [_analytic_signal(256, 13), _tone(bad_n, bad_bin),
               _analytic_signal(256, 55)]  # H3 aliases to bin 91, distinct from H2.
    rows = [{fn.__name__: fn(s, tclk=1e-9, n_samples=len(s.x), n_harmonics=5)
             for fn in (M.thd, M.thd_percent)} for s in signals]
    assert math.isnan(rows[1]['thd'])
    assert math.isnan(rows[1]['thd_percent'])
    for row in (rows[0], rows[2]):
        assert row['thd_percent'] == pytest.approx(5.0, abs=1e-10)
        assert row['thd'] == pytest.approx(-26.02059991328, abs=1e-10)


@pytest.mark.parametrize('fn', [M.thd, M.thd_percent, M.snr])
@pytest.mark.parametrize('bad_kw, message', [
    ({'n_harmonics': 1}, 'n_harmonics must be >= 2'),
    ({'tclk': 0}, 'tclk must be > 0'),
    ({'n_samples': 3}, 'n_samples must be >= 4'),
    ({'n_samples': 257}, 'exceeds trace end'),
    ({'tstart': -1}, 'before the start'),
])
def test_invalid_arguments_still_raise(fn, bad_kw, message):
    kw = dict(tclk=1e-9, n_samples=256, n_harmonics=5)
    kw.update(bad_kw)
    with pytest.raises(ValueError, match=message):
        fn(_tone(256, 128), **kw)


@pytest.mark.parametrize('fn', [M.thd, M.thd_percent])
def test_measurable_clean_tone_remains_a_finite_low_distortion_result(fn):
    value = fn(_tone(256, 13), tclk=1e-9, n_samples=256)
    assert math.isfinite(value)
    assert value < (-150 if fn is M.thd else 1e-6)


@pytest.mark.parametrize('amplitude', [0.0, 1.0, 1e150])
@pytest.mark.parametrize('n, harmonics', [(4, 2), (8, 4)])
def test_snr_empty_residual_is_nan_not_an_eps_denominator(amplitude, n, harmonics):
    sig = _tone(n, 1)
    sig.y = sig.y * amplitude
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        value = M.snr(sig, tclk=1e-9, n_samples=n, n_harmonics=harmonics)
    assert math.isnan(value)
    report = SpecReport('tb', 'dut', {'snr': value})
    report.check('snr > 60 dB', 'snr', '>', 60)
    assert not report.results[0].passed


def test_snr_nonempty_residual_keeps_precision_floor():
    value = M.snr(_tone(8, 1), tclk=1e-9, n_samples=8, n_harmonics=2)
    assert math.isfinite(value) and value > 150


def test_imd3_dc_collision_is_unavailable_and_a_separated_zero_product_is_measurable():
    # f1_bin=1, f2_bin=2: the lower product 2*f1-f2 folds exactly onto DC
    # (bin 0) -- unattributable regardless of the data (TASK-311). Before
    # the fix this bin-0 product was silently dropped by the old ``0 < b``
    # filter and the metric fell through to the surviving (also zero)
    # upper product, reporting a false -inf; it must now report nan.
    dc_kw = dict(tclk=1.0, n_samples=8, f1_hz=1/8, f2_hz=2/8)
    dead = Signal('v', 'voltage', 'time', 'V', np.arange(8), np.zeros(8))
    with pytest.warns(MetricWarning) as caught:
        value = M.imd3_db(dead, **dc_kw)
    assert math.isnan(value)
    assert caught[-1].message.code == 'unattributable_imd_product'
    report = SpecReport('tb', 'dut', {'imd3_db': value})
    report.check('IMD3 < -60 dB', 'imd3_db', '<', -60)
    assert not report.results[0].passed

    # Exact integer samples, pure tone at bin 2: the DC collision above
    # is a property of the requested tone bins, not of this data either.
    sig = Signal('v', 'voltage', 'time', 'V', np.arange(8),
                 np.array([1., 0., -1., 0., 1., 0., -1., 0.]))
    with pytest.warns(MetricWarning) as caught:
        value = M.imd3_db(sig, **dc_kw)
    assert math.isnan(value)
    assert caught[-1].message.code == 'unattributable_imd_product'

    # Move the requested tones to bins 2 and 3: both products (1 and 4)
    # are now separated from DC and from both tones, so this signal's
    # exact-zero product bins ARE measurable: exactly -inf.
    sep_kw = dict(tclk=1.0, n_samples=8, f1_hz=2/8, f2_hz=3/8)
    assert M.imd3_db(sig, **sep_kw) == -math.inf
