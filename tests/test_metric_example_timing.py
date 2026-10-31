"""Timing and jitter identities on inverter and ring-oscillator circuits."""
import math

import numpy as np
import pytest

from silk.examples.metrics import inverter_timing as inv
from silk.examples.metrics import ring_osc_jitter as ring
from silk.metrics import (
    allan_deviation, argmax_x, argmin_x, average, average_duty, average_period, average_power, crossing,
    crossings, cycle_frequency, cycle_to_cycle_jitter, dynamic_energy_per_cycle,
    integral, long_term_jitter, off_time, on_time, peak, peak_to_peak,
    period_jitter, propagation_delay, pulse_width, rise_time, rms, slew_rate, stddev, tie,
)
from tests._env import needs_ngspice, needs_pdk


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_inverter_metric_identities(tmp_path):
    sig = inv.simulate(str(tmp_path))
    vin, vout = sig['vin'], sig['vout']
    mean, quadratic = average(vout), rms(vout)
    assert slew_rate(vout) == pytest.approx(0.8 * peak_to_peak(vout) / rise_time(vout))
    assert stddev(vout) == pytest.approx(math.sqrt(quadratic ** 2 - mean ** 2))
    assert integral(vout) == pytest.approx(mean * inv.STOP, rel=1e-12, abs=0)
    assert on_time(vout, inv.VDD / 2) + off_time(vout, inv.VDD / 2) == pytest.approx(inv.STOP, rel=1e-12, abs=0)
    assert average_period(vout, threshold=inv.VDD / 2) == pytest.approx(inv.PER, rel=1e-4, abs=0)
    first = crossing(vout, inv.VDD / 2, edge='falling', n=1)
    third = crossing(vout, inv.VDD / 2, edge='falling', n=3)
    assert third == pytest.approx(first + 2 * inv.PER, rel=1e-4, abs=0)
    tp_hl = first - inv.T_IN_RISE_MID
    tp_lh = crossing(vout, inv.VDD / 2, edge='rising', n=1) - inv.T_IN_FALL_MID
    width_model = inv.PER / 2 - (tp_lh - tp_hl)
    assert pulse_width(vout) == pytest.approx(width_model, rel=0.002, abs=0)
    assert average_duty(vout) == pytest.approx(width_model / inv.PER, rel=0.002)
    assert on_time(vout, inv.VDD / 2) == pytest.approx(inv.N_CYCLES * width_model, rel=0.002, abs=0)
    assert off_time(vout, inv.VDD / 2) == pytest.approx(inv.STOP - inv.N_CYCLES * width_model, rel=0.002, abs=0)
    assert mean == pytest.approx(inv.VDD * average_duty(vout), rel=0.02)
    assert quadratic == pytest.approx(inv.VDD * math.sqrt(average_duty(vout)), rel=0.02)
    t0, t1 = inv.I_WIN
    assert argmax_x(sig['i_sup'], start=t0, end=t1) == pytest.approx(
        t0 + inv.T_IN_FALL_END, abs=inv.TEDGE)
    assert t0 <= argmin_x(sig['i_sup'], start=t0, end=t1) <= t0 + inv.TEDGE
    energy = dynamic_energy_per_cycle(sig['vdd'], sig['i_sup'], inv.PER)
    assert energy == pytest.approx(average_power(sig['vdd'], sig['i_sup']) * inv.PER,
                                   rel=1e-14, abs=0)
    explicit = propagation_delay(vin, vout, in_edge='rising', out_edge='falling')
    assert explicit == pytest.approx(tp_hl, rel=0.01, abs=0)
    assert propagation_delay(vin, vout, in_edge='any') == pytest.approx(explicit, rel=1e-12, abs=0)
    assert propagation_delay(vin, vout) > inv.PER / 2
    assert explicit < inv.PER / 10


@pytest.fixture(scope='module')
def ring_runs():
    clean = ring.simulate(vdd=ring.VDD, noise='0', stop=ring.STOP_SHORT)
    hi = ring.simulate(vdd=ring.VDD_HI, noise='0', stop=ring.STOP_SHORT)
    noisy = ring.simulate(vdd=ring.VDD, noise=ring.noise_expression(), stop=ring.STOP_LONG)
    model = ring.build_supply_model(average_period(clean['clock']), average_period(hi['clock']))
    return clean, noisy, model


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_ring_sample_statistics_and_curve_identities(ring_runs):
    clean, noisy, model = ring_runs
    clock = noisy['clock']
    edges = crossings(clock, edge='rising')
    periods = np.diff(edges)
    freq, tie_sig, allan = cycle_frequency(clock), tie(clock), allan_deviation(clock)
    assert len(freq.y) >= ring.MIN_CURVE_POINTS
    assert len(tie_sig.y) >= ring.MIN_CURVE_POINTS
    assert len(allan.y) >= ring.MIN_TAU_POINTS
    assert np.all(np.isfinite(allan.y))
    np.testing.assert_allclose(freq.y, 1 / periods, rtol=1e-14)
    np.testing.assert_allclose(freq.x, (edges[1:] + edges[:-1]) / 2, rtol=1e-14)
    # Period RMS and per-edge TIE sample statistics are not time-weighted stddev.
    assert period_jitter(clock) == pytest.approx(np.std(periods, ddof=0), rel=1e-12, abs=0)
    assert cycle_to_cycle_jitter(clock) == pytest.approx(np.sqrt(np.mean(np.diff(periods) ** 2)), rel=1e-12, abs=0)
    assert long_term_jitter(clock) == pytest.approx(np.std(tie_sig.y, ddof=1), rel=1e-12, abs=0)
    assert peak(tie_sig) == np.max(np.abs(tie_sig.y))
    assert abs(np.mean(tie_sig.y)) < 1e-20
    assert abs(np.polyfit(np.arange(len(tie_sig.y)), tie_sig.y, 1)[0]) < 1e-22
    # Allan at one period is the cycle-to-cycle RMS divided by sqrt(2) T.
    expected = np.sqrt(np.mean(np.diff(periods) ** 2) / 2) / np.mean(periods)
    assert allan.y[0] == pytest.approx(expected, rel=1e-12, abs=0)
    assert period_jitter(clean['clock']) < period_jitter(clock) / 100


@pytest.mark.slow
@needs_ngspice
@needs_pdk
def test_ring_supply_model_comparisons(ring_runs):
    _, noisy, model = ring_runs
    clock = noisy['clock']
    assert model.dT_dV < 0
    assert average_period(clock) == pytest.approx(model.t_nom, rel=0.01, abs=0)
    # The supply sensitivity is a first-order circuit model, not an exact identity.
    assert period_jitter(clock) == pytest.approx(ring.predict_period_jitter(model), rel=0.3, abs=0)
    assert long_term_jitter(clock) == pytest.approx(ring.predict_long_term_jitter(model), rel=0.3, abs=0)
    freq = cycle_frequency(clock)
    pred_freq = ring.predict_cycle_frequency(model, freq.x)
    assert np.std(freq.y, ddof=1) == pytest.approx(
        ring.predict_period_jitter(model) / model.t_nom ** 2, rel=0.3, abs=0)
    assert np.std(freq.y - pred_freq) / np.std(freq.y) < 0.5
    tie_sig = tie(clock)
    pred_tie = ring.predict_tie(model, tie_sig.x)
    assert np.corrcoef(tie_sig.y, pred_tie)[0, 1] > 0.9
    allan = allan_deviation(clock)
    pred_allan = ring.predict_allan(model, allan.x)
    # Long-tau estimates have few non-overlapping bins: retain their sampling uncertainty.
    groups = len(np.diff(noisy['edges'])) // np.rint(allan.x / model.t_nom).astype(int)
    uncertainty = 1 / np.sqrt(2 * np.maximum(1, groups - 1))
    assert np.all(np.abs(allan.y / pred_allan - 1) < 0.3 + 3 * uncertainty)
