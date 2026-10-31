"""TASK-414: band_noise_figure against cs_amp_ac_noise's hand-computed factor.

The example forms the noise factor by hand as (v_out / v_rs_out)^2 from the
integrated total output noise and the source resistor's own output
contribution, at the simulation temperature. band_noise_figure must agree
with it once the reference temperature is matched (temp_ref_k = T_sim), and
with its T0 = 290 K re-reference 1 + (F - 1) T_sim / T0 otherwise. Measured
2026-10-02: residual 1.5e-6 dB both ways (ngspice's per-instance and total
spectra agree on the gain to 3.5e-7 relative).
"""
import math

import pytest

from tests._env import needs_ngspice, needs_pdk
from tests._env import scratch_dir

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


@pytest.fixture(scope='module')
def noise_run():
    from silk.examples.metrics import cs_amp_ac_noise as ex
    mod = ex.build_cs_amp(scratch_dir())
    ckt = mod.run_test('noise').sim_results
    return ex, (ckt.load_signal('noise_in'), ckt.load_signal('noise_out'),
                ckt.load_signal('onoise_r.xdut.rrs'))


def test_band_nf_matches_example_factor(noise_run):
    from silk.metrics import band_noise_figure, integrated_noise
    ex, (nin, nout, nrs) = noise_run
    lo, hi = ex.SIG_BAND
    t_sim = ex.TEMP_C + 273.15
    factor = (integrated_noise(nout, lo, hi) / integrated_noise(nrs, lo, hi)) ** 2
    assert factor > 10.0          # a real, noisy stage, not a degenerate 0 dB
    same = band_noise_figure(nin, nout, lo, hi, ex.RS_OHM, t_sim, temp_ref_k=t_sim)
    assert same == pytest.approx(10 * math.log10(factor), abs=1e-4)
    at_t0 = band_noise_figure(nin, nout, lo, hi, ex.RS_OHM, t_sim)
    assert at_t0 == pytest.approx(
        10 * math.log10(1 + (factor - 1) * t_sim / 290.0), abs=1e-4)


def test_spot_nf_falls_with_frequency_past_the_flicker_corner(noise_run):
    from silk.metrics import noise_figure_at_freq
    ex, (nin, _, _) = noise_run
    t_sim = ex.TEMP_C + 273.15
    spots = [noise_figure_at_freq(nin, f, ex.RS_OHM, t_sim) for f in (10, 1e3, 1e5)]
    assert spots[0] > spots[1] > spots[2] > 0
