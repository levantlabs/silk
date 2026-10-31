"""Coherent common-source spectral recipes agree at their named bins."""
import numpy as np
import pytest

from silk.examples.metrics import cs_amp_distortion as single
from silk.examples.metrics import cs_amp_two_tone as two
from silk.metrics import harmonic
from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


def test_single_tone_spectral_equalities(tmp_path):
    mod = single.build_cs_amp(str(tmp_path))
    result = mod.run_all_tests()['cs_amp_distortion.tran'].sim_results
    m = single.measure(result.load_signal('vin'), result.load_signal('vout'))
    assert m['sinad_db'] == m['sndr_db']
    assert m['thd_percent'] == pytest.approx(100 * 10 ** (m['thd_db'] / 20))
    np.testing.assert_allclose(m['harmonic_freqs_true'],
                               single.F0_HZ * np.arange(1, 6))
    np.testing.assert_allclose(m['harmonic_freqs_observed'],
                               m['harmonic_freqs_true'])
    levels = [20 * np.log10(abs(m['harmonic'][k]) / abs(m['harmonic'][1]))
              for k in range(2, 6)]
    np.testing.assert_allclose(m['harmonic_levels_db'], levels, atol=1e-10)
    spur = m['sfdr']
    assert abs(spur.spur_hz - 2 * single.F0_HZ) < spur.bin_hz


@pytest.mark.parametrize('amplitude,label', [(two.AMPL_LO_V, 'lo'),
                                            (two.AMPL_HI_V, 'hi')])
def test_two_tone_bin_agreements(tmp_path, amplitude, label):
    mod = two.build_cs_amp(str(tmp_path), amplitude, label)
    result = mod.run_all_tests()[f'cs_amp_two_tone_{label}.tran'].sim_results
    vout = result.load_signal('vout')
    m = two.measure(vout, result.load_signal('vin'), amplitude)
    stronger = m['mp_f1'] if m['amp_f1'] >= m['amp_f2'] else m['mp_f2']
    assert harmonic(vout, 1, **two._SAMPLING) == pytest.approx(stronger, rel=1e-12, abs=1e-12)
    assert m['p_fund_dbm'] - m['p_imd3_dbm'] == pytest.approx(
        -m['imd3_db'], abs=1e-10)
    assert np.isfinite(m['iip3_dbm'])
