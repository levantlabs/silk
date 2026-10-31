"""The flash converter's nominal and misrouted ladders match their design."""

import numpy as np
import pytest

from silk import load_pdk
from silk.examples.metrics import flash_adc_codes as adc
from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


@pytest.mark.parametrize('misroute', [False, True], ids=['nominal', 'lost-code'])
def test_flash_linearity_matches_drawn_ladder(tmp_path, misroute):
    pdk = load_pdk('sky130')
    comparator = adc.build_comparator(str(tmp_path), pdk)
    name = 'flash3_lost' if misroute else 'flash3'
    flash = adc.build_flash(str(tmp_path), pdk, comparator, name,
                            name + '.cir', misroute=misroute)
    adc.attach_ramp_bench(flash)
    if not misroute:
        adc.attach_sine_bench(flash)
    flash.set_temperature(adc.TEMP_C)
    results = flash.run_all_tests()
    ramp = results[f'{name}.ramp'].sim_results
    static = adc.measure_static(ramp.load_signal('vin'), adc.decode(ramp))

    dnl = (adc.DNL_SHAPE_LOST if misroute else
           np.where(np.arange(6) == adc.MISMATCHED_STEP - 2,
                    adc.DNL_WIDE, adc.DNL_NORMAL))
    inl = adc.INL_SHAPE_LOST if misroute else adc.INL_SHAPE
    np.testing.assert_allclose(static['dnl'].dnl_lsb, dnl,
                               atol=adc.LINEARITY_TOL_LSB, rtol=0)
    np.testing.assert_allclose(static['inl'].inl_lsb, inl,
                               atol=adc.LINEARITY_TOL_LSB, rtol=0)
    assert abs(static['dnl'].max_abs_lsb - np.max(np.abs(dnl))) < (
        adc.LINEARITY_TOL_LSB)
    assert abs(static['inl'].max_abs_lsb - np.max(np.abs(inl))) < (
        adc.LINEARITY_TOL_LSB)
    assert static['is_monotonic'] is (not misroute)
    assert [int(code) for code in static['missing_codes']] == (
        [adc.LOST_TAP] if misroute else [])

    # The first crossing sample can lie up to one maximum ramp step late.
    np.testing.assert_allclose(static['transitions'].y,
                               adc.designed_transitions(misroute),
                               atol=adc.RAMP_DV, rtol=0)
    if not misroute:
        dynamic = adc.measure_dynamic(results['flash3.sine'].sim_results,
                                       static['transitions'])
        assert dynamic['enob'] == pytest.approx(
            dynamic['enob_static_model'], abs=0.05)
        drawn_current = (adc.N_CMP * adc.I_TAIL + adc.VDD /
                         (adc.R_BOT + sum(adc.ladder_values()) + adc.R_TOP))
        assert dynamic['power_w'] == pytest.approx(adc.VDD * drawn_current, rel=0.01)
