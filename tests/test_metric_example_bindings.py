"""Named metric bindings agree with direct calls on the same circuits."""

import pytest

from silk import load_pdk
from silk.examples.metrics import comparator_and_sampler as comparator
from silk.examples.metrics import diff_pair_rejection as rejection
from silk.metrics import cmrr_dB, hysteresis
from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


def test_comparator_farm_hysteresis_matches_separate_benches(tmp_path):
    pdk = load_pdk('sky130')
    dut = comparator.build_comparator(str(tmp_path), pdk)
    comparator.attach_comparator_benches(dut)
    results = dut.run_all_tests()
    up = results['comparator.ramp_up'].sim_results
    down = results['comparator.ramp_dn'].sim_results
    direct = hysteresis(up.load_signal('vin'), up.load_signal('vout'),
                        down.load_signal('vin'), down.load_signal('vout'),
                        out_threshold=comparator.VOUT_BAL)

    farm = comparator.build_ramp_farm(str(tmp_path), pdk)
    comparator.attach_farm_bench(farm)
    result = farm.run_all_tests()['ramp_farm.hyst']
    slow = result.metrics['hysteresis_slow_V']
    assert abs(slow - direct) / abs(direct) < comparator.HYST_ROUTE_TOL
    assert result.metrics['hysteresis_fast_V'] == pytest.approx(
        comparator.FAST * slow, rel=comparator.HYST_ROUTE_TOL)
    # Both named registrations must select the intended pair of instances.
    signals = result.sim_results
    for rate, tags in [('slow', ('u1', 'd1')), ('fast', ('u4', 'd4'))]:
        up_tag, down_tag = tags
        expected = hysteresis(
            signals.load_signal(f'vi_{up_tag}'),
            signals.load_signal(f'vo_{up_tag}'),
            signals.load_signal(f'vi_{down_tag}'),
            signals.load_signal(f'vo_{down_tag}'),
            out_threshold=comparator.VOUT_BAL)
        assert result.metrics[f'hysteresis_{rate}_V'] == pytest.approx(expected)


def test_repeated_pair_cmrr_matches_separate_benches(tmp_path):
    pdk = load_pdk('sky130')
    pair = rejection.build_pair(str(tmp_path), pdk)
    rejection.attach_pair_benches(pair)
    results = pair.run_all_tests()
    dm = results['diff_pair.dm'].sim_results.load_signal('voutn')
    cm = results['diff_pair.cm'].sim_results.load_signal('voutn')

    top = rejection.build_two_instance_pair(str(tmp_path), pdk)
    rejection.attach_declarative_bench(top)
    declared = top.run_all_tests()['cmrr_pair.cmrr'].metrics
    for frequency in rejection.REPORT_HZ:
        direct = cmrr_dB(dm, cm, freq=frequency)
        bound = declared[f'cmrr_{rejection._fname(frequency)}_dB']
        relative = abs(bound - direct) / abs(direct) if direct else abs(bound)
        assert relative < rejection.CMRR_ROUTE_TOL
