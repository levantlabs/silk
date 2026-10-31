"""MonteCarloResults.corners and Sensitivity.summary (TASK-493)."""
import numpy as np
import pytest

from silk.circuit.monte_carlo import (MonteCarloResults, MonteCarloSettings,
                                      SampleFailure, derive_sample_seed)
from silk.circuit.monte_carlo_draws import (DrawTable, Sensitivity,
                                            SensitivityRow)
from silk.circuit.spec_report import SpecReport


def _mc(corners, dead=()):
    """Two samples per corner; every sample of a corner in *dead* failed."""
    reports, failed, seeds = {}, {}, {}
    for c in corners:
        for i in range(2):
            key = (c, i)
            seeds[key] = derive_sample_seed(0, i)
            if c in dead:
                failed[key] = SampleFailure('SimulationAborted', seeds[key])
            else:
                reports[key] = SpecReport('t', 'm', {'y': float(i)})
    return MonteCarloResults(reports, failed=failed, seeds=seeds,
                             settings=MonteCarloSettings(n=2))


def test_corners_keep_the_run_order_not_sorted_order():
    assert _mc(['tt', 'ss', 'ff']).corners == ['tt', 'ss', 'ff']


def test_corners_include_a_corner_whose_every_sample_failed():
    mc = _mc(['tt', 'ss', 'ff'], dead={'ss'})
    assert not any(k[0] == 'ss' for k in mc.reports)
    assert mc.corners == ['tt', 'ss', 'ff']
    assert mc.corner('ss').corners == ['ss']


def test_corners_without_seeds_fall_back_to_sorted_order():
    mc = _mc(['tt', 'ss'])
    bare = MonteCarloResults(mc.reports, failed={}, seeds={},
                             settings=mc.settings)
    assert bare.corners == ['ss', 'tt']


def _sens(method, **kw):
    rows = [SensitivityRow('M1:vth0', 0.0021, 0.7, 0.8),
            SensitivityRow('M2:vth0', -0.0012, 0.2, -0.4),
            SensitivityRow('M1:u0', 0.0004, 0.1, 0.1)]
    return Sensitivity('y', method, 20, 3, kw.get('r2', 0.9876), rows, [])


def test_summary_least_squares_top_n_and_scaling():
    text = _sens('least_squares').summary(top=2, scale=1e3, unit=' mV')
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0] == '  M1:vth0        effect/sigma +2.100 mV  share 70.0%'
    assert 'M2:vth0' in lines[1] and 'M1:u0' not in text
    assert lines[2] == '  (least-squares fit R^2 = 0.9876)'


def test_summary_correlation_fallback_says_why():
    s = _sens('correlation', r2=None)
    s.rows = [r._replace(effect_per_sigma=None, share=None) for r in s.rows]
    lines = s.summary().splitlines()
    assert lines[0] == '  M1:vth0        correlation +0.800'
    assert 'ranked by correlation only' in lines[-1]
    assert len(lines) == 4


def test_summary_tolerates_unfitted_rows():
    s = _sens('least_squares')
    s.rows[0] = SensitivityRow('M1:vth0', None, 0.0, float('nan'))
    assert 'effect/sigma n/a  share 0.0%' in s.summary()


@pytest.mark.parametrize('top', [0, -1, 2.5, True, None])
def test_summary_rejects_a_bad_top(top):
    with pytest.raises(ValueError, match='top'):
        _sens('least_squares').summary(top=top)


def test_summary_of_a_real_ranking_matches_its_rows():
    rng = np.random.default_rng(3)
    n = 30
    x = rng.standard_normal((n, 2))
    y = 3 * x[:, 0] + 0.1 * x[:, 1]
    keys = [('tt', i) for i in range(n)]
    draws = DrawTable((('M1', 'a'), ('M2', 'a')),
                      {k: x[i] for i, k in enumerate(keys)}, raw=False)
    reports = {k: SpecReport('t', 'm', {'y': float(y[i])})
               for i, k in enumerate(keys)}
    mc = MonteCarloResults(
        reports, settings=MonteCarloSettings(n=n, record_draws=True),
        seeds={k: derive_sample_seed(0, k[1]) for k in keys}, draws=draws)
    s = mc.sensitivity('y')
    assert s.summary(top=1).splitlines()[0].split()[0] == s.rows[0].name
