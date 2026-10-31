"""The Monte Carlo diff-pair example runs end to end (TASK-432).

A small campaign (two corners, twelve samples) through the example's own
``run``: the printed summary, the sensitivity ranking and the HTML report.
"""
import os

import pytest

from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


@pytest.mark.parametrize('explicit_out_dir', [False, True],
                         ids=['default-output', 'explicit-output'])
def test_example_runs_and_writes_the_report(tmp_path, monkeypatch, capsys,
                                           explicit_out_dir):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    from silk.examples import mc_diff_pair_offset as ex
    output_dir = tmp_path / ('custom-output' if explicit_out_dir
                             else 'mc_diff_pair_offset')
    kwargs = {'out_dir': str(output_dir)} if explicit_out_dir else {}
    mc, html_path = ex.run(n=6, corners=['tt', 'ss'], n_workers=2, **kwargs)
    out = capsys.readouterr().out

    assert not mc.failed
    assert sorted(mc.reports) == [(c, i) for c in ('ss', 'tt')
                                  for i in range(6)]
    assert mc.draws is not None
    # Matched devices differ only through the drawn mismatch.
    assert mc.stat('offset_V', 'std') > 0.0
    assert html_path == str(output_dir / 'pair_monte_carlo.html')
    # One yield line under each corner, then the pooled line.
    assert out.count('\n    yield |offset| < limit:') == 2
    assert out.count('\n  aggregate yield |offset| < limit:') == 1
    assert '(informational)' in out
    assert 'yield per spec' not in out
    assert 'corner=ss: 6 run, 0 failed, 6 used' in out
    assert 'corner=tt: 6 run, 0 failed, 6 used' in out
    assert 'failed samples:' not in out
    assert 'top sensitivity rows' in out and 'correlation' in out
    assert os.path.isfile(html_path)
    with open(html_path, encoding='utf-8') as f:
        html = f.read()
    assert 'Monte Carlo' in html
    # silk keeps every sample's folder.
    assert os.path.isdir(os.path.join(
        str(output_dir), 'test_pair~offset', 'ss', 'mc_0005'))
