"""gm/Id lookup example (TASK-344): the real-ngspice run and its checks."""
import pytest

from tests._env import needs_ngspice, needs_pdk


@needs_pdk
@needs_ngspice
def test_gm_id_lookup_example_runs_and_closes_the_loop(tmp_path, monkeypatch):
    monkeypatch.setenv('SILK_EXAMPLES_OUT', str(tmp_path))
    from silk.examples import gm_id_lookup as g

    # One NMOS sweep: the four views exist, x axes are the documented ones.
    p = g.sweep('NMOS', 0.5, g.W_UM)
    v = g.lookup_views(p, g.W_UM)
    assert not v['notes']                       # NMOS gm/Id is monotonic
    assert 'id' in v['a'].xlabel                # x is the Id/W Signal
    assert v['b'].x[0] < v['b'].x[-1]           # gm/Id ascending (flipped)
    assert v['d'].ylabel is not None and 'vov' in v
    vov = v['d'].y
    assert vov.min() < 0 < vov.max()                # weak inversion < 0, strong end > 0
    assert vov[0] > 0.2                             # x is ascending gm/Id; first = strong end
    r = g.readoff(p, v, 'NMOS')
    op = g.op_point('NMOS', 0.5, r['w_um'], r['vg'])
    assert op.gm == pytest.approx(g.GM_TARGET_S, rel=0.25)

    # The hand check inside main() asserts to 1e-6; run it end to end.
    # PMOS Vov reads positive in strong inversion, like NMOS.
    pp = g.sweep('PMOS', 0.5, g.W_UM)
    assert g.lookup_views(pp, g.W_UM)['vov'].y.max() > 0.2

    html = g.main()
    assert open(html).read().count('data:image/png;base64,') == 2
