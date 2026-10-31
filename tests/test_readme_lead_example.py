"""The README's lead SPICE example must be complete and physically meaningful.

Guards REL-FIRST-USE (TASK-11.2): the previous lead snippet was an unpowered
two-device fragment that exited zero while measuring vout ~= 2e-55 V — a false
success. This test executes the *exact* code block published in README.md
against real ngspice + sky130 and asserts the measured output is a bounded,
rail-to-rail inverter response, not merely exit code zero.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_pdk, needs_ngspice]

README = Path(__file__).resolve().parents[1] / 'README.md'


def _lead_spice_block():
    """The first ```python fence in README.md that runs a sky130 simulation."""
    text = README.read_text()
    blocks = re.findall(r'```python\n(.*?)```', text, flags=re.DOTALL)
    for block in blocks:
        if "load_pdk('sky130')" in block and 'tb.run()' in block:
            return block
    raise AssertionError(
        "README.md no longer contains a lead sky130 SPICE example "
        "(a ```python fence with load_pdk('sky130') and tb.run()) — "
        "update this test alongside the README.")


def test_lead_example_is_complete_and_meaningful(tmp_path, monkeypatch, capsys):
    """Exec the verbatim README block; vout must swing rail-to-rail."""
    code = _lead_spice_block()

    monkeypatch.chdir(tmp_path)      # the block writes into ./out
    ns = {}
    exec(compile(code, str(README), 'exec'), ns)

    vout = ns['vout']
    y = np.asarray(vout.y, dtype=float)
    assert np.all(np.isfinite(y))
    # A real inverter driven rail-to-rail: low near 0 V, high near vdd = 1.8 V,
    # bounded by the rails plus modest coupling overshoot. The old broken
    # fragment measured ~2e-55 V and fails all of these.
    assert y.min() < 0.1
    assert y.max() > 1.7
    assert -0.3 < y.min() and y.max() < 2.1
    assert y.max() - y.min() > 1.5   # meaningful swing, not numerical noise

    # The printed measurement is the README's documented outcome.
    output = capsys.readouterr().out
    assert 'vout swings' in output
    assert re.search(r'^delay \d+ ps: PASS$', output, flags=re.MULTILINE)

    # Continue the same run with the published viewer fence, keeping it headless.
    blocks = re.findall(r'```python\n(.*?)```', README.read_text(), flags=re.DOTALL)
    viewer_code, = [block for block in blocks if block.strip() == 'tb.netlist.view()']
    viewer = ns['tb'].netlist
    original_view = viewer.view
    paths = []

    def headless_view(*args, **kwargs):
        paths.append(original_view(*args, **kwargs, open=False))
        return paths[-1]

    monkeypatch.setattr(viewer, 'view', headless_view)
    exec(compile(viewer_code, str(README), 'exec'), ns)
    assert Path(paths[0]).is_file()
    assert Path(paths[0]).name == 'view.html'
    assert Path(paths[0]).parent == Path(viewer.results_file).resolve().parent
    monkeypatch.setattr(viewer, 'view', original_view)
    headless_code = 'html_path = tb.netlist.view(open=False)'
    assert f'`{headless_code}`' in README.read_text()
    exec(compile(headless_code, str(README), 'exec'), ns)
    assert ns['html_path'] == paths[0]
