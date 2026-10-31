"""Both waveform producers drive the emitted, offline explorer identically."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from silk.signals.view import emit_view
from tests.test_view_e2e import _BROWSER_DIR, _find_chrome, _reason_to_skip

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
pytestmark = pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or '')


@pytest.mark.parametrize('fixture,inline', [
    ('large', False), ('large', True),
    ('d_multiplot_ac_tran.raw', False), ('f_nonascii_title.raw', True),
])
def test_open_raw_dual_producer(tmp_path, fixture, inline):
    if fixture == 'large':
        # Three pyramid levels, independently located extrema, both Y dtypes,
        # and real gaps exercise the viewer's union-X and nullification paths.
        n = 65536
        i = np.arange(n)
        y = np.sin(i * 0.013)
        gap = np.cos(i * 0.019)
        gap[80:96] = np.nan
        gap[1024] = np.inf
        gap[2048] = -np.inf
        y[0] = -0.0
        values = np.column_stack((i * 1e-9, y, np.sin(i * 0.027) * 1e40, gap))
        header = (
            'Title: dual producer pyramid\nDate: fixture\n'
            'Plotname: Transient Analysis\nFlags: real\n'
            f'No. Variables: 4\nNo. Points: {n}\nVariables:\n'
            '0 time time\n1 v(out) voltage\n2 v(wide) voltage\n'
            '3 v(gap) voltage\nBinary:\n'
        )
        raw = tmp_path / 'pyramid.raw'
        raw.write_bytes(header.encode('ascii') + values.astype('<f8').tobytes())
    else:
        raw = FIXTURES / fixture
    html = tmp_path / 'view.html'
    emit_view(raw, html, inline=inline, poll_ms=100,
              context={'title': 'Original context', 'subtitle': 'original scope'})
    result = subprocess.run(
        ['node', str(Path(_BROWSER_DIR) / 'open_raw_e2e.mjs'), str(html),
         str(raw), _find_chrome()],
        cwd=_BROWSER_DIR, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
