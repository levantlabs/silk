"""``Module.view()`` and the top-level ``silk.view()`` shortcut.

These are the low-friction on-ramps to the waveform explorer for someone who
just ran a sim: ``tb.simulate(); tb.view()`` (or ``silk.view("sim.raw")``).
Both are thin, duck-typed delegates to :func:`silk.signals.view.emit_view` —
the method reads only ``.results_file`` and never imports the circuit stack —
so they are exercised here against a lightweight stub and a fixture rawfile,
no PDK or real simulation required.
"""
import os
import webbrowser
from pathlib import Path

import pytest

import silk
from silk.circuit.modules_sim import _SimMethods
from silk.signals.view import ViewError

FIXTURES = Path(__file__).parent / 'fixtures' / 'rawfiles'
A_TRAN = FIXTURES / 'a_tran_bin.raw'


class _Stub:
    """Duck-typed stand-in for a simulated Module — ``Module.view`` reads only
    ``.results_file``, so this is all it needs."""
    def __init__(self, results_file):
        self.results_file = results_file

    # Bind the real method so we test production code, not a copy.
    view = _SimMethods.view


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    """Never actually launch a browser; record what would have been opened."""
    opened = []
    monkeypatch.setattr(webbrowser, 'open', opened.append)
    return opened


# ---------------------------------------------------------------------------
# Module.view()
# ---------------------------------------------------------------------------
class TestModuleView:
    def test_default_out_path_beside_rawfile(self, tmp_path, _no_browser):
        raw = tmp_path / 'rawfile.raw'
        raw.write_bytes(A_TRAN.read_bytes())
        stub = _Stub(str(raw))
        out = stub.view(open=False)
        assert out == str(tmp_path / 'view.html')
        assert os.path.exists(out)
        assert _no_browser == []  # open=False → no launch

    def test_explicit_out_path_honored(self, tmp_path):
        dest = tmp_path / 'sub' / 'my_view.html'
        out = _Stub(str(A_TRAN)).view(out_path=str(dest), open=False)
        assert out == str(dest)
        assert os.path.exists(dest)

    def test_open_true_launches_browser(self, tmp_path, _no_browser):
        out = _Stub(str(A_TRAN)).view(out_path=str(tmp_path / 'v.html'), open=True)
        assert len(_no_browser) == 1
        assert _no_browser[0].endswith('v.html')
        assert _no_browser[0].startswith('file://')

    def test_no_results_file_raises_viewerror(self):
        with pytest.raises(ViewError, match='not been simulated'):
            _Stub(None).view()

    def test_serve_with_inline_raises_viewerror(self, tmp_path):
        with pytest.raises(ViewError, match='not combinable'):
            _Stub(str(A_TRAN)).view(out_path=str(tmp_path / 'v.html'),
                                    serve=True, inline=True)

    def test_signal_selection_passthrough(self, tmp_path):
        # A selector matching nothing must fail loud (proves kwargs reach emit).
        with pytest.raises(ViewError):
            _Stub(str(A_TRAN)).view(out_path=str(tmp_path / 'v.html'),
                                    open=False, signals=['definitely_no_such_signal'])

    def test_real_module_has_the_method(self):
        # The API golden freezes this too; a direct check keeps the failure
        # legible if someone drops the method from the sim mixin.
        assert callable(getattr(silk.Module, 'view', None))


# ---------------------------------------------------------------------------
# silk.view() top-level shortcut
# ---------------------------------------------------------------------------
class TestTopLevelView:
    def test_path_source_emits_beside_rawfile(self, tmp_path, _no_browser):
        raw = tmp_path / 'rawfile.raw'
        raw.write_bytes(A_TRAN.read_bytes())
        out = silk.view(str(raw), open=False)
        assert out == str(tmp_path / 'view.html')
        assert os.path.exists(out)

    def test_open_true_launches_browser(self, tmp_path, _no_browser):
        silk.view(str(A_TRAN), out_path=str(tmp_path / 'v.html'), open=True)
        assert len(_no_browser) == 1
        assert _no_browser[0].endswith('v.html')

    def test_delegates_to_module_view_for_simulated_module(self, tmp_path):
        # A source carrying .results_file and .view is routed to the method,
        # so its defaults/provenance apply — assert the delegation, not a re-emit.
        calls = {}

        class FakeModule:
            results_file = str(A_TRAN)

            def view(self, *, out_path=None, open=True, **kw):
                calls['out_path'] = out_path
                calls['open'] = open
                calls['kw'] = kw
                return 'DELEGATED'

        result = silk.view(FakeModule(), out_path='x.html', open=False, title='T')
        assert result == 'DELEGATED'
        assert calls == {'out_path': 'x.html', 'open': False, 'kw': {'title': 'T'}}

    def test_view_is_exported(self):
        assert 'view' in silk.__all__
        assert silk.view is not None
