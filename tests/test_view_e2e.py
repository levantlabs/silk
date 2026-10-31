"""Opt-in browser end-to-end test for the waveform explorer.

Emits a real ``view.html`` from a fixture rawfile and drives it in headless
Chrome (``tests/browser/view_e2e.mjs``), asserting on screenshot diffs that
zoom changes the view and ``f`` fits it back, and that the help dialog opens
and closes. uPlot draws to canvas, so pixels are the only honest signal — this
is where interaction bugs surface that the payload-level tests cannot see.

Auto-skips unless Chrome, Node, and ``tests/browser/node_modules`` are all
present, so the default ``pytest`` run and browser-less CI are unaffected. See
``tests/browser/README.md`` for the one-time ``npm install``.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from silk.signals.view import emit_view
from tests._env import chrome_setting

_HERE = os.path.dirname(os.path.abspath(__file__))
_BROWSER_DIR = os.path.join(_HERE, "browser")
_MJS = os.path.join(_BROWSER_DIR, "view_e2e.mjs")
_FIXTURE = os.path.join(_HERE, "fixtures", "rawfiles", "c_tran_ascii_cli.raw")

_CHROME_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


def _find_chrome():
    env = chrome_setting()
    if env and os.path.exists(env):
        return env
    for name in _CHROME_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _reason_to_skip():
    if shutil.which("node") is None:
        return "node not found on PATH"
    if not os.path.isdir(os.path.join(_BROWSER_DIR, "node_modules")):
        return "tests/browser/node_modules missing (run `npm install` in tests/browser)"
    if _find_chrome() is None:
        return "no Chrome/Chromium found (set SILK_CHROME or install google-chrome)"
    return None


pytestmark = pytest.mark.skipif(_reason_to_skip() is not None, reason=_reason_to_skip() or "")


def test_view_interactions_e2e(tmp_path):
    """Drag-zoom (X, Y), the `f` fit key, and the help dialog all behave."""
    html = tmp_path / "view.html"
    emit_view(_FIXTURE, str(html), title="silk view e2e")
    assert html.exists()

    chrome = _find_chrome()
    proc = subprocess.run(
        ["node", _MJS, str(html), chrome],
        cwd=_BROWSER_DIR,
        capture_output=True,
        text=True,
        timeout=180,
    )
    # The node harness prints one PASS/FAIL line per check and exits nonzero
    # if any failed; surface its full output on failure for a usable report.
    assert proc.returncode == 0, (
        "browser e2e checks failed (exit %d):\n%s\n%s"
        % (proc.returncode, proc.stdout, proc.stderr)
    )


@pytest.mark.parametrize('inline', [False, True])
def test_page_mounts_shared_selector_modules_without_external_fetch(tmp_path, inline):
    html = tmp_path / 'modules.html'
    emit_view(_FIXTURE, str(html), inline=inline)
    script = Path(_BROWSER_DIR) / 'selector_modules_e2e.mjs'
    proc = subprocess.run(['node', str(script), str(html), _find_chrome()],
                          cwd=_BROWSER_DIR, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stdout + proc.stderr
