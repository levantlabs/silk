"""Required Python/JavaScript parity gate for the vendorable waveform core."""
import base64
import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading

import numpy as np
import pytest

from tests._env import browser_executable

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "silk/signals/view/assets/waveform-core.mjs"


def _require_node():
    node = shutil.which("node")
    if node:
        return node
    message = "Node >=20 is required for the waveform differential gate; install node or explicitly set SILK_ALLOW_NO_NODE=1"
    if os.environ.get("SILK_ALLOW_NO_NODE") == "1":
        pytest.skip(message)
    pytest.fail(message, pytrace=False)


def test_missing_node_requires_explicit_opt_out(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    for value in (None, "", "0", "true"):
        if value is None:
            monkeypatch.delenv("SILK_ALLOW_NO_NODE", raising=False)
        else:
            monkeypatch.setenv("SILK_ALLOW_NO_NODE", value)
        with pytest.raises(pytest.fail.Exception, match="Node >=20 is required"):
            _require_node()
    monkeypatch.setenv("SILK_ALLOW_NO_NODE", "1")
    with pytest.raises(pytest.skip.Exception, match="SILK_ALLOW_NO_NODE=1"):
        _require_node()


def test_whole_corpus_differential(tmp_path):
    node = _require_node()
    dump = subprocess.run(
        [sys.executable, str(ROOT / "tests/js/dump_waveform_expected.py"), "--output", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert dump.returncode == 0, dump.stdout + dump.stderr
    result = subprocess.run(
        [node, str(ROOT / "tests/js/differential.mjs"), "--module", str(CORE), "--expected", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_node_core_contract():
    result = subprocess.run(
        [_require_node(), "--test", str(ROOT / "tests/js/waveform_core.test.mjs")],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_node_explorer_union_contract():
    result = subprocess.run(
        [_require_node(), "--test", str(ROOT / "tests/js/explorer_union.test.mjs")],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_node_explorer_derive_differential(tmp_path):
    """TASK-374: the explorer's derive operations equal silk's Signal maths,
    including real ngspice transient and complex AC grids."""
    node = _require_node()
    dump = subprocess.run(
        [sys.executable, str(ROOT / "tests/js/dump_derive_expected.py"), "--output", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert dump.returncode == 0, dump.stdout + dump.stderr
    result = subprocess.run(
        [node, "--test", str(ROOT / "tests/js/explorer_derive.test.mjs")],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        env={**os.environ, "SILK_DERIVE_EXPECTED": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Execute the actual JS-produced code against silk probes of the source
    # rawfile, including whole vectors and final magnitude/phase projections.
    from silk.signals.spice import NgspiceSignal
    from silk.signals.rawfile import read_rawfile

    raw = read_rawfile(ROOT / "tests/fixtures/rawfiles/e_ng_ac.raw")

    class Bench:
        def probe(self, name, typ="V", analysis=None):
            return NgspiceSignal.from_rawfile(
                raw, name, typ)

    def decode(value):
        return np.frombuffer(base64.b64decode(value), dtype="<f8")

    for exported in json.loads((tmp_path / "derive_exports.json").read_text()):
        result = eval(exported["code"], {"tb": Bench(), "__builtins__": {}})
        expected = (decode(exported["expected"]) if "expected" in exported
                    else decode(exported["re"]) + 1j * decode(exported["im"]))
        np.testing.assert_allclose(result.y, expected, rtol=1e-12, atol=1e-12,
                                   err_msg=exported["code"])


def test_nonascii_fixture_is_generated_and_byte_exact():
    path = ROOT / "tests/fixtures/rawfiles/gen_browser.py"
    spec = importlib.util.spec_from_file_location("gen_browser", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    data = path.with_name("f_nonascii_title.raw").read_bytes()
    assert data == generator.nonascii_fixture()
    title = data.split(b"\n", 1)[0]
    assert b"\xc2\xb5" in title and b"\xe9" in title


def test_standalone_browser_module(tmp_path):
    # Explicit opt-in only. A browser binary being on PATH does not mean it
    # can run headless here (snap Firefox refuses outside its cgroup; system
    # Chrome's network service hangs on this machine even on about:blank),
    # and a 30 s hang followed by a red fast suite is the wrong signal for
    # supplementary evidence. The Node differential gate above is the
    # required one (agreement O2); this test adds real-browser evidence when
    # a working binary is named.
    browser = browser_executable()
    if not browser:
        pytest.skip("set SILK_FIREFOX=<path> or SILK_CHROME=<path> to run the browser smoke; "
                    "the Node differential gate remains required")
    if not Path(browser).is_file():
        pytest.fail(f"SILK_FIREFOX/SILK_CHROME names a missing executable: {browser}", pytrace=False)
    fixture = ROOT / "tests/fixtures/rawfiles/f_nonascii_title.raw"
    shutil.copyfile(CORE, tmp_path / "waveform-core.mjs")
    shutil.copyfile(fixture, tmp_path / "fixture.raw")
    shutil.copyfile(ROOT / "tests/js/waveform_browser.html", tmp_path / "index.html")
    complete = threading.Event()
    results = []

    class Handler(SimpleHTTPRequestHandler):
        def do_POST(self):
            results.append(self.rfile.read(int(self.headers["Content-Length"])).decode())
            self.send_response(200)
            self.end_headers()
            complete.set()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(tmp_path)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = tmp_path / "profile"
    profile.mkdir()
    firefox = "firefox" in Path(browser).name.lower()
    args = (["--headless", "--no-remote", "--profile", str(profile)] if firefox else
            ["--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
             f"--user-data-dir={profile}"])
    env = dict(os.environ)
    if firefox:
        # Like Chrome's --no-sandbox, needed inside restricted CI containers.
        env["MOZ_DISABLE_CONTENT_SANDBOX"] = "1"
    try:
        with (tmp_path / "browser.log").open("w") as log:
            proc = subprocess.Popen(
                [browser, *args, f"http://127.0.0.1:{server.server_port}/index.html"],
                stdout=log, stderr=log, env=env, start_new_session=os.name == "posix",
            )
            try:
                complete.wait(30)
            finally:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    run_id = hashlib.sha256(fixture.read_bytes() + b"\0[2,0,2,1]").hexdigest()[:16]
    state_key = hashlib.sha256(b"fixture.raw").hexdigest()[:8]
    assert results == [f"PASS {run_id} {state_key}"], (
        results, (tmp_path / "browser.log").read_text(),
    )
